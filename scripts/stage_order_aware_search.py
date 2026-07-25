#!/usr/bin/env python3
"""Frozen Tuning-only runner for the standalone order-aware STAGE model.

The runner deliberately keeps experiment orchestration outside ``STAGE.stage``.
Each physical unit trains exactly one declared candidate on one official
TSB-AD series and writes one JSON record atomically.  Labels are consumed only
inside the official metric call after scoring; no Eval result is available to
the selection code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SIX_METRICS = (
    "VUS-PR",
    "VUS-ROC",
    "R-based-F1",
    "AUC-PR",
    "AUC-ROC",
    "Standard-F1",
)
PLAN_NAME = "stage_order_aware_plan.json"
STATUS_NAME = "stage_order_aware_status.json"
SUMMARY_NAME = "stage_order_aware_summary.json"
SUMMARY_CSV = "stage_order_aware_summary.csv"
SELECTION_NAME = "stage_order_aware_selection.json"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz"}
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical_subset(track: str, dataset: str) -> str:
    if track not in {"U", "M"} or not dataset or "/" in dataset:
        raise ValueError(f"invalid subset {track}/{dataset}")
    return f"{track}/{dataset}"


def resolve_candidates(protocol: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    bases = protocol.get("base_config_by_subset")
    overlays = protocol.get("order_candidates")
    heads = protocol.get("fixed_head_by_subset")
    files = protocol.get("files")
    if not all(isinstance(value, Mapping) for value in (bases, heads, files)):
        raise ValueError("protocol lacks subset bases, fixed heads, or files")
    if not isinstance(overlays, list) or len(overlays) < 2:
        raise ValueError("order_candidates must contain at least two candidates")
    output: dict[str, list[dict[str, Any]]] = {}
    for track, datasets in files.items():
        if not isinstance(datasets, Mapping):
            raise ValueError("files must be grouped by track and dataset")
        for dataset, names in datasets.items():
            subset = canonical_subset(str(track), str(dataset))
            if not isinstance(names, list) or not names or len(names) != len(set(names)):
                raise ValueError(f"invalid files for {subset}")
            base = dict(bases[subset])
            head = dict(heads[subset])
            embedding_dim = int(base.get("embedding_dim", 64))
            rows: list[dict[str, Any]] = []
            seen: set[str] = set()
            for raw in overlays:
                if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
                    raise ValueError("every order candidate needs an id")
                candidate_id = str(raw["id"])
                if candidate_id in seen:
                    raise ValueError(f"duplicate order candidate {candidate_id}")
                seen.add(candidate_id)
                parameters = dict(base)
                parameters.update(head)
                parameters.update(dict(raw.get("parameters", {})))
                fraction = raw.get("order_dimension_fraction")
                if fraction is not None:
                    order_dim = int(round(embedding_dim * float(fraction)))
                    order_dim = min(max(2, order_dim), embedding_dim - 2)
                    parameters["order_dim"] = order_dim
                    parameters["invariant_dim"] = embedding_dim - order_dim
                parameters["embedding_dim"] = embedding_dim
                rows.append(
                    {
                        "id": candidate_id,
                        "parameters": parameters,
                        "config_fingerprint": fingerprint(parameters),
                    }
                )
            output[subset] = rows
    return output


def evaluator_hashes(metrics_root: Path) -> dict[str, str]:
    relative = (
        "PaAno/utils/basic_metrics.py",
        "PaAno/affiliation/metrics.py",
        "PaAno/affiliation/generics.py",
        "PaAno/affiliation/_affiliation_zone.py",
        "PaAno/affiliation/_integral_interval.py",
    )
    hashes: dict[str, str] = {}
    for name in relative:
        path = metrics_root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing official evaluator file: {path}")
        hashes[name] = sha256_file(path)
    return hashes


def build_plan(repo: Path, protocol_path: Path, result_root: Path, metrics_root: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    if protocol.get("selection_split") != "official TSB-AD Tuning only":
        raise ValueError("selection_split must be official TSB-AD Tuning only")
    if protocol.get("selection_metric") != "macro VUS-PR only" or protocol.get("eval_feedback") is not False:
        raise ValueError("the frozen selection rule must be Tuning macro VUS-PR only without Eval")
    seeds = protocol.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be a non-empty integer list")
    source = repo / "STAGE" / "stage.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    candidates = resolve_candidates(protocol)
    data_hashes: dict[str, str] = {}
    expected = 0
    for track, datasets in protocol["files"].items():
        for dataset, names in datasets.items():
            subset = canonical_subset(track, dataset)
            for name in names:
                path = repo / "data" / f"TSB-AD-{track}" / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                data_hashes[f"{subset}/{name}"] = sha256_file(path)
                expected += len(candidates[subset]) * len(seeds)
    payload: dict[str, Any] = {
        "schema_version": "stage-order-aware-plan-v1",
        "phase": protocol["phase"],
        "selection_split": protocol["selection_split"],
        "selection_metric": protocol["selection_metric"],
        "eval_feedback": False,
        "source_sha256": sha256_file(source),
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "evaluator_sha256": evaluator_hashes(metrics_root),
        "data_sha256": data_hashes,
        "files": protocol["files"],
        "seeds": seeds,
        "candidates_by_subset": candidates,
        "expected_units": expected,
        "baseline_reference": protocol.get("baseline_reference"),
    }
    payload["plan_fingerprint"] = fingerprint(payload)
    path = result_root / PLAN_NAME
    if path.is_file():
        prior = load_json(path)
        if prior != payload:
            raise RuntimeError("refusing to replace a different frozen plan")
    else:
        atomic_json(path, payload)
    return payload


def unit_path(result_root: Path, subset: str, candidate_id: str, seed: int, file_name: str) -> Path:
    track, dataset = subset.split("/", 1)
    return result_root / "units" / track / dataset / candidate_id / f"seed_{seed}" / f"{Path(file_name).stem}.json"


def expected_tasks(plan: Mapping[str, Any]) -> list[tuple[str, str, int, str]]:
    tasks: list[tuple[str, str, int, str]] = []
    for track, datasets in plan["files"].items():
        for dataset, names in datasets.items():
            subset = canonical_subset(track, dataset)
            for candidate in plan["candidates_by_subset"][subset]:
                for seed in plan["seeds"]:
                    for name in names:
                        tasks.append((subset, candidate["id"], int(seed), name))
    return tasks


def candidate_for(plan: Mapping[str, Any], subset: str, candidate_id: str) -> dict[str, Any]:
    rows = [row for row in plan["candidates_by_subset"][subset] if row["id"] == candidate_id]
    if len(rows) != 1:
        raise ValueError(f"unknown candidate {subset}/{candidate_id}")
    return dict(rows[0])


def valid_unit(value: Mapping[str, Any], plan: Mapping[str, Any], task: tuple[str, str, int, str]) -> bool:
    subset, candidate_id, seed, file_name = task
    try:
        candidate = candidate_for(plan, subset, candidate_id)
        record = value["record"]
        method = next(row for row in record["methods"] if row["method"] == "STAGE")
        observed = [float(method[name]) for name in SIX_METRICS]
        return bool(
            value.get("schema_version") == "stage-order-aware-unit-v1"
            and value.get("plan_fingerprint") == plan["plan_fingerprint"]
            and value.get("source_sha256") == plan["source_sha256"]
            and value.get("subset") == subset
            and value.get("candidate_id") == candidate_id
            and value.get("candidate_fingerprint") == candidate["config_fingerprint"]
            and int(value.get("seed")) == seed
            and value.get("file") == file_name
            and value.get("selection_split") == "official TSB-AD Tuning only"
            and value.get("eval_feedback") is False
            and all(math.isfinite(item) for item in observed)
            and not record.get("checkpoint")
            and not record.get("scores")
        )
    except (KeyError, TypeError, ValueError, StopIteration):
        return False


def verify_plan(repo: Path, protocol_path: Path, result_root: Path, metrics_root: Path) -> dict[str, Any]:
    plan = load_json(result_root / PLAN_NAME)
    if plan != build_plan(repo, protocol_path, result_root, metrics_root):
        raise RuntimeError("plan drift")
    if sha256_file(repo / "STAGE" / "stage.py") != plan["source_sha256"]:
        raise RuntimeError("source drift")
    if sha256_file(Path(__file__).resolve()) != plan["orchestrator_sha256"]:
        raise RuntimeError("orchestrator drift")
    for key, expected in plan["data_sha256"].items():
        track, _, name = key.split("/", 2)
        if sha256_file(repo / "data" / f"TSB-AD-{track}" / name) != expected:
            raise RuntimeError(f"data drift: {key}")
    if evaluator_hashes(metrics_root) != plan["evaluator_sha256"]:
        raise RuntimeError("evaluator drift")
    return plan


@contextmanager
def singleton(result_root: Path):
    path = result_root / "controller.pid"
    result_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"controller pid file already exists: {path}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def configure_determinism() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("strict CUDA worker lacks CUBLAS_WORKSPACE_CONFIG")
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_worker(repo: Path, result_root: Path, metrics_root: Path, plan: Mapping[str, Any], task: tuple[str, str, int, str], gpu: str) -> Path:
    configure_determinism()
    import torch

    from STAGE import stage as model

    subset, candidate_id, seed, file_name = task
    track, _ = subset.split("/", 1)
    candidate = candidate_for(plan, subset, candidate_id)
    parameters = dict(candidate["parameters"])
    parameters["seed"] = int(seed)
    config = model.StageConfig(**parameters)
    config.validate()
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    data_path = repo / "data" / f"TSB-AD-{track}" / file_name
    if sha256_file(data_path) != plan["data_sha256"][f"{subset}/{file_name}"]:
        raise RuntimeError("worker data hash drift")
    temporary_output = result_root / ".worker_scratch" / f"{os.getpid()}_{candidate_id}_{Path(file_name).stem}"
    if temporary_output.exists():
        raise RuntimeError(f"unexpected worker scratch path: {temporary_output}")
    temporary_output.mkdir(parents=True)
    try:
        record = model.evaluate_series(
            data_path,
            temporary_output,
            device,
            config,
            metrics_root,
            save_checkpoint=False,
            save_scores=False,
        )
        value = {
            "schema_version": "stage-order-aware-unit-v1",
            "plan_fingerprint": plan["plan_fingerprint"],
            "source_sha256": plan["source_sha256"],
            "selection_split": "official TSB-AD Tuning only",
            "eval_feedback": False,
            "subset": subset,
            "candidate_id": candidate_id,
            "candidate_fingerprint": candidate["config_fingerprint"],
            "seed": seed,
            "file": file_name,
            "physical_gpu": str(gpu),
            "strict_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "config": asdict(config),
            "record": record,
        }
        if not valid_unit(value, plan, task):
            raise RuntimeError("new unit failed schema validation")
        target = unit_path(result_root, subset, candidate_id, seed, file_name)
        if target.exists():
            raise RuntimeError(f"refusing to overwrite unit: {target}")
        atomic_json(target, value)
        return target
    finally:
        shutil.rmtree(temporary_output, ignore_errors=True)


def _worker_command(script: Path, repo: Path, protocol: Path, result_root: Path, metrics_root: Path, python: Path, task: tuple[str, str, int, str], gpu: str) -> list[str]:
    subset, candidate_id, seed, file_name = task
    return [
        str(python), str(script), "--repo", str(repo), "--protocol", str(protocol),
        "--result-root", str(result_root), "--metrics-root", str(metrics_root), "_worker",
        "--subset", subset, "--candidate", candidate_id, "--seed", str(seed),
        "--file", file_name, "--gpu", str(gpu),
    ]


def run_parent(repo: Path, protocol: Path, result_root: Path, metrics_root: Path, python: Path, gpus: Sequence[str], workers_per_gpu: int) -> None:
    plan = verify_plan(repo, protocol, result_root, metrics_root)
    tasks = []
    for task in expected_tasks(plan):
        target = unit_path(result_root, *task)
        if target.is_file():
            value = load_json(target)
            if valid_unit(value, plan, task):
                continue
            raise RuntimeError(f"invalid existing unit: {target}")
        tasks.append(task)
    work: queue.Queue[tuple[str, str, int, str]] = queue.Queue()
    for task in tasks:
        work.put(task)
    stop = threading.Event()
    errors: list[str] = []
    errors_lock = threading.Lock()
    active: dict[int, subprocess.Popen[str]] = {}
    active_lock = threading.Lock()
    completed = {"count": int(plan["expected_units"]) - len(tasks)}
    completed_lock = threading.Lock()
    script = Path(__file__).resolve()

    def terminate_peers(except_pid: int | None = None) -> None:
        with active_lock:
            peers = [process for pid, process in active.items() if pid != except_pid and process.poll() is None]
        for process in peers:
            process.terminate()

    def lane(gpu: str) -> None:
        while not stop.is_set():
            try:
                task = work.get_nowait()
            except queue.Empty:
                return
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
            environment.setdefault("OMP_NUM_THREADS", "1")
            environment.setdefault("MKL_NUM_THREADS", "1")
            environment.setdefault("OPENBLAS_NUM_THREADS", "1")
            process = subprocess.Popen(
                _worker_command(script, repo, protocol, result_root, metrics_root, python, task, gpu),
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            with active_lock:
                active[process.pid] = process
            output, _ = process.communicate()
            with active_lock:
                active.pop(process.pid, None)
            if process.returncode:
                stop.set()
                terminate_peers(process.pid)
                with errors_lock:
                    errors.append(f"unit {task} on GPU {gpu} failed ({process.returncode}):\n{output[-12000:]}")
                return
            with completed_lock:
                completed["count"] += 1
                print(f"[ORDER-AWARE {completed['count']}/{plan['expected_units']}] GPU={gpu} {task}", flush=True)

    threads = [
        threading.Thread(target=lane, args=(gpu,), daemon=False)
        for gpu in gpus
        for _ in range(int(workers_per_gpu))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("\n\n".join(errors))
    report = status(repo, protocol, result_root, metrics_root)
    if report["valid_units"] != plan["expected_units"] or report["invalid_units"]:
        raise RuntimeError("post-run audit is incomplete")
    atomic_json(result_root / STATUS_NAME, {**report, "status": "complete"})


def forbidden_files(result_root: Path) -> list[str]:
    return [str(path) for path in result_root.rglob("*") if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES]


def status(repo: Path, protocol: Path, result_root: Path, metrics_root: Path) -> dict[str, Any]:
    plan = verify_plan(repo, protocol, result_root, metrics_root)
    valid = 0
    invalid: list[str] = []
    progress: dict[str, dict[str, int]] = {}
    tasks = expected_tasks(plan)
    expected_paths = {unit_path(result_root, *task).resolve() for task in tasks}
    actual_paths = {
        path.resolve()
        for path in (result_root / "units").rglob("*.json")
        if path.is_file()
    }
    unexpected = sorted(str(path) for path in actual_paths - expected_paths)
    for task in tasks:
        subset, candidate_id, _, _ = task
        progress.setdefault(subset, {}).setdefault(candidate_id, 0)
        path = unit_path(result_root, *task)
        if not path.is_file():
            continue
        try:
            value = load_json(path)
        except Exception:
            invalid.append(str(path))
            continue
        if valid_unit(value, plan, task):
            valid += 1
            progress[subset][candidate_id] += 1
        else:
            invalid.append(str(path))
    forbidden = forbidden_files(result_root)
    return {
        "schema_version": "stage-order-aware-status-v1",
        "plan_fingerprint": plan["plan_fingerprint"],
        "expected_units": int(plan["expected_units"]),
        "valid_units": valid,
        "invalid_units": len(invalid),
        "invalid_paths": invalid,
        "unexpected_units": unexpected,
        "forbidden_artifacts": forbidden,
        "progress": progress,
    }


def summarize(repo: Path, protocol: Path, result_root: Path, metrics_root: Path) -> dict[str, Any]:
    plan = verify_plan(repo, protocol, result_root, metrics_root)
    report = status(repo, protocol, result_root, metrics_root)
    if (
        report["valid_units"] != plan["expected_units"]
        or report["invalid_units"]
        or report["unexpected_units"]
        or report["forbidden_artifacts"]
    ):
        raise RuntimeError("cannot summarize incomplete or invalid results")
    rows: list[dict[str, Any]] = []
    for subset, candidates in plan["candidates_by_subset"].items():
        for candidate in candidates:
            records = []
            track, dataset = subset.split("/", 1)
            for seed in plan["seeds"]:
                for name in plan["files"][track][dataset]:
                    value = load_json(unit_path(result_root, subset, candidate["id"], int(seed), name))
                    records.append(next(row for row in value["record"]["methods"] if row["method"] == "STAGE"))
            row = {"subset": subset, "candidate_id": candidate["id"], "series_seed_rows": len(records)}
            for metric in SIX_METRICS:
                values = np.asarray([float(record[metric]) for record in records], dtype=np.float64)
                row[metric] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=0))
            rows.append(row)
    ranked: dict[str, list[dict[str, Any]]] = {}
    for subset in sorted({row["subset"] for row in rows}):
        current = [row for row in rows if row["subset"] == subset]
        current.sort(key=lambda row: (-float(row["VUS-PR"]), str(row["candidate_id"])))
        ranked[subset] = [{**row, "rank": index + 1} for index, row in enumerate(current)]
    selection = {
        subset: {
            "candidate_id": values[0]["candidate_id"],
            "macro_vus_pr": values[0]["VUS-PR"],
            "runner_up": values[1]["candidate_id"],
            "margin": float(values[0]["VUS-PR"] - values[1]["VUS-PR"]),
        }
        for subset, values in ranked.items()
    }
    payload = {
        "schema_version": "stage-order-aware-summary-v1",
        "selection_split": plan["selection_split"],
        "selection_metric": plan["selection_metric"],
        "eval_feedback": False,
        "plan_fingerprint": plan["plan_fingerprint"],
        "rankings": ranked,
        "selection": selection,
    }
    target = result_root / SUMMARY_NAME
    if target.exists() and load_json(target) != payload:
        raise RuntimeError("refusing to replace a different summary")
    atomic_json(target, payload)
    atomic_json(result_root / SELECTION_NAME, {"plan_fingerprint": plan["plan_fingerprint"], "selection": selection})
    flat = [row for values in ranked.values() for row in values]
    temporary = result_root / f".{SUMMARY_CSV}.{os.getpid()}.tmp"
    columns = ["subset", "rank", "candidate_id", "series_seed_rows", *SIX_METRICS, *[f"{metric}_std" for metric in SIX_METRICS]]
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in columns} for row in flat])
    os.replace(temporary, result_root / SUMMARY_CSV)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    run = sub.add_parser("run")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--gpus", default="0,1")
    run.add_argument("--workers-per-gpu", type=int, default=2)
    sub.add_parser("status")
    sub.add_parser("summarize")
    worker = sub.add_parser("_worker")
    worker.add_argument("--subset", required=True)
    worker.add_argument("--candidate", required=True)
    worker.add_argument("--seed", required=True, type=int)
    worker.add_argument("--file", required=True)
    worker.add_argument("--gpu", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    protocol = args.protocol.resolve()
    result_root = args.result_root.resolve()
    metrics_root = args.metrics_root.resolve()
    if args.command == "_worker":
        plan = verify_plan(repo, protocol, result_root, metrics_root)
        task = (args.subset, args.candidate, int(args.seed), args.file)
        print(run_worker(repo, result_root, metrics_root, plan, task, args.gpu))
        return 0
    if args.command == "plan":
        plan = build_plan(repo, protocol, result_root, metrics_root)
        print(json.dumps({"expected_units": plan["expected_units"], "plan_fingerprint": plan["plan_fingerprint"]}, indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(status(repo, protocol, result_root, metrics_root), indent=2))
        return 0
    with singleton(result_root):
        if args.command == "run":
            gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
            if not gpus or int(args.workers_per_gpu) < 1:
                raise ValueError("run needs GPUs and positive workers-per-gpu")
            # Keep the virtual-environment launcher path intact.  ``Path.resolve``
            # dereferences ``.venv/bin/python`` to the system interpreter and
            # silently drops the environment's site-packages.
            python = Path(os.path.abspath(args.python))
            run_parent(repo, protocol, result_root, metrics_root, python, gpus, int(args.workers_per_gpu))
            return 0
        if args.command == "summarize":
            print(json.dumps(summarize(repo, protocol, result_root, metrics_root), indent=2))
            return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
