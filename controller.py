from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any

import pandas as pd

from common import METRICS, atomic_json, complete_record, dataset_name, sha256


NON_DEEP = ("KMeansAD", "PCA", "IForest", "LOF")
TSB_DEEP = (
    "PatchTST",
    "AnomalyTransformer",
    "TimesNet",
    "TranAD",
    "USAD",
    "OmniAnomaly",
)
EXTERNAL_DEEP = ("PaAno", "GBOC", "MEMTO", "DCdetector", "STAGE")
BASELINE_GPU_ORDER = ("PaAno", *TSB_DEEP, "GBOC", "MEMTO", "DCdetector")
METHOD_ORDER = (*NON_DEEP, *BASELINE_GPU_ORDER, "STAGE")
TARGETS = {
    "U": ("UCR", "Exathlon", "MSL", "SED", "TODS"),
    "M": ("CATSv2", "GHL", "LTDB", "SVDB", "TAO"),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def selected_files(repo: Path) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for track, datasets in TARGETS.items():
        frame = pd.read_csv(repo / "data" / "File_List" / f"TSB-AD-{track}-Eva.csv")
        names = frame["file_name"].astype(str).tolist()
        chosen = [name for name in names if dataset_name(name) in datasets]
        output[track] = chosen
    return output


def standard_path(result: Path, method: str, track: str, file_name: str) -> Path:
    return result / "units" / method / track / dataset_name(file_name) / f"{Path(file_name).stem}.json"


def complete_record_for_seed(path: Path, seed: int) -> bool:
    if not complete_record(path):
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return int(payload.get("seed")) == int(seed)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def normalized_external_record(method: str, source: dict[str, Any], track: str, file_name: str, seed: int) -> dict[str, Any]:
    metrics = source.get("metrics")
    if metrics is None and method == "STAGE":
        methods = source.get("methods", [])
        if len(methods) != 1:
            raise ValueError("STAGE output does not contain exactly one method")
        metrics = {name: methods[0][name] for name in METRICS}
    return {
        "method": method,
        "track": track,
        "dataset": dataset_name(file_name),
        "file": file_name,
        "seed": seed,
        "training_prefix_policy": "filename_declared_label_blind",
        "source_record": source,
        "metrics": {name: float(metrics[name]) for name in METRICS},
        "runtime_s": source.get("runtime_s", source.get("elapsed_seconds")),
        "error": None,
    }


def command_for(args: argparse.Namespace, method: str, track: str, file_name: str, target: Path, gpu: str | None, scratch: Path) -> tuple[list[str], dict[str, str], Path | None]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    python = str(args.python)
    common = [
        "--repo", str(args.repo), "--method", method, "--track", track,
        "--file", file_name, "--seed", str(args.seed), "--output", str(target),
        "--stage-source", str(args.stage_source), "--paano-root", str(args.paano_root),
    ]
    if method in NON_DEEP:
        return [python, str(args.experiment / "run_one_classical.py"), *common], env, None
    if method in TSB_DEEP:
        return [
            python, str(args.experiment / "run_one_deep.py"), *common,
            "--require-physical-gpu", str(gpu),
        ], env, None
    if method == "PaAno":
        raw = scratch / "paano.json"
        return [
            python, str(args.paano_one), "--paano-root", str(args.paano_root),
            "--data-file", str(args.repo / "data" / f"TSB-AD-{track}" / file_name),
            "--track", track, "--output", str(raw), "--seed", str(args.seed),
            "--require-physical-gpu", str(gpu),
        ], env, raw
    if method == "GBOC":
        raw_root = scratch / "gboc"
        raw = raw_root / "Canonical" / "gboc" / track / f"seed_{args.seed}" / "metrics_parts" / f"{Path(file_name).stem}.json"
        return [
            python, str(args.gboc_one), "--repo", str(args.repo),
            "--upstream", str(args.gboc_root), "--output", str(raw_root),
            "--config", str(args.gboc_config), "--track", track,
            "--file", file_name, "--seed", str(args.seed),
        ], env, raw
    if method == "MEMTO":
        return [
            python, str(args.memto_one), "--repo", str(args.repo),
            "--memto-root", str(args.memto_root), "--track", track,
            "--file", file_name, "--seed", str(args.seed), "--output", str(target),
            "--stage-source", str(args.stage_source), "--paano-root", str(args.paano_root),
            "--require-physical-gpu", str(gpu),
        ], env, None
    if method == "DCdetector":
        return [
            python, str(args.dcdetector_one), "--repo", str(args.repo),
            "--dcdetector-root", str(args.dcdetector_root), "--track", track,
            "--file", file_name, "--seed", str(args.seed), "--output", str(target),
            "--stage-source", str(args.stage_source), "--paano-root", str(args.paano_root),
            "--require-physical-gpu", str(gpu),
        ], env, None
    if method == "STAGE":
        raw_root = scratch / "stage"
        raw = raw_root / "series" / f"{Path(file_name).stem}.json"
        return [
            python, str(args.stage_source), "--data-root", str(args.repo / "data" / f"TSB-AD-{track}"),
            "--files", file_name, "--output", str(raw_root), "--metrics-root", str(args.paano_root.parent),
            "--device", "cuda:0", "--require-physical-gpu", str(gpu), "--seed", str(args.seed),
        ], env, raw
    raise ValueError(method)


def run_unit(args: argparse.Namespace, method: str, track: str, file_name: str, gpu: str | None) -> dict[str, Any]:
    target = standard_path(args.result, method, track, file_name)
    if complete_record(target):
        return {"status": "skipped", "method": method, "track": track, "file": file_name}
    log_path = args.result / "logs" / method / track / f"{Path(file_name).stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in (1, 2):
        with tempfile.TemporaryDirectory(prefix="stage-baseline-") as temporary:
            scratch = Path(temporary)
            command, env, raw_path = command_for(args, method, track, file_name, target, gpu, scratch)
            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{now()}] attempt={attempt} gpu={gpu} command={command!r}\n")
                process = subprocess.run(command, cwd=scratch, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
            try:
                if process.returncode != 0:
                    raise RuntimeError(f"exit code {process.returncode}")
                if raw_path is not None:
                    source = json.loads(raw_path.read_text(encoding="utf-8"))
                    atomic_json(target, normalized_external_record(method, source, track, file_name, args.seed))
                if not complete_record(target):
                    raise RuntimeError("unit did not produce six valid metrics")
                return {
                    "status": "complete",
                    "method": method,
                    "track": track,
                    "file": file_name,
                    "gpu": gpu,
                    "elapsed_s": time.time() - started,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
    if not target.exists() or complete_record(target):
        atomic_json(
            target,
            {
                "method": method,
                "track": track,
                "dataset": dataset_name(file_name),
                "file": file_name,
                "seed": args.seed,
                "metrics": {},
                "error": last_error,
            },
        )
    return {"status": "error", "method": method, "track": track, "file": file_name, "gpu": gpu, "error": last_error}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--stage-source", required=True, type=Path)
    parser.add_argument("--paano-root", required=True, type=Path)
    parser.add_argument("--paano-one", required=True, type=Path)
    parser.add_argument("--gboc-root", required=True, type=Path)
    parser.add_argument("--gboc-one", required=True, type=Path)
    parser.add_argument("--gboc-config", required=True, type=Path)
    parser.add_argument("--memto-root", required=True, type=Path)
    parser.add_argument("--memto-one", required=True, type=Path)
    parser.add_argument("--dcdetector-root", required=True, type=Path)
    parser.add_argument("--dcdetector-one", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--mode", choices=("smoke", "formal"), default="formal")
    parser.add_argument(
        "--phase",
        choices=("baseline", "target", "all"),
        default="baseline",
        help="Default is baseline-only; target requires complete same-seed baselines.",
    )
    args = parser.parse_args()
    # Do not Path.resolve() the Python executable: resolving a venv symlink can
    # silently replace it with the system interpreter and lose site-packages.
    args.python = Path(os.path.abspath(args.python))
    for name in ("repo", "experiment", "result", "stage_source", "paano_root", "paano_one", "gboc_root", "gboc_one", "gboc_config", "memto_root", "memto_one", "dcdetector_root", "dcdetector_one"):
        setattr(args, name, getattr(args, name).resolve())
    args.result.mkdir(parents=True, exist_ok=True)
    files = selected_files(args.repo)
    if args.mode == "smoke":
        files = {
            track: [min(names, key=lambda name: (args.repo / "data" / f"TSB-AD-{track}" / name).stat().st_size)]
            for track, names in files.items()
        }
    if args.phase == "baseline":
        methods = (*NON_DEEP, *BASELINE_GPU_ORDER)
    elif args.phase == "target":
        methods = ("STAGE",)
    else:
        methods = METHOD_ORDER

    if args.phase == "target":
        missing_baselines = [
            (method, track, file_name)
            for method in (*NON_DEEP, *BASELINE_GPU_ORDER)
            for track in ("U", "M")
            for file_name in files[track]
            if not complete_record_for_seed(
                standard_path(args.result, method, track, file_name), args.seed
            )
        ]
        if missing_baselines:
            sample = missing_baselines[:5]
            raise RuntimeError(
                f"target phase blocked: {len(missing_baselines)} same-seed baseline "
                f"units are incomplete; sample={sample}"
            )
    jobs = [
        (method, track, file_name)
        for method in methods
        for track in ("U", "M")
        for file_name in files[track]
    ]
    manifest = {
        "protocol": "stage-10subset-seed2026-v1",
        "status": "running",
        "mode": args.mode,
        "phase": args.phase,
        "seed": args.seed,
        "started_at": now(),
        "files": {track: len(names) for track, names in files.items()},
        "datasets": TARGETS,
        "methods": methods,
        "expected_units": len(jobs),
        "metrics": METRICS,
        "execution_policy": {
            "baselines_first": True,
            "gpu_priority": list(BASELINE_GPU_ORDER),
            "target_after_baselines": True,
            "runtime_eligible_for_paper": False,
        },
        "source_sha256": {
            "stage": sha256(args.stage_source),
            "controller": sha256(Path(__file__)),
            "deep_runner": sha256(args.experiment / "run_one_deep.py"),
            "classical_runner": sha256(args.experiment / "run_one_classical.py"),
            "paano_one": sha256(args.paano_one),
            "gboc_one": sha256(args.gboc_one),
            "gboc_config": sha256(args.gboc_config),
            "memto_one": sha256(args.memto_one),
            "dcdetector_one": sha256(args.dcdetector_one),
        },
        "completed_units": 0,
        "error_units": 0,
    }
    manifest_path = args.result / f"run_manifest_{args.phase}.json"
    atomic_json(manifest_path, manifest)
    lock = threading.Lock()
    completed = 0
    errors = 0

    def update(item: dict[str, Any]) -> None:
        nonlocal completed, errors
        with lock:
            if item["status"] in {"complete", "skipped"}:
                completed += 1
            else:
                errors += 1
            manifest.update(
                {
                    "completed_units": completed,
                    "error_units": errors,
                    "last_unit": item,
                    "updated_at": now(),
                }
            )
            atomic_json(manifest_path, manifest)
            print(f"[{completed + errors}/{len(jobs)}] {item}", flush=True)

    cpu_jobs = [job for job in jobs if job[0] in NON_DEEP]
    gpus = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    if len(gpus) != 2:
        raise ValueError("this locked run requires exactly two physical GPUs")

    def gpu_worker(gpu: str, queue: list[tuple[str, str, str]], queue_lock: threading.Lock) -> None:
        while True:
            with queue_lock:
                if not queue:
                    return
                job = queue.pop(0)
            update(run_unit(args, *job, gpu))

    with ThreadPoolExecutor(max_workers=args.cpu_workers) as cpu_pool, ThreadPoolExecutor(max_workers=2) as gpu_pool:
        cpu_futures = [cpu_pool.submit(run_unit, args, *job, None) for job in cpu_jobs]

        # Baseline GPU methods are strict phases. This makes PaAno finish before
        # the next GPU baseline begins and prevents STAGE from overlapping any
        # admitted baseline.
        for method in BASELINE_GPU_ORDER:
            queue = [job for job in jobs if job[0] == method]
            queue_lock = threading.Lock()
            phase = [gpu_pool.submit(gpu_worker, gpu, queue, queue_lock) for gpu in gpus]
            for future in as_completed(phase):
                future.result()

        for future in as_completed(cpu_futures):
            result = future.result()
            if isinstance(result, dict):
                update(result)

        if "STAGE" in methods and errors == 0:
            queue = [job for job in jobs if job[0] == "STAGE"]
            queue_lock = threading.Lock()
            phase = [gpu_pool.submit(gpu_worker, gpu, queue, queue_lock) for gpu in gpus]
            for future in as_completed(phase):
                future.result()
        elif "STAGE" in methods:
            manifest["target_skipped_due_to_baseline_error"] = True
            manifest["updated_at"] = now()
            atomic_json(manifest_path, manifest)

    manifest["status"] = "complete" if errors == 0 and completed == len(jobs) else "incomplete"
    manifest["completed_at"] = now()
    atomic_json(manifest_path, manifest)
    if args.phase == "baseline":
        return 0 if manifest["status"] == "complete" else 1
    summarize = subprocess.run(
        [str(args.python), str(args.experiment / "summarize.py"), "--protocol", str(args.experiment / "protocol.json"), "--result", str(args.result)],
        check=False,
    )
    return 0 if manifest["status"] == "complete" and summarize.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
