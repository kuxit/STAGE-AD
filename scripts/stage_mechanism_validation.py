#!/usr/bin/env python3
"""Strict Tuning-only mechanism validation for STAGE.

This runner is deliberately isolated from every formal result tree.  It uses
the frozen ten-dataset Tuning lock as the common configuration and varies only
the mechanism under test.  It never saves checkpoints, embeddings, scores, or
other sample-level arrays.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import atomic_json  # noqa: E402
from STAGE import stage as stage_impl  # noqa: E402


SCHEMA = "stage-mechanism-tuning-v1"
PLAN_SCHEMA = "stage-mechanism-plan-v1"
UNIT_SCHEMA = "stage-mechanism-unit-v1"
STATUS_SCHEMA = "stage-mechanism-status-v1"
SUMMARY_SCHEMA = "stage-mechanism-summary-v1"
PLAN_NAME = "stage_mechanism_plan.json"
STATUS_NAME = "stage_mechanism_status.json"
SUMMARY_NAME = "stage_mechanism_summary.json"
LOCK_NAME = ".stage_mechanism.lock"
SELECTION_SPLIT = "official TSB-AD Tuning only"
METRICS = tuple(stage_impl.SIX_METRICS)

TRAINING_VARIANTS: dict[str, dict[str, Any]] = {
    "full": {"alignment_objective": "both", "uniform_sampling": False},
    "no_timestamp_alignment": {
        "alignment_objective": "context_only",
        "uniform_sampling": False,
    },
    "token_only": {
        "alignment_objective": "timestamp_token_only",
        "uniform_sampling": False,
    },
    "interval_only": {
        "alignment_objective": "interval_only",
        "uniform_sampling": False,
    },
    "uniform_sampling": {
        "alignment_objective": "both",
        "uniform_sampling": True,
    },
}
FULL_MEMORY_VARIANTS = (
    "adaptive_exemplars",
    "uncompressed_memory",
    "same_size_uniform_observed",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def verify_protocol(protocol: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "study_id",
        "selection_split",
        "eval_feedback",
        "seeds",
        "targets",
        "files",
        "representative_priority",
        "expected_formal_head",
        "expected_ten_dataset_lock_fingerprint",
    }
    if set(protocol) != required:
        raise ValueError(f"protocol keys differ: {sorted(set(protocol) ^ required)}")
    if protocol["schema_version"] != SCHEMA:
        raise ValueError("unexpected protocol schema")
    if protocol["selection_split"] != SELECTION_SPLIT or protocol["eval_feedback"] is not False:
        raise ValueError("mechanism validation must be Tuning-only without Eval feedback")
    if protocol["seeds"] != [2026]:
        raise ValueError("the frozen first-pass protocol requires seed 2026")
    if set(protocol["targets"]) != {"U", "M"}:
        raise ValueError("targets must contain U and M")
    subsets = {
        f"{track}/{dataset}"
        for track, datasets in protocol["targets"].items()
        for dataset in datasets
    }
    if len(subsets) != 10:
        raise ValueError("the protocol requires exactly ten dataset subsets")
    if protocol["representative_priority"] != ["U/UCR", "U/SED", "M/CATSv2", "M/GHL"]:
        raise ValueError("representative_priority differs from the frozen cross-setting subset")
    declared = {
        f"{track}/{dataset}"
        for track, datasets in protocol["files"].items()
        for dataset in datasets
    }
    if declared != subsets:
        raise ValueError("files must cover every target subset exactly")
    if sum(len(files) for datasets in protocol["files"].values() for files in datasets.values()) != 22:
        raise ValueError("the frozen mechanism protocol requires exactly 22 Tuning series")


def resolved_config(
    dataset_lock: Mapping[str, Any],
    variant: str,
    seed: int,
) -> stage_impl.StageConfig:
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"unknown training variant: {variant}")
    kwargs = dict(dataset_lock["training_parameters"])
    kwargs["seed"] = int(seed)
    kwargs["top_k"] = int(dataset_lock["top_k"])
    kwargs["alignment_objective"] = TRAINING_VARIANTS[variant]["alignment_objective"]
    if TRAINING_VARIANTS[variant]["uniform_sampling"]:
        kwargs["gb_sampling_power"] = 1.0
    config = stage_impl.StageConfig(**kwargs)
    config.validate()
    return config


def create_plan(
    protocol_path: Path,
    lock_path: Path,
    data_repo: Path,
    metrics_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    verify_protocol(protocol)
    lock = load_json(lock_path)
    if lock.get("lock_fingerprint") != protocol["expected_ten_dataset_lock_fingerprint"]:
        raise RuntimeError("ten-dataset lock fingerprint mismatch")
    if lock.get("eval_feedback") is not False or lock.get("selection_split") != SELECTION_SPLIT:
        raise RuntimeError("ten-dataset lock is not Tuning-only")
    if set(lock.get("dataset_locks", {})) != {
        f"{track}/{dataset}"
        for track, datasets in protocol["targets"].items()
        for dataset in datasets
    }:
        raise RuntimeError("ten-dataset lock target mismatch")

    data_hashes: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []
    logical = 0
    for track in ("U", "M"):
        for dataset in protocol["targets"][track]:
            subset = f"{track}/{dataset}"
            item_lock = lock["dataset_locks"][subset]
            for file_name in protocol["files"][track][dataset]:
                path = data_repo / "data" / f"TSB-AD-{track}" / file_name
                if not path.is_file():
                    raise FileNotFoundError(path)
                data_hashes[f"{track}/{file_name}"] = sha256_file(path)
                weight = float(path.stat().st_size)
                for seed in protocol["seeds"]:
                    for variant in TRAINING_VARIANTS:
                        config = resolved_config(item_lock, variant, int(seed))
                        results = list(FULL_MEMORY_VARIANTS) if variant == "full" else ["adaptive_exemplars"]
                        task = {
                            "track": track,
                            "dataset": dataset,
                            "file": file_name,
                            "seed": int(seed),
                            "training_variant": variant,
                            "result_variants": results,
                            "config_fingerprint": fingerprint(asdict(config)),
                            "priority_tier": 0 if subset in protocol["representative_priority"] else 1,
                            "estimated_weight": weight * (1.7 if variant == "full" else 1.0),
                        }
                        tasks.append(task)
                        logical += len(results)

    source_files = {
        "stage.py": sha256_file(PROJECT_ROOT / "STAGE" / "stage.py"),
        "runner.py": sha256_file(Path(__file__)),
        "protocol.json": sha256_file(protocol_path),
        "ten_dataset_lock.json": sha256_file(lock_path),
        "metrics.py": sha256_file(metrics_root / "PaAno" / "utils" / "metrics.py"),
    }
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "study_id": protocol["study_id"],
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "confirmatory_claim_eligible": False,
        "paper_main_table_eligible": False,
        "runtime_eligible_for_paper": False,
        "formal_head": protocol["expected_formal_head"],
        "ten_dataset_lock_fingerprint": lock["lock_fingerprint"],
        "targets": protocol["targets"],
        "files": protocol["files"],
        "representative_priority": protocol["representative_priority"],
        "seeds": protocol["seeds"],
        "training_variants": TRAINING_VARIANTS,
        "full_memory_variants": list(FULL_MEMORY_VARIANTS),
        "dataset_locks": lock["dataset_locks"],
        "source_sha256": source_files,
        "data_sha256": data_hashes,
        "tasks": tasks,
        "expected_physical_units": len(tasks),
        "expected_logical_records": logical,
    }
    plan["plan_fingerprint"] = fingerprint(plan)
    plan["created_at"] = utc_now()
    target = result_root / PLAN_NAME
    if target.exists():
        prior = load_json(target)
        if prior.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise RuntimeError("refusing to replace a different mechanism plan")
        return prior
    atomic_json(target, plan)
    return plan


def unit_path(result_root: Path, task: Mapping[str, Any]) -> Path:
    return (
        result_root
        / "units"
        / str(task["track"])
        / str(task["dataset"])
        / str(task["training_variant"])
        / f"seed_{int(task['seed'])}"
        / (Path(str(task["file"])).stem + ".json")
    )


def expected_result_variants(task: Mapping[str, Any]) -> set[str]:
    return set(task["result_variants"])


def valid_unit(unit: Mapping[str, Any], plan: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    try:
        if unit.get("schema_version") != UNIT_SCHEMA or unit.get("error") is not None:
            return False
        if unit.get("selection_split") != SELECTION_SPLIT or unit.get("eval_feedback") is not False:
            return False
        if unit.get("plan_fingerprint") != plan["plan_fingerprint"]:
            return False
        for key in ("track", "dataset", "file", "seed", "training_variant", "config_fingerprint"):
            if unit.get(key) != task[key]:
                return False
        if unit.get("data_sha256") != plan["data_sha256"][f"{task['track']}/{task['file']}"]:
            return False
        rows = unit.get("results")
        if not isinstance(rows, list) or {row.get("memory_variant") for row in rows} != expected_result_variants(task):
            return False
        for row in rows:
            if set(row.get("metrics", {})) != set(METRICS):
                return False
            if not all(finite_number(row["metrics"][metric]) for metric in METRICS):
                return False
            if not isinstance(row.get("memory_rows"), int) or row["memory_rows"] < 1:
                return False
        diagnostics = unit.get("alignment_diagnostics")
        if not isinstance(diagnostics, dict) or not diagnostics:
            return False
        for item in diagnostics.values():
            if not isinstance(item, dict) or not all(finite_number(value) for value in item.values()):
                return False
        return True
    except Exception:
        return False


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }


def alignment_diagnostics(
    model: stage_impl.StageEncoder,
    values: np.ndarray,
    device: torch.device,
    config: stage_impl.StageConfig,
) -> dict[str, dict[str, float]]:
    model.eval()
    output: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for delta in config.overlap_deltas:
            eligible = len(values) - int(config.patch_size) - int(delta) + 1
            count = min(512, eligible)
            starts = np.unique(np.rint(np.linspace(0, eligible - 1, count)).astype(np.int64))
            token_distances: list[np.ndarray] = []
            interval_distances: list[np.ndarray] = []
            for offset in range(0, len(starts), 128):
                current = starts[offset : offset + 128]
                left = stage_impl.window_batch(values, current, config.patch_size).to(device=device, dtype=torch.float32)
                right = stage_impl.window_batch(values, current + int(delta), config.patch_size).to(device=device, dtype=torch.float32)
                tokens_left = model.encode_tokens(left)
                tokens_right = model.encode_tokens(right)
                trim = int(config.overlap_trim)
                aligned_left = F.normalize(tokens_left[:, int(delta) + trim : config.patch_size - trim], dim=-1)
                aligned_right = F.normalize(tokens_right[:, trim : config.patch_size - int(delta) - trim], dim=-1)
                token_distances.append((1.0 - (aligned_left * aligned_right).sum(dim=-1)).cpu().numpy().reshape(-1))
                emb_left = F.normalize(model.embed_tokens(aligned_left), dim=-1)
                emb_right = F.normalize(model.embed_tokens(aligned_right), dim=-1)
                interval_distances.append((1.0 - (emb_left * emb_right).sum(dim=-1)).cpu().numpy().reshape(-1))
            token = percentile_summary(np.concatenate(token_distances))
            interval = percentile_summary(np.concatenate(interval_distances))
            output[str(int(delta))] = {
                "token_mean": token["mean"],
                "token_median": token["median"],
                "token_p90": token["p90"],
                "interval_mean": interval["mean"],
                "interval_median": interval["median"],
                "interval_p90": interval["p90"],
            }
    model.train()
    return output


def uniform_observed_memory(embeddings: np.ndarray, rows: int) -> np.ndarray:
    count = int(len(embeddings))
    rows = min(int(rows), count)
    indices = np.linspace(0, count - 1, rows, dtype=np.int64)
    if len(np.unique(indices)) != rows:
        raise AssertionError("uniform matched-memory indices are not unique")
    return np.asarray(embeddings[indices], dtype=np.float32)


def score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    normal = scores[labels == 0]
    anomaly = scores[labels != 0]
    if not len(normal) or not len(anomaly):
        raise ValueError("mechanism diagnostics require normal and anomalous points")
    return {
        "normal_mean": float(np.mean(normal)),
        "normal_p90": float(np.quantile(normal, 0.90)),
        "anomaly_mean": float(np.mean(anomaly)),
        "anomaly_p10": float(np.quantile(anomaly, 0.10)),
        "separation_mean": float(np.mean(anomaly) - np.mean(normal)),
    }


def run_worker(
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    data_repo: Path,
    metrics_root: Path,
    result_root: Path,
) -> Path:
    target = unit_path(result_root, task)
    if target.exists():
        prior = load_json(target)
        if valid_unit(prior, plan, task):
            return target
        raise RuntimeError(f"refusing to overwrite invalid unit: {target}")
    for name, expected in plan["source_sha256"].items():
        actual_path = {
            "stage.py": PROJECT_ROOT / "STAGE" / "stage.py",
            "runner.py": Path(__file__),
            "metrics.py": metrics_root / "PaAno" / "utils" / "metrics.py",
        }.get(name)
        if actual_path is not None and sha256_file(actual_path) != expected:
            raise RuntimeError(f"source hash drift: {name}")
    data_path = data_repo / "data" / f"TSB-AD-{task['track']}" / str(task["file"])
    if sha256_file(data_path) != plan["data_sha256"][f"{task['track']}/{task['file']}"]:
        raise RuntimeError("Tuning data hash drift")
    subset = f"{task['track']}/{task['dataset']}"
    item_lock = plan["dataset_locks"][subset]
    config = resolved_config(item_lock, str(task["training_variant"]), int(task["seed"]))
    if fingerprint(asdict(config)) != task["config_fingerprint"]:
        raise RuntimeError("resolved config fingerprint drift")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    values, train_index, label_column = stage_impl.load_series(data_path)
    stage_impl.seed_everything(config.seed)
    model = stage_impl.StageEncoder(values.shape[1], config).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    torch.cuda.reset_peak_memory_stats(device)
    training = stage_impl.fit_encoder(values[:train_index], model, device, config)
    align_diag = alignment_diagnostics(model, values[:train_index], device, config)
    train_starts = np.arange(train_index - config.patch_size + 1, dtype=np.int64)
    full_starts = np.arange(len(values) - config.patch_size + 1, dtype=np.int64)
    train_raw = stage_impl.extract_embeddings(model, values[:train_index], train_starts, device, config, normalize=False)
    train_unit = train_raw / np.maximum(np.linalg.norm(train_raw, axis=1, keepdims=True), 1e-12)
    full_unit = stage_impl.extract_embeddings(model, values, full_starts, device, config, normalize=True)
    final_config = replace(config, gb_min_split=int(item_lock["final_gb_min_split"]))
    adaptive, partition = stage_impl.build_gb_exemplar_memory(train_unit, final_config, seed=int(config.seed) + 29)
    memories: dict[str, np.ndarray] = {"adaptive_exemplars": adaptive}
    if task["training_variant"] == "full":
        memories["uncompressed_memory"] = train_unit.astype(np.float32, copy=False)
        memories["same_size_uniform_observed"] = uniform_observed_memory(train_unit, len(adaptive))
    labels = stage_impl.load_labels_after_scoring(data_path, label_column, len(values))
    sliding_window = stage_impl.estimate_sliding_window(values)
    rows: list[dict[str, Any]] = []
    for memory_variant in task["result_variants"]:
        memory = memories[memory_variant]
        patch_scores = stage_impl.score_embeddings(full_unit, memory, device, final_config)
        point_scores = stage_impl.aggregate_patch_scores(patch_scores, len(values), config.patch_size)
        metrics = stage_impl.official_metrics(point_scores, labels, sliding_window, metrics_root)
        rows.append(
            {
                "memory_variant": memory_variant,
                "metrics": {metric: float(metrics[metric]) for metric in METRICS},
                "memory_rows": int(len(memory)),
                "memory_ratio": float(len(memory) / len(train_unit)),
                "score_diagnostics": score_summary(point_scores, labels),
            }
        )
    torch.cuda.synchronize(device)
    unit = {
        "schema_version": UNIT_SCHEMA,
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "confirmatory_claim_eligible": False,
        "paper_main_table_eligible": False,
        "runtime_eligible_for_paper": False,
        "plan_fingerprint": plan["plan_fingerprint"],
        "ten_dataset_lock_fingerprint": plan["ten_dataset_lock_fingerprint"],
        "track": task["track"],
        "dataset": task["dataset"],
        "file": task["file"],
        "seed": int(task["seed"]),
        "training_variant": task["training_variant"],
        "config_fingerprint": task["config_fingerprint"],
        "data_sha256": plan["data_sha256"][f"{task['track']}/{task['file']}"],
        "parameter_count": parameter_count,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "training": training,
        "alignment_diagnostics": align_diag,
        "final_partition": {
            "initial_k": int(partition.initial_k),
            "final_k": int(partition.final_k),
            "rounds": int(partition.rounds),
            "source_rows": int(partition.source_rows),
        },
        "results": rows,
        "error": None,
        "created_at": utc_now(),
    }
    if not valid_unit(unit, plan, task):
        raise RuntimeError("worker produced an invalid unit")
    atomic_json(target, unit)
    return target


def status(plan: Mapping[str, Any], result_root: Path) -> dict[str, Any]:
    raw_paths = list((result_root / "units").rglob("*.json")) if (result_root / "units").exists() else []
    valid = 0
    invalid = 0
    errors = 0
    logical = 0
    expected = {str(unit_path(result_root, task)): task for task in plan["tasks"]}
    seen: set[str] = set()
    for path in raw_paths:
        key = str(path)
        task = expected.get(key)
        if task is None or key in seen:
            invalid += 1
            continue
        seen.add(key)
        try:
            unit = load_json(path)
        except Exception:
            invalid += 1
            continue
        if unit.get("error") is not None:
            errors += 1
        elif valid_unit(unit, plan, task):
            valid += 1
            logical += len(unit["results"])
        else:
            invalid += 1
    payload = {
        "schema_version": STATUS_SCHEMA,
        "plan_fingerprint": plan["plan_fingerprint"],
        "physical_valid": valid,
        "physical_expected": int(plan["expected_physical_units"]),
        "logical_valid": logical,
        "logical_expected": int(plan["expected_logical_records"]),
        "raw": len(raw_paths),
        "invalid": invalid,
        "duplicate": max(0, len(raw_paths) - len(seen)),
        "error": errors,
        "complete": valid == int(plan["expected_physical_units"]) and logical == int(plan["expected_logical_records"]) and invalid == 0 and errors == 0,
        "updated_at": utc_now(),
    }
    atomic_json(result_root / STATUS_NAME, payload)
    return payload


def run_controller(
    plan: Mapping[str, Any],
    protocol_path: Path,
    lock_path: Path,
    data_repo: Path,
    metrics_root: Path,
    result_root: Path,
    gpus: list[int],
    workers_per_gpu: int,
) -> None:
    if os.name == "nt":
        raise RuntimeError("controller must run on Linux")
    import fcntl

    lock_handle = (result_root / LOCK_NAME).open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another mechanism controller already holds the lock") from exc
    pid_path = result_root / "stage_mechanism.pid"
    atomic_json(pid_path, {"pid": os.getpid(), "started_at": utc_now(), "plan_fingerprint": plan["plan_fingerprint"]})
    tasks = sorted(
        plan["tasks"],
        key=lambda item: (int(item["priority_tier"]), -float(item["estimated_weight"])),
    )
    pending = []
    for task in tasks:
        path = unit_path(result_root, task)
        if path.exists() and valid_unit(load_json(path), plan, task):
            continue
        pending.append(task)
    queue_lock = threading.Lock()

    def consume(gpu: int) -> None:
        while True:
            with queue_lock:
                if not pending:
                    return
                task = pending.pop(0)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            command = [
                sys.executable,
                str(Path(__file__)),
                "_worker",
                "--protocol", str(protocol_path),
                "--ten-dataset-lock", str(lock_path),
                "--data-repo", str(data_repo),
                "--metrics-root", str(metrics_root),
                "--result-root", str(result_root),
                "--task-json", json.dumps(task, separators=(",", ":")),
            ]
            completed = subprocess.run(command, env=env, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"worker failed for {task['track']}/{task['dataset']}/{task['training_variant']}/{task['file']}")

    futures = []
    with ThreadPoolExecutor(max_workers=len(gpus) * int(workers_per_gpu)) as pool:
        for gpu in gpus:
            for _ in range(int(workers_per_gpu)):
                futures.append(pool.submit(consume, int(gpu)))
        for future in as_completed(futures):
            future.result()
    final = status(plan, result_root)
    if not final["complete"]:
        raise RuntimeError(f"controller ended without complete strict status: {final}")


def summarize(plan: Mapping[str, Any], result_root: Path) -> dict[str, Any]:
    current = status(plan, result_root)
    if not current["complete"]:
        raise RuntimeError("cannot summarize an incomplete mechanism study")
    buckets: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    diagnostics: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in plan["tasks"]:
        unit = load_json(unit_path(result_root, task))
        subset = f"{task['track']}/{task['dataset']}"
        for row in unit["results"]:
            key = (subset, str(task["training_variant"]), str(row["memory_variant"]))
            buckets.setdefault(key, []).append(row["metrics"])
        diagnostics.setdefault((subset, str(task["training_variant"])), []).append(unit["alignment_diagnostics"])
    rows: list[dict[str, Any]] = []
    for (subset, training_variant, memory_variant), items in sorted(buckets.items()):
        row = {
            "subset": subset,
            "training_variant": training_variant,
            "memory_variant": memory_variant,
            "series": len(items),
            "metrics": {metric: float(np.mean([item[metric] for item in items])) for metric in METRICS},
        }
        rows.append(row)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "confirmatory_claim_eligible": False,
        "plan_fingerprint": plan["plan_fingerprint"],
        "rows": rows,
        "created_at": utc_now(),
    }
    summary["summary_fingerprint"] = fingerprint(summary)
    target = result_root / SUMMARY_NAME
    if target.exists():
        prior = load_json(target)
        if prior.get("summary_fingerprint") != summary["summary_fingerprint"]:
            raise RuntimeError("refusing to replace a different mechanism summary")
        return prior
    atomic_json(target, summary)
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    sub = value.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "status", "summarize", "_worker"):
        item = sub.add_parser(name)
        item.add_argument("--protocol", type=Path, required=True)
        item.add_argument("--ten-dataset-lock", type=Path, required=True)
        item.add_argument("--data-repo", type=Path, required=True)
        item.add_argument("--metrics-root", type=Path, required=True)
        item.add_argument("--result-root", type=Path, required=True)
        if name == "run":
            item.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
            item.add_argument("--workers-per-gpu", type=int, default=6)
        if name == "_worker":
            item.add_argument("--task-json", required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        plan_path = args.result_root / PLAN_NAME
        if args.command == "plan":
            plan = create_plan(args.protocol, args.ten_dataset_lock, args.data_repo, args.metrics_root, args.result_root)
            print(json.dumps({"plan_fingerprint": plan["plan_fingerprint"], "physical": plan["expected_physical_units"], "logical": plan["expected_logical_records"]}))
            return 0
        if not plan_path.is_file():
            raise FileNotFoundError(plan_path)
        plan = load_json(plan_path)
        if args.command == "_worker":
            task = json.loads(args.task_json)
            run_worker(task, plan, args.data_repo, args.metrics_root, args.result_root)
            return 0
        if args.command == "status":
            print(json.dumps(status(plan, args.result_root), ensure_ascii=False))
            return 0
        if args.command == "summarize":
            result = summarize(plan, args.result_root)
            print(json.dumps({"summary_fingerprint": result["summary_fingerprint"], "rows": len(result["rows"])}))
            return 0
        run_controller(plan, args.protocol, args.ten_dataset_lock, args.data_repo, args.metrics_root, args.result_root, args.gpus, args.workers_per_gpu)
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
