#!/usr/bin/env python3
"""Leakage-safe STAGE tuning, parameter locking, and multi-seed evaluation.

The workflow has deliberately separate stages:

1. build one deterministic 24-candidate search plan for every benchmark subset;
2. evaluate those candidates only on the official TSB-AD Tuning split;
3. freeze and hash one configuration per (track, dataset);
4. evaluate the frozen configuration on Eval;
5. continue to later seeds only when the predeclared seed-2026 gate passes.

No command persists checkpoints or anomaly-score arrays.  Eval results never
participate in parameter selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import METRICS, atomic_json, complete_record, dataset_name


TARGETS: dict[str, tuple[str, ...]] = {
    "U": ("UCR", "Exathlon", "MSL", "SED", "TODS"),
    "M": ("CATSv2", "GHL", "LTDB", "SVDB", "TAO"),
}
DEFAULT_STAGE_PARAMETERS: dict[str, Any] = {
    "patch_size": 96,
    "overlap_deltas": [24, 48],
    "overlap_trim": 8,
    "learning_rate": 0.0003,
    "dropout": 0.1,
    "steps": 1000,
    "gb_activation_fraction": 0.2,
    "gb_sampling_power": 0.5,
    "top_k": 3,
}
PLAN_NAME = "tuning_plan.json"
LOCK_NAME = "stage_locked_seed2026.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_metrics(item: Mapping[str, Any]) -> bool:
    metrics = item.get("metrics", {})
    return not item.get("error") and all(
        name in metrics and math.isfinite(float(metrics[name])) for name in METRICS
    )


def split_files(repo: Path, split: str) -> dict[str, dict[str, list[str]]]:
    """Return only the ten declared track/dataset subsets from an official split."""
    if split not in {"Tuning", "Eva"}:
        raise ValueError(f"unsupported split: {split}")
    selected: dict[str, dict[str, list[str]]] = {}
    for track, datasets in TARGETS.items():
        manifest = repo / "data" / "File_List" / f"TSB-AD-{track}-{split}.csv"
        names = pd.read_csv(manifest)["file_name"].astype(str).tolist()
        by_dataset = {
            dataset: [name for name in names if dataset_name(name) == dataset]
            for dataset in datasets
        }
        empty = [dataset for dataset, subset in by_dataset.items() if not subset]
        if empty:
            raise ValueError(f"{manifest} has no members for {empty}")
        for subset in by_dataset.values():
            for name in subset:
                path = repo / "data" / f"TSB-AD-{track}" / name
                if not path.is_file():
                    raise FileNotFoundError(path)
        selected[track] = by_dataset
    return selected


def candidate_parameters(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    tuning = policy["stage_tuning_policy"]
    space = tuning["search_space"]
    profiles = [dict(item) for item in space["alignment_profiles"]]
    axes = (
        profiles,
        space["learning_rate"],
        space["dropout"],
        space["steps"],
        space["gb_activation_fraction"],
        space["gb_sampling_power"],
        space["top_k"],
    )
    candidates: list[dict[str, Any]] = []
    for profile, learning_rate, dropout, steps, activation, sampling, top_k in itertools.product(*axes):
        item = {
            **profile,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "steps": steps,
            "gb_activation_fraction": activation,
            "gb_sampling_power": sampling,
            "top_k": top_k,
        }
        candidates.append(item)
    default_index = next(
        (index for index, item in enumerate(candidates) if item == DEFAULT_STAGE_PARAMETERS),
        None,
    )
    if default_index is None:
        raise ValueError("the declared search space does not contain the STAGE default")
    default = candidates.pop(default_index)
    sampler = random.Random(int(tuning["sampler_seed"]))
    sampler.shuffle(candidates)
    budget = int(tuning["max_trials_per_subset"])
    if budget < 1 or budget > len(candidates) + 1:
        raise ValueError(f"invalid tuning budget: {budget}")
    chosen = [default, *candidates[: budget - 1]]
    fixed = dict(tuning["fixed_parameters"])
    return [{**fixed, **item} for item in chosen]


def build_plan(repo: Path, tuning_root: Path) -> dict[str, Any]:
    policy_path = repo / "experiment_policy.json"
    protocol_path = repo / "protocol.json"
    policy = load_json(policy_path)
    protocol = load_json(protocol_path)
    tuning = policy["stage_tuning_policy"]
    if tuning.get("data") != "official TSB-AD Tuning only":
        raise ValueError("policy does not restrict STAGE selection to official Tuning")
    if tuning.get("eval_feedback") or not tuning.get("freeze_and_hash_before_eval"):
        raise ValueError("unsafe tuning policy")
    if tuning.get("exathlon_exception"):
        raise ValueError("dataset-specific search-budget exceptions are forbidden")
    files = split_files(repo, "Tuning")
    candidates = candidate_parameters(policy)
    stable = {
        "schema_version": "stage-tuning-plan-v1",
        "protocol": protocol["protocol_version"],
        "selection_split": "official TSB-AD Tuning only",
        "selection_seed": int(policy["seed"]),
        "objective": tuning["selection_objective"],
        "tie_breakers": list(tuning["tie_breakers"]),
        "trials_per_subset": len(candidates),
        "same_candidates_for_every_subset": True,
        "eval_feedback": False,
        "files": files,
        "candidates": [
            {
                "trial": index,
                "config_fingerprint": fingerprint(config),
                "parameters": config,
            }
            for index, config in enumerate(candidates)
        ],
        "source_sha256": sha256_file(repo / "STAGE" / "stage.py"),
        "policy_sha256": sha256_file(policy_path),
        "protocol_sha256": sha256_file(protocol_path),
    }
    plan = {**stable, "plan_fingerprint": fingerprint(stable), "created_at": utc_now()}
    target = tuning_root / PLAN_NAME
    if target.is_file():
        prior = load_json(target)
        if prior.get("plan_fingerprint") != plan["plan_fingerprint"]:
            raise RuntimeError(f"existing tuning plan differs: {target}")
        return prior
    atomic_json(target, plan)
    return plan


def tuning_unit_path(
    root: Path, track: str, dataset: str, trial: int, file_name: str
) -> Path:
    return (
        root
        / "units"
        / track
        / dataset
        / f"trial_{trial:02d}"
        / f"{Path(file_name).stem}.json"
    )


def eval_unit_path(root: Path, track: str, dataset: str, file_name: str) -> Path:
    return root / "units" / "STAGE" / track / dataset / f"{Path(file_name).stem}.json"


def stage_parameter_args(config: Mapping[str, Any]) -> list[str]:
    mapping = {
        "patch_size": "--patch-size",
        "channels": "--channels",
        "token_dim": "--token-dim",
        "embedding_dim": "--embedding-dim",
        "dilations": "--dilations",
        "group_norm_groups": "--group-norm-groups",
        "dropout": "--dropout",
        "batch_size": "--batch-size",
        "steps": "--steps",
        "gb_activation_fraction": "--gb-activation-fraction",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "grad_clip": "--grad-clip",
        "overlap_deltas": "--overlap-deltas",
        "overlap_trim": "--overlap-trim",
        "gb_min_split": "--gb-min-split",
        "gb_max_rounds": "--gb-max-rounds",
        "gb_sampling_power": "--gb-sampling-power",
        "top_k": "--top-k",
    }
    args: list[str] = []
    for name, flag in mapping.items():
        if name not in config:
            continue
        value = config[name]
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        args.extend((flag, str(value)))
    return args


def run_stage_once(
    *,
    repo: Path,
    python: Path,
    metrics_root: Path,
    track: str,
    file_name: str,
    config: Mapping[str, Any],
    seed: int,
    gpu: str,
    scratch_parent: Path,
) -> dict[str, Any]:
    scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage-", dir=scratch_parent) as temporary:
        output = Path(temporary) / "output"
        command = [
            str(python),
            str(repo / "STAGE" / "stage.py"),
            "--data-root",
            str(repo / "data" / f"TSB-AD-{track}"),
            "--files",
            file_name,
            "--output",
            str(output),
            "--metrics-root",
            str(metrics_root),
            "--device",
            "cuda:0",
            "--require-physical-gpu",
            gpu,
            "--seed",
            str(seed),
            *stage_parameter_args(config),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", "2")
        environment.setdefault("OPENBLAS_NUM_THREADS", "2")
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            tail = completed.stdout[-8000:]
            raise RuntimeError(
                f"STAGE failed for {track}/{file_name} on GPU {gpu}:\n{tail}"
            )
        raw_path = output / "series" / f"{Path(file_name).stem}.json"
        raw = load_json(raw_path)
        methods = raw.get("methods", [])
        if len(methods) != 1 or methods[0].get("method") != "STAGE":
            raise ValueError(f"unexpected STAGE record: {raw_path}")
        metrics = {name: float(methods[0][name]) for name in METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"non-finite metric: {raw_path}")
        return {
            "metrics": metrics,
            "runtime_s": raw.get("training", {}).get("elapsed_seconds"),
            "diagnostics": {
                "points": raw.get("points"),
                "channels": raw.get("channels"),
                "train_index": raw.get("train_index"),
                "parameter_count": raw.get("parameter_count"),
                "peak_cuda_bytes": raw.get("peak_cuda_bytes"),
                "final_gb": raw.get("final_gb"),
                "query_seconds": methods[0].get("query_seconds"),
                "query_ms_per_patch": methods[0].get("query_ms_per_patch"),
            },
        }


def run_lanes(
    tasks: Sequence[tuple[Any, ...]],
    gpus: Sequence[str],
    workers_per_gpu: int,
    worker,
) -> None:
    if not tasks:
        return
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    lanes = [gpu for gpu in gpus for _ in range(workers_per_gpu)]
    iterator = iter(tasks)
    iterator_lock = threading.Lock()
    stop = threading.Event()
    failures: list[BaseException] = []

    def lane(gpu: str) -> None:
        while not stop.is_set():
            with iterator_lock:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
            try:
                worker(gpu, task)
            except BaseException as exc:  # fail-stop after in-flight units settle
                failures.append(exc)
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [pool.submit(lane, gpu) for gpu in lanes]
        for future in as_completed(futures):
            future.result()
    if failures:
        raise RuntimeError(str(failures[0])) from failures[0]


def run_tuning(
    repo: Path,
    tuning_root: Path,
    python: Path,
    metrics_root: Path,
    gpus: Sequence[str],
    workers_per_gpu: int,
) -> None:
    plan = build_plan(repo, tuning_root)
    tasks: list[tuple[str, str, int, str, dict[str, Any], str]] = []
    for track, datasets in plan["files"].items():
        for dataset, files in datasets.items():
            for candidate in plan["candidates"]:
                trial = int(candidate["trial"])
                for file_name in files:
                    target = tuning_unit_path(tuning_root, track, dataset, trial, file_name)
                    if target.is_file():
                        item = load_json(target)
                        if (
                            finite_metrics(item)
                            and item.get("config_fingerprint")
                            == candidate["config_fingerprint"]
                            and int(item.get("seed", -1)) == int(plan["selection_seed"])
                        ):
                            continue
                    tasks.append(
                        (
                            track,
                            dataset,
                            trial,
                            file_name,
                            candidate["parameters"],
                            candidate["config_fingerprint"],
                        )
                    )
    total = sum(
        len(files) * len(plan["candidates"])
        for datasets in plan["files"].values()
        for files in datasets.values()
    )
    completed_before = total - len(tasks)
    counter = {"value": completed_before}
    counter_lock = threading.Lock()
    scratch = tuning_root / ".scratch"

    def worker(gpu: str, task: tuple[Any, ...]) -> None:
        track, dataset, trial, file_name, config, config_fingerprint = task
        outcome = run_stage_once(
            repo=repo,
            python=python,
            metrics_root=metrics_root,
            track=track,
            file_name=file_name,
            config=config,
            seed=int(plan["selection_seed"]),
            gpu=gpu,
            scratch_parent=scratch,
        )
        payload = {
            "kind": "STAGE official-Tuning trial",
            "method": "STAGE",
            "track": track,
            "dataset": dataset,
            "file": file_name,
            "trial": trial,
            "seed": int(plan["selection_seed"]),
            "selection_split": "Tuning",
            "config_fingerprint": config_fingerprint,
            "source_sha256": plan["source_sha256"],
            "metrics": outcome["metrics"],
            "runtime_s": outcome["runtime_s"],
            "diagnostics": outcome["diagnostics"],
            "error": None,
        }
        atomic_json(tuning_unit_path(tuning_root, track, dataset, trial, file_name), payload)
        with counter_lock:
            counter["value"] += 1
            print(
                f"[tuning {counter['value']}/{total}] GPU={gpu} "
                f"{track}/{dataset} trial={trial:02d} {file_name}",
                flush=True,
            )

    run_lanes(tasks, gpus, workers_per_gpu, worker)
    if scratch.exists():
        shutil.rmtree(scratch)
    atomic_json(
        tuning_root / "tuning_status.json",
        {
            "status": "complete",
            "expected_units": total,
            "completed_units": total,
            "error_units": 0,
            "plan_fingerprint": plan["plan_fingerprint"],
            "completed_at": utc_now(),
        },
    )


def freeze_config(repo: Path, tuning_root: Path, export_path: Path | None) -> dict[str, Any]:
    plan = build_plan(repo, tuning_root)
    if sha256_file(repo / "STAGE" / "stage.py") != plan["source_sha256"]:
        raise RuntimeError("STAGE source changed after the Tuning plan was created")
    selections: dict[str, Any] = {}
    for track, datasets in plan["files"].items():
        for dataset, files in datasets.items():
            ranking: list[dict[str, Any]] = []
            for candidate in plan["candidates"]:
                trial = int(candidate["trial"])
                records = []
                for file_name in files:
                    path = tuning_unit_path(tuning_root, track, dataset, trial, file_name)
                    if not path.is_file():
                        raise RuntimeError(f"missing Tuning result: {path}")
                    item = load_json(path)
                    if (
                        not finite_metrics(item)
                        or item.get("config_fingerprint") != candidate["config_fingerprint"]
                    ):
                        raise RuntimeError(f"invalid Tuning result: {path}")
                    records.append(item)
                means = {
                    metric: sum(float(item["metrics"][metric]) for item in records)
                    / len(records)
                    for metric in METRICS
                }
                objective = sum(means.values()) / len(METRICS)
                ranking.append(
                    {
                        "trial": trial,
                        "config_fingerprint": candidate["config_fingerprint"],
                        "parameters": candidate["parameters"],
                        "series": len(records),
                        "objective": objective,
                        "metric_means": means,
                    }
                )
            ranking.sort(
                key=lambda row: (
                    -row["objective"],
                    -row["metric_means"]["VUS-PR"],
                    -row["metric_means"]["VUS-ROC"],
                    -row["metric_means"]["R-based-F1"],
                    row["trial"],
                )
            )
            selections[f"{track}/{dataset}"] = ranking[0]
    stable = {
        "schema_version": "stage-locked-config-v1",
        "protocol": plan["protocol"],
        "selection_split": "official TSB-AD Tuning only",
        "selection_seed": int(plan["selection_seed"]),
        "eval_feedback": False,
        "one_config_per_track_dataset": True,
        "trials_per_subset": int(plan["trials_per_subset"]),
        "objective": plan["objective"],
        "tie_breakers": plan["tie_breakers"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "source_sha256": plan["source_sha256"],
        "selections": selections,
    }
    locked = {
        **stable,
        "locked_fingerprint": fingerprint(stable),
        "locked_at": utc_now(),
    }
    destinations = [tuning_root / LOCK_NAME]
    if export_path is not None:
        destinations.append(export_path)
    for destination in destinations:
        if destination.is_file():
            prior = load_json(destination)
            if prior.get("locked_fingerprint") != locked["locked_fingerprint"]:
                raise RuntimeError(f"refusing to overwrite a different locked config: {destination}")
        else:
            atomic_json(destination, locked)
    return locked


def validate_locked(repo: Path, locked_path: Path) -> dict[str, Any]:
    locked = load_json(locked_path)
    stable = {
        key: value
        for key, value in locked.items()
        if key not in {"locked_fingerprint", "locked_at"}
    }
    if fingerprint(stable) != locked.get("locked_fingerprint"):
        raise RuntimeError(f"locked-config fingerprint mismatch: {locked_path}")
    if sha256_file(repo / "STAGE" / "stage.py") != locked.get("source_sha256"):
        raise RuntimeError("STAGE source differs from the source used for parameter selection")
    if locked.get("eval_feedback") or locked.get("selection_split") != "official TSB-AD Tuning only":
        raise RuntimeError("locked config violates the no-Eval-feedback policy")
    return locked


def run_eval(
    repo: Path,
    result: Path,
    locked_path: Path,
    python: Path,
    metrics_root: Path,
    seed: int,
    gpus: Sequence[str],
    workers_per_gpu: int,
) -> None:
    locked = validate_locked(repo, locked_path)
    files = split_files(repo, "Eva")
    tasks: list[tuple[str, str, str, dict[str, Any], str]] = []
    for track, datasets in files.items():
        for dataset, names in datasets.items():
            selection = locked["selections"][f"{track}/{dataset}"]
            for file_name in names:
                target = eval_unit_path(result, track, dataset, file_name)
                if target.is_file():
                    item = load_json(target)
                    if (
                        finite_metrics(item)
                        and int(item.get("seed", -1)) == int(seed)
                        and item.get("locked_fingerprint") == locked["locked_fingerprint"]
                    ):
                        continue
                tasks.append(
                    (
                        track,
                        dataset,
                        file_name,
                        selection["parameters"],
                        selection["config_fingerprint"],
                    )
                )
    total = sum(len(names) for datasets in files.values() for names in datasets.values())
    counter = {"value": total - len(tasks)}
    counter_lock = threading.Lock()
    scratch = result / ".stage_scratch"

    def worker(gpu: str, task: tuple[Any, ...]) -> None:
        track, dataset, file_name, config, config_fingerprint = task
        outcome = run_stage_once(
            repo=repo,
            python=python,
            metrics_root=metrics_root,
            track=track,
            file_name=file_name,
            config=config,
            seed=seed,
            gpu=gpu,
            scratch_parent=scratch,
        )
        payload = {
            "method": "STAGE",
            "track": track,
            "dataset": dataset,
            "file": file_name,
            "seed": int(seed),
            "training_prefix_policy": "filename_declared_label_blind",
            "locked_fingerprint": locked["locked_fingerprint"],
            "config_fingerprint": config_fingerprint,
            "metrics": outcome["metrics"],
            "runtime_s": outcome["runtime_s"],
            "runtime_eligible_for_paper": False,
            "diagnostics": outcome["diagnostics"],
            "error": None,
        }
        atomic_json(eval_unit_path(result, track, dataset, file_name), payload)
        with counter_lock:
            counter["value"] += 1
            print(
                f"[Eval seed={seed} {counter['value']}/{total}] GPU={gpu} "
                f"{track}/{dataset} {file_name}",
                flush=True,
            )

    run_lanes(tasks, gpus, workers_per_gpu, worker)
    if scratch.exists():
        shutil.rmtree(scratch)
    atomic_json(
        result / f"stage_eval_seed{seed}_status.json",
        {
            "status": "complete",
            "seed": int(seed),
            "expected_units": total,
            "completed_units": total,
            "error_units": 0,
            "locked_fingerprint": locked["locked_fingerprint"],
            "completed_at": utc_now(),
        },
    )


def expected_baselines(repo: Path) -> list[str]:
    protocol = load_json(repo / "protocol.json")
    return [*protocol["non_deep_baselines"], *protocol["deep_baselines"]]


def baseline_completeness(repo: Path, result: Path, seed: int) -> dict[str, Any]:
    files = split_files(repo, "Eva")
    methods = expected_baselines(repo)
    missing: list[str] = []
    errors: list[str] = []
    for method in methods:
        for track, datasets in files.items():
            for dataset, names in datasets.items():
                for file_name in names:
                    path = result / "units" / method / track / dataset / f"{Path(file_name).stem}.json"
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    try:
                        item = load_json(path)
                        if int(item.get("seed", -1)) != seed or not complete_record(path):
                            errors.append(str(path))
                    except Exception:
                        errors.append(str(path))
    expected = len(methods) * sum(
        len(names) for datasets in files.values() for names in datasets.values()
    )
    return {
        "status": "complete" if not missing and not errors else "incomplete",
        "expected_units": expected,
        "valid_units": expected - len(missing) - len(errors),
        "missing_units": len(missing),
        "error_units": len(errors),
        "missing_sample": missing[:10],
        "error_sample": errors[:10],
    }


def compare_seed2026(repo: Path, result: Path, seed: int = 2026) -> dict[str, Any]:
    if seed != 2026:
        raise ValueError("the autonomous continuation gate is defined only for seed 2026")
    completeness = baseline_completeness(repo, result, seed)
    if completeness["status"] != "complete":
        raise RuntimeError(f"baseline result is incomplete: {completeness}")
    methods = [*expected_baselines(repo), "STAGE"]
    files = split_files(repo, "Eva")
    rows: list[dict[str, Any]] = []
    for method in methods:
        for track, datasets in files.items():
            for dataset, names in datasets.items():
                for file_name in names:
                    path = result / "units" / method / track / dataset / f"{Path(file_name).stem}.json"
                    if not path.is_file() or not complete_record(path):
                        raise RuntimeError(f"missing or invalid comparison unit: {path}")
                    item = load_json(path)
                    if int(item.get("seed", -1)) != seed:
                        raise RuntimeError(f"seed mismatch: {path}")
                    rows.append(
                        {
                            "method": method,
                            "track": track,
                            "dataset": dataset,
                            **{metric: float(item["metrics"][metric]) for metric in METRICS},
                        }
                    )
    frame = pd.DataFrame(rows)
    baselines = frame[frame.method != "STAGE"]
    target = frame[frame.method == "STAGE"]

    def cells(group_columns: list[str]) -> list[dict[str, Any]]:
        baseline_means = baselines.groupby([*group_columns, "method"])[list(METRICS)].mean()
        target_means = target.groupby(group_columns)[list(METRICS)].mean()
        output: list[dict[str, Any]] = []
        target_rows = target_means.reset_index().to_dict("records")
        for target_row in target_rows:
            selector: Any = tuple(target_row[column] for column in group_columns)
            if len(group_columns) == 1:
                selector = selector[0]
            subset = baseline_means.loc[selector]
            for metric in METRICS:
                best_method = str(subset[metric].idxmax())
                best_value = float(subset.loc[best_method, metric])
                stage_value = float(target_row[metric])
                output.append(
                    {
                        **{column: target_row[column] for column in group_columns},
                        "metric": metric,
                        "STAGE": stage_value,
                        "best_baseline": best_method,
                        "best_baseline_value": best_value,
                        "delta": stage_value - best_value,
                        "strict_win": stage_value > best_value,
                    }
                )
        return output

    # A synthetic constant grouping gives six global macro-per-series cells.
    frame["scope"] = "all"
    baselines = frame[frame.method != "STAGE"]
    target = frame[frame.method == "STAGE"]
    global_cells = cells(["scope"])
    track_cells = cells(["track"])
    dataset_cells = cells(["track", "dataset"])
    gate = {
        "definition": (
            "STAGE must strictly exceed the strongest admitted baseline on every "
            "one of the six global cells and all twelve U/M track-metric cells."
        ),
        "global_wins": sum(bool(item["strict_win"]) for item in global_cells),
        "global_cells": len(global_cells),
        "track_wins": sum(bool(item["strict_win"]) for item in track_cells),
        "track_cells": len(track_cells),
    }
    gate["passed"] = (
        gate["global_wins"] == gate["global_cells"]
        and gate["track_wins"] == gate["track_cells"]
    )
    report = {
        "schema_version": "stage-seed2026-comparison-v1",
        "seed": seed,
        "baseline_methods": expected_baselines(repo),
        "baseline_completeness": completeness,
        "gate": gate,
        "global": global_cells,
        "by_track": track_cells,
        "by_dataset": dataset_cells,
        "eval_feedback_used_for_selection": False,
        "created_at": utc_now(),
    }
    atomic_json(result / "stage_seed2026_comparison.json", report)
    return report


def parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpus:
        raise ValueError("provide at least one GPU")
    return gpus


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    tune = subparsers.add_parser("tune")
    add_runtime_args(tune)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--export-config", type=Path)
    evaluate = subparsers.add_parser("evaluate")
    add_runtime_args(evaluate)
    evaluate.add_argument("--locked-config", required=True, type=Path)
    evaluate.add_argument("--seed", required=True, type=int)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--seed", type=int, default=2026)
    autopilot = subparsers.add_parser("autopilot")
    add_runtime_args(autopilot)
    autopilot.add_argument("--export-config", required=True, type=Path)
    autopilot.add_argument("--next-seeds", default="2027,2028")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    result_root = args.result_root.resolve()
    tuning_root = result_root / "tuning_seed2026"
    if args.command == "plan":
        plan = build_plan(repo, tuning_root)
        print(json.dumps({"plan_fingerprint": plan["plan_fingerprint"]}, indent=2))
        return 0
    if args.command == "freeze":
        locked = freeze_config(repo, tuning_root, args.export_config)
        print(json.dumps({"locked_fingerprint": locked["locked_fingerprint"]}, indent=2))
        return 0
    if args.command == "compare":
        report = compare_seed2026(repo, result_root / f"seed{args.seed}", args.seed)
        print(json.dumps(report["gate"], indent=2))
        return 0 if report["gate"]["passed"] else 20

    # Do not resolve a venv executable symlink: doing so can silently replace
    # the selected environment with its base interpreter.
    python = Path(os.path.abspath(args.python))
    metrics_root = args.metrics_root.resolve()
    gpus = parse_gpus(args.gpus)
    if args.command == "tune":
        run_tuning(repo, tuning_root, python, metrics_root, gpus, args.workers_per_gpu)
        return 0
    if args.command == "evaluate":
        run_eval(
            repo,
            result_root / f"seed{args.seed}",
            args.locked_config.resolve(),
            python,
            metrics_root,
            args.seed,
            gpus,
            args.workers_per_gpu,
        )
        return 0
    if args.command == "autopilot":
        baseline_result = result_root / "seed2026"
        completeness = baseline_completeness(repo, baseline_result, 2026)
        atomic_json(baseline_result / "baseline_completeness_strict.json", completeness)
        if completeness["status"] != "complete":
            print(json.dumps(completeness, indent=2))
            return 10
        run_tuning(repo, tuning_root, python, metrics_root, gpus, args.workers_per_gpu)
        locked = freeze_config(repo, tuning_root, args.export_config.resolve())
        locked_path = tuning_root / LOCK_NAME
        run_eval(
            repo,
            baseline_result,
            locked_path,
            python,
            metrics_root,
            2026,
            gpus,
            args.workers_per_gpu,
        )
        report = compare_seed2026(repo, baseline_result, 2026)
        if not report["gate"]["passed"]:
            print(json.dumps(report["gate"], indent=2))
            return 20
        next_seeds = [int(item) for item in args.next_seeds.split(",") if item.strip()]
        for seed in next_seeds:
            if seed == 2026:
                continue
            run_eval(
                repo,
                result_root / f"seed{seed}",
                locked_path,
                python,
                metrics_root,
                seed,
                gpus,
                args.workers_per_gpu,
            )
        atomic_json(
            result_root / "stage_autopilot_complete.json",
            {
                "status": "complete",
                "selection_seed": 2026,
                "evaluated_seeds": [2026, *next_seeds],
                "locked_fingerprint": locked["locked_fingerprint"],
                "continuation_gate": report["gate"],
                "completed_at": utc_now(),
            },
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
