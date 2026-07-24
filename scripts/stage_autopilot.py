#!/usr/bin/env python3
"""Leakage-safe STAGE tuning, parameter locking, and multi-seed evaluation.

The workflow has deliberately separate stages:

1. require the frozen fourteen-baseline result to be strictly complete;
2. run the separately locked recent-baseline qualification and Eval barrier;
3. require all sixteen baseline methods to be strictly complete;
4. resume the deterministic STAGE Tuning plan on the official split;
5. freeze and hash one STAGE configuration per (track, dataset);
6. evaluate the frozen STAGE configuration on Eval;
7. continue to later seeds only when the predeclared seed-2026 gate passes.

No command persists checkpoints or anomaly-score arrays.  Eval results never
participate in parameter selection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
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

from common import METRICS, atomic_json, dataset_name


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
RECENT_METHODS: tuple[str, ...] = ("DPAD_AAAI24", "DNE_AAAI25")
RECENT_RESULT_NAME = "recent_seed2026"
RECENT_LOCK_RELATIVE = Path("configs/aaai_recent_locked_seed2026.json")
CONTROLLER_LOCK_NAME = ".stage_autopilot.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def controller_singleton(result_root: Path):
    """Hold one non-blocking OS lock for every STAGE controller command."""
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / CONTROLLER_LOCK_NAME
    handle = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(
                    f"another STAGE controller holds {lock_path}"
                ) from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    f"another STAGE controller holds {lock_path}"
                ) from exc
        locked = True
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps({"pid": os.getpid(), "acquired_at": utc_now()}, sort_keys=True)
            + "\n"
        )
        handle.flush()
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


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
    metrics = item.get("metrics")
    if item.get("error") not in (None, "") or not isinstance(metrics, Mapping):
        return False
    for name in METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
    return True


def valid_tuning_unit(
    item: Mapping[str, Any],
    *,
    track: str,
    dataset: str,
    trial: int,
    file_name: str,
    seed: int,
    config_fingerprint: str,
    source_sha256: str,
) -> bool:
    trial_value = item.get("trial")
    seed_value = item.get("seed")
    return finite_metrics(item) and all(
        (
            item.get("kind") == "STAGE official-Tuning trial",
            item.get("method") == "STAGE",
            item.get("track") == track,
            item.get("dataset") == dataset,
            item.get("file") == file_name,
            isinstance(trial_value, int) and not isinstance(trial_value, bool),
            trial_value == trial,
            isinstance(seed_value, int) and not isinstance(seed_value, bool),
            seed_value == seed,
            item.get("selection_split") == "Tuning",
            item.get("config_fingerprint") == config_fingerprint,
            item.get("source_sha256") == source_sha256,
        )
    )


def valid_stage_eval_unit(
    item: Mapping[str, Any],
    *,
    track: str,
    dataset: str,
    file_name: str,
    seed: int,
    locked_fingerprint: str,
    config_fingerprint: str,
) -> bool:
    seed_value = item.get("seed")
    return finite_metrics(item) and all(
        (
            item.get("method") == "STAGE",
            item.get("track") == track,
            item.get("dataset") == dataset,
            item.get("file") == file_name,
            isinstance(seed_value, int) and not isinstance(seed_value, bool),
            seed_value == seed,
            item.get("training_prefix_policy")
            == "filename_declared_label_blind",
            item.get("locked_fingerprint") == locked_fingerprint,
            item.get("config_fingerprint") == config_fingerprint,
            item.get("runtime_eligible_for_paper") is False,
        )
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
        "final_gb_min_split": "--final-gb-min-split",
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
                        if valid_tuning_unit(
                            item,
                            track=track,
                            dataset=dataset,
                            trial=trial,
                            file_name=file_name,
                            seed=int(plan["selection_seed"]),
                            config_fingerprint=candidate["config_fingerprint"],
                            source_sha256=plan["source_sha256"],
                        ):
                            continue
                        raise RuntimeError(
                            f"refusing to overwrite invalid existing Tuning unit: {target}"
                        )
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
                    if not valid_tuning_unit(
                        item,
                        track=track,
                        dataset=dataset,
                        trial=trial,
                        file_name=file_name,
                        seed=int(plan["selection_seed"]),
                        config_fingerprint=candidate["config_fingerprint"],
                        source_sha256=plan["source_sha256"],
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


def validate_recent_locked(
    repo: Path, locked_path: Path, seed: int = 2026
) -> dict[str, Any]:
    """Validate the immutable, no-Eval-feedback contract for recent baselines."""
    locked = load_json(locked_path)
    if locked.get("status") != "frozen":
        raise RuntimeError(f"recent-baseline config is not frozen: {locked_path}")
    if int(locked.get("seed", -1)) != int(seed):
        raise RuntimeError(f"recent-baseline seed mismatch: {locked_path}")
    if locked.get("eval_feedback") is not False:
        raise RuntimeError("recent-baseline lock must explicitly prohibit Eval feedback")
    if locked.get("selection_split") != "official TSB-AD Tuning only":
        raise RuntimeError("recent-baseline lock must select only on official Tuning")
    if (
        int(locked.get("extension_method_count", -1)) != len(RECENT_METHODS)
        or int(locked.get("series_count", -1)) != 193
        or int(locked.get("extension_unit_count", -1)) != len(RECENT_METHODS) * 193
    ):
        raise RuntimeError("recent-baseline frozen method/series/unit counts are invalid")
    stable = {
        key: value
        for key, value in locked.items()
        if key not in {"locked_fingerprint", "locked_at"}
    }
    if fingerprint(stable) != locked.get("locked_fingerprint"):
        raise RuntimeError(f"recent-baseline lock fingerprint mismatch: {locked_path}")

    methods = locked.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(RECENT_METHODS):
        raise RuntimeError(
            f"recent-baseline lock must contain exactly {list(RECENT_METHODS)}"
        )
    for method in RECENT_METHODS:
        specification = methods[method]
        if not isinstance(specification, Mapping):
            raise RuntimeError(f"invalid recent-baseline specification: {method}")
        parameters = specification.get("formal_hyperparameters")
        if not isinstance(parameters, Mapping) or not parameters:
            raise RuntimeError(f"formal hyperparameters are not frozen for {method}")
        if fingerprint(parameters) != specification.get("config_fingerprint"):
            raise RuntimeError(f"recent-baseline config fingerprint mismatch: {method}")
        if specification.get("formal_eval_eligible") is not True:
            raise RuntimeError(f"recent-baseline Eval eligibility is not frozen: {method}")

    plan_fingerprint = locked.get("plan_fingerprint")
    if not isinstance(plan_fingerprint, str) or len(plan_fingerprint) != 64:
        raise RuntimeError("recent-baseline plan fingerprint is absent or malformed")
    try:
        int(plan_fingerprint, 16)
    except ValueError as exc:
        raise RuntimeError("recent-baseline plan fingerprint is malformed") from exc
    if not isinstance(locked.get("plan"), Mapping):
        raise RuntimeError("recent-baseline plan contract is absent")
    if fingerprint(locked["plan"]) != plan_fingerprint:
        raise RuntimeError("recent-baseline plan fingerprint mismatch")

    source_files = locked.get("source_files")
    source_sha256 = locked.get("source_sha256")
    if (
        not isinstance(source_files, Mapping)
        or not isinstance(source_sha256, Mapping)
        or set(source_files) != set(source_sha256)
    ):
        raise RuntimeError("recent-baseline source path/hash maps are absent or differ")
    resolved_sources: set[Path] = set()
    for name, relative in source_files.items():
        source = (repo / str(relative)).resolve()
        if not source.is_relative_to(repo.resolve()):
            raise RuntimeError(f"recent-baseline source escapes repository: {relative}")
        if not source.is_file() or sha256_file(source) != source_sha256.get(name):
            raise RuntimeError(f"recent-baseline {name} source hash mismatch")
        resolved_sources.add(source)
    required_sources = {
        (repo / "extensions" / "aaai_recent" / "run_one_recent.py").resolve(),
        (repo / "extensions" / "aaai_recent" / "models.py").resolve(),
        (repo / "scripts" / "recent_baseline_autopilot.py").resolve(),
    }
    if not required_sources.issubset(resolved_sources):
        raise RuntimeError("recent-baseline source map is incomplete")

    base_protocol = (locked_path.parent / str(locked["base_protocol"])).resolve()
    if base_protocol != (repo / "protocol.json").resolve():
        raise RuntimeError("recent-baseline lock points to an unexpected base protocol")
    if sha256_file(base_protocol) != locked.get("base_protocol_sha256"):
        raise RuntimeError("recent-baseline base-protocol hash mismatch")
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
                    if valid_stage_eval_unit(
                        item,
                        track=track,
                        dataset=dataset,
                        file_name=file_name,
                        seed=seed,
                        locked_fingerprint=locked["locked_fingerprint"],
                        config_fingerprint=selection["config_fingerprint"],
                    ):
                        continue
                    raise RuntimeError(
                        f"refusing to overwrite invalid existing STAGE Eval unit: {target}"
                    )
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


def expected_base_baselines(repo: Path) -> list[str]:
    protocol = load_json(repo / "protocol.json")
    return [*protocol["non_deep_baselines"], *protocol["deep_baselines"]]


def expected_baselines(repo: Path) -> list[str]:
    """Return the complete comparison set after the recent-baseline barrier."""
    return [*expected_base_baselines(repo), *RECENT_METHODS]


def baseline_unit_path(
    root: Path, method: str, track: str, dataset: str, file_name: str
) -> Path:
    return root / "units" / method / track / dataset / f"{Path(file_name).stem}.json"


def recent_eval_result_root(recent_result_root: Path, seed: int = 2026) -> Path:
    return recent_result_root / f"eval_seed{seed}"


def _strict_baseline_root_completeness(
    repo: Path,
    result: Path,
    seed: int,
    methods: Sequence[str],
    *,
    locked_fingerprint: str | None = None,
    config_fingerprints: Mapping[str, str] | None = None,
    plan_fingerprint: str | None = None,
    required_split: str | None = None,
) -> dict[str, Any]:
    files = split_files(repo, "Eva")
    expected: dict[tuple[str, str, str, str], Path] = {}
    for method in methods:
        for track, datasets in files.items():
            for dataset, names in datasets.items():
                for file_name in names:
                    key = (method, track, dataset, file_name)
                    expected[key] = baseline_unit_path(
                        result, method, track, dataset, file_name
                    )

    observed: dict[tuple[str, str, str, str], list[Path]] = {}
    units_root = result / "units"
    if units_root.is_dir():
        for path in units_root.rglob("*.json"):
            try:
                item = load_json(path)
            except Exception:
                continue
            if item.get("method") not in methods:
                continue
            key = (
                str(item.get("method")),
                str(item.get("track")),
                str(item.get("dataset")),
                str(item.get("file")),
            )
            observed.setdefault(key, []).append(path)

    missing: list[str] = []
    errors: list[str] = []
    unexpected: list[str] = []
    valid_by_method = {method: 0 for method in methods}
    for key, paths in observed.items():
        if key not in expected:
            unexpected.extend(str(path) for path in paths)
        elif len(paths) != 1 or paths[0].resolve() != expected[key].resolve():
            errors.extend(str(path) for path in paths)

    for key, path in expected.items():
        method, track, dataset, file_name = key
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            item = load_json(path)
            item_seed = item.get("seed")
            identity_ok = (
                item.get("method") == method
                and item.get("track") == track
                and item.get("dataset") == dataset
                and item.get("file") == file_name
                and isinstance(item_seed, int)
                and not isinstance(item_seed, bool)
                and item_seed == seed
            )
            provenance_ok = (
                locked_fingerprint is None
                or item.get("locked_fingerprint") == locked_fingerprint
            )
            config_ok = (
                config_fingerprints is None
                or item.get("config_fingerprint") == config_fingerprints[method]
            )
            plan_ok = (
                plan_fingerprint is None
                or item.get("plan_fingerprint") == plan_fingerprint
            )
            split_ok = required_split is None or item.get("split") == required_split
            unique_ok = len(observed.get(key, [])) == 1
            if not (
                identity_ok
                and finite_metrics(item)
                and provenance_ok
                and config_ok
                and plan_ok
                and split_ok
                and unique_ok
            ):
                errors.append(str(path))
                continue
        except Exception:
            errors.append(str(path))
            continue
        valid_by_method[method] += 1

    expected_units = len(expected)
    valid_units = sum(valid_by_method.values())
    status = (
        "complete"
        if valid_units == expected_units and not missing and not errors and not unexpected
        else "incomplete"
    )
    return {
        "status": status,
        "methods": list(methods),
        "expected_units": expected_units,
        "valid_units": valid_units,
        "missing_units": len(missing),
        "error_units": len(set(errors)),
        "unexpected_units": len(set(unexpected)),
        "by_method": valid_by_method,
        "missing_sample": missing[:10],
        "error_sample": list(dict.fromkeys(errors))[:10],
        "unexpected_sample": list(dict.fromkeys(unexpected))[:10],
    }


def baseline_completeness(repo: Path, result: Path, seed: int) -> dict[str, Any]:
    """Strictly scan only the original fourteen-baseline frozen result."""
    report = _strict_baseline_root_completeness(
        repo, result, seed, expected_base_baselines(repo)
    )
    if len(report["methods"]) != 14 or int(report["expected_units"]) != 14 * 193:
        report["status"] = "incomplete"
        report["protocol_count_error"] = (
            "the frozen base gate must contain exactly 14 methods and 2702 units"
        )
    return report


def combined_baseline_completeness(
    repo: Path,
    base_result: Path,
    recent_result: Path,
    recent_locked_config: Path,
    seed: int = 2026,
) -> dict[str, Any]:
    """Require exactly 14 base + 2 recent methods before STAGE Eval starts."""
    base_resolved = base_result.resolve()
    recent_resolved = recent_result.resolve()
    if (
        base_resolved == recent_resolved
        or base_resolved in recent_resolved.parents
        or recent_resolved in base_resolved.parents
    ):
        raise RuntimeError("base and recent baseline result roots must be independent")
    recent_lock = validate_recent_locked(repo, recent_locked_config, seed)
    base = baseline_completeness(repo, base_result, seed)
    try:
        from scripts.recent_baseline_autopilot import COMPLETE_NAME, audit_all
    except ImportError as exc:
        raise RuntimeError("recent-baseline strict auditor is unavailable") from exc
    recent_audit = audit_all(repo, recent_result, recent_locked_config)
    completion_path = recent_result / COMPLETE_NAME
    if not completion_path.is_file():
        raise RuntimeError(f"recent-baseline completion marker is absent: {completion_path}")
    recent_completion = load_json(completion_path)
    completion_stable = {
        key: value
        for key, value in recent_completion.items()
        if key not in {"completion_fingerprint", "completed_at"}
    }
    completion_ok = (
        fingerprint(completion_stable)
        == recent_completion.get("completion_fingerprint")
        and recent_completion.get("status") == "complete"
        and int(recent_completion.get("seed", -1)) == int(seed)
        and recent_completion.get("methods") == list(RECENT_METHODS)
        and recent_completion.get("locked_fingerprint")
        == recent_lock["locked_fingerprint"]
        and recent_completion.get("contract_plan_fingerprint")
        == recent_lock["plan_fingerprint"]
        and recent_completion.get("eval_feedback") is False
        and recent_completion.get("runtime_eligible_for_paper") is False
    )
    phase_expectations = {"tuning": 44, "eval": 386}
    for phase, count in phase_expectations.items():
        phase_marker = recent_completion.get(phase)
        phase_audit = recent_audit.get(phase)
        marker_ok = isinstance(phase_marker, Mapping) and all(
            (
                phase_marker.get("status") == "complete",
                int(phase_marker.get("expected_units", -1)) == count,
                int(phase_marker.get("valid_units", -1)) == count,
                int(phase_marker.get("error_units", -1)) == 0,
            )
        )
        audit_ok = isinstance(phase_audit, Mapping) and all(
            (
                phase_audit.get("status") == "complete",
                int(phase_audit.get("valid_units", -1)) == count,
                int(phase_audit.get("error_units", -1)) == 0,
            )
        )
        completion_ok = completion_ok and marker_ok and audit_ok
    completion_ok = completion_ok and recent_audit.get("failstop_present") is False
    config_fingerprints = {
        method: str(recent_lock["methods"][method]["config_fingerprint"])
        for method in RECENT_METHODS
    }
    recent = _strict_baseline_root_completeness(
        repo,
        recent_eval_result_root(recent_result, seed),
        seed,
        RECENT_METHODS,
        locked_fingerprint=str(recent_lock["locked_fingerprint"]),
        config_fingerprints=config_fingerprints,
        plan_fingerprint=str(recent_lock["plan_fingerprint"]),
        required_split="Eval",
    )
    expected_units = int(base["expected_units"]) + int(recent["expected_units"])
    valid_units = int(base["valid_units"]) + int(recent["valid_units"])
    cross_root_units = [
        str(base_result / "units" / method)
        for method in RECENT_METHODS
        if (base_result / "units" / method).exists()
    ]
    cross_root_units.extend(
        str(recent_eval_result_root(recent_result, seed) / "units" / method)
        for method in expected_base_baselines(repo)
        if (
            recent_eval_result_root(recent_result, seed) / "units" / method
        ).exists()
    )
    status = (
        "complete"
        if base["status"] == recent["status"] == "complete"
        and completion_ok
        and len(expected_baselines(repo)) == 16
        and expected_units == 16 * 193
        and not cross_root_units
        else "incomplete"
    )
    return {
        "status": status,
        "seed": int(seed),
        "baseline_methods": expected_baselines(repo),
        "expected_units": expected_units,
        "valid_units": valid_units,
        "missing_units": int(base["missing_units"]) + int(recent["missing_units"]),
        "error_units": int(base["error_units"]) + int(recent["error_units"]),
        "unexpected_units": int(base["unexpected_units"])
        + int(recent["unexpected_units"]),
        "cross_root_method_directories": cross_root_units,
        "recent_locked_fingerprint": recent_lock["locked_fingerprint"],
        "recent_eval_result_root": str(recent_eval_result_root(recent_result, seed)),
        "recent_barrier_audit": recent_audit,
        "recent_completion_marker_valid": completion_ok,
        "base": base,
        "recent": recent,
    }


def invoke_recent_baseline_barrier(
    *,
    repo: Path,
    recent_result_root: Path,
    python: Path,
    metrics_root: Path,
    gpus: Sequence[str],
    workers_per_gpu: int,
    locked_config: Path,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run the independent qualification/Eval barrier through a narrow API."""
    if workers_per_gpu != 1:
        raise ValueError(
            "recent baselines require exactly one isolated worker per physical GPU"
        )
    validate_recent_locked(repo, locked_config, seed)
    try:
        from scripts.recent_baseline_autopilot import run_recent_baseline_barrier
    except ImportError as exc:
        raise RuntimeError(
            "scripts.recent_baseline_autopilot.run_recent_baseline_barrier "
            "is required before STAGE Eval"
        ) from exc

    summary = run_recent_baseline_barrier(
        repo=repo,
        recent_result_root=recent_result_root,
        python=python,
        metrics_root=metrics_root,
        seed=seed,
        gpus=tuple(gpus),
        workers_per_gpu=workers_per_gpu,
        locked_config=locked_config,
    )
    if not isinstance(summary, Mapping) or summary.get("status") != "complete":
        raise RuntimeError(f"recent-baseline barrier did not complete: {summary!r}")
    expected = {"tuning": 44, "eval": 386}
    for phase, expected_units in expected.items():
        phase_summary = summary.get(phase)
        if not isinstance(phase_summary, Mapping) or not (
            phase_summary.get("status") == "complete"
            and int(phase_summary.get("expected_units", -1)) == expected_units
            and int(phase_summary.get("valid_units", -1)) == expected_units
            and int(phase_summary.get("error_units", -1)) == 0
        ):
            raise RuntimeError(
                f"recent-baseline {phase} qualification is invalid: {phase_summary!r}"
            )
    return dict(summary)


def compare_seed2026(
    repo: Path,
    result: Path,
    recent_result: Path,
    recent_locked_config: Path,
    seed: int = 2026,
) -> dict[str, Any]:
    if seed != 2026:
        raise ValueError("the autonomous continuation gate is defined only for seed 2026")
    completeness = combined_baseline_completeness(
        repo, result, recent_result, recent_locked_config, seed
    )
    if completeness["status"] != "complete":
        raise RuntimeError(f"baseline result is incomplete: {completeness}")
    stage_locked_path = result.parent / "tuning_seed2026" / LOCK_NAME
    stage_locked = validate_locked(repo, stage_locked_path)
    methods = [*expected_baselines(repo), "STAGE"]
    files = split_files(repo, "Eva")
    rows: list[dict[str, Any]] = []
    for method in methods:
        for track, datasets in files.items():
            for dataset, names in datasets.items():
                for file_name in names:
                    method_root = (
                        recent_eval_result_root(recent_result, seed)
                        if method in RECENT_METHODS
                        else result
                    )
                    path = baseline_unit_path(
                        method_root, method, track, dataset, file_name
                    )
                    if not path.is_file():
                        raise RuntimeError(f"missing or invalid comparison unit: {path}")
                    item = load_json(path)
                    item_seed = item.get("seed")
                    identity_ok = (
                        item.get("method") == method
                        and item.get("track") == track
                        and item.get("dataset") == dataset
                        and item.get("file") == file_name
                        and isinstance(item_seed, int)
                        and not isinstance(item_seed, bool)
                        and item_seed == seed
                        and finite_metrics(item)
                    )
                    if method == "STAGE":
                        selection = stage_locked["selections"][f"{track}/{dataset}"]
                        identity_ok = valid_stage_eval_unit(
                            item,
                            track=track,
                            dataset=dataset,
                            file_name=file_name,
                            seed=seed,
                            locked_fingerprint=stage_locked["locked_fingerprint"],
                            config_fingerprint=selection["config_fingerprint"],
                        )
                    if not identity_ok:
                        raise RuntimeError(f"invalid comparison identity or metrics: {path}")
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
        "schema_version": "stage-seed2026-comparison-v2",
        "seed": seed,
        "baseline_methods": expected_baselines(repo),
        "base_baseline_methods": expected_base_baselines(repo),
        "recent_baseline_methods": list(RECENT_METHODS),
        "recent_locked_fingerprint": completeness["recent_locked_fingerprint"],
        "stage_locked_fingerprint": stage_locked["locked_fingerprint"],
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


def add_recent_result_args(
    parser: argparse.ArgumentParser, *, include_workers: bool = False
) -> None:
    parser.add_argument(
        "--recent-result-root",
        type=Path,
        help=f"separate recent-baseline root (default: <result-root>/{RECENT_RESULT_NAME})",
    )
    parser.add_argument(
        "--recent-locked-config",
        type=Path,
        help=f"frozen recent-baseline contract (default: <repo>/{RECENT_LOCK_RELATIVE})",
    )
    if include_workers:
        parser.add_argument("--recent-workers-per-gpu", type=int, default=1)


def recent_paths(
    repo: Path, result_root: Path, args: argparse.Namespace
) -> tuple[Path, Path]:
    recent_result = (
        args.recent_result_root.resolve()
        if args.recent_result_root is not None
        else (result_root / RECENT_RESULT_NAME).resolve()
    )
    recent_locked = (
        args.recent_locked_config.resolve()
        if args.recent_locked_config is not None
        else (repo / RECENT_LOCK_RELATIVE).resolve()
    )
    return recent_result, recent_locked


def validate_continuation_report(
    repo: Path,
    result_root: Path,
    stage_locked_path: Path,
    recent_locked_path: Path,
) -> dict[str, Any]:
    """Require the exact seed-2026 gate before any later-seed Eval command."""
    stage_locked = validate_locked(repo, stage_locked_path)
    recent_locked = validate_recent_locked(repo, recent_locked_path, 2026)
    report_path = result_root / "seed2026" / "stage_seed2026_comparison.json"
    if not report_path.is_file():
        raise RuntimeError(f"seed-2026 comparison gate is absent: {report_path}")
    report = load_json(report_path)
    gate = report.get("gate")
    if not (
        report.get("schema_version") == "stage-seed2026-comparison-v2"
        and int(report.get("seed", -1)) == 2026
        and isinstance(gate, Mapping)
        and gate.get("passed") is True
        and int(gate.get("global_wins", -1)) == int(gate.get("global_cells", -2)) == 6
        and int(gate.get("track_wins", -1)) == int(gate.get("track_cells", -2)) == 12
        and report.get("baseline_methods") == expected_baselines(repo)
        and report.get("stage_locked_fingerprint")
        == stage_locked["locked_fingerprint"]
        and report.get("recent_locked_fingerprint")
        == recent_locked["locked_fingerprint"]
        and report.get("eval_feedback_used_for_selection") is False
    ):
        raise RuntimeError("seed-2026 continuation gate is absent, stale, or failed")
    return report


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
    add_recent_result_args(evaluate)
    evaluate.add_argument("--locked-config", required=True, type=Path)
    evaluate.add_argument("--seed", required=True, type=int)
    compare = subparsers.add_parser("compare")
    add_recent_result_args(compare)
    compare.add_argument("--seed", type=int, default=2026)
    autopilot = subparsers.add_parser("autopilot")
    add_runtime_args(autopilot)
    add_recent_result_args(autopilot, include_workers=True)
    autopilot.add_argument("--export-config", required=True, type=Path)
    autopilot.add_argument("--next-seeds", default="2027,2028")
    return parser.parse_args(argv)


def _execute(
    args: argparse.Namespace, repo: Path, result_root: Path
) -> int:
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
        recent_result, recent_locked = recent_paths(repo, result_root, args)
        report = compare_seed2026(
            repo,
            result_root / f"seed{args.seed}",
            recent_result,
            recent_locked,
            args.seed,
        )
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
        if args.seed not in {2026, 2027, 2028}:
            raise RuntimeError("formal STAGE Eval is predeclared only for seeds 2026-2028")
        recent_result, recent_locked = recent_paths(repo, result_root, args)
        if args.seed == 2026:
            completeness = combined_baseline_completeness(
                repo,
                result_root / "seed2026",
                recent_result,
                recent_locked,
                2026,
            )
            atomic_json(
                result_root / "seed2026" / "baseline_completeness_16_strict.json",
                completeness,
            )
            if completeness["status"] != "complete":
                print(json.dumps(completeness, indent=2))
                return 10
        else:
            validate_continuation_report(
                repo,
                result_root,
                args.locked_config.resolve(),
                recent_locked,
            )
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
        recent_result, recent_locked = recent_paths(repo, result_root, args)
        recent_summary = invoke_recent_baseline_barrier(
            repo=repo,
            recent_result_root=recent_result,
            python=python,
            metrics_root=metrics_root,
            gpus=gpus,
            workers_per_gpu=args.recent_workers_per_gpu,
            locked_config=recent_locked,
            seed=2026,
        )
        completeness = combined_baseline_completeness(
            repo, baseline_result, recent_result, recent_locked, 2026
        )
        atomic_json(
            baseline_result / "baseline_completeness_16_strict.json", completeness
        )
        if completeness["status"] != "complete":
            print(json.dumps(completeness, indent=2))
            return 11
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
        report = compare_seed2026(
            repo, baseline_result, recent_result, recent_locked, 2026
        )
        if not report["gate"]["passed"]:
            print(json.dumps(report["gate"], indent=2))
            return 20
        next_seeds = [int(item) for item in args.next_seeds.split(",") if item.strip()]
        if next_seeds != [2027, 2028]:
            raise RuntimeError("autonomous continuation seeds are frozen to 2027,2028")
        for seed in next_seeds:
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
                "recent_locked_fingerprint": completeness[
                    "recent_locked_fingerprint"
                ],
                "baseline_method_count": len(completeness["baseline_methods"]),
                "baseline_valid_units": completeness["valid_units"],
                "recent_barrier": recent_summary,
                "continuation_gate": report["gate"],
                "completed_at": utc_now(),
            },
        )
        return 0
    raise AssertionError(args.command)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    result_root = args.result_root.resolve()
    with controller_singleton(result_root):
        return _execute(args, repo, result_root)


if __name__ == "__main__":
    raise SystemExit(main())
