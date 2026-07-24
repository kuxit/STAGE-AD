#!/usr/bin/env python3
"""Evaluate the frozen dataset-specific STAGE Tuning lock on official Eval.

This runner is deliberately separate from the VUS-PR search runner.  It first
materializes an immutable ten-dataset lock from the completed Stage2 selection,
then fingerprints the official Eval manifests, every Eval series, the STAGE
source, and the official evaluator before any GPU work starts.

Eval metrics never participate in parameter selection.  Units are atomic JSON
records; checkpoints, embeddings, anomaly-score arrays, and paper runtime
claims are not produced.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import METRICS, atomic_json, dataset_name  # noqa: E402
from stage_autopilot import run_stage_once  # noqa: E402


LOCK_SCHEMA = "stage-vuspr-eval-lock-v1"
PLAN_SCHEMA = "stage-vuspr-eval-plan-v1"
UNIT_SCHEMA = "stage-vuspr-eval-unit-v1"
STATUS_SCHEMA = "stage-vuspr-eval-status-v1"
SUMMARY_SCHEMA = "stage-vuspr-eval-summary-v1"
LOCK_NAME = "stage_vuspr_eval_lock.json"
PLAN_NAME = "stage_vuspr_eval_plan.json"
STATUS_NAME = "stage_vuspr_eval_status.json"
SUMMARY_JSON_NAME = "stage_vuspr_eval_summary.json"
SUMMARY_CSV_NAME = "stage_vuspr_eval_summary.csv"
CONTROLLER_LOCK_NAME = ".stage_vuspr_eval.lock"
SEED = 2026
SELECTION_SPLIT = "official TSB-AD Tuning only"
EVALUATION_SPLIT = "official TSB-AD Eval"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_TARGETS = {
    "U": ["UCR", "Exathlon", "MSL", "SED", "TODS"],
    "M": ["CATSv2", "GHL", "LTDB", "SVDB", "TAO"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selection_stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key not in {"selection_fingerprint", "created_at"}
    }


def _lock_stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key not in {"lock_fingerprint", "created_at"}
    }


def _plan_stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key not in {"plan_fingerprint", "created_at"}
    }


def _source_hashes(repo: Path) -> dict[str, str]:
    relative = (
        "STAGE/stage.py",
        "common.py",
        "scripts/stage_autopilot.py",
        "scripts/stage_vuspr_eval.py",
    )
    return {name: sha256_file(repo / name) for name in relative}


def _evaluator_hashes(metrics_root: Path) -> tuple[str, dict[str, str]]:
    root = metrics_root / "PaAno"
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(
        path for path in root.rglob("*.py") if path.is_file() and "__pycache__" not in path.parts
    )
    if not files:
        raise RuntimeError(f"no evaluator sources found under {root}")
    return (
        root.name,
        {path.relative_to(root).as_posix(): sha256_file(path) for path in files},
    )


def _candidate_map(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = protocol.get("training_candidates")
    if not isinstance(candidates, list):
        raise ValueError("Stage2 protocol has no training_candidates list")
    mapped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError("training candidate must be an object")
        candidate_id = item.get("id")
        parameters = item.get("parameters")
        if not isinstance(candidate_id, str) or not isinstance(parameters, Mapping):
            raise ValueError("training candidate id/parameters are malformed")
        if candidate_id in mapped:
            raise ValueError(f"duplicate training candidate id: {candidate_id}")
        mapped[candidate_id] = dict(parameters)
    return mapped


def build_lock(
    repo: Path,
    stage2_protocol_path: Path,
    stage2_result: Path,
    destination: Path,
) -> dict[str, Any]:
    protocol = load_json(stage2_protocol_path)
    selection_path = stage2_result / "stage_vuspr_search_selection.json"
    summary_path = stage2_result / "stage_vuspr_search_summary.json"
    plan_path = stage2_result / "stage_vuspr_search_plan.json"
    selection = load_json(selection_path)
    summary = load_json(summary_path)
    stage2_plan = load_json(plan_path)

    if (
        protocol.get("phase") != "stage2"
        or selection.get("phase") != "stage2"
        or summary.get("phase") != "stage2"
    ):
        raise RuntimeError("the lock source is not a completed Stage2 result")
    if (
        selection.get("selection_split") != SELECTION_SPLIT
        or selection.get("eval_feedback") is not False
        or summary.get("selection_split") != SELECTION_SPLIT
        or summary.get("eval_feedback") is not False
    ):
        raise RuntimeError("Stage2 selection is not Tuning-only")
    if fingerprint(_selection_stable(selection)) != selection.get("selection_fingerprint"):
        raise RuntimeError("Stage2 selection fingerprint mismatch")
    if summary.get("selection") != selection:
        raise RuntimeError("Stage2 summary and selection JSON disagree")
    if (
        int(summary.get("completed_units", -1)) != int(summary.get("expected_units", -2))
        or int(summary.get("completed_units", -1)) != int(stage2_plan.get("expected_units", -2))
    ):
        raise RuntimeError("Stage2 result is incomplete")
    if selection.get("plan_fingerprint") != stage2_plan.get("plan_fingerprint"):
        raise RuntimeError("Stage2 selection/plan lineage mismatch")
    if sha256_file(stage2_protocol_path) != stage2_plan.get("protocol_sha256"):
        raise RuntimeError("Stage2 protocol hash differs from its frozen plan")

    candidates = _candidate_map(protocol)
    expected_subsets = {
        f"{track}/{dataset}"
        for track, datasets in EXPECTED_TARGETS.items()
        for dataset in datasets
    }
    training_winners = selection.get("training_winners")
    selected_heads = selection.get("selected_heads")
    if (
        not isinstance(training_winners, Mapping)
        or set(training_winners) != expected_subsets
        or not isinstance(selected_heads, Mapping)
        or set(selected_heads) != expected_subsets
    ):
        raise RuntimeError("Stage2 does not contain ten dataset-specific selections")

    selections: dict[str, dict[str, Any]] = {}
    for subset in sorted(expected_subsets):
        candidate_id = training_winners[subset]
        head = selected_heads[subset]
        if candidate_id not in candidates or not isinstance(head, Mapping):
            raise RuntimeError(f"invalid Stage2 selection for {subset}")
        parameters = dict(candidates[candidate_id])
        parameters["final_gb_min_split"] = int(head["final_gb_min_split"])
        parameters["top_k"] = int(head["top_k"])
        selections[subset] = {
            "training_candidate_id": candidate_id,
            "head_id": str(head["head_id"]),
            "parameters": parameters,
            "config_fingerprint": fingerprint(parameters),
        }

    stable = {
        "schema_version": LOCK_SCHEMA,
        "method": "STAGE",
        "seed": SEED,
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "dataset_specific": True,
        "track_fallback": False,
        "stage2_protocol_sha256": sha256_file(stage2_protocol_path),
        "stage2_plan_sha256": sha256_file(plan_path),
        "stage2_plan_fingerprint": stage2_plan["plan_fingerprint"],
        "stage2_summary_sha256": sha256_file(summary_path),
        "stage2_selection_sha256": sha256_file(selection_path),
        "stage2_selection_fingerprint": selection["selection_fingerprint"],
        "tuning_source_sha256": stage2_plan["source_sha256"],
        "eval_source_sha256": _source_hashes(repo),
        "selections": selections,
    }
    payload = {
        **stable,
        "lock_fingerprint": fingerprint(stable),
        "created_at": utc_now(),
    }
    if destination.is_file():
        prior = load_json(destination)
        if prior.get("lock_fingerprint") != payload["lock_fingerprint"]:
            raise RuntimeError(f"refusing to overwrite a different Eval lock: {destination}")
        return prior
    atomic_json(destination, payload)
    return payload


def _finite_variant_metrics(variant: Mapping[str, Any]) -> bool:
    metrics = variant.get("metrics")
    return isinstance(metrics, Mapping) and set(metrics) == set(METRICS) and all(
        isinstance(metrics[name], (int, float))
        and not isinstance(metrics[name], bool)
        and math.isfinite(float(metrics[name]))
        for name in METRICS
    )


def build_partial_lock(
    repo: Path,
    stage2_protocol_path: Path,
    stage2_result: Path,
    destination: Path,
) -> dict[str, Any]:
    """Freeze every independently complete Stage2 dataset without CATS fallback."""

    protocol = load_json(stage2_protocol_path)
    stage2_plan_path = stage2_result / "stage_vuspr_search_plan.json"
    stage2_plan = load_json(stage2_plan_path)
    if protocol.get("phase") != "stage2" or stage2_plan.get("phase") != "stage2":
        raise RuntimeError("partial lock source is not Stage2")
    if (
        protocol.get("selection_split") != SELECTION_SPLIT
        or protocol.get("eval_feedback_used_by_runner") is not False
        or stage2_plan.get("selection_split") != SELECTION_SPLIT
    ):
        raise RuntimeError("partial lock source is not Tuning-only")
    if sha256_file(stage2_protocol_path) != stage2_plan.get("protocol_sha256"):
        raise RuntimeError("Stage2 protocol differs from its frozen plan")
    if stage2_plan.get("source_sha256") != {
        name: sha256_file(stage2_protocol_path.parents[1] / name)
        for name in stage2_plan.get("source_sha256", {})
    }:
        raise RuntimeError("frozen Stage2 source drift")

    candidates = _candidate_map(protocol)
    training_winners = stage2_plan.get("training_winners")
    if not isinstance(training_winners, Mapping):
        raise RuntimeError("Stage2 plan has no frozen training winners")

    unit_members: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    units_root = stage2_result / "units"
    for path in sorted(units_root.rglob("*.json")):
        item = load_json(path)
        subset = f"{item.get('track')}/{item.get('dataset')}"
        variants = item.get("variants")
        if (
            item.get("schema_version") != "stage-vuspr-search-unit-v1"
            or item.get("selection_split") != "Tuning"
            or item.get("eval_feedback") is not False
            or item.get("plan_fingerprint") != stage2_plan.get("plan_fingerprint")
            or item.get("source_sha256") != stage2_plan.get("source_sha256")
            or item.get("error") is not None
            or not isinstance(variants, list)
            or len(variants) != 20
        ):
            raise RuntimeError(f"invalid Stage2 unit while building partial lock: {path}")
        observed_heads = {
            (int(variant["final_gb_min_split"]), int(variant["top_k"]))
            for variant in variants
            if _finite_variant_metrics(variant)
        }
        expected_heads = {
            (int(split), int(top_k))
            for split in stage2_plan["final_gb_min_splits"]
            for top_k in stage2_plan["top_ks"]
        }
        if observed_heads != expected_heads:
            raise RuntimeError(f"incomplete Stage2 head coverage: {path}")
        data_key = f"{item['track']}/{item['dataset']}/{item['file']}"
        if item.get("data_sha256") != stage2_plan["data_sha256"].get(data_key):
            raise RuntimeError(f"Stage2 unit data lineage mismatch: {path}")
        unit_members.setdefault(subset, []).append((path, item))

    selections: dict[str, dict[str, Any]] = {}
    used_unit_hashes: dict[str, str] = {}
    incomplete_subsets: dict[str, dict[str, int]] = {}
    for track, datasets in stage2_plan["files"].items():
        for dataset, files in datasets.items():
            subset = f"{track}/{dataset}"
            expected_keys = {
                (file_name, int(seed))
                for file_name in files
                for seed in stage2_plan["seeds"]
            }
            members = unit_members.get(subset, [])
            observed_keys = {
                (str(item["file"]), int(item["seed"])) for _path, item in members
            }
            if observed_keys != expected_keys:
                incomplete_subsets[subset] = {
                    "valid_units": len(observed_keys),
                    "expected_units": len(expected_keys),
                }
                continue
            if len(observed_keys) != len(members):
                raise RuntimeError(f"duplicate Stage2 unit key in {subset}")

            head_scores: dict[tuple[int, int], list[float]] = {}
            for path, item in members:
                used_unit_hashes[path.relative_to(stage2_result).as_posix()] = sha256_file(path)
                for variant in item["variants"]:
                    head = (
                        int(variant["final_gb_min_split"]),
                        int(variant["top_k"]),
                    )
                    head_scores.setdefault(head, []).append(
                        float(variant["metrics"]["VUS-PR"])
                    )
            ranked = sorted(
                (
                    (
                        -sum(values) / len(values),
                        int(head[0]),
                        int(head[1]),
                    )
                    for head, values in head_scores.items()
                )
            )
            _negative_score, final_split, top_k = ranked[0]
            candidate_id = str(training_winners[subset])
            if candidate_id not in candidates:
                raise RuntimeError(f"unknown Stage2 training winner: {subset}")
            parameters = dict(candidates[candidate_id])
            parameters["final_gb_min_split"] = final_split
            parameters["top_k"] = top_k
            selections[subset] = {
                "training_candidate_id": candidate_id,
                "head_id": f"g{final_split}_k{top_k}",
                "parameters": parameters,
                "config_fingerprint": fingerprint(parameters),
                "tuning_mean_vus_pr": -float(_negative_score),
            }

    if not selections or len(selections) == 10:
        raise RuntimeError("prepare-partial requires between one and nine complete datasets")
    stable = {
        "schema_version": LOCK_SCHEMA,
        "method": "STAGE",
        "seed": SEED,
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "dataset_specific": True,
        "track_fallback": False,
        "partial": True,
        "complete_dataset_count": len(selections),
        "incomplete_subsets": incomplete_subsets,
        "stage2_protocol_sha256": sha256_file(stage2_protocol_path),
        "stage2_plan_sha256": sha256_file(stage2_plan_path),
        "stage2_plan_fingerprint": stage2_plan["plan_fingerprint"],
        "stage2_unit_sha256": used_unit_hashes,
        "stage2_units_fingerprint": fingerprint(used_unit_hashes),
        "tuning_source_sha256": stage2_plan["source_sha256"],
        "eval_source_sha256": _source_hashes(repo),
        "selections": selections,
    }
    payload = {
        **stable,
        "lock_fingerprint": fingerprint(stable),
        "created_at": utc_now(),
    }
    if destination.is_file():
        prior = load_json(destination)
        if prior.get("lock_fingerprint") != payload["lock_fingerprint"]:
            raise RuntimeError(f"refusing to overwrite a different partial lock: {destination}")
        return prior
    atomic_json(destination, payload)
    return payload


def validate_lock(repo: Path, path: Path) -> dict[str, Any]:
    lock = load_json(path)
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise RuntimeError("unexpected Eval lock schema")
    if fingerprint(_lock_stable(lock)) != lock.get("lock_fingerprint"):
        raise RuntimeError("Eval lock fingerprint mismatch")
    if (
        lock.get("selection_split") != SELECTION_SPLIT
        or lock.get("eval_feedback") is not False
        or lock.get("dataset_specific") is not True
        or lock.get("track_fallback") is not False
    ):
        raise RuntimeError("Eval lock violates the Tuning-only dataset-specific policy")
    if lock.get("eval_source_sha256") != _source_hashes(repo):
        raise RuntimeError("Eval source hash drift")
    return lock


def _eval_files(
    repo: Path, allowed_subsets: set[str]
) -> tuple[dict[str, dict[str, list[str]]], dict[str, str]]:
    selected: dict[str, dict[str, list[str]]] = {}
    manifests: dict[str, str] = {}
    for track, datasets in EXPECTED_TARGETS.items():
        selected_datasets = [
            dataset for dataset in datasets if f"{track}/{dataset}" in allowed_subsets
        ]
        if not selected_datasets:
            continue
        manifest = repo / "data" / "File_List" / f"TSB-AD-{track}-Eva.csv"
        manifests[track] = sha256_file(manifest)
        names = pd.read_csv(manifest)["file_name"].astype(str).tolist()
        by_dataset = {
            dataset: [name for name in names if dataset_name(name) == dataset]
            for dataset in selected_datasets
        }
        if any(not names for names in by_dataset.values()):
            raise RuntimeError(f"empty official Eval subset in {manifest}")
        selected[track] = by_dataset
    return selected, manifests


def build_plan(
    repo: Path,
    metrics_root: Path,
    lock_path: Path,
    destination: Path,
) -> dict[str, Any]:
    lock = validate_lock(repo, lock_path)
    allowed_subsets = set(lock["selections"])
    expected_subsets = {
        f"{track}/{dataset}"
        for track, datasets in EXPECTED_TARGETS.items()
        for dataset in datasets
    }
    if not allowed_subsets or not allowed_subsets.issubset(expected_subsets):
        raise RuntimeError("Eval lock contains an unknown or empty dataset set")
    files, manifest_hashes = _eval_files(repo, allowed_subsets)
    data_sha256: dict[str, str] = {}
    for track, datasets in files.items():
        for dataset, names in datasets.items():
            for file_name in names:
                source = repo / "data" / f"TSB-AD-{track}" / file_name
                if not source.is_file():
                    raise FileNotFoundError(source)
                data_sha256[f"{track}/{dataset}/{file_name}"] = sha256_file(source)
    evaluator_root, evaluator_sha256 = _evaluator_hashes(metrics_root)
    stable = {
        "schema_version": PLAN_SCHEMA,
        "method": "STAGE",
        "seed": SEED,
        "selection_split": EVALUATION_SPLIT,
        "selection_lock_split": SELECTION_SPLIT,
        "eval_feedback_used_for_selection": False,
        "runtime_eligible_for_paper": False,
        "targets": {
            track: list(datasets)
            for track, datasets in files.items()
        },
        "files": files,
        "expected_units": sum(
            len(names) for datasets in files.values() for names in datasets.values()
        ),
        "lock_sha256": sha256_file(lock_path),
        "lock_fingerprint": lock["lock_fingerprint"],
        "source_sha256": _source_hashes(repo),
        "manifest_sha256": manifest_hashes,
        "data_sha256": data_sha256,
        "evaluator": {
            "root": evaluator_root,
            "source_sha256": evaluator_sha256,
            "fingerprint": fingerprint(evaluator_sha256),
        },
        "execution_environment": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "torch_deterministic_algorithms": True,
            "runtime_eligible_for_paper": False,
        },
    }
    if len(allowed_subsets) == 10 and stable["expected_units"] != 193:
        raise RuntimeError(f"expected 193 official Eval units, got {stable['expected_units']}")
    payload = {
        **stable,
        "plan_fingerprint": fingerprint(stable),
        "created_at": utc_now(),
    }
    if destination.is_file():
        prior = load_json(destination)
        if prior.get("plan_fingerprint") != payload["plan_fingerprint"]:
            raise RuntimeError(f"refusing to overwrite a different Eval plan: {destination}")
        return prior
    atomic_json(destination, payload)
    return payload


def validate_plan(
    repo: Path, metrics_root: Path, lock_path: Path, plan_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = validate_lock(repo, lock_path)
    plan = load_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RuntimeError("unexpected Eval plan schema")
    if fingerprint(_plan_stable(plan)) != plan.get("plan_fingerprint"):
        raise RuntimeError("Eval plan fingerprint mismatch")
    if (
        plan.get("lock_sha256") != sha256_file(lock_path)
        or plan.get("lock_fingerprint") != lock.get("lock_fingerprint")
        or plan.get("source_sha256") != _source_hashes(repo)
    ):
        raise RuntimeError("Eval plan source/lock drift")
    files, manifests = _eval_files(repo, set(lock["selections"]))
    if plan.get("files") != files or plan.get("manifest_sha256") != manifests:
        raise RuntimeError("official Eval manifest drift")
    for key, expected in plan["data_sha256"].items():
        track, _dataset, file_name = key.split("/", 2)
        if sha256_file(repo / "data" / f"TSB-AD-{track}" / file_name) != expected:
            raise RuntimeError(f"Eval data hash drift: {key}")
    evaluator_root, evaluator_sha256 = _evaluator_hashes(metrics_root)
    if plan.get("evaluator") != {
        "root": evaluator_root,
        "source_sha256": evaluator_sha256,
        "fingerprint": fingerprint(evaluator_sha256),
    }:
        raise RuntimeError("official evaluator drift")
    return lock, plan


def unit_path(root: Path, track: str, dataset: str, file_name: str) -> Path:
    return root / "units" / "STAGE" / track / dataset / f"{Path(file_name).stem}.json"


def _iter_tasks(plan: Mapping[str, Any]) -> Iterable[tuple[str, str, str]]:
    for track, datasets in plan["files"].items():
        for dataset, names in datasets.items():
            for file_name in names:
                yield track, dataset, file_name


def _finite_metrics(payload: Mapping[str, Any]) -> bool:
    metrics = payload.get("metrics")
    return isinstance(metrics, Mapping) and set(metrics) == set(METRICS) and all(
        isinstance(metrics[name], (int, float))
        and not isinstance(metrics[name], bool)
        and math.isfinite(float(metrics[name]))
        for name in METRICS
    )


def valid_unit(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    track: str,
    dataset: str,
    file_name: str,
) -> bool:
    subset = f"{track}/{dataset}"
    selection = lock["selections"][subset]
    return _finite_metrics(payload) and all(
        (
            payload.get("schema_version") == UNIT_SCHEMA,
            payload.get("method") == "STAGE",
            payload.get("selection_split") == "Eval",
            payload.get("eval_feedback_used_for_selection") is False,
            payload.get("track") == track,
            payload.get("dataset") == dataset,
            payload.get("file") == file_name,
            payload.get("seed") == SEED,
            payload.get("training_prefix_policy") == "filename_declared_label_blind",
            payload.get("plan_fingerprint") == plan["plan_fingerprint"],
            payload.get("lock_fingerprint") == lock["lock_fingerprint"],
            payload.get("config_fingerprint") == selection["config_fingerprint"],
            payload.get("data_sha256") == plan["data_sha256"][f"{track}/{dataset}/{file_name}"],
            payload.get("source_sha256") == plan["source_sha256"],
            payload.get("evaluator_fingerprint") == plan["evaluator"]["fingerprint"],
            payload.get("runtime_eligible_for_paper") is False,
            payload.get("error") is None,
        )
    )


def _assert_no_unplanned_units(root: Path, plan: Mapping[str, Any]) -> None:
    expected = {
        unit_path(root, track, dataset, file_name).resolve()
        for track, dataset, file_name in _iter_tasks(plan)
    }
    units = root / "units"
    if not units.exists():
        return
    unexpected = [
        path for path in units.rglob("*.json") if path.resolve() not in expected
    ]
    if unexpected:
        raise RuntimeError(f"unplanned Eval unit artifact: {unexpected[0]}")


def _run_lanes(
    tasks: Sequence[tuple[str, str, str]],
    gpus: Sequence[str],
    workers_per_gpu: int,
    worker,
) -> None:
    lanes = [gpu for _ in range(workers_per_gpu) for gpu in gpus]
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
            except BaseException as exc:
                failures.append(exc)
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [pool.submit(lane, gpu) for gpu in lanes]
        for future in as_completed(futures):
            future.result()
    if failures:
        raise RuntimeError(str(failures[0])) from failures[0]


@contextmanager
def controller_singleton(result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / CONTROLLER_LOCK_NAME
    handle = path.open("a+", encoding="utf-8")
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
                raise RuntimeError(f"another Eval controller holds {path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another Eval controller holds {path}") from exc
        locked = True
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


def run_eval(
    repo: Path,
    metrics_root: Path,
    result_root: Path,
    lock_path: Path,
    plan_path: Path,
    python: Path,
    gpus: Sequence[str],
    workers_per_gpu: int,
) -> dict[str, Any]:
    lock, plan = validate_plan(repo, metrics_root, lock_path, plan_path)
    _assert_no_unplanned_units(result_root, plan)
    tasks: list[tuple[str, str, str]] = []
    for track, dataset, file_name in _iter_tasks(plan):
        path = unit_path(result_root, track, dataset, file_name)
        if path.is_file():
            if valid_unit(
                load_json(path),
                plan=plan,
                lock=lock,
                track=track,
                dataset=dataset,
                file_name=file_name,
            ):
                continue
            raise RuntimeError(f"refusing to overwrite invalid Eval unit: {path}")
        tasks.append((track, dataset, file_name))
    tasks.sort(
        key=lambda item: (
            -(repo / "data" / f"TSB-AD-{item[0]}" / item[2]).stat().st_size,
            *item,
        )
    )
    total = int(plan["expected_units"])
    counter = {"value": total - len(tasks)}
    counter_lock = threading.Lock()
    scratch = result_root / ".stage_eval_scratch"

    def worker(gpu: str, task: tuple[str, str, str]) -> None:
        track, dataset, file_name = task
        subset = f"{track}/{dataset}"
        selection = lock["selections"][subset]
        outcome = run_stage_once(
            repo=repo,
            python=python,
            metrics_root=metrics_root,
            track=track,
            file_name=file_name,
            config=selection["parameters"],
            seed=SEED,
            gpu=gpu,
            scratch_parent=scratch,
        )
        payload = {
            "schema_version": UNIT_SCHEMA,
            "method": "STAGE",
            "selection_split": "Eval",
            "eval_feedback_used_for_selection": False,
            "track": track,
            "dataset": dataset,
            "file": file_name,
            "seed": SEED,
            "training_prefix_policy": "filename_declared_label_blind",
            "training_candidate_id": selection["training_candidate_id"],
            "head_id": selection["head_id"],
            "plan_fingerprint": plan["plan_fingerprint"],
            "lock_fingerprint": lock["lock_fingerprint"],
            "config_fingerprint": selection["config_fingerprint"],
            "data_sha256": plan["data_sha256"][f"{track}/{dataset}/{file_name}"],
            "source_sha256": plan["source_sha256"],
            "evaluator_fingerprint": plan["evaluator"]["fingerprint"],
            "metrics": outcome["metrics"],
            "runtime_s": outcome["runtime_s"],
            "runtime_eligible_for_paper": False,
            "diagnostics": outcome["diagnostics"],
            "error": None,
            "completed_at": utc_now(),
        }
        atomic_json(unit_path(result_root, track, dataset, file_name), payload)
        with counter_lock:
            counter["value"] += 1
            print(
                f"[STAGE Eval {counter['value']}/{total}] GPU={gpu} "
                f"{track}/{dataset} {file_name}",
                flush=True,
            )

    _run_lanes(tasks, gpus, workers_per_gpu, worker)
    if scratch.exists():
        shutil.rmtree(scratch)
    return audit_and_write_status(result_root, plan, lock)


def audit_and_write_status(
    result_root: Path, plan: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    valid = 0
    for track, dataset, file_name in _iter_tasks(plan):
        path = unit_path(result_root, track, dataset, file_name)
        if not path.is_file() or not valid_unit(
            load_json(path),
            plan=plan,
            lock=lock,
            track=track,
            dataset=dataset,
            file_name=file_name,
        ):
            raise RuntimeError(f"Eval unit audit failed: {path}")
        valid += 1
    if valid != int(plan["expected_units"]):
        raise RuntimeError("Eval audit count differs from frozen plan")
    payload = {
        "schema_version": STATUS_SCHEMA,
        "status": "complete",
        "seed": SEED,
        "expected_units": int(plan["expected_units"]),
        "completed_units": valid,
        "error_units": 0,
        "plan_fingerprint": plan["plan_fingerprint"],
        "lock_fingerprint": lock["lock_fingerprint"],
        "completed_at": utc_now(),
    }
    atomic_json(result_root / STATUS_NAME, payload)
    return payload


def status(
    repo: Path,
    metrics_root: Path,
    result_root: Path,
    lock_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    lock, plan = validate_plan(repo, metrics_root, lock_path, plan_path)
    _assert_no_unplanned_units(result_root, plan)
    raw = valid = invalid = errors = 0
    by_subset: dict[str, dict[str, int]] = {}
    for track, dataset, file_name in _iter_tasks(plan):
        subset = f"{track}/{dataset}"
        slot = by_subset.setdefault(subset, {"expected": 0, "valid": 0})
        slot["expected"] += 1
        path = unit_path(result_root, track, dataset, file_name)
        if not path.is_file():
            continue
        raw += 1
        try:
            payload = load_json(path)
        except Exception:
            invalid += 1
            continue
        if payload.get("error") is not None:
            errors += 1
        if valid_unit(
            payload,
            plan=plan,
            lock=lock,
            track=track,
            dataset=dataset,
            file_name=file_name,
        ):
            valid += 1
            slot["valid"] += 1
        else:
            invalid += 1
    return {
        "ok": invalid == 0 and errors == 0,
        "expected": int(plan["expected_units"]),
        "raw": raw,
        "valid": valid,
        "invalid": invalid,
        "errors": errors,
        "missing": int(plan["expected_units"]) - valid,
        "by_subset": by_subset,
        "plan_fingerprint": plan["plan_fingerprint"],
        "lock_fingerprint": lock["lock_fingerprint"],
        "observed_at": utc_now(),
    }


def summarize(
    repo: Path,
    metrics_root: Path,
    result_root: Path,
    lock_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    lock, plan = validate_plan(repo, metrics_root, lock_path, plan_path)
    audit_and_write_status(result_root, plan, lock)
    grouped: dict[str, list[Mapping[str, float]]] = {}
    for track, dataset, file_name in _iter_tasks(plan):
        payload = load_json(unit_path(result_root, track, dataset, file_name))
        grouped.setdefault(f"{track}/{dataset}", []).append(payload["metrics"])
    rows: list[dict[str, Any]] = []
    for subset, members in grouped.items():
        track, dataset = subset.split("/", 1)
        row = {
            "track": track,
            "dataset": dataset,
            "series": len(members),
            **{
                metric: sum(float(item[metric]) for item in members) / len(members)
                for metric in METRICS
            },
        }
        rows.append(row)
    rows.sort(key=lambda item: (item["track"], item["dataset"]))
    global_macro = {
        metric: sum(float(row[metric]) for row in rows) / len(rows)
        for metric in METRICS
    }
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "method": "STAGE",
        "seed": SEED,
        "selection_split": "Eval",
        "eval_feedback_used_for_selection": False,
        "plan_fingerprint": plan["plan_fingerprint"],
        "lock_fingerprint": lock["lock_fingerprint"],
        "completed_units": int(plan["expected_units"]),
        "rows": rows,
        "global_dataset_macro": global_macro,
        "created_at": utc_now(),
    }
    target = result_root / SUMMARY_JSON_NAME
    if target.is_file():
        prior = load_json(target)
        comparable_prior = {k: v for k, v in prior.items() if k != "created_at"}
        comparable_new = {k: v for k, v in payload.items() if k != "created_at"}
        if comparable_prior != comparable_new:
            raise RuntimeError(f"refusing to overwrite a different Eval summary: {target}")
        return prior
    atomic_json(target, payload)
    _atomic_csv(
        result_root / SUMMARY_CSV_NAME,
        rows,
        ["track", "dataset", "series", *METRICS],
    )
    return payload


def _parse_gpus(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("--gpus must contain unique comma-separated devices")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    parser.add_argument("--stage2-protocol", required=True, type=Path)
    parser.add_argument("--stage2-result", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("prepare-partial")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--python", required=True, type=Path)
    run_parser.add_argument("--gpus", default="0,1")
    run_parser.add_argument("--workers-per-gpu", type=int, default=6)
    sub.add_parser("status")
    sub.add_parser("summarize")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    metrics_root = args.metrics_root.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / LOCK_NAME
    plan_path = result_root / PLAN_NAME
    if args.command in {"prepare", "prepare-partial"}:
        builder = build_lock if args.command == "prepare" else build_partial_lock
        lock = builder(
            repo,
            args.stage2_protocol.resolve(),
            args.stage2_result.resolve(),
            lock_path,
        )
        plan = build_plan(repo, metrics_root, lock_path, plan_path)
        print(
            json.dumps(
                {
                    "lock_fingerprint": lock["lock_fingerprint"],
                    "plan_fingerprint": plan["plan_fingerprint"],
                    "expected_units": plan["expected_units"],
                },
                indent=2,
            )
        )
    elif args.command == "run":
        with controller_singleton(result_root):
            payload = run_eval(
                repo,
                metrics_root,
                result_root,
                lock_path,
                plan_path,
                args.python.resolve(),
                _parse_gpus(args.gpus),
                int(args.workers_per_gpu),
            )
        print(json.dumps(payload, indent=2))
    elif args.command == "status":
        print(
            json.dumps(
                status(repo, metrics_root, result_root, lock_path, plan_path),
                indent=2,
            )
        )
    else:
        with controller_singleton(result_root):
            payload = summarize(
                repo, metrics_root, result_root, lock_path, plan_path
            )
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
