#!/usr/bin/env python3
"""Leakage-safe controller for the two frozen recent AAAI baselines.

The controller deliberately keeps the extension outside the original baseline
result tree.  Each method has exactly one configuration in a frozen lock.  The
official TSB-AD Tuning split is therefore used only as a 44-unit execution and
finite-metric qualification gate (22 series x 2 methods); no Tuning metric is
ranked and no configuration is selected.  The unchanged configurations are
then evaluated on 193 Eval series per method (386 units total).

Unit JSON files are the recovery authority.  They are written atomically by
``extensions/aaai_recent/run_one_recent.py`` and reused only after strict
identity, seed, split, six-metric, and fingerprint validation.  Checkpoints and
score arrays are forbidden.  Runtime collected under shared execution is
diagnostic only and is never eligible for a paper efficiency table.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import METRICS, atomic_json, dataset_name  # noqa: E402


METHODS: tuple[str, ...] = ("DPAD_AAAI24", "DNE_AAAI25")
TARGETS: dict[str, tuple[str, ...]] = {
    "U": ("UCR", "Exathlon", "MSL", "SED", "TODS"),
    "M": ("CATSv2", "GHL", "LTDB", "SVDB", "TAO"),
}
SELECTION_SEED = 2026
TUNING_SERIES = 22
EVAL_SERIES = 193
TUNING_UNITS = TUNING_SERIES * len(METHODS)
EVAL_UNITS = EVAL_SERIES * len(METHODS)
PLAN_NAME = "recent_baseline_execution_plan.json"
FAILSTOP_NAME = "recent_baseline_failstop.json"
COMPLETE_NAME = "recent_baseline_autopilot_complete.json"
LOCKFILE_NAME = ".recent_baseline_autopilot.lock"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt"}
FORBIDDEN_RECORD_KEYS = {
    "checkpoint",
    "checkpoints",
    "model_state_dict",
    "score_array",
    "score_arrays",
    "scores",
    "state_dict",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def parse_gpus(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = list(value)
    gpus = tuple(str(item).strip() for item in raw if str(item).strip())
    if not gpus:
        raise ValueError("at least one physical GPU must be provided")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"duplicate physical GPU identifiers: {gpus}")
    if any("," in gpu for gpu in gpus):
        raise ValueError(f"each lane must name exactly one physical GPU: {gpus}")
    return gpus


def read_locked_contract(repo: Path, locked_path: Path) -> dict[str, Any]:
    """Load and cryptographically validate the immutable two-method contract."""
    if not locked_path.is_file():
        raise FileNotFoundError(locked_path)
    locked = load_json(locked_path)
    if locked.get("status") != "frozen":
        raise RuntimeError("recent-baseline formal execution requires a frozen lock")
    if locked.get("selection_split") != "official TSB-AD Tuning only":
        raise RuntimeError("the frozen lock must select only on official TSB-AD Tuning")
    if locked.get("eval_feedback") is not False:
        raise RuntimeError("the frozen lock must explicitly prohibit Eval feedback")
    if int(locked.get("seed", -1)) != SELECTION_SEED:
        raise RuntimeError(f"recent-baseline lock seed must be {SELECTION_SEED}")

    stable = {
        key: value
        for key, value in locked.items()
        if key not in {"locked_fingerprint", "locked_at"}
    }
    locked_fingerprint = locked.get("locked_fingerprint")
    if not is_sha256(locked_fingerprint):
        raise RuntimeError("recent-baseline locked_fingerprint is absent or malformed")
    if stable_fingerprint(stable) != locked_fingerprint:
        raise RuntimeError("recent-baseline locked fingerprint mismatch")
    if not is_sha256(locked.get("plan_fingerprint")):
        raise RuntimeError("recent-baseline plan_fingerprint is absent or malformed")
    if not isinstance(locked.get("plan"), dict):
        raise RuntimeError("recent-baseline plan contract is absent")
    if stable_fingerprint(locked["plan"]) != locked["plan_fingerprint"]:
        raise RuntimeError("recent-baseline plan fingerprint mismatch")

    base_relative = locked.get("base_protocol")
    if not isinstance(base_relative, str) or not base_relative:
        raise RuntimeError("recent-baseline base_protocol is absent")
    base_protocol = (locked_path.parent / base_relative).resolve()
    if not base_protocol.is_file():
        raise FileNotFoundError(base_protocol)
    if base_protocol != (repo / "protocol.json").resolve():
        raise RuntimeError("recent-baseline lock points to an unexpected base protocol")
    if sha256_file(base_protocol) != locked.get("base_protocol_sha256"):
        raise RuntimeError("recent-baseline base protocol hash mismatch")

    source_files = locked.get("source_files")
    source_sha256 = locked.get("source_sha256")
    if (
        not isinstance(source_files, dict)
        or not source_files
        or not isinstance(source_sha256, dict)
        or set(source_files) != set(source_sha256)
    ):
        raise RuntimeError("recent-baseline source path/hash maps are absent or differ")
    resolved_sources: set[Path] = set()
    for name, relative in source_files.items():
        source = (repo / str(relative)).resolve()
        if not source.is_relative_to(repo):
            raise RuntimeError(f"recent-baseline source escapes repository: {relative}")
        expected_hash = source_sha256.get(name)
        if not is_sha256(expected_hash):
            raise RuntimeError(f"recent-baseline {name} source hash is absent or malformed")
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise RuntimeError(f"recent-baseline {name} source hash mismatch")
        resolved_sources.add(source)
    required_sources = {
        (repo / "extensions" / "aaai_recent" / "run_one_recent.py").resolve(),
        (repo / "extensions" / "aaai_recent" / "models.py").resolve(),
    }
    if not required_sources.issubset(resolved_sources):
        raise RuntimeError("frozen source map must include the recent runner and models")

    methods = locked.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(METHODS):
        raise RuntimeError(f"frozen lock must contain exactly {METHODS}")
    for method in METHODS:
        method_contract = methods[method]
        if not isinstance(method_contract, dict):
            raise RuntimeError(f"invalid method contract for {method}")
        parameters = method_contract.get("formal_hyperparameters")
        if not isinstance(parameters, dict) or not parameters:
            raise RuntimeError(f"exactly one non-empty frozen configuration is required for {method}")
        config_fingerprint = method_contract.get("config_fingerprint")
        if not is_sha256(config_fingerprint):
            raise RuntimeError(f"config fingerprint is absent or malformed for {method}")
        if stable_fingerprint(parameters) != config_fingerprint:
            raise RuntimeError(f"frozen configuration fingerprint mismatch for {method}")
        if not isinstance(method_contract.get("parameter_source"), str):
            raise RuntimeError(f"parameter provenance is absent for {method}")

    frozen_files = locked.get("files")
    if not isinstance(frozen_files, dict) or set(frozen_files) != {"Tuning", "Eval"}:
        raise RuntimeError("frozen lock must contain Tuning and Eval allowlist contracts")
    official = {
        "Tuning": official_split_files(repo, "Tuning"),
        "Eval": official_split_files(repo, "Eva"),
    }
    expected_counts = {"Tuning": TUNING_SERIES, "Eval": EVAL_SERIES}
    manifest_names = {"Tuning": "Tuning", "Eval": "Eva"}
    for split, official_tracks in official.items():
        contract = frozen_files.get(split)
        if not isinstance(contract, dict):
            raise RuntimeError(f"frozen {split} allowlist contract is malformed")
        if int(contract.get("expected_series", -1)) != expected_counts[split]:
            raise RuntimeError(f"frozen {split} series count is malformed")
        if stable_fingerprint(official_tracks) != contract.get("allowlist_fingerprint"):
            raise RuntimeError(f"frozen {split} allowlist fingerprint differs from official")
        manifests = contract.get("manifests")
        if not isinstance(manifests, dict) or set(manifests) != set(TARGETS):
            raise RuntimeError(f"frozen {split} manifest contract is malformed")
        for track in TARGETS:
            manifest_contract = manifests[track]
            expected_path = (
                repo
                / "data"
                / "File_List"
                / f"TSB-AD-{track}-{manifest_names[split]}.csv"
            ).resolve()
            actual_path = (repo / str(manifest_contract.get("path", ""))).resolve()
            if actual_path != expected_path:
                raise RuntimeError(f"frozen {split}/{track} manifest path mismatch")
            if sha256_file(actual_path) != manifest_contract.get("sha256"):
                raise RuntimeError(f"frozen {split}/{track} manifest hash mismatch")
    return locked


def official_split_files(repo: Path, split: str) -> dict[str, dict[str, list[str]]]:
    """Enumerate only the declared ten subsets from an official TSB-AD split."""
    if split not in {"Tuning", "Eva"}:
        raise ValueError(f"unsupported official split: {split}")
    selected: dict[str, dict[str, list[str]]] = {}
    seen: set[tuple[str, str]] = set()
    for track, datasets in TARGETS.items():
        manifest = repo / "data" / "File_List" / f"TSB-AD-{track}-{split}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            names = [str(row["file_name"]) for row in csv.DictReader(handle)]
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate file_name entries in {manifest}")
        by_dataset: dict[str, list[str]] = {}
        for dataset in datasets:
            subset = sorted(name for name in names if dataset_name(name) == dataset)
            if not subset:
                raise RuntimeError(f"{manifest} has no members for {track}/{dataset}")
            for file_name in subset:
                identity = (track, file_name)
                if identity in seen:
                    raise RuntimeError(f"duplicate official split identity: {identity}")
                seen.add(identity)
                source = repo / "data" / f"TSB-AD-{track}" / file_name
                if not source.is_file():
                    raise FileNotFoundError(source)
            by_dataset[dataset] = subset
        selected[track] = by_dataset
    expected = TUNING_SERIES if split == "Tuning" else EVAL_SERIES
    actual = sum(len(names) for datasets in selected.values() for names in datasets.values())
    if actual != expected:
        raise RuntimeError(f"official {split} series count {actual} != {expected}")
    return selected


def build_plan(repo: Path, result_root: Path, locked_path: Path) -> dict[str, Any]:
    locked = read_locked_contract(repo, locked_path)
    tuning_files = official_split_files(repo, "Tuning")
    eval_files = official_split_files(repo, "Eva")
    stable = {
        "schema_version": "recent-baseline-execution-plan-v1",
        "seed": SELECTION_SEED,
        "methods": {
            method: {
                "config_fingerprint": locked["methods"][method]["config_fingerprint"],
                "configuration_source": locked["methods"][method]["parameter_source"],
                "single_frozen_configuration": True,
            }
            for method in METHODS
        },
        "locked_fingerprint": locked["locked_fingerprint"],
        "contract_plan_fingerprint": locked["plan_fingerprint"],
        "source_sha256": locked["source_sha256"],
        "tuning": {
            "split": "official TSB-AD Tuning only",
            "purpose": "execution and finite-six-metric qualification only",
            "metrics_used_for_selection": False,
            "expected_series": TUNING_SERIES,
            "expected_units": TUNING_UNITS,
            "files": tuning_files,
        },
        "eval": {
            "split": "official TSB-AD Eval",
            "uses_unchanged_tuning_qualified_configurations": True,
            "eval_feedback": False,
            "expected_series": EVAL_SERIES,
            "expected_units": EVAL_UNITS,
            "files": eval_files,
        },
        "storage_policy": "six full-precision metrics; no checkpoint or score array",
        "runtime_eligible_for_paper": False,
    }
    plan = {
        **stable,
        "execution_plan_fingerprint": stable_fingerprint(stable),
        "created_at": utc_now(),
    }
    target = result_root / PLAN_NAME
    if target.is_file():
        prior = load_json(target)
        if prior.get("execution_plan_fingerprint") != plan["execution_plan_fingerprint"]:
            raise RuntimeError(f"refusing to replace a different execution plan: {target}")
        return prior
    atomic_json(target, plan)
    return plan


def phase_root(result_root: Path, phase: str) -> Path:
    if phase == "tuning":
        return result_root / f"tuning_seed{SELECTION_SEED}"
    if phase == "eval":
        return result_root / f"eval_seed{SELECTION_SEED}"
    raise ValueError(f"unsupported phase: {phase}")


def unit_path(
    result_root: Path,
    phase: str,
    method: str,
    track: str,
    dataset: str,
    file_name: str,
) -> Path:
    return (
        phase_root(result_root, phase)
        / "units"
        / method
        / track
        / dataset
        / f"{Path(file_name).stem}.json"
    )


def expected_units(
    result_root: Path, plan: Mapping[str, Any], phase: str
) -> list[dict[str, Any]]:
    split = "Tuning" if phase == "tuning" else "Eval"
    phase_plan = plan[phase]
    units: list[dict[str, Any]] = []
    targets: set[Path] = set()
    identities: set[tuple[str, str, str, str]] = set()
    for method in METHODS:
        for track, datasets in phase_plan["files"].items():
            for dataset, names in datasets.items():
                for file_name in names:
                    target = unit_path(
                        result_root, phase, method, track, dataset, file_name
                    )
                    identity = (method, track, dataset, file_name)
                    if target in targets or identity in identities:
                        raise RuntimeError(f"duplicate planned recent-baseline unit: {identity}")
                    targets.add(target)
                    identities.add(identity)
                    units.append(
                        {
                            "method": method,
                            "track": track,
                            "dataset": dataset,
                            "file": file_name,
                            "seed": SELECTION_SEED,
                            "split": split,
                            "target": target,
                            "config_fingerprint": plan["methods"][method][
                                "config_fingerprint"
                            ],
                        }
                    )
    expected = TUNING_UNITS if phase == "tuning" else EVAL_UNITS
    if len(units) != expected:
        raise RuntimeError(f"planned {phase} unit count {len(units)} != {expected}")
    return units


def assert_no_embedded_payloads(value: Any, location: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_RECORD_KEYS:
                raise RuntimeError(f"forbidden persisted payload key {key!r} in {location}")
            assert_no_embedded_payloads(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_embedded_payloads(child, f"{location}[{index}]")


def validate_unit(
    path: Path,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
    physical_gpu: str | None = None,
) -> dict[str, Any]:
    item = load_json(path)
    checks = {
        "method": expected["method"],
        "track": expected["track"],
        "dataset": expected["dataset"],
        "file": expected["file"],
        "seed": expected["seed"],
        "profile": "formal",
        "split": expected["split"],
        "locked_fingerprint": plan["locked_fingerprint"],
        "config_fingerprint": expected["config_fingerprint"],
        "plan_fingerprint": plan["contract_plan_fingerprint"],
        "configuration_source": plan["methods"][expected["method"]][
            "configuration_source"
        ],
        "source_sha256": plan["source_sha256"],
    }
    for field, wanted in checks.items():
        if item.get(field) != wanted:
            raise RuntimeError(
                f"identity/fingerprint mismatch at {path}: "
                f"{field}={item.get(field)!r}, expected {wanted!r}"
            )
    if item.get("error") is not None:
        raise RuntimeError(f"error unit at {path}: {item.get('error')}")
    if item.get("runtime_eligible_for_paper") is not False:
        raise RuntimeError(f"shared-run runtime must remain paper-ineligible: {path}")
    if not str(item.get("device", "")).startswith("cuda"):
        raise RuntimeError(f"formal recent-baseline unit did not use CUDA: {path}")
    recorded_gpu = item.get("physical_gpu")
    if not isinstance(recorded_gpu, str) or not recorded_gpu:
        raise RuntimeError(f"formal recent-baseline physical GPU is absent: {path}")
    if physical_gpu is not None and recorded_gpu != physical_gpu:
        raise RuntimeError(
            f"physical GPU mismatch at {path}: {recorded_gpu!r} != {physical_gpu!r}"
        )
    metrics = item.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(f"metrics object is absent: {path}")
    for metric in METRICS:
        if metric not in metrics:
            raise RuntimeError(f"missing {metric} at {path}")
        if isinstance(metrics[metric], bool):
            raise RuntimeError(f"boolean metric {metric} at {path}")
        try:
            value = float(metrics[metric])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"non-numeric {metric} at {path}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite {metric} at {path}")
    assert_no_embedded_payloads(item, str(path))
    return item


def assert_no_forbidden_artifacts(result_root: Path) -> None:
    if not result_root.exists():
        return
    forbidden: list[Path] = []
    for path in result_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(path)
            continue
        lowered = path.name.lower()
        if "checkpoint" in lowered or "score_array" in lowered:
            forbidden.append(path)
    if forbidden:
        preview = ", ".join(str(path) for path in forbidden[:8])
        raise RuntimeError(f"forbidden checkpoint/score-array artifacts: {preview}")


def audit_phase(
    result_root: Path, plan: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    planned = expected_units(result_root, plan, phase)
    expected_paths = {unit["target"] for unit in planned}
    units_root = phase_root(result_root, phase) / "units"
    if units_root.exists():
        actual_paths = set(units_root.rglob("*.json"))
        unexpected = sorted(actual_paths - expected_paths)
        if unexpected:
            raise RuntimeError(f"unexpected {phase} unit JSON: {unexpected[0]}")
        other_files = [
            path
            for path in units_root.rglob("*")
            if path.is_file()
            and path not in actual_paths
            and ".tmp." not in path.name
        ]
        if other_files:
            raise RuntimeError(f"unexpected non-JSON unit artifact: {other_files[0]}")

    completed = 0
    by_method = {method: 0 for method in METHODS}
    seen: set[tuple[str, str, str, str]] = set()
    for unit in planned:
        path = unit["target"]
        if not path.is_file():
            continue
        item = validate_unit(path, unit, plan)
        identity = (
            str(item["method"]),
            str(item["track"]),
            str(item["dataset"]),
            str(item["file"]),
        )
        if identity in seen:
            raise RuntimeError(f"duplicate valid {phase} unit identity: {identity}")
        seen.add(identity)
        completed += 1
        by_method[unit["method"]] += 1
    expected_count = len(planned)
    return {
        "phase": phase,
        "status": "complete" if completed == expected_count else "running",
        "completed_units": completed,
        "valid_units": completed,
        "expected_units": expected_count,
        "error_units": 0,
        "duplicate_units": 0,
        "invalid_units": 0,
        "by_method": by_method,
        "seed": SELECTION_SEED,
        "locked_fingerprint": plan["locked_fingerprint"],
        "contract_plan_fingerprint": plan["contract_plan_fingerprint"],
        "execution_plan_fingerprint": plan["execution_plan_fingerprint"],
        "metrics_used_for_selection": False,
        "runtime_eligible_for_paper": False,
        "audited_at": utc_now(),
    }


def write_phase_status(result_root: Path, phase: str, report: Mapping[str, Any]) -> None:
    payload = dict(report)
    payload["updated_at"] = utc_now()
    atomic_json(phase_root(result_root, phase) / f"{phase}_status.json", payload)


def run_lanes(
    tasks: Sequence[Mapping[str, Any]],
    gpus: Sequence[str],
    worker,
) -> None:
    """Run exactly one serial worker lane for each explicitly isolated GPU."""
    if not tasks:
        return
    iterator: Iterator[Mapping[str, Any]] = iter(tasks)
    iterator_lock = threading.Lock()
    stop = threading.Event()
    failures: list[BaseException] = []
    failure_lock = threading.Lock()

    def lane(gpu: str) -> None:
        while not stop.is_set():
            with iterator_lock:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
            try:
                worker(gpu, task)
            except BaseException as exc:  # first error stops assignment of new work
                with failure_lock:
                    if not failures:
                        failures.append(exc)
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(lane, gpu) for gpu in gpus]
        for future in as_completed(futures):
            future.result()
    if failures:
        raise RuntimeError(str(failures[0])) from failures[0]


def run_one_unit(
    *,
    repo: Path,
    result_root: Path,
    python: Path,
    metrics_root: Path,
    locked_path: Path,
    plan: Mapping[str, Any],
    gpu: str,
    unit: Mapping[str, Any],
) -> None:
    target = Path(unit["target"])
    if target.exists():
        raise RuntimeError(f"worker refuses to overwrite an existing unit: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(repo / "extensions" / "aaai_recent" / "run_one_recent.py"),
        "--repo",
        str(repo),
        "--method",
        str(unit["method"]),
        "--track",
        str(unit["track"]),
        "--file",
        str(unit["file"]),
        "--seed",
        str(unit["seed"]),
        "--output",
        str(target),
        "--stage-source",
        str(repo / "STAGE" / "stage.py"),
        "--paano-root",
        str(metrics_root / "PaAno"),
        "--profile",
        "formal",
        "--split",
        str(unit["split"]),
        "--locked-config",
        str(locked_path),
        "--require-physical-gpu",
        gpu,
    ]
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    # Required by torch.use_deterministic_algorithms(True) for CUDA >= 10.2.
    # This is a runtime reproducibility setting, not a model hyperparameter.
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
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
            f"recent baseline failed for {unit['split']} {unit['method']} "
            f"{unit['track']}/{unit['file']} on GPU {gpu}:\n{tail}"
        )
    if not target.is_file():
        raise RuntimeError(f"runner succeeded without atomically publishing {target}")
    validate_unit(target, unit, plan, physical_gpu=gpu)
    assert_no_forbidden_artifacts(result_root)


def run_phase(
    *,
    repo: Path,
    result_root: Path,
    python: Path,
    metrics_root: Path,
    locked_path: Path,
    plan: Mapping[str, Any],
    phase: str,
    gpus: Sequence[str],
) -> dict[str, Any]:
    if phase == "eval":
        qualification = audit_phase(result_root, plan, "tuning")
        if qualification["status"] != "complete":
            raise RuntimeError("formal Eval is forbidden before strict 44/44 Tuning qualification")
    assert_no_forbidden_artifacts(result_root)
    initial = audit_phase(result_root, plan, phase)
    write_phase_status(result_root, phase, initial)
    planned = expected_units(result_root, plan, phase)
    tasks = [unit for unit in planned if not Path(unit["target"]).is_file()]
    counter = {"value": int(initial["completed_units"])}
    counter_lock = threading.Lock()
    total = len(planned)

    def worker(gpu: str, unit: Mapping[str, Any]) -> None:
        run_one_unit(
            repo=repo,
            result_root=result_root,
            python=python,
            metrics_root=metrics_root,
            locked_path=locked_path,
            plan=plan,
            gpu=gpu,
            unit=unit,
        )
        with counter_lock:
            counter["value"] += 1
            progress = audit_phase(result_root, plan, phase)
            write_phase_status(result_root, phase, progress)
            label = "Tuning qualification" if phase == "tuning" else "Eval"
            print(
                f"[recent {label} {counter['value']}/{total}] GPU={gpu} "
                f"{unit['method']} {unit['track']}/{unit['dataset']} {unit['file']}",
                flush=True,
            )

    run_lanes(tasks, gpus, worker)
    final = audit_phase(result_root, plan, phase)
    if final["status"] != "complete":
        raise RuntimeError(
            f"{phase} ended at {final['completed_units']}/{final['expected_units']}"
        )
    final["completed_at"] = utc_now()
    if phase == "tuning":
        final["qualification_only"] = True
        final["configuration_selection_performed"] = False
        final["unchanged_frozen_configurations_qualified"] = list(METHODS)
    else:
        final["used_unchanged_tuning_qualified_configurations"] = True
        final["eval_feedback"] = False
    write_phase_status(result_root, phase, final)
    assert_no_forbidden_artifacts(result_root)
    return final


def run_tuning_qualification(
    *,
    repo: Path,
    result_root: Path,
    python: Path,
    metrics_root: Path,
    locked_config: Path,
    gpus: Sequence[str] = ("0", "1"),
) -> dict[str, Any]:
    """Run/resume the fixed-config 44-unit qualification without selection."""
    normalized_gpus = parse_gpus(gpus)
    plan = build_plan(repo, result_root, locked_config)
    return run_phase(
        repo=repo,
        result_root=result_root,
        python=python,
        metrics_root=metrics_root,
        locked_path=locked_config,
        plan=plan,
        phase="tuning",
        gpus=normalized_gpus,
    )


def run_formal_eval(
    *,
    repo: Path,
    result_root: Path,
    python: Path,
    metrics_root: Path,
    locked_config: Path,
    gpus: Sequence[str] = ("0", "1"),
) -> dict[str, Any]:
    """Run/resume 386 Eval units after strict qualification of the same configs."""
    normalized_gpus = parse_gpus(gpus)
    plan = build_plan(repo, result_root, locked_config)
    return run_phase(
        repo=repo,
        result_root=result_root,
        python=python,
        metrics_root=metrics_root,
        locked_path=locked_config,
        plan=plan,
        phase="eval",
        gpus=normalized_gpus,
    )


@contextmanager
def singleton_lock(result_root: Path) -> Iterator[None]:
    """Hold a non-blocking OS lock for one controller per independent root."""
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / LOCKFILE_NAME
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
                raise RuntimeError(f"another recent-baseline controller holds {lock_path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another recent-baseline controller holds {lock_path}") from exc
        locked = True
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "acquired_at": utc_now()},
                sort_keys=True,
            )
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


def validate_runtime_paths(
    repo: Path,
    result_root: Path,
    python: Path,
    metrics_root: Path,
    locked_config: Path,
) -> None:
    required = (
        repo / "extensions" / "aaai_recent" / "run_one_recent.py",
        repo / "extensions" / "aaai_recent" / "models.py",
        repo / "STAGE" / "stage.py",
        metrics_root / "PaAno",
        locked_config,
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not python.is_file():
        raise FileNotFoundError(python)
    if result_root == repo or repo in result_root.parents:
        raise RuntimeError("recent-baseline results must use an independent root outside the repo")


def _write_failstop(result_root: Path, command: str, exc: BaseException) -> None:
    target = result_root / FAILSTOP_NAME
    if target.exists():
        return
    atomic_json(
        target,
        {
            "status": "fail-stop",
            "command": command,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "pid": os.getpid(),
            "created_at": utc_now(),
            "automatic_restart_permitted": False,
        },
    )


def strict_barrier_summary(
    tuning: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the narrow, stable contract consumed by ``stage_autopilot``."""
    summary = {
        "status": "complete",
        "tuning": {
            "status": tuning.get("status"),
            "expected_units": int(tuning.get("expected_units", -1)),
            "valid_units": int(tuning.get("valid_units", -1)),
            "error_units": int(tuning.get("error_units", -1)),
        },
        "eval": {
            "status": evaluation.get("status"),
            "expected_units": int(evaluation.get("expected_units", -1)),
            "valid_units": int(evaluation.get("valid_units", -1)),
            "error_units": int(evaluation.get("error_units", -1)),
        },
    }
    required = {"tuning": TUNING_UNITS, "eval": EVAL_UNITS}
    for phase, expected in required.items():
        phase_summary = summary[phase]
        if not (
            phase_summary["status"] == "complete"
            and phase_summary["expected_units"] == expected
            and phase_summary["valid_units"] == expected
            and phase_summary["error_units"] == 0
        ):
            raise RuntimeError(f"invalid recent-baseline barrier summary: {summary}")
    return summary


def run_recent_baselines(
    *,
    repo: Path,
    result_root: Path,
    python: Path,
    metrics_root: Path,
    locked_config: Path,
    gpus: Sequence[str] = ("0", "1"),
) -> dict[str, Any]:
    """Importable all-in-one entry point used by the main STAGE autopilot."""
    repo = repo.resolve()
    result_root = result_root.resolve()
    # Preserve a selected virtual-environment executable instead of resolving
    # a symlink to its base interpreter.
    python = Path(os.path.abspath(python))
    metrics_root = metrics_root.resolve()
    locked_config = locked_config.resolve()
    normalized_gpus = parse_gpus(gpus)
    validate_runtime_paths(repo, result_root, python, metrics_root, locked_config)
    with singleton_lock(result_root):
        failstop = result_root / FAILSTOP_NAME
        if failstop.exists():
            raise RuntimeError(f"existing fail-stop requires explicit audit: {failstop}")
        try:
            plan = build_plan(repo, result_root, locked_config)
            tuning = run_phase(
                repo=repo,
                result_root=result_root,
                python=python,
                metrics_root=metrics_root,
                locked_path=locked_config,
                plan=plan,
                phase="tuning",
                gpus=normalized_gpus,
            )
            evaluation = run_phase(
                repo=repo,
                result_root=result_root,
                python=python,
                metrics_root=metrics_root,
                locked_path=locked_config,
                plan=plan,
                phase="eval",
                gpus=normalized_gpus,
            )
            summary = strict_barrier_summary(tuning, evaluation)
            stable = {
                "schema_version": "recent-baseline-autopilot-complete-v1",
                **summary,
                "seed": SELECTION_SEED,
                "methods": list(METHODS),
                "locked_fingerprint": plan["locked_fingerprint"],
                "contract_plan_fingerprint": plan["contract_plan_fingerprint"],
                "execution_plan_fingerprint": plan["execution_plan_fingerprint"],
                "metrics_used_for_selection": False,
                "eval_feedback": False,
                "runtime_eligible_for_paper": False,
            }
            complete = {
                **stable,
                "completion_fingerprint": stable_fingerprint(stable),
                "completed_at": utc_now(),
            }
            target = result_root / COMPLETE_NAME
            if target.is_file():
                prior = load_json(target)
                if prior.get("completion_fingerprint") != complete["completion_fingerprint"]:
                    raise RuntimeError(f"different completion marker already exists: {target}")
            else:
                atomic_json(target, complete)
            assert_no_forbidden_artifacts(result_root)
            return complete
        except BaseException as exc:
            _write_failstop(result_root, "autopilot", exc)
            raise


def run_recent_baseline_barrier(
    *,
    repo: Path,
    recent_result_root: Path,
    python: Path,
    metrics_root: Path,
    seed: int,
    gpus: Sequence[str],
    workers_per_gpu: int,
    locked_config: Path,
) -> Mapping[str, Any]:
    """Narrow integration API for the main STAGE controller.

    One worker per physical GPU is an invariant rather than a tunable runtime
    option.  The returned mapping intentionally contains only the barrier
    status and the two strict unit-count summaries expected by STAGE.
    """
    if int(seed) != SELECTION_SEED:
        raise RuntimeError(
            f"recent-baseline barrier is frozen to seed {SELECTION_SEED}, received {seed}"
        )
    if int(workers_per_gpu) != 1:
        raise RuntimeError("recent baselines require exactly one worker per physical GPU")
    complete = run_recent_baselines(
        repo=repo,
        result_root=recent_result_root,
        python=python,
        metrics_root=metrics_root,
        locked_config=locked_config,
        gpus=gpus,
    )
    return strict_barrier_summary(complete["tuning"], complete["eval"])


def audit_all(
    repo: Path,
    result_root: Path,
    locked_config: Path,
) -> dict[str, Any]:
    plan = build_plan(repo, result_root, locked_config)
    assert_no_forbidden_artifacts(result_root)
    return {
        "tuning": audit_phase(result_root, plan, "tuning"),
        "eval": audit_phase(result_root, plan, "eval"),
        "failstop_present": (result_root / FAILSTOP_NAME).is_file(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--metrics-root", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "command",
        choices=("plan", "audit", "tune", "evaluate", "autopilot"),
    )
    args = parser.parse_args(argv)
    if args.metrics_root is None:
        args.metrics_root = args.repo / "external"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    result_root = args.result_root.resolve()
    locked_config = args.locked_config.resolve()
    python = Path(os.path.abspath(args.python))
    metrics_root = args.metrics_root.resolve()
    gpus = parse_gpus(args.gpus)

    validate_runtime_paths(repo, result_root, python, metrics_root, locked_config)
    with singleton_lock(result_root):
        if args.command == "audit":
            print(json.dumps(audit_all(repo, result_root, locked_config), indent=2))
            return 0
        failstop = result_root / FAILSTOP_NAME
        if failstop.exists():
            raise RuntimeError(f"existing fail-stop requires explicit audit: {failstop}")
        try:
            plan = build_plan(repo, result_root, locked_config)
            if args.command == "plan":
                print(
                    json.dumps(
                        {
                            "execution_plan_fingerprint": plan[
                                "execution_plan_fingerprint"
                            ]
                        },
                        indent=2,
                    )
                )
                return 0
            if args.command == "tune":
                report = run_phase(
                    repo=repo,
                    result_root=result_root,
                    python=python,
                    metrics_root=metrics_root,
                    locked_path=locked_config,
                    plan=plan,
                    phase="tuning",
                    gpus=gpus,
                )
                print(json.dumps(report, indent=2))
                return 0
            if args.command == "evaluate":
                report = run_phase(
                    repo=repo,
                    result_root=result_root,
                    python=python,
                    metrics_root=metrics_root,
                    locked_path=locked_config,
                    plan=plan,
                    phase="eval",
                    gpus=gpus,
                )
                print(json.dumps(report, indent=2))
                return 0
            if args.command == "autopilot":
                # The lock is already held, so execute the two phases directly
                # rather than recursively entering run_recent_baselines().
                tuning = run_phase(
                    repo=repo,
                    result_root=result_root,
                    python=python,
                    metrics_root=metrics_root,
                    locked_path=locked_config,
                    plan=plan,
                    phase="tuning",
                    gpus=gpus,
                )
                evaluation = run_phase(
                    repo=repo,
                    result_root=result_root,
                    python=python,
                    metrics_root=metrics_root,
                    locked_path=locked_config,
                    plan=plan,
                    phase="eval",
                    gpus=gpus,
                )
                summary = strict_barrier_summary(tuning, evaluation)
                stable = {
                    "schema_version": "recent-baseline-autopilot-complete-v1",
                    **summary,
                    "seed": SELECTION_SEED,
                    "methods": list(METHODS),
                    "locked_fingerprint": plan["locked_fingerprint"],
                    "contract_plan_fingerprint": plan["contract_plan_fingerprint"],
                    "execution_plan_fingerprint": plan["execution_plan_fingerprint"],
                    "metrics_used_for_selection": False,
                    "eval_feedback": False,
                    "runtime_eligible_for_paper": False,
                }
                complete = {
                    **stable,
                    "completion_fingerprint": stable_fingerprint(stable),
                    "completed_at": utc_now(),
                }
                target = result_root / COMPLETE_NAME
                if target.is_file():
                    prior = load_json(target)
                    if prior.get("completion_fingerprint") != complete["completion_fingerprint"]:
                        raise RuntimeError(f"different completion marker exists: {target}")
                else:
                    atomic_json(target, complete)
                print(json.dumps(complete, indent=2))
                return 0
            raise AssertionError(args.command)
        except BaseException as exc:
            _write_failstop(result_root, args.command, exc)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
