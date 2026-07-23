#!/usr/bin/env python3
"""Read-only, strict progress audit for a frozen STAGE v2 search.

The scanner deliberately does not call ``ensure_plan`` and never writes into
the result tree.  Existing physical unit JSONs are validated by importing and
calling :func:`scripts.stage_v2_search.valid_unit`; no reduced unit schema is
maintained here.  Missing planned units are normal for a partial scan.

Only frozen TSB-AD Tuning metadata and byte-level hashes are inspected.  The
scanner never opens a CSV to read labels or performs candidate selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import stage_v2_search as search  # noqa: E402


WATCH_SCHEMA = "stage-v2-search-watch-v1"
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl"}
FORBIDDEN_DIRECTORY_NAMES = {
    "checkpoint",
    "checkpoints",
    "embedding",
    "embeddings",
    "score_array",
    "score_arrays",
}
ATOMIC_TEMP_RE = re.compile(r"\.json\.tmp\.\d+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check(expected: Any, actual: Any) -> dict[str, Any]:
    return {"expected": expected, "actual": actual, "match": actual == expected}


def _safe_sha256(path: Path) -> str | None:
    return search.sha256_file(path) if path.is_file() else None


def validate_frozen_plan(
    repo: Path,
    protocol_path: Path,
    metrics_root: Path,
    result_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the existing plan and all frozen byte hashes without writing."""

    plan_path = result_root / search.PLAN_NAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"frozen plan is missing: {plan_path}")
    plan = search.load_json(plan_path)
    integrity: dict[str, Any] = {
        "plan_path": str(plan_path),
        "plan_sha256": search.sha256_file(plan_path),
        "checks": {},
        "errors": [],
    }
    checks: dict[str, Any] = integrity["checks"]
    checks["plan_schema"] = _check(search.PLAN_SCHEMA, plan.get("schema_version"))
    checks["execution_environment"] = _check(
        {
            "CUBLAS_WORKSPACE_CONFIG": search.CUBLAS_WORKSPACE_CONFIG,
            "torch_deterministic_algorithms": True,
            "CUDA_VISIBLE_DEVICES": "isolated_physical_gpu_per_worker",
            "runtime_eligible_for_paper": False,
        },
        plan.get("execution_environment"),
    )
    try:
        actual_plan_fingerprint = search.fingerprint(
            search._stable_plan_from_frozen(plan)
        )
    except Exception as exc:  # malformed frozen JSON still gets a machine report
        actual_plan_fingerprint = None
        integrity["errors"].append(f"cannot fingerprint frozen plan: {exc}")
    checks["plan_fingerprint"] = _check(
        plan.get("plan_fingerprint"), actual_plan_fingerprint
    )

    try:
        protocol, protocol_sha256, protocol_fingerprint = (
            search.load_and_validate_protocol(protocol_path)
        )
        checks["protocol_sha256"] = _check(
            plan.get("protocol_sha256"), protocol_sha256
        )
        checks["protocol_fingerprint"] = _check(
            plan.get("protocol_fingerprint"), protocol_fingerprint
        )
        checks["protocol_phase"] = _check(plan.get("phase"), protocol.get("phase"))
        checks["selection_is_tuning_only"] = _check(
            (search.SELECTION_SPLIT, False, False),
            (
                protocol.get("selection_split"),
                protocol.get("eval_feedback_used_by_runner"),
                protocol.get("confirmatory_claim_eligible"),
            ),
        )
    except Exception as exc:
        integrity["errors"].append(f"protocol validation failed: {exc}")
        checks["protocol"] = {"match": False}

    source_paths = {
        "scripts/stage_v2_search.py": Path(search.__file__).resolve(),
        "STAGE/stage.py": repo / "STAGE" / "stage.py",
        "common.py": repo / "common.py",
    }
    planned_sources = plan.get("source_sha256", {})
    checks["source_set"] = _check(sorted(source_paths), sorted(planned_sources))
    source_checks: dict[str, Any] = {}
    for name, path in source_paths.items():
        source_checks[name] = _check(planned_sources.get(name), _safe_sha256(path))
    integrity["sources"] = source_checks

    manifest_checks: dict[str, Any] = {}
    planned_manifests = plan.get("official_tuning_manifest_sha256", {})
    if isinstance(planned_manifests, Mapping):
        for track, expected_hash in sorted(planned_manifests.items()):
            path = repo / "data" / "File_List" / f"TSB-AD-{track}-Tuning.csv"
            manifest_checks[str(track)] = _check(expected_hash, _safe_sha256(path))
    else:
        integrity["errors"].append("plan lacks official Tuning manifest hashes")
    integrity["official_tuning_manifests"] = manifest_checks

    try:
        evaluator_root, evaluator_hashes = search._evaluator_source_hashes(metrics_root)
        planned_evaluator = plan.get("official_evaluator", {})
        checks["evaluator_root"] = _check(
            planned_evaluator.get("root_name"), evaluator_root.name
        )
        checks["evaluator_hashes"] = _check(
            planned_evaluator.get("source_sha256"), evaluator_hashes
        )
    except Exception as exc:
        integrity["errors"].append(f"official evaluator validation failed: {exc}")
        checks["evaluator"] = {"match": False}

    data_checks: dict[str, Any] = {}
    planned_data = plan.get("data_sha256", {})
    try:
        for track, datasets in plan["files"].items():
            for dataset, names in datasets.items():
                for file_name in names:
                    key = f"{track}/{dataset}/{file_name}"
                    path = repo / "data" / f"TSB-AD-{track}" / file_name
                    data_checks[key] = _check(planned_data.get(key), _safe_sha256(path))
    except Exception as exc:
        integrity["errors"].append(f"cannot enumerate frozen data files: {exc}")
    integrity["data"] = {
        "checked": len(data_checks),
        "mismatches": [key for key, value in data_checks.items() if not value["match"]],
        "all_match": bool(data_checks) and all(
            value["match"] for value in data_checks.values()
        ),
    }

    try:
        tasks = list(search._iter_tasks(plan))
        paths = [
            search.unit_path(result_root, track, dataset, signature, seed, file_name)
            .resolve()
            for track, dataset, file_name, seed, signature in tasks
        ]
        checks["expected_physical_units"] = _check(
            plan.get("expected_units"), len(tasks)
        )
        checks["unique_unit_paths"] = _check(len(tasks), len(set(paths)))
        logical = sum(
            len(search.execution_group(plan, *task[:3], task[4])["candidate_ids"])
            for task in tasks
        )
        checks["expected_logical_units"] = _check(
            plan.get("expected_logical_units"), logical
        )
    except Exception as exc:
        integrity["errors"].append(f"frozen task enumeration failed: {exc}")
        checks["task_plan"] = {"match": False}

    match_values = [
        value.get("match") is True
        for value in checks.values()
        if isinstance(value, Mapping)
    ]
    source_ok = all(value["match"] for value in source_checks.values())
    manifest_ok = bool(manifest_checks) and all(
        value["match"] for value in manifest_checks.values()
    )
    integrity["valid"] = (
        bool(match_values)
        and all(match_values)
        and source_ok
        and manifest_ok
        and integrity["data"]["all_match"]
        and not integrity["errors"]
    )
    return plan, integrity


def _empty_subset(expected: int = 0, expected_logical: int = 0) -> dict[str, int]:
    return {
        "valid": 0,
        "expected": int(expected),
        "missing": int(expected),
        "raw": 0,
        "invalid": 0,
        "duplicates": 0,
        "errors": 0,
        "valid_logical": 0,
        "expected_logical": int(expected_logical),
    }


def _payload_key(payload: Mapping[str, Any]) -> tuple[Any, ...] | None:
    fields = ("track", "dataset", "file", "seed", "execution_signature")
    if any(field not in payload for field in fields):
        return None
    return tuple(payload[field] for field in fields)


def scan_units(result_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Scan all existing unit JSONs, calling the runner's exact validator."""

    tasks = list(search._iter_tasks(plan))
    expected_by_path: dict[Path, tuple[str, str, str, int, str]] = {}
    by_subset: dict[str, dict[str, int]] = {}
    for task in tasks:
        track, dataset, file_name, seed, signature = task
        path = search.unit_path(
            result_root, track, dataset, signature, seed, file_name
        ).resolve()
        if path in expected_by_path:
            raise RuntimeError(f"frozen task plan has a duplicate path: {path}")
        expected_by_path[path] = task
        subset = f"{track}/{dataset}"
        group = search.execution_group(plan, track, dataset, file_name, signature)
        if subset not in by_subset:
            by_subset[subset] = _empty_subset()
        by_subset[subset]["expected"] += 1
        by_subset[subset]["missing"] += 1
        by_subset[subset]["expected_logical"] += len(group["candidate_ids"])

    units_root = result_root / "units"
    raw_paths = sorted(
        (path.resolve() for path in units_root.rglob("*.json") if path.is_file()),
        key=str,
    ) if units_root.exists() else []
    raw_payloads: dict[Path, Mapping[str, Any] | None] = {}
    parse_error_paths: list[str] = []
    key_counts: Counter[tuple[Any, ...]] = Counter()
    for path in raw_paths:
        try:
            payload = search.load_json(path)
        except Exception:
            payload = None
            parse_error_paths.append(str(path.relative_to(result_root)))
        raw_payloads[path] = payload
        if payload is not None:
            key = _payload_key(payload)
            if key is not None:
                key_counts[key] += 1

    duplicate_keys = {key: count for key, count in key_counts.items() if count > 1}
    duplicate_total = sum(count - 1 for count in duplicate_keys.values())
    for key, count in duplicate_keys.items():
        subset = f"{key[0]}/{key[1]}"
        by_subset.setdefault(subset, _empty_subset())["duplicates"] += count - 1

    valid = invalid = error_units = valid_logical = 0
    invalid_paths: list[str] = []
    unexpected_paths: list[str] = []
    for path in raw_paths:
        payload = raw_payloads[path]
        task = expected_by_path.get(path)
        if task is not None:
            track, dataset, file_name, seed, signature = task
            subset = f"{track}/{dataset}"
        elif payload is not None:
            subset = f"{payload.get('track', '__unknown__')}/{payload.get('dataset', '__unknown__')}"
        else:
            subset = "__unplanned__"
        row = by_subset.setdefault(subset, _empty_subset())
        row["raw"] += 1

        is_error = payload is not None and payload.get("error") not in (None, "")
        if is_error:
            error_units += 1
            row["errors"] += 1

        is_valid = False
        logical_width = 0
        if task is not None and payload is not None:
            track, dataset, file_name, seed, signature = task
            group = search.execution_group(
                plan, track, dataset, file_name, signature
            )
            logical_width = len(group["candidate_ids"])
            is_valid = search.valid_unit(
                payload,
                plan=plan,
                track=track,
                dataset=dataset,
                file_name=file_name,
                seed=seed,
                group=group,
            )
        if is_valid:
            valid += 1
            valid_logical += logical_width
            row["valid"] += 1
            row["valid_logical"] += logical_width
            row["missing"] -= 1
        else:
            invalid += 1
            row["invalid"] += 1
            relative = str(path.relative_to(result_root))
            invalid_paths.append(relative)
            if task is None:
                unexpected_paths.append(relative)
            else:
                row["missing"] -= 1

    # Existing invalid planned files are not "missing"; they are explicit
    # fail-stop evidence.  Missing therefore counts only absent planned paths.
    missing = sum(1 for path in expected_by_path if path not in raw_payloads)

    return {
        "valid": valid,
        "expected": len(tasks),
        "missing": missing,
        "raw": len(raw_paths),
        "invalid": invalid,
        "duplicates": duplicate_total,
        "errors": error_units,
        "valid_logical": valid_logical,
        "expected_logical": int(plan["expected_logical_units"]),
        "by_subset": {key: by_subset[key] for key in sorted(by_subset)},
        "invalid_paths": invalid_paths,
        "unexpected_paths": unexpected_paths,
        "parse_error_paths": parse_error_paths,
        "duplicate_keys": [
            {
                "track": key[0],
                "dataset": key[1],
                "file": key[2],
                "seed": key[3],
                "execution_signature": key[4],
                "copies": count,
            }
            for key, count in sorted(duplicate_keys.items(), key=lambda item: repr(item[0]))
        ],
    }


def scan_forbidden_artifacts(result_root: Path) -> dict[str, Any]:
    forbidden: list[str] = []
    atomic_temps: list[str] = []
    if result_root.exists():
        for path in result_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(result_root)
            if ATOMIC_TEMP_RE.search(path.name):
                atomic_temps.append(str(relative))
                continue
            lowered_parts = {part.lower() for part in relative.parts[:-1]}
            if (
                path.suffix.lower() in FORBIDDEN_SUFFIXES
                or lowered_parts.intersection(FORBIDDEN_DIRECTORY_NAMES)
                or path.suffix.lower() == ".log"
            ):
                forbidden.append(str(relative))
    return {
        "count": len(forbidden),
        "paths": sorted(forbidden),
        "inflight_atomic_temp_count": len(atomic_temps),
        "inflight_atomic_temp_paths": sorted(atomic_temps),
    }


def _linux_process_table(proc_root: Path = Path("/proc")) -> dict[int, int]:
    parents: dict[int, int] = {}
    if os.name != "posix" or not proc_root.is_dir():
        return parents
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
            close = text.rfind(")")
            fields = text[close + 2 :].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    return parents


def process_status(pid_file: Path | None, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    status: dict[str, Any] = {
        "pid_file": None if pid_file is None else str(pid_file),
        "pid_file_exists": bool(pid_file is not None and pid_file.is_file()),
        "parent_pid": None,
        "parent_alive": False,
        "worker_count": 0,
        "worker_pids": [],
    }
    if pid_file is None or not pid_file.is_file():
        return status
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        status["pid_file_valid"] = False
        return status
    status["pid_file_valid"] = True
    status["parent_pid"] = pid
    parents = _linux_process_table(proc_root)
    if parents:
        status["parent_alive"] = pid in parents
        descendants = {pid}
        changed = True
        while changed:
            changed = False
            for child, parent in parents.items():
                if parent in descendants and child not in descendants:
                    descendants.add(child)
                    changed = True
        workers: list[int] = []
        for candidate in sorted(descendants - {pid}):
            try:
                command = (proc_root / str(candidate) / "cmdline").read_bytes().replace(
                    b"\0", b" "
                )
            except OSError:
                continue
            if b"stage_v2_search.py" in command and b"_worker" in command:
                workers.append(candidate)
        status["worker_count"] = len(workers)
        status["worker_pids"] = workers
    else:
        try:
            os.kill(pid, 0)
            status["parent_alive"] = True
        except (OSError, PermissionError):
            pass
    return status


def build_report(
    repo: Path,
    protocol_path: Path,
    result_root: Path,
    metrics_root: Path,
    pid_file: Path | None,
) -> dict[str, Any]:
    plan, integrity = validate_frozen_plan(
        repo, protocol_path, metrics_root, result_root
    )
    units = scan_units(result_root, plan)
    forbidden = scan_forbidden_artifacts(result_root)
    processes = process_status(pid_file)
    anomalies = (
        not integrity["valid"]
        or units["invalid"] > 0
        or units["duplicates"] > 0
        or units["errors"] > 0
        or forbidden["count"] > 0
    )
    complete = units["valid"] == units["expected"] and units["missing"] == 0
    if anomalies:
        state = "invalid"
    elif complete:
        state = "complete"
    elif processes["parent_alive"]:
        state = "running"
    else:
        state = "partial"
    return {
        "schema_version": WATCH_SCHEMA,
        "observed_at": utc_now(),
        "read_only": True,
        "selection_split": "official TSB-AD Tuning only",
        "eval_feedback_used": False,
        "runtime_eligible_for_paper": False,
        "state": state,
        "ok": not anomalies,
        "phase": plan.get("phase"),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "integrity": integrity,
        "units": units,
        "processes": processes,
        "forbidden_artifacts": forbidden,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(
            args.repo.resolve(),
            args.protocol.resolve(),
            args.result_root.resolve(),
            args.metrics_root.resolve(),
            None if args.pid_file is None else args.pid_file.resolve(),
        )
    except Exception as exc:
        report = {
            "schema_version": WATCH_SCHEMA,
            "observed_at": utc_now(),
            "read_only": True,
            "selection_split": "official TSB-AD Tuning only",
            "eval_feedback_used": False,
            "runtime_eligible_for_paper": False,
            "state": "scan_failed",
            "ok": False,
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
