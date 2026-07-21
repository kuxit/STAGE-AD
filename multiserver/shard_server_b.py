from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

from common import METRICS, atomic_json, dataset_name, sha256  # noqa: E402
from controller import EXTERNAL_DEEP, TSB_DEEP, run_unit, standard_path  # noqa: E402
from input_integrity import verify as verify_input_integrity  # noqa: E402


ALL_METHODS = (*TSB_DEEP, *EXTERNAL_DEEP)
DATASETS = {"SVDB", "TAO", "CATSv2"}
EXCLUDED = {"124_TAO_id_9_Environment_tr_500_1st_1.csv"}
PARITY_FILE = "124_TAO_id_9_Environment_tr_500_1st_1.csv"
EXPECTED_FILES = 43

EXPECTED_HASHES = {
    "common": "e1c8cb0d732dd18bde179ef552577eeea06151903e8f8821a1afe31e6e67ba59",
    "duoba": "94f31c50641c1c097b4b60cad50994d99dd6148148f2ae35cddc51579d242dd3",
    "controller": "88a394978d2547608e2ee58e30e9797a58a6bc950ca950c8c802a8b567dd8d78",
    "deep_runner": "a8bfb73293005ba5e9da9c3fc321e4f1b9cbae7db908434ac03568abd459840f",
    "classical_runner": "d97a1e4083bd1157c73afc3255b35eb871add08361c95c8e52cf8a95f90ca83c",
    "paano_one": "00437318c9a6b6ded357fd88892b336ea6cebc5fe6c7755a08bfa909d67ddf75",
    "gboc_one": "3861da207667fe4944a45712088a9072abec07fa7ce2167bbcb8751f29082e19",
    "gboc_config": "1839041de874a6a0586628274c51a14a4b6b66a049c2a7d46d38e2513d511fab",
    "memto_one": "e923c6187b0954e0c01cc3fb09eade7a037b3e66148386bab19327f3d6901371",
    "dcdetector_one": "2fb559dcfb3a7deb5b0a5c745ac5b524969f1cabf95946d2b05d1178dae1bb38",
}

# Approximate relative costs are used only to reduce scheduling tail latency.
# They never affect a method configuration, input, score, or metric.
METHOD_COST = {
    "GBOC": 100.0,
    "DuoBa": 8.0,
    "AnomalyTransformer": 5.0,
    "PatchTST": 4.5,
    "TimesNet": 4.0,
    "DCdetector": 3.0,
    "PaAno": 2.8,
    "USAD": 2.5,
    "TranAD": 2.4,
    "MEMTO": 2.3,
    "OmniAnomaly": 2.2,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def paths(root: Path, result: Path, python: Path) -> argparse.Namespace:
    experiment = root / "DuoBa-Baseline-Seed2027"
    return argparse.Namespace(
        repo=root,
        experiment=experiment,
        result=result,
        python=python,
        duoba_source=root / "GROVE-AD-V3" / "grove_ad.py",
        paano_root=root / "external" / "PaAno",
        paano_one=root / "GROVE-AD-V3" / "paano_official_one.py",
        gboc_root=root / "external" / "GBOC_official_55a2a31",
        gboc_one=root / "spectral_tsad_v5" / "baselines" / "gboc" / "run_one.py",
        gboc_config=root / "spectral_tsad_v5" / "baselines" / "gboc" / "configs" / "official_release.json",
        memto_root=root / "external" / "MEMTO_official",
        memto_one=experiment / "run_one_memto.py",
        dcdetector_root=root / "external" / "DCdetector",
        dcdetector_one=experiment / "run_one_dcdetector.py",
        seed=2027,
    )


def source_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "common": sha256(args.experiment / "common.py"),
        "duoba": sha256(args.duoba_source),
        "controller": sha256(args.experiment / "controller.py"),
        "deep_runner": sha256(args.experiment / "run_one_deep.py"),
        "classical_runner": sha256(args.experiment / "run_one_classical.py"),
        "paano_one": sha256(args.paano_one),
        "gboc_one": sha256(args.gboc_one),
        "gboc_config": sha256(args.gboc_config),
        "memto_one": sha256(args.memto_one),
        "dcdetector_one": sha256(args.dcdetector_one),
    }


def frozen_jobs(
    args: argparse.Namespace,
    methods: tuple[str, ...],
    files: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    if any(dataset_name(name) not in DATASETS or name in EXCLUDED for name in files):
        raise RuntimeError("split plan contains an out-of-bound data file")
    jobs = [(method, "M", name) for name in files for method in methods]
    expected_units = EXPECTED_FILES * len(methods)
    if len(files) != EXPECTED_FILES or len(jobs) != expected_units:
        raise RuntimeError(f"unexpected shard size: files={len(files)} units={len(jobs)}")

    def cost(job: tuple[str, str, str]) -> tuple[float, str, str]:
        method, track, name = job
        size = (args.repo / "data" / f"TSB-AD-{track}" / name).stat().st_size
        return (METHOD_COST[method] * math.log2(max(size, 2)), method, name)

    return sorted(jobs, key=cost, reverse=True)


def validate_inputs(args: argparse.Namespace) -> dict[str, str]:
    actual = source_hashes(args)
    if actual != EXPECTED_HASHES:
        raise RuntimeError(f"frozen source hash mismatch: {actual}")
    input_manifest = args.repo / "server_b_input_manifest.json"
    if not input_manifest.exists():
        raise RuntimeError(f"missing input manifest: {input_manifest}")
    payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "verified" or payload.get("seed") != 2027:
        raise RuntimeError("server B input manifest has not been verified")
    return actual


def valid_job_record(path: Path, job: tuple[str, str, str]) -> bool:
    method, track, file_name = job
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if payload.get("method") != method or payload.get("track") != track or payload.get("file") != file_name:
        return False
    if payload.get("dataset") != dataset_name(file_name):
        return False
    if payload.get("seed") != 2027 or payload.get("error") is not None:
        return False
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False
    try:
        values = [float(metrics[name]) for name in METRICS]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)


def count_valid(args: argparse.Namespace, jobs: list[tuple[str, str, str]]) -> int:
    return sum(valid_job_record(standard_path(args.result, *job), job) for job in jobs)


ENVIRONMENT_SCRIPT = r'''
import hashlib, json, pathlib, platform, subprocess, sys, torch
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
print(json.dumps({
 'python_executable':str(pathlib.Path(sys.executable).resolve()),
 'python':platform.python_version(),
 'torch':torch.__version__,
 'torch_cuda':torch.version.cuda,
 'cudnn':torch.backends.cudnn.version(),
 'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
 'pip_freeze_sha256':hashlib.sha256(freeze.encode()).hexdigest(),
}))
'''


def runtime_environment(python: Path) -> dict[str, Any]:
    return json.loads(subprocess.check_output([str(python), "-c", ENVIRONMENT_SCRIPT], text=True))


def environment_matches(reported: dict[str, Any], current: dict[str, Any]) -> bool:
    for key, value in reported.items():
        if key == "python_executable":
            if Path(str(value)).resolve() != Path(str(current.get(key, ""))).resolve():
                return False
        elif current.get(key) != value:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/AAAI"))
    parser.add_argument("--result", type=Path, default=Path("/root/autodl-tmp/AAAI/results/DUOBA_10SUBSET_BASELINES/seed2027_v2_server_b"))
    parser.add_argument("--python", type=Path, default=Path("/root/autodl-tmp/duoba-env/bin/python"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--parity-report", default="parity_report.json")
    parser.add_argument("--method-plan", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    cli = parser.parse_args()
    root = cli.root.resolve()
    result = cli.result.resolve()
    result.mkdir(parents=True, exist_ok=True)
    lock_stream = (result / "shard.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another formal shard process already owns this result root") from error
    methods = tuple(item.strip() for item in cli.methods.split(",") if item.strip())
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(ALL_METHODS):
        raise RuntimeError(f"invalid method shard: {methods}")
    if cli.method_plan:
        raw_plan = json.loads(cli.method_plan.resolve().read_text(encoding="utf-8"))
        if set(raw_plan) != set(methods):
            raise RuntimeError("method plan must cover exactly the formal method set")
    else:
        raw_plan = {
            method: {"python": str(Path(os.path.abspath(cli.python))), "parity_report": cli.parity_report}
            for method in methods
        }

    args_by_method: dict[str, argparse.Namespace] = {}
    method_execution: dict[str, dict[str, Any]] = {}
    environment_cache: dict[Path, dict[str, Any]] = {}
    hashes: dict[str, str] | None = None
    for method in methods:
        spec = raw_plan.get(method)
        if not isinstance(spec, dict):
            raise RuntimeError(f"invalid method plan entry for {method}")
        python = Path(os.path.abspath(str(spec.get("python", ""))))
        parity_name = spec.get("parity_report")
        if not python.is_file() or not isinstance(parity_name, str) or "/" in parity_name or "\\" in parity_name:
            raise RuntimeError(f"invalid execution environment for {method}")
        method_args = paths(root, result, python)
        method_hashes = validate_inputs(method_args)
        if hashes is None:
            hashes = method_hashes
        elif method_hashes != hashes:
            raise RuntimeError("method environments see different frozen sources")
        parity_path = root / "parity_pilot" / parity_name
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        environment = parity.get("candidate_environment")
        if python not in environment_cache:
            environment_cache[python] = runtime_environment(python)
        current_environment = environment_cache[python]
        expected_environment = spec.get("environment", current_environment)
        if parity.get("seed") != 2027 or parity.get("source_sha256") != method_hashes:
            raise RuntimeError(f"parity report does not match the frozen run for {method}")
        if parity.get("file") != PARITY_FILE:
            raise RuntimeError(f"parity report uses an unexpected pilot file for {method}")
        if parity.get("policy_set_before_run") != {
            "max_metric_absolute_difference": 0.005,
            "mean_metric_absolute_difference": 0.002,
        }:
            raise RuntimeError(f"parity report uses an unexpected gate for {method}")
        if not parity.get("methods", {}).get(method, {}).get("passed"):
            raise RuntimeError(f"{method} did not pass the frozen parity gate")
        candidate_name = parity.get("candidate_name")
        if not isinstance(candidate_name, str) or "/" in candidate_name or "\\" in candidate_name:
            raise RuntimeError(f"parity report does not identify its candidate for {method}")
        method_gate = parity["methods"][method]
        candidate_path = standard_path(root / "parity_pilot" / candidate_name, method, "M", PARITY_FILE)
        reference_path = root / "parity_pilot" / "reference" / f"{method}.json"
        pilot_job = (method, "M", PARITY_FILE)
        if not valid_job_record(candidate_path, pilot_job) or not valid_job_record(reference_path, pilot_job):
            raise RuntimeError(f"parity candidate or reference is invalid for {method}")
        if method_gate.get("candidate_record_sha256") != sha256(candidate_path):
            raise RuntimeError(f"parity candidate changed for {method}")
        if method_gate.get("reference_record_sha256") != sha256(reference_path):
            raise RuntimeError(f"parity reference changed for {method}")
        if not isinstance(environment, dict) or not environment_matches(environment, current_environment):
            raise RuntimeError(f"formal and parity interpreters differ for {method}")
        if expected_environment != current_environment:
            raise RuntimeError(f"formal environment fingerprint changed for {method}")
        args_by_method[method] = method_args
        method_execution[method] = {
            "python": str(python),
            "parity_report": parity_name,
            "parity_report_sha256": sha256(parity_path),
            "environment": current_environment,
        }
    if hashes is None:
        raise RuntimeError("no formal methods selected")
    base_args = args_by_method[methods[0]]
    split_plan_path = root / "server_b_split_plan.json"
    integrity = verify_input_integrity(root, split_plan_path)
    split_plan = json.loads(split_plan_path.read_text(encoding="utf-8"))
    formal_files = tuple(Path(record["path"]).name for record in split_plan["formal_files"])
    jobs = frozen_jobs(base_args, methods, formal_files)
    expected_units = EXPECTED_FILES * len(methods)
    gpus = tuple(item.strip() for item in cli.gpus.split(",") if item.strip())
    if gpus != ("0", "1"):
        raise RuntimeError("the formal shard requires physical GPUs 0 and 1")

    manifest_path = result / "shard_manifest.json"
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    immutable: dict[str, Any] = {
        "protocol": "duoba-10subset-seed2027-v2-server-b-shard1",
        "seed": 2027,
        "datasets": ["SVDB", "TAO", "CATSv2"],
        "excluded_files": sorted(EXCLUDED),
        "formal_files": list(formal_files),
        "split_plan_sha256": integrity["plan_sha256"],
        "input_digest": integrity["input_digest"],
        "methods": list(methods),
        "expected_units": expected_units,
        "method_execution": method_execution,
        "metrics": list(METRICS),
        "source_sha256": hashes,
    }
    if previous:
        mismatches = [key for key, value in immutable.items() if previous.get(key) != value]
        if mismatches:
            raise RuntimeError(f"refusing to resume a different formal shard identity: {mismatches}")
    manifest: dict[str, Any] = {
        **immutable,
        "status": "running",
        "started_at": previous.get("started_at", now()),
        "resumed_at": now() if previous else None,
        "resume_count": int(previous.get("resume_count", 0)) + (1 if previous else 0),
        "previous_error_units": int(previous.get("error_units", 0)) if previous else 0,
        "completed_units": count_valid(base_args, jobs),
        "error_units": 0,
    }
    expected_paths = {standard_path(result, *job).resolve() for job in jobs}
    units_root = result / "units"
    if units_root.exists():
        unexpected = [path for path in units_root.rglob("*.json") if path.resolve() not in expected_paths]
        if unexpected:
            raise RuntimeError(f"result root contains out-of-plan unit records: {unexpected[:3]}")
    if previous.get("status") == "complete" and manifest["completed_units"] == expected_units:
        print(f"[{now()}] server B shard was already verified complete", flush=True)
        return 0
    if cli.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "methods": list(methods),
                    "formal_files": len(formal_files),
                    "expected_units": expected_units,
                    "split_plan_sha256": integrity["plan_sha256"],
                    "completed_units": manifest["completed_units"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    atomic_json(manifest_path, manifest)
    print(f"[{now()}] server B shard starts at {manifest['completed_units']}/{expected_units}", flush=True)

    queue = [job for job in jobs if not valid_job_record(standard_path(base_args.result, *job), job)]
    queue_lock = threading.Lock()
    manifest_lock = threading.Lock()
    stop = threading.Event()

    def update(item: dict[str, Any]) -> None:
        with manifest_lock:
            completed = count_valid(base_args, jobs)
            if item.get("status") == "error":
                manifest["error_units"] = int(manifest.get("error_units", 0)) + 1
                stop.set()
            manifest.update({"completed_units": completed, "last_unit": item, "updated_at": now()})
            atomic_json(manifest_path, manifest)
            print(f"[{completed}/{expected_units}] {item}", flush=True)

    def worker(gpu: str) -> None:
        while not stop.is_set():
            with queue_lock:
                if not queue:
                    return
                job = queue.pop(0)
            item = run_unit(args_by_method[job[0]], *job, gpu)
            if item.get("status") == "complete" and not valid_job_record(standard_path(result, *job), job):
                item = {**item, "status": "error", "error": "runner produced an invalid formal record"}
            update(item)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()

    completed = count_valid(base_args, jobs)
    manifest["completed_units"] = completed
    manifest["status"] = "complete" if completed == expected_units and manifest["error_units"] == 0 else "incomplete"
    manifest["completed_at"] = now()
    atomic_json(manifest_path, manifest)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
