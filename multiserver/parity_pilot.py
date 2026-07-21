from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(EXPERIMENT))

from common import METRICS, atomic_json, complete_record, sha256  # noqa: E402
from controller import EXTERNAL_DEEP, TSB_DEEP, run_unit, standard_path  # noqa: E402
from shard_server_b import paths, source_hashes, EXPECTED_HASHES  # noqa: E402


FILE = "124_TAO_id_9_Environment_tr_500_1st_1.csv"
METHODS = (*TSB_DEEP, *EXTERNAL_DEEP)
MAX_ABS_DIFF = 0.005
MEAN_ABS_DIFF = 0.002


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/AAAI"))
    parser.add_argument("--python", type=Path, default=Path("/root/autodl-tmp/duoba-env/bin/python"))
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--report-name", default="parity_report.json")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--gpus", default="0,1")
    cli = parser.parse_args()
    python = Path(os.path.abspath(cli.python)).resolve()
    if Path(sys.executable).resolve() != python:
        raise RuntimeError("the pilot script and child runners must use the same interpreter")
    if "/" in cli.candidate_name or "\\" in cli.candidate_name:
        raise RuntimeError("candidate name must be a single path component")
    if "/" in cli.report_name or "\\" in cli.report_name:
        raise RuntimeError("report name must be a single path component")
    root = cli.root.resolve()
    result = root / "parity_pilot" / cli.candidate_name
    report_path = root / "parity_pilot" / cli.report_name
    if report_path.exists() or (result.exists() and any(result.iterdir())):
        raise RuntimeError("refusing to reuse an existing parity candidate or report")
    args = paths(root, result, python)
    hashes = source_hashes(args)
    if hashes != EXPECTED_HASHES:
        raise RuntimeError(f"source hash mismatch: {hashes}")

    methods = tuple(item.strip() for item in cli.methods.split(",") if item.strip())
    if not methods or len(set(methods)) != len(methods) or not set(methods).issubset(METHODS):
        raise RuntimeError(f"invalid parity method set: {methods}")
    gpus = tuple(item.strip() for item in cli.gpus.split(",") if item.strip())
    if not gpus or not set(gpus).issubset({"0", "1"}):
        raise RuntimeError(f"invalid parity GPU set: {gpus}")
    jobs = [(method, "M", FILE) for method in methods]
    queue = list(jobs)

    def worker(gpu: str) -> None:
        while queue:
            try:
                job = queue.pop(0)
            except IndexError:
                return
            item = run_unit(args, *job, gpu)
            print(item, flush=True)
            if item.get("status") == "error":
                raise RuntimeError(str(item))

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()

    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    report = {
        "status": "running",
        "file": FILE,
        "candidate_name": cli.candidate_name,
        "methods_requested": list(methods),
        "seed": 2027,
        "policy_set_before_run": {
            "max_metric_absolute_difference": MAX_ABS_DIFF,
            "mean_metric_absolute_difference": MEAN_ABS_DIFF,
        },
        "source_sha256": hashes,
        "candidate_environment": {
            "python_executable": str(python),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        },
        "methods": {},
        "checked_at": now(),
    }
    passed = True
    for method, track, file_name in jobs:
        candidate_path = standard_path(result, method, track, file_name)
        reference_path = root / "parity_pilot" / "reference" / f"{method}.json"
        if not complete_record(candidate_path) or not complete_record(reference_path):
            report["methods"][method] = {"passed": False, "reason": "missing valid record"}
            passed = False
            continue
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
        candidate = candidate_payload["metrics"]
        reference = reference_payload["metrics"]
        differences = {name: abs(float(candidate[name]) - float(reference[name])) for name in METRICS}
        maximum = max(differences.values())
        mean = sum(differences.values()) / len(differences)
        method_passed = maximum <= MAX_ABS_DIFF and mean <= MEAN_ABS_DIFF
        report["methods"][method] = {
            "passed": method_passed,
            "max_abs_diff": maximum,
            "mean_abs_diff": mean,
            "differences": differences,
            "candidate_record_sha256": sha256(candidate_path),
            "reference_record_sha256": sha256(reference_path),
            "pilot_gpu": candidate_payload.get("gpu"),
        }
        passed = passed and method_passed
    report["status"] = "passed" if passed else "failed"
    report["completed_at"] = now()
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
