from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import METRICS, atomic_json, complete_record, dataset_name  # noqa: E402


METHODS = ("DPAD_AAAI24", "DNE_AAAI25")
FILES = {
    "U": "140_MSL_id_1_Sensor_tr_530_1st_630.csv",
    "M": "124_TAO_id_9_Environment_tr_500_1st_1.csv",
}


def target_path(root: Path, method: str, track: str, file_name: str) -> Path:
    return root / "units" / method / track / dataset_name(file_name) / (Path(file_name).stem + ".json")


def validate(path: Path) -> dict[str, float]:
    if not complete_record(path):
        raise ValueError(f"invalid result {path}")
    item = json.loads(path.read_text(encoding="utf-8"))
    metrics = {name: float(item["metrics"][name]) for name in METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"non-finite metrics in {path}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the two recent AAAI extensions")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--result",
        type=Path,
        default=PROJECT / "results" / "smoke" / "aaai_recent_local_00gwk_validation",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpad-epochs", type=int, default=3)
    parser.add_argument("--dne-epochs", type=int, default=3)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    project = args.project.resolve()
    result = args.result.resolve()
    result.mkdir(parents=True, exist_ok=True)
    runner = project / "extensions" / "aaai_recent" / "run_one_recent.py"
    jobs = [(method, track, FILES[track]) for track in ("U", "M") for method in METHODS]
    manifest = {
        "schema_version": "stage-aaai-recent-smoke-v1",
        "purpose": "implementation and six-metric portability only",
        "runtime_eligible_for_paper": False,
        "seed": args.seed,
        "profile": "smoke",
        "methods": list(METHODS),
        "files": FILES,
        "expected_units": len(jobs),
        "completed_units": 0,
        "error_units": 0,
        "units": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    manifest_path = result / "smoke_manifest.json"
    atomic_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu

    for index, (method, track, file_name) in enumerate(jobs, start=1):
        output = target_path(result, method, track, file_name)
        command = [
            str(args.python),
            str(runner),
            "--repo", str(project),
            "--method", method,
            "--track", track,
            "--file", file_name,
            "--seed", str(args.seed),
            "--output", str(output),
            "--stage-source", str(project / "STAGE" / "stage.py"),
            "--paano-root", str(project / "external" / "PaAno"),
            "--profile", "smoke",
            "--dpad-epochs", str(args.dpad_epochs),
            "--dne-epochs", str(args.dne_epochs),
            "--dne-architecture", "resmlp",
        ]
        completed = subprocess.run(command, env=environment, check=False)
        unit = {"method": method, "track": track, "file": file_name, "status": "error"}
        if completed.returncode == 0:
            unit["status"] = "complete"
            unit["metrics"] = validate(output)
            manifest["completed_units"] += 1
        else:
            manifest["error_units"] += 1
            if output.exists():
                unit["error"] = json.loads(output.read_text(encoding="utf-8")).get("error")
        manifest["units"].append(unit)
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(manifest_path, manifest)
        print(f"[{index}/{len(jobs)}] {unit['status']} {method} {track} {file_name}", flush=True)

    manifest["status"] = "complete" if manifest["error_units"] == 0 else "incomplete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "completed_units": manifest["completed_units"],
        "error_units": manifest["error_units"],
        "result": str(result),
    }, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
