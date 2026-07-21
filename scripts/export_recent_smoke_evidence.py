from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from common import METRICS, complete_record  # noqa: E402


METHODS = ("DPAD_AAAI24", "DNE_AAAI25")
TRACKS = ("U", "M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export runtime-free recent-baseline smoke evidence")
    parser.add_argument(
        "--result",
        type=Path,
        default=PROJECT / "results" / "smoke" / "aaai_recent_local_00gwk_validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "validation" / "aaai_recent_local_00gwk_smoke.json",
    )
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    records: dict[tuple[str, str], dict] = {}
    for path in sorted((args.result.resolve() / "units").rglob("*.json")):
        if not complete_record(path):
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        key = (item.get("method"), item.get("track"))
        if key[0] not in METHODS or key[1] not in TRACKS:
            continue
        if item.get("seed") != args.seed or item.get("profile") != "smoke":
            raise ValueError(f"unexpected record identity in {path}")
        if key in records:
            raise ValueError(f"duplicate record {key}")
        metrics = {name: float(item["metrics"][name]) for name in METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"non-finite metrics in {path}")
        records[key] = {
            "method": key[0],
            "track": key[1],
            "dataset": item["dataset"],
            "file": item["file"],
            "profile": "smoke",
            "window": item["window"],
            "training_windows_used": item["training_windows_used"],
            "metrics": metrics,
        }

    expected = {(method, track) for method in METHODS for track in TRACKS}
    if set(records) != expected:
        raise ValueError(f"incomplete evidence: missing={sorted(expected - set(records))}")
    import torch

    evidence = {
        "schema_version": "duoba-aaai-recent-smoke-evidence-v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "implementation and six-metric portability validation only",
        "formal_eval_eligible": False,
        "formal_eval_note": "Formal settings remain blocked until official-Tuning freeze.",
        "runtime_eligible_for_paper": False,
        "runtime_note": "All elapsed-time fields are intentionally omitted.",
        "seed": args.seed,
        "methods": list(METHODS),
        "tracks": list(TRACKS),
        "valid_records": len(records),
        "error_records": 0,
        "metrics": list(METRICS),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
        },
        "integrity": {
            "base_protocol_sha256": sha256(PROJECT / "protocol.json"),
            "extension_protocol_sha256": sha256(
                PROJECT / "extensions" / "aaai_recent" / "protocol_extension.json"
            ),
            "model_source_sha256": sha256(PROJECT / "extensions" / "aaai_recent" / "models.py"),
            "runner_source_sha256": sha256(
                PROJECT / "extensions" / "aaai_recent" / "run_one_recent.py"
            ),
        },
        "records": [records[key] for key in sorted(records)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": evidence["status"],
        "valid_records": evidence["valid_records"],
        "error_records": evidence["error_records"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
