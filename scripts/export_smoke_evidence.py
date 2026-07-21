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

import controller  # noqa: E402
from common import METRICS, complete_record  # noqa: E402


METHODS = (*controller.NON_DEEP, *controller.TSB_DEEP, *controller.EXTERNAL_DEEP)
TRACKS = ("U", "M")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a reviewable, runtime-free local smoke-test record"
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=PROJECT / "results" / "smoke" / "local_00gwk_validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "validation" / "local_00gwk_smoke.json",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    result = args.result.resolve()
    records: dict[tuple[str, str], dict] = {}
    files: dict[str, set[str]] = {track: set() for track in TRACKS}

    for path in sorted((result / "units").rglob("*.json")):
        if not complete_record(path):
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        method = item.get("method")
        track = item.get("track")
        if method not in METHODS or track not in TRACKS:
            continue
        if item.get("seed") != args.seed:
            raise ValueError(f"unexpected seed in {path}")
        key = (method, track)
        if key in records:
            raise ValueError(f"duplicate method/track record: {key}")
        metrics = {name: float(item["metrics"][name]) for name in METRICS}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"non-finite metric in {path}")
        records[key] = {
            "method": method,
            "track": track,
            "dataset": item["dataset"],
            "file": item["file"],
            "metrics": metrics,
        }
        files[track].add(item["file"])

    expected = {(method, track) for method in METHODS for track in TRACKS}
    missing = sorted(expected - set(records))
    unexpected = sorted(set(records) - expected)
    if missing or unexpected:
        raise ValueError(f"incomplete evidence: missing={missing} unexpected={unexpected}")
    if any(len(names) != 1 for names in files.values()):
        raise ValueError(f"each track must use exactly one smoke file: {files}")

    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        torch_info = {"import_error": f"{type(exc).__name__}: {exc}"}

    formal_paano = PROJECT / "compat" / "paano_official_one_linux.py"
    local_paano = PROJECT / "compat" / "paano_official_one_windows.py"
    evidence = {
        "schema_version": "stage-local-smoke-evidence-v1",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "dependency, interface, and six-metric portability validation",
        "runtime_eligible_for_paper": False,
        "runtime_note": "All elapsed-time fields are intentionally omitted.",
        "seed": args.seed,
        "training_prefix_policy": "filename_declared_label_blind",
        "methods": list(METHODS),
        "tracks": list(TRACKS),
        "expected_records": len(expected),
        "valid_records": len(records),
        "error_records": 0,
        "files": {track: next(iter(names)) for track, names in files.items()},
        "metrics": list(METRICS),
        "environment": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch_info,
        },
        "integrity": {
            "protocol_sha256": sha256(PROJECT / "protocol.json"),
            "experiment_policy_sha256": sha256(PROJECT / "experiment_policy.json"),
            "stage_source_sha256": sha256(PROJECT / "STAGE" / "stage.py"),
            "controller_sha256": sha256(PROJECT / "controller.py"),
            "formal_paano_adapter_sha256": sha256(formal_paano),
            "windows_smoke_paano_adapter_sha256": sha256(local_paano),
        },
        "compatibility_note": (
            "On Windows smoke tests only, PaAno receives a POSIX-form path because "
            "the released loader splits paths on '/'. Formal Linux runs use the frozen adapter."
        ),
        "records": [records[key] for key in sorted(records)],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": evidence["status"],
        "valid_records": evidence["valid_records"],
        "error_records": evidence["error_records"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
