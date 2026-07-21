from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_TREE_DIGEST = {
    "tree_sha256": "0ed191fde5e5e2bb4d8b2a290238e5092b1799ef3146f391ccb5118293c2bd7b",
    "file_count": 9143,
    "total_bytes": 767676814,
    "selected_data_files": 44,
}
EXPECTED_PLAN_SHA256 = "543c884bfc761f5fa767392b5bc29c9b7c3db6f56a07d2abd641cd3f14c90178"
DATASETS = {"SVDB", "TAO", "CATSv2"}
PILOT_FILE = "124_TAO_id_9_Environment_tr_500_1st_1.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def plan_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("plan_sha256", None)
    return hashlib.sha256(canonical_json(unsigned)).hexdigest()


def snapshot(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment = root / "DuoBa-Baseline-Seed2027"
    sys.path.insert(0, str(experiment))
    from common import dataset_name  # noqa: PLC0415
    from controller import selected_files  # noqa: PLC0415

    names = [
        name
        for name in selected_files(root)["M"]
        if dataset_name(name) in DATASETS
    ]
    if PILOT_FILE not in names:
        names.append(PILOT_FILE)
    data_paths = [root / "data" / "TSB-AD-M" / name for name in sorted(set(names))]
    paths = list(data_paths)
    paths.append(root / "spectral_tsad_v5" / "scripts" / "efficiency_monitor.py")
    trees = [
        root / "external",
        root / "GROVE-AD-V3",
        root / "spectral_tsad_v5" / "baselines" / "gboc",
    ]
    for tree in trees:
        paths.extend(
            sorted(
                path
                for path in tree.rglob("*")
                if path.is_file()
                and ".git" not in path.parts
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
        )

    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    total = 0
    unique_paths = sorted(set(paths), key=lambda path: str(path.relative_to(root)).replace("\\", "/"))
    data_set = set(data_paths)
    data_records: list[dict[str, Any]] = []
    for path in unique_paths:
        relative = str(path.relative_to(root)).replace("\\", "/")
        size = path.stat().st_size
        digest = file_sha256(path)
        aggregate.update(relative.encode() + b"\0" + str(size).encode() + b"\0" + digest.encode() + b"\n")
        total += size
        record = {"path": relative, "size": size, "sha256": digest}
        records.append(record)
        if path in data_set:
            data_records.append(record)
    digest = {
        "tree_sha256": aggregate.hexdigest(),
        "file_count": len(records),
        "total_bytes": total,
        "selected_data_files": len(set(data_paths)),
    }
    return digest, data_records


def build_plan(root: Path) -> dict[str, Any]:
    digest, data_records = snapshot(root)
    formal = [record for record in data_records if not record["path"].endswith("/" + PILOT_FILE)]
    if digest != EXPECTED_TREE_DIGEST or len(formal) != 43:
        raise RuntimeError(f"unexpected frozen input snapshot: digest={digest}, formal_files={len(formal)}")
    payload: dict[str, Any] = {
        "protocol": "duoba-10subset-seed2027-v2-server-b-split1",
        "seed": 2027,
        "input_digest": digest,
        "datasets": ["SVDB", "TAO", "CATSv2"],
        "excluded_files": [PILOT_FILE],
        "formal_files": formal,
    }
    payload["plan_sha256"] = plan_sha256(payload)
    return payload


def verify(root: Path, plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    actual_plan_sha = plan_sha256(plan)
    if plan.get("plan_sha256") != actual_plan_sha:
        raise RuntimeError("split plan self-digest mismatch")
    if EXPECTED_PLAN_SHA256 != "TO_BE_FROZEN" and actual_plan_sha != EXPECTED_PLAN_SHA256:
        raise RuntimeError("split plan does not match the frozen plan digest")
    actual_digest, data_records = snapshot(root)
    actual_formal = [record for record in data_records if not record["path"].endswith("/" + PILOT_FILE)]
    if actual_digest != EXPECTED_TREE_DIGEST or plan.get("input_digest") != EXPECTED_TREE_DIGEST:
        raise RuntimeError("frozen input tree digest mismatch")
    if actual_formal != plan.get("formal_files"):
        raise RuntimeError("formal data file inventory mismatch")
    return {
        "status": "verified",
        "plan_sha256": actual_plan_sha,
        "input_digest": actual_digest,
        "formal_files": len(actual_formal),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--write-plan", type=Path)
    parser.add_argument("--verify-plan", type=Path)
    args = parser.parse_args()
    if bool(args.write_plan) == bool(args.verify_plan):
        raise RuntimeError("select exactly one of --write-plan or --verify-plan")
    if args.write_plan:
        payload = build_plan(args.root.resolve())
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        payload = verify(args.root.resolve(), args.verify_plan.resolve())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
