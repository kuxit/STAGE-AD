#!/usr/bin/env python3
"""Strict local audit for the STAGE VUS-PR-only Stage1A plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage_vuspr_search as search


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "stage_vuspr_stage1a.json",
    )
    parser.add_argument("--metrics-root", type=Path, default=ROOT / "external")
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    metrics_root = args.metrics_root.resolve()
    protocol, protocol_sha256, protocol_fingerprint = (
        search.load_and_validate_protocol(protocol_path)
    )
    plan = search._plan_stable_payload(repo, protocol_path, metrics_root)

    if protocol["selection"]["score"] != "macro mean VUS-PR only":
        raise RuntimeError("selection is not VUS-PR only")
    if protocol["selection"]["tie_breakers"] != ["canonical_id"]:
        raise RuntimeError("tie-breaking uses a metric other than VUS-PR")
    if protocol["selection_split"] != search.SELECTION_SPLIT:
        raise RuntimeError("selection split is not official TSB-AD Tuning")
    if protocol["eval_feedback_used_by_runner"] is not False:
        raise RuntimeError("Eval feedback is enabled")
    if plan["series_count"] != 22:
        raise RuntimeError("the frozen representative Tuning set must have 22 series")
    expected = 24 * 22
    if plan["expected_logical_units"] != expected:
        raise RuntimeError("logical unit count differs from 24 x 22")
    if plan["expected_units"] != expected:
        raise RuntimeError("an execution signature was unexpectedly deduplicated")

    behaviors = [
        behavior
        for by_signature in plan["execution_behaviors"].values()
        for behavior in by_signature.values()
    ]
    if len(behaviors) != expected:
        raise RuntimeError("execution behavior count differs from the frozen plan")
    if any(bool(item["short_series_update_cap"]) for item in behaviors):
        raise RuntimeError("a short-series update cap is still active")
    if min(int(item["effective_steps"]) for item in behaviors) != 750:
        raise RuntimeError("minimum declared update budget changed")
    if max(int(item["effective_steps"]) for item in behaviors) != 2000:
        raise RuntimeError("maximum declared update budget changed")

    serialized = json.dumps(plan, sort_keys=True)
    if "-Eval.csv" in serialized or "Eval only" in serialized:
        raise RuntimeError("Eval material leaked into the Tuning plan")

    report = {
        "status": "pass",
        "protocol_sha256": protocol_sha256,
        "protocol_fingerprint": protocol_fingerprint,
        "plan_fingerprint": search.fingerprint(plan),
        "training_candidates": len(protocol["training_candidates"]),
        "series": plan["series_count"],
        "physical_units": plan["expected_units"],
        "logical_units": plan["expected_logical_units"],
        "short_series_caps": 0,
        "selection_score": protocol["selection"]["score"],
        "tie_breakers": protocol["selection"]["tie_breakers"],
        "eval_feedback": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
