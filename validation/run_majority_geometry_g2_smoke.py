#!/usr/bin/env python3
"""Run one shortest frozen G2 unit and validate all shared-geometry outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stage_vuspr_search as search  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "stage_majority_geometry_g2.json",
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    repo = arguments.repo.resolve()
    protocol = arguments.protocol.resolve()
    result_root = arguments.result_root.resolve()
    metrics_root = arguments.metrics_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)

    plan = search.ensure_plan(repo, protocol, metrics_root, result_root)
    data_preflight = search.ensure_data_preflight(repo, result_root, plan)
    search.ensure_runtime_preflight(result_root, plan, data_preflight)
    tasks = sorted(
        search._iter_tasks(plan),
        key=lambda item: search._task_priority(plan, repo, item),
    )
    if not tasks:
        raise RuntimeError("the frozen G2 plan contains no tasks")
    track, dataset, file_name, seed, execution_signature = tasks[-1]
    group = search.execution_group(
        plan,
        track,
        dataset,
        file_name,
        execution_signature,
    )
    target = search.unit_path(
        result_root,
        track,
        dataset,
        execution_signature,
        seed,
        file_name,
    )
    if not target.is_file():
        command = [
            str(arguments.python.resolve()),
            str((repo / "scripts" / "stage_vuspr_search.py").resolve()),
            "--repo",
            str(repo),
            "--protocol",
            str(protocol),
            "--result-root",
            str(result_root),
            "--metrics-root",
            str(metrics_root),
            "_worker",
            "--track",
            track,
            "--dataset",
            dataset,
            "--file",
            file_name,
            "--seed",
            str(seed),
            "--execution-signature",
            execution_signature,
            "--physical-gpu",
            str(arguments.gpu),
        ]
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(arguments.gpu),
            CUBLAS_WORKSPACE_CONFIG=search.CUBLAS_WORKSPACE_CONFIG,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout[-12_000:])

    item = search.load_json(target)
    if not search.valid_unit(
        item,
        plan=plan,
        track=track,
        dataset=dataset,
        file_name=file_name,
        seed=seed,
        group=group,
    ):
        raise RuntimeError(f"smoke unit is invalid: {target}")
    expected_variants = len(group["candidate_ids"]) * int(
        plan["variants_per_unit"]
    )
    geometry_visuals = item["diagnostics"].get(
        "visual_diagnostics_by_geometry"
    )
    if (
        len(item["variants"]) != expected_variants
        or not isinstance(geometry_visuals, dict)
        or set(geometry_visuals) != set(group["candidate_ids"])
    ):
        raise RuntimeError("smoke output lacks complete geometry/head diagnostics")
    print(
        json.dumps(
            {
                "status": "passed",
                "unit": str(target),
                "subset": f"{track}/{dataset}",
                "file": file_name,
                "physical_encoder_trainings": 1,
                "logical_geometry_units": len(group["candidate_ids"]),
                "head_records": len(item["variants"]),
                "visual_diagnostics": len(geometry_visuals),
                "plan_fingerprint": plan["plan_fingerprint"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
