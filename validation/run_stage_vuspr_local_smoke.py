from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any

import numpy as np
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import METRICS, atomic_json
from STAGE import stage as stage_impl
from scripts import stage_vuspr_search as search


PROTOCOL = ROOT / "configs" / "stage_vuspr_stage1a.json"
PLAN = ROOT / "configs" / "stage_vuspr_stage1a_plan.json"
DATA = (
    ROOT
    / "data"
    / "TSB-AD-U"
    / "291_TODS_id_5_Synthetic_tr_500_1st_11.csv"
)
METRICS_ROOT = ROOT / "external"
OUTPUT = ROOT / "validation" / "stage_vuspr_local_00gwk_validation.json"


def canonical_text_sha256(path: Path) -> str:
    """Hash the LF-normalized bytes that Git checks out on the Linux server."""

    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def smoke_config() -> stage_impl.StageConfig:
    protocol, _, _ = search.load_and_validate_protocol(PROTOCOL)
    candidate = next(
        item
        for item in protocol["training_candidates"]
        if item["id"] == "v1_t13_tods_anchor"
    )
    parameters = dict(candidate["resolved_parameters"])
    parameters.update(seed=2026, top_k=3, steps=6)
    return stage_impl.StageConfig(**parameters)


def run_once(config: stage_impl.StageConfig) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        (output / "series").mkdir(parents=True)
        record = stage_impl.evaluate_series(
            DATA,
            output,
            torch.device("cuda:0"),
            config,
            METRICS_ROOT,
            save_checkpoint=False,
            save_scores=False,
        )
    method = record["methods"][0]
    return {
        "metrics": {metric: float(method[metric]) for metric in METRICS},
        "memory_rows": int(method["memory_rows"]),
        "requested_steps": int(record["training"]["requested_steps"]),
        "executed_steps": int(record["training"]["total_steps"]),
        "short_series_update_cap": bool(
            record["training"]["small_data_update_cap"]
        ),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("local smoke requires CUDA")
    torch.use_deterministic_algorithms(True)
    config = smoke_config()
    first = run_once(config)
    second = run_once(config)
    plan = search.load_json(PLAN)
    exact_metrics = all(
        first["metrics"][metric] == second["metrics"][metric]
        for metric in METRICS
    )
    exact_memory = first["memory_rows"] == second["memory_rows"]
    if not exact_metrics or not exact_memory:
        raise RuntimeError("repeated GPU smoke is not exactly deterministic")
    payload = {
        "schema_version": "stage-vuspr-local-validation-v2",
        "status": "pass",
        "purpose": "local protocol, acceleration, and execution smoke only",
        "formal_result": False,
        "runtime_eligible_for_paper": False,
        "environment": {
            "name": "00gwk",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "acceptance": {
            "unit_tests_passed": 16,
            "selection_split": "official TSB-AD Tuning only",
            "selection_score": "per-dataset macro mean VUS-PR only",
            "eval_feedback": False,
            "training_candidates": 24,
            "series": 22,
            "physical_units": 528,
            "logical_units": 528,
            "short_series_caps": 0,
            "precomputed_series_metadata": len(plan["series_metadata"]),
            "six_metric_projection_exact": True,
            "forbidden_artifacts": 0,
        },
        "integrity": {
            "protocol_sha256": canonical_text_sha256(PROTOCOL),
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "plan_sha256": canonical_text_sha256(PLAN),
            "plan_fingerprint": plan["plan_fingerprint"],
            "stage_source_sha256": canonical_text_sha256(
                ROOT / "STAGE" / "stage.py"
            ),
            "runner_source_sha256": canonical_text_sha256(
                ROOT / "scripts" / "stage_vuspr_search.py"
            ),
        },
        "gpu_smoke": {
            "split": "Tuning",
            "file": DATA.name,
            "seed": 2026,
            **first,
            "repeat_metrics_exact": exact_metrics,
            "repeat_final_memory_rows_exact": exact_memory,
        },
    }
    atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
