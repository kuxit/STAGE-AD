from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "stage_majority_encoder_e1.json"
OUTPUT = ROOT / "configs" / "stage_majority_geometry_d1.json"
TARGETS = {
    "U": ["MSL", "SED"],
    "M": ["CATSv2", "GHL"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_candidates = {
        str(item["id"]): dict(item["parameters"])
        for item in source["training_candidates"]
    }
    candidates: list[dict[str, object]] = []
    shortlists: dict[str, list[str]] = {}
    for track, datasets in TARGETS.items():
        for dataset in datasets:
            subset = f"{track}/{dataset}"
            prefix = dataset.lower()
            base = dict(source_candidates[f"{prefix}_res"])
            variants = {
                "d0_control": {},
                "d1_stats": {"patch_statistics_weight": 0.5},
                "d2_pairwise": {"timestamp_pairwise_weight": 0.25},
                "d3_geometry": {
                    "gb_min_split": 32,
                    "gb_refresh_fraction": 0.65,
                },
            }
            ids: list[str] = []
            for suffix, updates in variants.items():
                candidate_id = f"{prefix}_{suffix}"
                parameters = {**base, **updates}
                candidates.append(
                    {"id": candidate_id, "parameters": parameters}
                )
                ids.append(candidate_id)
            shortlists[subset] = ids

    protocol = {
        "schema_version": source["schema_version"],
        "phase": "geometry_d1",
        "study_id": "stage-majority-local-geometry-diagnosis-d1-20260725",
        "status": "posthoc_development_only",
        "selection_split": "official TSB-AD Tuning only",
        "eval_feedback_used_by_runner": False,
        "confirmatory_claim_eligible": False,
        "story_contract": source["story_contract"],
        "expected_source_sha256": sha256_file(ROOT / "STAGE" / "stage.py"),
        "seeds": [2026],
        "targets": TARGETS,
        "metrics": source["metrics"],
        "selection": {
            "score": "macro mean VUS-PR only",
            "tie_breakers": ["canonical_id"],
            "stage1_head": {
                "final_gb_min_split": 4,
                "top_k": 3,
            },
            "stage1_rule": (
                "Use complete and identical official Tuning coverage to "
                "diagnose the control, robust statistics, direct timestamp "
                "pairwise alignment, and refreshed coarser geometry axes."
            ),
            "stage2_rule": (
                "This diagnostic does not authorize Eval or parameter lock; "
                "any follow-up mechanism must be frozen in a new Tuning-only protocol."
            ),
            "dataset_specific_rule": (
                "All comparisons are dataset-specific; track fallback is forbidden."
            ),
        },
        "final_gb_min_splits": [4, 64],
        "top_ks": [1, 3],
        "training_candidates": candidates,
        "shortlist_size": 4,
        "training_shortlists": shortlists,
        "metadata": {
            "base_model": "STAGE_majority_local",
            "frozen_encoder_type": "dilated_residual",
            "diagnostic_projection": True,
            "diagnostic_final_gb_min_split": 4,
            "diagnostic_top_k": 1,
            "diagnostic_datasets": [
                "U/MSL",
                "U/SED",
                "M/CATSv2",
                "M/GHL",
            ],
            "diagnostic_axes": {
                "d0_control": "unchanged majority-local baseline",
                "d1_stats": "retain robust patch level and scale alongside normalized shape",
                "d2_pairwise": "direct timestamp and shared-interval cosine consistency",
                "d3_geometry": "coarser intermediate regions rebuilt once during training",
            },
            "visual_evidence": [
                "raw normal and anomalous time-series intervals",
                "training-normal, Eval-normal, Eval-anomaly embedding projection",
                "selected observed prototypes with support, radius, and source time",
                "normal-versus-anomaly score separation",
            ],
            "visuals_are_posthoc": True,
            "selection_uses_visuals": False,
            "eval_feedback": False,
        },
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "candidates": len(candidates),
                "shortlists": len(shortlists),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
