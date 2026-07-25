from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "configs" / "stage_majority_geometry_d1.json"
OUTPUT = ROOT / "configs" / "stage_majority_geometry_g2.json"
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
    modes = {
        "g0_control": {
            "final_geometry_mode": "control",
            "memory_radius_weight": 0.0,
            "temporal_transition_weight": 0.0,
        },
        "g1_independent_support": {
            "final_geometry_mode": "independent_support",
            "memory_radius_weight": 0.0,
            "temporal_support_fraction": 0.5,
            "temporal_transition_weight": 0.0,
        },
        "g2_support_radius": {
            "final_geometry_mode": "support_radius",
            "memory_radius_weight": 0.5,
            "temporal_support_fraction": 0.5,
            "temporal_transition_weight": 0.0,
        },
        "g3_support_radius_combined": {
            "final_geometry_mode": "support_radius_combined",
            "memory_radius_weight": 0.5,
            "temporal_support_fraction": 0.5,
            "temporal_transition_weight": 0.0,
        },
        "g4_support_radius_transition": {
            "final_geometry_mode": "support_radius_transition",
            "memory_radius_weight": 0.5,
            "temporal_support_fraction": 0.5,
            "temporal_transition_weight": 0.25,
        },
    }
    for track, datasets in TARGETS.items():
        for dataset in datasets:
            subset = f"{track}/{dataset}"
            prefix = dataset.lower()
            base = dict(source_candidates[f"{prefix}_d0_control"])
            ids: list[str] = []
            for suffix, updates in modes.items():
                candidate_id = f"{prefix}_{suffix}"
                candidates.append(
                    {
                        "id": candidate_id,
                        "parameters": {**base, **updates},
                    }
                )
                ids.append(candidate_id)
            shortlists[subset] = ids

    protocol = {
        "schema_version": source["schema_version"],
        "phase": "geometry_g2",
        "study_id": "stage-majority-local-support-geometry-g2-20260725",
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
                "Compare only complete official Tuning coverage after one "
                "shared encoder pass per series. Final geometry may use "
                "timestamp-distinct support, locally calibrated radii, and "
                "label-blind state-transition consistency."
            ),
            "stage2_rule": (
                "This diagnostic does not authorize Eval or parameter lock. "
                "A successful mechanism requires a separately frozen multi-seed "
                "Tuning protocol."
            ),
            "dataset_specific_rule": (
                "All comparisons are dataset-specific; track fallback is forbidden."
            ),
        },
        "final_gb_min_splits": [4, 64],
        "top_ks": [1, 3],
        "training_candidates": candidates,
        "shortlist_size": 5,
        "training_shortlists": shortlists,
        "metadata": {
            "base_model": "STAGE_majority_local",
            "frozen_encoder_type": "dilated_residual",
            "shared_encoder_per_series": True,
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
                "g0_control": "unchanged majority-local observed-exemplar memory",
                "g1_independent_support": (
                    "choose each observed representative after de-duplicating "
                    "overlapping timestamp support"
                ),
                "g2_support_radius": (
                    "penalize broad, strongly supported normal regions while "
                    "keeping the legacy representative"
                ),
                "g3_support_radius_combined": (
                    "combine timestamp-distinct representatives with "
                    "support-calibrated local radii"
                ),
                "g4_support_radius_transition": (
                    "add label-blind state-transition geometry to the combined memory"
                ),
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
                "physical_training_configs_per_dataset": 1,
                "geometry_candidates_per_dataset": len(modes),
                "candidates": len(candidates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
