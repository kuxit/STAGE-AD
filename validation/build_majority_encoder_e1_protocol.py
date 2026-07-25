from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PROTOCOL = ROOT / "configs" / "stage_vuspr_stage2.json"
SOURCE_SELECTION = Path(
    r"F:\python_project\AAAI\我们的论文\experiments"
    r"\stage_vuspr_eval_seed2026\stage2_tuning_selection.json"
)
OUTPUT = ROOT / "configs" / "stage_majority_encoder_e1.json"
ENCODERS = (
    "dilated_residual",
    "depthwise_tcn",
    "multiscale_depthwise_tcn",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    source_selection = json.loads(SOURCE_SELECTION.read_text(encoding="utf-8"))
    lookup = {
        str(item["id"]): dict(item["parameters"])
        for item in source["training_candidates"]
    }
    candidates: list[dict[str, object]] = []
    shortlists: dict[str, list[str]] = {}
    for subset, winner in source["training_winners"].items():
        dataset = subset.split("/", 1)[1].lower()
        candidate_ids: list[str] = []
        for encoder_type in ENCODERS:
            suffix = {
                "dilated_residual": "res",
                "depthwise_tcn": "dw",
                "multiscale_depthwise_tcn": "mdw",
            }[encoder_type]
            candidate_id = f"{dataset}_{suffix}"
            parameters = dict(lookup[str(winner)])
            parameters["encoder_type"] = encoder_type
            candidates.append(
                {"id": candidate_id, "parameters": parameters}
            )
            candidate_ids.append(candidate_id)
        shortlists[str(subset)] = candidate_ids
    protocol = {
        "schema_version": source["schema_version"],
        "phase": "encoder_e1",
        "study_id": "stage-majority-local-light-encoder-e1-20260725",
        "status": "posthoc_development_only",
        "selection_split": source["selection_split"],
        "eval_feedback_used_by_runner": False,
        "confirmatory_claim_eligible": False,
        "story_contract": source["story_contract"],
        "expected_source_sha256": sha256_file(ROOT / "STAGE" / "stage.py"),
        "seeds": [2026],
        "targets": source["targets"],
        "metrics": source["metrics"],
        "selection": {
            **source["selection"],
            "stage1_rule": (
                "With each dataset's prior Tuning-only training parameters and "
                "head fixed, compare the three declared timestamp-preserving "
                "encoder backbones by complete official Tuning macro VUS-PR."
            ),
            "stage2_rule": (
                "Only encoder candidates that pass the seed-2026 gate may be "
                "confirmed on additional seeds; no Eval feedback is permitted."
            ),
        },
        "final_gb_min_splits": source["final_gb_min_splits"],
        "top_ks": source["top_ks"],
        "training_candidates": candidates,
        "shortlist_size": 3,
        "training_shortlists": shortlists,
        "metadata": {
            "base_model": "STAGE_majority_local",
            "majority_reference_sha256": (
                "704916804f8f37386e480e9167c8b04ed1e3ec27166804865b27d0b338f9fb79"
            ),
            "frozen_axes": [
                "majority_local_alignment",
                "intermediate_granular_sampling",
                "final_observed_exemplar_memory",
                "dataset_training_parameters",
                "stage1_fixed_head",
            ],
            "encoder_candidates": list(ENCODERS),
            "selection_metric": "official Tuning macro VUS-PR only",
            "reference_selected_heads": {
                subset: {
                    "final_gb_min_split": int(head["final_gb_min_split"]),
                    "top_k": int(head["top_k"]),
                }
                for subset, head in source_selection["selected_heads"].items()
            },
            "global_encoder_rule": (
                "Select one encoder architecture for all ten datasets by the "
                "unweighted mean of dataset macro VUS-PR at each dataset's "
                "previously frozen head; canonical encoder_type breaks exact ties."
            ),
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
