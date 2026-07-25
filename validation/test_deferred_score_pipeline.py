from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common import METRICS, atomic_json
from scripts import stage_vuspr_search as search


class DeferredScorePipelineTests(unittest.TestCase):
    def _plan(self) -> dict:
        signature = "c" * 64
        candidate_id = "msl_o0_moments_control"
        return {
            "phase": "order_o1",
            "plan_fingerprint": "1" * 64,
            "protocol_fingerprint": "2" * 64,
            "source_sha256": {
                "scripts/stage_vuspr_search.py": "3" * 64,
                "STAGE/stage.py": "4" * 64,
                "common.py": "5" * 64,
            },
            "data_sha256": {"U/MSL/example.csv": "6" * 64},
            "files": {"U": {"MSL": ["example.csv"]}},
            "seeds": [2026],
            "final_gb_min_splits": [4],
            "top_ks": [3],
            "metadata": {"diagnostic_projection": False},
            "candidates_by_subset": {"U/MSL": [candidate_id]},
            "execution_signatures": {
                "U/MSL/example.csv": {candidate_id: signature}
            },
            "execution_behaviors": {
                "U/MSL/example.csv": {signature: {"patch_size": 16}}
            },
            "training_candidates": [
                {
                    "id": candidate_id,
                    "config_fingerprint": "7" * 64,
                    "resolved_parameters": {},
                }
            ],
        }

    def test_cpu_evaluator_consumes_and_removes_verified_cache(self) -> None:
        plan = self._plan()
        signature = "c" * 64
        candidate_id = "msl_o0_moments_control"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, array_path = search.score_cache_paths(
                root, "U", "MSL", signature, 2026, "example.csv"
            )
            labels = np.asarray([0, 0, 1, 1, 0, 0], dtype=np.int8)
            scores = np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.1])
            search._atomic_npz(
                array_path,
                {"labels": labels, "score_000": scores},
            )
            unit_base = {
                "schema_version": search.UNIT_SCHEMA,
                "kind": search.UNIT_KIND,
                "method": "STAGE",
                "selection_split": "Tuning",
                "eval_feedback": False,
                "track": "U",
                "dataset": "MSL",
                "file": "example.csv",
                "seed": 2026,
                "execution_signature": signature,
                "execution_behavior": {"patch_size": 16},
                "canonical_training_candidate_id": candidate_id,
                "training_candidate_ids": [candidate_id],
                "training_config_fingerprints": {
                    candidate_id: "7" * 64
                },
                "plan_fingerprint": plan["plan_fingerprint"],
                "protocol_fingerprint": plan["protocol_fingerprint"],
                "source_sha256": plan["source_sha256"],
                "data_sha256": "6" * 64,
                "training_prefix_policy": "filename_declared_label_blind",
                "diagnostics": {
                    "runtime_eligible_for_paper": False,
                    "visual_diagnostics": None,
                    "visual_diagnostics_by_geometry": None,
                    "environment": {
                        "CUBLAS_WORKSPACE_CONFIG": search.CUBLAS_WORKSPACE_CONFIG,
                        "torch_deterministic_algorithms": True,
                    },
                },
                "error": None,
            }
            pending = {
                "geometry_candidate_id": candidate_id,
                "geometry_mode": "control",
                "final_gb_min_split": 4,
                "canonical_final_gb_min_split": 4,
                "final_partition_fingerprint": "8" * 64,
                "memory_fingerprint": "9" * 64,
                "top_k": 3,
                "effective_top_k": 3,
                "top_k_clamped": False,
                "canonical_top_k": 3,
                "memory_rows": 10,
                "memory_ratio": 0.1,
                "score_index": 0,
            }
            manifest = {
                "schema_version": search.SCORE_CACHE_SCHEMA,
                "plan_fingerprint": plan["plan_fingerprint"],
                "protocol_fingerprint": plan["protocol_fingerprint"],
                "track": "U",
                "dataset": "MSL",
                "file": "example.csv",
                "seed": 2026,
                "execution_signature": signature,
                "sliding_window": 4,
                "score_count": 1,
                "score_array_sha256": search.sha256_file(array_path),
                "pending_variants": [pending],
                "unit_base": unit_base,
                "created_at": search.utc_now(),
            }
            atomic_json(manifest_path, manifest)
            fixed_metrics = {metric: 0.5 for metric in METRICS}
            with patch.object(
                search,
                "_metrics_checked",
                return_value=fixed_metrics,
            ):
                target = search.evaluate_score_cache(
                    root,
                    root,
                    plan,
                    track="U",
                    dataset="MSL",
                    file_name="example.csv",
                    seed=2026,
                    execution_signature=signature,
                    remove_cache=True,
                )
            self.assertTrue(target.is_file())
            self.assertFalse(manifest_path.exists())
            self.assertFalse(array_path.exists())
            self.assertEqual(
                search.load_json(target)["variants"][0]["metrics"],
                fixed_metrics,
            )

    def test_order_o1_protocol_has_four_execution_distinct_candidates_per_dataset(
        self,
    ) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "stage_majority_order_o1.json"
        )
        protocol, _, _ = search.load_and_validate_protocol(protocol_path)
        self.assertEqual(protocol["phase"], "order_o1")
        self.assertEqual(protocol["targets"], {"U": ["MSL"], "M": ["GHL"]})
        lookup = {
            candidate["id"]: candidate
            for candidate in protocol["training_candidates"]
        }
        for subset, file_name in (
            ("U/MSL", "x_MSL_tr_530_case.csv"),
            ("M/GHL", "x_GHL_tr_400_case.csv"),
        ):
            signatures = {
                search.training_execution_signature(
                    lookup[candidate_id],
                    file_name,
                )[0]
                for candidate_id in protocol["training_shortlists"][subset]
            }
            self.assertEqual(len(signatures), 4)


if __name__ == "__main__":
    unittest.main()
