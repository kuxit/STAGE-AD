from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from STAGE import stage  # noqa: E402
from scripts import stage_vuspr_search as search  # noqa: E402


class MajorityGeometrySupportTests(unittest.TestCase):
    def test_temporal_support_deduplicates_dense_windows(self) -> None:
        dense_starts = np.arange(0, 32, dtype=np.int64)
        sparse_starts = np.arange(0, 32 * 16, 16, dtype=np.int64)
        _, dense_support, dense_components = stage.temporal_independence_weights(
            dense_starts,
            patch_size=32,
            fraction=0.5,
        )
        _, sparse_support, sparse_components = stage.temporal_independence_weights(
            sparse_starts,
            patch_size=32,
            fraction=0.5,
        )
        self.assertLess(dense_support, sparse_support)
        self.assertEqual(dense_components, 1)
        self.assertEqual(sparse_components, 1)

    def test_support_memory_keeps_observed_exemplars(self) -> None:
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(64, 8)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        config = stage.StageConfig(
            patch_size=16,
            channels=8,
            token_dim=4,
            embedding_dim=8,
            dilations=(1, 2),
            group_norm_groups=4,
            gb_min_split=4,
            gb_max_rounds=8,
            max_gb_rows=64,
            temporal_support_fraction=0.5,
        )
        memory, _ = stage.build_support_calibrated_exemplar_memory(
            embeddings,
            config,
            seed=7,
            independent_representatives=True,
        )
        self.assertGreater(len(memory), 0)
        self.assertIsNotNone(memory.effective_support)
        self.assertIsNotNone(memory.temporal_components)
        np.testing.assert_allclose(
            memory.vectors,
            embeddings[memory.representative_source_rows],
            atol=1e-6,
        )

    def test_radius_penalty_respects_radius_and_support(self) -> None:
        memory = stage.CalibratedExemplarMemory(
            vectors=np.eye(2, dtype=np.float32),
            region_radii=np.asarray([0.10, 0.40], dtype=np.float32),
            region_sizes=np.asarray([5, 50], dtype=np.int64),
            representative_source_rows=np.asarray([0, 1], dtype=np.int64),
            effective_support=np.asarray([1.0, 4.0], dtype=np.float32),
        )
        penalty = stage.support_radius_penalty(memory, 0.5)
        self.assertGreater(float(penalty[1]), float(penalty[0]))
        np.testing.assert_allclose(penalty, [0.025, 0.2], atol=1e-7)

    def test_transition_geometry_separates_equal_states(self) -> None:
        state = np.asarray([[1.0, 0.0]], dtype=np.float32)
        positive = np.asarray([[0.0, 1.0]], dtype=np.float32)
        negative = np.asarray([[0.0, -1.0]], dtype=np.float32)
        reference = stage.augment_temporal_geometry(state, positive, 0.25)
        matching = stage.augment_temporal_geometry(state, positive, 0.25)
        opposing = stage.augment_temporal_geometry(state, negative, 0.25)
        match_distance = float(np.sum((reference - matching) ** 2))
        opposing_distance = float(np.sum((reference - opposing) ** 2))
        self.assertAlmostEqual(match_distance, 0.0, places=7)
        self.assertGreater(opposing_distance, 0.9)

    def test_geometry_only_candidates_share_training_signature(self) -> None:
        base = asdict(
            stage.StageConfig(
                patch_size=32,
                channels=16,
                token_dim=8,
                embedding_dim=8,
                dilations=(1, 2),
                group_norm_groups=8,
                overlap_deltas=(8, 16),
                steps=10,
            )
        )
        base.pop("seed")
        base.pop("top_k")
        candidate_a = {
            "id": "g0",
            "resolved_parameters": {
                **base,
                "final_geometry_mode": "control",
                "memory_radius_weight": 0.0,
                "temporal_transition_weight": 0.0,
            },
        }
        candidate_b = {
            "id": "g4",
            "resolved_parameters": {
                **base,
                "final_geometry_mode": "support_radius_transition",
                "memory_radius_weight": 0.5,
                "temporal_support_fraction": 0.5,
                "temporal_transition_weight": 0.25,
            },
        }
        signature_a, behavior_a = search.training_execution_signature(
            candidate_a,
            "Synthetic_tr_512_1024.csv",
        )
        signature_b, behavior_b = search.training_execution_signature(
            candidate_b,
            "Synthetic_tr_512_1024.csv",
        )
        self.assertEqual(signature_a, signature_b)
        self.assertEqual(behavior_a, behavior_b)

    def test_memory_penalty_is_added_before_nearest_neighbor(self) -> None:
        queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
        memory = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        config = stage.StageConfig(
            patch_size=8,
            channels=8,
            token_dim=4,
            embedding_dim=2,
            dilations=(1, 2),
            group_norm_groups=4,
        )
        score = search.score_embeddings_multi_k(
            queries,
            memory,
            torch.device("cpu"),
            config,
            [1],
            memory_penalty=np.asarray([0.3, 0.0], dtype=np.float32),
        )[1]
        self.assertAlmostEqual(float(score[0]), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
