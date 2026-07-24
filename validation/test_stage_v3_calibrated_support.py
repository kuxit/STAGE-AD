from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from STAGE import stage as stage_impl


class StageV3CalibratedSupportTests(unittest.TestCase):
    def test_robust_feature_calibration_is_finite_and_resists_one_outlier(self) -> None:
        values = np.asarray(
            [
                [0.0, 10.0],
                [1.0, 10.0],
                [2.0, 10.0],
                [3.0, 10.0],
                [1000.0, 10.0],
            ],
            dtype=np.float32,
        )
        center, scale = stage_impl.robust_feature_calibration(values)
        np.testing.assert_allclose(center, np.asarray([2.0, 10.0], dtype=np.float32))
        self.assertTrue(np.isfinite(scale).all())
        self.assertTrue(np.all(scale > 0.0))
        self.assertLess(float(scale[0]), 10.0)

    def test_statistics_branch_preserves_level_cues_removed_by_patch_revin(self) -> None:
        config = replace(
            stage_impl.StageConfig(),
            patch_size=8,
            channels=8,
            token_dim=4,
            embedding_dim=4,
            dilations=(1,),
            group_norm_groups=2,
            dropout=0.0,
            overlap_deltas=(2,),
            overlap_trim=1,
            batch_size=4,
            patch_statistics_weight=1.0,
        )
        model = stage_impl.StageEncoder(
            1,
            config,
            feature_center=np.asarray([0.0], dtype=np.float32),
            feature_scale=np.asarray([1.0], dtype=np.float32),
        ).eval()
        base = torch.linspace(-1.0, 1.0, 8).reshape(1, 1, 8)
        shifted = base + 5.0
        with torch.inference_mode():
            base_tokens = model.encode_tokens(base)
            shifted_tokens = model.encode_tokens(shifted)
            _, base_embedding = model(base)
            _, shifted_embedding = model(shifted)
        torch.testing.assert_close(base_tokens, shifted_tokens, rtol=1e-5, atol=1e-5)
        self.assertGreater(
            float(torch.linalg.vector_norm(base_embedding - shifted_embedding)),
            0.05,
        )

    def test_patch_statistics_calibration_centers_training_windows(self) -> None:
        values = np.arange(20, dtype=np.float32).reshape(-1, 1)
        feature_center, feature_scale = stage_impl.robust_feature_calibration(values)
        statistics_center, statistics_scale = (
            stage_impl.robust_patch_statistics_calibration(
                values,
                5,
                feature_center,
                feature_scale,
            )
        )
        self.assertEqual(statistics_center.shape, (2,))
        self.assertEqual(statistics_scale.shape, (2,))
        self.assertTrue(np.isfinite(statistics_center).all())
        self.assertTrue(np.isfinite(statistics_scale).all())
        self.assertTrue(np.all(statistics_scale > 0.0))

    def test_zero_statistics_weight_preserves_legacy_embedding_dimension(self) -> None:
        config = replace(
            stage_impl.StageConfig(),
            patch_size=8,
            channels=8,
            token_dim=4,
            embedding_dim=6,
            dilations=(1,),
            group_norm_groups=2,
            dropout=0.0,
            overlap_deltas=(2,),
            overlap_trim=1,
            batch_size=4,
            patch_statistics_weight=0.0,
        )
        model = stage_impl.StageEncoder(2, config).eval()
        with torch.inference_mode():
            _, embedding = model(torch.randn(3, 2, 8))
        self.assertEqual(tuple(embedding.shape), (3, 6))

    def test_region_radius_penalty_calibrates_nearest_reference_distance(self) -> None:
        memory = stage_impl.CalibratedExemplarMemory(
            vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            region_radii=np.asarray([0.5, 0.0], dtype=np.float32),
            region_sizes=np.asarray([20, 4], dtype=np.int64),
            representative_source_rows=np.asarray([0, 1], dtype=np.int64),
        )
        query = np.asarray([[1.0, 0.0]], dtype=np.float32)
        base = replace(
            stage_impl.StageConfig(),
            top_k=1,
            score_batch_size=1,
            memory_score_block_size=1,
            memory_radius_weight=0.0,
        )
        calibrated = replace(base, memory_radius_weight=1.0)
        legacy_score = stage_impl.score_embeddings(
            query,
            memory,
            torch.device("cpu"),
            base,
        )
        calibrated_score = stage_impl.score_embeddings(
            query,
            memory,
            torch.device("cpu"),
            calibrated,
        )
        np.testing.assert_allclose(legacy_score, np.asarray([0.0]), atol=1e-7)
        np.testing.assert_allclose(calibrated_score, np.asarray([0.5]), atol=1e-7)

    def test_calibrated_memory_retains_observed_unit_exemplars_and_metadata(self) -> None:
        rng = np.random.default_rng(9)
        embeddings = rng.normal(size=(36, 5)).astype(np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        config = replace(
            stage_impl.StageConfig(),
            gb_min_split=8,
            gb_max_rounds=8,
            max_gb_rows=36,
            memory_radius_quantile=0.75,
        )
        memory, partition = stage_impl.build_calibrated_exemplar_memory(
            embeddings,
            config,
            seed=3,
        )
        self.assertEqual(len(memory), partition.final_k)
        self.assertEqual(memory.region_radii.shape, (len(memory),))
        self.assertEqual(memory.region_sizes.shape, (len(memory),))
        self.assertTrue(np.all(memory.region_radii >= 0.0))
        np.testing.assert_allclose(
            np.linalg.norm(memory.vectors, axis=1),
            np.ones(len(memory)),
            rtol=1e-5,
            atol=1e-6,
        )
        for vector in memory.vectors:
            self.assertLess(
                float(np.min(np.linalg.norm(embeddings - vector[None, :], axis=1))),
                1e-5,
            )


if __name__ == "__main__":
    unittest.main()
