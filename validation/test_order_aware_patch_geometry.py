from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from STAGE import stage  # noqa: E402


def small_config(**overrides) -> stage.StageConfig:
    values = {
        "patch_size": 48,
        "channels": 32,
        "token_dim": 16,
        "embedding_dim": 16,
        "dilations": (1, 2, 4),
        "group_norm_groups": 8,
        "dropout": 0.0,
        "batch_size": 8,
        "steps": 4,
        "overlap_deltas": (12, 24),
        "overlap_trim": 4,
    }
    values.update(overrides)
    return stage.StageConfig(**values)


class OrderAwarePatchGeometryTests(unittest.TestCase):
    def test_moments_control_matches_legacy_formula(self) -> None:
        torch.manual_seed(3)
        model = stage.StageEncoder(2, small_config(patch_geometry_mode="moments"))
        tokens = torch.randn(5, 31, 16)
        unit = F.normalize(tokens, dim=-1)
        mean = unit.mean(dim=1)
        std = torch.sqrt(unit.var(dim=1, unbiased=False).clamp_min(1e-8))
        expected = model.embedding_head(torch.cat([mean, std], dim=1))
        actual = model.embed_tokens(tokens)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_moments_control_is_permutation_invariant(self) -> None:
        torch.manual_seed(5)
        model = stage.StageEncoder(2, small_config(patch_geometry_mode="moments"))
        tokens = torch.randn(4, 32, 16)
        permutation = torch.randperm(tokens.shape[1])
        original = model.embed_tokens(tokens)
        shuffled = model.embed_tokens(tokens[:, permutation])
        torch.testing.assert_close(original, shuffled, rtol=1e-5, atol=1e-6)

    def test_ordered_pyramid_detects_segment_reversal(self) -> None:
        torch.manual_seed(7)
        model = stage.StageEncoder(
            2,
            small_config(patch_geometry_mode="ordered_pyramid"),
        )
        tokens = torch.randn(3, 32, 16)
        original = model.order_features(tokens)
        reversed_features = model.order_features(tokens.flip(1))
        self.assertGreater(float((original - reversed_features).abs().mean()), 0.02)

    def test_deterministic_temporal_bins_match_adaptive_pool_forward(self) -> None:
        torch.manual_seed(8)
        for length in (16, 31, 32, 40, 56, 64, 96):
            tokens = torch.randn(3, length, 7, requires_grad=True)
            actual = stage.StageEncoder._deterministic_temporal_bins(tokens, 4)
            expected = F.adaptive_avg_pool1d(
                tokens.transpose(1, 2),
                output_size=4,
            ).transpose(1, 2)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
            actual.square().mean().backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_signed_transition_features_flip_under_reversal(self) -> None:
        angles = torch.linspace(-1.1, 1.2, 40)
        tokens = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        tokens = tokens.unsqueeze(0)
        model = stage.StageEncoder(
            1,
            stage.StageConfig(
                patch_size=40,
                channels=8,
                token_dim=2,
                embedding_dim=4,
                dilations=(1, 2),
                group_norm_groups=4,
                overlap_deltas=(10, 20),
                overlap_trim=2,
                patch_geometry_mode="order_relations",
            ),
        )
        original = model.order_features(tokens)
        reversed_features = model.order_features(tokens.flip(1))
        directional_dimensions = 3 * 2
        torch.testing.assert_close(
            original[:, :directional_dimensions],
            -reversed_features[:, :directional_dimensions],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_hybrid_embedding_has_finite_gradients(self) -> None:
        torch.manual_seed(11)
        config = small_config(
            patch_geometry_mode="hybrid",
            transition_alignment_weight=0.1,
        )
        model = stage.StageEncoder(3, config)
        left = torch.randn(8, 3, 48)
        right = torch.randn(8, 3, 48)
        left_tokens = model.encode_tokens(left)
        right_tokens = model.encode_tokens(right)
        delta = 12
        trim = 4
        left_overlap = left_tokens[:, delta + trim : 48 - trim]
        right_overlap = right_tokens[:, trim : 48 - delta - trim]
        loss, diagnostics = stage.temporal_overlap_loss(
            left_tokens,
            right_tokens,
            model.embed_tokens(left_overlap),
            model.embed_tokens(right_overlap),
            delta,
            config,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("transition_consistency", diagnostics)
        self.assertIn("transition_gate", diagnostics)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_transition_gate_does_not_force_opposing_changes(self) -> None:
        torch.manual_seed(13)
        left = torch.randn(4, 24, 8).cumsum(dim=1)
        matching_loss, matching_gate = stage.majority_local_transition_loss(
            left,
            left.clone(),
        )
        opposing_loss, opposing_gate = stage.majority_local_transition_loss(
            left,
            left.flip(1),
        )
        self.assertLess(float(matching_loss), 1e-6)
        self.assertGreater(float(matching_gate), 0.99)
        self.assertLess(float(opposing_gate), 0.15)
        self.assertTrue(torch.isfinite(opposing_loss))

    def test_hybrid_parameter_overhead_stays_below_fifteen_percent(self) -> None:
        base = stage.StageEncoder(
            19,
            stage.StageConfig(patch_geometry_mode="moments"),
        )
        hybrid = stage.StageEncoder(
            19,
            stage.StageConfig(patch_geometry_mode="hybrid"),
        )
        base_count = sum(parameter.numel() for parameter in base.parameters())
        hybrid_count = sum(parameter.numel() for parameter in hybrid.parameters())
        self.assertGreater(hybrid_count, base_count)
        self.assertLess(hybrid_count, 1.15 * base_count)

    def test_invalid_geometry_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            small_config(patch_geometry_mode="attention").validate()
        with self.assertRaises(ValueError):
            small_config(order_residual_gate_init=0.0).validate()
        with self.assertRaises(ValueError):
            small_config(transition_alignment_weight=-0.1).validate()

    def test_hybrid_cpu_training_smoke_is_finite_and_order_sensitive(self) -> None:
        rng = np.random.default_rng(2026)
        time = np.linspace(0.0, 8.0 * np.pi, 112, dtype=np.float32)
        values = np.stack(
            [
                np.sin(time) + 0.05 * rng.standard_normal(len(time)),
                np.cos(0.5 * time) + 0.05 * rng.standard_normal(len(time)),
            ],
            axis=1,
        ).astype(np.float32)
        config = stage.StageConfig(
            patch_size=24,
            channels=16,
            token_dim=8,
            embedding_dim=8,
            dilations=(1, 2),
            group_norm_groups=4,
            dropout=0.0,
            batch_size=4,
            steps=3,
            gb_activation_fraction=0.67,
            overlap_deltas=(6, 12),
            overlap_trim=2,
            gb_min_split=4,
            gb_max_rounds=64,
            max_gb_rows=128,
            embedding_batch_size=64,
            patch_geometry_mode="hybrid",
            transition_alignment_weight=0.1,
            seed=2026,
        )
        device = torch.device("cpu")
        stage.seed_everything(config.seed)
        model = stage.StageEncoder(2, config).to(device)
        training = stage.fit_encoder(values, model, device, config)
        starts = np.asarray([0, 12, 24, 36], dtype=np.int64)
        embedding = stage.extract_embeddings(
            model,
            values,
            starts,
            device,
            config,
            normalize=False,
        )
        reversed_values = values.copy()
        for start in starts:
            reversed_values[
                start : start + config.patch_size
            ] = reversed_values[start : start + config.patch_size][::-1]
        reversed_embedding = stage.extract_embeddings(
            model,
            reversed_values,
            starts,
            device,
            config,
            normalize=False,
        )
        self.assertEqual(training["total_steps"], 3)
        self.assertTrue(np.isfinite(embedding).all())
        self.assertGreater(
            float(np.mean(np.abs(embedding - reversed_embedding))),
            1e-4,
        )


if __name__ == "__main__":
    unittest.main()
