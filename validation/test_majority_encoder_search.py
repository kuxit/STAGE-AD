from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from STAGE import stage as candidate  # noqa: E402
from scripts import stage_vuspr_search as search  # noqa: E402


def load_majority_reference():
    path = Path(r"F:\Download\STAGE_majority_local.py")
    spec = importlib.util.spec_from_file_location("majority_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load majority-local reference")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MajorityEncoderSearchTests(unittest.TestCase):
    @unittest.skipUnless(
        Path(r"F:\Download\STAGE_majority_local.py").is_file(),
        "local majority-local reference is unavailable",
    )
    def test_default_encoder_exactly_matches_majority_reference(self) -> None:
        reference = load_majority_reference()
        common = dict(
            patch_size=48,
            channels=32,
            token_dim=16,
            embedding_dim=16,
            dilations=(1, 2, 4),
            group_norm_groups=8,
            dropout=0.0,
            overlap_deltas=(12, 24),
            overlap_trim=4,
            seed=19,
        )
        torch.manual_seed(19)
        expected_model = reference.StageEncoder(
            3,
            reference.StageConfig(**common),
        )
        torch.manual_seed(19)
        actual_model = candidate.StageEncoder(
            3,
            candidate.StageConfig(**common),
        )
        expected_state = expected_model.state_dict()
        actual_state = actual_model.state_dict()
        self.assertEqual(tuple(expected_state), tuple(actual_state))
        for name in expected_state:
            torch.testing.assert_close(
                expected_state[name],
                actual_state[name],
                rtol=0.0,
                atol=0.0,
            )
        values = torch.randn(5, 3, 48)
        expected_tokens, expected_embedding = expected_model(values)
        actual_tokens, actual_embedding = actual_model(values)
        torch.testing.assert_close(
            expected_tokens,
            actual_tokens,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            expected_embedding,
            actual_embedding,
            rtol=0.0,
            atol=0.0,
        )

    def test_all_encoders_preserve_timestamp_and_embedding_shapes(self) -> None:
        values = torch.randn(4, 5, 64)
        for encoder_type in (
            "dilated_residual",
            "depthwise_tcn",
            "multiscale_depthwise_tcn",
        ):
            with self.subTest(encoder_type=encoder_type):
                config = candidate.StageConfig(
                    encoder_type=encoder_type,
                    patch_size=64,
                    channels=32,
                    token_dim=16,
                    embedding_dim=24,
                    dilations=(1, 2, 4),
                    group_norm_groups=8,
                    dropout=0.0,
                    overlap_deltas=(16, 32),
                    overlap_trim=4,
                )
                model = candidate.StageEncoder(5, config)
                tokens, embedding = model(values)
                self.assertEqual(tokens.shape, (4, 64, 16))
                self.assertEqual(embedding.shape, (4, 24))
                self.assertTrue(torch.isfinite(tokens).all())
                self.assertTrue(torch.isfinite(embedding).all())

    def test_depthwise_candidates_are_materially_lighter(self) -> None:
        counts: dict[str, int] = {}
        for encoder_type in (
            "dilated_residual",
            "depthwise_tcn",
            "multiscale_depthwise_tcn",
        ):
            config = candidate.StageConfig(encoder_type=encoder_type)
            model = candidate.StageEncoder(1, config)
            counts[encoder_type] = sum(
                parameter.numel() for parameter in model.parameters()
            )
            self.assertGreater(
                sum(parameter.numel() for parameter in model.token_head.parameters()),
                0,
            )
            self.assertGreater(
                sum(
                    parameter.numel()
                    for parameter in model.embedding_head.parameters()
                ),
                0,
            )
        self.assertLess(
            counts["depthwise_tcn"],
            0.5 * counts["dilated_residual"],
        )
        self.assertLess(
            counts["multiscale_depthwise_tcn"],
            0.5 * counts["dilated_residual"],
        )

    def test_depthwise_candidates_have_finite_gradients(self) -> None:
        for encoder_type in (
            "depthwise_tcn",
            "multiscale_depthwise_tcn",
        ):
            with self.subTest(encoder_type=encoder_type):
                torch.manual_seed(23)
                config = candidate.StageConfig(
                    encoder_type=encoder_type,
                    patch_size=48,
                    channels=32,
                    token_dim=16,
                    embedding_dim=16,
                    dilations=(1, 2, 4),
                    group_norm_groups=8,
                    dropout=0.0,
                    overlap_deltas=(12, 24),
                    overlap_trim=4,
                )
                model = candidate.StageEncoder(2, config)
                tokens, embedding = model(torch.randn(8, 2, 48))
                loss = tokens.square().mean() + embedding.square().mean()
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(
                    all(torch.isfinite(gradient).all() for gradient in gradients)
                )

    def test_invalid_encoder_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            candidate.StageConfig(encoder_type="transformer").validate()

    def test_encoder_gate_uses_frozen_heads_for_one_global_encoder(self) -> None:
        targets = {"U": ["UCR"], "M": ["CATSv2"]}
        candidates = []
        candidates_by_subset = {}
        rows = []
        scores = {
            "dilated_residual": (0.70, 0.40),
            "depthwise_tcn": (0.72, 0.42),
            "multiscale_depthwise_tcn": (0.75, 0.45),
        }
        for track, datasets in targets.items():
            for dataset in datasets:
                subset = f"{track}/{dataset}"
                candidates_by_subset[subset] = []
                dataset_index = 0 if dataset == "UCR" else 1
                for encoder_type in search.ENCODER_TYPES:
                    candidate_id = f"{dataset.lower()}_{encoder_type}"
                    candidates.append(
                        {
                            "id": candidate_id,
                            "resolved_parameters": {
                                "encoder_type": encoder_type
                            },
                        }
                    )
                    candidates_by_subset[subset].append(candidate_id)
                    for head_index, (split, top_k) in enumerate(
                        ((4, 1), (16, 3))
                    ):
                        score = scores[encoder_type][dataset_index]
                        if head_index == 1:
                            score += 0.01
                        rows.append(
                            {
                                "track": track,
                                "dataset": dataset,
                                "training_candidate_id": candidate_id,
                                "final_gb_min_split": split,
                                "top_k": top_k,
                                "head_id": f"H{head_index:02d}",
                                "selection_score": score,
                                "seeds": 1,
                            }
                        )
        plan = {
            "targets": targets,
            "training_candidates": candidates,
            "candidates_by_subset": candidates_by_subset,
            "metadata": {
                "reference_selected_heads": {
                    "U/UCR": {"final_gb_min_split": 4, "top_k": 1},
                    "M/CATSv2": {"final_gb_min_split": 4, "top_k": 1},
                }
            },
            "variants_per_unit": 2,
        }
        selection = search._encoder_e1_selection(rows, plan)
        self.assertEqual(
            selection["selected_encoder_type_global"],
            "multiscale_depthwise_tcn",
        )
        self.assertTrue(
            all(
                head["head_id"] == "H01"
                for head in selection[
                    "selected_heads_for_global_encoder"
                ].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
