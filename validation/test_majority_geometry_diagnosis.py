from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from STAGE import stage  # noqa: E402
from scripts import stage_vuspr_search as search  # noqa: E402


class MajorityGeometryDiagnosisTests(unittest.TestCase):
    def test_zero_pairwise_weight_preserves_legacy_loss(self) -> None:
        torch.manual_seed(17)
        config = stage.StageConfig(
            patch_size=32,
            channels=16,
            token_dim=8,
            embedding_dim=8,
            group_norm_groups=8,
            overlap_deltas=(8, 16),
            overlap_trim=2,
            multiscale_alignment_weight=0.25,
            timestamp_pairwise_weight=0.0,
        )
        left = torch.randn(6, 32, 8)
        right = torch.randn(6, 32, 8)
        delta = 8
        overlap_left = left[:, delta + 2 : 30]
        overlap_right = right[:, 2 : 32 - delta - 2]
        embedding_left = torch.randn(6, 8)
        embedding_right = torch.randn(6, 8)
        loss, diagnostics = stage.temporal_overlap_loss(
            left,
            right,
            embedding_left,
            embedding_right,
            delta,
            config,
        )
        expected = (
            float(diagnostics["token_cc"])
            + float(diagnostics["overlap_embedding_cc"])
            + 0.25 * float(diagnostics["multiscale_cc"])
        )
        self.assertAlmostEqual(float(loss), expected, places=5)
        self.assertEqual(float(diagnostics["pairwise_weight"]), 0.0)

    def test_pairwise_weight_adds_direct_consistency_term(self) -> None:
        torch.manual_seed(23)
        base = stage.StageConfig(
            patch_size=32,
            channels=16,
            token_dim=8,
            embedding_dim=8,
            group_norm_groups=8,
            overlap_deltas=(8, 16),
            overlap_trim=2,
            multiscale_alignment_weight=0.0,
            timestamp_pairwise_weight=0.0,
        )
        weighted = stage.replace(base, timestamp_pairwise_weight=0.4)
        left = torch.randn(6, 32, 8)
        right = torch.randn(6, 32, 8)
        embedding_left = torch.randn(6, 8)
        embedding_right = torch.randn(6, 8)
        loss_base, diagnostics = stage.temporal_overlap_loss(
            left,
            right,
            embedding_left,
            embedding_right,
            8,
            base,
        )
        loss_weighted, _ = stage.temporal_overlap_loss(
            left,
            right,
            embedding_left,
            embedding_right,
            8,
            weighted,
        )
        pairwise = float(diagnostics["timestamp_pairwise"]) + float(
            diagnostics["interval_pairwise"]
        )
        self.assertAlmostEqual(
            float(loss_weighted - loss_base),
            0.4 * pairwise,
            places=5,
        )

    def test_geometry_refresh_builds_two_partition_snapshots(self) -> None:
        rng = np.random.default_rng(31)
        values = rng.normal(size=(72, 2)).astype(np.float32)
        config = stage.StageConfig(
            patch_size=16,
            channels=8,
            token_dim=4,
            embedding_dim=4,
            dilations=(1, 2),
            group_norm_groups=4,
            dropout=0.0,
            batch_size=4,
            steps=6,
            gb_activation_fraction=0.2,
            gb_refresh_fraction=0.7,
            overlap_deltas=(4, 8),
            overlap_trim=1,
            gb_min_split=4,
            gb_max_rounds=8,
            max_gb_rows=128,
            seed=31,
        )
        model = stage.StageEncoder(2, config)
        result = stage.fit_encoder(
            values,
            model,
            torch.device("cpu"),
            config,
        )
        self.assertIsNotNone(result["gb_refresh_step"])
        self.assertEqual(len(result["training_gb_history"]), 2)
        self.assertEqual(
            [item["built_at_step"] for item in result["training_gb_history"]],
            [result["gb_activation_step"], result["gb_refresh_step"]],
        )

    def test_visual_diagnostic_is_compact_and_label_posthoc(self) -> None:
        rng = np.random.default_rng(41)
        train = rng.normal(size=(64, 8)).astype(np.float32)
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        full = rng.normal(size=(96, 8)).astype(np.float32)
        full /= np.linalg.norm(full, axis=1, keepdims=True)
        memory = stage.CalibratedExemplarMemory(
            vectors=train[[2, 17, 43]],
            region_radii=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            region_sizes=np.asarray([30, 20, 14], dtype=np.int64),
            representative_source_rows=np.asarray([2, 17, 43], dtype=np.int64),
        )
        values = rng.normal(size=(103, 3)).astype(np.float32)
        labels = np.zeros(103, dtype=np.int8)
        labels[78:91] = 1
        scores = rng.random(96, dtype=np.float32)
        item = search.build_visual_diagnostics(
            train_unit=train,
            full_unit=full,
            memory=memory,
            values=values,
            labels=labels,
            train_index=70,
            patch_size=8,
            patch_scores=scores,
        )
        self.assertTrue(item["posthoc_labels_only"])
        self.assertFalse(item["selection_uses_visuals"])
        self.assertEqual(len(item["pca"]["prototypes"]["xy"]), 3)
        self.assertGreater(
            item["score_separation"]["anomaly"]["count"],
            0,
        )
        encoded = json.dumps(item)
        self.assertLess(len(encoded), 2_000_000)

    def test_renderer_writes_png_pdf_and_index(self) -> None:
        renderer_path = (
            ROOT
            / "analysis"
            / "stage_geometry_diagnostics"
            / "render_geometry_diagnostics.py"
        )
        namespace: dict[str, object] = {"__name__": "diagnostic_renderer"}
        exec(renderer_path.read_text(encoding="utf-8"), namespace)
        rng = np.random.default_rng(53)
        train = rng.normal(size=(32, 4)).astype(np.float32)
        train /= np.linalg.norm(train, axis=1, keepdims=True)
        full = rng.normal(size=(45, 4)).astype(np.float32)
        full /= np.linalg.norm(full, axis=1, keepdims=True)
        memory = stage.CalibratedExemplarMemory(
            vectors=train[[1, 9, 21]],
            region_radii=np.asarray([0.1, 0.2, 0.4], dtype=np.float32),
            region_sizes=np.asarray([14, 10, 8], dtype=np.int64),
            representative_source_rows=np.asarray([1, 9, 21], dtype=np.int64),
        )
        values = rng.normal(size=(52, 2)).astype(np.float32)
        labels = np.zeros(52, dtype=np.int8)
        labels[40:48] = 1
        visual = search.build_visual_diagnostics(
            train_unit=train,
            full_unit=full,
            memory=memory,
            values=values,
            labels=labels,
            train_index=35,
            patch_size=8,
            patch_scores=rng.random(45, dtype=np.float32),
        )
        unit = {
            "track": "U",
            "dataset": "Synthetic",
            "file": "Synthetic_35_52.csv",
            "canonical_training_candidate_id": "synthetic_d0_control",
            "seed": 2026,
            "diagnostics": {
                "train_patches": 32,
                "final_memories": [
                    {"final_gb_min_split": 4, "memory_rows": 3}
                ],
                "visual_diagnostics": visual,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            row = namespace["render_unit"](unit, output)
            namespace["write_index"]([row], output)
            self.assertTrue(Path(row["png"]).is_file())
            self.assertTrue(Path(row["pdf"]).is_file())
            self.assertTrue((output / "geometry_diagnostics_index.csv").is_file())


if __name__ == "__main__":
    unittest.main()
