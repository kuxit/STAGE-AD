from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from STAGE.stage import StageConfig  # noqa: E402
from stage_autopilot import stage_parameter_args  # noqa: E402
import stage_vuspr_eval as evaluation  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class StageVUSPREvalTests(unittest.TestCase):
    def test_prepare_subset_cli_accepts_exact_dataset_keys(self) -> None:
        args = evaluation.parse_args(
            [
                "--repo",
                str(ROOT),
                "--metrics-root",
                str(ROOT / "external"),
                "--stage2-protocol",
                "stage2.json",
                "--stage2-result",
                "stage2-result",
                "--result-root",
                "eval-result",
                "prepare-subset",
                "--subset",
                "M/CATSv2",
            ]
        )
        self.assertEqual(args.command, "prepare-subset")
        self.assertEqual(args.subset, ["M/CATSv2"])

    def test_final_memory_split_is_independent_and_cli_mapped(self) -> None:
        default = StageConfig()
        self.assertIsNone(default.final_gb_min_split)
        explicit = StageConfig(gb_min_split=4, final_gb_min_split=64)
        explicit.validate()
        args = stage_parameter_args(
            {"gb_min_split": 4, "final_gb_min_split": 64, "top_k": 9}
        )
        self.assertEqual(
            args,
            [
                "--gb-min-split",
                "4",
                "--final-gb-min-split",
                "64",
                "--top-k",
                "9",
            ],
        )
        with self.assertRaises(ValueError):
            StageConfig(final_gb_min_split=3).validate()

    def test_build_lock_freezes_all_ten_dataset_specific_heads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol_path = root / "stage2.json"
            result = root / "stage2_result"
            destination = root / "eval_lock.json"
            protocol = {
                "phase": "stage2",
                "training_candidates": [
                    {"id": "candidate", "parameters": {"gb_min_split": 4}}
                ],
            }
            write_json(protocol_path, protocol)
            subsets = {
                f"{track}/{dataset}"
                for track, datasets in evaluation.EXPECTED_TARGETS.items()
                for dataset in datasets
            }
            selection_stable = {
                "schema_version": "stage-vuspr-search-selection-v1",
                "phase": "stage2",
                "selection_split": evaluation.SELECTION_SPLIT,
                "eval_feedback": False,
                "score": "macro mean VUS-PR only",
                "tie_breakers": ["canonical_id"],
                "plan_fingerprint": "a" * 64,
                "training_winners": {subset: "candidate" for subset in subsets},
                "selected_heads": {
                    subset: {
                        "head_id": "g64_k9",
                        "final_gb_min_split": 64,
                        "top_k": 9,
                    }
                    for subset in subsets
                },
                "seeds": [2026, 2027, 2028],
            }
            selection = {
                **selection_stable,
                "selection_fingerprint": evaluation.fingerprint(selection_stable),
                "created_at": "test",
            }
            plan = {
                "expected_units": 66,
                "plan_fingerprint": "a" * 64,
                "protocol_sha256": evaluation.sha256_file(protocol_path),
                "source_sha256": {"STAGE/stage.py": "frozen-tuning-source"},
            }
            summary = {
                "phase": "stage2",
                "selection_split": evaluation.SELECTION_SPLIT,
                "eval_feedback": False,
                "completed_units": 66,
                "expected_units": 66,
                "selection": selection,
            }
            write_json(result / "stage_vuspr_search_selection.json", selection)
            write_json(result / "stage_vuspr_search_plan.json", plan)
            write_json(result / "stage_vuspr_search_summary.json", summary)

            lock = evaluation.build_lock(
                ROOT, protocol_path, result, destination
            )
            self.assertEqual(set(lock["selections"]), subsets)
            for frozen in lock["selections"].values():
                self.assertEqual(frozen["parameters"]["gb_min_split"], 4)
                self.assertEqual(frozen["parameters"]["final_gb_min_split"], 64)
                self.assertEqual(frozen["parameters"]["top_k"], 9)
            self.assertEqual(
                evaluation.fingerprint(evaluation._lock_stable(lock)),
                lock["lock_fingerprint"],
            )


if __name__ == "__main__":
    unittest.main()
