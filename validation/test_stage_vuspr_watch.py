from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import stage_vuspr_search as search
from scripts import stage_vuspr_watch as watch


def minimal_plan() -> dict:
    file_name = "x_UCR_tr_1000_case.csv"
    signature = "a" * 64
    candidate = "candidate"
    return {
        "files": {"U": {"UCR": [file_name]}},
        "seeds": [2026],
        "candidates_by_subset": {"U/UCR": [candidate]},
        "execution_signatures": {
            f"U/UCR/{file_name}": {candidate: signature},
        },
        "training_candidates": [
            {"id": candidate, "config_fingerprint": "b" * 64}
        ],
        "expected_units": 1,
        "expected_logical_units": 1,
    }


class StageVUSPRWatchTests(unittest.TestCase):
    def test_partial_scan_uses_exact_runner_validator_and_missing_is_not_invalid(self) -> None:
        plan = minimal_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = watch.scan_units(root, plan)
            self.assertEqual(empty["valid"], 0)
            self.assertEqual(empty["missing"], 1)
            self.assertEqual(empty["invalid"], 0)

            task = next(iter(search._iter_tasks(plan)))
            track, dataset, file_name, seed, signature = task
            path = search.unit_path(root, track, dataset, signature, seed, file_name)
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "track": track,
                        "dataset": dataset,
                        "file": file_name,
                        "seed": seed,
                        "execution_signature": signature,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(
                watch.search, "valid_unit", return_value=True
            ) as validator:
                result = watch.scan_units(root, plan)
            validator.assert_called_once()
            self.assertEqual(result["valid"], 1)
            self.assertEqual(result["missing"], 0)
            self.assertEqual(result["invalid"], 0)
            self.assertEqual(result["valid_logical"], 1)

    def test_unplanned_duplicate_is_reported_without_overwriting(self) -> None:
        plan = minimal_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = next(iter(search._iter_tasks(plan)))
            track, dataset, file_name, seed, signature = task
            payload = {
                "track": track,
                "dataset": dataset,
                "file": file_name,
                "seed": seed,
                "execution_signature": signature,
                "error": None,
            }
            planned = search.unit_path(root, track, dataset, signature, seed, file_name)
            planned.parent.mkdir(parents=True)
            planned.write_text(json.dumps(payload), encoding="utf-8")
            extra = root / "units" / "duplicate.json"
            extra.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(watch.search, "valid_unit", return_value=True):
                result = watch.scan_units(root, plan)
            self.assertEqual(result["valid"], 1)
            self.assertEqual(result["raw"], 2)
            self.assertEqual(result["invalid"], 1)
            self.assertEqual(result["duplicates"], 1)
            self.assertEqual(len(result["unexpected_paths"]), 1)

    def test_forbidden_scan_ignores_atomic_json_temp_but_flags_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "units").mkdir()
            (root / "units" / "unit.json.tmp.123").write_text("{}")
            (root / "units" / "scores.npy").write_bytes(b"x")
            result = watch.scan_forbidden_artifacts(root)
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["inflight_atomic_temp_count"], 1)


if __name__ == "__main__":
    unittest.main()
