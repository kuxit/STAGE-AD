from __future__ import annotations

import json
from pathlib import Path
import unittest

import stage_autopilot as autopilot


PROJECT = Path(__file__).resolve().parents[1]


class StageAutopilotTests(unittest.TestCase):
    def test_declared_split_counts(self) -> None:
        tuning = autopilot.split_files(PROJECT, "Tuning")
        evaluation = autopilot.split_files(PROJECT, "Eva")
        self.assertEqual(sum(len(v) for d in tuning.values() for v in d.values()), 22)
        self.assertEqual(sum(len(v) for d in evaluation.values() for v in d.values()), 193)
        self.assertEqual(
            {k: len(v) for k, v in evaluation["U"].items()},
            {"UCR": 70, "Exathlon": 30, "MSL": 7, "SED": 2, "TODS": 13},
        )
        self.assertEqual(
            {k: len(v) for k, v in evaluation["M"].items()},
            {"CATSv2": 5, "GHL": 23, "LTDB": 4, "SVDB": 28, "TAO": 11},
        )

    def test_candidates_are_deterministic_and_equal_budget(self) -> None:
        policy = json.loads(
            (PROJECT / "experiment_policy.json").read_text(encoding="utf-8")
        )
        first = autopilot.candidate_parameters(policy)
        second = autopilot.candidate_parameters(policy)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        for key, value in autopilot.DEFAULT_STAGE_PARAMETERS.items():
            self.assertEqual(first[0][key], value)
        self.assertEqual(len({autopilot.fingerprint(item) for item in first}), 24)

    def test_locked_fingerprint_excludes_only_audit_timestamp(self) -> None:
        stable = {"schema_version": "x", "selections": {"U/UCR": {"trial": 0}}}
        locked = {
            **stable,
            "locked_fingerprint": autopilot.fingerprint(stable),
            "locked_at": "later",
        }
        reconstructed = {
            key: value
            for key, value in locked.items()
            if key not in {"locked_fingerprint", "locked_at"}
        }
        self.assertEqual(
            autopilot.fingerprint(reconstructed), locked["locked_fingerprint"]
        )


if __name__ == "__main__":
    unittest.main()
