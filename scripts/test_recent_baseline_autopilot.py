from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import recent_baseline_autopilot as autopilot  # noqa: E402


LOCK_PATH = PROJECT / "configs" / "aaai_recent_locked_seed2026.json"


class RecentBaselineAutopilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_frozen_lock_and_all_source_hashes(self) -> None:
        validated = autopilot.read_locked_contract(PROJECT, LOCK_PATH)
        self.assertEqual(validated, self.lock)

        stable_lock = {
            key: value
            for key, value in self.lock.items()
            if key not in {"locked_fingerprint", "locked_at"}
        }
        self.assertEqual(
            autopilot.stable_fingerprint(stable_lock),
            self.lock["locked_fingerprint"],
        )
        self.assertEqual(
            autopilot.stable_fingerprint(self.lock["plan"]),
            self.lock["plan_fingerprint"],
        )

        self.assertEqual(
            set(self.lock["source_files"]), set(self.lock["source_sha256"])
        )
        for name, relative in self.lock["source_files"].items():
            with self.subTest(source=name):
                source = (PROJECT / relative).resolve()
                self.assertTrue(source.is_file())
                self.assertTrue(source.is_relative_to(PROJECT.resolve()))
                self.assertEqual(
                    autopilot.sha256_file(source),
                    self.lock["source_sha256"][name],
                )

        base_protocol = (LOCK_PATH.parent / self.lock["base_protocol"]).resolve()
        policy = (LOCK_PATH.parent / self.lock["experiment_policy"]).resolve()
        self.assertEqual(
            autopilot.sha256_file(base_protocol),
            self.lock["base_protocol_sha256"],
        )
        self.assertEqual(
            autopilot.sha256_file(policy),
            self.lock["experiment_policy_sha256"],
        )

    def test_official_allowlists_have_frozen_counts_and_fingerprints(self) -> None:
        expected = {
            "Tuning": ("Tuning", 22, 44),
            "Eval": ("Eva", 193, 386),
        }
        for contract_name, (manifest_name, series_count, unit_count) in expected.items():
            with self.subTest(split=contract_name):
                files = autopilot.official_split_files(PROJECT, manifest_name)
                actual_count = sum(
                    len(names)
                    for datasets in files.values()
                    for names in datasets.values()
                )
                contract = self.lock["files"][contract_name]
                self.assertEqual(actual_count, series_count)
                self.assertEqual(contract["expected_series"], series_count)
                self.assertEqual(
                    autopilot.stable_fingerprint(files),
                    contract["allowlist_fingerprint"],
                )
                self.assertEqual(actual_count * len(autopilot.METHODS), unit_count)

                for track, manifest in contract["manifests"].items():
                    path = (PROJECT / manifest["path"]).resolve()
                    self.assertEqual(
                        autopilot.sha256_file(path), manifest["sha256"]
                    )

        evaluation = autopilot.official_split_files(PROJECT, "Eva")
        self.assertEqual(
            {dataset: len(names) for dataset, names in evaluation["U"].items()},
            {"UCR": 70, "Exathlon": 30, "MSL": 7, "SED": 2, "TODS": 13},
        )
        self.assertEqual(
            {dataset: len(names) for dataset, names in evaluation["M"].items()},
            {"CATSv2": 5, "GHL": 23, "LTDB": 4, "SVDB": 28, "TAO": 11},
        )

    def test_single_frozen_configs_and_resource_budgets(self) -> None:
        self.assertEqual(set(self.lock["methods"]), set(autopilot.METHODS))

        dpad = self.lock["methods"]["DPAD_AAAI24"]
        dpad_parameters = dpad["formal_hyperparameters"]
        self.assertEqual(
            dpad_parameters,
            {
                "dpad_epochs": 100,
                "max_train_windows": 1024,
                "gamma": 0.01,
                "lambda": 10.0,
                "learning_rate": 0.001,
                "k": 5,
                "pairwise_reduction": "sum",
                "optimizer_betas": [0.9, 0.999],
                "optimizer_eps": 1e-08,
                "optimizer_weight_decay": 0.0,
                "optimizer_amsgrad": False,
            },
        )
        self.assertEqual(
            autopilot.stable_fingerprint(dpad_parameters),
            dpad["config_fingerprint"],
        )
        self.assertEqual(dpad_parameters["max_train_windows"] ** 2, 1_048_576)

        dne = self.lock["methods"]["DNE_AAAI25"]
        dne_parameters = dne["formal_hyperparameters"]
        self.assertEqual(
            dne_parameters,
            {
                "dne_epochs": 500,
                "max_train_windows": 4096,
                "declared_max_optimizer_steps_per_unit": 16000,
                "dne_architecture": "resmlp",
                "batch_size": 128,
                "learning_rate": 0.0001,
                "weight_decay": 0.0005,
                "optimizer_betas": [0.9, 0.999],
                "optimizer_eps": 1e-08,
                "optimizer_amsgrad": True,
                "sigma_max": 2.0,
                "noise_level_parts": 3,
                "noise_ratios": [0.5, 0.8, 1.0],
                "learning_rate_decay_epoch": 100,
                "learning_rate_decay_factor": 0.1,
            },
        )
        self.assertEqual(
            autopilot.stable_fingerprint(dne_parameters),
            dne["config_fingerprint"],
        )
        derived_steps = (
            math.ceil(
                dne_parameters["max_train_windows"] / dne_parameters["batch_size"]
            )
            * dne_parameters["dne_epochs"]
        )
        self.assertEqual(
            derived_steps, dne_parameters["declared_max_optimizer_steps_per_unit"]
        )

        self.assertFalse(self.lock["eval_feedback"])
        self.assertEqual(self.lock["candidate_count_per_method"], 1)
        self.assertEqual(self.lock["plan"]["acceptance"]["tuning_units"], 44)
        self.assertEqual(self.lock["plan"]["acceptance"]["eval_units"], 386)
        self.assertFalse(
            self.lock["plan"]["acceptance"]["tuning_metrics_used_for_selection"]
        )

    def test_resume_accepts_only_a_strictly_valid_atomic_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_root = Path(directory) / "recent-results"
            plan = autopilot.build_plan(PROJECT, result_root, LOCK_PATH)
            expected = autopilot.expected_units(result_root, plan, "tuning")[0]
            record = self._valid_record(expected, plan)
            target = expected["target"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(record), encoding="utf-8")

            self.assertEqual(
                autopilot.validate_unit(target, expected, plan, physical_gpu="0"),
                record,
            )
            audit = autopilot.audit_phase(result_root, plan, "tuning")
            self.assertEqual(audit["valid_units"], 1)
            self.assertEqual(audit["error_units"], 0)
            self.assertEqual(audit["by_method"][expected["method"]], 1)

            mutations = {
                "wrong seed": lambda item: item.__setitem__("seed", 2027),
                "wrong split": lambda item: item.__setitem__("split", "Eval"),
                "wrong lock": lambda item: item.__setitem__(
                    "locked_fingerprint", "0" * 64
                ),
                "wrong config": lambda item: item.__setitem__(
                    "config_fingerprint", "0" * 64
                ),
                "wrong plan": lambda item: item.__setitem__(
                    "plan_fingerprint", "0" * 64
                ),
                "wrong source map": lambda item: item.__setitem__(
                    "source_sha256", {"runner": "0" * 64}
                ),
                "error unit": lambda item: item.__setitem__("error", "failed"),
                "paper eligible runtime": lambda item: item.__setitem__(
                    "runtime_eligible_for_paper", True
                ),
                "cpu result": lambda item: item.__setitem__("device", "cpu"),
                "missing metric": lambda item: item["metrics"].pop(
                    autopilot.METRICS[0]
                ),
                "boolean metric": lambda item: item["metrics"].__setitem__(
                    autopilot.METRICS[0], True
                ),
                "non-finite metric": lambda item: item["metrics"].__setitem__(
                    autopilot.METRICS[0], float("nan")
                ),
                "embedded scores": lambda item: item.__setitem__("scores", [0.1]),
            }
            invalid_path = result_root / "candidate.json"
            for label, mutate in mutations.items():
                with self.subTest(rejects=label):
                    invalid = copy.deepcopy(record)
                    mutate(invalid)
                    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
                    with self.assertRaises(RuntimeError):
                        autopilot.validate_unit(
                            invalid_path, expected, plan, physical_gpu="0"
                        )

            with self.assertRaisesRegex(RuntimeError, "physical GPU mismatch"):
                autopilot.validate_unit(target, expected, plan, physical_gpu="1")

    def test_resume_refuses_a_different_persisted_execution_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_root = Path(directory) / "recent-results"
            plan = autopilot.build_plan(PROJECT, result_root, LOCK_PATH)
            resumed = autopilot.build_plan(PROJECT, result_root, LOCK_PATH)
            self.assertEqual(
                resumed["execution_plan_fingerprint"],
                plan["execution_plan_fingerprint"],
            )

            plan_path = result_root / autopilot.PLAN_NAME
            persisted = json.loads(plan_path.read_text(encoding="utf-8"))
            persisted["execution_plan_fingerprint"] = "0" * 64
            plan_path.write_text(json.dumps(persisted), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "different execution plan"):
                autopilot.build_plan(PROJECT, result_root, LOCK_PATH)

    @staticmethod
    def _valid_record(expected: dict, plan: dict) -> dict:
        return {
            "method": expected["method"],
            "track": expected["track"],
            "dataset": expected["dataset"],
            "file": expected["file"],
            "seed": expected["seed"],
            "profile": "formal",
            "split": expected["split"],
            "locked_fingerprint": plan["locked_fingerprint"],
            "config_fingerprint": expected["config_fingerprint"],
            "plan_fingerprint": plan["contract_plan_fingerprint"],
            "configuration_source": plan["methods"][expected["method"]][
                "configuration_source"
            ],
            "source_sha256": plan["source_sha256"],
            "error": None,
            "runtime_eligible_for_paper": False,
            "device": "cuda:0",
            "physical_gpu": "0",
            "metrics": {metric: 0.5 for metric in autopilot.METRICS},
        }


if __name__ == "__main__":
    unittest.main()
