from __future__ import annotations

from dataclasses import replace
import copy
import csv
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from STAGE import stage as stage_impl
from scripts import stage_vuspr_search as search


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "stage_vuspr_stage1a.json"


class StageVUSPRSearchTests(unittest.TestCase):
    def test_repository_protocol_and_full_stage1a_plan_validate(self) -> None:
        protocol, _, _ = search.load_and_validate_protocol(PROTOCOL)
        self.assertEqual(protocol["phase"], "stage1a")
        self.assertEqual(len(protocol["training_candidates"]), 24)
        self.assertEqual(protocol["final_gb_min_splits"], [4])
        self.assertEqual(protocol["top_ks"], [3])

        counts = {
            "U": {"UCR": 3, "Exathlon": 2, "MSL": 2, "SED": 1, "TODS": 3},
            "M": {"CATSv2": 1, "GHL": 3, "LTDB": 1, "SVDB": 3, "TAO": 3},
        }
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            metrics = Path(temporary) / "metrics"
            (repo / "STAGE").mkdir(parents=True)
            (repo / "data" / "File_List").mkdir(parents=True)
            shutil.copyfile(ROOT / "STAGE" / "stage.py", repo / "STAGE" / "stage.py")
            shutil.copyfile(ROOT / "common.py", repo / "common.py")
            (metrics / "utils").mkdir(parents=True)
            (metrics / "utils" / "metrics.py").write_text("# frozen test evaluator\n")
            for track, datasets in counts.items():
                data_dir = repo / "data" / f"TSB-AD-{track}"
                data_dir.mkdir(parents=True)
                names: list[str] = []
                for dataset, count in datasets.items():
                    for index in range(count):
                        name = f"x_{dataset}_tr_1000_case{index}.csv"
                        names.append(name)
                        (data_dir / name).write_text("value,label\n0,0\n")
                with (repo / "data" / "File_List" / f"TSB-AD-{track}-Tuning.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=["file_name"])
                    writer.writeheader()
                    writer.writerows({"file_name": name} for name in names)

            plan = search._plan_stable_payload(repo, PROTOCOL, metrics)
            self.assertEqual(plan["series_count"], 22)
            self.assertEqual(plan["expected_logical_units"], 528)
            self.assertEqual(plan["expected_units"], 528)
            self.assertEqual(plan["variants_per_unit"], 1)

    def test_protocol_rejects_candidate_scoring_axis(self) -> None:
        payload = search.load_json(PROTOCOL)
        payload["training_candidates"][0]["parameters"]["top_k"] = 3
        with self.assertRaisesRegex(ValueError, "must not set axes"):
            search.validate_protocol_payload(payload)

    def test_stage1b_candidate_scope_is_dataset_specific(self) -> None:
        payload = search.load_json(PROTOCOL)
        identifiers = [item["id"] for item in payload["training_candidates"]]
        payload["phase"] = "stage1b"
        payload["seeds"] = [2027, 2028]
        payload["prior_summary"] = {"path": "stage1a.json", "sha256": "0" * 64}
        payload["training_shortlists"] = {
            **{
                f"{track}/{dataset}": identifiers[6:9]
                for track, datasets in payload["targets"].items()
                for dataset in datasets
            },
        }
        payload["training_shortlists"]["U/SED"] = identifiers[:3]
        payload["training_shortlists"]["M/CATSv2"] = identifiers[3:6]
        payload["training_shortlists"]["M/LTDB"] = identifiers[3:6]
        protocol = search.validate_protocol_payload(payload)
        self.assertEqual(
            search._candidate_ids_for_subset(protocol, "U", "UCR"),
            identifiers[6:9],
        )
        self.assertEqual(
            search._candidate_ids_for_subset(protocol, "U", "SED"), identifiers[:3]
        )
        self.assertEqual(
            search._candidate_ids_for_subset(protocol, "M", "CATSv2"), identifiers[3:6]
        )

    def test_short_series_keeps_declared_training_budget(self) -> None:
        protocol, _, _ = search.load_and_validate_protocol(PROTOCOL)
        first = copy.deepcopy(protocol["training_candidates"][0])
        second = copy.deepcopy(first)
        first["resolved_parameters"]["steps"] = 1000
        first["resolved_parameters"]["gb_activation_fraction"] = 0.1
        second["resolved_parameters"]["steps"] = 2000
        second["resolved_parameters"]["gb_activation_fraction"] = 0.4
        one, behavior_one = search.training_execution_signature(
            first, "x_UCR_tr_150_case.csv"
        )
        two, behavior_two = search.training_execution_signature(
            second, "x_UCR_tr_150_case.csv"
        )
        self.assertNotEqual(one, two)
        self.assertNotEqual(behavior_one, behavior_two)
        self.assertEqual(behavior_one["effective_steps"], 1000)
        self.assertEqual(behavior_two["effective_steps"], 2000)
        self.assertFalse(behavior_one["short_series_update_cap"])
        self.assertFalse(behavior_two["short_series_update_cap"])

    def test_selection_uses_vus_pr_only_then_canonical_id(self) -> None:
        low_vus_high_other = {
            metric: (0.40 if metric == "VUS-PR" else 1.0)
            for metric in search.METRICS
        }
        high_vus_low_other = {
            metric: (0.41 if metric == "VUS-PR" else 0.0)
            for metric in search.METRICS
        }
        _, low_score = search._selection_statistics(low_vus_high_other)
        _, high_score = search._selection_statistics(high_vus_low_other)
        self.assertEqual(low_score, 0.40)
        self.assertEqual(high_score, 0.41)
        rows = [
            {
                "training_candidate_id": "z_candidate",
                "selection_score": 0.41,
            },
            {
                "training_candidate_id": "a_candidate",
                "selection_score": 0.41,
            },
        ]
        ranked = sorted(
            rows, key=lambda row: search._ranking_key(row, candidate=True)
        )
        self.assertEqual(ranked[0]["training_candidate_id"], "a_candidate")

    def test_scheduler_prioritizes_long_work_and_defaults_to_three_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            data_dir = repo / "data" / "TSB-AD-U"
            data_dir.mkdir(parents=True)
            short_name = "x_UCR_tr_500_short.csv"
            long_name = "x_UCR_tr_500_long.csv"
            (data_dir / short_name).write_bytes(b"x" * 10)
            (data_dir / long_name).write_bytes(b"x" * 100)
            short_signature = "a" * 64
            long_signature = "b" * 64
            plan = {
                "execution_behaviors": {
                    f"U/UCR/{short_name}": {
                        short_signature: {
                            "effective_steps": 750,
                            "batch_size": 64,
                            "patch_size": 48,
                            "channels": 64,
                        }
                    },
                    f"U/UCR/{long_name}": {
                        long_signature: {
                            "effective_steps": 2000,
                            "batch_size": 128,
                            "patch_size": 96,
                            "channels": 128,
                        }
                    },
                }
            }
            short_task = ("U", "UCR", short_name, 2026, short_signature)
            long_task = ("U", "UCR", long_name, 2026, long_signature)
            ordered = sorted(
                [short_task, long_task],
                key=lambda item: search._task_priority(plan, repo, item),
            )
            self.assertEqual(ordered[0], long_task)

        args = search.parse_args(
            [
                "--protocol",
                "protocol.json",
                "--result-root",
                "results",
                "--metrics-root",
                "external",
                "run",
            ]
        )
        self.assertEqual(args.workers_per_gpu, 3)

    def test_multi_k_cpu_matches_reference_including_clamping(self) -> None:
        rng = np.random.default_rng(7)
        queries = rng.normal(size=(17, 5)).astype(np.float32)
        memory = rng.normal(size=(7, 5)).astype(np.float32)
        config = replace(
            stage_impl.StageConfig(), score_batch_size=4, memory_score_block_size=3
        )
        requested = [1, 3, 9]
        actual = search.score_embeddings_multi_k(
            queries, memory, torch.device("cpu"), config, requested
        )
        for top_k in requested:
            expected = stage_impl.score_embeddings(
                queries, memory, torch.device("cpu"), replace(config, top_k=top_k)
            )
            np.testing.assert_allclose(actual[top_k], expected, rtol=1e-6, atol=1e-7)

    def test_worker_command_bypasses_controller_singleton(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            unit = base / "unit.json"
            argv = [
                "--repo",
                str(base),
                "--protocol",
                str(base / "protocol.json"),
                "--result-root",
                str(base / "results"),
                "--metrics-root",
                str(base / "metrics"),
                "_worker",
                "--track",
                "U",
                "--dataset",
                "UCR",
                "--file",
                "x_UCR_tr_1000_case.csv",
                "--seed",
                "2026",
                "--execution-signature",
                "a" * 64,
                "--physical-gpu",
                "0",
            ]
            with mock.patch.object(
                search, "controller_singleton"
            ) as controller:
                with mock.patch.object(
                    search, "load_frozen_plan_for_worker", return_value={}
                ):
                    with mock.patch.object(
                        search, "execute_unit", return_value=unit
                    ):
                        self.assertEqual(search.main(argv), 0)
                        controller.assert_not_called()

    def test_worker_determinism_environment_is_fail_stop(self) -> None:
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            with mock.patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "0", "CUBLAS_WORKSPACE_CONFIG": ":16:8"},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                    search.configure_worker_determinism("0")
            with mock.patch.dict(
                os.environ,
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "CUBLAS_WORKSPACE_CONFIG": search.CUBLAS_WORKSPACE_CONFIG,
                },
                clear=False,
            ):
                search.configure_worker_determinism("0")
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(prior)

    def test_stage1a_summary_freezes_dataset_specific_top3(self) -> None:
        candidates = [
            {"id": f"c{index}", "config_fingerprint": str(index) * 64}
            for index in range(4)
        ]
        targets = {"U": ["UCR", "SED"], "M": ["GHL", "CATSv2", "LTDB"]}
        files = {
            track: {
                dataset: [f"x_{dataset}_tr_1000_case.csv"] for dataset in datasets
            }
            for track, datasets in targets.items()
        }
        execution_signatures: dict[str, dict[str, str]] = {}
        subset_signatures: dict[str, dict[str, str]] = {}
        for track, datasets in targets.items():
            for dataset in datasets:
                subset = f"{track}/{dataset}"
                file_name = files[track][dataset][0]
                execution_signatures[f"{subset}/{file_name}"] = {
                    candidate["id"]: f"{index + 1:064x}"
                    for index, candidate in enumerate(candidates)
                }
                subset_signatures[subset] = {
                    candidate["id"]: f"{index + 11:064x}"
                    for index, candidate in enumerate(candidates)
                }
        plan = {
            "phase": "stage1a",
            "files": files,
            "targets": targets,
            "seeds": [2026],
            "training_candidates": candidates,
            "candidates_by_subset": {
                f"{track}/{dataset}": [candidate["id"] for candidate in candidates]
                for track, datasets in targets.items()
                for dataset in datasets
            },
            "execution_signatures": execution_signatures,
            "subset_execution_signatures": subset_signatures,
            "final_gb_min_splits": [4],
            "top_ks": [3],
            "shortlist_size": 3,
            "plan_fingerprint": "a" * 64,
            "protocol_fingerprint": "b" * 64,
            "expected_units": len(targets["U"] + targets["M"]) * len(candidates),
            "expected_logical_units": len(targets["U"] + targets["M"])
            * len(candidates),
        }
        with tempfile.TemporaryDirectory() as temporary:
            result_root = Path(temporary)
            for task in search._iter_tasks(plan):
                track, dataset, file_name, seed, signature = task
                candidate_index = int(signature, 16) - 1
                value = 0.9 - 0.1 * candidate_index
                path = search.unit_path(
                    result_root, track, dataset, signature, seed, file_name
                )
                search.atomic_json(
                    path,
                    {
                        "variants": [
                            {
                                "final_gb_min_split": 4,
                                "top_k": 3,
                                "metrics": {metric: value for metric in search.METRICS},
                                "memory_rows": 5,
                                "memory_ratio": 0.25,
                            }
                        ]
                    },
                )
            with mock.patch.object(
                search, "ensure_plan", return_value=plan
            ):
                with mock.patch.object(search, "valid_unit", return_value=True):
                    summary = search.summarize_results(
                        ROOT, PROTOCOL, result_root, ROOT
                    )
            shortlists = summary["selection"]["training_shortlists"]
            self.assertEqual(shortlists["U/UCR"], ["c0", "c1", "c2"])
            self.assertEqual(shortlists["U/SED"], ["c0", "c1", "c2"])
            self.assertEqual(shortlists["M/CATSv2"], ["c0", "c1", "c2"])
            self.assertEqual(shortlists["M/LTDB"], ["c0", "c1", "c2"])
            self.assertNotIn("U/__track__", shortlists)
            self.assertNotIn("M/__track__", shortlists)
            self.assertTrue((result_root / search.SELECTION_JSON_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
