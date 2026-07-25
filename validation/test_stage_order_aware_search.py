from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stage_order_aware_search",
    ROOT / "scripts" / "stage_order_aware_search.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_protocol_is_tuning_only_and_resolves_all_subsets() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "stage_order_aware_stage1.json").read_text(encoding="utf-8")
    )
    assert protocol["selection_split"] == "official TSB-AD Tuning only"
    assert protocol["selection_metric"] == "macro VUS-PR only"
    assert protocol["eval_feedback"] is False
    candidates = MODULE.resolve_candidates(protocol)
    assert len(candidates) == 10
    assert all(len(rows) == 6 for rows in candidates.values())
    assert sum(
        len(names) * len(candidates[f"{track}/{dataset}"])
        for track, datasets in protocol["files"].items()
        for dataset, names in datasets.items()
    ) == 132


def test_dimension_splits_are_valid_for_compact_encoder() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "stage_order_aware_stage1.json").read_text(encoding="utf-8")
    )
    for row in MODULE.resolve_candidates(protocol)["M/SVDB"]:
        parameters = row["parameters"]
        assert parameters["invariant_dim"] + parameters["order_dim"] == 32
        assert min(parameters["invariant_dim"], parameters["order_dim"]) >= 2


def test_every_resolved_candidate_is_accepted_by_the_exact_model() -> None:
    model_spec = importlib.util.spec_from_file_location(
        "latest_stage_order_aware",
        ROOT / "STAGE" / "stage.py",
    )
    assert model_spec is not None and model_spec.loader is not None
    model = importlib.util.module_from_spec(model_spec)
    sys.modules[model_spec.name] = model
    model_spec.loader.exec_module(model)
    protocol = json.loads(
        (ROOT / "configs" / "stage_order_aware_stage1.json").read_text(encoding="utf-8")
    )
    for rows in MODULE.resolve_candidates(protocol).values():
        for row in rows:
            config = model.StageConfig(**row["parameters"])
            config.validate()
