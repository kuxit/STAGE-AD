#!/usr/bin/env python3
"""Leakage-safe STAGE search selected by official-Tuning VUS-PR only.

Each unit trains one encoder for a unique
``(track, dataset, file, seed, training_candidate)`` key.  The worker extracts
the train and full-series embeddings once, rebuilds the final exemplar memory
for every declared ``final_gb_min_split``, and evaluates every declared
``top_k`` from one blocked nearest-neighbour distance pass per memory.

The script never writes checkpoints or embeddings.  Its standard path writes
only final unit JSONs.  The isolated ``order_o1`` path may instead write
transient, SHA-verified point-score caches so GPU work can continue while a
separate CPU process runs the official metrics; those caches are deleted after
strict final-unit validation.  It exposes five user-facing commands:

* ``plan`` validates and fingerprints the explicit JSON protocol;
* ``run`` executes/resumes atomic unit JSONs through isolated GPU subprocesses;
* ``run-scores`` runs GPU training/scoring without CPU metrics;
* ``evaluate-caches`` consumes transferred score caches on CPU;
* ``summarize`` strictly validates all units and emits full-precision aggregates.

An internal ``_worker`` command is used only by ``run`` so that every training
process receives its GPU through ``CUDA_VISIBLE_DEVICES`` before importing CUDA.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import METRICS, atomic_json, dataset_name  # noqa: E402
from STAGE import stage as stage_impl  # noqa: E402


PROTOCOL_SCHEMA = "stage-vuspr-search-v1"
PLAN_SCHEMA = "stage-vuspr-search-plan-v1"
UNIT_SCHEMA = "stage-vuspr-search-unit-v1"
SCORE_CACHE_SCHEMA = "stage-vuspr-score-cache-v1"
SUMMARY_SCHEMA = "stage-vuspr-search-summary-v1"
PLAN_NAME = "stage_vuspr_search_plan.json"
DATA_PREFLIGHT_NAME = "stage_vuspr_data_preflight.json"
RUNTIME_PREFLIGHT_NAME = "stage_vuspr_runtime_preflight.json"
STATUS_NAME = "stage_vuspr_search_status.json"
SUMMARY_JSON_NAME = "stage_vuspr_search_summary.json"
SUMMARY_CSV_NAME = "stage_vuspr_search_summary.csv"
SELECTION_JSON_NAME = "stage_vuspr_search_selection.json"
CONTROLLER_LOCK_NAME = ".stage_vuspr_search.lock"
SELECTION_SPLIT = "official TSB-AD Tuning only"
UNIT_KIND = "STAGE official-Tuning VUS-PR search unit"

SELECTION_METRIC = "VUS-PR"
SELECTION_TIE_BREAKERS = ("canonical_id",)
EXPECTED_TARGETS = {
    "U": ["UCR", "Exathlon", "MSL", "SED", "TODS"],
    "M": ["CATSv2", "GHL", "LTDB", "SVDB", "TAO"],
}
GEOMETRY_DIAGNOSTIC_TARGETS = {
    "U": ["MSL", "SED"],
    "M": ["CATSv2", "GHL"],
}
ORDER_DIAGNOSTIC_TARGETS = {
    "U": ["MSL"],
    "M": ["GHL"],
}
EXPECTED_FINAL_SPLITS = [4, 16, 64, 256]
EXPECTED_TOP_KS = [1, 3, 5, 9, 15]
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_SELECTION_SCORE = "macro mean VUS-PR only"
STORY_CONTRACT = {
    "canonical_guidance_sha256": (
        "bfca67c805c48d169aa5206f429fd9c71f60e0767fbc2f214ec8d02843b1f3a9"
    ),
    "fixed_problem": "coupled_context_and_patch_abundance_bias_within_normality",
    "alignment_role": (
        "reduce_context_induced_variation_of_shared_temporal_content"
    ),
    "intermediate_geometry_role": "temper_abundance_driven_encoder_exposure",
    "final_geometry_role": (
        "retain_supported_observed_exemplars_at_adaptive_granularity"
    ),
    "implementation_flexible": True,
    "main_selection_excludes_controls": True,
}

INTEGER_CONFIG_FIELDS = {
    "patch_size",
    "channels",
    "token_dim",
    "embedding_dim",
    "group_norm_groups",
    "batch_size",
    "steps",
    "overlap_trim",
    "gb_min_split",
    "gb_max_rounds",
    "embedding_batch_size",
    "score_batch_size",
    "memory_score_block_size",
    "max_gb_rows",
}
FLOAT_CONFIG_FIELDS = {
    "dropout",
    "gb_activation_fraction",
    "gb_refresh_fraction",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "gb_sampling_power",
    "multiscale_alignment_weight",
    "timestamp_pairwise_weight",
    "patch_statistics_weight",
    "memory_radius_weight",
    "memory_radius_quantile",
    "temporal_support_fraction",
    "temporal_transition_weight",
    "order_residual_gate_init",
    "transition_alignment_weight",
}
STRING_CONFIG_FIELDS = {
    "encoder_type",
    "final_geometry_mode",
    "patch_geometry_mode",
}
ENCODER_TYPES = (
    "dilated_residual",
    "depthwise_tcn",
    "multiscale_depthwise_tcn",
)
SEQUENCE_CONFIG_FIELDS = {"dilations", "overlap_deltas"}
FORBIDDEN_TRAINING_FIELDS = {"seed", "top_k"}
CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

if set(METRICS) != set(stage_impl.SIX_METRICS):
    raise RuntimeError("common.py and STAGE/stage.py disagree on the six metrics")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _require_int(value: Any, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _unique_int_list(
    value: Any, label: str, *, minimum: int, require_sorted: bool = True
) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty list")
    result = [_require_int(item, f"{label}[{index}]", minimum) for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    if require_sorted and result != sorted(result):
        raise ValueError(f"{label} must be sorted in ascending order")
    return result


def _normalize_candidate_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    valid_fields = {item.name for item in fields(stage_impl.StageConfig)}
    unknown = set(parameters) - valid_fields
    if unknown:
        raise ValueError(f"unknown StageConfig fields: {sorted(unknown)}")
    forbidden = set(parameters) & FORBIDDEN_TRAINING_FIELDS
    if forbidden:
        raise ValueError(
            "training candidates must not set axes declared separately: "
            f"{sorted(forbidden)}"
        )

    normalized: dict[str, Any] = {}
    for name, value in parameters.items():
        label = f"training parameter {name}"
        if name in INTEGER_CONFIG_FIELDS:
            normalized[name] = _require_int(value, label, 1)
        elif name in FLOAT_CONFIG_FIELDS:
            normalized[name] = _require_number(value, label)
        elif name in STRING_CONFIG_FIELDS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} must be a non-empty string")
            normalized[name] = value.strip()
            if name == "encoder_type" and normalized[name] not in ENCODER_TYPES:
                raise ValueError(
                    f"{label} must be one of {list(ENCODER_TYPES)}"
                )
        elif name in SEQUENCE_CONFIG_FIELDS:
            if not isinstance(value, list) or not value:
                raise ValueError(f"{label} must be a non-empty integer list")
            normalized[name] = tuple(
                _require_int(item, f"{label}[{index}]", 1)
                for index, item in enumerate(value)
            )
        else:
            raise ValueError(f"unsupported training parameter: {name}")
    return normalized


def _resolved_training_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    base = asdict(stage_impl.StageConfig())
    base.update(parameters)
    base["seed"] = 0
    base["top_k"] = 1
    config = stage_impl.StageConfig(**base)
    config.validate()
    resolved = asdict(config)
    resolved.pop("seed")
    resolved.pop("top_k")
    return resolved


def validate_protocol_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate and normalize a v2 search protocol."""

    required = {
        "schema_version",
        "study_id",
        "status",
        "selection_split",
        "eval_feedback_used_by_runner",
        "confirmatory_claim_eligible",
        "expected_source_sha256",
        "training_candidates",
        "final_gb_min_splits",
        "top_ks",
        "targets",
        "seeds",
        "metrics",
        "selection",
        "story_contract",
    }
    allowed = required | {
        "phase",
        "metadata",
        "shortlist_size",
        "training_shortlists",
        "training_winners",
        "prior_summary",
    }
    missing = required - set(payload)
    unknown = set(payload) - allowed
    if missing:
        raise ValueError(f"protocol is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"protocol contains unknown fields: {sorted(unknown)}")
    if payload.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError(f"schema_version must be {PROTOCOL_SCHEMA!r}")

    study_id = payload.get("study_id")
    if not isinstance(study_id, str) or not study_id.strip():
        raise ValueError("study_id must be a non-empty string")
    if payload.get("status") != "posthoc_development_only":
        raise ValueError("status must remain posthoc_development_only")
    if payload.get("selection_split") != SELECTION_SPLIT:
        raise ValueError(f"selection_split must be {SELECTION_SPLIT!r}")
    if payload.get("eval_feedback_used_by_runner") is not False:
        raise ValueError("eval_feedback_used_by_runner must be false")
    if payload.get("confirmatory_claim_eligible") is not False:
        raise ValueError("confirmatory_claim_eligible must be false")
    expected_source = payload.get("expected_source_sha256")
    if not isinstance(expected_source, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_source
    ):
        raise ValueError("expected_source_sha256 must be a SHA-256 hex digest")
    if payload.get("metrics") != list(METRICS):
        raise ValueError(f"metrics must be exactly {list(METRICS)}")
    if payload.get("story_contract") != STORY_CONTRACT:
        raise ValueError(
            "story_contract must preserve the canonical STAGE problem and "
            "the functional roles of alignment, intermediate geometry, and "
            "final observed-exemplar geometry"
        )

    raw_selection = payload.get("selection")
    expected_selection_fields = {
        "score",
        "tie_breakers",
        "stage1_head",
        "stage1_rule",
        "stage2_rule",
        "dataset_specific_rule",
    }
    if not isinstance(raw_selection, Mapping) or set(raw_selection) != expected_selection_fields:
        raise ValueError(
            f"selection must contain exactly {sorted(expected_selection_fields)}"
        )
    if raw_selection.get("score") != EXPECTED_SELECTION_SCORE:
        raise ValueError("selection.score must be macro mean Tuning VUS-PR only")
    if raw_selection.get("tie_breakers") != list(SELECTION_TIE_BREAKERS):
        raise ValueError(
            f"selection.tie_breakers must be {list(SELECTION_TIE_BREAKERS)}"
        )
    if raw_selection.get("stage1_head") != {
        "final_gb_min_split": 4,
        "top_k": 3,
    }:
        raise ValueError("selection.stage1_head must remain final_min=4, top_k=3")
    for name in ("stage1_rule", "stage2_rule", "dataset_specific_rule"):
        if not isinstance(raw_selection.get(name), str) or not raw_selection[name].strip():
            raise ValueError(f"selection.{name} must be a non-empty governance statement")

    phase = payload.get("phase", "stage1a")
    if phase not in {
        "stage1a",
        "stage1b",
        "stage2",
        "encoder_e1",
        "geometry_d1",
        "geometry_g2",
        "order_o1",
    }:
        raise ValueError(
            "phase must be stage1a, stage1b, stage2, encoder_e1, "
            "geometry_d1, geometry_g2, or order_o1"
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")

    seeds = _unique_int_list(payload["seeds"], "seeds", minimum=0)
    declared_final_splits = _unique_int_list(
        payload["final_gb_min_splits"], "final_gb_min_splits", minimum=4
    )
    declared_top_ks = _unique_int_list(payload["top_ks"], "top_ks", minimum=1)
    compact_diagnostic_phases = {"geometry_d1", "geometry_g2", "order_o1"}
    if phase not in compact_diagnostic_phases and (
        declared_final_splits != EXPECTED_FINAL_SPLITS
    ):
        raise ValueError(f"final_gb_min_splits must remain {EXPECTED_FINAL_SPLITS}")
    if phase not in compact_diagnostic_phases and (
        declared_top_ks != EXPECTED_TOP_KS
    ):
        raise ValueError(f"top_ks must remain {EXPECTED_TOP_KS}")
    if phase == "stage1a" and seeds != [2026]:
        raise ValueError("stage1a seeds are frozen to [2026]")
    if phase == "encoder_e1" and seeds != [2026]:
        raise ValueError("encoder_e1 seeds are frozen to [2026]")
    if phase in compact_diagnostic_phases and seeds != [2026]:
        raise ValueError(f"{phase} seeds are frozen to [2026]")
    if phase == "stage1b" and seeds != [2027, 2028]:
        raise ValueError("stage1b seeds are frozen to [2027, 2028]")
    if phase == "stage2" and seeds != [2026, 2027, 2028]:
        raise ValueError("stage2 seeds are frozen to [2026, 2027, 2028]")
    if phase in {"stage1a", "stage1b"}:
        final_splits = [4]
        top_ks = [3]
    elif phase in {"geometry_d1", "geometry_g2"}:
        if declared_final_splits != [4, 64]:
            raise ValueError(f"{phase} final_gb_min_splits must be [4, 64]")
        if declared_top_ks != [1, 3]:
            raise ValueError(f"{phase} top_ks must be [1, 3]")
        final_splits = declared_final_splits
        top_ks = declared_top_ks
    elif phase == "order_o1":
        if declared_final_splits != [4]:
            raise ValueError("order_o1 final_gb_min_splits must be [4]")
        if declared_top_ks != [3]:
            raise ValueError("order_o1 top_ks must be [3]")
        final_splits = declared_final_splits
        top_ks = declared_top_ks
    else:
        final_splits = declared_final_splits
        top_ks = declared_top_ks

    raw_targets = payload["targets"]
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise ValueError("targets must be a non-empty object")
    if not set(raw_targets).issubset({"U", "M"}):
        raise ValueError("target tracks are restricted to U and M")
    targets: dict[str, list[str]] = {}
    for track in ("U", "M"):
        if track not in raw_targets:
            continue
        datasets = raw_targets[track]
        if not isinstance(datasets, list) or not datasets:
            raise ValueError(f"targets.{track} must be a non-empty list")
        if any(not isinstance(item, str) or not item.strip() for item in datasets):
            raise ValueError(f"targets.{track} must contain non-empty dataset names")
        cleaned = [item.strip() for item in datasets]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError(f"targets.{track} contains duplicate datasets")
        targets[track] = cleaned
    expected_targets = (
        GEOMETRY_DIAGNOSTIC_TARGETS
        if phase in {"geometry_d1", "geometry_g2"}
        else ORDER_DIAGNOSTIC_TARGETS
        if phase == "order_o1"
        else EXPECTED_TARGETS
    )
    if targets != expected_targets:
        raise ValueError(f"targets must remain the frozen subsets: {expected_targets}")

    raw_candidates = payload["training_candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("training_candidates must be a non-empty list")
    candidate_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping) or set(raw) != {"id", "parameters"}:
            raise ValueError(
                f"training_candidates[{index}] must contain exactly id and parameters"
            )
        candidate_id = raw["id"]
        if not isinstance(candidate_id, str) or not CANDIDATE_ID_RE.fullmatch(candidate_id):
            raise ValueError(f"invalid training candidate id: {candidate_id!r}")
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate training candidate id: {candidate_id}")
        candidate_ids.add(candidate_id)
        if not isinstance(raw["parameters"], Mapping):
            raise ValueError(f"parameters for {candidate_id} must be an object")
        normalized_parameters = _normalize_candidate_parameters(raw["parameters"])
        resolved = _resolved_training_parameters(normalized_parameters)
        family_valid = (
            tuple(resolved["dilations"]) != (1, 2, 4, 8, 4, 2)
            or int(resolved["group_norm_groups"]) != 8
            or int(resolved["gb_max_rounds"]) != 64
            or str(resolved["alignment_objective"]) != "both"
            or not 0.0 <= float(resolved["gb_sampling_power"]) < 1.0
        )
        if family_valid or (
            phase not in {"geometry_d1", "geometry_g2", "order_o1"}
            and int(resolved["gb_min_split"]) != 4
        ) or (
            phase == "geometry_d1"
            and int(resolved["gb_min_split"]) not in {4, 32}
        ) or (
            phase == "geometry_g2"
            and int(resolved["gb_min_split"]) != 4
        ) or (
            phase == "order_o1"
            and int(resolved["gb_min_split"]) != 4
        ):
            raise ValueError(
                f"{candidate_id} violates the frozen story/same-family boundary"
            )
        candidates.append(
            {
                "id": candidate_id,
                "parameters": normalized_parameters,
                "resolved_parameters": resolved,
                "config_fingerprint": fingerprint(resolved),
                "path_id": f"candidate_{index:03d}_{candidate_id}",
            }
        )

    if not 8 <= len(candidates) <= 64:
        raise ValueError("the VUS-PR protocol requires 8--64 frozen candidates")
    candidate_ids_set = {item["id"] for item in candidates}
    subset_keys = {
        f"{track}/{dataset}"
        for track, datasets in targets.items()
        for dataset in datasets
    }
    if phase == "encoder_e1":
        reference_heads = metadata.get("reference_selected_heads")
        if (
            not isinstance(reference_heads, Mapping)
            or set(reference_heads) != subset_keys
        ):
            raise ValueError(
                "encoder_e1 metadata.reference_selected_heads must contain "
                "exactly every target subset"
            )
        normalized_reference_heads: dict[str, dict[str, int]] = {}
        for subset, raw_head in reference_heads.items():
            if not isinstance(raw_head, Mapping):
                raise ValueError(
                    f"metadata.reference_selected_heads.{subset} must be an object"
                )
            split = _require_int(
                raw_head.get("final_gb_min_split"),
                f"metadata.reference_selected_heads.{subset}.final_gb_min_split",
                4,
            )
            top_k = _require_int(
                raw_head.get("top_k"),
                f"metadata.reference_selected_heads.{subset}.top_k",
                1,
            )
            if split not in declared_final_splits or top_k not in declared_top_ks:
                raise ValueError(
                    f"metadata.reference_selected_heads.{subset} is outside "
                    "the declared head grid"
                )
            normalized_reference_heads[str(subset)] = {
                "final_gb_min_split": split,
                "top_k": top_k,
            }
        metadata = dict(metadata)
        metadata["reference_selected_heads"] = normalized_reference_heads
    shortlist_size = _require_int(payload.get("shortlist_size", 3), "shortlist_size", 1)
    expected_shortlist_size = (
        4
        if phase in {"geometry_d1", "order_o1"}
        else 5
        if phase == "geometry_g2"
        else 3
    )
    if shortlist_size != expected_shortlist_size:
        raise ValueError(
            f"shortlist_size must remain {expected_shortlist_size}"
        )
    raw_shortlists = payload.get("training_shortlists")
    training_shortlists: dict[str, list[str]] | None = None
    if phase in {
        "stage1b",
        "encoder_e1",
        "geometry_d1",
        "geometry_g2",
        "order_o1",
    }:
        required_shortlist_keys = subset_keys
        if not isinstance(raw_shortlists, Mapping) or set(raw_shortlists) != required_shortlist_keys:
            raise ValueError(
                f"{phase} training_shortlists must contain every dataset subset"
            )
        training_shortlists = {}
        for key, value in raw_shortlists.items():
            if (
                not isinstance(value, list)
                or len(value) != shortlist_size
                or len(value) != len(set(value))
            ):
                raise ValueError(
                    f"training_shortlists.{key} must contain exactly "
                    f"{shortlist_size} unique candidates"
                )
            if any(item not in candidate_ids_set for item in value):
                raise ValueError(f"training_shortlists.{key} references an unknown candidate")
            training_shortlists[str(key)] = [str(item) for item in value]
    elif raw_shortlists is not None:
        raise ValueError(
            "training_shortlists is allowed only for stage1b, encoder_e1, "
            "geometry_d1, geometry_g2, or order_o1"
        )

    raw_winners = payload.get("training_winners")
    training_winners: dict[str, str] | None = None
    if phase == "stage2":
        if not isinstance(raw_winners, Mapping) or set(raw_winners) != subset_keys:
            raise ValueError("stage2 training_winners must contain exactly every target subset")
        training_winners = {}
        for key, value in raw_winners.items():
            if not isinstance(value, str) or value not in candidate_ids_set:
                raise ValueError(f"training_winners.{key} references an unknown candidate")
            training_winners[str(key)] = value
    elif raw_winners is not None:
        raise ValueError("training_winners is allowed only for stage2")

    prior_summary = payload.get("prior_summary")
    if phase in {"stage1b", "stage2"} and prior_summary is None:
        raise ValueError(f"{phase} requires a fingerprinted prior_summary")
    if prior_summary is not None:
        if not isinstance(prior_summary, Mapping) or set(prior_summary) != {"path", "sha256"}:
            raise ValueError("prior_summary must contain exactly path and sha256")
        if not isinstance(prior_summary["path"], str) or not prior_summary["path"]:
            raise ValueError("prior_summary.path must be a non-empty string")
        if not isinstance(prior_summary["sha256"], str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", prior_summary["sha256"]
        ):
            raise ValueError("prior_summary.sha256 must be a SHA-256 hex digest")

    stable = {
        "schema_version": PROTOCOL_SCHEMA,
        "study_id": study_id.strip(),
        "status": "posthoc_development_only",
        "selection_split": SELECTION_SPLIT,
        "eval_feedback_used_by_runner": False,
        "confirmatory_claim_eligible": False,
        "story_contract": dict(STORY_CONTRACT),
        "expected_source_sha256": expected_source.lower(),
        "phase": phase,
        "metadata": dict(metadata),
        "metrics": list(METRICS),
        "selection": dict(raw_selection),
        "training_candidates": candidates,
        "declared_final_gb_min_splits": declared_final_splits,
        "declared_top_ks": declared_top_ks,
        "final_gb_min_splits": final_splits,
        "top_ks": top_ks,
        "targets": targets,
        "seeds": seeds,
        "shortlist_size": shortlist_size,
        "training_shortlists": training_shortlists,
        "training_winners": training_winners,
        "prior_summary": None if prior_summary is None else dict(prior_summary),
    }
    return stable


def load_and_validate_protocol(path: Path) -> tuple[dict[str, Any], str, str]:
    raw = load_json(path)
    normalized = validate_protocol_payload(raw)
    return normalized, sha256_file(path), fingerprint(normalized)


def _read_official_tuning_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "file_name" not in reader.fieldnames:
            raise ValueError(f"manifest lacks file_name: {path}")
        names = [str(row["file_name"]).strip() for row in reader]
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"manifest has empty or duplicate file names: {path}")
    return names


def _evaluator_source_hashes(metrics_root: Path) -> tuple[Path, dict[str, str]]:
    root = metrics_root.resolve()
    candidates = (root, root / "PaAno")
    evaluator_root = next(
        (candidate for candidate in candidates if (candidate / "utils" / "metrics.py").is_file()),
        None,
    )
    if evaluator_root is None:
        raise FileNotFoundError(
            f"cannot locate PaAno/utils/metrics.py below metrics root: {root}"
        )
    files: list[Path] = []
    for relative in (Path("utils"), Path("affiliation")):
        directory = evaluator_root / relative
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    if not files:
        raise RuntimeError(f"official evaluator source set is empty: {evaluator_root}")
    hashes = {
        path.relative_to(evaluator_root).as_posix(): sha256_file(path) for path in files
    }
    return evaluator_root, hashes


def _candidate_ids_for_subset(
    protocol: Mapping[str, Any], track: str, dataset: str
) -> list[str]:
    all_ids = [str(item["id"]) for item in protocol["training_candidates"]]
    phase = protocol["phase"]
    subset = f"{track}/{dataset}"
    if phase == "stage1a":
        return all_ids
    if phase in {
        "stage1b",
        "encoder_e1",
        "geometry_d1",
        "geometry_g2",
        "order_o1",
    }:
        shortlists = protocol["training_shortlists"]
        return [str(item) for item in shortlists[subset]]
    return [str(protocol["training_winners"][subset])]


def training_execution_signature(
    candidate: Mapping[str, Any], file_name: str
) -> tuple[str, dict[str, Any]]:
    """Fingerprint the behavior that actually reaches ``fit_encoder``.

    Every series executes the protocol-declared update budget.  Requested steps
    and activation fractions therefore remain part of the effective behavior
    and cannot be silently deduplicated on short series.
    """

    config = stage_impl.StageConfig(
        **dict(candidate["resolved_parameters"]), seed=0, top_k=1
    )
    train_index = stage_impl.parse_train_index(file_name)
    eligible_count = train_index - int(config.patch_size) - max(config.overlap_deltas) + 1
    if eligible_count < 2:
        raise ValueError(f"training prefix is too short for candidate {candidate['id']}: {file_name}")
    effective_steps, short_series_cap = stage_impl.resolve_training_steps(
        eligible_count, int(config.batch_size), int(config.steps)
    )
    activation_step = int(round(effective_steps * float(config.gb_activation_fraction)))
    activation_step = min(max(1, activation_step), max(1, effective_steps - 1))
    refresh_step = (
        None
        if config.gb_refresh_fraction is None
        else int(round(effective_steps * float(config.gb_refresh_fraction)))
    )
    if refresh_step is not None:
        refresh_step = min(
            max(activation_step + 1, refresh_step),
            max(activation_step + 1, effective_steps - 1),
        )
    behavior = dict(candidate["resolved_parameters"])
    behavior.pop("steps")
    behavior.pop("gb_activation_fraction")
    behavior.pop("gb_refresh_fraction")
    # Final-memory geometry is evaluated after a shared encoder pass.  These
    # fields must not create duplicate GPU training units.
    for geometry_field in (
        "final_geometry_mode",
        "memory_radius_weight",
        "memory_radius_quantile",
        "temporal_support_fraction",
        "temporal_transition_weight",
    ):
        behavior.pop(geometry_field)
    behavior.update(
        effective_steps=int(effective_steps),
        gb_activation_step=int(activation_step),
        gb_refresh_step=(
            None if refresh_step is None else int(refresh_step)
        ),
        short_series_update_cap=bool(short_series_cap),
    )
    return fingerprint(behavior), behavior


def _plan_stable_payload(
    repo: Path, protocol_path: Path, metrics_root: Path
) -> dict[str, Any]:
    protocol, protocol_sha256, protocol_fingerprint = load_and_validate_protocol(
        protocol_path
    )
    prior_summary: dict[str, Any] | None = None
    if protocol["phase"] in {"stage1b", "stage2"}:
        prior_summary = _load_prior_summary(protocol, protocol_path)
    files: dict[str, dict[str, list[str]]] = {}
    manifest_sha256: dict[str, str] = {}
    data_sha256: dict[str, str] = {}
    series_metadata: dict[str, dict[str, int]] = {}
    for track, requested_datasets in protocol["targets"].items():
        manifest = repo / "data" / "File_List" / f"TSB-AD-{track}-Tuning.csv"
        names = _read_official_tuning_manifest(manifest)
        manifest_sha256[track] = sha256_file(manifest)
        by_dataset = {
            dataset: [name for name in names if dataset_name(name) == dataset]
            for dataset in requested_datasets
        }
        missing = [dataset for dataset, selected in by_dataset.items() if not selected]
        if missing:
            raise ValueError(f"official {track} Tuning manifest has no members for {missing}")
        for dataset, selected in by_dataset.items():
            for file_name in selected:
                source = repo / "data" / f"TSB-AD-{track}" / file_name
                if not source.is_file():
                    raise FileNotFoundError(source)
                data_key = f"{track}/{dataset}/{file_name}"
                data_sha256[data_key] = sha256_file(source)
                values, train_index, _ = stage_impl.load_series(source)
                series_metadata[data_key] = {
                    "points": int(len(values)),
                    "channels": int(values.shape[1]),
                    "train_index": int(train_index),
                    "data_bytes": int(source.stat().st_size),
                    "sliding_window": int(stage_impl.estimate_sliding_window(values)),
                }
        files[track] = by_dataset

    script_path = Path(__file__).resolve()
    stage_path = repo / "STAGE" / "stage.py"
    common_path = repo / "common.py"
    for source in (script_path, stage_path, common_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    evaluator_root, evaluator_hashes = _evaluator_source_hashes(metrics_root)

    actual_stage_hash = sha256_file(stage_path)
    if actual_stage_hash != protocol["expected_source_sha256"]:
        raise RuntimeError(
            "STAGE/stage.py does not match protocol expected_source_sha256"
        )

    series_count = sum(
        len(names) for datasets in files.values() for names in datasets.values()
    )
    if protocol["phase"] == "stage1a":
        if series_count != 22:
            raise RuntimeError("stage1a governance requires the 22 frozen Tuning series")

    candidate_lookup = {
        str(item["id"]): item for item in protocol["training_candidates"]
    }
    candidates_by_subset: dict[str, list[str]] = {}
    execution_signatures: dict[str, dict[str, str]] = {}
    execution_behaviors: dict[str, dict[str, dict[str, Any]]] = {}
    subset_execution_signatures: dict[str, dict[str, str]] = {}
    expected_execution_units = 0
    expected_logical_units = 0
    for track, datasets in files.items():
        for dataset, names in datasets.items():
            subset = f"{track}/{dataset}"
            candidate_ids = _candidate_ids_for_subset(protocol, track, dataset)
            candidates_by_subset[subset] = candidate_ids
            per_candidate_series: dict[str, list[str]] = {
                candidate_id: [] for candidate_id in candidate_ids
            }
            for file_name in names:
                data_key = f"{track}/{dataset}/{file_name}"
                execution_signatures[data_key] = {}
                execution_behaviors[data_key] = {}
                signatures_for_file: set[str] = set()
                for candidate_id in candidate_ids:
                    signature, behavior = training_execution_signature(
                        candidate_lookup[candidate_id], file_name
                    )
                    execution_signatures[data_key][candidate_id] = signature
                    if (
                        signature in execution_behaviors[data_key]
                        and execution_behaviors[data_key][signature] != behavior
                    ):
                        raise RuntimeError(
                            "execution signature collision with different behavior"
                        )
                    execution_behaviors[data_key][signature] = behavior
                    per_candidate_series[candidate_id].append(signature)
                    signatures_for_file.add(signature)
                expected_execution_units += len(signatures_for_file) * len(protocol["seeds"])
                expected_logical_units += len(candidate_ids) * len(protocol["seeds"])
            subset_execution_signatures[subset] = {
                candidate_id: fingerprint(per_candidate_series[candidate_id])
                for candidate_id in candidate_ids
            }
    stable = {
        "schema_version": PLAN_SCHEMA,
        "study_id": protocol["study_id"],
        "status": protocol["status"],
        "confirmatory_claim_eligible": False,
        "phase": protocol["phase"],
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "story_contract": protocol["story_contract"],
        "protocol_schema": protocol["schema_version"],
        "protocol_sha256": protocol_sha256,
        "protocol_fingerprint": protocol_fingerprint,
        "source_sha256": {
            "scripts/stage_vuspr_search.py": sha256_file(script_path),
            "STAGE/stage.py": actual_stage_hash,
            "common.py": sha256_file(common_path),
        },
        "official_tuning_manifest_sha256": manifest_sha256,
        "data_sha256": data_sha256,
        "series_metadata": series_metadata,
        "official_evaluator": {
            "root_name": evaluator_root.name,
            "source_sha256": evaluator_hashes,
        },
        "targets": protocol["targets"],
        "files": files,
        "seeds": protocol["seeds"],
        "training_candidates": protocol["training_candidates"],
        "candidates_by_subset": candidates_by_subset,
        "execution_signatures": execution_signatures,
        "execution_behaviors": execution_behaviors,
        "subset_execution_signatures": subset_execution_signatures,
        "declared_final_gb_min_splits": protocol["declared_final_gb_min_splits"],
        "declared_top_ks": protocol["declared_top_ks"],
        "final_gb_min_splits": protocol["final_gb_min_splits"],
        "top_ks": protocol["top_ks"],
        "selection": protocol["selection"],
        "metadata": protocol["metadata"],
        "execution_environment": {
            "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
            "torch_deterministic_algorithms": True,
            "CUDA_VISIBLE_DEVICES": "isolated_physical_gpu_per_worker",
            "runtime_eligible_for_paper": False,
        },
        "shortlist_size": protocol["shortlist_size"],
        "training_shortlists": protocol["training_shortlists"],
        "training_winners": protocol["training_winners"],
        "prior_summary": protocol["prior_summary"],
        "series_count": series_count,
        "expected_logical_units": expected_logical_units,
        "expected_units": expected_execution_units,
        "variants_per_unit": len(protocol["final_gb_min_splits"])
        * len(protocol["top_ks"]),
    }
    if prior_summary is not None:
        prior_summary = _load_prior_summary(stable, protocol_path)
        stable["prior_summary_validation"] = {
            "summary_sha256": str(protocol["prior_summary"]["sha256"]).lower(),
            "phase": prior_summary["phase"],
            "plan_fingerprint": prior_summary["plan_fingerprint"],
            "selection_fingerprint": prior_summary["selection"]["selection_fingerprint"],
        }
    else:
        stable["prior_summary_validation"] = None
    return stable


def _stable_plan_from_frozen(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in plan.items()
        if key not in {"plan_fingerprint", "created_at"}
    }


def validate_frozen_plan_identity(
    repo: Path,
    protocol_path: Path,
    metrics_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Fast identity audit for a locally precomputed or resumed plan.

    The expensive task expansion and per-file data hashes are computed once
    before the server session.  This audit still checks the plan fingerprint,
    protocol, executable sources, official manifests, and evaluator sources.
    The parent then verifies all 22 CSV hashes once before any worker starts.
    """

    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RuntimeError("frozen VUS-PR plan has the wrong schema")
    if fingerprint(_stable_plan_from_frozen(plan)) != plan.get("plan_fingerprint"):
        raise RuntimeError("frozen VUS-PR plan fingerprint is invalid")

    protocol, protocol_sha256, protocol_fingerprint = load_and_validate_protocol(
        protocol_path
    )
    if (
        plan.get("protocol_sha256") != protocol_sha256
        or plan.get("protocol_fingerprint") != protocol_fingerprint
        or plan.get("phase") != protocol.get("phase")
    ):
        raise RuntimeError("protocol differs from the frozen VUS-PR plan")

    source_paths = {
        "scripts/stage_vuspr_search.py": Path(__file__).resolve(),
        "STAGE/stage.py": repo / "STAGE" / "stage.py",
        "common.py": repo / "common.py",
    }
    if set(plan.get("source_sha256", {})) != set(source_paths):
        raise RuntimeError("frozen VUS-PR plan source set is invalid")
    for name, source_path in source_paths.items():
        if (
            not source_path.is_file()
            or sha256_file(source_path) != plan["source_sha256"][name]
        ):
            raise RuntimeError(f"source hash drift in frozen plan: {name}")

    manifest_hashes = plan.get("official_tuning_manifest_sha256")
    if not isinstance(manifest_hashes, Mapping):
        raise RuntimeError("frozen VUS-PR plan lacks manifest hashes")
    for track, expected_hash in manifest_hashes.items():
        manifest = repo / "data" / "File_List" / f"TSB-AD-{track}-Tuning.csv"
        if not manifest.is_file() or sha256_file(manifest) != expected_hash:
            raise RuntimeError(f"official {track} Tuning manifest hash drift")

    evaluator_root, evaluator_hashes = _evaluator_source_hashes(metrics_root)
    evaluator_plan = plan.get("official_evaluator")
    if not isinstance(evaluator_plan, Mapping) or (
        evaluator_plan.get("root_name") != evaluator_root.name
        or evaluator_plan.get("source_sha256") != evaluator_hashes
    ):
        raise RuntimeError("official evaluator differs from the frozen VUS-PR plan")
    return dict(plan)


def _data_preflight_stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key not in {"preflight_fingerprint", "completed_at"}
    }


def ensure_data_preflight(
    repo: Path,
    result_root: Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every frozen Tuning CSV once per result root, not once per unit."""

    stable = {
        "schema_version": "stage-vuspr-data-preflight-v1",
        "status": "complete",
        "plan_fingerprint": plan["plan_fingerprint"],
        "data_sha256": plan["data_sha256"],
    }
    expected_fingerprint = fingerprint(stable)
    target = result_root / DATA_PREFLIGHT_NAME
    if target.is_file():
        prior = load_json(target)
        if (
            fingerprint(_data_preflight_stable(prior)) != expected_fingerprint
            or prior.get("preflight_fingerprint") != expected_fingerprint
        ):
            raise RuntimeError("existing Tuning data preflight is invalid")
        return prior

    def verify(item: tuple[str, str]) -> None:
        data_key, expected_hash = item
        track, _dataset, file_name = data_key.split("/", 2)
        path = repo / "data" / f"TSB-AD-{track}" / file_name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"Tuning data hash drift during preflight: {data_key}")

    items = list(plan["data_sha256"].items())
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
        futures = [pool.submit(verify, item) for item in items]
        for future in as_completed(futures):
            future.result()
    payload = {
        **stable,
        "preflight_fingerprint": expected_fingerprint,
        "completed_at": utc_now(),
    }
    atomic_json(target, payload)
    return payload


def _runtime_preflight_stable(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if key not in {"preflight_fingerprint", "completed_at"}
    }


def ensure_runtime_preflight(
    result_root: Path,
    plan: Mapping[str, Any],
    data_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the one-time parent audits into a small worker trust token."""

    plan_path = result_root / PLAN_NAME
    stable = {
        "schema_version": "stage-vuspr-runtime-preflight-v1",
        "status": "complete",
        "plan_fingerprint": plan["plan_fingerprint"],
        "plan_sha256": sha256_file(plan_path),
        "protocol_sha256": plan["protocol_sha256"],
        "source_sha256": plan["source_sha256"],
        "official_tuning_manifest_sha256": plan[
            "official_tuning_manifest_sha256"
        ],
        "official_evaluator": plan["official_evaluator"],
        "data_preflight_fingerprint": data_preflight["preflight_fingerprint"],
    }
    expected_fingerprint = fingerprint(stable)
    target = result_root / RUNTIME_PREFLIGHT_NAME
    if target.is_file():
        prior = load_json(target)
        if (
            prior.get("preflight_fingerprint") != expected_fingerprint
            or fingerprint(_runtime_preflight_stable(prior))
            != expected_fingerprint
        ):
            raise RuntimeError("existing runtime preflight is invalid")
        return prior
    payload = {
        **stable,
        "preflight_fingerprint": expected_fingerprint,
        "completed_at": utc_now(),
    }
    atomic_json(target, payload)
    return payload


def ensure_plan(
    repo: Path, protocol_path: Path, metrics_root: Path, result_root: Path
) -> dict[str, Any]:
    target = result_root / PLAN_NAME
    if target.is_file():
        return validate_frozen_plan_identity(
            repo, protocol_path, metrics_root, load_json(target)
        )

    stable = _plan_stable_payload(repo, protocol_path, metrics_root)
    plan = {
        **stable,
        "plan_fingerprint": fingerprint(stable),
        "created_at": utc_now(),
    }
    atomic_json(target, plan)
    return plan


def load_frozen_plan_for_worker(
    repo: Path,
    protocol_path: Path,
    metrics_root: Path,
    result_root: Path,
    *,
    track: str,
    dataset: str,
    file_name: str,
    seed: int,
    execution_signature: str,
) -> dict[str, Any]:
    """Validate a frozen plan without rescanning every manifest/data file.

    The parent verifies protocol/source/evaluator identity and all Tuning CSV
    hashes once.  Each worker then checks the byte hash of that frozen plan and
    the two preflight tokens before validating task membership.  This avoids
    repeating the same source/data scans for hundreds of units.
    """

    plan_path = result_root / PLAN_NAME
    if not plan_path.is_file():
        raise FileNotFoundError(f"worker requires an existing frozen plan: {plan_path}")
    plan = load_json(plan_path)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise RuntimeError("frozen worker plan has the wrong schema")
    if plan.get("execution_environment") != {
        "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG,
        "torch_deterministic_algorithms": True,
        "CUDA_VISIBLE_DEVICES": "isolated_physical_gpu_per_worker",
        "runtime_eligible_for_paper": False,
    }:
        raise RuntimeError("frozen worker plan has the wrong execution environment")
    runtime_preflight_path = result_root / RUNTIME_PREFLIGHT_NAME
    if not runtime_preflight_path.is_file():
        raise RuntimeError("worker requires the completed runtime preflight")
    runtime_preflight = load_json(runtime_preflight_path)
    if (
        runtime_preflight.get("preflight_fingerprint")
        != fingerprint(_runtime_preflight_stable(runtime_preflight))
        or runtime_preflight.get("plan_fingerprint")
        != plan.get("plan_fingerprint")
        or runtime_preflight.get("plan_sha256") != sha256_file(plan_path)
        or runtime_preflight.get("protocol_sha256")
        != plan.get("protocol_sha256")
        or runtime_preflight.get("source_sha256") != plan.get("source_sha256")
        or runtime_preflight.get("official_tuning_manifest_sha256")
        != plan.get("official_tuning_manifest_sha256")
        or runtime_preflight.get("official_evaluator")
        != plan.get("official_evaluator")
    ):
        raise RuntimeError("worker runtime preflight is invalid")

    if track not in plan.get("files", {}) or dataset not in plan["files"][track]:
        raise ValueError(f"task target is outside the frozen plan: {track}/{dataset}")
    if file_name not in plan["files"][track][dataset]:
        raise ValueError(f"task file is outside the frozen plan: {file_name}")
    if seed not in plan.get("seeds", []):
        raise ValueError(f"task seed is outside the frozen plan: {seed}")
    execution_group(plan, track, dataset, file_name, execution_signature)

    data_path = repo / "data" / f"TSB-AD-{track}" / file_name
    expected_data_hash = plan.get("data_sha256", {}).get(
        f"{track}/{dataset}/{file_name}"
    )
    if not isinstance(expected_data_hash, str) or not data_path.is_file():
        raise RuntimeError("worker task lacks a frozen data source")
    data_preflight_path = result_root / DATA_PREFLIGHT_NAME
    if not data_preflight_path.is_file():
        raise RuntimeError("worker requires the completed Tuning data preflight")
    data_preflight = load_json(data_preflight_path)
    expected_preflight = {
        "schema_version": "stage-vuspr-data-preflight-v1",
        "status": "complete",
        "plan_fingerprint": plan["plan_fingerprint"],
        "data_sha256": plan["data_sha256"],
    }
    if (
        data_preflight.get("preflight_fingerprint")
        != fingerprint(expected_preflight)
        or fingerprint(_data_preflight_stable(data_preflight))
        != fingerprint(expected_preflight)
        or runtime_preflight.get("data_preflight_fingerprint")
        != data_preflight.get("preflight_fingerprint")
    ):
        raise RuntimeError("worker Tuning data preflight is invalid")
    return plan


def _candidate_by_id(plan: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    matches = [
        item for item in plan["training_candidates"] if item.get("id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"plan has no unique candidate {candidate_id!r}")
    return matches[0]


def execution_group(
    plan: Mapping[str, Any], track: str, dataset: str, file_name: str, signature: str
) -> dict[str, Any]:
    subset = f"{track}/{dataset}"
    data_key = f"{track}/{dataset}/{file_name}"
    candidate_ids = [
        candidate_id
        for candidate_id in plan["candidates_by_subset"][subset]
        if plan["execution_signatures"][data_key][candidate_id] == signature
    ]
    if not candidate_ids:
        raise ValueError(f"unknown execution signature for {data_key}: {signature}")
    candidates = [_candidate_by_id(plan, candidate_id) for candidate_id in candidate_ids]
    return {
        "execution_signature": signature,
        "canonical_candidate_id": candidate_ids[0],
        "candidate_ids": candidate_ids,
        "config_fingerprints": {
            str(candidate["id"]): str(candidate["config_fingerprint"])
            for candidate in candidates
        },
    }


def unit_path(
    result_root: Path,
    track: str,
    dataset: str,
    execution_signature: str,
    seed: int,
    file_name: str,
) -> Path:
    return (
        result_root
        / "units"
        / track
        / dataset
        / f"execution_{execution_signature}"
        / f"seed_{int(seed)}"
        / f"{Path(file_name).stem}.json"
    )


def score_cache_paths(
    result_root: Path,
    track: str,
    dataset: str,
    execution_signature: str,
    seed: int,
    file_name: str,
) -> tuple[Path, Path]:
    base = (
        result_root
        / "score_cache"
        / track
        / dataset
        / f"execution_{execution_signature}"
        / f"seed_{int(seed)}"
        / Path(file_name).stem
    )
    return base.with_suffix(".json"), base.with_suffix(".npz")


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_metric_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(METRICS):
        return False
    for metric in METRICS:
        item = value.get(metric)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return False
        if not math.isfinite(float(item)):
            return False
    return True


def _finite_points(value: Any, *, columns: int, maximum: int) -> bool:
    if not isinstance(value, list) or len(value) > maximum:
        return False
    for row in value:
        if (
            not isinstance(row, list)
            or len(row) != columns
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in row
            )
        ):
            return False
    return True


def _valid_visual_diagnostics(value: Any) -> bool:
    if not isinstance(value, Mapping) or not all(
        (
            value.get("schema_version")
            == "stage-geometry-visual-diagnostics-v1",
            value.get("posthoc_labels_only") is True,
            value.get("selection_uses_visuals") is False,
        )
    ):
        return False
    pca = value.get("pca")
    if not isinstance(pca, Mapping):
        return False
    explained = pca.get("explained_variance_ratio")
    if (
        not isinstance(explained, list)
        or len(explained) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            for item in explained
        )
    ):
        return False
    for key, maximum in (
        ("train", 1000),
        ("normal_queries", 1000),
        ("anomaly_queries", 1000),
    ):
        group = pca.get(key)
        if (
            not isinstance(group, Mapping)
            or not isinstance(group.get("indices"), list)
            or len(group["indices"]) != len(group.get("xy", []))
            or not _finite_points(group.get("xy"), columns=2, maximum=maximum)
        ):
            return False
    anomaly = pca["anomaly_queries"]
    if (
        not isinstance(anomaly.get("anomaly_fraction"), list)
        or len(anomaly["anomaly_fraction"]) != len(anomaly["indices"])
    ):
        return False
    prototypes = pca.get("prototypes")
    prototype_fields = (
        "indices",
        "region_sizes",
        "region_radii",
        "source_starts",
    )
    if (
        not isinstance(prototypes, Mapping)
        or any(not isinstance(prototypes.get(field), list) for field in prototype_fields)
        or len({len(prototypes[field]) for field in prototype_fields})
        != 1
        or len(prototypes["indices"]) != len(prototypes.get("xy", []))
        or not _finite_points(prototypes.get("xy"), columns=2, maximum=1500)
    ):
        return False
    raw = value.get("raw_series")
    if (
        not isinstance(raw, Mapping)
        or not isinstance(raw.get("indices"), list)
        or len(raw["indices"]) > 2500
        or not isinstance(raw.get("labels"), list)
        or len(raw["labels"]) != len(raw["indices"])
        or not isinstance(raw.get("selected_channels"), list)
        or not 1 <= len(raw["selected_channels"]) <= 3
        or not _finite_points(
            raw.get("z_values"),
            columns=len(raw["selected_channels"]),
            maximum=2500,
        )
        or len(raw["z_values"]) != len(raw["indices"])
        or isinstance(raw.get("train_index"), bool)
        or not isinstance(raw.get("train_index"), int)
    ):
        return False
    separation = value.get("score_separation")
    if not isinstance(separation, Mapping):
        return False
    for key in ("normal", "anomaly"):
        quantiles = separation.get(key)
        if not isinstance(quantiles, Mapping) or set(quantiles) != {
            "count",
            "q10",
            "median",
            "q90",
        }:
            return False
        count = quantiles["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return False
        numbers = (quantiles["q10"], quantiles["median"], quantiles["q90"])
        if count == 0 and numbers != (None, None, None):
            return False
        if count > 0 and any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in numbers
        ):
            return False
    examples = value.get("prototype_examples")
    if not isinstance(examples, list) or len(examples) > 8:
        return False
    for example in examples:
        if (
            not isinstance(example, Mapping)
            or not isinstance(example.get("signal_rms_z"), list)
            or len(example["signal_rms_z"]) > 64
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in example["signal_rms_z"]
            )
        ):
            return False
    return True


def valid_unit(
    item: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    track: str,
    dataset: str,
    file_name: str,
    seed: int,
    group: Mapping[str, Any],
) -> bool:
    if not all(
        (
            item.get("schema_version") == UNIT_SCHEMA,
            item.get("kind") == UNIT_KIND,
            item.get("method") == "STAGE",
            item.get("selection_split") == "Tuning",
            item.get("eval_feedback") is False,
            item.get("track") == track,
            item.get("dataset") == dataset,
            item.get("file") == file_name,
            isinstance(item.get("seed"), int)
            and not isinstance(item.get("seed"), bool)
            and item.get("seed") == seed,
            item.get("execution_signature") == group["execution_signature"],
            item.get("canonical_training_candidate_id")
            == group["canonical_candidate_id"],
            item.get("training_candidate_ids") == group["candidate_ids"],
            item.get("training_config_fingerprints")
            == group["config_fingerprints"],
            item.get("execution_behavior")
            == plan["execution_behaviors"][f"{track}/{dataset}/{file_name}"][
                group["execution_signature"]
            ],
            item.get("plan_fingerprint") == plan["plan_fingerprint"],
            item.get("protocol_fingerprint") == plan["protocol_fingerprint"],
            item.get("source_sha256") == plan["source_sha256"],
            item.get("data_sha256")
            == plan["data_sha256"].get(f"{track}/{dataset}/{file_name}"),
            item.get("training_prefix_policy") == "filename_declared_label_blind",
            item.get("error") in (None, ""),
        )
    ):
        return False
    variants = item.get("variants")
    if not isinstance(variants, list):
        return False
    geometry_candidate_ids = (
        list(group["candidate_ids"])
        if plan["phase"] == "geometry_g2"
        else [str(group["canonical_candidate_id"])]
    )
    expected = {
        (candidate_id, int(split), int(top_k))
        for candidate_id in geometry_candidate_ids
        for split in plan["final_gb_min_splits"]
        for top_k in plan["top_ks"]
    }
    observed: set[tuple[str, int, int]] = set()
    partition_metadata: dict[str, tuple[int, str, int]] = {}
    canonical_k: dict[tuple[str, int], int] = {}
    cached_metrics: dict[tuple[str, int], Mapping[str, Any]] = {}
    split_by_partition: dict[str, list[int]] = {}
    for variant in variants:
        if not isinstance(variant, Mapping):
            return False
        split = variant.get("final_gb_min_split")
        top_k = variant.get("top_k")
        effective = variant.get("effective_top_k")
        geometry_candidate_id = variant.get("geometry_candidate_id")
        if (
            geometry_candidate_id not in geometry_candidate_ids
            or not isinstance(variant.get("geometry_mode"), str)
            or
            isinstance(split, bool)
            or not isinstance(split, int)
            or isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or isinstance(effective, bool)
            or not isinstance(effective, int)
            or effective < 1
            or effective > top_k
            or variant.get("top_k_clamped") is not (effective < top_k)
            or not isinstance(variant.get("canonical_final_gb_min_split"), int)
            or variant.get("canonical_final_gb_min_split") not in plan["final_gb_min_splits"]
            or not isinstance(variant.get("canonical_top_k"), int)
            or variant.get("canonical_top_k") not in plan["top_ks"]
            or not isinstance(variant.get("final_partition_fingerprint"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", variant.get("final_partition_fingerprint"))
            or not isinstance(variant.get("memory_fingerprint"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", variant.get("memory_fingerprint"))
            or not _finite_metric_mapping(variant.get("metrics"))
        ):
            return False
        key = (str(geometry_candidate_id), split, top_k)
        if key in observed:
            return False
        observed.add(key)
        memory_rows = variant.get("memory_rows")
        memory_ratio = variant.get("memory_ratio")
        if (
            isinstance(memory_rows, bool)
            or not isinstance(memory_rows, int)
            or memory_rows < 1
            or isinstance(memory_ratio, bool)
            or not isinstance(memory_ratio, (int, float))
            or not math.isfinite(float(memory_ratio))
            or float(memory_ratio) <= 0.0
            or int(effective) != min(int(top_k), int(memory_rows))
        ):
            return False
        partition_fp = str(variant["final_partition_fingerprint"])
        partition_value = (
            int(variant["canonical_final_gb_min_split"]),
            str(variant["memory_fingerprint"]),
            int(memory_rows),
        )
        if partition_fp in partition_metadata and partition_metadata[partition_fp] != partition_value:
            return False
        partition_metadata.setdefault(partition_fp, partition_value)
        effective_key = (partition_fp, int(effective))
        if effective_key in canonical_k and canonical_k[effective_key] != int(
            variant["canonical_top_k"]
        ):
            return False
        canonical_k.setdefault(effective_key, int(variant["canonical_top_k"]))
        expected_canonical_k = next(
            int(requested)
            for requested in plan["top_ks"]
            if min(int(requested), int(memory_rows)) == int(effective)
        )
        if int(variant["canonical_top_k"]) != expected_canonical_k:
            return False
        if effective_key in cached_metrics and cached_metrics[effective_key] != variant["metrics"]:
            return False
        cached_metrics.setdefault(effective_key, variant["metrics"])
        split_by_partition.setdefault(partition_fp, []).append(int(split))
    for partition_fp, splits in split_by_partition.items():
        if partition_metadata[partition_fp][0] != min(splits):
            return False
    diagnostics = item.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get(
        "runtime_eligible_for_paper"
    ) is not False:
        return False
    visual_required = bool(
        plan.get("metadata", {}).get("diagnostic_projection", False)
    )
    visual = diagnostics.get("visual_diagnostics")
    if visual_required and not _valid_visual_diagnostics(visual):
        return False
    if not visual_required and visual is not None:
        return False
    geometry_visuals = diagnostics.get("visual_diagnostics_by_geometry")
    if plan["phase"] == "geometry_g2" and visual_required:
        if (
            not isinstance(geometry_visuals, Mapping)
            or set(geometry_visuals) != set(geometry_candidate_ids)
            or any(
                not _valid_visual_diagnostics(value)
                for value in geometry_visuals.values()
            )
        ):
            return False
    elif plan["phase"] != "geometry_g2" and geometry_visuals is not None:
        if (
            not isinstance(geometry_visuals, Mapping)
            or set(geometry_visuals) != {
                str(group["canonical_candidate_id"])
            }
        ):
            return False
    environment = diagnostics.get("environment")
    if not isinstance(environment, Mapping) or (
        environment.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG
        or environment.get("torch_deterministic_algorithms") is not True
    ):
        return False
    return observed == expected


def score_embeddings_multi_k(
    queries: np.ndarray,
    memory: np.ndarray,
    device: torch.device,
    config: stage_impl.StageConfig,
    top_ks: Sequence[int],
    *,
    memory_penalty: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Return several top-k distance means from one distance pass per memory.

    The blocked matrix multiplication is executed once for each query/memory
    block pair.  The largest requested nearest-neighbour set is retained, then
    sorted once so cumulative means provide every smaller requested ``k``.
    Requested values larger than the memory use the same clamping semantics as
    :func:`STAGE.stage.score_embeddings`.
    """

    query_array = np.asarray(queries, dtype=np.float32)
    memory_array = np.asarray(memory, dtype=np.float32)
    requested = [int(value) for value in top_ks]
    if (
        query_array.ndim != 2
        or memory_array.ndim != 2
        or query_array.shape[1] != memory_array.shape[1]
        or not len(query_array)
        or not len(memory_array)
        or not np.isfinite(query_array).all()
        or not np.isfinite(memory_array).all()
    ):
        raise ValueError("queries and memory must be non-empty compatible finite matrices")
    if not requested or len(requested) != len(set(requested)) or min(requested) < 1:
        raise ValueError("top_ks must be unique positive integers")
    effective = {value: min(value, len(memory_array)) for value in requested}
    maximum_k = max(effective.values())
    memory_tensor = F.normalize(torch.from_numpy(memory_array).to(device), dim=1)
    if memory_penalty is None:
        penalty_array = np.zeros(len(memory_array), dtype=np.float32)
    else:
        penalty_array = np.asarray(memory_penalty, dtype=np.float32)
        if (
            penalty_array.shape != (len(memory_array),)
            or not np.isfinite(penalty_array).all()
            or np.any(penalty_array < 0.0)
        ):
            raise ValueError("memory penalty must be a finite non-negative vector")
    penalty_tensor = torch.from_numpy(penalty_array).to(device)
    outputs: dict[int, list[np.ndarray]] = {value: [] for value in requested}

    with torch.inference_mode():
        for offset in range(0, len(query_array), int(config.score_batch_size)):
            query = torch.from_numpy(
                query_array[offset : offset + int(config.score_batch_size)]
            ).to(device)
            query = F.normalize(query, dim=1)
            best = torch.full(
                (len(query), maximum_k),
                float("inf"),
                device=device,
                dtype=query.dtype,
            )
            block_size = int(config.memory_score_block_size)
            for memory_offset in range(0, len(memory_tensor), block_size):
                block = memory_tensor[memory_offset : memory_offset + block_size]
                block_penalty = penalty_tensor[
                    memory_offset : memory_offset + block_size
                ]
                squared_distance = (
                    (2.0 - 2.0 * (query @ block.T)).clamp_min_(0.0)
                    + block_penalty[None, :]
                )
                block_k = min(maximum_k, len(block))
                block_best = torch.topk(
                    squared_distance,
                    k=block_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
                best = torch.topk(
                    torch.cat([best, block_best], dim=1),
                    k=maximum_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
            ordered = torch.sort(best, dim=1).values
            cumulative = torch.cumsum(ordered, dim=1)
            for requested_k in requested:
                k = effective[requested_k]
                mean = cumulative[:, k - 1] / float(k)
                outputs[requested_k].append(mean.cpu().numpy())

    return {
        requested_k: np.concatenate(parts).astype(np.float64)
        for requested_k, parts in outputs.items()
    }


def _metrics_checked(
    scores: np.ndarray,
    labels: np.ndarray,
    sliding_window: int,
    metrics_root: Path,
) -> dict[str, float]:
    result = stage_impl.official_metrics(scores, labels, sliding_window, metrics_root)
    metrics = {metric: float(result[metric]) for metric in METRICS}
    if not _finite_metric_mapping(metrics):
        raise ValueError("official evaluator returned incomplete or non-finite metrics")
    return metrics


def _environment_identity() -> dict[str, Any]:
    try:
        import sklearn

        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "sklearn": sklearn_version,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def final_partition_fingerprint(partition: stage_impl.GranularPartition) -> str:
    """Fingerprint exact terminal-cell membership, independent of head labels."""

    return fingerprint(
        {
            "source_rows": int(partition.source_rows),
            "balls": [
                [int(index) for index in np.asarray(ball, dtype=np.int64).tolist()]
                for ball in partition.balls
            ],
        }
    )


def array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def configure_worker_determinism(physical_gpu: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != physical_gpu:
        raise RuntimeError(
            f"worker requires CUDA_VISIBLE_DEVICES={physical_gpu}; received {visible!r}"
        )
    cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "").strip()
    if cublas != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "worker requires CUBLAS_WORKSPACE_CONFIG="
            f"{CUBLAS_WORKSPACE_CONFIG}; received {cublas!r}"
        )
    torch.use_deterministic_algorithms(True)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms could not be enabled")


def _spaced_indices(count: int, limit: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, count - 1, int(limit))).astype(np.int64)
    )


def _patch_anomaly_fraction(labels: np.ndarray, patch_size: int) -> np.ndarray:
    binary = (np.asarray(labels).reshape(-1) > 0).astype(np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(binary)])
    return (
        cumulative[int(patch_size) :] - cumulative[: -int(patch_size)]
    ) / float(patch_size)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"count": 0, "q10": None, "median": None, "q90": None}
    return {
        "count": int(len(array)),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "q90": float(np.quantile(array, 0.90)),
    }


def build_visual_diagnostics(
    *,
    train_unit: np.ndarray,
    full_unit: np.ndarray,
    memory: stage_impl.CalibratedExemplarMemory,
    values: np.ndarray,
    labels: np.ndarray,
    train_index: int,
    patch_size: int,
    patch_scores: np.ndarray,
) -> dict[str, Any]:
    """Emit compact, label-after-scoring evidence without raw embeddings."""

    fit_index = _spaced_indices(len(train_unit), 4096)
    fit = np.asarray(train_unit[fit_index], dtype=np.float64)
    center = fit.mean(axis=0, keepdims=True)
    centered = fit - center
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    basis = right[:2].T
    variance = np.square(singular)
    explained = variance[:2] / max(float(variance.sum()), 1e-12)

    fractions = _patch_anomaly_fraction(labels, patch_size)
    normal_candidates = np.flatnonzero(fractions == 0.0)
    anomaly_candidates = np.flatnonzero(fractions > 0.0)
    normal_index = normal_candidates[
        _spaced_indices(len(normal_candidates), 1000)
    ]
    anomaly_index = anomaly_candidates[
        _spaced_indices(len(anomaly_candidates), 1000)
    ]
    train_index_sample = _spaced_indices(len(train_unit), 1000)

    region_sizes = np.asarray(memory.region_sizes, dtype=np.int64)
    if len(memory) <= 1500:
        prototype_index = np.arange(len(memory), dtype=np.int64)
    else:
        large = np.argsort(region_sizes)[-750:]
        spread = _spaced_indices(len(memory), 750)
        prototype_index = np.unique(np.concatenate([large, spread]))

    def project(array: np.ndarray, index: np.ndarray) -> list[list[float]]:
        if len(index) == 0:
            return []
        coordinates = (np.asarray(array[index], dtype=np.float64) - center) @ basis
        return coordinates.astype(np.float32).tolist()

    standardized_center = np.median(values[:train_index], axis=0)
    mad = 1.4826 * np.median(
        np.abs(values[:train_index] - standardized_center[None, :]), axis=0
    )
    fallback = np.std(values[:train_index], axis=0)
    scale = np.where(mad > 1e-6, mad, fallback)
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = (values - standardized_center[None, :]) / scale[None, :]
    channel_variance = np.var(standardized[:train_index], axis=0)
    selected_channels = np.argsort(channel_variance)[-min(3, values.shape[1]) :]
    raw_index = _spaced_indices(len(values), 2500)

    representative_rows = np.asarray(
        memory.representative_source_rows, dtype=np.int64
    )
    ranked_large = np.argsort(region_sizes)[::-1][:4]
    ranked_small = np.argsort(region_sizes)[:4]
    exemplar_examples = []
    for memory_row in np.unique(np.concatenate([ranked_large, ranked_small])):
        start = int(representative_rows[memory_row])
        stop = min(start + int(patch_size), len(values))
        local = standardized[start:stop, selected_channels]
        local_signal = np.sqrt(np.mean(np.square(local), axis=1))
        local_index = _spaced_indices(len(local_signal), 64)
        exemplar_examples.append(
            {
                "memory_row": int(memory_row),
                "source_start": start,
                "region_size": int(region_sizes[memory_row]),
                "region_radius": float(memory.region_radii[memory_row]),
                "signal_rms_z": local_signal[local_index]
                .astype(np.float32)
                .tolist(),
            }
        )

    normal_scores = patch_scores[normal_candidates]
    anomaly_scores = patch_scores[anomaly_candidates]
    return {
        "schema_version": "stage-geometry-visual-diagnostics-v1",
        "posthoc_labels_only": True,
        "selection_uses_visuals": False,
        "pca": {
            "explained_variance_ratio": explained.astype(np.float64).tolist(),
            "train": {
                "indices": train_index_sample.tolist(),
                "xy": project(train_unit, train_index_sample),
            },
            "normal_queries": {
                "indices": normal_index.tolist(),
                "xy": project(full_unit, normal_index),
            },
            "anomaly_queries": {
                "indices": anomaly_index.tolist(),
                "anomaly_fraction": fractions[anomaly_index]
                .astype(np.float32)
                .tolist(),
                "xy": project(full_unit, anomaly_index),
            },
            "prototypes": {
                "indices": prototype_index.tolist(),
                "region_sizes": region_sizes[prototype_index].tolist(),
                "region_radii": np.asarray(
                    memory.region_radii[prototype_index], dtype=np.float32
                ).tolist(),
                "source_starts": representative_rows[prototype_index].tolist(),
                "xy": project(memory.vectors, prototype_index),
            },
        },
        "score_separation": {
            "normal": _quantiles(normal_scores),
            "anomaly": _quantiles(anomaly_scores),
            "median_margin": (
                None
                if len(normal_scores) == 0 or len(anomaly_scores) == 0
                else float(
                    np.median(anomaly_scores) - np.median(normal_scores)
                )
            ),
        },
        "raw_series": {
            "indices": raw_index.tolist(),
            "labels": (np.asarray(labels)[raw_index] > 0).astype(np.int8).tolist(),
            "selected_channels": selected_channels.tolist(),
            "z_values": standardized[raw_index][:, selected_channels]
            .astype(np.float32)
            .tolist(),
            "train_index": int(train_index),
        },
        "prototype_examples": exemplar_examples,
    }


def execute_unit(
    repo: Path,
    result_root: Path,
    metrics_root: Path,
    plan: Mapping[str, Any],
    *,
    track: str,
    dataset: str,
    file_name: str,
    seed: int,
    execution_signature: str,
    physical_gpu: str,
    defer_metrics: bool = False,
) -> Path:
    configure_worker_determinism(physical_gpu)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the isolated v2 worker")
    if track not in plan["files"] or dataset not in plan["files"][track]:
        raise ValueError(f"task target is outside the frozen plan: {track}/{dataset}")
    if file_name not in plan["files"][track][dataset]:
        raise ValueError(f"task file is outside the frozen plan: {file_name}")
    if seed not in plan["seeds"]:
        raise ValueError(f"task seed is outside the frozen plan: {seed}")
    group = execution_group(
        plan, track, dataset, file_name, execution_signature
    )
    candidate = _candidate_by_id(plan, group["canonical_candidate_id"])
    target = unit_path(
        result_root, track, dataset, execution_signature, seed, file_name
    )
    if target.is_file():
        prior = load_json(target)
        if valid_unit(
            prior,
            plan=plan,
            track=track,
            dataset=dataset,
            file_name=file_name,
            seed=seed,
            group=group,
        ):
            return target
        raise RuntimeError(f"refusing to overwrite invalid existing unit: {target}")

    data_path = repo / "data" / f"TSB-AD-{track}" / file_name
    expected_data_hash = plan["data_sha256"][f"{track}/{dataset}/{file_name}"]
    config_kwargs = dict(candidate["resolved_parameters"])
    config_kwargs.update(seed=int(seed), top_k=max(int(item) for item in plan["top_ks"]))
    training_config = stage_impl.StageConfig(**config_kwargs)
    training_config.validate()
    device = torch.device("cuda:0")

    values, train_index, label_column = stage_impl.load_series(data_path)
    series_metadata = plan["series_metadata"][f"{track}/{dataset}/{file_name}"]
    if (
        int(len(values)) != int(series_metadata["points"])
        or int(values.shape[1]) != int(series_metadata["channels"])
        or int(train_index) != int(series_metadata["train_index"])
    ):
        raise RuntimeError(f"Tuning series metadata drift: {data_path}")
    if train_index < training_config.patch_size + max(training_config.overlap_deltas):
        raise ValueError(f"training prefix is too short in {file_name}")
    train_values = values[:train_index]
    feature_center, feature_scale = stage_impl.robust_feature_calibration(
        train_values
    )
    statistics_center, statistics_scale = (
        stage_impl.robust_patch_statistics_calibration(
            train_values,
            training_config.patch_size,
            feature_center,
            feature_scale,
        )
    )
    stage_impl.seed_everything(seed)
    model = stage_impl.StageEncoder(
        values.shape[1],
        training_config,
        feature_center=feature_center,
        feature_scale=feature_scale,
        statistics_center=statistics_center,
        statistics_scale=statistics_scale,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    torch.cuda.reset_peak_memory_stats(device)
    train_window_bank = stage_impl.DeviceWindowBank(
        train_values, device, training_config.patch_size
    )
    training = stage_impl.fit_encoder(
        train_values,
        model,
        device,
        training_config,
        window_bank=train_window_bank,
    )
    raw_starts = np.arange(
        train_index - training_config.patch_size + 1, dtype=np.int64
    )
    full_starts = np.arange(
        len(values) - training_config.patch_size + 1, dtype=np.int64
    )
    train_raw = stage_impl.extract_embeddings(
        model,
        train_values,
        raw_starts,
        device,
        training_config,
        normalize=False,
        window_bank=train_window_bank,
    )
    train_unit = train_raw / np.maximum(
        np.linalg.norm(train_raw, axis=1, keepdims=True), 1e-12
    )
    full_unit = stage_impl.extract_embeddings(
        model,
        values,
        full_starts,
        device,
        training_config,
        normalize=True,
    )
    del train_window_bank
    sliding_window = int(series_metadata["sliding_window"])

    pending_variants: list[dict[str, Any]] = []
    memory_diagnostics: list[dict[str, Any]] = []
    partition_cache: dict[str, dict[str, Any]] = {}
    point_score_cache: dict[tuple[str, int], np.ndarray] = {}
    diagnostic_memories: dict[str, stage_impl.CalibratedExemplarMemory] = {}
    diagnostic_patch_scores: dict[str, np.ndarray] = {}
    diagnostic_split = int(
        plan.get("metadata", {}).get(
            "diagnostic_final_gb_min_split",
            plan["final_gb_min_splits"][0],
        )
    )
    diagnostic_k = int(
        plan.get("metadata", {}).get(
            "diagnostic_top_k",
            plan["top_ks"][0],
        )
    )
    geometry_candidate_ids = (
        list(group["candidate_ids"])
        if plan["phase"] == "geometry_g2"
        else [str(group["canonical_candidate_id"])]
    )
    transition_lag = max(1, int(training_config.patch_size) // 4)
    train_transitions: np.ndarray | None = None
    full_transitions: np.ndarray | None = None
    for geometry_candidate_id in geometry_candidate_ids:
        geometry_candidate = _candidate_by_id(plan, geometry_candidate_id)
        geometry_kwargs = dict(geometry_candidate["resolved_parameters"])
        geometry_kwargs.update(
            seed=int(seed),
            top_k=max(int(item) for item in plan["top_ks"]),
        )
        geometry_config = stage_impl.StageConfig(**geometry_kwargs)
        geometry_config.validate()
        geometry_mode = str(geometry_config.final_geometry_mode)
        if geometry_mode == "support_radius_transition":
            if train_transitions is None or full_transitions is None:
                train_transitions = stage_impl.temporal_transition_embeddings(
                    train_unit, transition_lag
                )
                full_transitions = stage_impl.temporal_transition_embeddings(
                    full_unit, transition_lag
                )

        for final_split in plan["final_gb_min_splits"]:
            final_config = replace(
                geometry_config,
                gb_min_split=int(final_split),
            )
            memory_started = time.perf_counter()
            if geometry_mode == "control":
                calibrated_memory, partition = (
                    stage_impl.build_calibrated_exemplar_memory(
                        train_unit,
                        final_config,
                        seed=int(seed) + 29,
                    )
                )
            else:
                calibrated_memory, partition = (
                    stage_impl.build_support_calibrated_exemplar_memory(
                        train_unit,
                        final_config,
                        seed=int(seed) + 29,
                        independent_representatives=geometry_mode
                        in {
                            "independent_support",
                            "support_radius_combined",
                            "support_radius_transition",
                        },
                        transition_embeddings=(
                            train_transitions
                            if geometry_mode == "support_radius_transition"
                            else None
                        ),
                    )
                )
            memory = calibrated_memory.vectors
            query_vectors = full_unit
            memory_vectors = memory
            memory_penalty = np.zeros(len(memory), dtype=np.float32)
            if geometry_mode in {
                "support_radius",
                "support_radius_combined",
                "support_radius_transition",
            }:
                memory_penalty = stage_impl.support_radius_penalty(
                    calibrated_memory,
                    float(final_config.memory_radius_weight),
                )
            if geometry_mode == "support_radius_transition":
                if (
                    full_transitions is None
                    or calibrated_memory.transition_vectors is None
                ):
                    raise RuntimeError("transition geometry was not constructed")
                query_vectors = stage_impl.augment_temporal_geometry(
                    full_unit,
                    full_transitions,
                    float(final_config.temporal_transition_weight),
                )
                memory_vectors = stage_impl.augment_temporal_geometry(
                    memory,
                    calibrated_memory.transition_vectors,
                    float(final_config.temporal_transition_weight),
                )
            build_seconds = float(time.perf_counter() - memory_started)
            base_partition_fingerprint = final_partition_fingerprint(partition)
            partition_fingerprint = fingerprint(
                {
                    "geometry_candidate_id": geometry_candidate_id,
                    "geometry_mode": geometry_mode,
                    "base_partition_fingerprint": base_partition_fingerprint,
                }
            )
            memory_fingerprint = array_fingerprint(memory_vectors)
            prior_partition = partition_cache.get(partition_fingerprint)
            if prior_partition is None:
                canonical_split = int(final_split)
                unique_effective_top_ks = list(
                    dict.fromkeys(
                        min(int(top_k), int(len(memory)))
                        for top_k in plan["top_ks"]
                    )
                )
                query_started = time.perf_counter()
                patch_scores_by_effective = score_embeddings_multi_k(
                    query_vectors,
                    memory_vectors,
                    device,
                    final_config,
                    unique_effective_top_ks,
                    memory_penalty=memory_penalty,
                )
                if int(final_split) == diagnostic_split:
                    effective_diagnostic_k = min(
                        diagnostic_k,
                        int(len(memory)),
                    )
                    diagnostic_memories[geometry_candidate_id] = calibrated_memory
                    diagnostic_patch_scores[geometry_candidate_id] = (
                        patch_scores_by_effective[effective_diagnostic_k].copy()
                    )
                query_seconds = float(time.perf_counter() - query_started)
                for effective_k, patch_score in patch_scores_by_effective.items():
                    point_score_cache[
                        (partition_fingerprint, int(effective_k))
                    ] = stage_impl.aggregate_patch_scores(
                        patch_score,
                        len(values),
                        training_config.patch_size,
                    )
                partition_cache[partition_fingerprint] = {
                    "canonical_final_gb_min_split": canonical_split,
                    "memory_fingerprint": memory_fingerprint,
                    "memory_rows": int(len(memory)),
                }
                partition_reused = False
                distance_passes = 1
            else:
                if (
                    prior_partition["memory_fingerprint"] != memory_fingerprint
                    or int(prior_partition["memory_rows"]) != int(len(memory))
                ):
                    raise RuntimeError(
                        "identical geometry identity produced different memory"
                    )
                canonical_split = int(
                    prior_partition["canonical_final_gb_min_split"]
                )
                query_seconds = 0.0
                partition_reused = True
                distance_passes = 0

            canonical_k_by_effective: dict[int, int] = {}
            for requested_k in plan["top_ks"]:
                effective_k = min(int(requested_k), int(len(memory)))
                canonical_k_by_effective.setdefault(
                    effective_k,
                    int(requested_k),
                )
                pending_variants.append(
                    {
                        "geometry_candidate_id": geometry_candidate_id,
                        "geometry_mode": geometry_mode,
                        "final_gb_min_split": int(final_split),
                        "canonical_final_gb_min_split": canonical_split,
                        "final_partition_fingerprint": partition_fingerprint,
                        "memory_fingerprint": memory_fingerprint,
                        "top_k": int(requested_k),
                        "effective_top_k": effective_k,
                        "top_k_clamped": bool(
                            effective_k < int(requested_k)
                        ),
                        "canonical_top_k": canonical_k_by_effective[
                            effective_k
                        ],
                        "memory_rows": int(len(memory)),
                        "memory_ratio": float(len(memory) / len(train_unit)),
                        "score_cache_key": (
                            partition_fingerprint,
                            effective_k,
                        ),
                    }
                )
            memory_diagnostics.append(
                {
                    "geometry_candidate_id": geometry_candidate_id,
                    "geometry_mode": geometry_mode,
                    "final_gb_min_split": int(final_split),
                    "canonical_final_gb_min_split": canonical_split,
                    "final_partition_fingerprint": partition_fingerprint,
                    "base_partition_fingerprint": base_partition_fingerprint,
                    "memory_fingerprint": memory_fingerprint,
                    "partition_reused": partition_reused,
                    "initial_k": int(partition.initial_k),
                    "final_k": int(partition.final_k),
                    "rounds": int(partition.rounds),
                    "source_rows": int(partition.source_rows),
                    "memory_rows": int(len(memory)),
                    "mean_effective_support": (
                        None
                        if calibrated_memory.effective_support is None
                        else float(
                            np.mean(calibrated_memory.effective_support)
                        )
                    ),
                    "mean_temporal_components": (
                        None
                        if calibrated_memory.temporal_components is None
                        else float(
                            np.mean(calibrated_memory.temporal_components)
                        )
                    ),
                    "mean_radius_penalty": float(np.mean(memory_penalty)),
                    "transition_lag": (
                        transition_lag
                        if geometry_mode == "support_radius_transition"
                        else None
                    ),
                    "build_seconds": build_seconds,
                    "query_seconds_all_top_ks": query_seconds,
                    "distance_passes": distance_passes,
                    "effective_top_ks": {
                        str(int(top_k)): min(int(top_k), int(len(memory)))
                        for top_k in plan["top_ks"]
                    },
                }
            )

    # The remaining work is label loading plus CPU-only official metrics.  Drop
    # model/embedding allocations before it starts so other workers can use the
    # device while this process evaluates VUS-PR and the reporting metrics.
    train_patches = int(len(train_unit))
    full_patches = int(len(full_unit))
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(device))
    del model, train_raw
    torch.cuda.empty_cache()

    # Labels are loaded only after training, memory construction, and every
    # anomaly-score vector have completed.  They are used solely by the
    # official Tuning evaluator and never feed back into this worker.
    labels = stage_impl.load_labels_after_scoring(data_path, label_column, len(values))
    if defer_metrics:
        if bool(plan.get("metadata", {}).get("diagnostic_projection", False)):
            raise RuntimeError(
                "deferred CPU evaluation currently requires "
                "metadata.diagnostic_projection=false"
            )
        cache_manifest_path, cache_array_path = score_cache_paths(
            result_root,
            track,
            dataset,
            execution_signature,
            seed,
            file_name,
        )
        if cache_manifest_path.exists() or cache_array_path.exists():
            raise RuntimeError(
                "refusing to overwrite an existing deferred score cache: "
                f"{cache_manifest_path}"
            )
        ordered_cache_keys = sorted(
            point_score_cache,
            key=lambda item: (str(item[0]), int(item[1])),
        )
        cache_index = {
            key: index for index, key in enumerate(ordered_cache_keys)
        }
        cached_pending: list[dict[str, Any]] = []
        for pending in pending_variants:
            cache_key = tuple(pending["score_cache_key"])
            cached_pending.append(
                {
                    **{
                        key: value
                        for key, value in pending.items()
                        if key != "score_cache_key"
                    },
                    "score_index": int(cache_index[cache_key]),
                }
            )
        cache_arrays = {
            "labels": np.asarray(labels, dtype=np.int8),
            **{
                f"score_{index:03d}": np.asarray(
                    point_score_cache[key], dtype=np.float64
                )
                for index, key in enumerate(ordered_cache_keys)
            },
        }
        _atomic_npz(cache_array_path, cache_arrays)
        unit_base = {
            "schema_version": UNIT_SCHEMA,
            "kind": UNIT_KIND,
            "method": "STAGE",
            "selection_split": "Tuning",
            "eval_feedback": False,
            "track": track,
            "dataset": dataset,
            "file": file_name,
            "seed": int(seed),
            "execution_signature": execution_signature,
            "execution_behavior": plan["execution_behaviors"][
                f"{track}/{dataset}/{file_name}"
            ][execution_signature],
            "canonical_training_candidate_id": group[
                "canonical_candidate_id"
            ],
            "training_candidate_ids": group["candidate_ids"],
            "training_config_fingerprints": group["config_fingerprints"],
            "plan_fingerprint": plan["plan_fingerprint"],
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "source_sha256": plan["source_sha256"],
            "data_sha256": expected_data_hash,
            "training_prefix_policy": "filename_declared_label_blind",
            "diagnostics": {
                "points": int(len(values)),
                "channels": int(values.shape[1]),
                "train_index": int(train_index),
                "train_patches": int(len(train_unit)),
                "full_patches": int(len(full_unit)),
                "parameter_count": parameter_count,
                "peak_cuda_bytes": peak_cuda_bytes,
                "sliding_window": int(sliding_window),
                "training": training,
                "final_memories": memory_diagnostics,
                "visual_diagnostics": None,
                "visual_diagnostics_by_geometry": None,
                "unique_final_partitions": len(partition_cache),
                "unique_effective_heads": len(point_score_cache),
                "environment": _environment_identity(),
                "runtime_eligible_for_paper": False,
                "gpu_cpu_pipeline": "deferred_official_metrics",
            },
            "error": None,
        }
        cache_manifest = {
            "schema_version": SCORE_CACHE_SCHEMA,
            "plan_fingerprint": plan["plan_fingerprint"],
            "protocol_fingerprint": plan["protocol_fingerprint"],
            "track": track,
            "dataset": dataset,
            "file": file_name,
            "seed": int(seed),
            "execution_signature": execution_signature,
            "sliding_window": int(sliding_window),
            "score_count": len(ordered_cache_keys),
            "score_array_sha256": sha256_file(cache_array_path),
            "pending_variants": cached_pending,
            "unit_base": unit_base,
            "created_at": utc_now(),
        }
        atomic_json(cache_manifest_path, cache_manifest)
        del train_unit, full_unit
        torch.cuda.empty_cache()
        return cache_manifest_path

    variants: list[dict[str, Any]] = []
    metric_cache: dict[tuple[str, int], dict[str, float]] = {}
    for pending in pending_variants:
        cache_key = tuple(pending.pop("score_cache_key"))
        if cache_key not in metric_cache:
            metric_cache[cache_key] = _metrics_checked(
                point_score_cache[cache_key], labels, sliding_window, metrics_root
            )
        variants.append({**pending, "metrics": metric_cache[cache_key]})

    visual_diagnostics = None
    visual_diagnostics_by_geometry: dict[str, dict[str, Any]] | None = None
    if bool(plan.get("metadata", {}).get("diagnostic_projection", False)):
        missing_diagnostics = set(geometry_candidate_ids) - set(
            diagnostic_memories
        )
        if missing_diagnostics or set(diagnostic_memories) != set(
            diagnostic_patch_scores
        ):
            raise RuntimeError(
                "diagnostic memory/score heads were not generated for "
                f"{sorted(missing_diagnostics)}"
            )
        visual_diagnostics_by_geometry = {}
        for geometry_candidate_id in geometry_candidate_ids:
            visual_diagnostics_by_geometry[geometry_candidate_id] = (
                build_visual_diagnostics(
                    train_unit=train_unit,
                    full_unit=full_unit,
                    memory=diagnostic_memories[geometry_candidate_id],
                    values=values,
                    labels=labels,
                    train_index=train_index,
                    patch_size=training_config.patch_size,
                    patch_scores=diagnostic_patch_scores[
                        geometry_candidate_id
                    ],
                )
            )
        visual_diagnostics = visual_diagnostics_by_geometry[
            str(group["canonical_candidate_id"])
        ]
    del train_unit, full_unit

    payload = {
        "schema_version": UNIT_SCHEMA,
        "kind": UNIT_KIND,
        "method": "STAGE",
        "selection_split": "Tuning",
        "eval_feedback": False,
        "track": track,
        "dataset": dataset,
        "file": file_name,
        "seed": int(seed),
        "execution_signature": execution_signature,
        "execution_behavior": plan["execution_behaviors"][
            f"{track}/{dataset}/{file_name}"
        ][execution_signature],
        "canonical_training_candidate_id": group["canonical_candidate_id"],
        "training_candidate_ids": group["candidate_ids"],
        "training_config_fingerprints": group["config_fingerprints"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "source_sha256": plan["source_sha256"],
        "data_sha256": expected_data_hash,
        "training_prefix_policy": "filename_declared_label_blind",
        "variants": variants,
        "diagnostics": {
            "points": int(len(values)),
            "channels": int(values.shape[1]),
            "train_index": int(train_index),
            "train_patches": train_patches,
            "full_patches": full_patches,
            "parameter_count": parameter_count,
            "peak_cuda_bytes": peak_cuda_bytes,
            "sliding_window": int(sliding_window),
            "training": training,
            "final_memories": memory_diagnostics,
            "visual_diagnostics": visual_diagnostics,
            "visual_diagnostics_by_geometry": (
                visual_diagnostics_by_geometry
            ),
            "unique_final_partitions": len(partition_cache),
            "unique_effective_heads": len(metric_cache),
            "environment": _environment_identity(),
            "runtime_eligible_for_paper": False,
        },
        "error": None,
        "completed_at": utc_now(),
    }
    atomic_json(target, payload)
    torch.cuda.empty_cache()
    return target


def valid_score_cache(
    manifest: Mapping[str, Any],
    array_path: Path,
    *,
    plan: Mapping[str, Any],
    track: str,
    dataset: str,
    file_name: str,
    seed: int,
    execution_signature: str,
) -> bool:
    if not all(
        (
            manifest.get("schema_version") == SCORE_CACHE_SCHEMA,
            manifest.get("plan_fingerprint") == plan["plan_fingerprint"],
            manifest.get("protocol_fingerprint")
            == plan["protocol_fingerprint"],
            manifest.get("track") == track,
            manifest.get("dataset") == dataset,
            manifest.get("file") == file_name,
            manifest.get("seed") == int(seed),
            manifest.get("execution_signature") == execution_signature,
            isinstance(manifest.get("score_count"), int),
            int(manifest.get("score_count", 0)) >= 1,
            isinstance(manifest.get("pending_variants"), list),
            isinstance(manifest.get("unit_base"), Mapping),
            array_path.is_file(),
            manifest.get("score_array_sha256") == sha256_file(array_path),
        )
    ):
        return False
    unit_base = manifest["unit_base"]
    if (
        unit_base.get("plan_fingerprint") != plan["plan_fingerprint"]
        or unit_base.get("source_sha256") != plan["source_sha256"]
        or unit_base.get("data_sha256")
        != plan["data_sha256"].get(f"{track}/{dataset}/{file_name}")
    ):
        return False
    score_count = int(manifest["score_count"])
    pending = manifest["pending_variants"]
    if not pending:
        return False
    return all(
        isinstance(item, Mapping)
        and isinstance(item.get("score_index"), int)
        and 0 <= int(item["score_index"]) < score_count
        for item in pending
    )


def evaluate_score_cache(
    result_root: Path,
    metrics_root: Path,
    plan: Mapping[str, Any],
    *,
    track: str,
    dataset: str,
    file_name: str,
    seed: int,
    execution_signature: str,
    remove_cache: bool,
) -> Path:
    group = execution_group(
        plan, track, dataset, file_name, execution_signature
    )
    target = unit_path(
        result_root, track, dataset, execution_signature, seed, file_name
    )
    if target.is_file():
        if valid_unit(
            load_json(target),
            plan=plan,
            track=track,
            dataset=dataset,
            file_name=file_name,
            seed=seed,
            group=group,
        ):
            return target
        raise RuntimeError(f"refusing to overwrite invalid existing unit: {target}")
    manifest_path, array_path = score_cache_paths(
        result_root, track, dataset, execution_signature, seed, file_name
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    if not valid_score_cache(
        manifest,
        array_path,
        plan=plan,
        track=track,
        dataset=dataset,
        file_name=file_name,
        seed=seed,
        execution_signature=execution_signature,
    ):
        raise RuntimeError(f"invalid deferred score cache: {manifest_path}")

    with np.load(array_path, allow_pickle=False) as arrays:
        expected_names = {
            "labels",
            *{
                f"score_{index:03d}"
                for index in range(int(manifest["score_count"]))
            },
        }
        if set(arrays.files) != expected_names:
            raise RuntimeError(f"score cache array set is invalid: {array_path}")
        labels = np.asarray(arrays["labels"], dtype=np.int8)
        score_arrays = {
            index: np.asarray(arrays[f"score_{index:03d}"], dtype=np.float64)
            for index in range(int(manifest["score_count"]))
        }
    if any(len(scores) != len(labels) for scores in score_arrays.values()):
        raise RuntimeError(f"score/label length mismatch: {array_path}")

    metric_cache = {
        index: _metrics_checked(
            scores,
            labels,
            int(manifest["sliding_window"]),
            metrics_root,
        )
        for index, scores in score_arrays.items()
    }
    variants = []
    for pending in manifest["pending_variants"]:
        score_index = int(pending["score_index"])
        variants.append(
            {
                **{
                    key: value
                    for key, value in pending.items()
                    if key != "score_index"
                },
                "metrics": metric_cache[score_index],
            }
        )
    payload = {
        **dict(manifest["unit_base"]),
        "variants": variants,
        "completed_at": utc_now(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, payload)
    if not valid_unit(
        load_json(target),
        plan=plan,
        track=track,
        dataset=dataset,
        file_name=file_name,
        seed=seed,
        group=group,
    ):
        raise RuntimeError(f"CPU-evaluated unit failed strict audit: {target}")
    if remove_cache:
        manifest_path.unlink()
        array_path.unlink()
    return target


def _iter_tasks(plan: Mapping[str, Any]) -> Iterable[tuple[str, str, str, int, str]]:
    for track, datasets in plan["files"].items():
        for dataset, files in datasets.items():
            subset = f"{track}/{dataset}"
            for file_name in files:
                data_key = f"{track}/{dataset}/{file_name}"
                signatures = list(
                    dict.fromkeys(
                        plan["execution_signatures"][data_key][candidate_id]
                        for candidate_id in plan["candidates_by_subset"][subset]
                    )
                )
                for seed in plan["seeds"]:
                    for execution_signature in signatures:
                        yield (
                            track,
                            dataset,
                            file_name,
                            int(seed),
                            str(execution_signature),
                        )


def _assert_no_unplanned_unit_artifacts(
    result_root: Path, plan: Mapping[str, Any]
) -> None:
    expected = {
        unit_path(result_root, track, dataset, signature, seed, file_name).resolve()
        for track, dataset, file_name, seed, signature in _iter_tasks(plan)
    }
    if len(expected) != int(plan["expected_units"]):
        raise RuntimeError("unit paths are not unique for the frozen task plan")
    units_root = result_root / "units"
    if not units_root.exists():
        return
    unexpected = [
        path
        for path in units_root.rglob("*")
        if path.is_file() and path.resolve() not in expected
    ]
    if unexpected:
        preview = ", ".join(str(path) for path in unexpected[:5])
        raise RuntimeError(f"unplanned artifact found under v2 units: {preview}")


def _parse_gpus(value: str) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique comma-separated device identifiers")
    return gpus


def _run_lanes(
    tasks: Sequence[tuple[str, str, str, int, str]],
    gpus: Sequence[str],
    workers_per_gpu: int,
    worker,
) -> None:
    if not tasks:
        return
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be positive")
    # Interleave physical GPUs so the first dispatched tasks cannot all land on
    # one device when worker startup times differ.
    lanes = [gpu for _ in range(workers_per_gpu) for gpu in gpus]
    iterator = iter(tasks)
    iterator_lock = threading.Lock()
    stop = threading.Event()
    failures: list[BaseException] = []

    def lane(gpu: str) -> None:
        while not stop.is_set():
            with iterator_lock:
                try:
                    task = next(iterator)
                except StopIteration:
                    return
            try:
                worker(gpu, task)
            except BaseException as exc:
                failures.append(exc)
                stop.set()
                return

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        futures = [pool.submit(lane, gpu) for gpu in lanes]
        for future in as_completed(futures):
            future.result()
    if failures:
        raise RuntimeError(str(failures[0])) from failures[0]


def _task_priority(
    plan: Mapping[str, Any],
    repo: Path,
    task: tuple[str, str, str, int, str],
) -> tuple[Any, ...]:
    """Longest-estimated task first, with a canonical deterministic tie."""

    track, dataset, file_name, seed, execution_signature = task
    data_key = f"{track}/{dataset}/{file_name}"
    behavior = plan["execution_behaviors"][data_key][execution_signature]
    training_work = (
        int(behavior["effective_steps"])
        * int(behavior["batch_size"])
        * int(behavior["patch_size"])
        * int(behavior["channels"])
    )
    data_bytes = int(plan["series_metadata"][data_key]["data_bytes"])
    scoring_work = data_bytes * int(behavior["patch_size"]) * int(
        behavior["channels"]
    )
    estimated = training_work + scoring_work
    return (
        -int(estimated),
        track,
        dataset,
        file_name,
        int(seed),
        execution_signature,
    )


def run_parent(
    repo: Path,
    protocol_path: Path,
    result_root: Path,
    metrics_root: Path,
    python: Path,
    gpus: Sequence[str],
    workers_per_gpu: int,
) -> dict[str, Any]:
    plan = ensure_plan(repo, protocol_path, metrics_root, result_root)
    data_preflight = ensure_data_preflight(repo, result_root, plan)
    ensure_runtime_preflight(result_root, plan, data_preflight)
    _assert_no_unplanned_unit_artifacts(result_root, plan)
    tasks: list[tuple[str, str, str, int, str]] = []
    for task in _iter_tasks(plan):
        track, dataset, file_name, seed, execution_signature = task
        group = execution_group(
            plan, track, dataset, file_name, execution_signature
        )
        target = unit_path(
            result_root, track, dataset, execution_signature, seed, file_name
        )
        if target.is_file():
            item = load_json(target)
            if valid_unit(
                item,
                plan=plan,
                track=track,
                dataset=dataset,
                file_name=file_name,
                seed=seed,
                group=group,
            ):
                continue
            raise RuntimeError(f"refusing to overwrite invalid existing unit: {target}")
        tasks.append(task)
    tasks.sort(key=lambda item: _task_priority(plan, repo, item))

    total = int(plan["expected_units"])
    counter = {"value": total - len(tasks)}
    counter_lock = threading.Lock()
    script = Path(__file__).resolve()

    def worker(gpu: str, task: tuple[str, str, str, int, str]) -> None:
        track, dataset, file_name, seed, execution_signature = task
        command = [
            str(python),
            str(script),
            "--repo",
            str(repo),
            "--protocol",
            str(protocol_path),
            "--result-root",
            str(result_root),
            "--metrics-root",
            str(metrics_root),
            "_worker",
            "--track",
            track,
            "--dataset",
            dataset,
            "--file",
            file_name,
            "--seed",
            str(seed),
            "--execution-signature",
            execution_signature,
            "--physical-gpu",
            gpu,
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        environment.setdefault("OMP_NUM_THREADS", "2")
        environment.setdefault("MKL_NUM_THREADS", "2")
        environment.setdefault("OPENBLAS_NUM_THREADS", "2")
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"VUS-PR unit failed on GPU {gpu} for {track}/{dataset}/{file_name} "
                f"seed={seed} execution={execution_signature}:\n{completed.stdout[-8000:]}"
            )
        with counter_lock:
            counter["value"] += 1
            print(
                f"[STAGE VUS-PR {counter['value']}/{total}] GPU={gpu} "
                f"{track}/{dataset} seed={seed} execution={execution_signature[:12]} "
                f"{file_name}",
                flush=True,
            )

    _run_lanes(tasks, gpus, workers_per_gpu, worker)
    _assert_no_unplanned_unit_artifacts(result_root, plan)
    validated_units = 0
    for track, dataset, file_name, seed, execution_signature in _iter_tasks(plan):
        group = execution_group(
            plan, track, dataset, file_name, execution_signature
        )
        path = unit_path(
            result_root, track, dataset, execution_signature, seed, file_name
        )
        if not path.is_file() or not valid_unit(
            load_json(path),
            plan=plan,
            track=track,
            dataset=dataset,
            file_name=file_name,
            seed=seed,
            group=group,
        ):
            raise RuntimeError(f"post-run unit audit failed: {path}")
        validated_units += 1
    if validated_units != total:
        raise RuntimeError("post-run unit audit count differs from the frozen plan")
    status = {
        "schema_version": "stage-vuspr-search-status-v1",
        "status": "complete",
        "expected_units": total,
        "expected_logical_units": int(plan["expected_logical_units"]),
        "completed_units": validated_units,
        "error_units": 0,
        "plan_fingerprint": plan["plan_fingerprint"],
        "completed_at": utc_now(),
    }
    atomic_json(result_root / STATUS_NAME, status)
    return status


def run_scores_parent(
    repo: Path,
    protocol_path: Path,
    result_root: Path,
    metrics_root: Path,
    python: Path,
    gpus: Sequence[str],
    workers_per_gpu: int,
) -> dict[str, Any]:
    """Run only GPU training/scoring and leave official metrics to CPU."""

    plan = ensure_plan(repo, protocol_path, metrics_root, result_root)
    data_preflight = ensure_data_preflight(repo, result_root, plan)
    ensure_runtime_preflight(result_root, plan, data_preflight)
    if plan["phase"] != "order_o1":
        raise RuntimeError("run-scores is restricted to the isolated order_o1 phase")
    tasks: list[tuple[str, str, str, int, str]] = []
    for task in _iter_tasks(plan):
        track, dataset, file_name, seed, execution_signature = task
        group = execution_group(
            plan, track, dataset, file_name, execution_signature
        )
        target = unit_path(
            result_root, track, dataset, execution_signature, seed, file_name
        )
        if target.is_file():
            if valid_unit(
                load_json(target),
                plan=plan,
                track=track,
                dataset=dataset,
                file_name=file_name,
                seed=seed,
                group=group,
            ):
                continue
            raise RuntimeError(f"invalid existing unit: {target}")
        manifest_path, array_path = score_cache_paths(
            result_root,
            track,
            dataset,
            execution_signature,
            seed,
            file_name,
        )
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            if valid_score_cache(
                manifest,
                array_path,
                plan=plan,
                track=track,
                dataset=dataset,
                file_name=file_name,
                seed=seed,
                execution_signature=execution_signature,
            ):
                continue
            raise RuntimeError(f"invalid existing score cache: {manifest_path}")
        if array_path.exists():
            raise RuntimeError(f"orphan score array: {array_path}")
        tasks.append(task)
    tasks.sort(key=lambda item: _task_priority(plan, repo, item))

    total = int(plan["expected_units"])
    counter = {"value": total - len(tasks)}
    counter_lock = threading.Lock()
    script = Path(__file__).resolve()

    def worker(gpu: str, task: tuple[str, str, str, int, str]) -> None:
        track, dataset, file_name, seed, execution_signature = task
        command = [
            str(python),
            str(script),
            "--repo",
            str(repo),
            "--protocol",
            str(protocol_path),
            "--result-root",
            str(result_root),
            "--metrics-root",
            str(metrics_root),
            "_score_worker",
            "--track",
            track,
            "--dataset",
            dataset,
            "--file",
            file_name,
            "--seed",
            str(seed),
            "--execution-signature",
            execution_signature,
            "--physical-gpu",
            gpu,
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"order_o1 score unit failed on GPU {gpu} for "
                f"{track}/{dataset}/{file_name} seed={seed} "
                f"execution={execution_signature}:\n{completed.stdout[-8000:]}"
            )
        with counter_lock:
            counter["value"] += 1
            print(
                f"[STAGE O1 SCORES {counter['value']}/{total}] GPU={gpu} "
                f"{track}/{dataset} {file_name}",
                flush=True,
            )

    _run_lanes(tasks, gpus, workers_per_gpu, worker)
    cached_or_complete = 0
    for track, dataset, file_name, seed, execution_signature in _iter_tasks(plan):
        target = unit_path(
            result_root, track, dataset, execution_signature, seed, file_name
        )
        if target.is_file():
            cached_or_complete += 1
            continue
        manifest_path, array_path = score_cache_paths(
            result_root,
            track,
            dataset,
            execution_signature,
            seed,
            file_name,
        )
        if not manifest_path.is_file() or not valid_score_cache(
            load_json(manifest_path),
            array_path,
            plan=plan,
            track=track,
            dataset=dataset,
            file_name=file_name,
            seed=seed,
            execution_signature=execution_signature,
        ):
            raise RuntimeError(f"post-run score cache audit failed: {manifest_path}")
        cached_or_complete += 1
    return {
        "schema_version": "stage-vuspr-score-status-v1",
        "status": "scores_complete",
        "expected_units": total,
        "cached_or_complete_units": cached_or_complete,
        "plan_fingerprint": plan["plan_fingerprint"],
        "completed_at": utc_now(),
    }


def evaluate_caches_parent(
    repo: Path,
    protocol_path: Path,
    result_root: Path,
    metrics_root: Path,
    workers: int,
    remove_cache: bool,
) -> dict[str, Any]:
    """Consume transferred score caches without touching CUDA."""

    if workers < 1:
        raise ValueError("--workers must be positive")
    plan = ensure_plan(repo, protocol_path, metrics_root, result_root)
    if plan["phase"] != "order_o1":
        raise RuntimeError(
            "evaluate-caches is restricted to the isolated order_o1 phase"
        )
    available: list[tuple[str, str, str, int, str]] = []
    completed = 0
    missing = 0
    for task in _iter_tasks(plan):
        track, dataset, file_name, seed, execution_signature = task
        target = unit_path(
            result_root, track, dataset, execution_signature, seed, file_name
        )
        if target.is_file():
            completed += 1
            continue
        manifest_path, array_path = score_cache_paths(
            result_root,
            track,
            dataset,
            execution_signature,
            seed,
            file_name,
        )
        if manifest_path.is_file() and array_path.is_file():
            available.append(task)
        else:
            missing += 1

    def evaluate(task: tuple[str, str, str, int, str]) -> Path:
        track, dataset, file_name, seed, execution_signature = task
        return evaluate_score_cache(
            result_root,
            metrics_root,
            plan,
            track=track,
            dataset=dataset,
            file_name=file_name,
            seed=seed,
            execution_signature=execution_signature,
            remove_cache=remove_cache,
        )

    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(available)))) as pool:
        futures = [pool.submit(evaluate, task) for task in available]
        for future in as_completed(futures):
            future.result()
            completed += 1
    total = int(plan["expected_units"])
    return {
        "schema_version": "stage-vuspr-cpu-eval-status-v1",
        "status": "complete" if completed == total else "partial",
        "expected_units": total,
        "completed_units": completed,
        "missing_score_caches": missing,
        "evaluated_this_call": len(available),
        "plan_fingerprint": plan["plan_fingerprint"],
        "completed_at": utc_now(),
    }


def _selection_statistics(metric_means: Mapping[str, float]) -> tuple[float, float]:
    all6 = sum(float(metric_means[metric]) for metric in METRICS) / len(METRICS)
    return all6, float(metric_means[SELECTION_METRIC])


def _aggregate_metric_rows(
    records: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        key = tuple(record[field] for field in group_fields)
        grouped.setdefault(key, []).append(record)
    rows: list[dict[str, Any]] = []
    for key, members in grouped.items():
        metric_means = {
            metric: sum(float(member[metric]) for member in members) / len(members)
            for metric in METRICS
        }
        all6, selection_score = _selection_statistics(metric_means)
        row = {field: value for field, value in zip(group_fields, key)}
        row.update(
            {
                **metric_means,
                "all6": all6,
                "selection_score": selection_score,
                "units": sum(int(member.get("units", 1)) for member in members),
                "series": len(
                    {
                        (member.get("dataset"), member.get("file"))
                        for member in members
                        if member.get("file") is not None
                    }
                ),
                "memory_rows": sum(float(member["memory_rows"]) for member in members)
                / len(members),
                "memory_ratio": sum(float(member["memory_ratio"]) for member in members)
                / len(members),
            }
        )
        rows.append(row)
    return rows


def _combine_seed_rows(
    seed_rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    rows = _aggregate_metric_rows(seed_rows, group_fields)
    seeds_by_key: dict[tuple[Any, ...], set[int]] = {}
    for row in seed_rows:
        key = tuple(row[field] for field in group_fields)
        seeds_by_key.setdefault(key, set()).add(int(row["seed"]))
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        row["seeds"] = len(seeds_by_key[key])
        row["series"] = max(
            int(item.get("series", 0))
            for item in seed_rows
            if tuple(item[field] for field in group_fields) == key
        )
    return rows


def _head_id(plan: Mapping[str, Any], final_split: int, top_k: int) -> str:
    heads = [
        (int(split), int(k))
        for split in plan["final_gb_min_splits"]
        for k in plan["top_ks"]
    ]
    return f"H{heads.index((int(final_split), int(top_k))):02d}"


def _ranking_key(row: Mapping[str, Any], *, candidate: bool) -> tuple[Any, ...]:
    if candidate:
        return (
            -float(row["selection_score"]),
            str(row["training_candidate_id"]),
            str(row["head_id"]),
        )
    return (
        -float(row["selection_score"]),
        str(row["head_id"]),
    )


def _encoder_e1_selection(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one global encoder using only each dataset's frozen reference head."""

    candidate_encoder = {
        str(item["id"]): str(item["resolved_parameters"]["encoder_type"])
        for item in plan["training_candidates"]
    }
    reference_heads = plan["metadata"]["reference_selected_heads"]
    subset_scores: dict[str, dict[str, float]] = {
        encoder_type: {} for encoder_type in ENCODER_TYPES
    }
    for track, datasets in plan["targets"].items():
        for dataset in datasets:
            subset = f"{track}/{dataset}"
            reference = reference_heads[subset]
            for encoder_type in ENCODER_TYPES:
                candidates = [
                    candidate_id
                    for candidate_id in plan["candidates_by_subset"][subset]
                    if candidate_encoder[candidate_id] == encoder_type
                ]
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"{subset} must contain exactly one {encoder_type} candidate"
                    )
                matches = [
                    row
                    for row in rows
                    if row["track"] == track
                    and row["dataset"] == dataset
                    and row["training_candidate_id"] == candidates[0]
                    and int(row["final_gb_min_split"])
                    == int(reference["final_gb_min_split"])
                    and int(row["top_k"]) == int(reference["top_k"])
                ]
                if len(matches) != 1 or int(matches[0]["seeds"]) != 1:
                    raise RuntimeError(
                        f"{subset} {encoder_type} lacks its complete frozen-head row"
                    )
                subset_scores[encoder_type][subset] = float(
                    matches[0]["selection_score"]
                )

    architecture_scores = []
    subset_count = sum(len(items) for items in plan["targets"].values())
    for encoder_type in ENCODER_TYPES:
        scores = subset_scores[encoder_type]
        if len(scores) != subset_count:
            raise RuntimeError(
                f"{encoder_type} lacks complete ten-dataset encoder coverage"
            )
        architecture_scores.append(
            {
                "encoder_type": encoder_type,
                "mean_dataset_macro_vus_pr": sum(scores.values()) / len(scores),
                "dataset_macro_vus_pr": scores,
            }
        )
    architecture_scores.sort(
        key=lambda item: (
            -float(item["mean_dataset_macro_vus_pr"]),
            str(item["encoder_type"]),
        )
    )
    selected_encoder = str(architecture_scores[0]["encoder_type"])

    selected_heads: dict[str, dict[str, Any]] = {}
    for track, datasets in plan["targets"].items():
        for dataset in datasets:
            subset = f"{track}/{dataset}"
            selected_candidate = next(
                candidate_id
                for candidate_id in plan["candidates_by_subset"][subset]
                if candidate_encoder[candidate_id] == selected_encoder
            )
            candidates = [
                row
                for row in rows
                if row["track"] == track
                and row["dataset"] == dataset
                and row["training_candidate_id"] == selected_candidate
            ]
            if len(candidates) != int(plan["variants_per_unit"]):
                raise RuntimeError(
                    f"{subset} selected encoder lacks its complete head grid"
                )
            candidates.sort(key=lambda row: _ranking_key(row, candidate=False))
            winner = candidates[0]
            selected_heads[subset] = {
                "training_candidate_id": selected_candidate,
                "head_id": str(winner["head_id"]),
                "final_gb_min_split": int(winner["final_gb_min_split"]),
                "top_k": int(winner["top_k"]),
                "selection_score": float(winner["selection_score"]),
            }
    return {
        "global_encoder_rule": (
            "maximize the unweighted mean of ten dataset-level official Tuning "
            "macro VUS-PR values at the previously frozen dataset-specific heads; "
            "canonical encoder_type breaks exact ties"
        ),
        "architecture_scores": architecture_scores,
        "selected_encoder_type_global": selected_encoder,
        "selected_heads_for_global_encoder": selected_heads,
    }


def _rank_groups(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    *,
    candidate: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        ranked = sorted(grouped[key], key=lambda row: _ranking_key(row, candidate=candidate))
        for rank, row in enumerate(ranked, 1):
            row["rank"] = rank
            output.append(row)
    return output


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=path.name + ".tmp.",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_prior_summary(
    plan: Mapping[str, Any], protocol_path: Path
) -> dict[str, Any]:
    reference = plan.get("prior_summary")
    if not isinstance(reference, Mapping):
        raise RuntimeError(f"{plan.get('phase')} requires a prior summary reference")
    path = Path(str(reference["path"]))
    if not path.is_absolute():
        path = (protocol_path.parent / path).resolve()
    if not path.is_file() or sha256_file(path) != str(reference["sha256"]).lower():
        raise RuntimeError(f"prior summary is missing or has the wrong SHA-256: {path}")
    summary = load_json(path)
    expected_phase = "stage1a" if plan["phase"] == "stage1b" else "stage1b"
    if not all(
        (
            summary.get("schema_version") == SUMMARY_SCHEMA,
            summary.get("phase") == expected_phase,
            summary.get("selection_split") == SELECTION_SPLIT,
            summary.get("eval_feedback") is False,
            isinstance(summary.get("selection_series_rows"), list),
            summary.get("completed_units") == summary.get("expected_units"),
            summary.get("completed_logical_units")
            == summary.get("expected_logical_units"),
        )
    ):
        raise RuntimeError("prior summary is not a complete compatible phase summary")
    prior_plan_path = path.parent / PLAN_NAME
    if not prior_plan_path.is_file():
        raise RuntimeError(f"prior summary lacks its frozen plan: {prior_plan_path}")
    prior_plan = load_json(prior_plan_path)
    prior_plan_fingerprint = prior_plan.get("plan_fingerprint")
    if (
        not isinstance(prior_plan_fingerprint, str)
        or not re.fullmatch(r"[0-9a-f]{64}", prior_plan_fingerprint)
        or fingerprint(_stable_plan_from_frozen(prior_plan))
        != prior_plan_fingerprint
        or summary.get("plan_fingerprint") != prior_plan_fingerprint
    ):
        raise RuntimeError("prior summary/plan fingerprint validation failed")
    prior_selection = summary.get("selection")
    if not isinstance(prior_selection, Mapping):
        raise RuntimeError("prior summary lacks its frozen selection payload")
    selection_fingerprint = prior_selection.get("selection_fingerprint")
    stable_selection = {
        str(key): value
        for key, value in prior_selection.items()
        if key not in {"selection_fingerprint", "created_at"}
    }
    if (
        not isinstance(selection_fingerprint, str)
        or fingerprint(stable_selection) != selection_fingerprint
        or prior_selection.get("plan_fingerprint") != prior_plan_fingerprint
    ):
        raise RuntimeError("prior selection fingerprint validation failed")
    if fingerprint(prior_plan.get("training_candidates")) != fingerprint(
        plan.get("training_candidates")
    ):
        raise RuntimeError("training candidates differ from the prior frozen plan")
    lineage_fields = (
        "files",
        "data_sha256",
        "official_tuning_manifest_sha256",
        "source_sha256",
        "official_evaluator",
        "targets",
    )
    if "files" in plan:
        for field in lineage_fields:
            if fingerprint(prior_plan.get(field)) != fingerprint(plan.get(field)):
                raise RuntimeError(f"prior/current lineage differs at {field}")
    if plan["phase"] == "stage1b" and plan.get("training_shortlists") != prior_selection.get(
        "training_shortlists"
    ):
        raise RuntimeError(
            "Stage1B training_shortlists differ from the frozen Stage1A selection"
        )
    if plan["phase"] == "stage2" and plan.get("training_winners") != prior_selection.get(
        "training_winners"
    ):
        raise RuntimeError(
            "Stage2 training_winners differ from the frozen Stage1B selection"
        )
    return summary


def _logical_series_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["track"],
        row["dataset"],
        row["file"],
        int(row["seed"]),
        row["training_candidate_id"],
        int(row["final_gb_min_split"]),
        int(row["top_k"]),
    )


def _validate_series_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not _finite_metric_mapping(
            {metric: row.get(metric) for metric in METRICS}
        ):
            raise RuntimeError("summary series row has invalid metrics")
        key = _logical_series_key(row)
        if key in seen:
            raise RuntimeError(f"duplicate logical series row: {key}")
        seen.add(key)


def _track_execution_signature(
    plan: Mapping[str, Any], track: str, candidate_id: str
) -> str:
    return fingerprint(
        [
            (
                dataset,
                plan["subset_execution_signatures"][f"{track}/{dataset}"][
                    candidate_id
                ],
            )
            for dataset in plan["targets"][track]
        ]
    )


def _take_unique_training_candidates(
    ranked_rows: Sequence[Mapping[str, Any]],
    signature_for_candidate,
    count: int,
) -> list[str]:
    selected: list[str] = []
    seen_signatures: set[str] = set()
    for row in ranked_rows:
        candidate_id = str(row["training_candidate_id"])
        signature = str(signature_for_candidate(candidate_id))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(candidate_id)
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"ranking contains fewer than {count} execution-distinct candidates"
    )


def _fixed_head_verification(
    current_rows: Sequence[Mapping[str, Any]],
    prior_rows: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    prior = {_logical_series_key(row): row for row in prior_rows}
    compared = 0
    maximum_delta = 0.0
    for row in current_rows:
        if (int(row["final_gb_min_split"]), int(row["top_k"])) != (4, 3):
            continue
        key = _logical_series_key(row)
        reference = prior.get(key)
        if reference is None:
            raise RuntimeError(f"fixed-head verification lacks prior row: {key}")
        for metric in METRICS:
            delta = abs(float(row[metric]) - float(reference[metric]))
            maximum_delta = max(maximum_delta, delta)
            if delta > tolerance:
                raise RuntimeError(
                    f"Stage2 fixed-head mismatch for {key} {metric}: {delta} > {tolerance}"
                )
        compared += 1
    return {
        "enabled": True,
        "tolerance": float(tolerance),
        "compared_series_seed_rows": compared,
        "maximum_metric_delta": maximum_delta,
        "passed": True,
    }


def summarize_results(
    repo: Path,
    protocol_path: Path,
    result_root: Path,
    metrics_root: Path,
) -> dict[str, Any]:
    plan = ensure_plan(repo, protocol_path, metrics_root, result_root)
    _assert_no_unplanned_unit_artifacts(result_root, plan)
    current_series_rows: list[dict[str, Any]] = []
    seen_units = 0
    seen_logical_units = 0
    for task in _iter_tasks(plan):
        track, dataset, file_name, seed, execution_signature = task
        group = execution_group(
            plan, track, dataset, file_name, execution_signature
        )
        path = unit_path(
            result_root, track, dataset, execution_signature, seed, file_name
        )
        if not path.is_file():
            raise RuntimeError(f"cannot summarize incomplete search; missing {path}")
        item = load_json(path)
        if not valid_unit(
            item,
            plan=plan,
            track=track,
            dataset=dataset,
            file_name=file_name,
            seed=seed,
            group=group,
        ):
            raise RuntimeError(f"invalid v2 unit: {path}")
        seen_units += 1
        seen_logical_units += len(group["candidate_ids"])
        for variant in item["variants"]:
            candidate_ids_for_variant = (
                [str(variant["geometry_candidate_id"])]
                if plan["phase"] == "geometry_g2"
                else list(group["candidate_ids"])
            )
            for candidate_id in candidate_ids_for_variant:
                current_series_rows.append(
                    {
                    "file": file_name,
                    "track": track,
                    "dataset": dataset,
                    "seed": int(seed),
                    "training_candidate_id": candidate_id,
                    "training_config_fingerprint": group["config_fingerprints"][
                        candidate_id
                    ],
                    "execution_signature": execution_signature,
                    "final_gb_min_split": int(variant["final_gb_min_split"]),
                    "top_k": int(variant["top_k"]),
                    **{metric: float(variant["metrics"][metric]) for metric in METRICS},
                    "memory_rows": int(variant["memory_rows"]),
                    "memory_ratio": float(variant["memory_ratio"]),
                    }
                )

    if seen_units != int(plan["expected_units"]):
        raise RuntimeError("execution unit count differs from the frozen plan")
    if seen_logical_units != int(plan["expected_logical_units"]):
        raise RuntimeError("logical unit count differs from the frozen plan")
    _validate_series_rows(current_series_rows)

    phase = str(plan["phase"])
    prior_summary: dict[str, Any] | None = None
    selection_series_rows = list(current_series_rows)
    if phase == "stage1b":
        prior_summary = _load_prior_summary(plan, protocol_path)
        allowed_by_subset = {
            subset: set(candidate_ids)
            for subset, candidate_ids in plan["candidates_by_subset"].items()
        }
        prior_rows = [
            dict(row)
            for row in prior_summary["selection_series_rows"]
            if row["training_candidate_id"]
            in allowed_by_subset[f"{row['track']}/{row['dataset']}"]
        ]
        selection_series_rows = [*prior_rows, *current_series_rows]
        _validate_series_rows(selection_series_rows)
    elif phase == "stage2":
        prior_summary = _load_prior_summary(plan, protocol_path)

    subset_seed_fields = (
        "track",
        "dataset",
        "training_candidate_id",
        "training_config_fingerprint",
        "final_gb_min_split",
        "top_k",
        "seed",
    )
    subset_fields = subset_seed_fields[:-1]
    seed_rows = _aggregate_metric_rows(selection_series_rows, subset_seed_fields)
    rows = _combine_seed_rows(seed_rows, subset_fields)
    for row in rows:
        row["head_id"] = _head_id(
            plan, int(row["final_gb_min_split"]), int(row["top_k"])
        )

    if phase in {
        "stage1a",
        "stage1b",
        "encoder_e1",
        "geometry_d1",
        "geometry_g2",
        "order_o1",
    }:
        rows = _rank_groups(rows, ("track", "dataset"), candidate=True)
        track_seed_fields = (
            "track",
            "training_candidate_id",
            "training_config_fingerprint",
            "final_gb_min_split",
            "top_k",
            "seed",
        )
        track_fields = track_seed_fields[:-1]
        track_seed_rows = _aggregate_metric_rows(
            selection_series_rows, track_seed_fields
        )
        track_rows = _combine_seed_rows(track_seed_rows, track_fields)
        for row in track_rows:
            row["dataset"] = "__track__"
            row["head_id"] = _head_id(
                plan, int(row["final_gb_min_split"]), int(row["top_k"])
            )
        track_rows = _rank_groups(track_rows, ("track",), candidate=True)
    else:
        rows = _rank_groups(
            rows,
            ("track", "dataset", "training_candidate_id"),
            candidate=False,
        )
        track_seed_fields = (
            "track",
            "final_gb_min_split",
            "top_k",
            "seed",
        )
        track_fields = track_seed_fields[:-1]
        track_selection_series = [
            row
            for row in selection_series_rows
            if row["training_candidate_id"]
            == plan["training_winners"][f"{row['track']}/{row['dataset']}"]
        ]
        track_seed_rows = _aggregate_metric_rows(track_selection_series, track_seed_fields)
        track_rows = _combine_seed_rows(track_seed_rows, track_fields)
        for row in track_rows:
            row.update(
                dataset="__track__",
                training_candidate_id="__frozen_subset_winners__",
                training_config_fingerprint="mixed_by_subset",
            )
            row["head_id"] = _head_id(
                plan, int(row["final_gb_min_split"]), int(row["top_k"])
            )
        track_rows = _rank_groups(track_rows, ("track",), candidate=False)

    selection_payload: dict[str, Any] = {
        "schema_version": "stage-vuspr-search-selection-v1",
        "phase": phase,
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "score": EXPECTED_SELECTION_SCORE,
        "tie_breakers": list(SELECTION_TIE_BREAKERS),
        "plan_fingerprint": plan["plan_fingerprint"],
    }
    shortlist_size = int(plan["shortlist_size"])
    if phase in {"stage1a", "geometry_d1", "geometry_g2", "order_o1"}:
        training_shortlists: dict[str, list[str]] = {}
        for track, datasets in plan["targets"].items():
            for dataset in datasets:
                subset = f"{track}/{dataset}"
                ranked = [
                    row
                    for row in rows
                    if row["track"] == track
                    and row["dataset"] == dataset
                ]
                if phase == "geometry_g2":
                    selected: list[str] = []
                    for row in ranked:
                        candidate_id = str(row["training_candidate_id"])
                        if candidate_id not in selected:
                            selected.append(candidate_id)
                        if len(selected) == shortlist_size:
                            break
                    if len(selected) != shortlist_size:
                        raise RuntimeError(
                            f"{subset} lacks complete geometry candidates"
                        )
                    training_shortlists[subset] = selected
                else:
                    training_shortlists[subset] = (
                        _take_unique_training_candidates(
                            ranked,
                            lambda candidate_id, s=subset: plan[
                                "subset_execution_signatures"
                            ][s][candidate_id],
                            shortlist_size,
                        )
                    )
        selection_payload["training_shortlists"] = training_shortlists
        if phase == "geometry_d1":
            selection_payload["diagnostic_only"] = True
            selection_payload["diagnostic_axes"] = [
                "robust_level_scale",
                "direct_timestamp_pairwise",
                "coarser_refreshed_intermediate_geometry",
            ]
        elif phase == "geometry_g2":
            selection_payload["diagnostic_only"] = True
            selection_payload["diagnostic_axes"] = [
                "independent_timestamp_support",
                "support_calibrated_local_radius",
                "combined_support_and_radius",
                "joint_state_transition_geometry",
            ]
        elif phase == "order_o1":
            selection_payload["diagnostic_only"] = True
            selection_payload["diagnostic_axes"] = [
                "moments_control",
                "ordered_temporal_pyramid_residual",
                "directional_lag_relation_residual",
                "hybrid_residual_with_reliability_gated_transition_alignment",
            ]
    elif phase == "encoder_e1":
        selection_payload.update(
            **_encoder_e1_selection(rows, plan),
            seeds=[2026],
            preliminary_gate=True,
        )
    elif phase == "stage1b":
        expected_seeds = 3
        shortlist_map = plan["training_shortlists"]
        training_winners: dict[str, str] = {}
        for track, datasets in plan["targets"].items():
            for dataset in datasets:
                subset = f"{track}/{dataset}"
                allowed = set(shortlist_map[subset])
                ranked = [
                    row
                    for row in rows
                    if row["track"] == track
                    and row["dataset"] == dataset
                    and row["training_candidate_id"] in allowed
                ]
                if not ranked or any(
                    int(row["seeds"]) != expected_seeds for row in ranked
                ):
                    raise RuntimeError(f"{subset} shortlist lacks three complete seeds")
                ranked.sort(key=lambda row: _ranking_key(row, candidate=True))
                training_winners[subset] = str(
                    ranked[0]["training_candidate_id"]
                )
        selection_payload.update(
            training_winners=training_winners,
            seeds=[2026, 2027, 2028],
        )
    else:
        if any(int(row["seeds"]) != 3 for row in rows) or any(
            int(row["seeds"]) != 3 for row in track_rows
        ):
            raise RuntimeError("Stage2 head selection requires all three seeds")
        selected_heads: dict[str, dict[str, Any]] = {}
        for track, datasets in plan["targets"].items():
            for dataset in datasets:
                subset = f"{track}/{dataset}"
                ranked = [
                    row
                    for row in rows
                    if row["track"] == track
                    and row["dataset"] == dataset
                    and row["training_candidate_id"]
                    == plan["training_winners"][subset]
                ]
                ranked.sort(key=lambda row: _ranking_key(row, candidate=False))
                winner = ranked[0]
                selected_heads[subset] = {
                    "head_id": winner["head_id"],
                    "final_gb_min_split": int(winner["final_gb_min_split"]),
                    "top_k": int(winner["top_k"]),
                }
        selection_payload.update(
            training_winners=plan["training_winners"],
            selected_heads=selected_heads,
            seeds=[2026, 2027, 2028],
        )
        if bool(plan.get("metadata", {}).get("verify_stage2_fixed_head", False)):
            tolerance = float(
                plan.get("metadata", {}).get(
                    "fixed_head_verification_tolerance", 1e-10
                )
            )
            selection_payload["fixed_head_verification"] = _fixed_head_verification(
                current_series_rows,
                prior_summary["selection_series_rows"],
                tolerance=tolerance,
            )

    frozen_selection = dict(selection_payload)
    selection_payload["selection_fingerprint"] = fingerprint(frozen_selection)
    selection_payload["created_at"] = utc_now()

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "phase": phase,
        "selection_split": SELECTION_SPLIT,
        "eval_feedback": False,
        "plan_fingerprint": plan["plan_fingerprint"],
        "protocol_fingerprint": plan["protocol_fingerprint"],
        "selection_score": EXPECTED_SELECTION_SCORE,
        "tie_breakers": list(SELECTION_TIE_BREAKERS),
        "completed_units": seen_units,
        "expected_units": plan["expected_units"],
        "completed_logical_units": seen_logical_units,
        "expected_logical_units": plan["expected_logical_units"],
        "current_series_rows": current_series_rows,
        "selection_series_rows": selection_series_rows,
        "seed_rows": seed_rows,
        "rows": rows,
        "track_rows": track_rows,
        "selection": selection_payload,
        "created_at": utc_now(),
    }
    atomic_json(result_root / SUMMARY_JSON_NAME, summary)
    columns = [
        "track",
        "dataset",
        "rank",
        "head_id",
        "training_candidate_id",
        "training_config_fingerprint",
        "final_gb_min_split",
        "top_k",
        "series",
        "seeds",
        "units",
        *METRICS,
        "all6",
        "selection_score",
        "memory_rows",
        "memory_ratio",
    ]
    _atomic_csv(result_root / SUMMARY_CSV_NAME, rows, columns)
    atomic_json(result_root / SELECTION_JSON_NAME, selection_payload)
    return summary


@contextmanager
def controller_singleton(result_root: Path):
    result_root.mkdir(parents=True, exist_ok=True)
    lock_path = result_root / CONTROLLER_LOCK_NAME
    handle = lock_path.open("a+", encoding="utf-8")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(" ")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f"another v2 controller holds {lock_path}") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"another v2 controller holds {lock_path}") from exc
        locked = True
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "acquired_at": utc_now()}) + "\n")
        handle.flush()
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--metrics-root", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="validate and freeze the explicit Tuning plan")
    run = subparsers.add_parser("run", help="run or resume all planned Tuning units")
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--gpus", default="0,1")
    run.add_argument("--workers-per-gpu", type=int, default=3)
    run_scores = subparsers.add_parser(
        "run-scores",
        help="run GPU training/scoring only and emit compact deferred caches",
    )
    run_scores.add_argument("--python", type=Path, default=Path(sys.executable))
    run_scores.add_argument("--gpus", default="0,1")
    run_scores.add_argument("--workers-per-gpu", type=int, default=3)
    evaluate_caches = subparsers.add_parser(
        "evaluate-caches",
        help="evaluate transferred score caches on CPU without CUDA",
    )
    evaluate_caches.add_argument("--workers", type=int, default=4)
    evaluate_caches.add_argument(
        "--keep-cache",
        action="store_true",
        help="retain verified transient score caches after final unit creation",
    )
    subparsers.add_parser("summarize", help="strictly aggregate all completed units")
    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--track", required=True, choices=("U", "M"))
    worker.add_argument("--dataset", required=True)
    worker.add_argument("--file", required=True)
    worker.add_argument("--seed", required=True, type=int)
    worker.add_argument("--execution-signature", required=True)
    worker.add_argument("--physical-gpu", required=True)
    score_worker = subparsers.add_parser("_score_worker", help=argparse.SUPPRESS)
    score_worker.add_argument("--track", required=True, choices=("U", "M"))
    score_worker.add_argument("--dataset", required=True)
    score_worker.add_argument("--file", required=True)
    score_worker.add_argument("--seed", required=True, type=int)
    score_worker.add_argument("--execution-signature", required=True)
    score_worker.add_argument("--physical-gpu", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = args.repo.resolve()
    protocol_path = args.protocol.resolve()
    result_root = args.result_root.resolve()
    metrics_root = args.metrics_root.resolve()
    if args.command in {"_worker", "_score_worker"}:
        plan = load_frozen_plan_for_worker(
            repo,
            protocol_path,
            metrics_root,
            result_root,
            track=args.track,
            dataset=args.dataset,
            file_name=args.file,
            seed=int(args.seed),
            execution_signature=args.execution_signature,
        )
        target = execute_unit(
            repo,
            result_root,
            metrics_root,
            plan,
            track=args.track,
            dataset=args.dataset,
            file_name=args.file,
            seed=int(args.seed),
            execution_signature=args.execution_signature,
            physical_gpu=args.physical_gpu,
            defer_metrics=args.command == "_score_worker",
        )
        print(
            json.dumps(
                {
                    (
                        "score_cache"
                        if args.command == "_score_worker"
                        else "unit"
                    ): str(target)
                },
                indent=2,
            )
        )
        return 0

    with controller_singleton(result_root):
        if args.command == "plan":
            plan = ensure_plan(repo, protocol_path, metrics_root, result_root)
            print(
                json.dumps(
                    {
                        "plan_fingerprint": plan["plan_fingerprint"],
                        "series_count": plan["series_count"],
                        "expected_units": plan["expected_units"],
                        "variants_per_unit": plan["variants_per_unit"],
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "run":
            python = Path(os.path.abspath(args.python))
            status = run_parent(
                repo,
                protocol_path,
                result_root,
                metrics_root,
                python,
                _parse_gpus(args.gpus),
                int(args.workers_per_gpu),
            )
            print(json.dumps(status, indent=2))
            return 0
        if args.command == "run-scores":
            python = Path(os.path.abspath(args.python))
            status = run_scores_parent(
                repo,
                protocol_path,
                result_root,
                metrics_root,
                python,
                _parse_gpus(args.gpus),
                int(args.workers_per_gpu),
            )
            print(json.dumps(status, indent=2))
            return 0
        if args.command == "evaluate-caches":
            status = evaluate_caches_parent(
                repo,
                protocol_path,
                result_root,
                metrics_root,
                int(args.workers),
                not bool(args.keep_cache),
            )
            print(json.dumps(status, indent=2))
            return 0
        if args.command == "summarize":
            summary = summarize_results(
                repo, protocol_path, result_root, metrics_root
            )
            print(
                json.dumps(
                    {
                        "completed_units": summary["completed_units"],
                        "rows": len(summary["rows"]),
                        "summary_json": str(result_root / SUMMARY_JSON_NAME),
                        "summary_csv": str(result_root / SUMMARY_CSV_NAME),
                    },
                    indent=2,
                )
            )
            return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
