from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from common import (  # noqa: E402
    METRICS,
    atomic_json,
    dataset_name,
    official_metrics,
    parse_train_index,
    seed_everything,
)
from extensions.aaai_recent.models import (  # noqa: E402
    DenseProjection,
    NoiseEvaluationMLP,
    NoiseEvaluationResMLP,
    diverse_gaussian_noise,
)


METHODS = ("DPAD_AAAI24", "DNE_AAAI25")
TARGETS = {
    "U": ("UCR", "Exathlon", "MSL", "SED", "TODS"),
    "M": ("CATSv2", "GHL", "LTDB", "SVDB", "TAO"),
}


def make_windows(values: np.ndarray, length: int) -> np.ndarray:
    length = min(max(int(length), 1), len(values))
    view = np.lib.stride_tricks.sliding_window_view(values, length, axis=0)
    return np.ascontiguousarray(view.reshape(len(values) - length + 1, -1), dtype=np.float32)


def to_points(scores: np.ndarray, length: int, total: int) -> np.ndarray:
    difference_sum = np.zeros(total + 1, dtype=np.float64)
    difference_count = np.zeros(total + 1, dtype=np.float64)
    starts = np.arange(len(scores), dtype=np.int64)
    np.add.at(difference_sum, starts, scores)
    np.add.at(difference_sum, starts + length, -scores)
    np.add.at(difference_count, starts, 1.0)
    np.add.at(difference_count, starts + length, -1.0)
    sums = np.cumsum(difference_sum[:-1])
    counts = np.cumsum(difference_count[:-1])
    return sums / np.maximum(counts, 1.0)


def choose_window(repo: Path, normalized: np.ndarray, boundary: int, track: str) -> int:
    if track == "M":
        return 1
    tsb_root = repo / "external" / "PAI" / "third_party" / "TSB-AD"
    sys.path.insert(0, str(tsb_root))
    from TSB_AD.utils.slidingWindows import find_length_rank

    return min(max(int(find_length_rank(normalized[:boundary], rank=2)), 1), boundary)


def deterministic_subset(values: np.ndarray, maximum: int | None) -> np.ndarray:
    if maximum is None or len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, num=maximum, dtype=np.int64)
    return values[indices]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_dpad(
    train_values: np.ndarray,
    query_values: np.ndarray,
    device: torch.device,
    epochs: int,
    gamma: float,
    regularization: float,
    learning_rate: float,
    neighbors: int,
    pairwise_reduction: str,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
    optimizer_weight_decay: float,
    optimizer_amsgrad: bool,
    execution_profile: str,
) -> tuple[np.ndarray, dict]:
    input_dim = int(train_values.shape[1])
    hidden = min(256, max(64, input_dim * 2))
    output_dim = min(32, max(2, input_dim // 2))
    model = DenseProjection(input_dim, (hidden, max(32, hidden // 2)), output_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_eps,
        weight_decay=optimizer_weight_decay,
        amsgrad=optimizer_amsgrad,
    )
    training = torch.from_numpy(train_values).to(device)
    model.train()
    final_loss = math.nan
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(training)
        squared_distances = torch.cdist(embeddings, embeddings, p=2).square()
        weights = torch.exp(-gamma * squared_distances.detach())
        weighted_distances = squared_distances * weights
        if pairwise_reduction == "mean":
            density_loss = weighted_distances.mean()
        elif pairwise_reduction == "sum":
            density_loss = weighted_distances.sum()
        else:
            raise ValueError(f"unsupported DPAD pairwise reduction: {pairwise_reduction}")
        loss = density_loss + regularization * model.anti_collapse_penalty()
        if not torch.isfinite(loss):
            raise ValueError("DPAD produced a non-finite training loss")
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        reference = model(training)
        if neighbors < 1 or len(reference) < neighbors:
            raise ValueError(
                f"DPAD requires at least k={neighbors} reference windows; received {len(reference)}"
            )
        k = neighbors
        chunks = []
        for start in range(0, len(query_values), 1024):
            query = torch.from_numpy(query_values[start : start + 1024]).to(device)
            distances = torch.cdist(model(query), reference, p=2)
            chunks.append(torch.topk(distances, k=k, largest=False, dim=1).values.mean(dim=1).cpu())
        scores = torch.cat(chunks).numpy().astype(np.float64)
    details = {
        "network": "bias-free ReLU MLP",
        "widths": [input_dim, hidden, max(32, hidden // 2), output_dim],
        "epochs": epochs,
        "gamma": gamma,
        "lambda": regularization,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "optimizer_betas": list(optimizer_betas),
        "optimizer_eps": optimizer_eps,
        "optimizer_weight_decay": optimizer_weight_decay,
        "optimizer_amsgrad": optimizer_amsgrad,
        "k": neighbors,
        "pairwise_reduction": pairwise_reduction,
        "final_training_loss": final_loss,
        "execution_profile": execution_profile,
    }
    return scores, details


def train_dne(
    train_values: np.ndarray,
    query_values: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    architecture: str,
    learning_rate: float,
    weight_decay: float,
    optimizer_betas: tuple[float, float],
    optimizer_eps: float,
    optimizer_amsgrad: bool,
    sigma_max: float,
    noise_level_parts: int,
    noise_ratios: tuple[float, ...],
    learning_rate_decay_epoch: int,
    learning_rate_decay_factor: float,
    max_optimizer_steps: int,
    execution_profile: str,
) -> tuple[np.ndarray, dict]:
    input_dim = int(train_values.shape[1])
    if architecture == "resmlp":
        model: nn.Module = NoiseEvaluationResMLP(input_dim)
    else:
        model = NoiseEvaluationMLP(input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=optimizer_betas,
        eps=optimizer_eps,
        weight_decay=weight_decay,
        amsgrad=optimizer_amsgrad,
    )
    dataset = TensorDataset(torch.from_numpy(train_values))
    loader_generator = torch.Generator().manual_seed(torch.initial_seed())
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )
    planned_steps = epochs * math.ceil(len(dataset) / min(batch_size, len(dataset)))
    if planned_steps > max_optimizer_steps:
        raise ValueError(
            f"DNE planned optimizer steps {planned_steps} exceed frozen cap "
            f"{max_optimizer_steps}"
        )
    model.train()
    final_loss = math.nan
    optimizer_steps = 0
    for epoch in range(epochs):
        if epoch == learning_rate_decay_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= learning_rate_decay_factor
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(clean), torch.zeros_like(clean))
            for ratio in noise_ratios:
                noise = diverse_gaussian_noise(
                    clean, sigma_max, noise_level_parts, ratio
                )
                loss = loss + nn.functional.mse_loss(model(clean + noise), noise.abs())
            if not torch.isfinite(loss):
                raise ValueError("DNE produced a non-finite training loss")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            final_loss = float(loss.detach().cpu())

    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(query_values), 2048):
            query = torch.from_numpy(query_values[start : start + 2048]).to(device)
            chunks.append(model(query).amax(dim=1).cpu())
    scores = torch.cat(chunks).numpy().astype(np.float64)
    details = {
        "architecture": architecture,
        "hidden_width": 64 if input_dim <= 64 else 256,
        "epochs": epochs,
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer_betas": list(optimizer_betas),
        "optimizer_eps": optimizer_eps,
        "optimizer_amsgrad": optimizer_amsgrad,
        "learning_rate_decay_epoch": learning_rate_decay_epoch,
        "learning_rate_decay_factor": learning_rate_decay_factor,
        "declared_max_optimizer_steps_per_unit": max_optimizer_steps,
        "optimizer_steps": optimizer_steps,
        "noise_type": "Gaussian",
        "sigma_max": sigma_max,
        "noise_level_parts": noise_level_parts,
        "noise_ratios": list(noise_ratios),
        "noisy_instances_per_clean_instance": len(noise_ratios),
        "score_aggregation": "maximum model output",
        "final_training_loss": final_loss,
        "execution_profile": execution_profile,
    }
    return scores, details


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_keys(mapping: Mapping[str, Any], keys: set[str], name: str) -> None:
    missing = sorted(keys.difference(mapping))
    if missing:
        raise RuntimeError(f"{name} is missing required keys: {missing}")


def _strict_complete_record(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        metrics = item.get("metrics")
        if item.get("error") not in (None, "") or not isinstance(metrics, Mapping):
            return False
        if any(item.get(key) != value for key, value in expected.items()):
            return False
        return all(
            name in metrics
            and not isinstance(metrics[name], bool)
            and isinstance(metrics[name], (int, float))
            and math.isfinite(float(metrics[name]))
            for name in METRICS
        )
    except Exception:
        return False


def _formal_device(args: argparse.Namespace) -> tuple[torch.device, str]:
    if args.profile == "smoke":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), "unconstrained"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_parts = [part.strip() for part in visible.split(",") if part.strip()]
    if len(visible_parts) != 1 or visible_parts[0] != args.require_physical_gpu:
        raise RuntimeError(
            "formal execution requires CUDA_VISIBLE_DEVICES to contain exactly "
            f"the requested physical GPU {args.require_physical_gpu!r}; received {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal execution requires exactly one visible CUDA device")
    torch.use_deterministic_algorithms(True)
    return torch.device("cuda:0"), visible_parts[0]


def _official_allowlist(
    repo: Path, split: str, contract: Mapping[str, Any]
) -> dict[str, dict[str, list[str]]]:
    manifest_split = "Eva" if split == "Eval" else "Tuning"
    selected: dict[str, dict[str, list[str]]] = {}
    manifests = contract.get("manifests")
    if not isinstance(manifests, Mapping) or set(manifests) != set(TARGETS):
        raise RuntimeError(f"invalid frozen manifest contract for {split}")
    for track, datasets in TARGETS.items():
        item = manifests[track]
        _require_keys(item, {"path", "sha256"}, f"{split}/{track} manifest")
        manifest = (repo / str(item["path"])).resolve()
        if not manifest.is_relative_to(repo.resolve()):
            raise RuntimeError(f"manifest path escapes repository: {manifest}")
        expected_path = repo / "data" / "File_List" / f"TSB-AD-{track}-{manifest_split}.csv"
        if manifest != expected_path.resolve() or file_sha256(manifest) != item["sha256"]:
            raise RuntimeError(f"frozen {split}/{track} manifest mismatch")
        names = pd.read_csv(manifest)["file_name"].astype(str).tolist()
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate file_name in {manifest}")
        selected[track] = {
            dataset: sorted(name for name in names if dataset_name(name) == dataset)
            for dataset in datasets
        }
        if any(not names for names in selected[track].values()):
            raise RuntimeError(f"empty declared subset in {manifest}")
    count = sum(
        len(names) for datasets in selected.values() for names in datasets.values()
    )
    if count != int(contract["expected_series"]):
        raise RuntimeError(f"frozen {split} series count mismatch: {count}")
    if stable_fingerprint(selected) != contract["allowlist_fingerprint"]:
        raise RuntimeError(f"frozen {split} allowlist fingerprint mismatch")
    return selected


def load_profile(args: argparse.Namespace) -> tuple[dict, dict]:
    if args.profile == "smoke":
        common = {
            "max_train_windows": args.max_train_windows,
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "optimizer_weight_decay": 0.0,
            "optimizer_amsgrad": False,
        }
        if args.method == "DPAD_AAAI24":
            parameters = {
                **common,
                "dpad_epochs": args.dpad_epochs,
                "gamma": 0.01,
                "lambda": 10.0,
                "learning_rate": 1e-3,
                "k": 5,
                "pairwise_reduction": "sum",
            }
        else:
            parameters = {
                **common,
                "dne_epochs": args.dne_epochs,
                "dne_architecture": args.dne_architecture,
                "batch_size": 128,
                "learning_rate": 1e-4,
                "weight_decay": 5e-4,
                "optimizer_amsgrad": True,
                "sigma_max": 2.0,
                "noise_level_parts": 3,
                "noise_ratios": [0.5, 0.8, 1.0],
                "learning_rate_decay_epoch": 100,
                "learning_rate_decay_factor": 0.1,
                "declared_max_optimizer_steps_per_unit": 2000,
            }
        return (
            parameters,
            {
                "configuration_source": "smoke_command_line",
                "config_fingerprint": stable_fingerprint(parameters),
            },
        )
    lock_path = args.locked_config.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    _require_keys(
        lock,
        {
            "status",
            "seed",
            "selection_split",
            "eval_feedback",
            "base_protocol",
            "base_protocol_sha256",
            "source_files",
            "source_sha256",
            "files",
            "methods",
            "plan",
            "plan_fingerprint",
            "locked_fingerprint",
        },
        "recent-baseline lock",
    )
    if lock["status"] != "frozen":
        raise RuntimeError("formal execution is disabled until the recent-baseline lock is frozen")
    stable = {
        key: value
        for key, value in lock.items()
        if key not in {"locked_fingerprint", "locked_at"}
    }
    if stable_fingerprint(stable) != lock["locked_fingerprint"]:
        raise RuntimeError("recent-baseline locked fingerprint mismatch")
    if stable_fingerprint(lock["plan"]) != lock["plan_fingerprint"]:
        raise RuntimeError("recent-baseline plan fingerprint mismatch")
    if int(lock["seed"]) != args.seed:
        raise RuntimeError(f"formal seed must be {lock['seed']}, received {args.seed}")
    if lock["selection_split"] != "official TSB-AD Tuning only" or lock["eval_feedback"] is not False:
        raise RuntimeError("recent-baseline lock violates the no-Eval-feedback policy")
    base_protocol = (lock_path.parent / lock["base_protocol"]).resolve()
    actual_base_hash = file_sha256(base_protocol)
    if actual_base_hash != lock["base_protocol_sha256"]:
        raise RuntimeError("base protocol hash does not match the frozen extension contract")
    if set(lock["source_files"]) != set(lock["source_sha256"]):
        raise RuntimeError("recent-baseline source path/hash keys differ")
    for name, relative in lock["source_files"].items():
        path = (args.repo.resolve() / str(relative)).resolve()
        if not path.is_relative_to(args.repo.resolve()):
            raise RuntimeError(f"recent-baseline source path escapes repository: {relative}")
        if file_sha256(path) != lock["source_sha256"][name]:
            raise RuntimeError(f"recent-baseline {name} source hash mismatch")
    expected_stage_source = (
        args.repo.resolve() / str(lock["source_files"]["metric_adapter"])
    ).resolve()
    if args.stage_source.resolve() != expected_stage_source:
        raise RuntimeError("--stage-source is not the evaluator frozen in the lock")
    metric_frontend = (
        args.repo.resolve() / str(lock["source_files"]["metric_frontend"])
    ).resolve()
    expected_paano_root = metric_frontend.parents[1]
    if args.paano_root.resolve() != expected_paano_root:
        raise RuntimeError("--paano-root is not the evaluator root frozen in the lock")
    file_contract = lock["files"][args.split]
    _require_keys(
        file_contract,
        {"manifests", "expected_series", "allowlist_fingerprint"},
        f"{args.split} allowlist contract",
    )
    allowlist = _official_allowlist(args.repo.resolve(), args.split, file_contract)
    allowed = allowlist[args.track][dataset_name(args.file)]
    if args.file not in allowed:
        raise RuntimeError(
            f"{args.file} is not in the frozen {args.split} allowlist for {args.track}"
        )
    method = lock["methods"][args.method]
    _require_keys(
        method,
        {"formal_hyperparameters", "config_fingerprint", "parameter_source"},
        f"lock method {args.method}",
    )
    parameters = method["formal_hyperparameters"]
    if stable_fingerprint(parameters) != method["config_fingerprint"]:
        raise RuntimeError(f"formal config fingerprint mismatch for {args.method}")
    return parameters, {
        "configuration_source": method["parameter_source"],
        "locked_fingerprint": lock["locked_fingerprint"],
        "config_fingerprint": method["config_fingerprint"],
        "plan_fingerprint": lock["plan_fingerprint"],
        "source_sha256": lock["source_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--track", required=True, choices=("U", "M"))
    parser.add_argument("--file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage-source", required=True, type=Path)
    parser.add_argument("--paano-root", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--split", choices=("Smoke", "Tuning", "Eval"), default="Smoke")
    parser.add_argument(
        "--locked-config",
        type=Path,
        default=PROJECT / "configs" / "aaai_recent_locked_seed2026.json",
    )
    parser.add_argument("--require-physical-gpu")
    parser.add_argument("--dpad-epochs", type=int, default=3)
    parser.add_argument("--dne-epochs", type=int, default=3)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--dne-architecture", choices=("mlp", "resmlp"), default="resmlp")
    args = parser.parse_args()

    if args.profile == "smoke" and args.split != "Smoke":
        raise ValueError("smoke profile requires --split Smoke")
    if args.profile == "formal" and args.split not in {"Tuning", "Eval"}:
        raise ValueError("formal profile requires --split Tuning or Eval")
    if args.profile == "formal" and args.require_physical_gpu is None:
        raise ValueError("formal profile requires --require-physical-gpu")

    profile, configuration = load_profile(args)
    expected_identity = {
        "method": args.method,
        "track": args.track,
        "dataset": dataset_name(args.file),
        "file": args.file,
        "seed": int(args.seed),
        "profile": args.profile,
        "split": args.split,
        "config_fingerprint": configuration["config_fingerprint"],
    }
    if args.profile == "formal":
        expected_identity["locked_fingerprint"] = configuration["locked_fingerprint"]
        expected_identity["plan_fingerprint"] = configuration["plan_fingerprint"]
    target = args.output.resolve()
    if target.is_file():
        if _strict_complete_record(target, expected_identity):
            return 0
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        mismatched = {
            key: (existing.get(key), value)
            for key, value in expected_identity.items()
            if existing.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"refusing to overwrite stale/mismatched output: {mismatched}")
    started = time.time()
    record = {
        **expected_identity,
        "training_prefix_policy": "filename_declared_label_blind",
        "runtime_eligible_for_paper": False,
        "error": None,
    }
    try:
        seed_everything(args.seed)
        record.update(configuration)
        data_path = args.repo / "data" / f"TSB-AD-{args.track}" / args.file
        values = pd.read_csv(
            data_path, usecols=lambda column: column != "Label"
        ).to_numpy(dtype=np.float32)
        boundary = parse_train_index(args.file)
        if boundary < 2 or boundary > len(values):
            raise ValueError(f"invalid training boundary {boundary} for {len(values)} points")
        train_mean = values[:boundary].mean(axis=0, keepdims=True)
        train_std = values[:boundary].std(axis=0, keepdims=True)
        normalized = np.nan_to_num(
            (values - train_mean) / np.where(train_std < 1e-8, 1.0, train_std)
        ).astype(np.float32)
        window = choose_window(args.repo, normalized, boundary, args.track)
        training_all = make_windows(normalized[:boundary], window)
        training = deterministic_subset(training_all, profile["max_train_windows"])
        query = make_windows(normalized, window)
        device, physical_gpu = _formal_device(args)

        if args.method == "DPAD_AAAI24":
            _require_keys(
                profile,
                {
                    "dpad_epochs",
                    "gamma",
                    "lambda",
                    "learning_rate",
                    "k",
                    "pairwise_reduction",
                    "max_train_windows",
                    "optimizer_betas",
                    "optimizer_eps",
                    "optimizer_weight_decay",
                    "optimizer_amsgrad",
                },
                "DPAD parameters",
            )
            patch_scores, hyperparameters = train_dpad(
                training,
                query,
                device,
                epochs=int(profile["dpad_epochs"]),
                gamma=float(profile["gamma"]),
                regularization=float(profile["lambda"]),
                learning_rate=float(profile["learning_rate"]),
                neighbors=int(profile["k"]),
                pairwise_reduction=str(profile["pairwise_reduction"]),
                optimizer_betas=tuple(float(value) for value in profile["optimizer_betas"]),
                optimizer_eps=float(profile["optimizer_eps"]),
                optimizer_weight_decay=float(profile["optimizer_weight_decay"]),
                optimizer_amsgrad=bool(profile["optimizer_amsgrad"]),
                execution_profile=args.profile,
            )
        else:
            _require_keys(
                profile,
                {
                    "dne_epochs",
                    "batch_size",
                    "dne_architecture",
                    "learning_rate",
                    "weight_decay",
                    "optimizer_betas",
                    "optimizer_eps",
                    "optimizer_amsgrad",
                    "sigma_max",
                    "noise_level_parts",
                    "noise_ratios",
                    "learning_rate_decay_epoch",
                    "learning_rate_decay_factor",
                    "declared_max_optimizer_steps_per_unit",
                    "max_train_windows",
                },
                "DNE parameters",
            )
            patch_scores, hyperparameters = train_dne(
                training,
                query,
                device,
                epochs=int(profile["dne_epochs"]),
                batch_size=int(profile["batch_size"]),
                architecture=str(profile["dne_architecture"]),
                learning_rate=float(profile["learning_rate"]),
                weight_decay=float(profile["weight_decay"]),
                optimizer_betas=tuple(float(value) for value in profile["optimizer_betas"]),
                optimizer_eps=float(profile["optimizer_eps"]),
                optimizer_amsgrad=bool(profile["optimizer_amsgrad"]),
                sigma_max=float(profile["sigma_max"]),
                noise_level_parts=int(profile["noise_level_parts"]),
                noise_ratios=tuple(float(value) for value in profile["noise_ratios"]),
                learning_rate_decay_epoch=int(profile["learning_rate_decay_epoch"]),
                learning_rate_decay_factor=float(profile["learning_rate_decay_factor"]),
                max_optimizer_steps=int(
                    profile["declared_max_optimizer_steps_per_unit"]
                ),
                execution_profile=args.profile,
            )
        if not np.isfinite(patch_scores).all():
            raise ValueError("non-finite raw anomaly score")
        point_scores = to_points(patch_scores, window, len(values))
        if not np.isfinite(point_scores).all():
            raise ValueError("non-finite point-level anomaly score")
        labels = pd.read_csv(data_path, usecols=["Label"])["Label"].to_numpy(dtype=np.int64)
        metrics, sliding_window = official_metrics(
            point_scores,
            labels,
            values,
            args.stage_source,
            args.paano_root.parent,
        )
        record.update(
            {
                "train_index": int(boundary),
                "points": int(len(values)),
                "channels": int(values.shape[1]),
                "window": int(window),
                "feature_dimension": int(query.shape[1]),
                "training_windows_available": int(len(training_all)),
                "training_windows_used": int(len(training)),
                "device": str(device),
                "physical_gpu": physical_gpu,
                "normalization": "channel z-score fitted only on filename-declared training prefix",
                "hyperparameters": hyperparameters,
                "sliding_window": sliding_window,
                "metrics": metrics,
                "runtime_s": float(time.time() - started),
            }
        )
        if args.profile == "smoke":
            record["training_prefix_anomaly_count_diagnostic_only"] = int(
                np.count_nonzero(labels[:boundary])
            )
    except Exception as exc:
        record.update(
            {
                "metrics": {},
                "runtime_s": float(time.time() - started),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    atomic_json(target, record)
    return 0 if record["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
