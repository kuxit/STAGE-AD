from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from common import (  # noqa: E402
    atomic_json,
    complete_record,
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
) -> tuple[np.ndarray, dict]:
    input_dim = int(train_values.shape[1])
    hidden = min(256, max(64, input_dim * 2))
    output_dim = min(32, max(2, input_dim // 2))
    model = DenseProjection(input_dim, (hidden, max(32, hidden // 2)), output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training = torch.from_numpy(train_values).to(device)
    model.train()
    final_loss = math.nan
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(training)
        squared_distances = torch.cdist(embeddings, embeddings, p=2).square()
        weights = torch.exp(-gamma * squared_distances.detach())
        density_loss = (squared_distances * weights).mean()
        loss = density_loss + regularization * model.anti_collapse_penalty()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    with torch.no_grad():
        reference = model(training)
        k = min(max(1, neighbors), len(reference))
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
        "k": neighbors,
        "pairwise_reduction": "mean",
        "final_training_loss": final_loss,
        "formal_status": "smoke-only engineering configuration; Tuning freeze required",
    }
    return scores, details


def train_dne(
    train_values: np.ndarray,
    query_values: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    architecture: str,
) -> tuple[np.ndarray, dict]:
    input_dim = int(train_values.shape[1])
    if architecture == "resmlp":
        model: nn.Module = NoiseEvaluationResMLP(input_dim)
    else:
        model = NoiseEvaluationMLP(input_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=5e-4, amsgrad=True
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
    ratios = (0.5, 0.8, 1.0)
    sigma_max = 2.0
    parts = 3
    model.train()
    final_loss = math.nan
    for epoch in range(epochs):
        if epoch == 100:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        for (clean_cpu,) in loader:
            clean = clean_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(clean), torch.zeros_like(clean))
            for ratio in ratios:
                noise = diverse_gaussian_noise(clean, sigma_max, parts, ratio)
                loss = loss + nn.functional.mse_loss(model(clean + noise), noise.abs())
            loss.backward()
            optimizer.step()
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
        "optimizer": "Adam AMSGrad",
        "learning_rate": 1e-4,
        "weight_decay": 5e-4,
        "learning_rate_decay": "factor 0.1 at epoch 100",
        "noise_type": "Gaussian",
        "sigma_max": sigma_max,
        "noise_level_parts": parts,
        "noise_ratios": list(ratios),
        "noisy_instances_per_clean_instance": 3,
        "score_aggregation": "maximum model output",
        "final_training_loss": final_loss,
        "formal_status": "paper settings with smoke-reduced epochs; Tuning freeze required",
    }
    return scores, details


def load_profile(args: argparse.Namespace) -> dict:
    if args.profile == "smoke":
        return {
            "dpad_epochs": args.dpad_epochs,
            "dne_epochs": args.dne_epochs,
            "max_train_windows": args.max_train_windows,
            "dne_architecture": args.dne_architecture,
        }
    protocol_path = PROJECT / "extensions" / "aaai_recent" / "protocol_extension.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("formal execution is disabled until protocol_extension.json is frozen")
    if int(protocol.get("seed", -1)) != args.seed:
        raise RuntimeError(f"formal seed must be {protocol.get('seed')}, received {args.seed}")
    base_protocol = (protocol_path.parent / protocol["base_protocol"]).resolve()
    actual_base_hash = file_sha256(base_protocol)
    if actual_base_hash != protocol.get("base_protocol_sha256"):
        raise RuntimeError("base protocol hash does not match the frozen extension contract")
    parameters = protocol["methods"][args.method].get("formal_hyperparameters")
    if not parameters:
        raise RuntimeError(f"formal hyperparameters are not frozen for {args.method}")
    return parameters


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
    parser.add_argument("--dpad-epochs", type=int, default=3)
    parser.add_argument("--dne-epochs", type=int, default=3)
    parser.add_argument("--max-train-windows", type=int, default=512)
    parser.add_argument("--dne-architecture", choices=("mlp", "resmlp"), default="resmlp")
    args = parser.parse_args()

    target = args.output.resolve()
    if complete_record(target):
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("profile") == args.profile:
            return 0
        raise RuntimeError(
            f"refusing to reuse {existing.get('profile')!r} output for {args.profile!r} execution"
        )
    started = time.time()
    record = {
        "method": args.method,
        "track": args.track,
        "dataset": dataset_name(args.file),
        "file": args.file,
        "seed": int(args.seed),
        "profile": args.profile,
        "training_prefix_policy": "filename_declared_label_blind",
        "runtime_eligible_for_paper": False,
        "error": None,
    }
    try:
        seed_everything(args.seed)
        profile = load_profile(args)
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
        training = deterministic_subset(training_all, profile.get("max_train_windows"))
        query = make_windows(normalized, window)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.method == "DPAD_AAAI24":
            patch_scores, hyperparameters = train_dpad(
                training,
                query,
                device,
                epochs=int(profile["dpad_epochs"]),
                gamma=float(profile.get("gamma", 0.01)),
                regularization=float(profile.get("lambda", 10.0)),
                learning_rate=float(profile.get("learning_rate", 1e-3)),
                neighbors=int(profile.get("k", 5)),
            )
        else:
            patch_scores, hyperparameters = train_dne(
                training,
                query,
                device,
                epochs=int(profile["dne_epochs"]),
                batch_size=int(profile.get("batch_size", 128)),
                architecture=str(profile["dne_architecture"]),
            )
        point_scores = to_points(np.nan_to_num(patch_scores), window, len(values))
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
                "training_prefix_anomaly_count_diagnostic_only": int(np.count_nonzero(labels[:boundary])),
                "points": int(len(values)),
                "channels": int(values.shape[1]),
                "window": int(window),
                "feature_dimension": int(query.shape[1]),
                "training_windows_available": int(len(training_all)),
                "training_windows_used": int(len(training)),
                "device": str(device),
                "normalization": "channel z-score fitted only on filename-declared training prefix",
                "hyperparameters": hyperparameters,
                "sliding_window": sliding_window,
                "metrics": metrics,
                "runtime_s": float(time.time() - started),
            }
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
