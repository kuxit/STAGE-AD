from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import atomic_json, complete_record, dataset_name, official_metrics, parse_train_index, seed_everything


def starts_for(points: int, window: int, stride: int) -> np.ndarray:
    if points < window:
        raise ValueError(f"series length {points} is shorter than DCdetector window {window}")
    starts = list(range(0, points - window + 1, stride))
    if starts[-1] != points - window:
        starts.append(points - window)
    return np.asarray(starts, dtype=np.int64)


def make_windows(values: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    return np.stack([values[index : index + window] for index in starts]).astype(np.float32)


def kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    result = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(result, dim=-1), dim=1)


def discrepancy(series: list[torch.Tensor], prior: list[torch.Tensor], window: int) -> tuple[torch.Tensor, torch.Tensor]:
    series_loss: torch.Tensor | float = 0.0
    prior_loss: torch.Tensor | float = 0.0
    for series_item, prior_item in zip(series, prior):
        normalized_prior = prior_item / torch.unsqueeze(torch.sum(prior_item, dim=-1), dim=-1).repeat(1, 1, 1, window)
        series_loss = series_loss + torch.mean(kl_loss(series_item, normalized_prior.detach())) + torch.mean(kl_loss(normalized_prior.detach(), series_item))
        prior_loss = prior_loss + torch.mean(kl_loss(normalized_prior, series_item.detach())) + torch.mean(kl_loss(series_item.detach(), normalized_prior))
    return series_loss / len(prior), prior_loss / len(prior)


@torch.no_grad()
def validation(model: nn.Module, loader: DataLoader, device: torch.device, window: int) -> float:
    model.eval()
    values = []
    for (batch,) in loader:
        series, prior = model(batch.to(device))
        series_loss, prior_loss = discrepancy(series, prior, window)
        values.append(float((prior_loss - series_loss).detach().cpu()))
    return float(np.mean(values))


def fit(model: nn.Module, train_loader: DataLoader, validation_loader: DataLoader, device: torch.device, window: int) -> tuple[int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float("inf")
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    epochs = 0
    for epoch in range(3):
        model.train()
        for (batch,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            series, prior = model(batch.to(device))
            series_loss, prior_loss = discrepancy(series, prior, window)
            loss = prior_loss - series_loss
            loss.backward()
            optimizer.step()
        current = validation(model, validation_loader, device, window)
        epochs = epoch + 1
        if current < best_loss:
            best_loss = current
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * (0.5 ** epoch)
    model.load_state_dict(best_state)
    return epochs, best_loss


@torch.no_grad()
def scores(model: nn.Module, loader: DataLoader, starts: np.ndarray, points: int, window: int, device: torch.device) -> np.ndarray:
    model.eval()
    windows = []
    for (batch,) in loader:
        series, prior = model(batch.to(device))
        series_token = None
        prior_token = None
        for series_item, prior_item in zip(series, prior):
            normalized_prior = prior_item / torch.unsqueeze(torch.sum(prior_item, dim=-1), dim=-1).repeat(1, 1, 1, window)
            current_series = kl_loss(series_item, normalized_prior.detach()) * 50.0
            current_prior = kl_loss(normalized_prior, series_item.detach()) * 50.0
            series_token = current_series if series_token is None else series_token + current_series
            prior_token = current_prior if prior_token is None else prior_token + current_prior
        windows.append(torch.softmax(-series_token - prior_token, dim=-1).cpu().numpy())
    token_scores = np.concatenate(windows, axis=0)
    sums = np.zeros(points, dtype=np.float64)
    counts = np.zeros(points, dtype=np.float64)
    for start, score in zip(starts, token_scores):
        sums[start : start + window] += score
        counts[start : start + window] += 1.0
    if np.any(counts == 0):
        raise AssertionError("DCdetector scoring left uncovered timestamps")
    return sums / counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--dcdetector-root", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=("U", "M"))
    parser.add_argument("--file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-duoba-source", required=True, type=Path)
    parser.add_argument("--paano-root", required=True, type=Path)
    parser.add_argument("--require-physical-gpu", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() != args.require_physical_gpu:
        raise RuntimeError("physical GPU isolation mismatch")
    target = args.output.resolve()
    if complete_record(target):
        return 0
    started = time.time()
    record = {
        "method": "DCdetector", "track": args.track, "dataset": dataset_name(args.file),
        "file": args.file, "seed": args.seed,
        "training_prefix_policy": "filename_declared_label_blind", "error": None,
    }
    try:
        seed_everything(args.seed)
        device = torch.device("cuda:0")
        sys.path.insert(0, str(args.dcdetector_root.resolve()))
        from model.DCdetector import DCdetector

        frame = pd.read_csv(args.repo / "data" / f"TSB-AD-{args.track}" / args.file)
        values = frame.drop(columns=["Label"]).to_numpy(dtype=np.float32)
        labels = frame["Label"].to_numpy(dtype=np.int64)
        boundary = parse_train_index(args.file)
        mean = values[:boundary].mean(axis=0, keepdims=True)
        std = values[:boundary].std(axis=0, keepdims=True)
        normalized = np.nan_to_num((values - mean) / np.where(std < 1e-8, 1.0, std)).astype(np.float32)
        window = 105 if args.track == "U" else 60
        patch_sizes = [3, 5, 7] if args.track == "U" else [3, 5]
        batch_size = 128
        train_starts = starts_for(boundary, window, 1)
        train_windows = make_windows(normalized[:boundary], train_starts, window)
        split = min(max(int(round(0.8 * len(train_windows))), 1), max(len(train_windows) - 1, 1))
        training = train_windows[:split] if len(train_windows) > 1 else train_windows
        validation_windows = train_windows[split:] if len(train_windows) > 1 else train_windows
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(training)), batch_size=batch_size, shuffle=True, generator=generator, drop_last=False)
        validation_loader = DataLoader(TensorDataset(torch.from_numpy(validation_windows)), batch_size=batch_size, shuffle=False)
        model = DCdetector(
            win_size=window, enc_in=values.shape[1], c_out=values.shape[1], n_heads=1,
            d_model=256, e_layers=3, patch_size=patch_sizes, channel=values.shape[1],
            d_ff=512, dropout=0.0, output_attention=True,
        ).to(device)
        trained_epochs, best_validation = fit(model, train_loader, validation_loader, device, window)
        full_starts = starts_for(len(values), window, window)
        full_windows = make_windows(normalized, full_starts, window)
        full_loader = DataLoader(TensorDataset(torch.from_numpy(full_windows)), batch_size=batch_size, shuffle=False)
        point_scores = scores(model, full_loader, full_starts, len(values), window, device)
        metrics, sliding_window = official_metrics(
            point_scores, labels, values, args.frozen_duoba_source, args.paano_root.parent
        )
        record.update(
            {
                "train_index": int(boundary),
                "training_prefix_anomaly_count": int(np.count_nonzero(labels[:boundary])),
                "points": int(len(values)), "channels": int(values.shape[1]),
                "sliding_window": sliding_window,
                "hyperparameters": {
                    "window": window, "patch_sizes": patch_sizes, "batch_size": batch_size,
                    "d_model": 256, "n_heads": 1, "encoder_layers": 3,
                    "learning_rate": 1e-4, "epochs": 3, "temperature": 50,
                },
                "protocol_adapter": "official architecture, discrepancy objective and score; generic TSB-AD prefix loader; in-memory model selection",
                "trained_epochs": trained_epochs, "best_validation_loss": best_validation,
                "metrics": metrics, "runtime_s": float(time.time() - started),
            }
        )
    except Exception as exc:
        record.update(
            {
                "metrics": {}, "runtime_s": float(time.time() - started),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            }
        )
    atomic_json(target, record)
    return 0 if record["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
