from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import atomic_json, complete_record, dataset_name, official_metrics, parse_train_index, seed_everything


def starts_for(points: int, window: int, stride: int) -> np.ndarray:
    if points < window:
        raise ValueError(f"series length {points} is shorter than MEMTO window {window}")
    starts = list(range(0, points - window + 1, stride))
    if starts[-1] != points - window:
        starts.append(points - window)
    return np.asarray(starts, dtype=np.int64)


def make_windows(values: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    return np.stack([values[index : index + window] for index in starts]).astype(np.float32)


def snapshot(model: nn.Module) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    memory = model.mem_module.mem.detach().cpu().clone()
    return state, memory


def restore(model: nn.Module, state: dict[str, torch.Tensor], memory: torch.Tensor, device: torch.device) -> None:
    model.load_state_dict(state)
    model.mem_module.mem = memory.to(device)


def entropy_loss(attention: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum(-attention * torch.log(attention + 1e-12), dim=-1))


@torch.no_grad()
def validation_loss(model: nn.Module, loader: DataLoader, device: torch.device, coefficient: float) -> float:
    model.eval()
    losses = []
    for (batch,) in loader:
        batch = batch.to(device)
        output = model(batch)
        loss = torch.mean((output["out"] - batch) ** 2) + coefficient * entropy_loss(output["attn"])
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def fit_phase(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    coefficient: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, int, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state, best_memory = snapshot(model)
    stale = 0
    epochs = 0
    for epoch in range(max_epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = torch.mean((output["out"] - batch) ** 2) + coefficient * entropy_loss(output["attn"])
            loss.backward()
            optimizer.step()
        current = validation_loss(model, validation_loader, device, coefficient)
        epochs = epoch + 1
        if current < best_loss - 1e-7:
            best_loss = current
            best_state, best_memory = snapshot(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    restore(model, best_state, best_memory, device)
    return best_state, best_memory, epochs, best_loss


@torch.no_grad()
def encoder_queries(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    queries = []
    for (batch,) in loader:
        batch = batch.to(device)
        embedded = model.embedding(batch)
        query = model.encoder(embedded)
        queries.append(query.reshape(-1, query.shape[-1]).cpu().numpy())
    return np.concatenate(queries, axis=0)


@torch.no_grad()
def anomaly_scores(
    model: nn.Module,
    loader: DataLoader,
    starts: np.ndarray,
    points: int,
    window: int,
    device: torch.device,
    temperature: float,
) -> np.ndarray:
    model.eval()
    token_scores = []
    for (batch,) in loader:
        batch = batch.to(device)
        output = model(batch)
        reconstruction = torch.mean((output["out"] - batch) ** 2, dim=-1)
        queries = output["queries"]
        memory = output["mem"]
        distances = torch.sum(
            (queries.unsqueeze(-2) - memory.reshape(1, 1, memory.shape[0], memory.shape[1])) ** 2,
            dim=-1,
        )
        gathering = torch.min(distances, dim=-1).values
        weights = torch.softmax(gathering / temperature, dim=-1)
        token_scores.append((weights * reconstruction).cpu().numpy())
    windows = np.concatenate(token_scores, axis=0)
    sums = np.zeros(points, dtype=np.float64)
    counts = np.zeros(points, dtype=np.float64)
    for start, score in zip(starts, windows):
        sums[start : start + window] += score
        counts[start : start + window] += 1.0
    if np.any(counts == 0):
        raise AssertionError("MEMTO scoring left uncovered timestamps")
    return sums / counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--memto-root", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=("U", "M"))
    parser.add_argument("--file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage-source", required=True, type=Path)
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
        "method": "MEMTO",
        "track": args.track,
        "dataset": dataset_name(args.file),
        "file": args.file,
        "seed": args.seed,
        "training_prefix_policy": "filename_declared_label_blind",
        "error": None,
    }
    try:
        seed_everything(args.seed)
        device = torch.device("cuda:0")
        sys.path.insert(0, str(args.memto_root.resolve()))
        from model.Transformer import TransformerVar

        data_path = args.repo / "data" / f"TSB-AD-{args.track}" / args.file
        frame = pd.read_csv(data_path)
        values = frame.drop(columns=["Label"]).to_numpy(dtype=np.float32)
        labels = frame["Label"].to_numpy(dtype=np.int64)
        boundary = parse_train_index(args.file)
        mean = values[:boundary].mean(axis=0, keepdims=True)
        std = values[:boundary].std(axis=0, keepdims=True)
        normalized = np.nan_to_num((values - mean) / np.where(std < 1e-8, 1.0, std)).astype(np.float32)

        window = 100
        stride = 100
        batch_size = 256
        d_model = 512
        memory_items = 10
        train_starts = starts_for(boundary, window, stride)
        train_windows = make_windows(normalized[:boundary], train_starts, window)
        split = min(max(int(round(0.8 * len(train_windows))), 1), max(len(train_windows) - 1, 1))
        if len(train_windows) == 1:
            training = validation = train_windows
        else:
            training, validation = train_windows[:split], train_windows[split:]
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(training)), batch_size=batch_size, shuffle=True, generator=generator)
        validation_loader = DataLoader(TensorDataset(torch.from_numpy(validation)), batch_size=batch_size, shuffle=False)
        all_train_loader = DataLoader(TensorDataset(torch.from_numpy(train_windows)), batch_size=batch_size, shuffle=False)

        phase1 = TransformerVar(
            win_size=window, enc_in=values.shape[1], c_out=values.shape[1], n_memory=memory_items,
            d_model=d_model, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, device=device,
            memory_initial=False, phase_type=None, dataset_name="TSBAD",
        ).to(device)
        _, _, phase1_epochs, phase1_best = fit_phase(
            phase1, train_loader, validation_loader, device,
            learning_rate=1e-4, max_epochs=100, patience=10, coefficient=0.01,
        )
        queries = encoder_queries(phase1, all_train_loader, device)
        centers = KMeans(n_clusters=min(memory_items, len(queries)), n_init=10, random_state=args.seed).fit(queries).cluster_centers_
        if len(centers) < memory_items:
            repeats = np.repeat(centers[-1:], memory_items - len(centers), axis=0)
            centers = np.concatenate([centers, repeats], axis=0)
        del phase1
        torch.cuda.empty_cache()

        phase2 = TransformerVar(
            win_size=window, enc_in=values.shape[1], c_out=values.shape[1], n_memory=memory_items,
            d_model=d_model, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, device=device,
            memory_init_embedding=torch.from_numpy(centers).float().to(device),
            memory_initial=False, phase_type="second_train", dataset_name="TSBAD",
        ).to(device)
        phase2_state, phase2_memory, phase2_epochs, phase2_best = fit_phase(
            phase2, train_loader, validation_loader, device,
            learning_rate=5e-5, max_epochs=100, patience=10, coefficient=0.01,
        )
        restore(phase2, phase2_state, phase2_memory, device)
        phase2.mem_module.phase_type = "test"
        full_starts = starts_for(len(normalized), window, stride)
        full_windows = make_windows(normalized, full_starts, window)
        full_loader = DataLoader(TensorDataset(torch.from_numpy(full_windows)), batch_size=batch_size, shuffle=False)
        scores = anomaly_scores(phase2, full_loader, full_starts, len(values), window, device, temperature=0.1)
        metrics, sliding_window = official_metrics(
            scores, labels, values, args.stage_source, args.paano_root.parent
        )
        record.update(
            {
                "train_index": int(boundary),
                "training_prefix_anomaly_count": int(np.count_nonzero(labels[:boundary])),
                "points": int(len(values)),
                "channels": int(values.shape[1]),
                "sliding_window": sliding_window,
                "hyperparameters": {
                    "window": window, "stride": stride, "batch_size": batch_size,
                    "d_model": d_model, "n_heads": 8, "encoder_layers": 3,
                    "memory_items": memory_items, "entropy_weight": 0.01,
                    "phase1_lr": 1e-4, "phase2_lr": 5e-5,
                    "max_epochs_per_phase": 100, "early_stopping_patience": 10,
                    "temperature": 0.1,
                },
                "protocol_adapter": "official architecture and two-phase memory initialization; generic TSB-AD prefix loader; in-memory early stopping; sklearn KMeans",
                "phase1_epochs": phase1_epochs,
                "phase2_epochs": phase2_epochs,
                "phase1_best_validation_loss": phase1_best,
                "phase2_best_validation_loss": phase2_best,
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
