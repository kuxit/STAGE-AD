#!/usr/bin/env python3
"""Run one official PaAno TSB-AD series without retaining model or scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


SIX_METRICS = (
    "VUS-PR",
    "VUS-ROC",
    "R-based-F1",
    "AUC-PR",
    "AUC-ROC",
    "Standard-F1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paano-root", required=True, type=Path)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--track", required=True, choices=("U", "M"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--require-physical-gpu", default="1")
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != args.require_physical_gpu:
        raise RuntimeError(
            f"launch with CUDA_VISIBLE_DEVICES={args.require_physical_gpu}; got {visible!r}"
        )
    paano_root = args.paano_root.resolve()
    sys.path.insert(0, str(paano_root))
    from model import PatchEncoder
    from train import train_model
    from utils.data_preprocess import (
        PatchCreator,
        find_length_rank,
        load_and_split_data,
        preprocess_to_patches,
    )
    from utils.evaluation import (
        calculate_anomaly_scores,
        distribute_patch_scores_to_points,
    )
    from utils.metrics import get_metrics
    from utils.utils import create_memory_bank

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    data_file = args.data_file.resolve()
    train_data, train_labels, test_data, test_labels = load_and_split_data(str(data_file))
    train_data = np.asarray(train_data, dtype=np.float32)
    test_data = np.asarray(test_data, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if train_data.size == 0 or test_data.size == 0:
        raise ValueError(f"empty split for {data_file}")
    # Official run_uni.sh/run_mul.sh pass --use_revin.  The declared training
    # prefix is therefore consumed label-blind with no label filtering.
    full_data = np.concatenate([train_data, test_data], axis=0)
    full_labels = np.concatenate([train_labels, test_labels], axis=0)
    sliding_input = (
        full_data.reshape(-1, 1)
        if full_data.ndim == 1
        else full_data[:, 0].reshape(-1, 1)
    )
    sliding_window = int(find_length_rank(sliding_input, rank=1))
    patch_creator = PatchCreator(L=args.patch_size, s=1, random_seed=seed)
    train_loader, test_loader, _ = patch_creator.create_dataloaders(
        train_data, full_data, full_labels, batch_size=args.batch_size
    )
    first_batch, _ = next(iter(train_loader))
    model = PatchEncoder(in_channels=first_batch.shape[1], use_revin=True).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    total_started = time.perf_counter()
    training_started = time.perf_counter()
    try:
        train_model(
            model,
            train_loader,
            preprocess_to_patches(train_data, patch_size=args.patch_size, stride=1),
            device,
            num_iter=args.num_iters,
            pretext_step=args.patch_size,
            lr=args.learning_rate,
            see_loss=False,
        )
        training_seconds = float(time.perf_counter() - training_started)
        memory_started = time.perf_counter()
        memory_bank, _ = create_memory_bank(
            model, train_loader, device, num_cores=0.1
        )
        memory_seconds = float(time.perf_counter() - memory_started)
        query_started = time.perf_counter()
        patch_scores = calculate_anomaly_scores(
            model, test_loader, memory_bank, top_k=3, device=device
        )
        point_scores = distribute_patch_scores_to_points(
            patch_scores, patch_size=args.patch_size, num_points=len(full_labels)
        )
        query_seconds = float(time.perf_counter() - query_started)
        metric_started = time.perf_counter()
        metrics = get_metrics(
            point_scores,
            full_labels,
            slidingWindow=sliding_window,
            pred=None,
            version="opt",
            thre=250,
        )
        metric_seconds = float(time.perf_counter() - metric_started)
        record = {
            "method": "official_paano",
            "track": args.track,
            "file": data_file.name,
            "seed": seed,
            "points": int(len(full_labels)),
            "channels": int(1 if full_data.ndim == 1 else full_data.shape[1]),
            "train_index": int(len(train_data)),
            "training_prefix_policy": "filename_declared_label_blind",
            "training_prefix_anomaly_count": int(np.count_nonzero(train_labels)),
            "sliding_window": sliding_window,
            "parameter_count": parameter_count,
            "memory_rows": int(len(memory_bank)),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "training_seconds": training_seconds,
            "memory_seconds": memory_seconds,
            "query_seconds": query_seconds,
            "metric_seconds": metric_seconds,
            "elapsed_seconds": float(time.perf_counter() - total_started),
            "config": {
                "patch_size": int(args.patch_size),
                "num_iters": int(args.num_iters),
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "use_revin": True,
                "memory_bank_ratio": 0.1,
                "top_k": 3,
            },
            "source_sha256": {
                relative: sha256_file(paano_root / relative)
                for relative in (
                    "model.py",
                    "train.py",
                    "utils/data_preprocess.py",
                    "utils/utils.py",
                    "utils/evaluation.py",
                    "utils/metrics.py",
                )
            },
            "metrics": {name: float(metrics[name]) for name in SIX_METRICS},
        }
        atomic_json(args.output.resolve(), record)
    finally:
        # Official train_model writes this relative checkpoint.  The controller
        # runs in an ephemeral per-unit directory; remove it before commit.
        checkpoint = Path.cwd() / "best_trained_encoder.pth"
        if checkpoint.is_file():
            checkpoint.unlink()
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
