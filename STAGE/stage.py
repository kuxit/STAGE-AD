#!/usr/bin/env python3
"""Standalone STAGE research implementation.

This file intentionally contains the complete model-side pipeline:

* CSV loading and train-prefix validation;
* unit-stride temporal patch access without materialising the whole patch set;
* a dilated residual token encoder;
* exact-overlap cross-correlation learning with redundancy reduction;
* a one-shot granular sampler and a final granular exemplar memory;
* a small-data update-budget safeguard;
* top-k squared unit-Euclidean patch scoring and overlap-to-point aggregation;
* a portable command-line experiment runner.

The model and training code do not import PaAno, GBOC, or any project module.
For benchmark reporting only, ``--metrics-root`` may point to the repository's
official TSB-AD-compatible evaluator.  That evaluator is kept outside the
model path so that exported checkpoints and inference remain standalone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import nn
from torch.nn import functional as F


SIX_METRICS = (
    "VUS-PR",
    "VUS-ROC",
    "AUC-PR",
    "AUC-ROC",
    "R-based-F1",
    "Standard-F1",
)


@dataclass(frozen=True)
class StageConfig:
    patch_size: int = 96
    channels: int = 128
    token_dim: int = 64
    embedding_dim: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 4, 2)
    group_norm_groups: int = 8
    dropout: float = 0.10
    batch_size: int = 256
    steps: int = 1000
    gb_activation_fraction: float = 0.20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    overlap_deltas: tuple[int, ...] = (24, 48)
    overlap_trim: int = 8
    alignment_objective: str = "both"
    gb_min_split: int = 4
    gb_max_rounds: int = 64
    gb_sampling_power: float = 0.5
    top_k: int = 3
    embedding_batch_size: int = 2048
    score_batch_size: int = 2048
    memory_score_block_size: int = 8192
    max_gb_rows: int = 50000
    seed: int = 2026

    def validate(self) -> None:
        if self.patch_size < 8:
            raise ValueError("patch_size must be at least 8")
        if not self.dilations:
            raise ValueError("at least one dilation is required")
        if self.channels % self.group_norm_groups:
            raise ValueError("channels must be divisible by group_norm_groups")
        if self.token_dim < 2 or self.embedding_dim < 2:
            raise ValueError("token and embedding dimensions must exceed one")
        if self.batch_size < 4:
            raise ValueError("batch_size must be at least four")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if not 0.0 <= self.gb_activation_fraction < 1.0:
            raise ValueError("gb_activation_fraction must be in [0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(delta <= 0 or delta >= self.patch_size for delta in self.overlap_deltas):
            raise ValueError("overlap deltas must lie strictly inside a patch")
        if any(self.patch_size - delta - 2 * self.overlap_trim < 1 for delta in self.overlap_deltas):
            raise ValueError("overlap_trim removes the complete aligned region")
        if self.alignment_objective not in {
            "both",
            "timestamp_token_only",
            "interval_only",
            "context_only",
        }:
            raise ValueError("unsupported alignment_objective")
        if self.gb_min_split < 4:
            raise ValueError("gb_min_split must be at least four")
        if not 0.0 <= self.gb_sampling_power <= 1.0:
            raise ValueError("gb_sampling_power must be in [0, 1]")
        if self.score_batch_size < 1 or self.memory_score_block_size < 1:
            raise ValueError("score block sizes must be positive")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_train_index(filename: str) -> int:
    match = re.search(r"_tr_(\d+)_", Path(filename).stem)
    if match is None:
        raise ValueError(f"cannot parse training prefix from {filename}")
    return int(match.group(1))


def load_series(path: Path) -> tuple[np.ndarray, int, str]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    if not columns:
        raise ValueError(f"empty CSV: {path}")
    label_candidates = [column for column in columns if column.lower() == "label"]
    label_column = label_candidates[-1] if label_candidates else columns[-1]
    values = pd.read_csv(
        path, usecols=lambda column: column != label_column
    ).to_numpy(dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError(f"no feature columns in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite feature value in {path}")
    train_index = parse_train_index(path.name)
    if not 0 < train_index <= len(values):
        raise ValueError(f"invalid training prefix in {path}")
    return values, train_index, label_column


def load_labels_after_scoring(path: Path, label_column: str, expected_length: int) -> np.ndarray:
    labels = pd.read_csv(path, usecols=[label_column])[label_column].to_numpy(
        dtype=np.int64, copy=True
    )
    if len(labels) != expected_length:
        raise ValueError("features and labels have different lengths")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"labels must be binary in {path}")
    return labels


def load_manifest(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload.get("files", []) if isinstance(payload, dict) else payload
    result = [str(item) for item in files]
    if not result or len(result) != len(set(result)):
        raise ValueError("manifest must contain unique file names")
    return result


def window_batch(values: np.ndarray, starts: np.ndarray, patch_size: int) -> torch.Tensor:
    starts = np.asarray(starts, dtype=np.int64).reshape(-1)
    if not len(starts):
        raise ValueError("cannot construct an empty patch batch")
    offsets = np.arange(int(patch_size), dtype=np.int64)[None, :]
    indices = starts[:, None] + offsets
    if int(indices.min()) < 0 or int(indices.max()) >= len(values):
        raise IndexError("patch indices escape the time series")
    patches = values[indices]
    return torch.from_numpy(np.transpose(patches, (0, 2, 1)).copy())


class PatchRevIN(nn.Module):
    """Per-patch, per-channel normalization without learnable affine terms."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + self.eps)


class DilatedResidualBlock(nn.Module):
    """Non-causal dilated temporal mixing with a stable residual path."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = int(dilation)
        self.norm1 = nn.GroupNorm(int(groups), int(channels))
        self.conv1 = nn.Conv1d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        self.norm2 = nn.GroupNorm(int(groups), int(channels))
        self.conv2 = nn.Conv1d(
            int(channels),
            int(channels),
            kernel_size=3,
            padding=padding,
            dilation=int(dilation),
            bias=False,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.layer_scale = nn.Parameter(torch.full((1, int(channels), 1), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.dropout(x)
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + self.layer_scale * x


class StageEncoder(nn.Module):
    """Dilated residual token encoder used by STAGE."""

    def __init__(self, in_channels: int, config: StageConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.revin = PatchRevIN()
        self.stem = nn.Sequential(
            nn.Conv1d(
                int(in_channels),
                int(config.channels),
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(int(config.group_norm_groups), int(config.channels)),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                DilatedResidualBlock(
                    channels=config.channels,
                    dilation=dilation,
                    groups=config.group_norm_groups,
                    dropout=config.dropout,
                )
                for dilation in config.dilations
            ]
        )
        self.token_head = nn.Sequential(
            nn.GroupNorm(int(config.group_norm_groups), int(config.channels)),
            nn.GELU(),
            nn.Conv1d(int(config.channels), int(config.token_dim), kernel_size=1),
        )
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(2 * int(config.token_dim)),
            nn.Linear(2 * int(config.token_dim), int(config.embedding_dim)),
        )

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool the same directional token geometry used by overlap learning."""

        if tokens.ndim != 3 or tokens.shape[-1] != self.config.token_dim:
            raise ValueError("tokens must have shape [batch, time, token_dim]")
        unit_tokens = F.normalize(tokens, dim=-1)
        mean = unit_tokens.mean(dim=1)
        std = torch.sqrt(unit_tokens.var(dim=1, unbiased=False).clamp_min(1e-8))
        return self.embedding_head(torch.cat([mean, std], dim=1))

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("StageEncoder expects [batch, channels, time]")
        x = self.stem(self.revin(x))
        for block in self.blocks:
            x = block(x)
        return self.token_head(x).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode_tokens(x)
        return tokens, self.embed_tokens(tokens)


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("off_diagonal expects a square matrix")
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def cross_correlation_identity_loss(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match corresponding dimensions while removing cross-dimension redundancy."""

    if left.ndim != 2 or right.shape != left.shape or len(left) < 2:
        raise ValueError("cross-correlation inputs must be matching 2-D batches")
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    left = left / torch.sqrt(left.pow(2).mean(dim=0, keepdim=True) + 1e-4)
    right = right / torch.sqrt(right.pow(2).mean(dim=0, keepdim=True) + 1e-4)
    correlation = left.T @ right / float(len(left))
    diagonal = (torch.diagonal(correlation) - 1.0).pow(2).mean()
    redundancy = off_diagonal(correlation).pow(2).mean()
    return diagonal + redundancy, diagonal, redundancy


def temporal_overlap_loss(
    tokens_left: torch.Tensor,
    tokens_right: torch.Tensor,
    overlap_embedding_left: torch.Tensor,
    overlap_embedding_right: torch.Tensor,
    delta: int,
    config: StageConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align only tokens that correspond to the same raw timestamp."""

    patch_size = int(tokens_left.shape[1])
    trim = int(config.overlap_trim)
    left_start = int(delta) + trim
    left_stop = patch_size - trim
    right_start = trim
    right_stop = patch_size - int(delta) - trim
    if left_stop - left_start != right_stop - right_start or left_stop <= left_start:
        raise ValueError("invalid temporal-overlap alignment geometry")
    aligned_left_raw = tokens_left[:, left_start:left_stop]
    aligned_right_raw = tokens_right[:, right_start:right_stop]
    token_loss, token_diagonal, token_redundancy = cross_correlation_identity_loss(
        aligned_left_raw.reshape(-1, aligned_left_raw.shape[-1]),
        aligned_right_raw.reshape(-1, aligned_right_raw.shape[-1]),
    )
    embedding_loss, embedding_diagonal, embedding_redundancy = (
        cross_correlation_identity_loss(
            overlap_embedding_left,
            overlap_embedding_right,
        )
    )
    if config.alignment_objective == "both":
        total = token_loss + embedding_loss
    elif config.alignment_objective == "timestamp_token_only":
        total = token_loss
    elif config.alignment_objective == "interval_only":
        total = embedding_loss
    else:
        raise ValueError(
            "temporal_overlap_loss does not implement context_only; "
            "fit_encoder handles that matched-budget control directly"
        )
    diagnostics = {
        "loss": float(total.detach().cpu()),
        "token_cc": float(token_loss.detach().cpu()),
        "token_cc_diagonal": float(token_diagonal.detach().cpu()),
        "token_redundancy": float(token_redundancy.detach().cpu()),
        "overlap_embedding_cc": float(embedding_loss.detach().cpu()),
        "overlap_embedding_cc_diagonal": float(embedding_diagonal.detach().cpu()),
        "embedding_redundancy": float(embedding_redundancy.detach().cpu()),
        "aligned_tokens": int(left_stop - left_start),
    }
    return total, diagnostics


def extract_embeddings(
    model: StageEncoder,
    values: np.ndarray,
    starts: np.ndarray,
    device: torch.device,
    config: StageConfig,
    *,
    normalize: bool = True,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    starts = np.asarray(starts, dtype=np.int64)
    with torch.inference_mode():
        for offset in range(0, len(starts), int(config.embedding_batch_size)):
            current = starts[offset : offset + int(config.embedding_batch_size)]
            batch = window_batch(values, current, config.patch_size).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            _, embedding = model(batch)
            if normalize:
                embedding = F.normalize(embedding, dim=1)
            rows.append(embedding.cpu().numpy().astype(np.float32, copy=False))
    if not rows:
        raise ValueError("no embeddings were extracted")
    result = np.concatenate(rows, axis=0)
    if not np.isfinite(result).all():
        raise FloatingPointError("encoder produced non-finite embeddings")
    return result


def two_means_split(
    values: np.ndarray,
    members: np.ndarray,
    *,
    max_iter: int = 12,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Deterministic two-means initialized by two distant observed rows."""

    members = np.asarray(members, dtype=np.int64)
    if len(members) < 2:
        return None
    subset = values[members].astype(np.float64, copy=False)
    center = subset.mean(axis=0)
    first = int(np.argmax(np.sum((subset - center) ** 2, axis=1)))
    second = int(np.argmax(np.sum((subset - subset[first]) ** 2, axis=1)))
    if first == second:
        return None
    centers = np.stack([subset[first], subset[second]])
    prior_labels: np.ndarray | None = None
    for _ in range(int(max_iter)):
        distance = np.sum((subset[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distance, axis=1)
        if not np.any(labels == 0) or not np.any(labels == 1):
            return None
        if prior_labels is not None and np.array_equal(labels, prior_labels):
            break
        centers = np.stack([subset[labels == label].mean(axis=0) for label in (0, 1)])
        prior_labels = labels.copy()
    first_members = members[labels == 0]
    second_members = members[labels == 1]
    if not len(first_members) or not len(second_members):
        return None
    return first_members, second_members


def ball_sse(values: np.ndarray, members: np.ndarray) -> float:
    subset = values[np.asarray(members, dtype=np.int64)].astype(np.float64, copy=False)
    center = subset.mean(axis=0)
    return float(np.sum((subset - center[None, :]) ** 2))


def split_improves_bic(
    values: np.ndarray,
    parent: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> bool:
    """Compare one versus two isotropic normal-support cells.

    The likelihood term rewards a real compactness reduction while the BIC
    parameter term charges for an additional center, scale, and mixture mass.
    This prevents the always-split behaviour of an unpenalized inverse-radius
    density rule and leaves the final number of cells data dependent.
    """

    count = int(len(parent))
    dimensions = int(values.shape[1])
    parent_sse = max(ball_sse(values, parent), 1e-12)
    child_sse = max(ball_sse(values, left) + ball_sse(values, right), 1e-12)
    observations = max(1, count * dimensions)
    parent_parameters = dimensions + 1
    child_parameters = 2 * (dimensions + 1) + 1
    parent_bic = observations * math.log(parent_sse / observations) + (
        parent_parameters * math.log(max(2, count))
    )
    child_bic = observations * math.log(child_sse / observations) + (
        child_parameters * math.log(max(2, count))
    )
    return bool(child_bic + 1e-9 < parent_bic)


@dataclass
class GranularPartition:
    balls: list[np.ndarray]
    initial_k: int
    final_k: int
    rounds: int
    source_rows: int


def build_granular_partition(
    values: np.ndarray,
    config: StageConfig,
    *,
    seed: int,
) -> GranularPartition:
    """Construct a natural-K partition without importing any project GB code.

    The construction starts from a deterministic ``floor(sqrt(N))`` coarse
    cover, then recursively accepts only binary refinements that improve a
    compactness BIC after paying for the extra support cell.  Unlike the frozen
    V1 implementation, valid small terminal balls are not deleted afterward.
    """

    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("granular partition expects at least two finite rows")
    count = len(x)
    initial_k = max(1, min(int(math.sqrt(count)), count))
    if count > 30000:
        initializer = MiniBatchKMeans(
            n_clusters=initial_k,
            random_state=int(seed),
            n_init=3,
            batch_size=min(4096, count),
            max_iter=100,
            reassignment_ratio=0.0,
        )
    else:
        initializer = KMeans(
            n_clusters=initial_k,
            random_state=int(seed),
            n_init=5,
            max_iter=200,
            algorithm="lloyd",
        )
    labels = initializer.fit_predict(x)
    balls = [
        np.flatnonzero(labels == label).astype(np.int64)
        for label in range(initial_k)
        if np.any(labels == label)
    ]
    rounds = 0
    for round_index in range(int(config.gb_max_rounds)):
        changed = False
        updated: list[np.ndarray] = []
        for ball in balls:
            if len(ball) < int(config.gb_min_split):
                updated.append(ball)
                continue
            children = two_means_split(x, ball)
            if children is None:
                updated.append(ball)
                continue
            left, right = children
            minimum_child = max(2, int(config.gb_min_split) // 4)
            if (
                len(left) >= minimum_child
                and len(right) >= minimum_child
                and split_improves_bic(x, ball, left, right)
            ):
                updated.extend((left, right))
                changed = True
            else:
                updated.append(ball)
        balls = updated
        rounds = round_index + 1
        if not changed:
            break
    else:
        raise RuntimeError("granular splitting did not converge")
    balls = sorted(balls, key=lambda members: int(np.min(members)))
    return GranularPartition(
        balls=balls,
        initial_k=int(initial_k),
        final_k=int(len(balls)),
        rounds=int(rounds),
        source_rows=int(count),
    )


def select_gb_rows(total_rows: int, maximum: int) -> np.ndarray:
    if total_rows <= maximum:
        return np.arange(total_rows, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, total_rows - 1, int(maximum))).astype(np.int64)
    )


def build_gb_exemplar_memory(
    embeddings: np.ndarray,
    config: StageConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, GranularPartition]:
    selected = select_gb_rows(len(embeddings), int(config.max_gb_rows))
    source = np.asarray(embeddings[selected], dtype=np.float32)
    partition = build_granular_partition(source, config, seed=int(seed))
    representatives: list[int] = []
    for ball in partition.balls:
        members = source[ball]
        center = members.mean(axis=0)
        local = int(np.argmin(np.sum((members - center[None, :]) ** 2, axis=1)))
        representatives.append(int(ball[local]))
    representatives_array = np.asarray(representatives, dtype=np.int64)
    memory = source[representatives_array]
    memory = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-12)
    return memory.astype(np.float32), partition


def sample_from_partition(
    partition: GranularPartition,
    eligible_starts: np.ndarray,
    batch_size: int,
    power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled, _ = sample_from_partition_with_ids(
        partition,
        eligible_starts,
        batch_size,
        power,
        rng,
    )
    return sampled


def sample_from_partition_with_ids(
    partition: GranularPartition,
    eligible_starts: np.ndarray,
    batch_size: int,
    power: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample rows and expose region ids for mechanism diagnostics."""

    sizes = np.asarray([len(ball) for ball in partition.balls], dtype=np.float64)
    probability = np.power(sizes, float(power))
    probability /= probability.sum()
    ball_ids = rng.choice(len(partition.balls), size=int(batch_size), p=probability)
    local_rows = np.empty(int(batch_size), dtype=np.int64)
    for position, ball_id in enumerate(ball_ids):
        ball = partition.balls[int(ball_id)]
        local_rows[position] = int(ball[int(rng.integers(0, len(ball)))])
    return eligible_starts[local_rows], np.asarray(ball_ids, dtype=np.int64)


def resolve_training_steps(
    eligible_count: int,
    batch_size: int,
    requested_steps: int,
) -> tuple[int, bool]:
    """Cap undersized-series updates at ``ceil(N / batch)``."""

    if eligible_count < 1 or batch_size < 1 or requested_steps < 1:
        raise ValueError("training-step inputs must be positive")
    small_data_update_cap = eligible_count < 2 * batch_size
    if not small_data_update_cap:
        return requested_steps, False
    effective_steps = min(
        requested_steps,
        max(1, math.ceil(eligible_count / float(batch_size))),
    )
    return effective_steps, True


def fit_encoder(
    values: np.ndarray,
    model: StageEncoder,
    device: torch.device,
    config: StageConfig,
) -> dict[str, Any]:
    seed_everything(config.seed)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    max_delta = max(config.overlap_deltas)
    eligible_count = len(values) - int(config.patch_size) - int(max_delta) + 1
    if eligible_count < 2:
        raise ValueError("training prefix is too short for overlap training")
    requested_steps = int(config.steps)
    effective_steps, small_data_update_cap = resolve_training_steps(
        eligible_count,
        int(config.batch_size),
        requested_steps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, effective_steps)
    )
    eligible_starts = np.arange(eligible_count, dtype=np.int64)
    rng = np.random.default_rng(int(config.seed))
    gb_activation_step = int(round(effective_steps * config.gb_activation_fraction))
    gb_activation_step = min(max(1, gb_activation_step), max(1, effective_steps - 1))
    partition: GranularPartition | None = None
    partition_starts: np.ndarray | None = None
    partition_sample_counts: np.ndarray | None = None
    history: list[dict[str, Any]] = []
    peak_grad_norm = 0.0
    started = time.perf_counter()

    for step in range(effective_steps):
        if step == gb_activation_step:
            warm_local_rows = select_gb_rows(
                len(eligible_starts), int(config.max_gb_rows)
            )
            partition_starts = eligible_starts[warm_local_rows]
            warm_embeddings = extract_embeddings(
                model,
                values,
                partition_starts,
                device,
                config,
                normalize=True,
            )
            partition = build_granular_partition(
                warm_embeddings, config, seed=int(config.seed) + 17
            )
            partition_sample_counts = np.zeros(len(partition.balls), dtype=np.int64)
            model.train()
        if partition is None:
            starts = rng.choice(
                eligible_starts,
                size=int(config.batch_size),
                replace=len(eligible_starts) < int(config.batch_size),
            )
        else:
            if partition_starts is None:
                raise AssertionError("partition starts are missing")
            starts, sampled_ball_ids = sample_from_partition_with_ids(
                partition,
                partition_starts,
                config.batch_size,
                config.gb_sampling_power,
                rng,
            )
            if partition_sample_counts is None:
                raise AssertionError("partition sample counts are missing")
            partition_sample_counts += np.bincount(
                sampled_ball_ids,
                minlength=len(partition.balls),
            )
        delta = int(config.overlap_deltas[step % len(config.overlap_deltas)])
        left = window_batch(values, starts, config.patch_size).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        right = window_batch(values, starts + delta, config.patch_size).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        tokens_left = model.encode_tokens(left)
        tokens_right = model.encode_tokens(right)
        trim = int(config.overlap_trim)
        left_overlap = tokens_left[:, delta + trim : config.patch_size - trim]
        right_overlap = tokens_right[:, trim : config.patch_size - delta - trim]
        overlap_embedding_left = model.embed_tokens(left_overlap)
        overlap_embedding_right = model.embed_tokens(right_overlap)
        if config.alignment_objective == "context_only":
            context_left = model.embed_tokens(tokens_left)
            context_right = model.embed_tokens(tokens_right)
            loss, diagonal, redundancy = cross_correlation_identity_loss(
                context_left,
                context_right,
            )
            diagnostics = {
                "loss": float(loss.detach().cpu()),
                "token_cc": 0.0,
                "token_cc_diagonal": 0.0,
                "token_redundancy": 0.0,
                "overlap_embedding_cc": float(loss.detach().cpu()),
                "overlap_embedding_cc_diagonal": float(diagonal.detach().cpu()),
                "embedding_redundancy": float(redundancy.detach().cpu()),
                "aligned_tokens": 0,
            }
        else:
            loss, diagnostics = temporal_overlap_loss(
                tokens_left,
                tokens_right,
                overlap_embedding_left,
                overlap_embedding_right,
                delta,
                config,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
        )
        peak_grad_norm = max(peak_grad_norm, grad_norm)
        optimizer.step()
        scheduler.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == effective_steps:
            history.append(
                {
                    "step": int(step + 1),
                    "delta": int(delta),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_norm),
                    "sampling": "uniform" if partition is None else "gb_tempered",
                    **diagnostics,
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "elapsed_seconds": float(time.perf_counter() - started),
        "gb_activation_step": int(gb_activation_step),
        "requested_steps": int(requested_steps),
        "total_steps": int(effective_steps),
        "small_data_update_cap": bool(small_data_update_cap),
        "peak_grad_norm": float(peak_grad_norm),
        "history": history,
        "training_gb": (
            None
            if partition is None
            else {
                "initial_k": partition.initial_k,
                "final_k": partition.final_k,
                "rounds": partition.rounds,
                "rows": partition.source_rows,
                "region_sizes": [int(len(ball)) for ball in partition.balls],
                "region_sample_counts": (
                    []
                    if partition_sample_counts is None
                    else [int(item) for item in partition_sample_counts]
                ),
                "sampling_power": float(config.gb_sampling_power),
            }
        ),
    }


def score_embeddings(
    queries: np.ndarray,
    memory: np.ndarray,
    device: torch.device,
    config: StageConfig,
) -> np.ndarray:
    if len(memory) < 1:
        raise ValueError("memory is empty")
    top_k = min(int(config.top_k), len(memory))
    memory_tensor = torch.from_numpy(np.asarray(memory, dtype=np.float32)).to(device)
    memory_tensor = F.normalize(memory_tensor, dim=1)
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(queries), int(config.score_batch_size)):
            query = torch.from_numpy(
                np.asarray(queries[offset : offset + int(config.score_batch_size)], dtype=np.float32)
            ).to(device)
            query = F.normalize(query, dim=1)
            best = torch.full(
                (len(query), top_k),
                float("inf"),
                device=device,
                dtype=query.dtype,
            )
            block_size = int(config.memory_score_block_size)
            for memory_offset in range(0, len(memory_tensor), block_size):
                block = memory_tensor[memory_offset : memory_offset + block_size]
                squared_distance = (2.0 - 2.0 * (query @ block.T)).clamp_min_(0.0)
                block_k = min(top_k, len(block))
                block_best = torch.topk(
                    squared_distance,
                    k=block_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
                best = torch.topk(
                    torch.cat([best, block_best], dim=1),
                    k=top_k,
                    dim=1,
                    largest=False,
                    sorted=False,
                ).values
            output.append(best.mean(dim=1).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def aggregate_patch_scores(patch_scores: np.ndarray, points: int, patch_size: int) -> np.ndarray:
    scores = np.asarray(patch_scores, dtype=np.float64)
    if len(scores) != points - int(patch_size) + 1:
        raise ValueError("patch scores do not cover the complete series")
    total_difference = np.zeros(points + 1, dtype=np.float64)
    count_difference = np.zeros(points + 1, dtype=np.float64)
    starts = np.arange(len(scores), dtype=np.int64)
    np.add.at(total_difference, starts, scores)
    np.add.at(total_difference, starts + int(patch_size), -scores)
    np.add.at(count_difference, starts, 1.0)
    np.add.at(count_difference, starts + int(patch_size), -1.0)
    total = np.cumsum(total_difference[:-1])
    count = np.cumsum(count_difference[:-1])
    if np.any(count <= 0):
        raise AssertionError("at least one point is not covered by a patch")
    return total / count


def estimate_sliding_window(values: np.ndarray) -> int:
    if values.shape[1] > 1:
        return 100
    from scipy.signal import argrelextrema
    from statsmodels.tsa.stattools import acf

    data = values[: min(20000, len(values)), 0]
    try:
        correlation = acf(data, nlags=400, fft=True)[3:]
        local_maxima = argrelextrema(correlation, np.greater)[0]
        ranked = np.argsort(correlation[local_maxima])[::-1]
        position = int(local_maxima[int(ranked[0])])
        return 125 if position < 3 or position > 300 else position + 3
    except Exception:
        return 125


def official_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    sliding_window: int,
    metrics_root: Path | None,
) -> dict[str, float]:
    if metrics_root is None:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "AUC-PR": float(average_precision_score(labels, scores)),
            "AUC-ROC": float(roc_auc_score(labels, scores)),
        }
    root = metrics_root.resolve()
    # The official PaAno/TSB-AD evaluator keeps the top-level ``affiliation``
    # package inside its PaAno directory.  Add both roots for evaluation only;
    # neither path is used by the model, training, memory, or scoring code.
    for candidate in (root, root / "PaAno"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    module = importlib.import_module("PaAno.utils.metrics")
    result = module.get_metrics(
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        slidingWindow=int(sliding_window),
        pred=None,
        version="opt",
        thre=250,
    )
    return {name: float(result[name]) for name in SIX_METRICS}


def evaluate_series(
    path: Path,
    output: Path,
    device: torch.device,
    config: StageConfig,
    metrics_root: Path | None,
    *,
    save_checkpoint: bool,
    save_scores: bool,
) -> dict[str, Any]:
    values, train_index, label_column = load_series(path)
    if train_index < config.patch_size + max(config.overlap_deltas):
        raise ValueError(f"training prefix is too short in {path.name}")
    seed_everything(config.seed)
    model = StageEncoder(values.shape[1], config).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training = fit_encoder(values[:train_index], model, device, config)
    raw_starts = np.arange(train_index - config.patch_size + 1, dtype=np.int64)
    full_starts = np.arange(len(values) - config.patch_size + 1, dtype=np.int64)
    train_raw = extract_embeddings(
        model, values[:train_index], raw_starts, device, config, normalize=False
    )
    train_unit = train_raw / np.maximum(np.linalg.norm(train_raw, axis=1, keepdims=True), 1e-12)
    full_unit = extract_embeddings(model, values, full_starts, device, config, normalize=True)

    memory_started = time.perf_counter()
    gb_memory, final_partition = build_gb_exemplar_memory(
        train_unit, config, seed=int(config.seed) + 29
    )
    gb_build_seconds = float(time.perf_counter() - memory_started)
    sliding_window = estimate_sliding_window(values)
    started = time.perf_counter()
    patch_score = score_embeddings(full_unit, gb_memory, device, config)
    point_score = aggregate_patch_scores(patch_score, len(values), config.patch_size)
    query_seconds = float(time.perf_counter() - started)
    labels = load_labels_after_scoring(path, label_column, len(values))
    metrics = official_metrics(point_score, labels, sliding_window, metrics_root)
    methods = [
        {
            "method": "STAGE",
            **metrics,
            "memory_rows": int(len(gb_memory)),
            "memory_ratio": float(len(gb_memory) / len(train_unit)),
            "query_seconds": query_seconds,
            "query_ms_per_patch": float(1000.0 * query_seconds / len(full_unit)),
        }
    ]
    artifact = output / "artifacts" / path.stem
    if save_checkpoint or save_scores:
        artifact.mkdir(parents=True, exist_ok=True)
    if save_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "in_channels": int(values.shape[1]),
            },
            artifact / "stage_encoder.pt",
        )
    if save_scores:
        np.savez_compressed(
            artifact / "point_scores.npz",
            STAGE=point_score.astype(np.float32),
        )
    record = {
        "file": path.name,
        "points": int(len(values)),
        "channels": int(values.shape[1]),
        "train_index": int(train_index),
        "training_prefix_policy": "filename_declared_label_blind",
        "training_prefix_anomaly_count": int(np.count_nonzero(labels[:train_index])),
        "train_patches": int(len(train_unit)),
        "full_patches": int(len(full_unit)),
        "parameter_count": parameter_count,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "sliding_window": int(sliding_window),
        "training": training,
        "final_gb": {
            "initial_k": final_partition.initial_k,
            "final_k": final_partition.final_k,
            "rounds": final_partition.rounds,
            "source_rows": final_partition.source_rows,
            "build_seconds": gb_build_seconds,
        },
        "methods": methods,
    }
    write_json(output / "series" / f"{path.stem}.json", record)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def summarize(records: Sequence[Mapping[str, Any]], output: Path) -> pd.DataFrame:
    rows = [
        {"file": record["file"], **method}
        for record in records
        for method in record["methods"]
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics.csv", index=False)
    aggregations: dict[str, tuple[str, str]] = {
        metric: (metric, "mean") for metric in SIX_METRICS if metric in frame.columns
    }
    aggregations.update(
        memory_rows=("memory_rows", "mean"),
        memory_ratio=("memory_ratio", "mean"),
        query_ms_per_patch=("query_ms_per_patch", "mean"),
        series=("file", "nunique"),
    )
    means = frame.groupby("method", as_index=False).agg(**aggregations)
    means.to_csv(output / "method_means.csv", index=False)
    return means


def build_config(args: argparse.Namespace) -> StageConfig:
    return StageConfig(
        patch_size=int(args.patch_size),
        channels=int(args.channels),
        token_dim=int(args.token_dim),
        embedding_dim=int(args.embedding_dim),
        dilations=tuple(int(item) for item in args.dilations.split(",") if item),
        group_norm_groups=int(args.group_norm_groups),
        dropout=float(args.dropout),
        batch_size=int(args.batch_size),
        steps=int(args.steps),
        gb_activation_fraction=float(args.gb_activation_fraction),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        overlap_deltas=tuple(int(item) for item in args.overlap_deltas.split(",") if item),
        overlap_trim=int(args.overlap_trim),
        gb_min_split=int(args.gb_min_split),
        gb_max_rounds=int(args.gb_max_rounds),
        gb_sampling_power=float(args.gb_sampling_power),
        top_k=int(args.top_k),
        embedding_batch_size=int(args.embedding_batch_size),
        score_batch_size=int(args.score_batch_size),
        memory_score_block_size=int(args.memory_score_block_size),
        max_gb_rows=int(args.max_gb_rows),
        seed=int(args.seed),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-root", type=Path)
    parser.add_argument(
        "--allow-partial-metrics",
        action="store_true",
        help="Permit AUC-only diagnostics when the official evaluator is unavailable.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-physical-gpu", default="1")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dilations", default="1,2,4,8,4,2")
    parser.add_argument("--group-norm-groups", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--gb-activation-fraction",
        type=float,
        default=0.20,
        help="Fraction of updates completed before building the frozen training GB.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--overlap-deltas", default="24,48")
    parser.add_argument("--overlap-trim", type=int, default=8)
    parser.add_argument("--gb-min-split", type=int, default=4)
    parser.add_argument("--gb-max-rounds", type=int, default=64)
    parser.add_argument("--gb-sampling-power", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-batch-size", type=int, default=2048)
    parser.add_argument("--score-batch-size", type=int, default=2048)
    parser.add_argument("--memory-score-block-size", type=int, default=8192)
    parser.add_argument("--max-gb-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    config.validate()
    if args.metrics_root is None and not bool(args.allow_partial_metrics):
        raise ValueError(
            "formal runs require --metrics-root for the complete six-metric protocol"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if args.require_physical_gpu and visible != str(args.require_physical_gpu):
        raise RuntimeError(
            f"launch with CUDA_VISIBLE_DEVICES={args.require_physical_gpu}; got {visible!r}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.manifest is not None:
        files = load_manifest(args.manifest)
    elif args.files:
        files = [str(item) for item in args.files]
    else:
        raise ValueError("provide --manifest or --files")
    offset = int(args.offset)
    if offset < 0 or offset > len(files):
        raise ValueError("offset escapes the file list")
    if int(args.limit) >= 0:
        files = files[offset : offset + int(args.limit)]
    else:
        files = files[offset:]
    if not files:
        raise ValueError("no series selected")
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "series").mkdir(exist_ok=True)
    source = Path(__file__).resolve()
    run_config = {
        "method": "STAGE",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "data_root": str(args.data_root.resolve()),
        "manifest": None if args.manifest is None else str(args.manifest.resolve()),
        "manifest_sha256": (
            None if args.manifest is None else sha256_file(args.manifest.resolve())
        ),
        "files": files,
        "device": str(device),
        "cuda_visible_devices": visible,
        "metrics_root": None if args.metrics_root is None else str(args.metrics_root.resolve()),
        "model": asdict(config),
        "save_checkpoints": bool(args.save_checkpoints),
        "save_scores": bool(args.save_scores),
    }
    fingerprint_payload = json.dumps(
        json_ready(run_config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    run_config["protocol_fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
    config_path = output / "config.json"
    if args.resume and config_path.is_file():
        prior_config = json.loads(config_path.read_text(encoding="utf-8"))
        if prior_config.get("protocol_fingerprint") != run_config["protocol_fingerprint"]:
            raise ValueError("resume configuration/source fingerprint mismatch")
    else:
        write_json(config_path, run_config)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for position, filename in enumerate(files, 1):
        series_path = output / "series" / f"{Path(filename).stem}.json"
        if args.resume and series_path.is_file():
            record = json.loads(series_path.read_text(encoding="utf-8"))
        else:
            print(f"[{position}/{len(files)}] {filename}", flush=True)
            record = evaluate_series(
                args.data_root / filename,
                output,
                device,
                config,
                args.metrics_root,
                save_checkpoint=bool(args.save_checkpoints),
                save_scores=bool(args.save_scores),
            )
            gb_row = next(
                method for method in record["methods"] if method["method"] == "STAGE"
            )
            print(
                "  params=%d finalK=%d VUS-PR=%s elapsed=%.1fs"
                % (
                    record["parameter_count"],
                    record["final_gb"]["final_k"],
                    (
                        "NA"
                        if "VUS-PR" not in gb_row
                        else f"{float(gb_row['VUS-PR']):.6f}"
                    ),
                    record["training"]["elapsed_seconds"],
                ),
                flush=True,
            )
        records.append(record)
        means = summarize(records, output)
        print(means.to_string(index=False), flush=True)
    completed = {
        "series": len(records),
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_sha256": run_config["source_sha256"],
    }
    write_json(output / "completed.json", completed)
    print(json.dumps(completed, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
