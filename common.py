from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np


METRICS = (
    "VUS-PR",
    "VUS-ROC",
    "R-based-F1",
    "AUC-PR",
    "AUC-ROC",
    "Standard-F1",
)
ALIASES = {
    "VP": "VUS-PR",
    "VR": "VUS-ROC",
    "RF": "R-based-F1",
    "AP": "AUC-PR",
    "AR": "AUC-ROC",
    "PF": "Standard-F1",
}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def parse_train_index(file_name: str) -> int:
    parts = Path(file_name).stem.split("_")
    try:
        return int(parts[parts.index("tr") + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"cannot parse tr_<N> from {file_name}") from exc


def dataset_name(file_name: str) -> str:
    parts = Path(file_name).stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot parse dataset from {file_name}")
    return parts[1]


def load_stage_source(source: Path):
    spec = importlib.util.spec_from_file_location("stage_frozen_metrics", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def official_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    values: np.ndarray,
    stage_source: Path,
    paano_root: Path,
) -> tuple[dict[str, float], int]:
    scores = np.nan_to_num(
        np.asarray(scores, dtype=np.float64),
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
    module = load_stage_source(stage_source)
    sliding_window = int(module.estimate_sliding_window(values))
    result = module.official_metrics(scores, labels, sliding_window, paano_root)
    metrics = {name: float(result[name]) for name in METRICS}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("non-finite metric")
    return metrics, sliding_window


def complete_record(path: Path) -> bool:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        metrics = item.get("metrics", {})
        return not item.get("error") and all(
            name in metrics and math.isfinite(float(metrics[name])) for name in METRICS
        )
    except Exception:
        return False
