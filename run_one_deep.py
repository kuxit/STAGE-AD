from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

from common import METRICS, atomic_json, dataset_name, official_metrics, parse_train_index, seed_everything


METHOD_MAP = {
    "PatchTST": "PatchTST",
    "AnomalyTransformer": "AnomalyTransformer",
    "TimesNet": "TimesNet",
    "TranAD": "TranAD",
    "USAD": "USAD",
    "OmniAnomaly": "OmniAnomaly",
    "FITS": "FITS",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=tuple(METHOD_MAP))
    parser.add_argument("--track", required=True, choices=("U", "M"))
    parser.add_argument("--file", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage-source", required=True, type=Path)
    parser.add_argument("--paano-root", required=True, type=Path)
    parser.add_argument("--require-physical-gpu", required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible != args.require_physical_gpu:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={args.require_physical_gpu}, got {visible!r}")
    target = args.output.resolve()
    if target.exists():
        from common import complete_record

        if complete_record(target):
            return 0

    started = time.time()
    record = {
        "method": args.method,
        "track": args.track,
        "dataset": dataset_name(args.file),
        "file": args.file,
        "seed": int(args.seed),
        "training_prefix_policy": "filename_declared_label_blind",
        "error": None,
    }
    try:
        seed_everything(args.seed)
        data_path = args.repo / "data" / f"TSB-AD-{args.track}" / args.file
        frame = pd.read_csv(data_path)
        if "Label" not in frame.columns:
            raise ValueError("Label column missing")
        values = frame.drop(columns=["Label"]).to_numpy(dtype=np.float32)
        labels = frame["Label"].to_numpy(dtype=np.int64)
        if not np.isfinite(values).all():
            raise ValueError("non-finite input")
        boundary = parse_train_index(args.file)
        if not 0 < boundary < len(values):
            raise ValueError(f"invalid train boundary {boundary}")

        tsb_root = args.repo / "external" / "PAI" / "third_party" / "TSB-AD"
        sys.path.insert(0, str(tsb_root))
        from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict, Optimal_Uni_algo_HP_dict
        from TSB_AD.model_wrapper import run_Semisupervise_AD

        official_name = METHOD_MAP[args.method]
        hp_dict = Optimal_Uni_algo_HP_dict if args.track == "U" else Optimal_Multi_algo_HP_dict
        hyperparameters = dict(hp_dict.get(official_name, {}))
        previous_cwd = Path.cwd()
        with tempfile.TemporaryDirectory(prefix=f"{args.method}-{args.file}-") as temporary:
            os.chdir(temporary)
            try:
                scores = run_Semisupervise_AD(
                    official_name,
                    values[:boundary],
                    values,
                    **hyperparameters,
                )
            finally:
                os.chdir(previous_cwd)
        if not isinstance(scores, np.ndarray):
            raise RuntimeError(f"official wrapper returned {str(scores)[:300]}")
        scores = np.nan_to_num(np.asarray(scores, dtype=np.float64).reshape(-1))
        if len(scores) != len(labels):
            raise ValueError(f"score length {len(scores)} != {len(labels)}")
        metrics, sliding_window = official_metrics(
            scores,
            labels,
            values,
            args.stage_source,
            args.paano_root.parent,
        )
        record.update(
            {
                "train_index": int(boundary),
                "training_prefix_anomaly_count": int(np.count_nonzero(labels[:boundary])),
                "points": int(len(values)),
                "channels": int(values.shape[1]),
                "official_tsb_name": official_name,
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
