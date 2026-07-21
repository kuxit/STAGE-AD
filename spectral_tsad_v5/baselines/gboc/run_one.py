from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings(
    "ignore",
    message="Number of distinct clusters.*smaller than n_clusters.*",
    category=ConvergenceWarning,
)


METRICS = ["VUS-PR", "VUS-ROC", "R-based-F1", "AUC-PR", "AUC-ROC", "Standard-F1"]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_previous_error(base: Path, stem: str) -> None:
    error_path = base / "errors" / f"{stem}.json"
    if not error_path.is_file():
        return
    history = base / "error_history"
    history.mkdir(parents=True, exist_ok=True)
    destination = history / f"{stem}.attempt_{time.time_ns()}.json"
    os.replace(error_path, destination)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one full-series GBOC reproduction record")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--track", choices=("U", "M"), required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--efficiency-only", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hp = dict(config["shared"])
    hp.update(config["tracks"][args.track])
    stem = Path(args.file).stem
    base = output / "Canonical" / "gboc" / args.track / f"seed_{args.seed}"
    target = base / "metrics_parts" / f"{stem}.json"
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
        except Exception:
            previous = {"error": "unreadable record"}
        if not previous.get("error") and all(name in previous.get("metrics", {}) for name in METRICS):
            print(f"skip complete {target}", flush=True)
            return 0

    started = time.time()
    record = {
        "subset": args.track,
        "track": args.track,
        "file": args.file,
        "method": "GBOC",
        "seed": args.seed,
        "stochastic": True,
        "normal_only": True,
        "config": config,
        "upstream_revision": "55a2a31e95055a638b5839c6b19335cbe02bd148",
        "error": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("formal GBOC run requires the single visible CUDA GPU")
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"single-GPU isolation violated: visible CUDA device count is {torch.cuda.device_count()}"
            )
        seed_everything(args.seed)

        data_path = repo / "data" / f"TSB-AD-{args.track}" / args.file
        frame = pd.read_csv(data_path).dropna()
        if "Label" not in frame.columns:
            raise ValueError(f"Label column missing in {data_path}")
        raw = frame.drop(columns=["Label"]).to_numpy(dtype=np.float32)
        boundary = int(stem.split("_")[-3])
        if not (hp["window_size"] * 2 <= boundary < len(raw)):
            raise ValueError(
                f"invalid normal-only boundary {boundary} for length {len(raw)} and window {hp['window_size']}"
            )
        train_raw = raw[:boundary]
        train_mean = train_raw.mean(axis=0, keepdims=True)
        train_std = train_raw.std(axis=0, keepdims=True)
        safe_std = np.where(train_std < 1e-8, 1.0, train_std)
        normalized = np.nan_to_num((raw - train_mean) / safe_std).astype(np.float32)

        scripts = repo / "spectral_tsad_v5" / "scripts"
        sys.path.insert(0, str(scripts))
        from efficiency_monitor import EfficiencyMonitor, atomic_json as atomic_efficiency_json, tagged_record

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from protocol_adapter import build_protocol_gboc

        ProtocolGBOC = build_protocol_gboc(upstream)
        model = ProtocolGBOC(
            win_size=int(hp["window_size"]),
            feats=normalized.shape[1],
            batch_size=int(hp["batch_size"]),
            epochs=int(hp["outer_epochs"]),
            patience=int(hp["early_stopping_patience"]),
            lr=float(hp["learning_rate"]),
            validation_size=float(hp["validation_fraction"]),
            hidden_dim=int(hp["hidden_dim"]),
            num_layers=int(hp["num_layers"]),
            dis_mode=str(hp["discrimination_mode"]),
            alpha=float(hp["alpha"]),
        )
        train_monitor = EfficiencyMonitor(include_gpu=True).start()
        try:
            model.fit(normalized[:boundary])
        finally:
            train_efficiency = train_monitor.stop()

        inference_monitor = EfficiencyMonitor(include_gpu=True).start()
        try:
            scores = np.asarray(model.decision_function(normalized), dtype=float).reshape(-1)
        finally:
            inference_efficiency = inference_monitor.stop()
        scores = np.nan_to_num(scores, nan=0.0, posinf=np.finfo(float).max, neginf=0.0)
        score_path = base / "scores" / f"{stem}.npy"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_tmp = score_path.with_suffix(f".tmp.{os.getpid()}.npy")
        np.save(score_tmp, scores)
        os.replace(score_tmp, score_path)

        # Labels enter only here, after fitting and scoring are complete.
        labels = frame["Label"].to_numpy(dtype=int)
        tsb_root = repo / "external" / "PAI" / "third_party" / "TSB-AD"
        if not tsb_root.exists():
            tsb_root = repo / "external" / "TSB-AD"
        sys.path.insert(0, str(tsb_root))
        from TSB_AD.evaluation.metrics import get_metrics
        from TSB_AD.utils.slidingWindows import find_length_rank

        metric_monitor = EfficiencyMonitor(include_gpu=False).start()
        try:
            values = get_metrics(
                scores,
                labels,
                slidingWindow=find_length_rank(normalized, rank=1),
                pred=None,
            )
        finally:
            metric_efficiency = metric_monitor.stop()

        efficiency_paths = {}
        for phase, efficiency in (
            ("train", train_efficiency),
            ("inference", inference_efficiency),
            ("metric", metric_efficiency),
        ):
            path = base / "efficiency" / f"{stem}.{phase}.json"
            payload = tagged_record(
                efficiency,
                method="GBOC",
                track=args.track,
                seed=args.seed,
                phase=phase,
                scope="per_series",
                files_count=1,
                points_count=len(raw),
                file=args.file,
            )
            payload.update(
                {
                    "paper_efficiency_valid": bool(args.efficiency_only),
                    "invalid_reason": None if args.efficiency_only else "accuracy seeds may share non-GPU host load",
                }
            )
            atomic_efficiency_json(path, payload)
            efficiency_paths[phase] = str(path)

        record.update(
            {
                "train_boundary": boundary,
                "series_length": len(raw),
                "channels": raw.shape[1],
                "data_sha256": sha256(data_path),
                "normalization": {
                    "fit_scope": "normal-only training prefix",
                    "mean": train_mean.reshape(-1).astype(float).tolist(),
                    "std": train_std.reshape(-1).astype(float).tolist(),
                },
                "retained_training_ball_count": int(len(model.train_centers_)),
                "best_validation_loss": float(model.best_validation_loss_),
                "score_path": str(score_path),
                "score_sha256": sha256(score_path),
                "efficiency_records": efficiency_paths,
                "metrics": {name: float(values[name]) for name in METRICS},
                "runtime_s": time.time() - started,
                "completed_at_epoch_s": time.time(),
            }
        )
        archive_previous_error(base, stem)
        atomic_json(target, record)
        print(json.dumps({"status": "complete", "record": str(target)}, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "runtime_s": time.time() - started,
            }
        )
        archive_previous_error(base, stem)
        error_path = base / "errors" / f"{stem}.json"
        atomic_json(error_path, record)
        print(json.dumps({"status": "error", "record": str(error_path), "error": record["error"]}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
