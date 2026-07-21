from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from common import atomic_json, complete_record, dataset_name, official_metrics, parse_train_index, seed_everything


METHODS = ("KMeansAD", "KNN", "PCA", "IForest", "LOF")


def windows(data: np.ndarray, length: int) -> np.ndarray:
    length = min(max(int(length), 1), len(data))
    view = np.lib.stride_tricks.sliding_window_view(data, length, axis=0)
    return view.reshape(len(data) - length + 1, -1)


def instance_zscore(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    return np.nan_to_num((data - mean) / np.where(std < 1e-8, 1.0, std))


def protocol_window_normalization(
    train_windows: np.ndarray,
    query_windows: np.ndarray,
    track: str,
) -> tuple[np.ndarray, np.ndarray]:
    if track == "U":
        mean = train_windows.mean(axis=0, keepdims=True)
        std = train_windows.std(axis=0, keepdims=True)
        safe = np.where(std < 1e-8, 1.0, std)
        return (
            np.nan_to_num((train_windows - mean) / safe),
            np.nan_to_num((query_windows - mean) / safe),
        )
    return instance_zscore(train_windows), instance_zscore(query_windows)


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
    args = parser.parse_args()
    target = args.output.resolve()
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
        values = frame.drop(columns=["Label"]).to_numpy(dtype=np.float32)
        labels = frame["Label"].to_numpy(dtype=np.int64)
        boundary = parse_train_index(args.file)
        train_mean = values[:boundary].mean(axis=0, keepdims=True)
        train_std = values[:boundary].std(axis=0, keepdims=True)
        normalized = np.nan_to_num((values - train_mean) / np.where(train_std < 1e-8, 1.0, train_std))

        import sys

        tsb_root = args.repo / "external" / "PAI" / "third_party" / "TSB-AD"
        sys.path.insert(0, str(tsb_root))
        from TSB_AD.utils.slidingWindows import find_length_rank

        if args.track == "U":
            periodicity = {"KMeansAD": 2, "KNN": 2, "PCA": 1, "IForest": 1, "LOF": 2}[args.method]
            window = min(max(int(find_length_rank(normalized[:boundary], rank=periodicity)), 1), boundary)
        elif args.method == "KMeansAD":
            window = min(40, boundary)
        elif args.method == "IForest":
            window = min(100, boundary)
        else:
            window = 1

        train_windows, query_windows = protocol_window_normalization(
            windows(normalized[:boundary], window),
            windows(normalized, window),
            args.track,
        )
        if args.method == "KMeansAD":
            clusters = min(10, len(train_windows))
            model = KMeans(n_clusters=clusters, n_init=10, random_state=args.seed)
            model.fit(train_windows)
            assignment = model.predict(query_windows)
            patch_scores = np.linalg.norm(query_windows - model.cluster_centers_[assignment], axis=1)
            hp = {
                "n_clusters": 10,
                "n_init": 10,
                "window": window,
                "periodicity": 2 if args.track == "U" else None,
                "official_name": "KMeansAD_U" if args.track == "U" else "KMeansAD",
            }
        elif args.method == "KNN":
            neighbors = min(50, len(train_windows))
            model = NearestNeighbors(n_neighbors=neighbors, metric="euclidean", n_jobs=1)
            model.fit(train_windows)
            distances, _ = model.kneighbors(query_windows)
            patch_scores = distances.mean(axis=1)
            hp = {
                "n_neighbors": 50,
                "aggregation": "mean",
                "window": window,
                "periodicity": 2 if args.track == "U" else None,
                "official_name": "Sub_KNN" if args.track == "U" else "KNN",
            }
        elif args.method == "IForest":
            estimators = 150 if args.track == "U" else 25
            max_features = 1.0 if args.track == "U" else 0.8
            model = IsolationForest(n_estimators=estimators, max_features=max_features, random_state=args.seed, n_jobs=1)
            model.fit(train_windows)
            patch_scores = -model.decision_function(query_windows)
            hp = {"n_estimators": estimators, "max_features": max_features, "window": window, "official_name": "Sub_IForest" if args.track == "U" else "IForest"}
        elif args.method == "LOF":
            configured_neighbors = 30 if args.track == "U" else 50
            metric = "minkowski" if args.track == "U" else "euclidean"
            neighbors = min(configured_neighbors, max(1, len(train_windows) - 1))
            model = LocalOutlierFactor(n_neighbors=neighbors, metric=metric, novelty=True, n_jobs=1)
            model.fit(train_windows)
            patch_scores = -model.score_samples(query_windows)
            hp = {"n_neighbors": configured_neighbors, "metric": metric, "novelty": True, "window": window, "official_name": "Sub_LOF" if args.track == "U" else "LOF"}
        else:
            feature_mean = train_windows.mean(axis=0, keepdims=True)
            feature_std = train_windows.std(axis=0, keepdims=True)
            safe_feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
            train_pca = np.nan_to_num((train_windows - feature_mean) / safe_feature_std)
            query_pca = np.nan_to_num((query_windows - feature_mean) / safe_feature_std)
            components = None if args.track == "U" else 0.25
            model = SklearnPCA(n_components=components, random_state=args.seed)
            model.fit(train_pca)
            selected = model.components_
            weights = np.maximum(np.nan_to_num(model.explained_variance_ratio_), 1e-12)
            pca_scores = np.sum(cdist(query_pca, selected) / weights.reshape(1, -1), axis=1)
            point_scores = to_points(np.nan_to_num(pca_scores), window, len(values))
            hp = {
                "window": window,
                "n_components": components,
                "weighted": True,
                "official_name": "Sub_PCA" if args.track == "U" else "PCA",
                "numeric_stability": "zero eigenvalue weights floored at 1e-12",
            }
            patch_scores = None

        if patch_scores is not None:
            point_scores = to_points(np.asarray(patch_scores, dtype=np.float64), window, len(values))
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
                "training_prefix_anomaly_count": int(np.count_nonzero(labels[:boundary])),
                "points": int(len(values)),
                "channels": int(values.shape[1]),
                "hyperparameters": hp,
                "normalization": "channel z-score fitted on training prefix; window instance z-score",
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
