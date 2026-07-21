"""Leakage-safe adapter around the official AAAI 2026 GBOC implementation.

The network, granular-ball construction, joint loss, optimizer, scheduler, and
training loop follow the released implementation.  The adapter changes only
the benchmark boundary conditions needed for a PaAno-compatible comparison:
normalization statistics and granular-ball centers are learned from the
official normal-only training prefix and are reused unchanged at inference.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

import numpy as np


def add_upstream_to_path(upstream: Path) -> None:
    upstream = upstream.resolve()
    if not (upstream / "models" / "GBOC.py").is_file():
        raise FileNotFoundError(f"official GBOC source is missing: {upstream}")
    text = str(upstream)
    if text not in sys.path:
        sys.path.insert(0, text)


def build_protocol_gboc(upstream: Path):
    """Return the adapted class after loading the pinned official source."""

    add_upstream_to_path(upstream)
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from models.GBOC import GBOC, calculate_anomaly_scores
    import granularball_computing.granularball_generation as official_granular_balls
    import utils.dataset as official_dataset
    from utils.dataset import GB_ReconstructDataset

    def safe_assign_granular_ball_labels(features, centers):
        """Official Euclidean assignment with round-off-safe squared distances."""
        features = np.asarray(features)
        centers = np.asarray(centers)
        squared_distances = (
            np.sum(features ** 2, axis=1, keepdims=True)
            + np.sum(centers ** 2, axis=1, keepdims=True).T
            - 2.0 * np.dot(features, centers.T)
        )
        return np.argmin(np.sqrt(np.maximum(squared_distances, 0.0)), axis=1)

    # The released scoring function already applies the same clamp.  Patch the
    # training/validation label helper so round-off cannot create NaN labels.
    official_dataset.assign_granular_ball_labels = safe_assign_granular_ball_labels

    original_divide_gb_k = official_granular_balls.divide_gb_k

    def safe_divide_gb_k(data, indices, k):
        # Released KMeans can return fewer distinct labels than requested when
        # embeddings are duplicated. Its empty cluster is then np.array([])
        # with float dtype and cannot index NumPy arrays. Empty clusters contain
        # no samples and have no mathematical contribution, so discard them and
        # make the remaining index dtype explicit.
        clusters = original_divide_gb_k(data, indices, k)
        return [
            np.asarray(cluster, dtype=np.int64)
            for cluster in clusters
            if len(cluster) > 0
        ]

    official_granular_balls.divide_gb_k = safe_divide_gb_k

    original_split_ball_2 = official_granular_balls.spilt_ball_2

    def safe_split_ball_2(data, membership):
        first, second = original_split_ball_2(data, membership)
        first = np.asarray(first, dtype=np.int64)
        second = np.asarray(second, dtype=np.int64)
        original = np.asarray(membership, dtype=np.int64)
        # The release represents a failed two-way split as (original,
        # original).  Downstream code mistakes that for two valid children and
        # can duplicate the ball forever. Represent "no split" explicitly with
        # an empty second child; division_2_2 already handles this by retaining
        # the parent.
        if (
            len(first) == len(original)
            and len(second) == len(original)
            and np.array_equal(np.sort(first), np.sort(original))
            and np.array_equal(np.sort(second), np.sort(original))
        ):
            return original, np.asarray([], dtype=np.int64)
        return first, second

    def safe_normalized_ball(data, gb_list, radius_detect):
        retained = []
        for membership in gb_list:
            membership = np.asarray(membership, dtype=np.int64)
            if len(membership) <= 8:
                continue
            first, second = safe_split_ball_2(data[membership], membership)
            unsplittable = len(first) == 0 or len(second) == 0
            if (
                unsplittable
                or official_granular_balls.get_radius(data[membership], membership)
                <= 2 * radius_detect
            ):
                retained.append(membership)
            else:
                retained.extend([first, second])
        # The method needs at least one normal-support center. In the degenerate
        # case where every ball is pruned by the released <=8 rule, retain only
        # the largest support ball instead of returning an undefined empty
        # memory.
        if not retained and gb_list:
            retained = [
                np.asarray(max(gb_list, key=len), dtype=np.int64)
            ]
        return retained

    official_granular_balls.spilt_ball_2 = safe_split_ball_2
    official_granular_balls.normalized_ball = safe_normalized_ball

    class ProtocolGBOC(GBOC):
        def fit(self, data):
            data = np.asarray(data, dtype=np.float32)
            if data.ndim != 2:
                raise ValueError(f"expected [time, channels], received {data.shape}")
            if len(data) < 2 * self.win_size:
                raise ValueError(
                    f"normal-only prefix length {len(data)} is too short for window {self.win_size}"
                )

            split = int((1.0 - self.validation_size) * len(data))
            split = min(max(split, self.win_size), len(data) - self.win_size)
            train_data, valid_data = data[:split], data[split:]
            best_val_loss = float("inf")
            best_model_state = None

            for outer_epoch in range(1, self.epochs + 1):
                patience_counter = 0
                epoch_best_val_loss = float("inf")
                epoch_best_model = copy.deepcopy(self.model.state_dict())

                trainset = GB_ReconstructDataset(
                    train_data,
                    self.model,
                    dis_mode=self.dis_mode,
                    window_size=self.win_size,
                    normalize=False,
                )
                if len(trainset) == 0 or len(trainset.gb_centers) == 0:
                    raise RuntimeError("official granular-ball construction returned no training centers")
                train_loader = DataLoader(
                    trainset, batch_size=self.batch_size, shuffle=True
                )
                validset = GB_ReconstructDataset(
                    valid_data,
                    self.model,
                    dis_mode=self.dis_mode,
                    gb_centers=trainset.gb_centers,
                    window_size=self.win_size,
                    normalize=False,
                )
                valid_loader = DataLoader(
                    validset, batch_size=self.batch_size, shuffle=False
                )
                centers = torch.as_tensor(
                    trainset.gb_centers, dtype=torch.float32, device=self.device
                )

                for _inner_epoch in range(1, 11):
                    self.model.train(True)
                    training_loss = 0.0
                    final_loss = None
                    for x, _target, gb_y in train_loader:
                        x = torch.nan_to_num(x).to(self.device).permute(1, 0, 2)
                        gb_y = gb_y.to(self.device)
                        batch = x.shape[1]
                        elem = x[-1, :, :].view(1, batch, self.feats)
                        self.optimizer.zero_grad()
                        features, reconstruction = self.model(x, elem)
                        # The released code passes [1, B, C] against [B, C].
                        # Removing the leading singleton is mathematically
                        # identical to that broadcast and prevents one warning
                        # (and a log write) for every training batch.
                        reconstruction_loss = self.criterion(reconstruction, elem.squeeze(0))
                        geometry_loss = F.mse_loss(features, centers[gb_y])
                        final_loss = (
                            self.alpha * reconstruction_loss
                            + (1.0 - self.alpha) * geometry_loss
                        )
                        final_loss.backward(retain_graph=True)
                        self.optimizer.step()
                        training_loss += float(final_loss.detach().cpu())

                    if final_loss is None or torch.isnan(final_loss):
                        break

                    if len(valid_loader):
                        self.model.eval()
                        validation_loss = 0.0
                        with torch.no_grad():
                            for x, _target, gb_y in valid_loader:
                                x = torch.nan_to_num(x).to(self.device).permute(1, 0, 2)
                                gb_y = gb_y.to(self.device)
                                batch = x.shape[1]
                                elem = x[-1, :, :].view(1, batch, self.feats)
                                features, reconstruction = self.model(x, elem)
                                reconstruction_loss = self.criterion(reconstruction, elem.squeeze(0))
                                geometry_loss = F.mse_loss(features, centers[gb_y])
                                loss = (
                                    self.alpha * reconstruction_loss
                                    + (1.0 - self.alpha) * geometry_loss
                                )
                                validation_loss += float(loss.detach().cpu())
                        average_loss = validation_loss / len(valid_loader)
                    else:
                        average_loss = training_loss / max(len(train_loader), 1)

                    self.scheduler.step()
                    if average_loss < epoch_best_val_loss:
                        epoch_best_val_loss = average_loss
                        patience_counter = 0
                        epoch_best_model = copy.deepcopy(self.model.state_dict())
                    else:
                        patience_counter += 1
                    if patience_counter >= self.early_stopping.patience:
                        self.model.load_state_dict(epoch_best_model)
                        break

                if epoch_best_val_loss < best_val_loss:
                    best_val_loss = epoch_best_val_loss
                    best_model_state = copy.deepcopy(epoch_best_model)

            if best_model_state is None:
                raise RuntimeError("GBOC training did not produce a finite validation state")
            self.model.load_state_dict(best_model_state)
            self.model.eval()

            # Rebuild the memory once in the final learned representation using
            # only the full normal-only prefix.  This is the frozen memory used
            # for every Eval point; no full-series/test ball is ever created.
            final_trainset = GB_ReconstructDataset(
                data,
                self.model,
                dis_mode=self.dis_mode,
                window_size=self.win_size,
                normalize=False,
            )
            self.train_centers_ = np.asarray(final_trainset.gb_centers, dtype=np.float32)
            if not len(self.train_centers_):
                raise RuntimeError("final normal-only GBOC memory is empty")
            self.best_validation_loss_ = float(best_val_loss)
            return self

        def decision_function(self, data):
            if not hasattr(self, "train_centers_"):
                raise RuntimeError("fit must be called before decision_function")
            data = np.asarray(data, dtype=np.float32)
            testset = GB_ReconstructDataset(
                data,
                self.model,
                dis_mode=self.dis_mode,
                gb_centers=self.train_centers_,
                window_size=self.win_size,
                normalize=False,
            )
            test_loader = DataLoader(
                testset, batch_size=self.batch_size, shuffle=False
            )
            scores = []
            self.model.eval()
            with torch.no_grad():
                for x, _target, _gb_y in test_loader:
                    x = torch.nan_to_num(x).to(self.device).permute(1, 0, 2)
                    features = self.model.obtain_features(x).cpu().numpy()
                    scores.append(calculate_anomaly_scores(features, self.train_centers_))
            if not scores:
                raise RuntimeError("GBOC produced no window scores")
            anomaly_score = np.concatenate(scores).astype(float, copy=False)
            if len(anomaly_score) < len(data):
                left = math.ceil((self.win_size - 1) / 2)
                right = (self.win_size - 1) // 2
                anomaly_score = np.asarray(
                    [anomaly_score[0]] * left
                    + anomaly_score.tolist()
                    + [anomaly_score[-1]] * right,
                    dtype=float,
                )
            if len(anomaly_score) != len(data):
                raise RuntimeError(
                    f"point score length {len(anomaly_score)} != series length {len(data)}"
                )
            self._GBOC__anomaly_score = anomaly_score
            return anomaly_score

    ProtocolGBOC.__name__ = "ProtocolGBOC"
    return ProtocolGBOC
