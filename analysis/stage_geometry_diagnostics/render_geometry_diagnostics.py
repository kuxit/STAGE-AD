#!/usr/bin/env python3
"""Render compact, publication-ready STAGE geometry diagnostics.

The training runner writes only sampled two-dimensional coordinates and
downsampled raw-series evidence.  This script turns those compact JSON fields
into reproducible PNG/PDF figures without loading checkpoints, embeddings, or
score arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = {
    "train": "#8C8C8C",
    "normal": "#0072B2",
    "anomaly": "#D55E00",
    "prototype": "#E69F00",
    "accent": "#009E73",
    "grid": "#D9D9D9",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def iter_units(result_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    units_root = result_root / "units"
    if not units_root.is_dir():
        return
    for path in sorted(units_root.rglob("*.json")):
        item = load_json(path)
        visual = item.get("diagnostics", {}).get("visual_diagnostics")
        if isinstance(visual, Mapping):
            yield path, item


def _array(value: Any, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if columns is not None:
        if result.size == 0:
            return np.empty((0, columns), dtype=np.float64)
        result = result.reshape(-1, columns)
    return result


def _style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, color=PALETTE["grid"], linewidth=0.55, alpha=0.55)
    axis.set_axisbelow(True)


def _shade_anomalies(
    axis: plt.Axes,
    x: np.ndarray,
    labels: np.ndarray,
) -> None:
    if len(x) < 2:
        return
    active = labels > 0
    transitions = np.diff(np.concatenate([[False], active, [False]]).astype(int))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    for start, stop in zip(starts, stops):
        left = x[start]
        right = x[min(stop, len(x) - 1)]
        axis.axvspan(left, right, color=PALETTE["anomaly"], alpha=0.10, lw=0)


def _plot_raw_series(axis: plt.Axes, visual: Mapping[str, Any]) -> None:
    raw = visual["raw_series"]
    indices = _array(raw["indices"])
    values = _array(raw["z_values"])
    labels = _array(raw["labels"])
    if values.ndim != 2 or len(values) != len(indices):
        raise ValueError("raw-series diagnostic has inconsistent dimensions")
    x = indices / max(float(indices[-1]), 1.0)
    _shade_anomalies(axis, x, labels)
    colors = (PALETTE["normal"], PALETTE["accent"], PALETTE["prototype"])
    channels = list(raw["selected_channels"])
    for column in range(values.shape[1]):
        axis.plot(
            x,
            values[:, column],
            lw=0.75,
            alpha=0.80,
            color=colors[column % len(colors)],
            label=f"channel {channels[column]}",
        )
    axis.axvline(
        float(raw["train_index"]) / max(float(indices[-1]), 1.0),
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        label="train boundary",
    )
    axis.set_title("(a) Raw time series and anomaly intervals", loc="left")
    axis.set_xlabel("Normalized time")
    axis.set_ylabel("Robust z-value")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    _style_axis(axis)


def _scatter(
    axis: plt.Axes,
    xy: np.ndarray,
    *,
    color: str,
    label: str,
    size: float,
    alpha: float,
    marker: str = "o",
    edgecolor: str = "none",
    linewidth: float = 0.0,
) -> None:
    if len(xy):
        axis.scatter(
            xy[:, 0],
            xy[:, 1],
            s=size,
            c=color,
            alpha=alpha,
            marker=marker,
            edgecolors=edgecolor,
            linewidths=linewidth,
            label=label,
            rasterized=True,
        )


def _plot_embedding(axis: plt.Axes, visual: Mapping[str, Any]) -> None:
    pca = visual["pca"]
    train = _array(pca["train"]["xy"], 2)
    normal = _array(pca["normal_queries"]["xy"], 2)
    anomaly = _array(pca["anomaly_queries"]["xy"], 2)
    prototypes = _array(pca["prototypes"]["xy"], 2)
    sizes = _array(pca["prototypes"]["region_sizes"])
    _scatter(
        axis,
        train,
        color=PALETTE["train"],
        label="training normal",
        size=7,
        alpha=0.20,
    )
    _scatter(
        axis,
        normal,
        color=PALETTE["normal"],
        label="Eval normal",
        size=8,
        alpha=0.35,
    )
    _scatter(
        axis,
        anomaly,
        color=PALETTE["anomaly"],
        label="Eval anomaly",
        size=13,
        alpha=0.62,
        marker="^",
    )
    if len(prototypes):
        marker_sizes = 12.0 + 24.0 * np.sqrt(
            sizes / max(float(np.max(sizes)), 1.0)
        )
        axis.scatter(
            prototypes[:, 0],
            prototypes[:, 1],
            s=marker_sizes,
            c=PALETTE["prototype"],
            alpha=0.78,
            edgecolors="#111111",
            linewidths=0.35,
            label="selected prototypes",
            rasterized=True,
        )
    explained = pca["explained_variance_ratio"]
    axis.set_title("(b) Embedding geometry and prototypes", loc="left")
    axis.set_xlabel(f"Training-PCA 1 ({100 * float(explained[0]):.1f}%)")
    axis.set_ylabel(f"Training-PCA 2 ({100 * float(explained[1]):.1f}%)")
    axis.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    _style_axis(axis)


def _plot_prototype_geometry(axis: plt.Axes, visual: Mapping[str, Any]) -> None:
    prototype = visual["pca"]["prototypes"]
    sizes = _array(prototype["region_sizes"])
    radii = _array(prototype["region_radii"])
    source = _array(prototype["source_starts"])
    if len(sizes):
        scatter = axis.scatter(
            sizes,
            radii,
            c=source,
            cmap="viridis",
            s=14,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        colorbar = axis.figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.03)
        colorbar.set_label("Prototype source time", fontsize=8)
        if np.max(sizes) / max(float(np.min(sizes)), 1.0) > 20:
            axis.set_xscale("log")
    axis.set_title("(c) Prototype support and radius", loc="left")
    axis.set_xlabel("Region support (training patches)")
    axis.set_ylabel("Region radius")
    _style_axis(axis)


def _plot_scores_and_examples(axis: plt.Axes, visual: Mapping[str, Any]) -> None:
    separation = visual["score_separation"]
    x_positions = (0.0, 1.0)
    for x, key, color in zip(
        x_positions,
        ("normal", "anomaly"),
        (PALETTE["normal"], PALETTE["anomaly"]),
    ):
        stats = separation[key]
        if stats["median"] is None:
            continue
        axis.vlines(
            x,
            float(stats["q10"]),
            float(stats["q90"]),
            color=color,
            linewidth=5.0,
            alpha=0.40,
        )
        axis.scatter(
            [x],
            [float(stats["median"])],
            s=45,
            color=color,
            edgecolors="#111111",
            linewidths=0.5,
            zorder=3,
        )
    axis.set_xticks(x_positions, ("normal score", "anomaly score"))
    axis.set_ylabel("Patch anomaly score")
    margin = separation.get("median_margin")
    margin_text = "n/a" if margin is None else f"{float(margin):+.3f}"
    axis.set_title(
        f"(d) Score separation and prototype shapes (median Δ={margin_text})",
        loc="left",
    )
    _style_axis(axis)

    inset = axis.inset_axes([0.43, 0.50, 0.54, 0.43])
    examples = list(visual.get("prototype_examples", []))
    if examples:
        max_support = max(int(item["region_size"]) for item in examples)
        for index, item in enumerate(examples):
            signal = _array(item["signal_rms_z"])
            if len(signal) == 0:
                continue
            x = np.linspace(0.0, 1.0, len(signal))
            color = plt.cm.viridis(
                int(item["region_size"]) / max(float(max_support), 1.0)
            )
            inset.plot(x, signal, color=color, lw=0.75, alpha=0.75)
        inset.set_title("Selected prototype raw RMS-z shapes", fontsize=7)
        inset.set_xlabel("Patch position", fontsize=6)
        inset.set_ylabel("RMS-z", fontsize=6)
        inset.tick_params(labelsize=6)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
    else:
        inset.axis("off")


def render_unit(item: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    visual = item["diagnostics"]["visual_diagnostics"]
    if visual.get("schema_version") != "stage-geometry-visual-diagnostics-v1":
        raise ValueError("unsupported visual-diagnostic schema")
    if (
        visual.get("posthoc_labels_only") is not True
        or visual.get("selection_uses_visuals") is not False
    ):
        raise ValueError("visual diagnostic violates the frozen label policy")

    track = str(item["track"])
    dataset = str(item["dataset"])
    candidate = str(item["canonical_training_candidate_id"])
    seed = int(item["seed"])
    stem = f"{track}_{dataset}_{candidate}_seed{seed}_{str(item['file']).replace('.csv', '')}"
    stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in stem)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    _plot_raw_series(axes[0, 0], visual)
    _plot_embedding(axes[0, 1], visual)
    _plot_prototype_geometry(axes[1, 0], visual)
    _plot_scores_and_examples(axes[1, 1], visual)
    figure.suptitle(
        f"STAGE geometry diagnostic — {track}/{dataset}, {candidate}, seed {seed}",
        fontsize=11.5,
        fontweight="bold",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    memories = item["diagnostics"]["final_memories"]
    selected_memory = next(
        (
            memory
            for memory in memories
            if int(memory["final_gb_min_split"])
            == int(item.get("diagnostic_final_gb_min_split", 4))
        ),
        memories[0],
    )
    return {
        "track": track,
        "dataset": dataset,
        "file": str(item["file"]),
        "candidate": candidate,
        "seed": seed,
        "normal_score_median": visual["score_separation"]["normal"]["median"],
        "anomaly_score_median": visual["score_separation"]["anomaly"]["median"],
        "score_median_margin": visual["score_separation"]["median_margin"],
        "prototype_rows": int(selected_memory["memory_rows"]),
        "training_patches": int(item["diagnostics"]["train_patches"]),
        "memory_ratio": (
            int(selected_memory["memory_rows"])
            / max(int(item["diagnostics"]["train_patches"]), 1)
        ),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def write_index(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fields = [
        "track",
        "dataset",
        "file",
        "candidate",
        "seed",
        "normal_score_median",
        "anomaly_score_median",
        "score_median_margin",
        "prototype_rows",
        "training_patches",
        "memory_ratio",
        "png",
        "pdf",
    ]
    with (output_dir / "geometry_diagnostics_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# STAGE geometry diagnostic figures",
        "",
        "Labels are loaded only after scoring. Figures are post-hoc evidence and "
        "are not used for Tuning selection.",
        "",
        "| Dataset | Candidate | Median score gap | Memory ratio | Figure |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        margin = row["score_median_margin"]
        margin_text = "n/a" if margin is None or not math.isfinite(float(margin)) else f"{float(margin):.3f}"
        lines.append(
            f"| {row['track']}/{row['dataset']} | {row['candidate']} | "
            f"{margin_text} | {float(row['memory_ratio']):.3f} | "
            f"[PNG]({Path(row['png']).name}) |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()

    rows = [
        render_unit(item, arguments.output_dir)
        for _, item in iter_units(arguments.result_root)
    ]
    if not rows:
        raise RuntimeError("no completed unit contains visual diagnostics")
    write_index(rows, arguments.output_dir)
    print(
        json.dumps(
            {
                "units_rendered": len(rows),
                "output_dir": str(arguments.output_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
