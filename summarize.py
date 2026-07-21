from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import ALIASES, METRICS, atomic_json, complete_record


def markdown_tables(frame: pd.DataFrame, track: str) -> str:
    lines = [f"# TSB-AD-{track}: STAGE and locked baselines (seed 2026)", ""]
    for dataset in sorted(frame.loc[frame.track == track, "dataset"].unique()):
        group = frame[(frame.track == track) & (frame.dataset == dataset)]
        summary = group.groupby("method", as_index=False)[list(METRICS)].mean()
        rendered = pd.DataFrame({"Method": summary["method"]})
        for alias, metric in ALIASES.items():
            rendered[alias] = summary[metric].map(lambda value: f"{value:.3f}")
        headers = rendered.columns.tolist()
        lines.extend(
            [
                f"## {dataset}",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        for row in rendered.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_methods = set(protocol["non_deep_baselines"] + protocol["deep_baselines"] + [protocol["target_method"]])
    rows = []
    errors = []
    for path in args.result.glob("units/*/*/*/*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("method") not in expected_methods:
            continue
        if not complete_record(path):
            errors.append({"path": str(path), "error": item.get("error")})
            continue
        row = {name: item[name] for name in ("method", "track", "dataset", "file", "seed")}
        row.update({name: float(item["metrics"][name]) for name in METRICS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        atomic_json(args.result / "summary_status.json", {"status": "no_valid_records", "errors": errors})
        return 1
    frame = frame.sort_values(["track", "dataset", "method", "file"])
    frame.to_csv(args.result / "per_series_full_precision.csv", index=False)
    subset = frame.groupby(["track", "dataset", "method"], as_index=False)[list(METRICS)].mean()
    subset.to_csv(args.result / "by_subset_full_precision.csv", index=False)
    track = frame.groupby(["track", "method"], as_index=False)[list(METRICS)].mean()
    track.to_csv(args.result / "by_track_full_precision.csv", index=False)
    for name in ("U", "M"):
        chosen = subset[subset.track == name].copy()
        paper = chosen[["dataset", "method"]].copy()
        for alias, metric in ALIASES.items():
            paper[alias] = chosen[metric].map(lambda value: f"{value:.3f}")
        paper.to_csv(args.result / f"{name}_by_subset_3dec.csv", index=False)
        (args.result / f"{name}_tables_3dec.md").write_text(markdown_tables(frame, name), encoding="utf-8")
    atomic_json(
        args.result / "summary_status.json",
        {
            "status": "complete" if not errors else "incomplete",
            "valid_records": len(frame),
            "error_records": len(errors),
            "errors": errors,
            "metric_aliases": protocol["metrics"],
        },
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
