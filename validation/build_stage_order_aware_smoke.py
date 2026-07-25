#!/usr/bin/env python3
"""Derive a disposable two-candidate CUDA smoke from the frozen protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol["schema_version"] = "stage-order-aware-smoke-v1"
    protocol["phase"] = "order_aware_strict_cuda_smoke"
    protocol["files"] = {
        "U": {
            "SED": ["235_SED_id_2_Medical_tr_2499_1st_3840.csv"]
        }
    }
    protocol["order_candidates"] = protocol["order_candidates"][:2]
    protocol["base_config_by_subset"] = {
        "U/SED": {
            **protocol["base_config_by_subset"]["U/SED"],
            "steps": 2,
            "batch_size": 8,
            "embedding_batch_size": 256,
            "score_batch_size": 256,
            "memory_score_block_size": 2048
        }
    }
    protocol["fixed_head_by_subset"] = {
        "U/SED": protocol["fixed_head_by_subset"]["U/SED"]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
