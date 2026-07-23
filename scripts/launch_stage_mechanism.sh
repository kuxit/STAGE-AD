#!/usr/bin/env bash
set -euo pipefail

APP=/root/autodl-tmp/STAGE-AD-v2-mechanism
PYTHON=/root/autodl-tmp/envs/stage-py310/bin/python
PROTOCOL="$APP/configs/stage_mechanism_tuning_seed2026.json"
TEN_LOCK=/root/autodl-tmp/results/STAGE_v2_dataset_tuning_supplement/stage_v2_ten_dataset_tuning_lock.json
DATA_REPO=/root/autodl-tmp/STAGE-AD
METRICS_ROOT=/root/autodl-tmp/STAGE-AD/external
RESULT_ROOT=/root/autodl-tmp/results/STAGE_v2_mechanism_tuning/seed2026
RUNNER="$APP/scripts/stage_mechanism_validation.py"

mkdir -p "$RESULT_ROOT"

"$PYTHON" "$RUNNER" plan \
  --protocol "$PROTOCOL" \
  --ten-dataset-lock "$TEN_LOCK" \
  --data-repo "$DATA_REPO" \
  --metrics-root "$METRICS_ROOT" \
  --result-root "$RESULT_ROOT"

exec "$PYTHON" "$RUNNER" run \
  --protocol "$PROTOCOL" \
  --ten-dataset-lock "$TEN_LOCK" \
  --data-repo "$DATA_REPO" \
  --metrics-root "$METRICS_ROOT" \
  --result-root "$RESULT_ROOT" \
  --gpus 0 1 \
  --workers-per-gpu 6
