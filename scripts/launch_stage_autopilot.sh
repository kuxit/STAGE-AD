#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_ROOT="${STAGE_RESULT_ROOT:-/root/autodl-tmp/results/STAGE}"
PYTHON="${STAGE_PYTHON:-/root/autodl-tmp/envs/stage-py310/bin/python}"
METRICS_ROOT="${STAGE_METRICS_ROOT:-$REPO/external}"
RECENT_RESULT_ROOT="${STAGE_RECENT_RESULT_ROOT:-$RESULT_ROOT/recent_seed2026}"
RECENT_LOCKED_CONFIG="${STAGE_RECENT_LOCKED_CONFIG:-$REPO/configs/aaai_recent_locked_seed2026.json}"

exec "$PYTHON" "$REPO/scripts/stage_autopilot.py" \
  --repo "$REPO" \
  --result-root "$RESULT_ROOT" \
  autopilot \
  --python "$PYTHON" \
  --metrics-root "$METRICS_ROOT" \
  --gpus "${STAGE_GPUS:-0,1}" \
  --workers-per-gpu "${STAGE_WORKERS_PER_GPU:-2}" \
  --recent-result-root "$RECENT_RESULT_ROOT" \
  --recent-locked-config "$RECENT_LOCKED_CONFIG" \
  --recent-workers-per-gpu "${STAGE_RECENT_WORKERS_PER_GPU:-1}" \
  --export-config "$REPO/configs/stage_locked_seed2026.json" \
  --next-seeds "${STAGE_NEXT_SEEDS:-2027,2028}" \
  "$@"
