#!/usr/bin/env bash
set -euo pipefail

APP="${STAGE_GEOMETRY_APP:-/root/autodl-tmp/STAGE-AD-majority-geometry-support}"
FORMAL="${STAGE_FORMAL_REPO:-/root/autodl-tmp/STAGE-AD}"
PYTHON="${STAGE_PYTHON:-/root/autodl-tmp/envs/stage-py310/bin/python}"
RESULT_ROOT="${STAGE_GEOMETRY_RESULT_ROOT:-/root/autodl-tmp/results/STAGE_majority_geometry_support/g2_seed2026}"
PROTOCOL="$APP/configs/stage_majority_geometry_g2.json"
PRECOMPUTED_PLAN="$APP/configs/stage_majority_geometry_g2_plan.json"
METRICS_ROOT="$FORMAL/external"
DATA_SOURCE="$FORMAL/data"

test -x "$PYTHON"
test -d "$DATA_SOURCE"
test -d "$METRICS_ROOT"
test -f "$PROTOCOL"
test -f "$PRECOMPUTED_PLAN"
git -C "$APP" diff --quiet
git -C "$APP" diff --cached --quiet

(
  cd "$APP"
  sha256sum -c checksums/stage-majority-geometry-g2.sha256
)

if [[ ! -e "$APP/data" ]]; then
  ln -s "$DATA_SOURCE" "$APP/data"
fi
if [[ "$(readlink -f "$APP/data")" != "$(readlink -f "$DATA_SOURCE")" ]]; then
  echo "APP/data does not resolve to the frozen formal data directory" >&2
  exit 1
fi
if [[ ! -e "$APP/external" ]]; then
  ln -s "$METRICS_ROOT" "$APP/external"
fi
if [[ "$(readlink -f "$APP/external")" != "$(readlink -f "$METRICS_ROOT")" ]]; then
  echo "APP/external does not resolve to the frozen official evaluator" >&2
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -lt 1 ]]; then
  echo "at least one GPU is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$RESULT_ROOT")"
available_kb="$(df -Pk "$(dirname "$RESULT_ROOT")" | awk 'NR==2 {print $4}')"
if [[ "$available_kb" -lt 15728640 ]]; then
  echo "less than 15 GiB is available below the result root" >&2
  exit 1
fi
mkdir -p "$RESULT_ROOT"
if [[ ! -e "$RESULT_ROOT/stage_vuspr_search_plan.json" ]]; then
  cp "$PRECOMPUTED_PLAN" "$RESULT_ROOT/stage_vuspr_search_plan.json"
fi

"$PYTHON" "$APP/scripts/stage_vuspr_search.py" \
  --repo "$APP" \
  --protocol "$PROTOCOL" \
  --result-root "$RESULT_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  plan

workers_per_gpu="${STAGE_WORKERS_PER_GPU:-6}"
if ! [[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "STAGE_WORKERS_PER_GPU must be a positive integer" >&2
  exit 1
fi
gpu_ids="$(seq -s, 0 $((gpu_count - 1)))"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "launching geometry G2 with ${workers_per_gpu} workers/GPU on ${gpu_count} GPU(s)"

exec "$PYTHON" "$APP/scripts/stage_vuspr_search.py" \
  --repo "$APP" \
  --protocol "$PROTOCOL" \
  --result-root "$RESULT_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  run \
  --python "$PYTHON" \
  --gpus "$gpu_ids" \
  --workers-per-gpu "$workers_per_gpu"
