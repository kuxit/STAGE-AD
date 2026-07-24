#!/usr/bin/env bash
set -euo pipefail

APP="${STAGE_VUSPR_APP:-/root/autodl-tmp/STAGE-AD-vuspr-stage2}"
FORMAL="${STAGE_FORMAL_REPO:-/root/autodl-tmp/STAGE-AD}"
PYTHON="${STAGE_PYTHON:-/root/autodl-tmp/envs/stage-py310/bin/python}"
RESULT_ROOT="${STAGE_VUSPR_RESULT_ROOT:-/root/autodl-tmp/results/STAGE_vuspr_tuning/stage2_heads_seeds2026_2028}"
PROTOCOL="$APP/configs/stage_vuspr_stage2.json"
PRECOMPUTED_PLAN="$APP/configs/stage_vuspr_stage2_plan.json"
METRICS_ROOT="$FORMAL/external"
DATA_SOURCE="$FORMAL/data"

test -x "$PYTHON"
git -C "$APP" rev-parse --is-inside-work-tree >/dev/null
test -d "$DATA_SOURCE"
test -d "$METRICS_ROOT"
test -f "$PROTOCOL"
test -f "$PRECOMPUTED_PLAN"

if [[ -n "$(git -C "$APP" status --porcelain)" ]]; then
  echo "refusing to run from a dirty STAGE VUS-PR Stage2 worktree" >&2
  exit 1
fi

(
  cd "$APP"
  sha256sum -c checksums/stage-vuspr-tuning.sha256
)

if [[ ! -e "$APP/data" ]]; then
  ln -s "$DATA_SOURCE" "$APP/data"
fi
if [[ "$(readlink -f "$APP/data")" != "$(readlink -f "$DATA_SOURCE")" ]]; then
  echo "APP/data does not resolve to the frozen formal data directory" >&2
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -lt 2 ]]; then
  echo "two GPUs are required; found $gpu_count" >&2
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

if [[ -n "${STAGE_WORKERS_PER_GPU:-}" ]]; then
  workers_per_gpu="$STAGE_WORKERS_PER_GPU"
else
  minimum_memory_mib="$(
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
      | awk 'NR == 1 {minimum=$1} $1 < minimum {minimum=$1} END {print int(minimum)}'
  )"
  if [[ "$minimum_memory_mib" -ge 40000 ]]; then
    workers_per_gpu=8
  elif [[ "$minimum_memory_mib" -ge 22000 ]]; then
    workers_per_gpu=6
  elif [[ "$minimum_memory_mib" -ge 14000 ]]; then
    workers_per_gpu=4
  else
    workers_per_gpu=3
  fi
fi
if ! [[ "$workers_per_gpu" =~ ^[1-9][0-9]*$ ]]; then
  echo "STAGE_WORKERS_PER_GPU must be a positive integer" >&2
  exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "launching STAGE Stage2 with ${workers_per_gpu} workers/GPU on ${gpu_count} GPUs"

exec "$PYTHON" "$APP/scripts/stage_vuspr_search.py" \
  --repo "$APP" \
  --protocol "$PROTOCOL" \
  --result-root "$RESULT_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  run \
  --python "$PYTHON" \
  --gpus 0,1 \
  --workers-per-gpu "$workers_per_gpu"
