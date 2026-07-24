#!/usr/bin/env bash
set -euo pipefail

APP="${STAGE_VUSPR_APP:-/root/autodl-tmp/STAGE-AD-vuspr-tuning}"
FORMAL="${STAGE_FORMAL_REPO:-/root/autodl-tmp/STAGE-AD}"
PYTHON="${STAGE_PYTHON:-/root/autodl-tmp/envs/stage-py310/bin/python}"
RESULT_ROOT="${STAGE_VUSPR_RESULT_ROOT:-/root/autodl-tmp/results/STAGE_vuspr_tuning/stage1a_seed2026}"
PROTOCOL="$APP/configs/stage_vuspr_stage1a.json"
METRICS_ROOT="$FORMAL/external"
DATA_SOURCE="$FORMAL/data"

test -x "$PYTHON"
git -C "$APP" rev-parse --is-inside-work-tree >/dev/null
test -d "$DATA_SOURCE"
test -d "$METRICS_ROOT"
test -f "$PROTOCOL"

if [[ -n "$(git -C "$APP" status --porcelain)" ]]; then
  echo "refusing to run from a dirty STAGE VUS-PR worktree" >&2
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

"$PYTHON" "$APP/scripts/stage_vuspr_search.py" \
  --repo "$APP" \
  --protocol "$PROTOCOL" \
  --result-root "$RESULT_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  plan

exec "$PYTHON" "$APP/scripts/stage_vuspr_search.py" \
  --repo "$APP" \
  --protocol "$PROTOCOL" \
  --result-root "$RESULT_ROOT" \
  --metrics-root "$METRICS_ROOT" \
  run \
  --python "$PYTHON" \
  --gpus 0,1 \
  --workers-per-gpu 3
