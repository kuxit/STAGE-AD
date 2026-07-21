#!/usr/bin/env bash
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${STAGE_WORKSPACE_ROOT:-$(dirname "$EXP")}"
OUT="${STAGE_RESULT_ROOT:-$ROOT/results/STAGE/seed2026}"
PY="${STAGE_PYTHON:-python}"
MEMTO_PY="${STAGE_MEMTO_PYTHON:-$PY}"

exec "$PY" "$EXP/controller.py" \
  --repo "$EXP" \
  --experiment "$EXP" \
  --result "$OUT" \
  --python "$PY" \
  --memto-python "$MEMTO_PY" \
  --stage-source "$EXP/STAGE/stage.py" \
  --paano-root "$EXP/external/PaAno" \
  --paano-one "$EXP/compat/paano_official_one_linux.py" \
  --gboc-root "$EXP/external/GBOC_official_55a2a31" \
  --gboc-one "$EXP/spectral_tsad_v5/baselines/gboc/run_one.py" \
  --gboc-config "$EXP/spectral_tsad_v5/baselines/gboc/configs/official_release.json" \
  --memto-root "$EXP/external/MEMTO_official" \
  --memto-one "$EXP/run_one_memto.py" \
  --dcdetector-root "$EXP/external/DCdetector" \
  --dcdetector-one "$EXP/run_one_dcdetector.py" \
  --seed 2026 \
  --gpus 0,1 \
  --gpu-cpu-map "${STAGE_GPU_CPU_MAP:-}" \
  --cpu-workers "${STAGE_CPU_WORKERS:-32}" \
  --scheduler "${STAGE_SCHEDULER:-throughput}" \
  --gboc-workers-per-gpu "${STAGE_GBOC_WORKERS_PER_GPU:-4}" \
  --phase "${STAGE_PHASE:-baseline}" \
  "$@"
