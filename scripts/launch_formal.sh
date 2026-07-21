#!/usr/bin/env bash
set -euo pipefail

EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${STAGE_WORKSPACE_ROOT:-$(dirname "$EXP")}"
OUT="${STAGE_RESULT_ROOT:-$ROOT/results/STAGE/seed2026}"
PY="${STAGE_PYTHON:-python}"

exec "$PY" "$EXP/controller.py" \
  --repo "$EXP" \
  --experiment "$EXP" \
  --result "$OUT" \
  --python "$PY" \
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
  --cpu-workers 8 \
  "$@"
