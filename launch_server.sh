#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/taoxie/AAAI
EXP="$ROOT/DuoBa-Baseline-Seed2027"
OUT="$ROOT/results/DUOBA_10SUBSET_BASELINES/seed2027_v2"
PY="$ROOT/.venv/bin/python"

exec "$PY" "$EXP/controller.py" \
  --repo "$ROOT" \
  --experiment "$EXP" \
  --result "$OUT" \
  --python "$PY" \
  --duoba-source "$ROOT/GROVE-AD-V3/grove_ad.py" \
  --paano-root "$ROOT/external/PaAno" \
  --paano-one "$ROOT/GROVE-AD-V3/paano_official_one.py" \
  --gboc-root "$ROOT/external/GBOC_official_55a2a31" \
  --gboc-one "$ROOT/spectral_tsad_v5/baselines/gboc/run_one.py" \
  --gboc-config "$ROOT/spectral_tsad_v5/baselines/gboc/configs/official_release.json" \
  --memto-root "$ROOT/external/MEMTO_official" \
  --memto-one "$EXP/run_one_memto.py" \
  --dcdetector-root "$ROOT/external/DCdetector" \
  --dcdetector-one "$EXP/run_one_dcdetector.py" \
  --seed 2027 \
  --gpus 0,1 \
  --cpu-workers 8 \
  "$@"
