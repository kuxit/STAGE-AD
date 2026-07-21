#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_ROOT="${STAGE_SERVER_WORK_ROOT:-/root/autodl-tmp}"
CONDA="${STAGE_CONDA:-/root/miniconda3/bin/conda}"
MAIN_ENV="${STAGE_MAIN_ENV:-$WORK_ROOT/envs/stage-py310}"
MEMTO_ENV="${STAGE_MEMTO_ENV:-$WORK_ROOT/envs/stage-memto-py310}"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORK_ROOT/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORK_ROOT/pip_cache}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$(dirname "$MAIN_ENV")"

if [[ ! -x "$MAIN_ENV/bin/python" ]]; then
  "$CONDA" create -y -p "$MAIN_ENV" python=3.10 pip
fi
"$MAIN_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$MAIN_ENV/bin/python" -m pip install \
  filelock typing-extensions sympy networkx jinja2 fsspec
"$MAIN_ENV/bin/python" -m pip install \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$MAIN_ENV/bin/python" -m pip install -r "$PROJECT/environment/requirements-formal.txt"
"$MAIN_ENV/bin/python" -m pip install reformer-pytorch==1.4.4
"$MAIN_ENV/bin/python" -m pip check

if [[ ! -x "$MEMTO_ENV/bin/python" ]]; then
  "$CONDA" create -y -p "$MEMTO_ENV" --clone "$MAIN_ENV"
fi
"$MEMTO_ENV/bin/python" -m pip install \
  torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$MEMTO_ENV/bin/python" -m pip check

"$MAIN_ENV/bin/python" - <<'PY'
import json
import platform
from pathlib import Path

import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
Path("/root/autodl-tmp/envs/stage-main-runtime.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
if not payload["cuda_available"] or payload["gpu_count"] != 2:
    raise SystemExit("main environment cannot see exactly two GPUs")
PY

"$MEMTO_ENV/bin/python" - <<'PY'
import json
import platform
from pathlib import Path

import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
Path("/root/autodl-tmp/envs/stage-memto-runtime.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
if not payload["cuda_available"] or payload["gpu_count"] != 2:
    raise SystemExit("MEMTO environment cannot see exactly two GPUs")
PY

"$MAIN_ENV/bin/python" -m pip freeze --all \
  > "$WORK_ROOT/envs/stage-main-pip-freeze.txt"
"$MEMTO_ENV/bin/python" -m pip freeze --all \
  > "$WORK_ROOT/envs/stage-memto-pip-freeze.txt"
touch "$WORK_ROOT/envs/STAGE_ENVIRONMENTS_READY"
