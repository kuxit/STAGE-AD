#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
ENV="$ROOT/duoba-env"
CONDA=/root/miniconda3/bin/conda

if [[ ! -x "$ENV/bin/python" ]]; then
  "$CONDA" create -p "$ENV" python=3.10 pip -y
fi

"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install \
  torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$ENV/bin/python" -m pip install \
  -r "$ROOT/AAAI/DuoBa-Baseline-Seed2027/multiserver/requirements-server-b.txt"
"$ENV/bin/python" -m pip check

"$ENV/bin/python" - <<'PY'
import json
import platform
from pathlib import Path

import numpy
import pandas
import scipy
import sklearn
import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "scipy": scipy.__version__,
    "sklearn": sklearn.__version__,
}
Path("/root/autodl-tmp/AAAI/server_b_environment.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
