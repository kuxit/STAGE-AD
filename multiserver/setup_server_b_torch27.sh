#!/usr/bin/env bash
set -euo pipefail

SOURCE=/root/autodl-tmp/duoba-env
TARGET=/root/autodl-tmp/duoba-env27
CONDA=/root/miniconda3/bin/conda

if [[ ! -x "$TARGET/bin/python" ]]; then
  "$CONDA" create -p "$TARGET" --clone "$SOURCE" -y
fi

"$TARGET/bin/python" -m pip install --no-cache-dir \
  torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$TARGET/bin/python" -m pip check
"$TARGET/bin/python" - <<'PY'
import json
import platform
from pathlib import Path

import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
Path("/root/autodl-tmp/AAAI/server_b_environment_torch27.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
