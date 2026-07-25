from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from STAGE.stage import StageConfig, StageEncoder, fit_encoder  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(2026)
    values = rng.normal(size=(512, 3)).astype(np.float32)
    records = []
    for encoder_type in (
        "dilated_residual",
        "depthwise_tcn",
        "multiscale_depthwise_tcn",
    ):
        config = StageConfig(
            encoder_type=encoder_type,
            patch_size=48,
            channels=32,
            token_dim=16,
            embedding_dim=16,
            dilations=(1, 2, 4),
            group_norm_groups=8,
            dropout=0.0,
            batch_size=16,
            steps=4,
            overlap_deltas=(12, 24),
            overlap_trim=4,
            gb_activation_fraction=0.5,
            gb_min_split=4,
            gb_max_rounds=8,
            max_gb_rows=256,
            seed=2026,
        )
        config.validate()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = StageEncoder(values.shape[1], config).to(device)
        diagnostics = fit_encoder(values, model, device, config)
        model.eval()
        with torch.inference_mode():
            tokens, embedding = model(
                torch.from_numpy(values[:48].T[None]).to(device)
            )
        if not torch.isfinite(tokens).all() or not torch.isfinite(embedding).all():
            raise RuntimeError(f"{encoder_type} produced a non-finite output")
        records.append(
            {
                "encoder_type": encoder_type,
                "device": str(device),
                "steps": int(diagnostics["total_steps"]),
                "peak_grad_norm": float(diagnostics["peak_grad_norm"]),
                "token_shape": list(tokens.shape),
                "embedding_shape": list(embedding.shape),
            }
        )
    print(json.dumps({"records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
