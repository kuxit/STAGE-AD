from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from STAGE.stage import StageConfig, StageEncoder  # noqa: E402


PLAN = ROOT / "configs" / "stage_majority_encoder_e1_plan.json"
OUTPUT = ROOT / "docs" / "majority_encoder_cost_audit.json"


def count_macs(model: nn.Module, values: torch.Tensor) -> int:
    macs = 0
    hooks = []

    def conv_hook(module: nn.Conv1d, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal macs
        batch, output_channels, output_length = output.shape
        kernel = int(module.kernel_size[0])
        per_output = (int(module.in_channels) // int(module.groups)) * kernel
        macs += int(batch * output_channels * output_length * per_output)

    def linear_hook(module: nn.Linear, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal macs
        output_elements = int(output.numel())
        macs += output_elements * int(module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    with torch.inference_mode():
        model(values)
    for hook in hooks:
        hook.remove()
    return macs


def main() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    metadata = plan["series_metadata"]
    channels_by_subset: dict[str, int] = {}
    for key, item in metadata.items():
        track, dataset, _file = key.split("/", 2)
        subset = f"{track}/{dataset}"
        channels = int(item["channels"])
        previous = channels_by_subset.setdefault(subset, channels)
        if previous != channels:
            raise RuntimeError(f"{subset} has inconsistent channel counts")

    records = []
    for subset, candidate_ids in plan["candidates_by_subset"].items():
        candidate_lookup = {
            str(item["id"]): item for item in plan["training_candidates"]
        }
        channels = channels_by_subset[subset]
        for candidate_id in candidate_ids:
            candidate = candidate_lookup[candidate_id]
            parameters = dict(candidate["resolved_parameters"])
            config_fields = {
                name: parameters[name]
                for name in StageConfig.__dataclass_fields__
                if name in parameters and parameters[name] is not None
            }
            config = StageConfig(**config_fields)
            model = StageEncoder(channels, config).eval()
            values = torch.zeros(1, channels, int(config.patch_size))
            records.append(
                {
                    "subset": subset,
                    "candidate_id": candidate_id,
                    "encoder_type": config.encoder_type,
                    "channels": channels,
                    "patch_size": int(config.patch_size),
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "trainable_parameters": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "forward_macs_per_patch": count_macs(model, values),
                    "token_head_retained": sum(
                        parameter.numel()
                        for parameter in model.token_head.parameters()
                    ),
                    "embedding_head_retained": sum(
                        parameter.numel()
                        for parameter in model.embedding_head.parameters()
                    ),
                }
            )
    payload = {
        "schema_version": "stage-majority-encoder-cost-audit-v1",
        "scope": "encoder plus retained token and embedding heads",
        "mac_definition": "Conv1d and Linear multiply-accumulate operations for one patch",
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
