from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class DenseProjection(nn.Module):
    """Bias-free projection network used by the DPAD equation."""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], output_dim: int):
        super().__init__()
        widths = (input_dim, *tuple(hidden_dims), output_dim)
        layers: list[nn.Module] = []
        self.linear_layers = nn.ModuleList()
        for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(left, right, bias=False)
            self.linear_layers.append(linear)
            layers.append(linear)
            if index < len(widths) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)

    def anti_collapse_penalty(self) -> torch.Tensor:
        penalties = [torch.abs(torch.linalg.matrix_norm(layer.weight) - 1.0) for layer in self.linear_layers]
        return torch.stack(penalties).sum()


class NoiseEvaluationMLP(nn.Module):
    """Four-layer ReLU MLP from Figure 7a of the DNE paper."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden = 64 if input_dim <= 64 else 256
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ResidualMLPBlock(nn.Module):
    """Three affine/ReLU stages with a residual connection."""

    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.body(values))


class NoiseEvaluationResMLP(nn.Module):
    """Five-block ResMLP interpretation of Figure 7b in the DNE appendix."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden = 64 if input_dim <= 64 else 256
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.blocks = nn.Sequential(*(ResidualMLPBlock(hidden) for _ in range(5)))
        self.output_layer = nn.Linear(hidden, input_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.blocks(self.input_layer(values)))


def diverse_gaussian_noise(
    values: torch.Tensor,
    sigma_max: float,
    parts: int,
    ratio: float,
) -> torch.Tensor:
    """Algorithm-1-style feature-wise noise with shuffled level intervals."""

    if parts < 1:
        raise ValueError("parts must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    batch, dimension = values.shape
    shuffled_positions = torch.rand((batch, dimension), device=values.device).argsort(dim=1)
    ordered_groups = torch.div(
        torch.arange(dimension, device=values.device) * parts,
        max(dimension, 1),
        rounding_mode="floor",
    ).clamp_max(parts - 1)
    lower = torch.arange(parts, device=values.device, dtype=values.dtype) * (sigma_max / parts)
    levels = lower.unsqueeze(0) + torch.rand(
        (batch, parts), device=values.device, dtype=values.dtype
    ) * (sigma_max / parts)
    ordered_sigmas = levels[:, ordered_groups]
    sigmas = torch.zeros_like(values).scatter(1, shuffled_positions, ordered_sigmas)
    noise = torch.randn_like(values) * sigmas
    if ratio < 1.0:
        mask = torch.rand((batch, dimension), device=values.device) < ratio
        noise = noise * mask
    return noise
