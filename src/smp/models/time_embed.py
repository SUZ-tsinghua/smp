"""Sinusoidal timestep embedding for diffusion models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
  """Standard sinusoidal positional embedding for diffusion timesteps."""

  freqs: torch.Tensor

  def __init__(self, dim: int) -> None:
    super().__init__()
    if dim % 2 != 0:
      msg = f"SinusoidalTimeEmbedding requires even dim, got {dim}"
      raise ValueError(msg)
    self.dim = dim
    half = dim // 2
    freqs = torch.exp(
      -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )
    self.register_buffer("freqs", freqs, persistent=False)

  def forward(self, t: torch.Tensor) -> torch.Tensor:
    """Embed integer timesteps. Shape: (B,) -> (B, dim)."""
    args = t.float()[:, None] * self.freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
