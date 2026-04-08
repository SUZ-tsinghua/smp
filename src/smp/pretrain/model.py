"""Transformer denoiser + sinusoidal time embedding for SMP."""

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


class DiffusionDenoiser(nn.Module):
  """Transformer encoder denoiser, epsilon-prediction parameterization.

  Input:  (B, window_size, feature_dim) noised motion window + (B,) timesteps
  Output: (B, window_size, feature_dim) predicted noise
  """

  def __init__(
    self,
    feature_dim: int,
    window_size: int,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 1024,
    dropout: float = 0.0,
  ) -> None:
    super().__init__()
    self.feature_dim = feature_dim
    self.window_size = window_size
    self.d_model = d_model

    self.input_proj = nn.Linear(feature_dim, d_model)
    self.pos_embed = nn.Parameter(torch.zeros(1, window_size, d_model))
    nn.init.trunc_normal_(self.pos_embed, std=0.02)

    self.time_embed = nn.Sequential(
      SinusoidalTimeEmbedding(d_model),
      nn.Linear(d_model, d_model),
      nn.SiLU(),
      nn.Linear(d_model, d_model),
    )

    self.layers = nn.ModuleList(
      [
        nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          activation="gelu",
          batch_first=True,
          norm_first=True,
        )
        for _ in range(num_layers)
      ]
    )

    self.output_norm = nn.LayerNorm(d_model)
    self.output_proj = nn.Linear(d_model, feature_dim)

  def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    h = self.input_proj(x_t) + self.pos_embed
    t_emb = self.time_embed(t).unsqueeze(1)  # (B, 1, d_model) — broadcast over tokens
    for layer in self.layers:
      h = layer(h + t_emb)
    h = self.output_norm(h)
    return self.output_proj(h)
