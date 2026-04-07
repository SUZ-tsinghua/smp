"""Transformer denoiser and DDPM scheduler for SMP."""

from __future__ import annotations

import torch
import torch.nn as nn

from smp.models.time_embed import SinusoidalTimeEmbedding


class DiffusionDenoiser(nn.Module):
  """2-layer transformer encoder, epsilon-prediction parameterization.

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


class DDPMScheduler(nn.Module):
  """Minimal DDPM noise scheduler. Linear beta schedule, epsilon parameterization.

  Subclassing nn.Module so buffers move with .to(device) automatically.
  Only the two sqrt(alpha_bar) buffers are kept; the raw betas/alphas are not
  used by add_noise and would only be needed for a sampler (not implemented).
  """

  sqrt_alphas_cumprod: torch.Tensor
  sqrt_one_minus_alphas_cumprod: torch.Tensor

  def __init__(
    self,
    num_timesteps: int = 50,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
  ) -> None:
    super().__init__()
    self.num_timesteps = num_timesteps
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
    self.register_buffer(
      "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
    )

  def add_noise(
    self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
  ) -> torch.Tensor:
    """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
    shape = (-1, *([1] * (x_0.ndim - 1)))
    return (
      self.sqrt_alphas_cumprod[t].view(shape) * x_0
      + self.sqrt_one_minus_alphas_cumprod[t].view(shape) * noise
    )

  def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(
      0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
    )
