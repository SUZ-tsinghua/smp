"""DDPM noise scheduler with cosine beta schedule."""

from __future__ import annotations

import torch
import torch.nn as nn


def _cosine_betas(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
  """Cosine ᾱ schedule from Nichol & Dhariwal 2021.

  ᾱ_t = cos²((t/T + s) / (1 + s) · π/2),  normalized so ᾱ_0 = 1.
  Reaches ᾱ_T ≈ 0 even for small T (e.g. T=50), unlike the standard linear
  schedule which only works at T≈1000. Returns the per-step β values.
  """
  steps = num_timesteps + 1
  x = torch.linspace(0, num_timesteps, steps, dtype=torch.float32)
  alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
  betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
  return torch.clip(betas, 0.0, 0.999)


class DDPMScheduler(nn.Module):
  """Minimal DDPM noise scheduler with a cosine beta schedule.

  Subclassing nn.Module so buffers move with .to(device) automatically.
  """

  sqrt_alphas_cumprod: torch.Tensor
  sqrt_one_minus_alphas_cumprod: torch.Tensor
  betas: torch.Tensor
  alphas_cumprod: torch.Tensor
  alphas_cumprod_prev: torch.Tensor
  sqrt_recip_alphas_cumprod: torch.Tensor
  sqrt_recipm1_alphas_cumprod: torch.Tensor
  posterior_variance: torch.Tensor
  posterior_mean_coef1: torch.Tensor
  posterior_mean_coef2: torch.Tensor

  def __init__(
    self,
    num_timesteps: int = 50,
  ) -> None:
    super().__init__()
    self.num_timesteps = num_timesteps
    betas = _cosine_betas(num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
      [torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0
    )
    self.register_buffer("betas", betas)
    self.register_buffer("alphas_cumprod", alphas_cumprod)
    self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
    self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
    self.register_buffer(
      "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
    )
    self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
    self.register_buffer(
      "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
    )
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    self.register_buffer("posterior_variance", posterior_variance)
    self.register_buffer(
      "posterior_mean_coef1",
      betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
    )
    self.register_buffer(
      "posterior_mean_coef2",
      (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
    )

  def step(self, eps: torch.Tensor, x_t: torch.Tensor, t: int) -> torch.Tensor:
    """One ancestral DDPM denoising step: x_t -> x_{t-1}.

    Predicts x_0 from eps, computes the posterior mean, and adds noise (except
    when t == 0). Operates on a single integer timestep broadcast across batch.
    """
    x_0_hat = (
      self.sqrt_recip_alphas_cumprod[t] * x_t
      - self.sqrt_recipm1_alphas_cumprod[t] * eps
    )
    mean = self.posterior_mean_coef1[t] * x_0_hat + self.posterior_mean_coef2[t] * x_t
    if t == 0:
      return mean
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(self.posterior_variance[t]) * noise

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
