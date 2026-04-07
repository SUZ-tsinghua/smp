"""Diffusion pretraining configuration."""

from __future__ import annotations

from dataclasses import dataclass

from smp.utils import detect_device


@dataclass
class PretrainCfg:
  """Configuration for diffusion model pretraining."""

  # Data
  data_dir: str = "datasets/npz"
  normalize: bool = True
  train_split: float = 0.9

  # Model
  d_model: int = 256
  nhead: int = 8
  num_layers: int = 2
  dim_feedforward: int = 1024
  dropout: float = 0.0

  # Diffusion
  num_timesteps: int = 50
  beta_start: float = 1e-4
  beta_end: float = 0.02

  # Training
  batch_size: int = 1024
  num_epochs: int = 500
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0

  # Logging
  log_interval: int = 10
  save_interval: int = 50
  log_dir: str = "logs/pretrain"
  wandb_project: str = "smp"
  wandb_run_name: str = "pretrain"
  use_wandb: bool = True

  # Device
  device: str = ""

  def __post_init__(self) -> None:
    if not self.device:
      self.device = detect_device()
