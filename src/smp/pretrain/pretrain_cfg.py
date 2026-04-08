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
  num_layers: int = 4
  dim_feedforward: int = 1024
  dropout: float = 0.1

  # Diffusion
  num_timesteps: int = 50

  # EMA
  use_ema: bool = True
  ema_decay: float = 0.9999

  # Training
  batch_size: int = 1024
  num_epochs: int = 2000
  lr: float = 3e-4
  weight_decay: float = 1e-4
  max_grad_norm: float = 1.0

  # Logging
  log_interval: int = 10
  save_interval: int = 100
  log_dir: str = "logs/pretrain"
  wandb_project: str = "smp"
  wandb_run_name: str = "pretrain"
  use_wandb: bool = True

  # Device
  device: str = ""

  # Reproducibility
  seed: int = 0

  def __post_init__(self) -> None:
    if not self.device:
      self.device = detect_device()
