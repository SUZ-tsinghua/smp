"""Shared utilities."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def detect_device() -> str:
  """Auto-detect best available device: cuda > mps > cpu."""
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


def count_parameters(module: nn.Module) -> int:
  """Total number of parameters in a module."""
  return sum(p.numel() for p in module.parameters())


def seed_everything(seed: int, deterministic: bool = False) -> None:
  """Seed Python, NumPy and PyTorch RNGs for reproducibility.

  Args:
    seed: integer RNG seed.
    deterministic: if True, force cuDNN into deterministic mode (slower).
  """
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
