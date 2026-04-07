"""Shared utilities."""

from __future__ import annotations

import torch
import torch.nn as nn


def detect_device() -> str:
  """Auto-detect best available device: cuda > mps > cpu."""
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def count_parameters(module: nn.Module) -> int:
  """Total number of parameters in a module."""
  return sum(p.numel() for p in module.parameters())
