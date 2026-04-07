"""Motion window dataset for diffusion pretraining."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionWindowDataset(Dataset[torch.Tensor]):
  """Loads pre-windowed NPZs produced by scripts/csv_to_npz.py.

  Each NPZ contains a `windows` array of shape (N, window_size, feature_dim).
  Per-frame layout: motion_anchor_pos_b (3) + motion_anchor_ori_b (6) +
  robot_body_pos_b (B*3) + robot_body_ori_b (B*6) + joint_pos (J).
  """

  def __init__(
    self,
    data_dir: str | Path,
    normalize: bool = True,
  ) -> None:
    npz_files = sorted(Path(data_dir).glob("*.npz"))
    if not npz_files:
      msg = f"No NPZ files found in {data_dir}"
      raise FileNotFoundError(msg)

    chunks: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for npz_file in npz_files:
      with np.load(npz_file, allow_pickle=False) as npz:
        windows = npz["windows"].astype(np.float32, copy=False)
      if windows.ndim != 3:
        msg = (
          f"{npz_file.name}: 'windows' has shape {windows.shape}, expected (N, W, S)"
        )
        raise ValueError(msg)
      if expected_shape is None:
        expected_shape = (int(windows.shape[1]), int(windows.shape[2]))
      elif (windows.shape[1], windows.shape[2]) != expected_shape:
        msg = (
          f"{npz_file.name}: shape {windows.shape} mismatches "
          f"first file's (*, {expected_shape[0]}, {expected_shape[1]})"
        )
        raise ValueError(msg)
      chunks.append(windows)

    assert expected_shape is not None  # npz_files is non-empty above
    self.window_size, self.feature_dim = expected_shape

    data = np.concatenate(chunks, axis=0)

    if normalize:
      flat = data.reshape(-1, self.feature_dim)
      self.mean = flat.mean(axis=0).astype(np.float32)
      self.std = (flat.std(axis=0) + 1e-8).astype(np.float32)
      data -= self.mean
      data /= self.std
    else:
      self.mean = np.zeros(self.feature_dim, dtype=np.float32)
      self.std = np.ones(self.feature_dim, dtype=np.float32)

    self.windows = torch.from_numpy(data)

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    mean = torch.from_numpy(self.mean).to(x.device, x.dtype)
    std = torch.from_numpy(self.std).to(x.device, x.dtype)
    return x * std + mean

  def __len__(self) -> int:
    return self.windows.shape[0]

  def __getitem__(self, idx: int) -> torch.Tensor:
    return self.windows[idx]
