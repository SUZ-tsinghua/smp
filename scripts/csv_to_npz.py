"""Convert CSV motion files to windowed base+joint NPZ files.

Each output NPZ contains a ``windows`` array of shape
``(N, window_size, 3+6+num_joints)`` with the per-frame layout:

  motion_anchor_pos_b   (3)            base pos in window-frame-0 yaw-aligned frame
  motion_anchor_ori_b   (6)            first 2 cols of the rotation matrix
  joint_pos             (num_joints)   CSV-order joint angles

No forward kinematics / body-pose info is stored — the base (pelvis) pose and
joint angles come straight from the interpolated motion clip, so we don't need
to spin up a MuJoCo sim at all.

Usage:
  uv run scripts/csv_to_npz.py --input-dir datasets/csv --output-dir datasets/npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
  yaw_quat,
)

from smp.utils import detect_device

NUM_JOINTS = 29  # G1 has 29 1-DoF revolute joints (CSV column order).


@dataclass
class Cfg:
  input_dir: str = "datasets/csv"
  """Directory of input CSV motion files."""
  output_dir: str = "datasets/npz"
  """Directory to write output NPZ window files."""
  window_size: int = 10
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  input_fps: int = 30
  """CSV frame rate."""
  output_fps: int = 50
  """Output frame rate after interpolation."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  shard_index: int = 0
  """Index of this shard (for parallel runs). Files are sliced as [shard_index::num_shards]."""
  num_shards: int = 1
  """Total number of shards (for parallel runs)."""


@torch.no_grad()
def _load_motion(
  csv_path: Path, input_fps: int, output_fps: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Load + interpolate a CSV motion; return (base_pos, base_quat_wxyz, dof_pos)."""
  motion = CsvMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
  )
  return motion.motion_base_poss, motion.motion_base_rots, motion.motion_dof_poss


def _compute_windows(
  base_pos: torch.Tensor,
  base_quat: torch.Tensor,
  joint_pos: torch.Tensor,
  window_size: int,
  stride: int,
) -> torch.Tensor | None:
  """Slice into windows and compute features relative to each window's first frame.

  Motion is left-padded with (window_size - 1) copies of frame 0 so every real
  frame i becomes the LAST frame of one window — early windows contain
  frame-0 padding, matching how a runtime policy sees a back-padded history at
  the start of an episode.

  Returns ``(num_windows, window_size, 3+6+J)`` or ``None`` on empty input.
  """
  T = base_pos.shape[0]
  if T < 1:
    return None

  pad_n = window_size - 1
  if pad_n > 0:
    base_pos = torch.cat([base_pos[0:1].repeat(pad_n, 1), base_pos], dim=0)
    base_quat = torch.cat([base_quat[0:1].repeat(pad_n, 1), base_quat], dim=0)
    joint_pos = torch.cat([joint_pos[0:1].repeat(pad_n, 1), joint_pos], dim=0)
  T_padded = base_pos.shape[0]

  # Build all window indices at once.
  starts = torch.arange(
    0, T_padded - window_size + 1, stride, device=base_pos.device, dtype=torch.long
  )
  offsets = torch.arange(window_size, device=base_pos.device, dtype=torch.long)
  win_idx = starts[:, None] + offsets[None, :]  # (N, W)
  N, W = win_idx.shape[0], window_size

  flat_idx = win_idx.reshape(-1)
  win_base_pos = base_pos.index_select(0, flat_idx).reshape(N, W, 3)
  win_base_quat = base_quat.index_select(0, flat_idx).reshape(N, W, 4)
  win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, -1)

  # Frame-0 anchor is YAW-ONLY: forward = +x in the horizontal plane. This
  # lets the roll/pitch of the pelvis at frame 0 leak into motion_anchor_ori_b,
  # so the world orientation is recoverable up to an unknown yaw.
  anchor_pos_0 = win_base_pos[:, 0:1, :].expand(N, W, 3)
  anchor_quat_0: torch.Tensor = yaw_quat(win_base_quat[:, 0])[:, None, :].expand(
    N, W, 4
  )

  m_pos, m_quat = subtract_frame_transforms(
    anchor_pos_0.reshape(-1, 3),
    anchor_quat_0.reshape(-1, 4),
    win_base_pos.reshape(-1, 3),
    win_base_quat.reshape(-1, 4),
  )
  m_pos = m_pos.reshape(N, W, 3)
  m_ori_6d = matrix_from_quat(m_quat)[..., :2].reshape(N, W, 6)

  return torch.cat([m_pos, m_ori_6d, win_joint], dim=-1)


def main(cfg: Cfg) -> None:
  if not cfg.device:
    cfg.device = detect_device()
  print(f"Device: {cfg.device}")

  in_dir = Path(cfg.input_dir)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  csv_files = sorted(in_dir.glob("*.csv"))
  if not csv_files:
    msg = f"No CSV files found in {in_dir}"
    raise FileNotFoundError(msg)
  if cfg.num_shards > 1:
    csv_files = csv_files[cfg.shard_index :: cfg.num_shards]
    print(f"Shard {cfg.shard_index}/{cfg.num_shards}: {len(csv_files)} files")

  feature_dims = [3, 6, NUM_JOINTS]
  total_feature_dim = sum(feature_dims)

  print(f"Files: {len(csv_files)} in {in_dir}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.output_fps}")
  print(f"Feature dim: {total_feature_dim} (= 3 + 6 + {NUM_JOINTS})")

  for i, csv_path in enumerate(csv_files):
    print(f"\n[{i + 1}/{len(csv_files)}] {csv_path.name}")
    base_pos, base_quat, joint_pos = _load_motion(
      csv_path, cfg.input_fps, cfg.output_fps, cfg.device
    )
    if joint_pos.shape[-1] != NUM_JOINTS:
      msg = (
        f"{csv_path.name}: expected {NUM_JOINTS} dof columns, got {joint_pos.shape[-1]}"
      )
      raise ValueError(msg)
    windows = _compute_windows(
      base_pos, base_quat, joint_pos, cfg.window_size, cfg.stride
    )
    if windows is None:
      print(f"  [SKIP] too short for window_size={cfg.window_size}")
      continue

    out_path = out_dir / f"{csv_path.stem}.npz"
    np.savez_compressed(
      out_path,
      windows=windows.cpu().numpy().astype(np.float32),
      fps=np.array([cfg.output_fps], dtype=np.float32),
      window_size=np.array([cfg.window_size], dtype=np.int32),
      stride=np.array([cfg.stride], dtype=np.int32),
      feature_dims=np.array(feature_dims, dtype=np.int32),
    )
    print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
