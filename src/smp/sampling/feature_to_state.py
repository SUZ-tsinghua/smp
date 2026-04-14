"""Convert SMP normalized feature windows back to robot world-frame states."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import combine_frame_transforms, quat_from_matrix

NUM_JOINTS = 29


def slice_features(frame: torch.Tensor) -> dict[str, torch.Tensor]:
  """Slice a feature vector into named components.

  Layout:
    [0:3]         motion_anchor_pos_b   pelvis pos in frame-0 yaw frame
    [3:9]         motion_anchor_ori_b   6D rot (first 2 cols of rotation matrix)
    [9:12]        base_lin_vel_b        linear velocity in frame-0 yaw frame
    [12:15]       base_ang_vel_b        angular velocity in frame-0 yaw frame
    [15:15+J]     joint_pos             J G1 joints (typically 29)
    [15+J:15+2J]  joint_vel             joint velocities
  """
  expected = 3 + 6 + 3 + 3 + NUM_JOINTS + NUM_JOINTS
  if frame.shape[-1] != expected:
    msg = f"expected feature_dim={expected}; got {frame.shape[-1]}"
    raise ValueError(msg)
  J = NUM_JOINTS
  return {
    "anchor_pos_b": frame[..., 0:3],
    "anchor_ori_6d": frame[..., 3:9],
    "base_lin_vel_b": frame[..., 9:12],
    "base_ang_vel_b": frame[..., 12:15],
    "joint_pos": frame[..., 15 : 15 + J],
    "joint_vel": frame[..., 15 + J : 15 + 2 * J],
  }


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D rotation representation (first 2 columns of R) to a 3x3 matrix.

  Layout matches scripts/csv_to_npz.py, which writes
  ``matrix_from_quat(q)[..., :2].reshape(..., 6)`` — i.e. the (3, 2) sub-matrix
  flattened row-major, so the first two columns are interleaved as
  ``[r00, r01, r10, r11, r20, r21]``.
  """
  m = d6.reshape(*d6.shape[:-1], 3, 2)  # (..., 3, 2): two columns of R
  c1 = m[..., 0]
  c2 = m[..., 1]
  c1 = torch.nn.functional.normalize(c1, dim=-1)
  c2 = c2 - (c1 * c2).sum(dim=-1, keepdim=True) * c1
  c2 = torch.nn.functional.normalize(c2, dim=-1)
  c3 = torch.cross(c1, c2, dim=-1)
  return torch.stack([c1, c2, c3], dim=-1)


def rot6d_to_quat(d6: torch.Tensor) -> torch.Tensor:
  """Convert 6D rotation representation to (w, x, y, z) quaternion."""
  return quat_from_matrix(rot6d_to_matrix(d6))


def window_to_pelvis_trajectory(
  window: torch.Tensor,
  frame0_pelvis_pos_w: torch.Tensor,
  frame0_pelvis_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Reconstruct world-frame pelvis pose + joint pos for each frame in a window.

  Args:
    window: (W, F) DENORMALIZED feature window.
    frame0_pelvis_pos_w: (3,) world position of the pelvis at window frame 0.
    frame0_pelvis_quat_w: (4,) world (w,x,y,z) orientation of the pelvis at frame 0.

  Returns:
    pelvis_pos_w: (W, 3)
    pelvis_quat_w: (W, 4) wxyz
    joint_pos:    (W, 29)
  """
  parts = slice_features(window)
  anchor_pos_b = parts["anchor_pos_b"]  # (W, 3) in frame-0 frame
  anchor_quat_b = rot6d_to_quat(parts["anchor_ori_6d"])  # (W, 4)
  W = window.shape[0]

  parent_pos = frame0_pelvis_pos_w.to(window).expand(W, 3)
  parent_quat = frame0_pelvis_quat_w.to(window).expand(W, 4)

  pelvis_pos_w, pelvis_quat_w = combine_frame_transforms(
    parent_pos, parent_quat, anchor_pos_b, anchor_quat_b
  )
  return pelvis_pos_w, pelvis_quat_w, parts["joint_pos"]
