"""Convert CSV motion files to windowed relative-feature NPZ files.

Each output NPZ contains windows of shape (N, window_size, feature_dim) where
features are computed relative to the first frame's anchor body pose. Per-frame
layout (matching mjlab's tracking observations, but expressed in the window's
frame-0 anchor frame instead of the live robot anchor frame):

  motion_anchor_pos_b   (3)              anchor pos in frame-0 anchor frame
  motion_anchor_ori_b   (6)              first 2 cols of rotation matrix
  robot_body_pos_b      (num_bodies * 3) tracked body pos in frame-i anchor frame
  robot_body_ori_b      (num_bodies * 6) tracked body ori (6D) in frame-i anchor frame
  joint_pos             (num_joints)     raw joint positions

The CSV is replayed through a MuJoCo G1 sim to obtain forward-kinematics-derived
body poses (mjlab's csv_to_npz.MotionLoader handles loading + fps interpolation).

Usage:
  uv run python scripts/csv_to_npz.py --input-dir datasets/csv --output-dir datasets/npz
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.scripts.csv_to_npz import MotionLoader as CsvMotionLoader
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

from smp.utils import detect_device

# Joint name order must match the CSV column order. Copied from
# mjlab/scripts/csv_to_npz.py main() — these are the 29 G1 joints in CSV order.
JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


@dataclass
class Cfg:
  input_dir: str = "datasets/csv"
  """Directory of input CSV motion files."""
  output_dir: str = "datasets/npz"
  """Directory to write output NPZ window files."""
  window_size: int = 20
  """Number of frames per window."""
  stride: int = 1
  """Stride between consecutive windows."""
  input_fps: int = 30
  """CSV frame rate."""
  output_fps: int = 50
  """Output (and sim) frame rate after interpolation."""
  device: str = ""
  """Compute device. Empty = auto (cuda if available else cpu)."""
  shard_index: int = 0
  """Index of this shard (for parallel runs). Files are sliced as [shard_index::num_shards]."""
  num_shards: int = 1
  """Total number of shards (for parallel runs)."""


def _setup_sim(device: str) -> tuple[Simulation, Scene, tuple[str, ...], str]:
  """Build the G1 sim once and pull tracked-body config from the env_cfg."""
  anchor_body_name = "pelvis"
  body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  return sim, scene, body_names, anchor_body_name


@torch.no_grad()
def _fk_motion(
  csv_path: Path,
  sim: Simulation,
  scene: Scene,
  joint_indexes: torch.Tensor,
  body_indexes: torch.Tensor,
  input_fps: int,
  output_fps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Replay a CSV through the sim and return FK'd (body_pos, body_quat, joint_pos)."""
  motion = CsvMotionLoader(
    motion_file=str(csv_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
  )
  robot: Entity = scene["robot"]

  body_pos: list[torch.Tensor] = []
  body_quat: list[torch.Tensor] = []
  joint_pos: list[torch.Tensor] = []

  scene.reset()
  for _ in range(motion.output_frames):
    state, _ = motion.get_next_state()
    base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel = state

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos_full = robot.data.default_joint_pos.clone()
    joint_vel_full = robot.data.default_joint_vel.clone()
    joint_pos_full[:, joint_indexes] = dof_pos
    joint_vel_full[:, joint_indexes] = dof_vel
    robot.write_joint_state_to_sim(joint_pos_full, joint_vel_full)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    body_pos.append(robot.data.body_link_pos_w[0, body_indexes].clone())
    body_quat.append(robot.data.body_link_quat_w[0, body_indexes].clone())
    joint_pos.append(robot.data.joint_pos[0, joint_indexes].clone())

  return torch.stack(body_pos), torch.stack(body_quat), torch.stack(joint_pos)


def _compute_windows(
  body_pos: torch.Tensor,
  body_quat: torch.Tensor,
  joint_pos: torch.Tensor,
  anchor_local_idx: int,
  window_size: int,
  stride: int,
) -> torch.Tensor | None:
  """Slice into windows and compute features relative to each window's first frame.

  The motion is left-padded with (window_size - 1) copies of frame 0 before
  windowing, so each real frame i becomes the LAST frame of one window. Early
  windows therefore contain frame-0 padding (matching how a runtime policy sees
  a back-padded history at the start of an episode).

  Returns (num_windows, window_size, feature_dim) or None if the motion is empty.
  """
  T, num_bodies, _ = body_pos.shape
  if T < 1:
    return None

  pad_n = window_size - 1
  if pad_n > 0:
    body_pos = torch.cat([body_pos[0:1].repeat(pad_n, 1, 1), body_pos], dim=0)
    body_quat = torch.cat([body_quat[0:1].repeat(pad_n, 1, 1), body_quat], dim=0)
    joint_pos = torch.cat([joint_pos[0:1].repeat(pad_n, 1), joint_pos], dim=0)
  T_padded = body_pos.shape[0]

  # Build all windows at once via index_select: indices shape (N, W).
  starts = torch.arange(
    0, T_padded - window_size + 1, stride, device=body_pos.device, dtype=torch.long
  )
  offsets = torch.arange(window_size, device=body_pos.device, dtype=torch.long)
  win_idx = starts[:, None] + offsets[None, :]  # (N, W)
  N = win_idx.shape[0]
  W = window_size
  B = num_bodies

  flat_idx = win_idx.reshape(-1)  # (N*W,)
  win_body_pos = body_pos.index_select(0, flat_idx).reshape(N, W, B, 3)
  win_body_quat = body_quat.index_select(0, flat_idx).reshape(N, W, B, 4)
  win_joint = joint_pos.index_select(0, flat_idx).reshape(N, W, -1)

  # Per-frame anchor pose and per-window frame-0 anchor.
  anchor_pos_t = win_body_pos[:, :, anchor_local_idx, :]  # (N, W, 3)
  anchor_quat_t = win_body_quat[:, :, anchor_local_idx, :]  # (N, W, 4)
  anchor_pos_0 = anchor_pos_t[:, 0:1, :].expand(N, W, 3)  # (N, W, 3)
  anchor_quat_0 = anchor_quat_t[:, 0:1, :].expand(N, W, 4)  # (N, W, 4)

  # motion_anchor_*_b: anchor at frame t expressed in frame-0 anchor frame.
  m_pos, m_quat = subtract_frame_transforms(
    anchor_pos_0.reshape(-1, 3),
    anchor_quat_0.reshape(-1, 4),
    anchor_pos_t.reshape(-1, 3),
    anchor_quat_t.reshape(-1, 4),
  )
  m_pos = m_pos.reshape(N, W, 3)
  m_ori_6d = matrix_from_quat(m_quat)[..., :2].reshape(N, W, 6)

  # robot_body_*_b: bodies at frame t in frame-t's own anchor frame
  # (matches mjlab's live observation).
  parent_pos = anchor_pos_t[:, :, None, :].expand(N, W, B, 3).reshape(-1, 3)
  parent_quat = anchor_quat_t[:, :, None, :].expand(N, W, B, 4).reshape(-1, 4)
  body_pos_b, body_quat_b = subtract_frame_transforms(
    parent_pos,
    parent_quat,
    win_body_pos.reshape(-1, 3),
    win_body_quat.reshape(-1, 4),
  )
  body_pos_b = body_pos_b.reshape(N, W, B * 3)
  body_ori_b_6d = matrix_from_quat(body_quat_b)[..., :2].reshape(N, W, B * 6)

  return torch.cat([m_pos, m_ori_6d, body_pos_b, body_ori_b_6d, win_joint], dim=-1)


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

  sim, scene, body_names, anchor_body_name = _setup_sim(cfg.device)
  robot: Entity = scene["robot"]
  joint_indexes = torch.tensor(
    robot.find_joints(list(JOINT_NAMES), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  body_indexes = torch.tensor(
    robot.find_bodies(list(body_names), preserve_order=True)[0],
    dtype=torch.long,
    device=sim.device,
  )
  anchor_local_idx = body_names.index(anchor_body_name)

  num_bodies = len(body_names)
  num_joints = len(JOINT_NAMES)
  feature_dims = [3, 6, num_bodies * 3, num_bodies * 6, num_joints]
  # feature_names = [
  #   "motion_anchor_pos_b",
  #   "motion_anchor_ori_b",
  #   "robot_body_pos_b",
  #   "robot_body_ori_b",
  #   "joint_pos",
  # ]
  total_feature_dim = sum(feature_dims)

  print(f"Files: {len(csv_files)} in {in_dir}")
  print(f"Output: {out_dir}")
  print(f"Window: size={cfg.window_size} stride={cfg.stride} fps={cfg.output_fps}")
  print(f"Bodies: {num_bodies} (anchor={anchor_body_name}) | Joints: {num_joints}")
  print(f"Feature dim: {total_feature_dim}")

  for i, csv_path in enumerate(csv_files):
    print(f"\n[{i + 1}/{len(csv_files)}] {csv_path.name}")
    body_pos, body_quat, joint_pos = _fk_motion(
      csv_path,
      sim,
      scene,
      joint_indexes,
      body_indexes,
      input_fps=cfg.input_fps,
      output_fps=cfg.output_fps,
    )
    windows = _compute_windows(
      body_pos,
      body_quat,
      joint_pos,
      anchor_local_idx,
      cfg.window_size,
      cfg.stride,
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
      anchor_body_name=np.array(anchor_body_name),
      body_names=np.array(body_names),
      feature_dims=np.array(feature_dims, dtype=np.int32),
    )
    print(f"  saved {out_path.name}: windows={tuple(windows.shape)}")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
