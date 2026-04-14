"""Startup / reset events that own the SMP feature buffer and denoiser bundle.

These run from mjlab's event manager so the task can be a plain
``ManagerBasedRlEnv`` and we can use mjlab's built-in train/play scripts.
"""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from smp.rl.utils import DiffNormalizer, PelvisAnchoredFeatureBuffer, load_denoiser
from smp.sampling.feature_to_state import rot6d_to_quat

NUM_JOINTS = 29


def init_smp_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None = None,
  ckpt_path: str = "",
) -> None:
  """Startup-mode event: allocate buffer, load frozen denoiser, prime buffer.

  The ``ckpt_path`` is passed through the ``EventTermCfg.params`` so the
  checkpoint path is a first-class config field rather than an env var.
  """
  del env_ids
  if not ckpt_path:
    msg = (
      "init_smp_state called without `ckpt_path`. Set it on the EventTermCfg: "
      "EventTermCfg(func=init_smp_state, mode='startup', "
      "params={'ckpt_path': '/path/to/pretrained.pt'})."
    )
    raise RuntimeError(msg)
  env._smp_bundle = load_denoiser(ckpt_path, env.device)  # type: ignore[attr-defined]
  window_size = env._smp_bundle[5]  # type: ignore[attr-defined]
  env._smp_buffer = PelvisAnchoredFeatureBuffer(  # type: ignore[attr-defined]
    num_envs=env.num_envs,
    window_size=window_size,
    num_joints=NUM_JOINTS,
    device=env.device,
  )
  num_timesteps = env._smp_bundle[1].num_timesteps  # type: ignore[attr-defined]
  env._smp_normalizer = DiffNormalizer(num_timesteps, env.device)  # type: ignore[attr-defined]

  gsi_reset(env)


@torch.no_grad()
def gsi_reset(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
  """Generative State Initialization.

  Sample a full window from the SMP denoiser, write the LAST frame's complete
  state (pos, rot, lin_vel, ang_vel, joint_pos, joint_vel) to sim, and fill the
  feature buffer with the entire trajectory.  Must run AFTER ``reset_base``.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = int(env_ids.numel())
  if n == 0:
    return

  model, scheduler, q_low, q_high, feature_dim, window_size = env._smp_bundle  # type: ignore[attr-defined]
  J = NUM_JOINTS
  W = window_size

  # Sample a full window via DDPM ancestral sampling.
  x_t = torch.randn(n, W, feature_dim, device=env.device)
  for t in reversed(range(scheduler.num_timesteps)):
    t_batch = torch.full((n,), t, dtype=torch.long, device=env.device)
    eps = model(x_t, t_batch)
    x_t = scheduler.step(eps, x_t, t)

  # Denormalize [-1, 1] → raw feature space.
  window = (x_t + 1.0) / 2.0 * (q_high - q_low) + q_low

  base_pos_b = window[..., 0:3]
  base_ori_6d = window[..., 3:9]
  base_lin_vel_b = window[..., 9:12]
  base_ang_vel_b = window[..., 12:15]
  joint_pos = window[..., 15 : 15 + J]
  joint_vel = window[..., 15 + J : 15 + 2 * J]

  base_quat_b = rot6d_to_quat(base_ori_6d.reshape(-1, 6)).reshape(n, W, 4)

  # Write the last frame's full root + joint state to sim.
  robot = env.scene["robot"]
  last_root_state = torch.cat(
    [
      base_pos_b[:, -1],
      base_quat_b[:, -1],
      base_lin_vel_b[:, -1],
      base_ang_vel_b[:, -1],
    ],
    dim=-1,
  )
  robot.write_root_state_to_sim(last_root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(joint_pos[:, -1], joint_vel[:, -1], env_ids=env_ids)

  # Fill the feature buffer with the full sampled trajectory.
  buf: PelvisAnchoredFeatureBuffer = env._smp_buffer  # type: ignore[attr-defined]
  buf.reset(
    env_ids,
    base_pos_b,
    base_quat_b,
    base_lin_vel_b,
    base_ang_vel_b,
    joint_pos,
    joint_vel,
  )
