"""Reward functions for SMP RL tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from smp.rl.utils import DiffNormalizer

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def smp_guidance_reward(
  env: ManagerBasedRlEnv,
  fixed_timesteps: tuple[int, ...] = (8, 15, 22),
  ws: float = 4.0,
) -> torch.Tensor:
  """Score-distillation reward with adaptive per-timestep normalization.

  Evaluates at a fixed set of timesteps and normalizes each timestep's MSE
  by a ``DiffNormalizer`` (count-based running mean) so all timesteps
  contribute equally regardless of their raw loss scale.

  The denoiser bundle, feature buffer, and normalizer are owned by the env
  (created by ``smp.rl.events.init_smp_state`` at startup).
  """
  device = torch.device(env.device)
  model, scheduler, q_low, q_high, _, _ = env._smp_bundle  # type: ignore[attr-defined]
  # Guard against the RL runner flipping modules into train mode (dropout=0.1).
  model.eval()
  buffer = env._smp_buffer  # type: ignore[attr-defined]
  normalizer: DiffNormalizer = env._smp_normalizer  # type: ignore[attr-defined]

  robot = env.scene["robot"]
  buffer.update(
    robot.data.root_link_pos_w,
    robot.data.root_link_quat_w,
    robot.data.root_link_lin_vel_w,
    robot.data.root_link_ang_vel_w,
    robot.data.joint_pos,
    robot.data.joint_vel,
  )

  features = buffer.compute_features()
  x_0 = 2.0 * (features - q_low) / (q_high - q_low + 1e-8) - 1.0
  num_envs = x_0.shape[0]

  total_err = torch.zeros(num_envs, device=device)
  with torch.no_grad():
    for t_scalar in fixed_timesteps:
      if not 0 <= t_scalar < scheduler.num_timesteps:
        msg = f"fixed_timestep {t_scalar} out of range [0, {scheduler.num_timesteps})"
        raise ValueError(msg)
      t = torch.full((num_envs,), t_scalar, dtype=torch.long, device=device)
      noise = torch.randn_like(x_0)
      x_t = scheduler.add_noise(x_0, noise, t)
      eps_hat = model(x_t, t)
      mse_per_env = ((eps_hat - noise) ** 2).mean(dim=(-1, -2))
      total_err += normalizer.update_and_normalize(t_scalar, mse_per_env)

  err = total_err / len(fixed_timesteps)
  return torch.exp(-err * ws)
