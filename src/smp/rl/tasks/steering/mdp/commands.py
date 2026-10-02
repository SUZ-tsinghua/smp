"""Steering command — target xy direction + speed + face direction.

Each env carries a periodically-resampled world-frame target dir, speed, and
face dir; the exposed command is in the robot's local heading frame, so the
observation is yaw-invariant.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _dir_world_to_local(dir_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  """Rotate a (N, 2) world-frame xy direction into the heading-aligned frame."""
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = dir_w[..., 0], dir_w[..., 1]
  return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


class SteeringCommand(CommandTerm):
  """Periodic target dir + speed + face dir command (world frame internally)."""

  cfg: SteeringCommandCfg

  def __init__(self, cfg: SteeringCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # World-frame state.
    self.tar_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.face_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.tar_speed = torch.zeros(self.num_envs, device=self.device)
    self.tar_dir_w[..., 0] = 1.0
    self.face_dir_w[..., 0] = 1.0

    # Heading-frame command exposed to the policy: [tar_dir_x, tar_dir_y,
    # tar_speed, face_dir_x, face_dir_y].
    self.command_b = torch.zeros(self.num_envs, 5, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_face"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_speed: viser.GuiSliderHandle | None = None
    self._gui_tar_angle: viser.GuiSliderHandle | None = None
    self._gui_face_angle: viser.GuiSliderHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    tar_vel_w = self.tar_speed.unsqueeze(-1) * self.tar_dir_w
    vel_err = torch.norm(tar_vel_w - self.robot.data.root_link_lin_vel_w[:, :2], dim=-1)
    self.metrics["error_vel_xy"] += vel_err / max_step

    heading_w = self.robot.data.heading_w
    char_face_w = torch.stack([torch.cos(heading_w), torch.sin(heading_w)], dim=-1)
    face_dot = (self.face_dir_w * char_face_w).sum(dim=-1)
    self.metrics["error_face"] += (1.0 - face_dot.clamp(-1.0, 1.0)) / max_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    r = torch.empty(n, device=self.device)

    if self.cfg.rand_tar_dir:
      theta = r.uniform_(-math.pi, math.pi)
    else:
      theta = torch.zeros(n, device=self.device)
    self.tar_dir_w[env_ids, 0] = torch.cos(theta)
    self.tar_dir_w[env_ids, 1] = torch.sin(theta)

    self.tar_speed[env_ids] = torch.empty(n, device=self.device).uniform_(
      self.cfg.tar_speed_min, self.cfg.tar_speed_max
    )

    if self.cfg.rand_face_dir:
      face_theta = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    else:
      face_theta = theta
    self.face_dir_w[env_ids, 0] = torch.cos(face_theta)
    self.face_dir_w[env_ids, 1] = torch.sin(face_theta)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    heading_w = self.robot.data.heading_w
    self.command_b[:, 0:2] = _dir_world_to_local(self.tar_dir_w, heading_w)
    self.command_b[:, 2] = self.tar_speed
    self.command_b[:, 3:5] = _dir_world_to_local(self.face_dir_w, heading_w)

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create steering joystick sliders in the Viser viewer."""
    from viser import Icon

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      speed_slider = server.gui.add_slider(
        "tar_speed",
        min=0.0,
        max=float(self.cfg.tar_speed_max),
        step=0.1,
        initial_value=1.0,
      )
      tar_angle_slider = server.gui.add_slider(
        "tar_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
      face_angle_slider = server.gui.add_slider(
        "face_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
      zero_btn = server.gui.add_button("Zero speed", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        speed_slider.value = 0.0

    self._gui_enabled = enabled
    self._gui_speed = speed_slider
    self._gui_tar_angle = tar_angle_slider
    self._gui_face_angle = face_angle_slider
    self._gui_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super().compute(dt, env_ids)
    if self._gui_enabled is None or not self._gui_enabled.value:
      return
    assert self._gui_get_env_idx is not None
    assert self._gui_speed is not None
    assert self._gui_tar_angle is not None
    assert self._gui_face_angle is not None
    idx = self._gui_get_env_idx()
    tar_a = float(self._gui_tar_angle.value)
    face_a = float(self._gui_face_angle.value)
    self.tar_dir_w[idx, 0] = math.cos(tar_a)
    self.tar_dir_w[idx, 1] = math.sin(tar_a)
    self.face_dir_w[idx, 0] = math.cos(face_a)
    self.face_dir_w[idx, 1] = math.sin(face_a)
    self.tar_speed[idx] = float(self._gui_speed.value)
    # Refresh heading-frame command so the override shows up in the obs.
    self._update_command()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    tar_dir_ws = self.tar_dir_w.cpu().numpy()
    tar_speed_s = self.tar_speed.cpu().numpy()
    face_dir_ws = self.face_dir_w.cpu().numpy()
    lin_vel_ws = self.robot.data.root_link_lin_vel_w.cpu().numpy()

    z = float(self.cfg.viz.z_offset)
    scale = float(self.cfg.viz.scale)

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      origin = base_pos_w + np.array([0.0, 0.0, z])

      # Commanded velocity arrow (blue): tar_dir * tar_speed in world frame.
      cmd_vec = (
        np.array(
          [
            tar_dir_ws[batch, 0] * tar_speed_s[batch],
            tar_dir_ws[batch, 1] * tar_speed_s[batch],
            0.0,
          ]
        )
        * scale
      )
      visualizer.add_arrow(
        origin, origin + cmd_vec, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Face direction arrow (red): unit-length world-frame face_dir.
      face_vec = np.array([face_dir_ws[batch, 0], face_dir_ws[batch, 1], 0.0]) * scale
      visualizer.add_arrow(
        origin, origin + face_vec, color=(0.8, 0.0, 0.0, 0.7), width=0.015
      )

      # Actual linear velocity arrow (cyan) in world frame.
      vel_vec = np.array([lin_vel_ws[batch, 0], lin_vel_ws[batch, 1], 0.0]) * scale
      visualizer.add_arrow(
        origin, origin + vel_vec, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class SteeringCommandCfg(CommandTermCfg):
  entity_name: str
  rand_tar_dir: bool = True
  rand_face_dir: bool = True
  tar_speed_min: float = 0.5
  tar_speed_max: float = 3.0

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.viz is None:
      self.viz = SteeringCommandCfg.VizCfg()

  def build(self, env: "ManagerBasedRlEnv") -> SteeringCommand:
    return SteeringCommand(self, env)
