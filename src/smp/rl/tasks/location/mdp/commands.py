"""Location command — periodic world-frame xy goal.

Each env carries a world-frame xy target (random angle + distance from the
current position); the command exposed to the policy is the local-frame offset
to the goal, so the observation is yaw-invariant.
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


def _xy_world_to_local(vec_w: torch.Tensor, heading_w: torch.Tensor) -> torch.Tensor:
  """Rotate a (N, 2) world-frame xy vector into the heading-aligned frame."""
  cos_h = torch.cos(heading_w)
  sin_h = torch.sin(heading_w)
  x_w, y_w = vec_w[..., 0], vec_w[..., 1]
  return torch.stack([cos_h * x_w + sin_h * y_w, -sin_h * x_w + cos_h * y_w], dim=-1)


class LocationCommand(CommandTerm):
  """Periodic world-frame xy goal command."""

  cfg: LocationCommandCfg

  def __init__(self, cfg: LocationCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    # World-frame state.
    self.tar_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

    # Heading-frame command exposed to the policy: [local_tar_x, local_tar_y].
    self.command_b = torch.zeros(self.num_envs, 2, device=self.device)

    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._gui_enabled: viser.GuiCheckboxHandle | None = None
    self._gui_distance: viser.GuiSliderHandle | None = None
    self._gui_angle: viser.GuiSliderHandle | None = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.command_b

  def _update_metrics(self) -> None:
    max_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    pos_err = torch.norm(
      self.tar_pos_w - self.robot.data.root_link_pos_w[:, :2], dim=-1
    )
    self.metrics["error_pos"] += pos_err / max_step

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = int(env_ids.numel())
    root_pos_xy = self.robot.data.root_link_pos_w[env_ids, :2]

    rand_dist = torch.empty(n, device=self.device).uniform_(
      self.cfg.tar_dist_min, self.cfg.tar_dist_max
    )
    rand_theta = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)
    self.tar_pos_w[env_ids, 0] = root_pos_xy[:, 0] + rand_dist * torch.cos(rand_theta)
    self.tar_pos_w[env_ids, 1] = root_pos_xy[:, 1] + rand_dist * torch.sin(rand_theta)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    root_pos_xy = self.robot.data.root_link_pos_w[:, :2]
    heading_w = self.robot.data.heading_w
    self.command_b[:] = _xy_world_to_local(self.tar_pos_w - root_pos_xy, heading_w)

  # GUI.

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create location goal sliders in the Viser viewer."""
    from viser import Icon

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      distance_slider = server.gui.add_slider(
        "tar_distance",
        min=0.0,
        max=float(self.cfg.tar_dist_max),
        step=0.1,
        initial_value=float(self.cfg.tar_dist_max * 0.5),
      )
      angle_slider = server.gui.add_slider(
        "tar_angle (rad)",
        min=-math.pi,
        max=math.pi,
        step=0.05,
        initial_value=0.0,
      )
      resample_btn = server.gui.add_button("Resample target", icon=Icon.REFRESH)

      @resample_btn.on_click
      def _(_) -> None:
        # Force a fresh random target on the visualized env.
        idx = get_env_idx()
        env_ids = torch.tensor([idx], device=self.device, dtype=torch.long)
        self._resample_command(env_ids)
        self._update_command()

    self._gui_enabled = enabled
    self._gui_distance = distance_slider
    self._gui_angle = angle_slider
    self._gui_get_env_idx = get_env_idx

  def compute(
    self, dt: float | torch.Tensor, env_ids: torch.Tensor | None = None
  ) -> None:
    super().compute(dt, env_ids)
    if self._gui_enabled is None or not self._gui_enabled.value:
      return
    assert self._gui_get_env_idx is not None
    assert self._gui_distance is not None
    assert self._gui_angle is not None
    idx = self._gui_get_env_idx()
    distance = float(self._gui_distance.value)
    angle = float(self._gui_angle.value)
    root_pos_xy = self.robot.data.root_link_pos_w[idx, :2]
    self.tar_pos_w[idx, 0] = float(root_pos_xy[0]) + distance * math.cos(angle)
    self.tar_pos_w[idx, 1] = float(root_pos_xy[1]) + distance * math.sin(angle)
    # Refresh heading-frame command so the override shows up in the obs.
    self._update_command()

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    tar_pos_ws = self.tar_pos_w.cpu().numpy()

    z = float(self.cfg.viz.z_offset)
    radius = float(self.cfg.viz.marker_radius)
    pole_height = float(self.cfg.viz.pole_height)

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      tar_xy = tar_pos_ws[batch]
      marker_base = np.array([tar_xy[0], tar_xy[1], 0.0])
      marker_top = np.array([tar_xy[0], tar_xy[1], pole_height])

      # Marker sphere on the ground.
      visualizer.add_sphere(marker_base, radius=radius, color=(0.85, 0.0, 0.0, 0.85))
      # Vertical pole so the goal stays visible from far away.
      visualizer.add_cylinder(
        marker_base, marker_top, radius=radius * 0.25, color=(0.85, 0.0, 0.0, 0.6)
      )
      # Heading-height line from char to target.
      line_start = base_pos_w + np.array([0.0, 0.0, z])
      line_end = np.array([tar_xy[0], tar_xy[1], line_start[2]])
      visualizer.add_arrow(line_start, line_end, color=(1.0, 0.3, 0.3, 0.6), width=0.01)


@dataclass(kw_only=True)
class LocationCommandCfg(CommandTermCfg):
  entity_name: str
  tar_dist_min: float = 1.0
  tar_dist_max: float = 10.0

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    marker_radius: float = 0.12
    pole_height: float = 1.0

  viz: VizCfg = None  # type: ignore[assignment]

  def __post_init__(self) -> None:
    if self.viz is None:
      self.viz = LocationCommandCfg.VizCfg()
    if self.tar_dist_max < self.tar_dist_min:
      msg = (
        f"tar_dist_max ({self.tar_dist_max}) must be >= "
        f"tar_dist_min ({self.tar_dist_min})."
      )
      raise ValueError(msg)

  def build(self, env: "ManagerBasedRlEnv") -> LocationCommand:
    return LocationCommand(self, env)
