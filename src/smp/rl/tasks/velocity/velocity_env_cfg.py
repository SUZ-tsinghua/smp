"""G1 velocity-tracking task with SMP guidance reward.

Builds on the minimal ``g1_smp_env_cfg`` and adds the twist command,
tracking observations/rewards, and fall/base-height terminations from
mjlab's stock velocity task.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg

# from mjlab.managers.command_manager import CommandTermCfg
# from mjlab.managers.curriculum_manager import CurriculumTermCfg
# from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp

from smp.rl.env_cfg import g1_smp_env_cfg
from smp.rl.tasks.velocity.mdp import (
  UniformVelocityCommandYawCfg,
  track_angular_velocity_yaw,
  track_linear_velocity_yaw,
)


def g1_velocity_smp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the G1 velocity tracking env cfg with SMP guidance."""
  cfg = g1_smp_env_cfg(play=play)

  # --- Commands ------------------------------------------------------------
  cfg.commands["twist"] = UniformVelocityCommandYawCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_forward_envs=1.0,
    debug_vis=True,
    ranges=UniformVelocityCommandYawCfg.Ranges(
      lin_vel_x=(0.5, 3.0),
      lin_vel_y=(-0.1, 0.1),
      ang_vel_z=(-0.1, 0.1),
    ),
  )

  # --- Curriculum ----------------------------------------------------------
  # cfg.curriculum = {
  #   "command_vel": CurriculumTermCfg(
  #     func=mdp.commands_vel,
  #     params={
  #       "command_name": "twist",
  #       "velocity_stages": [
  #         {"step": 0, "lin_vel_x": (0.5, 1.5)},
  #         {"step": 3000 * 24, "lin_vel_x": (0.5, 2.5)},
  #         {"step": 6000 * 24, "lin_vel_x": (0.5, 3.0)},
  #       ],
  #     },
  #   ),
  # }

  # --- Observations --------------------------------------------------------
  command_obs = ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "twist"},
  )
  cfg.observations["actor"].terms["command"] = command_obs
  cfg.observations["critic"].terms["command"] = command_obs

  # --- Rewards -------------------------------------------------------------
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_linear_velocity_yaw,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(1.0)},
  )
  cfg.rewards["track_angular_velocity"] = RewardTermCfg(
    func=track_angular_velocity_yaw,
    weight=1.0,
    params={"command_name": "twist", "std": math.sqrt(0.5)},
  )

  # --- Events --------------------------------------------------------------
  cfg.events["init_smp_state"].params["ckpt_path"] = (
    "datasets/pretrain_ckpt/pretrained_loco.pt"
  )
  #   cfg.events["reset_base"] = EventTermCfg(
  #       func=mdp.reset_root_state_uniform,
  #       mode="reset",
  #       params={
  #         "pose_range": {
  #           "x": (-0.5, 0.5),
  #           "y": (-0.5, 0.5),
  #           "z": (0.01, 0.05),
  #           "yaw": (-0.0, 0.0),
  #         },
  #         "velocity_range": {},
  #       },
  #     )

  # --- Contact sensor: arms vs ground --------------------------------------
  upper_body_ground_cfg = ContactSensorCfg(
    name="upper_body_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern="torso_link",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (upper_body_ground_cfg,)

  # --- Terminations --------------------------------------------------------
  cfg.terminations["upper_body_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": upper_body_ground_cfg.name},
  )
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={
      "minimum_height": 0.3,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  return cfg
