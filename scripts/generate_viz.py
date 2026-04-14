"""Unconditionally generate a motion window with a trained SMP diffusion model
and visualize the predicted trajectory in a viser viewer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
import viser
from mjlab.entity import Entity
from mjlab.viewer.viser.scene import MjlabViserScene

from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.scheduler import DDPMScheduler
from smp.sampling.feature_to_state import window_to_pelvis_trajectory
from smp.utils import detect_device


@dataclass
class Cfg:
  ckpt_path: str = ""
  """Path to a local SMP diffusion checkpoint .pt file. Mutually exclusive with --wandb-run."""
  wandb_run: str = ""
  """W&B run path '<entity>/<project>/<run_id>'. Downloads the latest .pt from the run."""
  device: str = ""
  """Compute device. Empty = auto."""
  fps: float = 50.0
  """Playback frame rate."""


def _resolve_ckpt_path(cfg: Cfg) -> str:
  """Return a local ckpt path, downloading from wandb if --wandb-run is set."""
  if bool(cfg.ckpt_path) == bool(cfg.wandb_run):
    msg = "Specify exactly one of --ckpt-path or --wandb-run"
    raise ValueError(msg)
  if cfg.ckpt_path:
    return cfg.ckpt_path

  import wandb

  api = wandb.Api()
  run = api.run(cfg.wandb_run)
  pt_files = [f for f in run.files() if f.name.endswith(".pt")]
  if not pt_files:
    msg = f"No .pt files in wandb run {cfg.wandb_run}"
    raise FileNotFoundError(msg)
  target = next(
    (f for f in pt_files if Path(f.name).name == "pretrained.pt"),
    sorted(pt_files, key=lambda f: f.name)[-1],
  )
  download_dir = Path("logs") / "wandb_ckpt_cache" / cfg.wandb_run.replace("/", "_")
  download_dir.mkdir(parents=True, exist_ok=True)
  target.download(root=str(download_dir), replace=True)
  local = download_dir / target.name
  print(f"Downloaded {target.name} from {cfg.wandb_run} -> {local}")
  return str(local)


def _build_model_and_scheduler(
  ckpt: dict, device: torch.device
) -> tuple[DiffusionDenoiser, DDPMScheduler, np.ndarray, np.ndarray]:
  cfg = ckpt["cfg"]
  model = DiffusionDenoiser(
    feature_dim=cfg["feature_dim"],
    window_size=cfg["window_size"],
    d_model=cfg.get("d_model", 256),
    nhead=cfg.get("nhead", 8),
    num_layers=cfg.get("num_layers", 2),
    dim_feedforward=cfg.get("dim_feedforward", 1024),
    dropout=cfg.get("dropout", 0.0),
  ).to(device)
  state = ckpt.get("model_ema") or ckpt["model"]
  model.load_state_dict(state)
  model.eval()
  scheduler = DDPMScheduler(
    num_timesteps=cfg.get("num_timesteps", 50),
  ).to(device)
  return model, scheduler, ckpt["q_low"], ckpt["q_high"]


def _setup_g1_sim(device: str):
  """Build a single G1 sim. Mirrors scripts/csv_to_npz.py:_setup_sim."""
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg
  from mjlab.tasks.tracking.config.g1.env_cfgs import (
    unitree_g1_flat_tracking_env_cfg,
  )

  sim_cfg = SimulationCfg()
  env_cfg = unitree_g1_flat_tracking_env_cfg()
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return sim, scene


def _quantile_denormalize(
  x: torch.Tensor, q_low: torch.Tensor, q_high: torch.Tensor
) -> torch.Tensor:
  return (x + 1.0) / 2.0 * (q_high - q_low) + q_low


@torch.no_grad()
def _run_generate(
  model: DiffusionDenoiser,
  scheduler: DDPMScheduler,
  q_low: np.ndarray,
  q_high: np.ndarray,
  window_size: int,
  feature_dim: int,
  device: torch.device,
) -> torch.Tensor:
  """Unconditional DDPM ancestral sampling. Returns (W, F) denormalized window on CPU."""
  x_t = torch.randn(1, window_size, feature_dim, device=device)
  for t in reversed(range(scheduler.num_timesteps)):
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    eps = model(x_t, t_batch)
    x_t = scheduler.step(eps, x_t, t)
  q_low_t = torch.from_numpy(q_low).float().to(device)
  q_high_t = torch.from_numpy(q_high).float().to(device)
  return _quantile_denormalize(x_t.squeeze(0), q_low_t, q_high_t).cpu()


def _write_pose_to_robot(
  robot: Entity,
  pelvis_pos: np.ndarray,
  pelvis_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
  device: str,
) -> None:
  """Mirror scripts/csv_to_npz.py:_fk_motion's per-frame state write."""
  root = robot.data.default_root_state.clone()
  root[:, 0:3] = torch.as_tensor(pelvis_pos, device=device, dtype=root.dtype)
  root[:, 3:7] = torch.as_tensor(pelvis_quat_wxyz, device=device, dtype=root.dtype)
  robot.write_root_state_to_sim(root)
  jp = robot.data.default_joint_pos.clone()
  jp[:] = torch.as_tensor(joint_pos, device=device, dtype=jp.dtype)
  jv = robot.data.default_joint_vel.clone()
  robot.write_joint_state_to_sim(jp, jv)


def main(cfg: Cfg) -> None:
  device_str = cfg.device or detect_device()
  device = torch.device(device_str)
  print(f"Device: {device_str}")

  ckpt_path = _resolve_ckpt_path(cfg)
  ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
  model, scheduler, q_low, q_high = _build_model_and_scheduler(ckpt, device)
  print(f"Loaded checkpoint epoch={ckpt.get('epoch')} from {ckpt_path}")

  feature_dim = int(ckpt["cfg"]["feature_dim"])
  window_size = int(ckpt["cfg"]["window_size"])

  sim_device = device_str
  sim, scene = _setup_g1_sim(sim_device)
  robot: Entity = scene["robot"]
  mj_model = sim.mj_model

  def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_denorm = _run_generate(
      model,
      scheduler,
      q_low,
      q_high,
      window_size,
      feature_dim,
      device,
    )
    p0 = torch.zeros(3)
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    p_pos, p_quat, p_joint = window_to_pelvis_trajectory(pred_denorm, p0, q0)
    return p_pos.cpu().numpy(), p_quat.cpu().numpy(), p_joint.cpu().numpy()

  state: dict = {"pred": run()}

  server = viser.ViserServer()
  viser_scene = MjlabViserScene(server, mj_model, num_envs=1)
  viser_scene.debug_visualization_enabled = True

  with server.gui.add_folder("Generate"):
    frame_slider = server.gui.add_slider(
      "Frame", min=0, max=window_size - 1, step=1, initial_value=0
    )
    play_btn = server.gui.add_button("Play / Pause")
    resample_btn = server.gui.add_button("Resample")

  playing = {"v": True}

  @play_btn.on_click
  def _(_evt) -> None:
    playing["v"] = not playing["v"]

  @resample_btn.on_click
  def _(_evt) -> None:
    state["pred"] = run()

  def render(frame: int) -> None:
    p_pos, p_quat, p_joint = state["pred"]
    _write_pose_to_robot(robot, p_pos[frame], p_quat[frame], p_joint[frame], sim_device)
    sim.forward()
    wd = sim.wp_data
    viser_scene.update_from_arrays(
      body_xpos=np.asarray(wd.xpos.numpy()),
      body_xmat=np.asarray(wd.xmat.numpy()),
      qpos=np.asarray(wd.qpos.numpy()),
      env_idx=0,
    )
    viser_scene.refresh_visualization()

  print("Viser server running. Open the printed URL.")
  dt = 1.0 / cfg.fps
  try:
    while True:
      render(int(frame_slider.value))
      if playing["v"]:
        nxt = (int(frame_slider.value) + 1) % window_size
        frame_slider.value = nxt
      time.sleep(dt)
  except KeyboardInterrupt:
    print("Shutting down.")


if __name__ == "__main__":
  main(tyro.cli(Cfg))
