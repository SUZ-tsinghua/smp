"""Diffusion model pretraining loop."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import tyro
from torch.utils.data import DataLoader, random_split

from smp.pretrain.dataset import MotionWindowDataset
from smp.pretrain.model import DiffusionDenoiser
from smp.pretrain.pretrain_cfg import PretrainCfg
from smp.pretrain.scheduler import DDPMScheduler
from smp.utils import count_parameters, seed_everything


class _Ema:
  """Exponential moving average shadow of a model.

  Standard formula: ``θ_ema ← decay·θ_ema + (1−decay)·θ``, applied in-place
  over every entry of ``state_dict()`` (covers params and buffers).
  """

  def __init__(self, model: torch.nn.Module, decay: float) -> None:
    self.decay = decay
    self.shadow = copy.deepcopy(model)
    self.shadow.eval()
    for p in self.shadow.parameters():
      p.requires_grad_(False)

  @torch.no_grad()
  def update(self, model: torch.nn.Module) -> None:
    src = model.state_dict()
    dst = self.shadow.state_dict()
    for k, v_src in src.items():
      v_dst = dst[k]
      if v_dst.is_floating_point():
        v_dst.mul_(self.decay).add_(v_src.detach(), alpha=1.0 - self.decay)
      else:
        v_dst.copy_(v_src)


def _diffusion_loss(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  x_0: torch.Tensor,
) -> torch.Tensor:
  """Standard DDPM epsilon-prediction loss for one batch."""
  t = scheduler.sample_timesteps(x_0.shape[0], x_0.device)
  noise = torch.randn_like(x_0)
  x_t = scheduler.add_noise(x_0, noise, t)
  return F.mse_loss(model(x_t, t), noise)


def _save_checkpoint(
  path: Path,
  epoch: int,
  model: DiffusionDenoiser,
  dataset: MotionWindowDataset,
  feature_dim: int,
  cfg: PretrainCfg,
  optimizer: torch.optim.Optimizer | None = None,
  ema: _Ema | None = None,
) -> None:
  data: dict[str, Any] = {
    "epoch": epoch,
    "model": model.state_dict(),
    "norm_mean": dataset.mean,
    "norm_std": dataset.std,
    "cfg": {
      **vars(cfg),
      "feature_dim": feature_dim,
      "window_size": dataset.window_size,
    },
  }
  if optimizer is not None:
    data["optimizer"] = optimizer.state_dict()
  if ema is not None:
    data["model_ema"] = ema.shadow.state_dict()
  torch.save(data, path)


def pretrain(cfg: PretrainCfg) -> Path:
  """Run diffusion pretraining."""
  seed_everything(cfg.seed)
  print(f"[INFO] seed={cfg.seed}")
  device = torch.device(cfg.device)

  dataset = MotionWindowDataset(cfg.data_dir, normalize=cfg.normalize)
  feature_dim = dataset.feature_dim
  window_size = dataset.window_size

  n_train = int(len(dataset) * cfg.train_split)
  n_val = len(dataset) - n_train
  print(
    f"Dataset: {len(dataset)} windows, n_train={n_train}, n_val={n_val}, "
    f"feature_dim={feature_dim}, window_size={window_size}"
  )

  train_set, val_set = random_split(dataset, [n_train, n_val])
  pin_memory = device.type == "cuda"
  train_loader = DataLoader(
    train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=pin_memory,
  )
  val_loader = DataLoader(
    val_set, batch_size=cfg.batch_size, shuffle=False, pin_memory=pin_memory
  )

  model = DiffusionDenoiser(
    feature_dim=feature_dim,
    window_size=window_size,
    d_model=cfg.d_model,
    nhead=cfg.nhead,
    num_layers=cfg.num_layers,
    dim_feedforward=cfg.dim_feedforward,
    dropout=cfg.dropout,
  ).to(device)
  scheduler = DDPMScheduler(
    num_timesteps=cfg.num_timesteps,
  ).to(device)
  print(f"Denoiser: {count_parameters(model):,} params")

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )

  ema = _Ema(model, decay=cfg.ema_decay) if cfg.use_ema else None
  if ema is not None:
    print(f"EMA enabled (decay={cfg.ema_decay})")

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = Path(cfg.log_dir) / timestamp
  save_dir.mkdir(parents=True, exist_ok=True)

  wandb_run = None
  if cfg.use_wandb:
    import wandb

    wandb_run = wandb.init(
      project=cfg.wandb_project, name=cfg.wandb_run_name, config=vars(cfg)
    )

  for epoch in range(cfg.num_epochs):
    model.train()
    epoch_loss = torch.zeros((), device=device)
    n_batches = 0

    for batch in train_loader:
      x_0 = batch.to(device, non_blocking=pin_memory)
      loss = _diffusion_loss(model, scheduler, x_0)

      optimizer.zero_grad()
      loss.backward()
      if cfg.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
      optimizer.step()
      if ema is not None:
        ema.update(model)

      epoch_loss += loss.detach()
      n_batches += 1

    avg_loss = (epoch_loss / max(n_batches, 1)).item()

    if epoch % cfg.log_interval == 0:
      eval_model = ema.shadow if ema is not None else model
      val_loss = _validate(eval_model, scheduler, val_loader, device, pin_memory)
      print(f"Epoch {epoch:4d} | train={avg_loss:.6f} | val={val_loss:.6f}")
      if wandb_run is not None:
        wandb_run.log({"epoch": epoch, "train/loss": avg_loss, "val/loss": val_loss})

    if epoch % cfg.save_interval == 0 or epoch == cfg.num_epochs - 1:
      ckpt_path = save_dir / f"checkpoint_{epoch:05d}.pt"
      _save_checkpoint(
        ckpt_path, epoch, model, dataset, feature_dim, cfg, optimizer, ema
      )
      if wandb_run is not None:
        wandb_run.save(str(ckpt_path), base_path=str(save_dir))

  final_path = save_dir / "pretrained.pt"
  _save_checkpoint(
    final_path, cfg.num_epochs, model, dataset, feature_dim, cfg, ema=ema
  )
  print(f"Saved final checkpoint to {final_path}")

  if wandb_run is not None:
    wandb_run.save(str(final_path), base_path=str(save_dir))
    wandb_run.finish()

  return final_path


@torch.no_grad()
def _validate(
  model: torch.nn.Module | DiffusionDenoiser,
  scheduler: DDPMScheduler,
  val_loader: DataLoader[torch.Tensor],
  device: torch.device,
  pin_memory: bool,
) -> float:
  model.eval()
  total = torch.zeros((), device=device)
  n = 0
  for batch in val_loader:
    x_0 = batch.to(device, non_blocking=pin_memory)
    total += _diffusion_loss(model, scheduler, x_0)
    n += 1
  return (total / max(n, 1)).item()


if __name__ == "__main__":
  pretrain(tyro.cli(PretrainCfg))
