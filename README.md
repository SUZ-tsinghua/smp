# SMP

Score-Matching Motion Priors for humanoid motion tracking.  Trains a small
diffusion model on motion windows; the frozen score is reused as a reward
signal during PPO tracking.  Inspired by Mu et al. (arXiv:2512.03028).

## Status

Only the simplest downstream task is implemented so far: **gait adaptation
to different velocity commands**, trained from **three motion clips**
(walk / jog / run).  The goal is for the policy to pick and execute the
right gait as the commanded forward speed changes.

The end-to-end pipeline:

1. **CSV → NPZ** — slice long motion clips into fixed-length windows with
   pelvis-anchored features.
2. **Normalization stats** — compute per-feature q01/q99 quantiles.
3. **Diffusion pretraining** — train the denoiser on the windowed dataset.
4. **RL** — PPO with the frozen denoiser as an SMP guidance reward, plus a
   velocity tracking term.

## 1. CSV → NPZ

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv \
  --output-dir datasets/npz \
  --window-size 20 \
  --stride 1 \
  --input-fps 30 \
  --output-fps 50
```

Each input CSV holds base pose + DoF trajectories for a motion clip.  The
output NPZ stores `(N, W, F)` windows with a **pelvis-anchored, yaw-only**
frame-0 anchor — the feature layout matches what the online feature buffer
in `smp.rl.utils.PelvisAnchoredFeatureBuffer` reproduces at RL time.

## 2. Normalization stats

```bash
uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/lafan \
  --output datasets/norm_stats.npz
```

**Important**: compute the quantile stats from the **full LAFAN dataset**,
not just the small `loco` subset used for pretraining.  The RL policy
occasionally produces motions that are outside the narrow walk/jog/run
distribution, and a normalizer fit only on that subset makes those states
look severely OOD.  The denoiser's score signal then collapses on the very
cases where we most need it to be reliable.  Fitting on the full dataset
gives a broader [-1, 1] mapping and keeps the score meaningful across the
range of motions the RL policy actually explores.

## 3. Diffusion pretraining

```bash
uv run scripts/pretrain.py \
  --data-dir datasets/npz/loco/ \
  --num-layers 2 \
  --no-use-ema \
  --save-interval 5000 \
  --num-epochs 50000 \
  --train-split 1.0 \
  --log-interval 1000 \
  --wandb-run-name pretrain-loco-window10
```

Standard ε-prediction DDPM with cosine-β schedule (50 timesteps), EMA on
weights, and multi-noise-sample loss for lower-variance gradients.  The
checkpoint is written to `logs/pretrain/<timestamp>/pretrained.pt`.

A pretrained checkpoint is already shipped at
`datasets/pretrain_ckpt/pretrained_loco.pt`, so this step can be skipped
for the walk/jog/run setup.

### Visualize unconditional samples

`scripts/generate_viz.py` loads a checkpoint, runs unconditional DDPM
ancestral sampling to produce a full motion window, and plays it back in
a viser viewer.  Use it to sanity-check a pretrained checkpoint before
wiring it into RL.

```bash
uv run scripts/generate_viz.py \
  --ckpt-path datasets/pretrain_ckpt/pretrained_loco.pt
```

Open the printed URL to see the Frame / Play-Pause / Resample controls.

## 4. RL

The SMP task is registered in `smp.rl.tasks` and uses mjlab's
`ManagerBasedRlEnv`.  The denoiser path is hardcoded on the
`init_smp_state` event inside `src/smp/rl/env_cfg.py` — edit it to point
at your pretrained checkpoint.

```bash
# Train
uv run scripts/train.py Smp-Velocity-G1

# Play
uv run scripts/play.py Smp-Velocity-G1 --wandb-run-path <org>/<project>/<run>
```

The velocity task combines:

- **Track lin/ang vel** rewards in the **yaw-only** base frame (so the
  robot can tilt without being penalized).
- **SMP guidance reward** — SDS-style MSE at a fixed set of diffusion
  timesteps, normalized per-timestep by a count-based running mean
  (MimicKit-style `DiffNormalizer`).
- **GSI reset** — each episode reset draws a full window from the
  denoiser, writes the last frame's state to sim, and fills the feature
  buffer with the sampled trajectory.
