# SMP — Score-Matching Motion Priors (reproduction)

A reproduction of **SMP: Reusable Score-Matching Motion Priors for Physics-Based
Character Control** (Mu et al., 2025) on the **Unitree G1** humanoid — the
original MimicKit implementation does not include a G1 setup, so this repo ports
the method to G1 end to end (motion features, priors, tasks, and rewards).

A small diffusion model (DDPM) is pretrained on motion windows; its **frozen
score** is then reused as an SDS-style *guidance reward* during PPO, so a policy
learns naturalistic motion for a downstream task without any per-task motion
clip or adversarial discriminator.

This is a reproduction for a course project. It re-implements the SMP idea on top
of [**mjlab**](https://github.com/mujocolab/mjlab) (the `ManagerBasedRlEnv` and
`mjlab.scripts.train` / `play` entrypoints are reused). The original method and
reference implementation are:

- **Paper:** SMP, Mu et al. 2025 — [arXiv:2512.03028](https://arxiv.org/abs/2512.03028) · [project page](https://yxmu.foo/smp-page/)
- **Original code:** [`xbpeng/MimicKit`](https://github.com/xbpeng/MimicKit) (see `docs/README_SMP.md`)

> The main intentional divergence from the original is the reward composition —
> see [Reward design](#reward-design-task--smp) below.

## Provided pretrained priors

To let you skip pretraining and run RL directly, **three pretrained diffusion
priors are shipped** in `datasets/pretrain_ckpt/`. Each task's env config already
points its `init_smp_state` event at the right one, so no setup is needed:

| Checkpoint                       | Trained on            | Used by                          |
| -------------------------------- | --------------------- | -------------------------------- |
| `pretrained_loco.pt`             | walk / jog / run      | `Smp-Forward-G1`                 |
| `pretrained_lafan_run.pt`        | LAFAN run subset      | `Smp-Steering-G1`, `Smp-Location-G1` |
| `pretrained_getup_f2s2.pt`       | get-up (fall→stand)   | `Smp-Getup-G1`                   |

## Setup

[`uv`](https://docs.astral.sh/uv/) is the canonical package manager; dependencies
(including the pinned `mjlab` git rev) are locked in `uv.lock`.

```bash
uv sync
```

## Pipeline

1. **Data processing** (CSV → windowed NPZ → normalization stats) — documented below.
2. **Diffusion pretraining** (DDPM ε-predictor on motion windows) — documented below.
   You can skip this entirely using the [shipped checkpoints](#provided-pretrained-priors).
3. **RL** (PPO with the frozen prior as a guidance reward) — documented below.

---

## Data processing

Stage 1 turns raw motion CSVs into the windowed feature tensors the diffusion
prior trains on, plus the normalization stats shared by pretraining and RL. Both
scripts are `tyro`-driven — run them under `uv` from the project root.

### Input CSVs (LAFAN1 retargeting format)

Inputs must follow the same per-frame CSV layout as the
**[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)**
(`g1` split) — this is the format `mjlab`'s `MotionLoader` reads. Each file is
header-less, comma-separated, one row per frame at **30 fps**, with **36 columns**:

| Columns | Field                       | Notes                                                              |
| ------- | --------------------------- | ------------------------------------------------------------------ |
| 0–2     | root position `x y z`       | world frame, metres                                                |
| 3–6     | root orientation quaternion | **`x y z w`** order                                                |
| 7–35    | 29 G1 joint angles          | radians; joint order = `JOINT_NAMES` in `scripts/csv_to_npz.py`    |

The CSVs are **not** shipped with this repo — download them yourself. To get the
full G1 set, grab the dataset's
[`g1/`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1)
folder into `datasets/csv/lafan/`:

```bash
# e.g. with the Hugging Face CLI:  pip install -U huggingface_hub
hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
  --include "g1/*.csv" --local-dir datasets/csv/_lafan_dl
mv datasets/csv/_lafan_dl/g1/*.csv datasets/csv/lafan/
```

> `csv_to_npz.py` globs `*.csv` **non-recursively**, so the CSV files must sit
> directly under `--input-dir` (not in a nested `g1/`).

### CSV → windowed NPZ

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/lafan \
  --output-dir datasets/npz/lafan
```

For each CSV this replays the motion through the G1 sim, forward-kinematics the
tracked end-effectors, interpolates 30 → 50 fps, and slices the result into
**pelvis-anchored, yaw-only** windows of shape `(N, window_size, 59)` — one `.npz`
per clip. The 59-dim per-frame feature layout (reproduced online by the RL feature
buffer) is `[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3),
root_ang_vel(3)]`.

Useful flags: `--window-size` (default `10`), `--stride` (`1`), `--input-fps`
(`30`), `--output-fps` (`50`), and `--shard-index / --num-shards` to split a large
corpus across parallel runs.

### Normalization stats

```bash
uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/lafan \
  --output datasets/norm_stats.npz
```

This concatenates every window under `--input-dir` and writes per-feature
**q01/q99 quantiles** (`q_low` / `q_high`; tunable via `--q-low / --q-high`) used
to map features to `[-1, 1]`.

**A `datasets/norm_stats.npz` computed over the full LAFAN G1 set is checked into
the repo**, and pretraining defaults to it (`--norm-stats-file
datasets/norm_stats.npz`). When (re)training a prior, **prefer reusing this
LAFAN-computed file** rather than recomputing stats on your own (often narrow)
clips — recompute only if you change the feature layout. The note below explains
why a wide normalizer matters.

> **Compute these on a diverse dataset (all of LAFAN), not a narrow one.** The
> q01/q99 range defines the feature normalization and is **baked into the
> pretrained checkpoint** — the exact mapping the frozen denoiser sees at RL time.
> A PPO policy routinely wanders outside the pretraining motion distribution; if the
> normalizer was fit to a small set (e.g. just the walk/jog/run clips), those
> out-of-range features saturate toward ±1, the denoiser receives
> **out-of-distribution** inputs, and its score estimate — hence the SMP guidance
> reward — degrades exactly where the policy needs it. A wide normalizer keeps the
> score reliable across the states RL actually visits.

---

## Diffusion pretraining

### Convert the CSV dataset to NPZ

First convert the corresponding CSV dataset into windowed NPZ files. For the
forward prior:

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/forward \
  --output-dir datasets/npz/forward
```

### Train the forward prior

```bash
uv run scripts/pretrain.py --data-dir datasets/npz/forward/ --num-layers 2 --no-use-ema --save-interval 5000 --num-epochs 10000 --train-split 1.0 --d-model 128
```

---

## RL

Four downstream tasks are registered with `mjlab.tasks.registry` (importing
`smp.rl.tasks` self-registers them):

| Task              | Demo | Description                              |
| ----------------- | :--: | ---------------------------------------- |
| `Smp-Forward-G1`  | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/forward.gif" width="200"/> | walk / jog / run at a commanded `+x` speed |
| `Smp-Steering-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/steering.gif" width="200"/> | track a commanded velocity + facing direction |
| `Smp-Location-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/location.gif" width="200"/> | walk to a world-frame xy goal |
| `Smp-Getup-G1`    | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/getup.gif" width="200"/> | stand up from a fallen pose |

### Train / play

```bash
# Train (checkpoints land under logs/)
uv run scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096

# Play a trained policy from a W&B run
uv run scripts/play.py Smp-Forward-G1 --wandb-run-path <org>/<project>/<run> --num-envs 4
```

Swap the task id for any of the four. Because the priors are shipped and already
wired into each env config, no editing is required before training.

### Reward design: `task × SMP`

Every task uses a single **multiplicative** reward term, `task_smp_product`:

```
r  =  ( Σᵢ wᵢ · taskᵢ(s) )  ×  r_smp(s)
```

where `r_smp = exp(−wₛ/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)` is the SDS guidance reward
(the frozen denoiser's ε-prediction error at a fixed set of diffusion timesteps
`K`, per-timestep normalized).

This is the **key divergence from the original SMP / MimicKit**, which combines
the two **additively** and balances them with separate weights
(`task_reward_weight`, `smp_reward_weight`):

```
# original (additive):     r = task_reward_weight · task  +  smp_reward_weight · r_smp
# here     (multiplicative): r = task · r_smp
```

We want the policy to **complete the task _while_ keeping the SMP reward high** —
which is exactly what a product expresses: it is large only when *both* factors
are large, and collapses toward 0 if *either* drops. This makes reward tuning
**easier and more robust**:

- **No task-vs-prior weight to balance.** The additive form needs a
  `task_reward_weight : smp_reward_weight` ratio whose sweet spot shifts per task
  (and per training stage); the product removes that knob entirely.
- **Neither term can be farmed alone.** Additively, a policy can max one term and
  ignore the other — e.g. stand still looking natural (high prior, no task
  progress) or lunge at the goal off-manifold (high task, low prior). With the
  product both failure modes score ≈ 0, so the only way to earn reward is to do
  the task *and* stay on the motion manifold.

Per-task `taskᵢ` components (each weighted, summed, then gated by `r_smp`):

- **Forward** — velocity tracking only: `exp(−s·‖v_cmd − v_xy‖²)`, zeroed when the
  velocity projects backwards onto the target direction. Fixed `+x` heading,
  commanded speed 0.5–5 m/s.
- **Steering** — `0.5·` velocity tracking `+ 0.5·` facing alignment
  `max(face_dir · heading, 0)`; randomized target direction + facing, speed 0.5–2 m/s.
- **Location** — position tracking only: `exp(−s·‖xy_goal − xy_robot‖)` toward a
  periodically resampled world-frame goal (uses `ws=4`).
- **Get-up** — `0.7·` upward head velocity `+ 0.3·` head-height tracking, each
  `exp(−s·max(target − ·, 0)²)`, from a fallen GSI start.

### Generative State Initialization (GSI)

On every reset, an init state is drawn from a pool of windows pre-sampled from the
frozen prior; its last frame seeds the sim state and the whole window primes the
online feature buffer, so `r_smp` is meaningful from step 0. Each env is reset to
its own scene origin while the feature buffer is kept **env-origin-relative**, so
the guidance reward is invariant to where the env sits in the world grid.

### Motion features

The guidance reward scores a rolling window of motion features rebuilt online by
`smp.rl.utils.MotionFeatureBuffer`, matching the pretraining layout (59-dim/frame
for G1), anchored to the last frame's yaw-only local frame:

```
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
```

## Citation & acknowledgements

This repository reproduces SMP; please cite the original work and credit the
reference implementation:

- **SMP** — Mu et al., *Reusable Score-Matching Motion Priors for Physics-Based Character Control*, 2025. [arXiv:2512.03028](https://arxiv.org/abs/2512.03028)
- **MimicKit** — the original SMP implementation: <https://github.com/xbpeng/MimicKit>
- **mjlab** — RL environment backbone: <https://github.com/mujocolab/mjlab>
