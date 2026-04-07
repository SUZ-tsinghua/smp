# SMP

Score-Matching Motion Priors for humanoid motion tracking. Trains a small
diffusion model on motion windows; the frozen score is reused as a reward
signal during PPO tracking. Inspired by Mu et al., arXiv:2512.03028.
