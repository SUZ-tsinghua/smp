"""Entry point for SMP diffusion pretraining."""

from __future__ import annotations

import tyro

from smp.config.pretrain_cfg import PretrainCfg
from smp.training.pretrain import pretrain


def main() -> None:
  pretrain(tyro.cli(PretrainCfg))


if __name__ == "__main__":
  main()
