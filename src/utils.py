"""Seeding, EMA, checkpoint IO, logging.

Checkpoint dict contract (V35): model, ema, config, iter, metrics, git.
build_model(ckpt["config"]) must load the stored state dict with strict=True.

Owner: trainer.
"""

from __future__ import annotations

from typing import Any


def seed_everything(seed: int) -> None:
    """Seed random, numpy, torch and torch.cuda (V44)."""
    raise NotImplementedError("seed_everything: not implemented yet")


def git_sha() -> str:
    """Short git SHA of HEAD, for the checkpoint and the experiment ledger."""
    raise NotImplementedError("git_sha: not implemented yet")


class EMA:
    """Exponential moving average of weights. The shipped checkpoint uses EMA."""

    def __init__(self, model: Any, decay: float = 0.999) -> None:
        raise NotImplementedError("EMA.__init__: not implemented yet")
