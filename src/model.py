"""Network architectures. Public entry point: build_model(cfg).

SPEC 7.1: NAFNet-style body at LR resolution + global bilinear-upsample residual skip
+ single PixelShuffle(2) head, single channel in and out. Plain U-Net baseline selectable
by config (SPEC 10 requires a learned baseline).

Owner: model-core (see CLAUDE.md FILE OWNERSHIP MAP).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """Construct a model from a config mapping.

    This signature is depended on by inference.py and by the checkpoint contract
    (V35: build_model(ckpt["config"]) must load the stored state dict with strict=True).
    Do not change it without updating both.

    Must be fully convolutional: any (H, W) in -> exactly (2H, 2W) out. No hard-coded 128
    or 256 (SPEC_ADDENDUM section 1 -- the dataset is uniformly 128->256, so the 256->512
    path is exercised only by the synthetic fixture required by SPEC T6).
    """
    raise NotImplementedError("build_model: not implemented yet")
