"""Building blocks: NAFBlock, SimpleGate, SCA, PixelShuffle head (SPEC 7.1).

No BatchNorm anywhere -- it is batch-size dependent at inference. LayerNorm or none.
No dropout or any stochastic layer active in eval().

Owner: model-core.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Channel-split elementwise product (NAFNet). No activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SimpleGate.forward: not implemented yet")


class SCA(nn.Module):
    """Simplified Channel Attention (NAFNet)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        raise NotImplementedError("SCA.__init__: not implemented yet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SCA.forward: not implemented yet")


class NAFBlock(nn.Module):
    """LN -> Conv1x1 -> DWConv3x3 -> SimpleGate -> SCA -> Conv1x1 (+res, layerscale)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        raise NotImplementedError("NAFBlock.__init__: not implemented yet")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("NAFBlock.forward: not implemented yet")
