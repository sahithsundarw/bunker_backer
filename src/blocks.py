"""Building blocks: LayerNorm2d, SimpleGate, SCA, NAFBlock, PixelShuffle head (SPEC 7.1).

Design rules enforced here (all are verifier-relevant, not stylistic):

* **No BatchNorm anywhere.** BN's running statistics make the forward pass batch-size
  dependent at inference; the shipped batch size is a tuning knob, so BN would make the
  score depend on it. Normalisation is channel LayerNorm (`LayerNorm2d`) or none.
* **No dropout and no stochastic layer.** V24 asserts bit-identical repeat runs in
  ``eval()``; the cheapest way to guarantee that is for no stochastic op to exist at all.
* **Fully convolutional, required size multiple 1.** Nothing here reads a spatial extent
  from a config or holds a positional table. `SCA` uses a global mean over H, W, which is
  defined for any (H, W) including odd and non-square (SPEC_ADDENDUM section 1).
* **No clipping, no input normalisation.** Inputs legitimately reach [-0.28, 2.16]
  (SPEC_ADDENDUM section 4); clipping happens at save time only.

Module-level imports are deliberately minimal: `inference.py` transitively imports this
module and its import cost is inside KLA's measured wall-clock (docs/decisions.md D7).

Owner: model-core.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LayerNorm2d",
    "SimpleGate",
    "SCA",
    "NAFBlock",
    "PixelShuffleHead",
    "ConvReLU",
    "bilinear_upsample",
]


def bilinear_upsample(x: torch.Tensor, scale: int = 2) -> torch.Tensor:
    """Global residual skip: bilinear upsample by an integer factor.

    ``align_corners=False`` per SPEC 7.1. `recompute_scale_factor` is left at its default
    so the output is exactly ``(scale*H, scale*W)`` for any input, including odd sizes.
    Deterministic in both train and eval.
    """
    return F.interpolate(x, scale_factor=float(scale), mode="bilinear", align_corners=False)


class LayerNorm2d(nn.Module):
    """Channel LayerNorm for NCHW tensors (normalises over C at each spatial location).

    Chosen over BatchNorm because it is independent of batch size and of the other images
    in the batch, and over InstanceNorm/GroupNorm-over-space because it does not couple
    the statistics to the spatial extent -- a model trained on 64x64 LR patches is then
    used unchanged on 128x128 and 256x256 whole images.

    **Implemented via ``F.layer_norm`` on an NHWC view, not by hand.** The obvious manual
    form ``(x - x.mean(1)) / sqrt(x.var(1) + eps)`` is arithmetically identical but slower
    and much heavier: it is bandwidth-bound and autograd saves every intermediate.

    MEASURED, NAFSR w48 n16 on an RTX 4060 Laptop GPU, bf16 autocast + channels_last;
    400 images at 128x128 batch 16 for inference, batch 32 at 64x64 for the train step.
    Five repeats per variant, interleaved to cancel thermal drift, medians reported:

        manual reduction : 400 img 4939 ms | train step 305 ms | peak 4970 MiB
        F.layer_norm     : 400 img 4233 ms | train step 208 ms | peak 3486 MiB
                           -> 1.17x inference, 1.46x training, 1.43x less VRAM

    The inference gap is outside run-to-run noise (manual range 4931-5269 ms, fused
    4196-4329 ms -- non-overlapping). An earlier single-shot comparison put the inference
    win at only 1.09x, which was inside the ~9% run variance and should not have been
    quoted; interleaved repeats were needed to make the claim honestly.

    Profiling attributes 32.8% of forward CUDA time to layer_norm even in the fused form
    -- autocast promotes it to fp32 by policy -- so that is the floor for the op as
    written, not a remaining bug.

    ``permute`` on a channels_last tensor is a metadata-only view, so the transposition is
    free in the configuration we actually run (``channels_last: true``, SPEC 9).
    Do not "simplify" this back to the manual form.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (int(channels),)
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
        )
        return y.permute(0, 3, 1, 2)

    def extra_repr(self) -> str:
        return f"channels={self.weight.numel()}, eps={self.eps}"


class SimpleGate(nn.Module):
    """Channel-split elementwise product (NAFNet). Not an activation function.

    ``(B, 2C, H, W) -> (B, C, H, W)``. Halves the channel count, so the caller must size
    the following layer at ``C``, not ``2C``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class SCA(nn.Module):
    """Simplified Channel Attention (NAFNet): global mean -> 1x1 conv -> multiply.

    No sigmoid and no non-linearity, matching NAFNet. The global mean is taken over all
    spatial positions, which is why this module is size-agnostic but *not* shift-local:
    every output pixel depends on the whole input. That is intentional (it is where the
    block gets its global context) but it means the model must never be applied tile-wise
    without overlap, or tiles would get inconsistent gains. Whole images only.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean over (H, W) keeping dims -> (B, C, 1, 1); defined for any H, W >= 1.
        w = self.conv(x.mean(dim=(2, 3), keepdim=True))
        return x * w


class NAFBlock(nn.Module):
    """NAFNet block, exactly as SPEC 7.1 specifies.

    ``LN -> Conv1x1 -> DWConv3x3 -> SimpleGate -> SCA -> Conv1x1 (+res, layerscale)``
    then ``LN -> Conv1x1 -> SimpleGate -> Conv1x1 (+res, layerscale)``.

    No activation functions: `SimpleGate` supplies the non-linearity as a channel-split
    elementwise product.

    `layerscale_init` deviates from the official NAFNet release, which initialises the
    residual scales (`beta`, `gamma`) to **zero**. Zero-init makes every block an exact
    identity at step 0, so branch weights get gradient only in proportion to `beta`, which
    must itself grow away from zero first. The default here is 1.0 -- a plain residual
    with a learnable per-channel gain.

    UNMEASURED: this is an argument from the shape of the gradient, not a result. No
    convergence comparison has been run, because no training run exists yet. It is exposed
    as a config key precisely so it can be ablated (0.0 vs 1.0 vs 1e-2) once there is a
    training loop to ablate with; until then it is a default, not a finding.
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        layerscale_init: float = 1.0,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        dw_c = channels * dw_expand
        ffn_c = channels * ffn_expand
        if dw_c % 2 or ffn_c % 2:
            raise ValueError(
                "SimpleGate halves the channel count, so channels*dw_expand and "
                f"channels*ffn_expand must be even (got {dw_c}, {ffn_c})"
            )

        # --- spatial branch -------------------------------------------------------
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_c, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_c, dw_c, kernel_size=3, padding=1, groups=dw_c, bias=True,
            padding_mode=str(padding_mode),
        )
        self.gate1 = SimpleGate()
        self.sca = SCA(dw_c // 2)
        self.conv2 = nn.Conv2d(dw_c // 2, channels, kernel_size=1, bias=True)

        # --- channel-MLP branch ---------------------------------------------------
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_c, kernel_size=1, bias=True)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_c // 2, channels, kernel_size=1, bias=True)

        # --- per-channel residual scales (layerscale) -----------------------------
        self.beta = nn.Parameter(torch.full((1, channels, 1, 1), float(layerscale_init)))
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), float(layerscale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.gate2(y)
        y = self.conv4(y)
        return x + y * self.gamma


class PixelShuffleHead(nn.Module):
    """The only module that changes resolution: Conv3x3(C -> s^2*C) -> PixelShuffle(s).

    A single PixelShuffle at the very end keeps all heavy computation at LR resolution,
    where it is ``s^2`` times cheaper (SPEC 7.1). ``out_ch`` is produced by a final 3x3
    conv at HR resolution, which is the only HR-resolution work in the network.
    """

    def __init__(
        self,
        channels: int,
        out_ch: int = 1,
        scale: int = 2,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        if scale < 1:
            raise ValueError(f"scale must be >= 1, got {scale}")
        self.scale = int(scale)
        self.padding_mode = str(padding_mode)
        self.expand = nn.Conv2d(
            channels, channels * scale * scale, kernel_size=3, padding=1,
            padding_mode=self.padding_mode,
        )
        self.shuffle = nn.PixelShuffle(scale)
        self.project = nn.Conv2d(
            channels, out_ch, kernel_size=3, padding=1, padding_mode=self.padding_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.shuffle(self.expand(x)))


class ConvReLU(nn.Module):
    """Conv3x3 + LeakyReLU, the plain-U-Net baseline's unit. No norm, no dropout.

    Deliberately unremarkable: the baseline exists to be a fair, boring reference point
    (V28), so it gets none of the NAFNet machinery. No BatchNorm here either -- the
    baseline must be batch-size independent for the comparison to mean anything.
    """

    def __init__(self, in_ch: int, out_ch: int, slope: float = 0.2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))
