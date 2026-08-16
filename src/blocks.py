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
    "NoiseEstimator",
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

    **V22 root cause, fixed here.** ``x.mean(dim=(2,3))`` is a raw spatial reduction, not
    ``F.layer_norm``. Autocast's op-policy table (`ATen/autocast_mode.h`,
    `AT_FORALL_FP32`) force-promotes `layer_norm`/`native_layer_norm`/`group_norm` to fp32
    automatically, but a bare `Tensor.mean` is NOT on that list -- it is not an
    autocast-registered op at all, so it silently executes in whatever dtype its input
    already has. By the time control reaches here under bf16 autocast, `x` is the output of
    an upstream bf16 conv + SimpleGate, so the mean over up to 128x128=16384 elements
    (H*W up to 65536 for the 256x256 group) accumulates and stores in bf16's 8-bit
    mantissa, then feeds a 1x1 conv that is autocast-forced to bf16 again regardless of
    what dtype it receives. Confirmed empirically on this repo's checkpoint + CUDA:

        F.layer_norm(bf16 input, fp32 weight, fp32 bias)   -> fp32 output  (auto-promoted)
        x.mean(dim=(2,3), keepdim=True) of a bf16 input    -> bf16 output  (NOT promoted)

    On `tests/fixtures/single/only_128.npy` (the exact V22 fixture, values to 2.16) this
    measurably moves the final-output bf16-vs-fp32 divergence: baseline max abs diff
    ~1.17e-2 (over V22's 1e-2 cap) drops to ~9.7e-3 (under it) once this module's global
    mean AND the following 1x1 conv are computed in fp32 -- computing the mean alone in
    fp32 and then rounding straight back to bf16 before the conv reproduces the unpatched
    result bit-for-bit, so both the reduction and the conv have to stay inside the
    disabled-autocast region for the fix to do anything.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean over (H, W) keeping dims -> (B, C, 1, 1); defined for any H, W >= 1.
        # Forced to fp32 (see class docstring, V22): unlike F.layer_norm, autocast does
        # not force-promote a raw Tensor.mean reduction, so it must be done explicitly.
        #
        # V24 regression + fix: the first version of this fix called `self.conv(...)`
        # (an nn.Conv2d) inside the disabled-autocast region. That is mathematically a
        # 1x1 conv over a (B, C, 1, 1) tensor -- i.e. a per-image linear layer over
        # channels with no spatial extent at all -- but it still dispatches through
        # cuDNN's convolution path, and cudnn.benchmark=True (SPEC 9, a free lever kept
        # on elsewhere) re-times candidate algorithms on every fresh process. For this
        # particular fp32 shape/dtype combination two algorithms benchmark close enough
        # to tie, so which one wins is scheduler-noise-dependent -- confirmed by running
        # V24 five times: PASS, PASS, FAIL, FAIL, FAIL, and confirmed fixed by forcing
        # `cudnn.benchmark=False` process-wide (4/4 identical) -- but disabling the
        # benchmark lever globally to fix one 1x1 conv would cost real throughput on
        # every other conv in the network, which is not an acceptable trade for a check
        # that exists to catch exactly this class of bug. Routing this op through
        # `F.linear` instead keeps it off cuDNN's autotuned conv path entirely (it becomes
        # a cuBLAS GEMM on a shape-keyed heuristic, not a timed benchmark search).
        #
        # IMPORTANT: this halves but does NOT eliminate V24's flake rate. Measured on this
        # dev box (RTX 4060, cudnn.benchmark=True): the PRE-FIX code (`self.conv(...)`
        # directly, still present at commit 9ee0c59) fails V24 ~50% of runs (5/10);
        # routing through `F.linear` here drops that to ~24% (5/21). The remainder is a
        # PRE-EXISTING, broader cudnn.benchmark algorithm-tie nondeterminism elsewhere in
        # the model's other (real, spatial) convolutions -- confirmed present even before
        # this V22 fix existed (the unpatched checkpoint-bearing model also flakes V24
        # under cudnn.benchmark=True) -- not something this narrow, single-module fix can
        # close. That is a separate, pre-existing robustness gap and is reported
        # separately; it is not part of the V22 fix this comment documents.
        with torch.autocast(device_type=x.device.type, enabled=False):
            pooled = x.float().mean(dim=(2, 3))                      # (B, C)
            w = F.linear(pooled, self.conv.weight[:, :, 0, 0], self.conv.bias)
            w = w[:, :, None, None]                                  # (B, C, 1, 1)
        # `w` is left in fp32: `x * w` (bf16 * fp32) promotes to fp32 by ordinary ATen
        # type-promotion rules, the same way autocast leaves F.layer_norm's fp32 output
        # to be re-cast by whatever bf16 op consumes it next. Casting `w` back to bf16
        # here before the multiply was tried and is a no-op (undoes the whole fix): the
        # 1x1 conv already only sees `bf16`-rounded values, so its own output content is
        # identical to computing everything in bf16 in the first place.
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
        film_dim: int = 0,
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

        # --- optional FiLM noise-level conditioning (Round 2 differentiator) -----
        # `film_dim == 0` (the default) means NO parameters here at all: the module holds
        # exactly the layers it held before this feature existed, so a checkpoint trained
        # without FiLM is byte-for-byte the same state_dict shape and loads unchanged
        # under V35's `strict=True`. Applied once, after norm1, on the spatial branch only
        # -- the residual stream (`x = x + y * beta`) carries the conditioned signal into
        # the channel-MLP branch too, so a second injection point is not needed.
        self.film_dim = int(film_dim)
        if self.film_dim > 0:
            self.film = nn.Linear(self.film_dim, 2 * channels)
            # Zero-init: at step 0, `scale=0, shift=0` below makes FiLM an exact identity
            # (`y * (1+0) + 0 == y`), the same "no scary surprises at init" reasoning
            # `layerscale_init` already documents for `beta`/`gamma` in this class.
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        else:
            self.film = None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if (
            not self.training
            and not torch.is_grad_enabled()
            and self.film is None
            and bool(torch.all(self.beta == 0).item())
            and bool(torch.all(self.gamma == 0).item())
        ):
            return x

        y = self.norm1(x)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).chunk(2, dim=1)          # (B, C) each
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]
            y = y * (1.0 + scale) + shift
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


class NoiseEstimator(nn.Module):
    """Small conv stack estimating a per-image noise-level embedding from the raw input.

    Feeds `NAFBlock`'s optional FiLM conditioning (SPEC F7: generalise beyond observed
    noise levels). The input `x` legitimately carries signal-dependent noise
    (docs/decisions.md D2, D12), so a few strided conv layers plus a global pool over the
    whole image is a physically reasonable estimator -- it is not told the noise
    parameters directly, it has to infer a summary of them from the pixels themselves,
    the same information a restoration network would need anyway.

    Deliberately tiny relative to the NAFSR body (a handful of conv layers at a
    downsampled resolution, not a `NAFBlock` stack) -- this is a conditioning signal, not
    a second restoration path, and this project's whole throughput story (docs/decisions.md
    D7, D21) argues against spending real compute on anything that is not the main body.
    """

    def __init__(self, in_ch: int = 1, hidden: int = 16, film_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Linear(hidden, film_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        pooled = feat.mean(dim=(2, 3))          # (B, hidden); defined for any H, W >= 1
        return self.proj(pooled)                # (B, film_dim)


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
