"""Network architectures. Public entry point: ``build_model(cfg)``.

SPEC 7.1: NAFNet-style body at LR resolution + global bilinear-upsample residual skip
+ a single PixelShuffle(2) head, single channel in and out. The plain U-Net baseline
required by the rubric (SPEC 7.2, V28) lives in this same file and is selected by
``cfg["name"]`` so that both architectures share one loader, one checkpoint contract and
one set of shape guarantees.

Contract (frozen -- `inference.py`, `train.py` and `scripts/evaluate.py` code against it):

    build_model(cfg: Mapping[str, Any]) -> torch.nn.Module

    cfg = {"name": "NAFSR" | "UNetSR", "width": int, "num_blocks": int,
           "scale": 2, "in_ch": 1, "out_ch": 1}

    forward: (B, in_ch, h, w) float32, UNCLIPPED  ->  (B, out_ch, scale*h, scale*w)

    NAFSR's `forward` additionally accepts `return_uncertainty: bool = False` (default
    False everywhere this frozen contract is exercised -- `inference.py` never passes it);
    when True AND the checkpoint was built with `uncertainty=True`, it returns
    `(tensor, log_var_tensor)` instead. This is an opt-in addition for `train.py`'s
    auxiliary loss and offline demo tooling, never the default single-tensor path above.

`cfg` is a plain mapping -- never a path, never a YAML file, never an ``argparse.Namespace``.
Reading YAML is `train.py`'s job. Every key is optional and falls back to a documented
default, because ``build_model(ckpt["config"])`` is called on configs written months apart
(V35). A full training config (with ``model:`` / ``data:`` / ``optim:`` sections) is also
accepted: the ``model`` sub-mapping is used if present.

Invariants that are load-bearing and cheap to break:

* **Size-agnostic, required size multiple 1.** Any (H, W) in, exactly (scale*H, scale*W)
  out -- odd, non-square, anything. The released dataset is uniformly 128->256
  (SPEC_ADDENDUM section 1: there are *zero* 512-GT pairs), so the 256->512 path is
  exercised only by the synthetic fixture SPEC T6 demands. That fixture is the only guard
  against silently baking in 128->256, hence nothing here may read a resolution from a
  config. `NAFSR` never downsamples, so its multiple is 1 by construction; `UNetSR` pads
  internally and crops back, so its *external* multiple is also 1.
* **No BatchNorm, no dropout, no stochastic layer.** See src/blocks.py.
* **No clipping and no input renormalisation inside the model.** Inputs legitimately span
  [-0.28, 2.16] (SPEC_ADDENDUM section 4); clipping to [0,1] happens at save time only,
  never in the model and never in the loss (SPEC 8).
* **Trained from scratch.** Nothing here loads an external checkpoint
  (docs/decisions.md D13).

Module-level imports are minimal on purpose: `inference.py` imports this module and its
import cost is ~85-95% of the scored wall-clock (docs/decisions.md D7).

Owner: model-core (see CLAUDE.md FILE OWNERSHIP MAP).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvReLU, NAFBlock, NoiseEstimator, PixelShuffleHead, bilinear_upsample
from .unrolling import UnrolledSR

__all__ = ["build_model", "NAFSR", "UNetSR", "UnrolledSR", "count_parameters", "estimate_macs"]


# --------------------------------------------------------------------------------------
# Defaults. Anything absent from a stored config resolves here, so these values are part
# of the checkpoint contract: changing one silently re-interprets old checkpoints.
# Add new keys with a default that reproduces the previous behaviour.
# --------------------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "name": "NAFSR",
    "width": 48,          # SPEC 9
    "num_blocks": 16,     # SPEC 9 (NAFSR body depth)
    "levels": 4,          # UNetSR encoder-decoder depth (SPEC 7.2 "4 levels")
    "scale": 2,           # F2: fixed x2, verified on all 3200 pairs
    "in_ch": 1,           # grayscale end to end (V32)
    "out_ch": 1,
    "dw_expand": 2,
    "ffn_expand": 2,
    "layerscale_init": 1.0,
    "padding_mode": "zeros",
    "film_dim": 0,        # 0 = FiLM noise-conditioning disabled (default, unchanged behaviour)
    "uncertainty": False, # add an optional per-pixel log-variance head (NAFSR only)
}

# Name aliases. "UNetBaseline" is accepted because early configs used it; keeping the
# alias is free and means an old checkpoint never fails to load (V35).
_ALIASES: dict[str, str] = {
    "nafsr": "NAFSR",
    "naf": "NAFSR",
    "nafnet": "NAFSR",
    "nafnetsr": "NAFSR",
    "unetsr": "UNetSR",
    "unet": "UNetSR",
    "unetbaseline": "UNetSR",
    "baselineunet": "UNetSR",
    "unrolledsr": "UnrolledSR",
    "unrolled": "UnrolledSR",
    "ista": "UnrolledSR",
    "istanet": "UnrolledSR",
    "algorithmunrolling": "UnrolledSR",
}

#: UnrolledSR-specific defaults (Round 2 Phase 4 stretch goal, docs/decisions.md D56). Kept
#: separate from _DEFAULTS above since num_steps/denoiser_blocks/share_denoiser/step_size_init
#: have no meaning for NAFSR/UNetSR and would be confusing merged into one shared dict.
_UNROLLED_DEFAULTS: dict[str, Any] = {
    "width": 32,
    "num_steps": 6,
    "denoiser_blocks": 2,
    "scale": 2,
    "in_ch": 1,
    "out_ch": 1,
    "share_denoiser": True,
    "step_size_init": 0.05,
}


class NAFSR(nn.Module):
    """NAFNet-style restoration body at LR resolution + PixelShuffle head (SPEC 7.1).

    ``x -> Conv3x3(in_ch->C) -> [NAFBlock] * N -> Conv3x3(C->C) + long skip
        -> Conv3x3(C->s^2 C) -> PixelShuffle(s) -> Conv3x3(C->out_ch)  +  bilinear(x)``

    Two design points worth defending:

    1. **Flat body, no downsampling.** With the scale factor fixed at x2, spending compute
       at LR resolution is 4x cheaper than at HR, and a flat body means the required input
       size multiple is 1 -- no padding logic, no crop-back, nothing to get wrong on an
       unexpected resolution (SPEC 7.3, SPEC_ADDENDUM section 1).
    2. **Global bilinear-upsample residual skip.** The network only has to predict the
       residual detail that decimation destroyed and the noise it must remove, rather than
       re-synthesising the image. This is also why the model is not a clean-bicubic SR
       model: the measured degradation is a *sharpening* downsample plus signal-dependent
       noise applied after decimation (docs/decisions.md D1, D2), so denoising and
       upsampling are learned jointly in one pass.

    Two optional, additive, default-off Round-2 differentiators (docs/decisions.md, Round-2
    plan): **FiLM noise-level conditioning** (`film_dim > 0`) adds a small `NoiseEstimator`
    (src/blocks.py) whose output conditions every `NAFBlock` via FiLM, targeting SPEC F7's
    "generalise beyond observed noise levels" -- and **an uncertainty head**
    (`uncertainty=True`) adds a second, LR-resolution-fed `PixelShuffleHead` predicting a
    per-pixel log-variance map. Both default OFF and add zero parameters when off, so a
    checkpoint trained before this feature existed loads unchanged under V35's
    `strict=True`. Neither changes the frozen `forward(x) -> tensor` contract used by
    `inference.py`: FiLM is computed internally from `x`, and the uncertainty map is
    returned only when the caller explicitly opts in via `return_uncertainty=True` (never
    done by `inference.py`, only by `train.py`'s optional auxiliary loss and by the
    standalone demo script that renders uncertainty maps for the deck).
    """

    def __init__(
        self,
        width: int = 48,
        num_blocks: int = 16,
        scale: int = 2,
        in_ch: int = 1,
        out_ch: int = 1,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        layerscale_init: float = 1.0,
        padding_mode: str = "zeros",
        film_dim: int = 0,
        uncertainty: bool = False,
    ) -> None:
        super().__init__()
        if width < 1 or num_blocks < 1:
            raise ValueError(f"width and num_blocks must be >= 1 (got {width}, {num_blocks})")
        self.scale = int(scale)
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.width = int(width)
        self.num_blocks = int(num_blocks)
        self.padding_mode = str(padding_mode)
        self.film_dim = int(film_dim)
        self.has_uncertainty = bool(uncertainty)

        self.stem = nn.Conv2d(
            in_ch, width, kernel_size=3, padding=1, padding_mode=self.padding_mode
        )
        # A ModuleList, not nn.Sequential: FiLM conditioning must reach every block, and
        # nn.Sequential only ever forwards a single positional argument between modules.
        # Integer-indexed submodules ("body.0.*", "body.1.*", ...) serialise identically
        # either way, so this does NOT change the state_dict layout of any existing
        # checkpoint.
        self.body = nn.ModuleList(
            [
                NAFBlock(
                    width,
                    dw_expand=dw_expand,
                    ffn_expand=ffn_expand,
                    layerscale_init=layerscale_init,
                    padding_mode=self.padding_mode,
                    film_dim=self.film_dim,
                )
                for _ in range(num_blocks)
            ]
        )
        self.body_tail = nn.Conv2d(
            width, width, kernel_size=3, padding=1, padding_mode=self.padding_mode
        )
        self.head = PixelShuffleHead(
            width, out_ch=out_ch, scale=self.scale, padding_mode=self.padding_mode
        )
        self.noise_est = (
            NoiseEstimator(in_ch=in_ch, film_dim=self.film_dim) if self.film_dim > 0 else None
        )
        self.uncertainty_head = (
            PixelShuffleHead(width, out_ch=1, scale=self.scale, padding_mode=self.padding_mode)
            if self.has_uncertainty else None
        )

    def forward(
        self, x: torch.Tensor, return_uncertainty: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        skip = bilinear_upsample(x, self.scale)
        if self.out_ch != self.in_ch:
            # Only reachable for a non 1->1 config; keeps the skip well-defined.
            skip = skip.mean(dim=1, keepdim=True).expand(-1, self.out_ch, -1, -1)
        feat = self.stem(x)
        cond = self.noise_est(x) if self.noise_est is not None else None
        y = feat
        for blk in self.body:
            y = blk(y, cond=cond)
        y = self.body_tail(y) + feat
        out = self.head(y) + skip
        if self.uncertainty_head is not None and return_uncertainty:
            # Log-variance is not a pixel value: no global skip, no [0,1] framing.
            log_var = self.uncertainty_head(y)
            return out, log_var
        return out


class UNetSR(nn.Module):
    """Plain U-Net encoder-decoder at LR resolution + PixelShuffle head (SPEC 7.2 baseline).

    Deliberately ordinary -- no NAFNet blocks, no attention, no normalisation: it is the
    learned baseline the rubric requires (V28), and it is only informative if it is a
    conventional design trained under the same budget. It keeps the two things that are
    part of the *task* rather than part of the architecture: the global bilinear skip and
    the single PixelShuffle head, so the comparison isolates the body.

    Encoder-decoder downsampling means the body needs H, W divisible by
    ``2**(levels-1)``. That is handled inside ``forward`` by reflect-padding and cropping
    the output back, so the module's external size requirement is still 1 -- an unexpected
    odd resolution produces a correct (2H, 2W) result rather than a crash. (This padding
    dance is precisely the complexity NAFSR's flat body avoids.)
    """

    def __init__(
        self,
        width: int = 32,
        levels: int = 4,
        scale: int = 2,
        in_ch: int = 1,
        out_ch: int = 1,
    ) -> None:
        super().__init__()
        if width < 1 or levels < 1:
            raise ValueError(f"width and levels must be >= 1 (got {width}, {levels})")
        self.scale = int(scale)
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.levels = int(levels)
        self.size_multiple = 2 ** (self.levels - 1)

        chans = [width * (2**i) for i in range(levels)]
        self.stem = ConvReLU(in_ch, chans[0])

        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(levels - 1):
            self.enc.append(nn.Sequential(ConvReLU(chans[i], chans[i]), ConvReLU(chans[i], chans[i])))
            self.down.append(nn.Conv2d(chans[i], chans[i + 1], kernel_size=3, stride=2, padding=1))

        self.bottleneck = nn.Sequential(
            ConvReLU(chans[-1], chans[-1]), ConvReLU(chans[-1], chans[-1])
        )

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(levels - 1)):
            self.up.append(ConvReLU(chans[i + 1], chans[i]))
            self.dec.append(
                nn.Sequential(ConvReLU(2 * chans[i], chans[i]), ConvReLU(chans[i], chans[i]))
            )

        self.body_tail = nn.Conv2d(chans[0], chans[0], kernel_size=3, padding=1)
        self.head = PixelShuffleHead(chans[0], out_ch=out_ch, scale=self.scale)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        m = self.size_multiple
        h, w = x.shape[-2:]
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            # Reflect needs pad < extent; fall back to replicate for very small inputs.
            mode = "reflect" if (ph < h and pw < w) else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return x, ph, pw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = bilinear_upsample(x, self.scale)
        if self.out_ch != self.in_ch:
            skip = skip.mean(dim=1, keepdim=True).expand(-1, self.out_ch, -1, -1)

        xp, ph, pw = self._pad(x)
        feat = self.stem(xp)

        y = feat
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.enc, self.down):
            y = enc(y)
            skips.append(y)
            y = down(y)
        y = self.bottleneck(y)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            y = up(F.interpolate(y, scale_factor=2.0, mode="nearest"))
            y = dec(torch.cat([y, s], dim=1))

        y = self.body_tail(y) + feat
        out = self.head(y)
        if ph or pw:
            s = self.scale
            out = out[..., : out.shape[-2] - ph * s, : out.shape[-1] - pw * s]
        return out + skip


def _model_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept either a bare model config or a full training config with a ``model:`` block."""
    if not isinstance(cfg, Mapping):
        raise TypeError(
            "build_model(cfg) takes a plain mapping of model hyper-parameters, not "
            f"{type(cfg).__name__}. Load YAML in train.py and pass the resulting dict; "
            "argparse.Namespace and file paths are not accepted."
        )
    inner = cfg.get("model")
    if isinstance(inner, Mapping):
        return inner
    return cfg


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """Construct a model from a config mapping. The only public entry point.

    Frozen signature -- `inference.py`, `train.py` and `scripts/evaluate.py` depend on it,
    and V35 requires ``build_model(ckpt["config"])`` to accept the stored ``state_dict``
    with ``strict=True``. Do not change it without telling the main session.

    Args:
        cfg: model hyper-parameters, or a full training config containing a ``model``
            sub-mapping. Recognised keys: ``name`` (``"NAFSR"`` | ``"UNetSR"``), ``width``,
            ``num_blocks`` (NAFSR), ``levels`` (UNetSR), ``scale``, ``in_ch``, ``out_ch``,
            ``dw_expand``, ``ffn_expand``, ``layerscale_init``, ``padding_mode``,
            ``film_dim`` and ``uncertainty`` (both NAFSR-only, default off). Every key is
            optional; missing keys fall back to ``_DEFAULTS``. Unrecognised keys are
            ignored, so a config carrying extra sections or future keys still loads.

    Returns:
        An ``nn.Module`` mapping ``(B, in_ch, H, W)`` to ``(B, out_ch, scale*H, scale*W)``
        for any ``H, W >= 1``. Output is unclipped.

    Raises:
        TypeError: `cfg` is not a mapping.
        ValueError: unknown ``name``, or a non-positive size parameter.
    """
    m = _model_section(cfg)

    def get(key: str) -> Any:
        v = m.get(key, _DEFAULTS[key])
        return _DEFAULTS[key] if v is None else v

    raw_name = str(get("name"))
    name = _ALIASES.get(raw_name.replace("_", "").replace("-", "").lower())
    if name is None:
        raise ValueError(
            f"unknown model name {raw_name!r}; expected one of "
            f"{sorted(set(_ALIASES.values()))} (aliases: {sorted(_ALIASES)})"
        )

    scale = int(get("scale"))
    in_ch = int(get("in_ch"))
    out_ch = int(get("out_ch"))

    if name == "NAFSR":
        return NAFSR(
            width=int(get("width")),
            num_blocks=int(get("num_blocks")),
            scale=scale,
            in_ch=in_ch,
            out_ch=out_ch,
            dw_expand=int(get("dw_expand")),
            ffn_expand=int(get("ffn_expand")),
            layerscale_init=float(get("layerscale_init")),
            padding_mode=str(get("padding_mode")),
            film_dim=int(get("film_dim")),
            uncertainty=bool(get("uncertainty")),
        )

    if name == "UnrolledSR":
        # Separate default table (_UNROLLED_DEFAULTS): num_steps/denoiser_blocks/
        # share_denoiser/step_size_init have no meaning for NAFSR/UNetSR, so they are not
        # merged into the shared _DEFAULTS dict `get()` above reads from.
        def get_u(key: str) -> Any:
            v = m.get(key, _UNROLLED_DEFAULTS[key])
            return _UNROLLED_DEFAULTS[key] if v is None else v

        return UnrolledSR(
            width=int(get_u("width")),
            num_steps=int(get_u("num_steps")),
            denoiser_blocks=int(get_u("denoiser_blocks")),
            scale=scale,
            in_ch=in_ch,
            out_ch=out_ch,
            share_denoiser=bool(get_u("share_denoiser")),
            step_size_init=float(get_u("step_size_init")),
        )

    # UNetSR (name == "UNetSR" -- the only remaining value _ALIASES can resolve to).
    # `num_blocks` is accepted as a synonym for `levels` only if `levels` is absent, so a
    # config written for NAFSR still builds a sane baseline.
    levels = m.get("levels", m.get("num_blocks_unet", None))
    if levels is None:
        levels = _DEFAULTS["levels"]
    return UNetSR(
        width=int(get("width")),
        levels=int(levels),
        scale=scale,
        in_ch=in_ch,
        out_ch=out_ch,
    )


# --------------------------------------------------------------------------------------
# Reporting helpers. Used by the self-test below and by scripts/benchmark_runtime.py for
# the V43 record (param count + checkpoint size in results/runtime_report.md).
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def estimate_macs(model: nn.Module, shape: tuple[int, int, int, int]) -> int:
    """Multiply-accumulate estimate for one forward pass at ``shape`` (N, C, H, W).

    Counts Conv2d and Linear only -- they are >99% of the arithmetic here. Normalisation,
    SimpleGate, the elementwise residuals and the bilinear skip are ignored, so this is a
    lower bound on total arithmetic and a good proxy for cost. FLOPs ~= 2 x MACs.
    """
    total = 0

    def hook(mod: nn.Module, inp: Any, out: Any) -> None:
        nonlocal total
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            total += out.numel() * (mod.in_channels // mod.groups) * k
        elif isinstance(mod, nn.Linear):
            total += out.numel() * mod.in_features

    handles = [
        mod.register_forward_hook(hook)
        for mod in model.modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(shape, dtype=torch.float32))
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return total


def _selftest() -> int:
    """Shape, purity and cost self-test. ``py -3.12 -m src.model``."""
    torch.manual_seed(0)
    shapes = [(1, 1, 128, 128), (1, 1, 256, 256), (1, 1, 61, 97), (2, 1, 1, 1)]
    configs = {
        "NAFSR w48 n16": {"name": "NAFSR", "width": 48, "num_blocks": 16, "scale": 2,
                          "in_ch": 1, "out_ch": 1},
        "UNetSR w32 L4": {"name": "UNetSR", "width": 32, "levels": 4, "scale": 2,
                          "in_ch": 1, "out_ch": 1},
        "NAFSR w48 n16 FiLM+uncertainty": {
            "name": "NAFSR", "width": 48, "num_blocks": 16, "scale": 2,
            "in_ch": 1, "out_ch": 1, "film_dim": 16, "uncertainty": True,
        },
        "UnrolledSR w32 steps6": {
            "name": "UnrolledSR", "width": 32, "num_steps": 6, "denoiser_blocks": 2,
            "scale": 2, "in_ch": 1, "out_ch": 1,
        },
    }
    bad = 0
    for label, cfg in configs.items():
        model = build_model(cfg).eval()
        params = count_parameters(model)
        macs = estimate_macs(model, (1, 1, 128, 128))
        print(f"{label}: params={params:,} ({params / 1e6:.3f} M)  "
              f"MACs@128x128={macs / 1e9:.3f} G  FLOPs~={2 * macs / 1e9:.3f} G")
        for name_, mod in model.named_modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.SyncBatchNorm, nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                                nn.AlphaDropout, nn.FeatureAlphaDropout)):
                print(f"  FAIL forbidden layer {name_}: {type(mod).__name__}")
                bad += 1
        with torch.no_grad():
            for s in shapes:
                x = torch.randn(s) * 0.5 + 0.5
                x[0, 0, 0, 0] = 2.16
                y = model(x)
                want = (s[0], 1, s[2] * 2, s[3] * 2)
                ok = tuple(y.shape) == want and torch.isfinite(y).all().item()
                bad += 0 if ok else 1
                print(f"  {'ok ' if ok else 'FAIL'} {tuple(s)} -> {tuple(y.shape)} "
                      f"(want {want}) range=[{y.min():.3f}, {y.max():.3f}]")
            # Determinism in eval(): same input twice must be bit-identical (V24).
            x = torch.randn(1, 1, 40, 40)
            same = torch.equal(model(x), model(x))
        print(f"  {'ok ' if same else 'FAIL'} bit-identical on repeat: {same}")
        bad += 0 if same else 1

        if getattr(model, "has_uncertainty", False):
            with torch.no_grad():
                x = torch.randn(1, 1, 64, 64)
                out_only = model(x)
                out_pair, log_var = model(x, return_uncertainty=True)
                shapes_ok = (
                    tuple(out_only.shape) == (1, 1, 128, 128)
                    and tuple(log_var.shape) == (1, 1, 128, 128)
                    and torch.equal(out_only, out_pair)  # same restoration either way
                    and torch.isfinite(log_var).all().item()
                )
            print(f"  {'ok ' if shapes_ok else 'FAIL'} return_uncertainty=True gives "
                  f"(restoration, log_var) of matching shape, restoration unchanged")
            bad += 0 if shapes_ok else 1

        if getattr(model, "film_dim", 0) > 0:
            # Zero-init sanity control: at construction, FiLM's Linear is zero-initialised,
            # so conditioning must be an exact identity -- output must match a twin model
            # with film_dim=0 given the SAME body/head weights. Confirms "no scary
            # surprises at init" is true, not just asserted.
            plain_cfg = dict(cfg)
            plain_cfg["film_dim"] = 0
            plain_cfg["uncertainty"] = False
            twin = build_model(plain_cfg).eval()
            twin.load_state_dict(model.state_dict(), strict=False)
            with torch.no_grad():
                x = torch.randn(1, 1, 48, 48)
                film_out = model(x)
                twin_out = twin(x)
            film_identity_ok = torch.equal(film_out, twin_out)
            print(f"  {'ok ' if film_identity_ok else 'FAIL'} FiLM zero-init is an exact "
                  f"identity vs an un-conditioned twin")
            bad += 0 if film_identity_ok else 1
    print("SELFTEST", "PASS" if bad == 0 else f"FAIL ({bad} problems)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
