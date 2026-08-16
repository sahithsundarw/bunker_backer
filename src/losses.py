"""Loss functions: Charbonnier, SSIM/MS-SSIM, FFT, optional LPIPS (SPEC 8).

L_total = 1.0*Charbonnier + 0.15*(1-MS_SSIM) + 0.05*L_fft + 0.02*L_lpips

Compute the loss on the UNCLIPPED network output; clip only at save time. Clipping inside
the loss zero-grads saturated pixels (SPEC 8, SPEC 18 pitfall 3). Nothing in this module
clamps ``pred``.

MS-SSIM needs >=161 px across its 5 scales. At 128x128 GT patches it will throw -- use
single-scale SSIM at 128 or raise the patch size to 192+. ``structural_loss`` picks
automatically from the tensor's own spatial size, so a 128 patch silently gets single-scale
SSIM instead of a RuntimeError. This is a measured dead end, not a guess: see the
"Do NOT retry" list in docs/STATE.md.

No adversarial loss (SPEC 7.2): hallucinated structure is the worst failure in inspection.

Everything here is computed in float32 even under bf16/fp16 autocast: torch.fft has no
reduced-precision kernel, and the pytorch-msssim gaussian window is float32.

Owner: loss-metrics.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

__all__ = [
    "CHARBONNIER_EPS",
    "MS_SSIM_MIN_SIDE",
    "LOG_VAR_CLAMP",
    "DEFAULT_WEIGHTS",
    "charbonnier",
    "fft_loss",
    "ssim_loss",
    "ms_ssim_loss",
    "structural_loss",
    "supports_ms_ssim",
    "lpips_loss",
    "heteroscedastic_nll_loss",
    "RestorationLoss",
    "build_loss",
]

#: SPEC 8: Charbonnier epsilon.
CHARBONNIER_EPS: float = 1e-3

#: pytorch-msssim's 5-scale pyramid with win_size=11 needs min(H,W) > (11-1)*2**4 = 160.
MS_SSIM_MIN_SIDE: int = 161

#: Clamp range for the predicted log-variance before `exp()` -- an unclamped log-variance
#: can drift to a value whose exp() overflows fp32 (or, at the other end, underflows to a
#: near-zero denominator that blows up the NLL), especially early in training before the
#: uncertainty head has learned anything. This is a numerical-safety measure, not a
#: modelling choice: +/-10 covers a variance range of ~4.5e-5 to ~2.2e4, far wider than any
#: plausible pixel-intensity variance in a [0,1]-ish signal.
LOG_VAR_CLAMP: tuple[float, float] = (-10.0, 10.0)

#: SPEC 8 blend. Keys are the component names used in the returned log dict. `uncertainty`
#: defaults to 0.0 (off) -- Round-2 differentiator, opt-in only (docs/decisions.md).
DEFAULT_WEIGHTS: dict[str, float] = {
    "charbonnier": 1.0,
    "structural": 0.15,
    "fft": 0.05,
    "lpips": 0.02,
    "uncertainty": 0.0,
}

#: LPIPS turns on only after this fraction of training (SPEC 8: "~50%").
DEFAULT_LPIPS_START_FRAC: float = 0.5


def _as_nchw(t: torch.Tensor, label: str) -> torch.Tensor:
    """Accept (N,C,H,W), (N,H,W) or (H,W); return (N,C,H,W) float32."""
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[:, None]
    elif t.dim() != 4:
        raise ValueError(f"{label}: expected 2/3/4-D tensor, got shape {tuple(t.shape)}")
    return t.float()


def _pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = _as_nchw(pred, "pred")
    t = _as_nchw(target, "target")
    if p.shape != t.shape:
        raise ValueError(f"shape mismatch: pred {tuple(p.shape)} vs target {tuple(t.shape)}")
    return p, t


# --------------------------------------------------------------------------------------
# 1. Charbonnier -- the PSNR driver
# --------------------------------------------------------------------------------------
def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = CHARBONNIER_EPS) -> torch.Tensor:
    """sqrt((pred-target)^2 + eps^2). Smooth L1; drives PSNR."""
    diff = pred.float() - target.float()
    return torch.sqrt(diff * diff + float(eps) ** 2).mean()


# --------------------------------------------------------------------------------------
# 2. Structural -- single-scale SSIM at 128, MS-SSIM only when the patch is big enough
# --------------------------------------------------------------------------------------
def _msssim_lib() -> Any:
    try:
        import pytorch_msssim
    except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
        raise RuntimeError(
            "pytorch-msssim is required for the structural loss term; "
            f"install it (pip install pytorch-msssim). Original error: {exc}"
        ) from exc
    return pytorch_msssim


def supports_ms_ssim(height: int, width: int) -> bool:
    """True when a 5-scale MS-SSIM pyramid is legal for this spatial size (>=161 px)."""
    return min(int(height), int(width)) >= MS_SSIM_MIN_SIDE


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """1 - single-scale SSIM. The correct choice at 128x128 GT patches."""
    p, t = _pair(pred, target)
    return 1.0 - _msssim_lib().ssim(p, t, data_range=float(data_range), size_average=True)


def ms_ssim_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """1 - MS-SSIM. Requires min(H,W) >= 161; raises early with a readable message."""
    p, t = _pair(pred, target)
    h, w = int(p.shape[-2]), int(p.shape[-1])
    if not supports_ms_ssim(h, w):
        raise ValueError(
            f"MS-SSIM needs min(H,W) >= {MS_SSIM_MIN_SIDE} for its 5 scales, got {h}x{w}. "
            "Use ssim_loss() at 128 px patches or raise the patch size to 192+ (SPEC 8)."
        )
    return 1.0 - _msssim_lib().ms_ssim(p, t, data_range=float(data_range), size_average=True)


def structural_loss(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
                    allow_ms_ssim: bool = True) -> tuple[torch.Tensor, str]:
    """SPEC 8 term 2, auto-selecting the variant that is legal for this patch size.

    Returns ``(loss, kind)`` where ``kind`` is ``'ms_ssim'`` or ``'ssim'`` so the trainer can
    log which one was actually used -- the two are NOT numerically comparable.
    """
    p, t = _pair(pred, target)
    h, w = int(p.shape[-2]), int(p.shape[-1])
    if allow_ms_ssim and supports_ms_ssim(h, w):
        return ms_ssim_loss(p, t, data_range=data_range), "ms_ssim"
    return ssim_loss(p, t, data_range=data_range), "ssim"


# --------------------------------------------------------------------------------------
# 3. FFT magnitude L1 -- recovers high frequency without hallucinating it
# --------------------------------------------------------------------------------------
def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between FFT magnitudes. Penalises missing high-frequency content."""
    p, t = _pair(pred, target)
    fp = torch.fft.fft2(p, norm="ortho")
    ft = torch.fft.fft2(t, norm="ortho")
    return torch.abs(torch.abs(fp) - torch.abs(ft)).mean()


# --------------------------------------------------------------------------------------
# 4. LPIPS -- small auxiliary term, enabled only after warmup
# --------------------------------------------------------------------------------------
_LPIPS_NET: Any = None
_LPIPS_KEY: tuple[str, str] | None = None


def lpips_net(device: torch.device | str, net: str = "alex") -> Any:
    """Lazily build ONE frozen LPIPS network and cache it module-globally."""
    global _LPIPS_NET, _LPIPS_KEY
    try:
        import lpips as _lpips_lib
    except ImportError as exc:
        raise RuntimeError(
            "the lpips package is required for the LPIPS loss term; either install it or "
            f"run with use_lpips=False (SPEC 8 allows dropping it). Original error: {exc}"
        ) from exc

    key = (str(device), str(net))
    if _LPIPS_NET is None or _LPIPS_KEY != key:
        _LPIPS_NET = _lpips_lib.LPIPS(net=str(net)).to(device).eval()
        for prm in _LPIPS_NET.parameters():
            prm.requires_grad_(False)   # frozen feature extractor; grads still reach pred
        _LPIPS_KEY = key
    return _LPIPS_NET


def lpips_loss(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> torch.Tensor:
    """Perceptual distance. Grayscale -> repeat(1,3,1,1); [0,1] -> [-1,1].

    ``pred`` is deliberately NOT clamped: gradients must keep flowing through saturated
    pixels (SPEC 8). LPIPS tolerates mildly out-of-range inputs.
    """
    p, t = _pair(pred, target)
    model = lpips_net(p.device, net=net)
    p3 = p.repeat(1, 3, 1, 1) * 2.0 - 1.0
    t3 = t.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return model(p3, t3).mean()


# --------------------------------------------------------------------------------------
# 5. Heteroscedastic NLL -- the uncertainty head's supervision (Round 2 differentiator)
# --------------------------------------------------------------------------------------
def heteroscedastic_nll_loss(pred: torch.Tensor, log_var: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood with a learned per-pixel log-variance.

    ``0.5 * exp(-log_var) * (pred-target)^2 + 0.5 * log_var``, mean-reduced. This is what
    lets the uncertainty head's output mean something (a calibrated confidence estimate,
    not a hand-tuned "distance from clean-looking"): the network is only rewarded for a
    high-uncertainty prediction on a pixel where the restoration is *actually* far from the
    target, and is penalised for claiming high confidence on a pixel it gets wrong.

    ``log_var`` is clamped to ``LOG_VAR_CLAMP`` before use -- see that constant's docstring
    for why an unclamped exp() is a real numerical hazard here, not a hypothetical one.
    """
    p, t = _pair(pred, target)
    lv = _as_nchw(log_var, "log_var")
    if lv.shape != p.shape:
        raise ValueError(f"log_var shape {tuple(lv.shape)} != pred shape {tuple(p.shape)}")
    lv = lv.clamp(*LOG_VAR_CLAMP)
    diff2 = (p - t) ** 2
    return (0.5 * torch.exp(-lv) * diff2 + 0.5 * lv).mean()


# --------------------------------------------------------------------------------------
# The composite loss
# --------------------------------------------------------------------------------------
class RestorationLoss(nn.Module):
    """SPEC 8 composite loss.

    ``forward(pred, target, progress)`` returns ``(total, logs)``:

    * ``total`` -- a finite scalar tensor carrying grad.
    * ``logs``  -- floats for the CSV: one entry per component (unweighted value), plus
      ``total``, ``structural_kind`` and ``lpips_active``.

    ``progress`` is the fraction of training completed, in [0,1]. The LPIPS term is skipped
    entirely (no forward pass, no cost) until ``progress >= lpips_start_frac``.

    Pass the UNCLIPPED network output. This module never clamps.
    """

    def __init__(self, weights: Mapping[str, float] | None = None, *,
                 use_lpips: bool = False,
                 lpips_start_frac: float = DEFAULT_LPIPS_START_FRAC,
                 lpips_net_name: str = "alex",
                 allow_ms_ssim: bool = True,
                 eps: float = CHARBONNIER_EPS,
                 data_range: float = 1.0) -> None:
        super().__init__()
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(DEFAULT_WEIGHTS)
            if unknown:
                raise ValueError(f"unknown loss weight key(s): {sorted(unknown)}")
            w.update({k: float(v) for k, v in weights.items()})
        self.weights = w
        self.use_lpips = bool(use_lpips)
        self.lpips_start_frac = float(lpips_start_frac)
        self.lpips_net_name = str(lpips_net_name)
        self.allow_ms_ssim = bool(allow_ms_ssim)
        self.eps = float(eps)
        self.data_range = float(data_range)

    def lpips_enabled(self, progress: float) -> bool:
        """LPIPS is on only if requested, weighted, and past the warmup fraction."""
        return (self.use_lpips and self.weights.get("lpips", 0.0) != 0.0
                and float(progress) >= self.lpips_start_frac)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                progress: float = 0.0,
                log_var: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        p, t = _pair(pred, target)

        terms: dict[str, torch.Tensor] = {}
        terms["charbonnier"] = charbonnier(p, t, eps=self.eps)
        struct, kind = structural_loss(p, t, data_range=self.data_range,
                                       allow_ms_ssim=self.allow_ms_ssim)
        terms["structural"] = struct
        terms["fft"] = fft_loss(p, t)

        lp_on = self.lpips_enabled(progress)
        if lp_on:
            terms["lpips"] = lpips_loss(p, t, net=self.lpips_net_name)

        # Uncertainty term: only computed when the caller actually has a log_var to supply
        # (i.e. the model was built with uncertainty=True and the trainer requested it via
        # return_uncertainty=True) AND it is weighted. `log_var is None` is the default for
        # every model/config that predates this feature -- zero cost, zero behaviour
        # change, same as the LPIPS gate above.
        if log_var is not None and self.weights.get("uncertainty", 0.0) != 0.0:
            terms["uncertainty"] = heteroscedastic_nll_loss(p, log_var, t)

        total = p.new_zeros(())
        for name, value in terms.items():
            total = total + self.weights.get(name, 0.0) * value

        logs: dict[str, Any] = {k: float(v.detach()) for k, v in terms.items()}
        logs["total"] = float(total.detach())
        logs["structural_kind"] = kind
        logs["lpips_active"] = bool(lp_on)
        return total, logs

    def extra_repr(self) -> str:
        return (f"weights={self.weights}, use_lpips={self.use_lpips}, "
                f"lpips_start_frac={self.lpips_start_frac}, allow_ms_ssim={self.allow_ms_ssim}")


def build_loss(cfg: Mapping[str, Any] | None = None) -> RestorationLoss:
    """Construct the loss from a config mapping, e.g. ``cfg['loss']`` in a YAML.

    Recognised keys: ``weights`` (mapping), ``use_lpips``, ``lpips_start_frac``,
    ``lpips_net``, ``allow_ms_ssim``, ``charbonnier_eps``, ``data_range``.
    """
    cfg = dict(cfg or {})
    return RestorationLoss(
        weights=cfg.get("weights"),
        use_lpips=bool(cfg.get("use_lpips", False)),
        lpips_start_frac=float(cfg.get("lpips_start_frac", DEFAULT_LPIPS_START_FRAC)),
        lpips_net_name=str(cfg.get("lpips_net", "alex")),
        allow_ms_ssim=bool(cfg.get("allow_ms_ssim", True)),
        eps=float(cfg.get("charbonnier_eps", CHARBONNIER_EPS)),
        data_range=float(cfg.get("data_range", 1.0)),
    )


# --------------------------------------------------------------------------------------
# Self-test: every term must return a FINITE SCALAR on a random batch, and grads must flow
# through pixels outside [0,1] (the whole point of not clipping inside the loss).
# --------------------------------------------------------------------------------------
def _self_test(seed: int = 20260815) -> int:
    torch.manual_seed(seed)
    ok = True

    def show(label: str, value: torch.Tensor) -> None:
        nonlocal ok
        finite = bool(torch.isfinite(value).all()) and value.dim() == 0
        ok = ok and finite
        print("  %-34s %12.6f  scalar=%s finite=%s"
              % (label, float(value.detach()), value.dim() == 0, bool(torch.isfinite(value).all())))

    for side in (128, 192):
        # Deliberately out of [0,1] on both ends, like a real unclipped network output.
        pred = (torch.rand(2, 1, side, side) * 1.6 - 0.3).requires_grad_(True)
        target = torch.rand(2, 1, side, side)
        pd = pred.detach()
        print(f"random batch (2,1,{side},{side}), pred range "
              f"[{float(pd.min()):.3f}, {float(pd.max()):.3f}]:")
        show("charbonnier", charbonnier(pred, target))
        s, kind = structural_loss(pred, target)
        show(f"structural ({kind})", s)
        show("fft", fft_loss(pred, target))
        print("    supports_ms_ssim(%d,%d) = %s" % (side, side, supports_ms_ssim(side, side)))

    # Heteroscedastic NLL: finite for a real log_var, and a confident-but-wrong prediction
    # must score worse than an equally-wrong prediction that honestly reports high variance.
    pred_u = torch.rand(2, 1, 64, 64)
    target_u = torch.rand(2, 1, 64, 64)
    diff = (pred_u - target_u) ** 2
    confident = torch.full_like(diff, -6.0)   # claims low variance everywhere
    honest = torch.full_like(diff, 6.0)       # claims high variance everywhere
    nll_confident = heteroscedastic_nll_loss(pred_u, confident, target_u)
    nll_honest = heteroscedastic_nll_loss(pred_u, honest, target_u)
    show("heteroscedastic_nll (confident)", nll_confident)
    show("heteroscedastic_nll (honest)", nll_honest)
    ok = ok and bool(nll_confident > nll_honest)  # over-confidence on real error is punished
    print("  confident > honest (over-confidence punished): %s" % (nll_confident > nll_honest))

    # RestorationLoss with log_var wired through: off by default (weight 0.0), on when
    # both a nonzero weight AND a log_var are supplied.
    crit_unc = build_loss({"weights": {"uncertainty": 0.01}})
    t_no_lv, l_no_lv = crit_unc(pred_u, target_u, log_var=None)
    assert "uncertainty" not in l_no_lv, "uncertainty term must be absent without log_var"
    t_lv, l_lv = crit_unc(pred_u, target_u, log_var=confident)
    assert "uncertainty" in l_lv, "uncertainty term must appear once log_var is supplied"
    print("  logs (with log_var): uncertainty=%.6f" % l_lv["uncertainty"])

    # MS-SSIM at 128 must raise, not silently produce garbage.
    try:
        ms_ssim_loss(torch.rand(1, 1, 128, 128), torch.rand(1, 1, 128, 128))
        print("  FAIL: ms_ssim_loss accepted a 128px patch")
        ok = False
    except ValueError as exc:
        print("  ms_ssim_loss(128) correctly refused: %s" % str(exc).split(".")[0])

    # Composite, without LPIPS (must work with no lpips package involved at all).
    crit = build_loss({"use_lpips": False})
    pred = (torch.rand(2, 1, 128, 128) * 1.6 - 0.3).requires_grad_(True)
    target = torch.rand(2, 1, 128, 128)
    total, logs = crit(pred, target, progress=0.9)
    show("composite total (no lpips)", total)
    total.backward()
    grad = pred.grad
    assert grad is not None
    sat = pred.detach() > 1.0
    frac_sat_with_grad = float((grad[sat] != 0).float().mean()) if bool(sat.any()) else 1.0
    print("  logs: %s" % logs)
    print("  grad finite=%s  fraction of >1.0 pixels with nonzero grad=%.4f"
          % (bool(torch.isfinite(grad).all()), frac_sat_with_grad))
    ok = ok and bool(torch.isfinite(grad).all()) and frac_sat_with_grad > 0.99

    # LPIPS gating: off before the warmup fraction, on after.
    crit_lp = build_loss({"use_lpips": True, "lpips_start_frac": 0.5})
    assert not crit_lp.lpips_enabled(0.49)
    assert crit_lp.lpips_enabled(0.50)
    t0, l0 = crit_lp(pred.detach(), target, progress=0.1)
    print("  progress=0.1 -> lpips_active=%s" % l0["lpips_active"])
    try:
        pred_lp = pred.detach().clone().requires_grad_(True)
        t1, l1 = crit_lp(pred_lp, target, progress=0.9)
        show("composite total (with lpips)", t1)
        t1.backward()
        print("  lpips term = %.6f, grad finite=%s"
              % (l1["lpips"], bool(torch.isfinite(pred_lp.grad).all())))
        ok = ok and bool(torch.isfinite(pred_lp.grad).all())
    except RuntimeError as exc:
        print("  LPIPS term unavailable in this environment: %s" % exc)
        print("  (training still works with use_lpips=False -- SPEC 8 permits dropping it)")

    print("\nlosses self-test %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
