"""Metric implementations with PINNED settings (SPEC 10, asserted by V31).

    psnr : data_range=1.0
    ssim : data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
    lpips: net='alex', grayscale -> repeat(1,3,1,1), scaled [0,1] -> [-1,1]

Deviating makes our numbers incomparable. State these settings in the deck.

V30: score the RELOADED on-disk artifacts, not in-memory tensors.

Predictions must ALREADY be clipped to [0,1] when they arrive here -- clip only, never
renormalise (docs/decisions.md D3: per-image min-max renorm costs -4.66 dB PSNR and loses
191/200 images). ``score_pair`` refuses out-of-range predictions by default so that an
unclipped artifact on disk fails loudly instead of quietly scoring low.

Reported dispersion is the population standard deviation (``np.std``, ddof=0), matching the
anchor numbers in docs/decisions.md D3.

This module may import skimage/lpips freely -- it is not inference.py and not inside the
measured window. ``lpips`` is imported lazily so that PSNR/SSIM work without it.

Owner: loss-metrics.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

__all__ = [
    "METRIC_NAMES",
    "METRIC_SETTINGS",
    "LOWER_IS_BETTER",
    "PAIRED_T_CRIT",
    "PAIRED_MIN_N",
    "psnr",
    "ssim",
    "lpips_score",
    "score_pair",
    "aggregate",
    "format_mean_std",
    "clip_prediction",
    "check_clipped",
    "compare",
    "paired_verdict",
    "paired_compare",
]

#: Canonical metric order for every table and report we emit.
METRIC_NAMES: tuple[str, ...] = ("psnr", "ssim", "lpips")

#: LPIPS is a distance: lower is better. PSNR/SSIM: higher is better.
LOWER_IS_BETTER: frozenset[str] = frozenset({"lpips"})

#: Deck-ready statement of the pinned settings. Quote this verbatim on the metrics slide.
METRIC_SETTINGS: dict[str, str] = {
    "psnr": "skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=1.0)",
    "ssim": ("skimage.metrics.structural_similarity(gt, pred, data_range=1.0, "
             "gaussian_weights=True, sigma=1.5, use_sample_covariance=False)"),
    "lpips": ("lpips.LPIPS(net='alex').eval(); grayscale -> repeat(1,3,1,1); "
              "[0,1] -> [-1,1]; torch.no_grad()"),
    "dispersion": "population standard deviation, np.std(..., ddof=0)",
    "artifact": "metrics are computed on the reloaded on-disk .npy, not an in-memory tensor",
    "postprocess": "np.clip(pred, 0.0, 1.0) only -- never per-image renormalisation",
}

_TOL = 1e-6


# --------------------------------------------------------------------------------------
# Input hygiene
# --------------------------------------------------------------------------------------
def _as_hw(a: np.ndarray, label: str) -> np.ndarray:
    """Return ``a`` as a contiguous 2-D float array, dropping a singleton channel axis."""
    arr = np.asarray(a)
    if arr.ndim == 3:
        squeezable = [i for i, s in enumerate(arr.shape) if s == 1]
        if len(squeezable) != 1:
            raise ValueError(f"{label}: expected HxW (or a singleton channel axis), got {arr.shape}")
        arr = np.squeeze(arr, axis=squeezable[0])
    if arr.ndim != 2:
        raise ValueError(f"{label}: expected a 2-D grayscale array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"{label}: expected a float array, got dtype {arr.dtype}")
    if not np.all(np.isfinite(arr)):
        n = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{label}: contains {n} non-finite value(s)")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _pair(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = _as_hw(pred, "pred")
    g = _as_hw(gt, "gt")
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    return p, g


def clip_prediction(pred: np.ndarray) -> np.ndarray:
    """The one and only permitted post-processing step (docs/decisions.md D3).

    Clip to [0,1] and cast to float32. Never renormalise: per-image min-max renorm was
    measured at -4.66 dB PSNR because 95.5% of predictions overshoot 1.0, so the renorm
    divides by an outlier-driven range.
    """
    return np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)


def check_clipped(pred: np.ndarray, tol: float = _TOL) -> None:
    """Raise if ``pred`` is not already inside [0,1]; scoring an unclipped file is a bug."""
    arr = np.asarray(pred)
    lo, hi = float(arr.min()), float(arr.max())
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(
            f"prediction is not clipped: range [{lo:.6f}, {hi:.6f}] escapes [0,1]. "
            "Clip with np.clip(pred, 0.0, 1.0) at save time (docs/io_contract.md)."
        )


# --------------------------------------------------------------------------------------
# The pinned metrics (SPEC 10). Do not change these call sites -- V31 inspects them.
# --------------------------------------------------------------------------------------
def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: float32 HxW in [0,1]; pred ALREADY clipped."""
    p, g = _pair(pred, gt)
    return float(sk_psnr(g, p, data_range=1.0))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Wang et al. (2004) reference settings -- see module docstring."""
    p, g = _pair(pred, gt)
    return float(sk_ssim(g, p, data_range=1.0,
                         gaussian_weights=True, sigma=1.5,
                         use_sample_covariance=False))


_LPIPS_MODEL: Any = None
_LPIPS_DEVICE: str | None = None


def resolve_device(device: str = "cuda") -> str:
    """Fall back to CPU when CUDA was requested but is unavailable."""
    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def lpips_model(device: str = "cuda") -> Any:
    """Lazily build ONE AlexNet LPIPS model per device and cache it module-globally."""
    global _LPIPS_MODEL, _LPIPS_DEVICE
    import lpips as _lpips_lib

    dev = resolve_device(device)
    if _LPIPS_MODEL is None or _LPIPS_DEVICE != dev:
        # AlexNet is the standard reporting backbone (SPEC 10).
        _LPIPS_MODEL = _lpips_lib.LPIPS(net='alex').to(dev).eval()
        for prm in _LPIPS_MODEL.parameters():
            prm.requires_grad_(False)
        _LPIPS_DEVICE = dev
    return _LPIPS_MODEL


def lpips_score(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> float:
    """AlexNet backbone, grayscale replicated to 3 channels, scaled to [-1,1]."""
    import torch

    p, g = _pair(pred, gt)
    dev = resolve_device(device)
    model = lpips_model(dev)

    def prep(a: np.ndarray) -> "torch.Tensor":
        t = torch.from_numpy(a)[None, None].to(dev)   # 1,1,H,W
        t = t.repeat(1, 3, 1, 1)                      # grayscale -> 3ch
        return t * 2.0 - 1.0                          # [0,1] -> [-1,1]

    with torch.no_grad():
        return float(model(prep(p), prep(g)).item())


# --------------------------------------------------------------------------------------
# Aggregation helpers
# --------------------------------------------------------------------------------------
def score_pair(pred: np.ndarray, gt: np.ndarray, *, with_lpips: bool = True,
               device: str = "cuda", require_clipped: bool = True) -> dict[str, float]:
    """Score one reloaded prediction against its GT. Returns {'psnr','ssim','lpips'}."""
    p, g = _pair(pred, gt)
    if require_clipped:
        check_clipped(p)
    out: dict[str, float] = {"psnr": psnr(p, g), "ssim": ssim(p, g)}
    if with_lpips:
        out["lpips"] = lpips_score(p, g, device=device)
    return out


def aggregate(rows: Iterable[Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean / population std / min / max / n per metric over a list of per-image dicts."""
    rows = list(rows)
    if not rows:
        raise ValueError("aggregate: no per-image scores to aggregate")
    out: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        vals = [float(r[name]) for r in rows if name in r and np.isfinite(float(r[name]))]
        if not vals:
            continue
        a = np.asarray(vals, dtype=np.float64)
        out[name] = {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=0)),   # population sd, matches docs/decisions.md D3
            "min": float(a.min()),
            "max": float(a.max()),
            "n": int(a.size),
        }
    return out


def format_mean_std(stat: Mapping[str, float] | None, digits: int = 4) -> str:
    """'23.4247 +/- 2.8319' -- the required mean +/- std reporting form (SPEC 10, V27)."""
    if not stat:
        return "n/a"
    return f"{stat['mean']:.{digits}f} +/- {stat['std']:.{digits}f}"


def compare(candidate: Mapping[str, Mapping[str, float]],
            reference: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, Any]]:
    """Per-metric delta of candidate vs reference, sign-corrected for LPIPS.

    ``better`` is True when the candidate wins on that metric. Used for V27 (beat bicubic
    on PSNR and SSIM with lower LPIPS) and V28 (beat the U-Net on >= 2 of 3).
    """
    out: dict[str, dict[str, Any]] = {}
    for name in METRIC_NAMES:
        if name not in candidate or name not in reference:
            continue
        delta = candidate[name]["mean"] - reference[name]["mean"]
        better = delta < 0.0 if name in LOWER_IS_BETTER else delta > 0.0
        out[name] = {"delta": float(delta), "better": bool(better)}
    return out


#: Two-tailed 95% critical value for the large-sample paired t-test below. Mirrors
#: ``scripts/verify_all.py``'s V28 statistic exactly -- kept here too because V28 is a
#: per-image PAIRED test, not a comparison of two independent means, and a scoring script
#: must not substitute the latter for the former (docs/decisions.md D31).
PAIRED_T_CRIT = 1.96
#: Minimum number of paired observations before a per-image comparison is trusted at all.
PAIRED_MIN_N = 30


def paired_verdict(cand_per_image: Mapping[str, Mapping[str, float]],
                    ref_per_image: Mapping[str, Mapping[str, float]],
                    key: str, higher_is_better: bool) -> dict[str, Any] | None:
    """Paired per-image significance test for one metric, keyed by filename.

    Both directories must score the SAME images for a mean-delta comparison to be
    meaningful; this computes the per-image differences on files common to both, then a
    paired t-statistic over those differences. A win requires BOTH the correct sign AND
    ``|t| >= PAIRED_T_CRIT``; anything else -- including a positive mean with no
    significance -- is a tie, never a win. Returns ``None`` if fewer than ``PAIRED_MIN_N``
    pairs are available (anti-vacuity: too few pairs to conclude anything).

    This is the exact test ``scripts/verify_all.py``'s ``check_V28`` runs. Counting a mean
    difference of e.g. +0.000135 SSIM as a "win" when the candidate only wins 172/400 images
    is precisely the self-serving reading this test exists to prevent (docs/decisions.md D31).
    """
    common = sorted(set(cand_per_image) & set(ref_per_image))
    diffs = [cand_per_image[f][key] - ref_per_image[f][key] for f in common
             if key in cand_per_image[f] and key in ref_per_image[f]]
    n = len(diffs)
    if n < PAIRED_MIN_N:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sem = (var / n) ** 0.5
    t = (mean / sem) if sem > 0 else 0.0
    better = (mean > 0) if higher_is_better else (mean < 0)
    sig = abs(t) >= PAIRED_T_CRIT
    n_better = sum(1 for d in diffs if ((d > 0) if higher_is_better else (d < 0)))
    return {"test": "paired", "n": n, "mean_diff": mean, "sem": sem, "t": t,
            "images_better": n_better, "significant": sig,
            "win": bool(better and sig), "loss": bool((not better) and sig)}


def paired_compare(cand_per_image: Iterable[Mapping[str, Any]] | None,
                    ref_per_image: Iterable[Mapping[str, Any]] | None,
                    metric_names: Sequence[str] = METRIC_NAMES) -> dict[str, dict[str, Any]]:
    """Per-metric paired win/loss/tie verdicts, keyed by metric name.

    ``cand_per_image`` / ``ref_per_image`` are lists of per-image score dicts (each with a
    ``"file"`` key plus metric values), the same shape as ``score_dir``'s ``per_image``
    output. Returns ``{}`` if either side is empty/missing or files don't overlap -- the
    caller must NOT fall back to an unpaired mean-delta verdict in that case, only report
    that no paired verdict is available (docs/decisions.md D31).
    """
    if not cand_per_image or not ref_per_image:
        return {}
    cand_by_file = {str(r["file"]): {k: float(v) for k, v in r.items()
                                     if k != "file" and isinstance(v, (int, float))}
                    for r in cand_per_image if isinstance(r, Mapping) and r.get("file")}
    ref_by_file = {str(r["file"]): {k: float(v) for k, v in r.items()
                                    if k != "file" and isinstance(v, (int, float))}
                   for r in ref_per_image if isinstance(r, Mapping) and r.get("file")}
    out: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        v = paired_verdict(cand_by_file, ref_by_file, name,
                           higher_is_better=(name not in LOWER_IS_BETTER))
        if v is not None:
            out[name] = v
    return out


def settings_block() -> str:
    """Deck/report-ready text of the pinned settings."""
    lines = ["Pinned metric settings (SPEC 10, asserted by V31):"]
    for key in ("psnr", "ssim", "lpips", "dispersion", "artifact", "postprocess"):
        lines.append(f"  {key:12s} {METRIC_SETTINGS[key]}")
    return "\n".join(lines)


def _self_test(seed: int = 20260815) -> int:
    """Sanity: identical arrays score perfectly; noise scores worse. Finite everywhere."""
    rng = np.random.default_rng(seed)
    gt = rng.random((64, 64), dtype=np.float32)
    same = gt.copy()
    noisy = clip_prediction(gt + rng.normal(0, 0.05, gt.shape).astype(np.float32))

    assert np.isinf(psnr(same, gt)) or psnr(same, gt) > 90.0
    assert abs(ssim(same, gt) - 1.0) < 1e-6
    s_noisy = score_pair(noisy, gt, with_lpips=True, device="cpu")
    for k, v in s_noisy.items():
        assert np.isfinite(v), (k, v)
    assert s_noisy["psnr"] < 60.0 and s_noisy["ssim"] < 1.0

    agg = aggregate([s_noisy, s_noisy])
    assert agg["psnr"]["std"] == 0.0 and agg["psnr"]["n"] == 2

    try:
        score_pair(gt + 2.0, gt, with_lpips=False)
    except ValueError:
        pass
    else:
        raise AssertionError("unclipped prediction was accepted")

    print(settings_block())
    print("\nmetrics self-test OK")
    print("  identical  psnr=%.2f ssim=%.6f" % (psnr(same, gt), ssim(same, gt)))
    print("  noisy      psnr=%.4f ssim=%.5f lpips=%.5f"
          % (s_noisy["psnr"], s_noisy["ssim"], s_noisy["lpips"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
