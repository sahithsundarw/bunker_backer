"""Degradation simulator -- built to MEASUREMENTS, not to SPEC 6.4.

Binding parameterisation: docs/decisions.md D12, docs/SPEC_ADDENDUM.md section 12.

Measured facts this module honours:
  * Downsample is a SHARPENING kernel, not a box. The recovered 4x4 kernel is applied as a
    fixed conv; bicubic(antialias=False) is within 1.22e-05 of optimal and is kept as a
    minority randomisation alternative (docs/decisions.md D1). Both resamplers are
    implemented from scratch here -- nothing depends on a third-party library's
    undocumented antialias or clipping behaviour (docs/SPEC_ADDENDUM.md section 5).
  * Noise is applied AFTER downsampling for the CANONICAL (modal) order -- residual
    autocorrelation is ~0 or slightly negative at lags (0,1),(1,0),(1,1) (docs/decisions.md
    D2). This stays the majority case. But KLA's brief requires robustness to "any order" of
    {downsample, speckle+shot, additive Gaussian} on a hidden test set that may not match this
    measurement, so every one of the 6 orderings of those three operations is reachable on a
    synthetic sample, at documented frequencies, with the canonical order kept modal.
  * There is NO additive Gaussian floor. The three-parameter fit gives sigma=0.000000,
    a=0.011253 (shot/linear), v=0.015745 (speckle/quadratic). SPEC 6.4's add_speckle
    implements only the quadratic term and over-noises darks by up to 12.5x.
    An additive Gaussian term is nevertheless retained as a cheap hedge, drawn from
    U(0, GAUSS_SIGMA_RANGE[1]) *including zero*, because SPEC F3 names it and F7 warns test
    noise levels may vary (docs/decisions.md D12).
  * Synthetic LR is NOT clipped to [0,1]. Real NoisyLR spans [-0.28, 2.16] (SPEC F5).
    Only the noise *variance* is floored at zero -- never a pixel value.
  * The randomisation ranges on a/v/sigma are widened beyond the raw +/-30% fit tolerance
    specifically so the synthetic upper tail reaches the real one: measured synthetic max was
    1.7177 against a real TRAIN max of 2.0735 (dataset-wide 2.1580) -- see the tail-coverage
    note below FIDELITY_TOLERANCE. This widening does not touch V33/fidelity_report, which
    compares against the exact FITTED (unrandomised) parameters, not the training-time range.

Numpy only, deliberately: this module is import-light, has no torch dependency, and is
callable from scripts and from DataLoader workers alike.

Run the V33 fidelity evidence job with:
    py -3.12 -m src.degrade --data-root C:/kla-data

Owner: data-pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ======================================================================================
# MEASURED CONSTANTS  (docs/decisions.md D1, D2, D12 -- do not re-derive, do not "improve")
# ======================================================================================

#: Recovered 4x4 downsample kernel, offsets [-1,0,1,2] from 2i in both axes.
#: Least-squares recovery over 3,125,000 equations, n=200 pairs. docs/decisions.md D1.
#: Centre weights ~0.320 with negative surround lobes ~-0.045: a sharpening kernel.
#: Sums to 0.99979320 -- flux preserving. A 2x2 box (0.25 x4) is REFUTED.
RECOVERED_KERNEL_4X4: np.ndarray = np.array(
    [
        [0.014066, -0.038645, -0.045098, 0.007462],
        [-0.045368, 0.327878, 0.318776, -0.033904],
        [-0.048204, 0.321710, 0.312729, -0.039037],
        [0.017182, -0.039238, -0.039416, 0.008900],
    ],
    dtype=np.float64,
)
KERNEL_OFFSETS: tuple[int, ...] = (-1, 0, 1, 2)

# Global three-parameter noise fit. docs/decisions.md D2/D12.
NOISE_SIGMA_FITTED = 0.0        # additive Gaussian: fits to exactly zero
NOISE_A_FITTED = 0.011253       # shot / linear term
NOISE_V_FITTED = 0.015745       # speckle / quadratic term

# Randomise a and v over +/-120% (not +/-30%) around the fit, and widen the additive-Gaussian
# hedge from U(0,0.02) to U(0,0.065). This is a MEASURED widening, not a guess: D25 found the
# synthetic upper tail (max 1.7177 under the +/-30%/U(0,0.02) ranges, over 2800 GT images x
# multiple draws) falls well short of the real TRAIN max (2.0735, dataset-wide 2.158).
# Empirically swept on the real 2800 non-val train GT images (this module, 15 draws/image,
# 42,000 synthetic samples per setting):
#
#   randomise_frac   gauss_sigma_hi   synthetic max   reaches train max (2.0735)?
#   0.30 (old)        0.02 (old)       1.79-1.92        no
#   1.00              0.06             2.01             borderline
#   1.10              0.06             2.12             yes
#   1.20              0.065            2.13             yes, ~0.06 headroom to dataset max
#   1.50              0.08             2.19             yes, exceeds dataset max
#
# 1.20 / 0.065 was chosen as the smallest widening in the sweep that cleared the train max
# with margin across reseeded repeats, without overshooting the dataset-wide max by much.
# This widening ONLY affects sample_noise_params/degrade() (the training-time randomised
# path) -- it does not touch degrade_fitted or FITTED_NOISE, so it cannot move the V33
# fidelity numbers, which are computed against the exact (unrandomised) fit.
#
# Known side effect, recorded rather than hidden: because the speckle term v*x^2 is symmetric
# in x, widening v (needed to reach the bright tail) also inflates the already over-produced
# <0 tail (D25: synthetic 0.62% vs real 0.26%, ~2.4x over) to roughly 1.8-1.9% at this
# setting, ~7x over. No V-check gates on the <0 fraction today; flagged in docs/decisions.md
# for a follow-up (asymmetric a/v treatment) rather than silently accepted.
NOISE_RANDOMISE_FRAC = 1.20     # randomise a and v by +/-120% (was +/-30%; docs/decisions.md)

# Additive Gaussian is retained as a hedge even though it fits to zero: SPEC F3 names it a
# benchmark degradation and F7 warns test noise levels may vary. Sampling from zero upward
# is free when the true value is zero. docs/decisions.md D12. Upper bound widened 0.02->0.065
# for the same tail-coverage reason as NOISE_RANDOMISE_FRAC above.
GAUSS_SIGMA_RANGE: tuple[float, float] = (0.0, 0.065)

# Fraction of samples that use bicubic(antialias=False) instead of the recovered kernel,
# for randomisation diversity per SPEC 6.3. Keep it a minority.
BICUBIC_ALT_PROB = 0.25

# SPEC F4/5.3 ask for the degradation *order* to be randomised where the order is undisclosed,
# and KLA's brief separately asks the trained model to "handle them even when applied in any
# order". D2 measured noise-after-downsampling as the modal real-data order, so it stays the
# majority case -- but ALL 6 orderings of {downsample D, shot+speckle S, additive Gaussian G}
# must be reachable, not just the Gaussian-only hedge this used to be. Each of S and G is
# independently moved before D with the probability below; when both move, or neither moves,
# a further coin flip decides which of the two noise terms is applied first. See degrade().
GAUSS_PRE_DOWN_PROB = 0.15
SHOT_SPECKLE_PRE_DOWN_PROB = 0.15   # symmetric hedge: shot+speckle before D, same rate as G
POST_ORDER_SWAP_PROB = 0.10         # when neither moves pre-D, chance G is applied before S
PRE_ORDER_SWAP_PROB = 0.50          # when both move pre-D, coin flip for their relative order

# The two-parameter values SPEC 5.2 asks for. Correct answer to the question as posed;
# MUST NEVER be used to generate data (they over-noise the darkest bin by 12.5x).
# Kept only so the V33 evidence figure can show the failure. docs/decisions.md D2.
SPEC_2PAR_SIGMA = 0.036991
SPEC_2PAR_V = 0.026781

#: V33 tolerance, claimed and documented.
#:
#: Set from the MEASURED whole-train-set agreement (2800 non-val pairs, 45,875,200 px,
#: 24 quantile bins), with ~25% headroom on each figure so the check is not knife-edge:
#:
#:   quantity                          measured    tolerance
#:   mean abs rel err (all bins)         0.3885    <= 0.50
#:   mean abs rel err (x >= 0.1)         0.2764    <= 0.35
#:   max abs rel err                     1.6698    <= 2.20   (at bin centre x=0.0288)
#:   std(synth resid)/std(real)          1.0554    in [0.90, 1.15]
#:   binned R^2                          0.9804    >= 0.97
#:   gain over SPEC 2-par, mean bin      1.894x    >= 1.5x   (2-par: 0.7359)
#:   gain over SPEC 2-par, worst bin     5.917x    >= 3.0x   (2-par: 9.8799)
#:
#: The residual disagreement is a property of the global three-parameter fit itself, not of
#: this simulator: docs/decisions.md D2 reports mean abs rel err 0.3356 for that fit on its
#: own bins, and the parameters are binding (D12) and must not be re-fitted here. The error
#: is concentrated in the darkest bins, where the absolute variance is ~1e-4 and the linear
#: shot term slightly overshoots. The relevant reference point is SPEC 5.2's two-parameter
#: model, which is 12.5x off in that same regime -- hence the explicit
#: "beats the 2-parameter model by >= 3x" clause, which is what makes this check load-bearing
#: rather than a loose band.
FIDELITY_TOLERANCE: dict[str, Any] = {
    "mean_abs_rel_err": 0.50,             # over all quantile bins of the clean LR intensity
    "mean_abs_rel_err_x_ge_0p1": 0.35,    # over bins with centre >= 0.1
    "max_abs_rel_err": 2.20,              # dominated by the darkest bins, where var ~ 1e-4
    "resid_std_ratio": (0.90, 1.15),      # std(synthetic residual) / std(real residual)
    "binned_r2_min": 0.97,                # R^2 of synthetic vs real binned variance
    "min_gain_over_spec_2par": 1.5,       # mean-over-bins rel err vs SPEC 5.2's 2-par model
    "min_gain_over_spec_2par_worst_bin": 3.0,   # worst-bin rel err vs the same
}


# ======================================================================================
# RESAMPLERS -- implemented from scratch, no third-party resize
# ======================================================================================
def _catmull_rom(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys bicubic with a=-0.5, the OpenCV/PIL convention (same as scripts/fit_degradation.py)."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    ax2 = ax * ax
    ax3 = ax2 * ax
    out = np.zeros_like(ax)
    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    out[m1] = (a + 2.0) * ax3[m1] - (a + 3.0) * ax2[m1] + 1.0
    out[m2] = a * ax3[m2] - 5.0 * a * ax2[m2] + 8.0 * a * ax[m2] - 4.0 * a
    return out


def _bicubic_aa_off_kernel() -> np.ndarray:
    """bicubic(antialias=False) at scale x2 is exactly a separable 4x4 conv.

    Centre-aligned mapping puts output centre i at input coordinate 2i+0.5, so the four
    taps sit at offsets (-1,0,1,2) with distances (-1.5,-0.5,0.5,1.5) -- identical support
    to the recovered kernel, which is why the two are interchangeable at zero code cost.
    """
    taps = _catmull_rom(np.array([-1.5, -0.5, 0.5, 1.5]))
    taps = taps / taps.sum()
    return np.outer(taps, taps)


#: bicubic(antialias=False), the minority alternative resampler (docs/decisions.md D1).
BICUBIC_AA_OFF_KERNEL_4X4: np.ndarray = _bicubic_aa_off_kernel()


def conv_downsample_2x(
    gt: np.ndarray,
    kernel: np.ndarray | None = None,
    offsets: Sequence[int] = KERNEL_OFFSETS,
) -> np.ndarray:
    """Exact x2 downsample as a fixed conv: ``LR[i,j] = sum_ab w[a,b] * GT[2i+a, 2j+b]``.

    Edge-replicated at the borders, which is what the least-squares recovery's clipped-index
    accumulation did (scripts/fit_degradation.py weight_matrix). The kernel is NOT
    renormalised -- its measured sum of 0.99979 is part of the measurement.

    Args:
        gt: 2-D array with even height and width.
        kernel: KxK weights; defaults to the recovered kernel.
        offsets: per-axis taps relative to ``2i``; must match the kernel size.

    Returns:
        float32 array of shape ``(H//2, W//2)``. Never clipped.
    """
    a = np.asarray(gt)
    if a.ndim != 2:
        raise ValueError(f"conv_downsample_2x expects a 2-D array, got shape {a.shape}")
    h, w = a.shape
    if h % 2 or w % 2:
        raise ValueError(f"conv_downsample_2x expects even dimensions, got {(h, w)}")
    k = RECOVERED_KERNEL_4X4 if kernel is None else np.asarray(kernel, dtype=np.float64)
    off = tuple(int(o) for o in offsets)
    if k.shape != (len(off), len(off)):
        raise ValueError(f"kernel shape {k.shape} does not match offsets {off}")

    pad_lo = max(0, -min(off))
    pad_hi = max(0, max(off) - 1)
    src = np.pad(a.astype(np.float64, copy=False), ((pad_lo, pad_hi), (pad_lo, pad_hi)), mode="edge")

    ho, wo = h // 2, w // 2
    out = np.zeros((ho, wo), dtype=np.float64)
    for ai, oa in enumerate(off):
        r0 = pad_lo + oa
        rows = src[r0 : r0 + 2 * ho : 2, :]
        for bi, ob in enumerate(off):
            wgt = k[ai, bi]
            if wgt == 0.0:
                continue
            c0 = pad_lo + ob
            out += wgt * rows[:, c0 : c0 + 2 * wo : 2]
    return out.astype(np.float32)


# ======================================================================================
# CONFIG / PARAMETERS
# ======================================================================================
@dataclass(frozen=True)
class NoiseParams:
    """One draw of the three-parameter noise model. ``var = sigma^2 + a*x + v*x^2``."""

    sigma: float = NOISE_SIGMA_FITTED
    a: float = NOISE_A_FITTED
    v: float = NOISE_V_FITTED


#: The exact global fit, unrandomised. Used for the V33 fidelity comparison.
FITTED_NOISE = NoiseParams(NOISE_SIGMA_FITTED, NOISE_A_FITTED, NOISE_V_FITTED)


@dataclass(frozen=True)
class DegradeConfig:
    """Randomisation ranges for synthetic pair generation (SPEC 6.3, docs/decisions.md D12)."""

    a: float = NOISE_A_FITTED
    v: float = NOISE_V_FITTED
    randomise_frac: float = NOISE_RANDOMISE_FRAC
    gauss_sigma_range: tuple[float, float] = GAUSS_SIGMA_RANGE
    bicubic_alt_prob: float = BICUBIC_ALT_PROB
    gauss_pre_down_prob: float = GAUSS_PRE_DOWN_PROB
    shot_speckle_pre_down_prob: float = SHOT_SPECKLE_PRE_DOWN_PROB
    post_order_swap_prob: float = POST_ORDER_SWAP_PROB

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any] | None) -> "DegradeConfig":
        """Build from a ``data.degrade`` config block. Unknown keys are rejected loudly.

        ``clip_lr`` must stay false (SPEC F5): a config that flips it would silently generate
        training data from a different distribution than the (unclipped) test inputs, so it
        raises instead.

        ``noise_after_downsample`` is accepted but no longer enforced as a single global
        switch: the degradation now randomises the actual application order per sample (see
        ``degrade()``), with the measured D2 order kept as the majority case rather than the
        only reachable one -- KLA's brief requires the trained model to handle any order of
        {downsample, speckle+shot, additive Gaussian}, which a hard-coded order cannot satisfy.
        The key is kept in ``known`` purely for backward compatibility with existing configs
        that still set it; it is otherwise a no-op.
        """
        if not cfg:
            return cls()
        known = {
            "a", "v", "randomise_frac", "gauss_sigma_range", "bicubic_alt_prob",
            "gauss_pre_down_prob", "shot_speckle_pre_down_prob", "post_order_swap_prob",
            "kernel", "clip_lr", "noise_after_downsample",
        }
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ValueError(f"unknown data.degrade keys: {unknown}")
        if cfg.get("clip_lr", False):
            raise ValueError(
                "data.degrade.clip_lr must be false: real NoisyLR spans [-0.28, 2.16] and "
                "clipping synthetic LR would give training inputs a different distribution "
                "from the test inputs (SPEC F5, docs/decisions.md D12)"
            )
        kernel = str(cfg.get("kernel", "recovered_4x4"))
        if kernel not in ("recovered_4x4", "bicubic_aa_off"):
            raise ValueError(f"unknown data.degrade.kernel: {kernel!r}")
        gsr = cfg.get("gauss_sigma_range", GAUSS_SIGMA_RANGE)
        return cls(
            a=float(cfg.get("a", NOISE_A_FITTED)),
            v=float(cfg.get("v", NOISE_V_FITTED)),
            randomise_frac=float(cfg.get("randomise_frac", NOISE_RANDOMISE_FRAC)),
            gauss_sigma_range=(float(gsr[0]), float(gsr[1])),
            # kernel: recovered_4x4 is primary; asking for bicubic_aa_off makes it primary
            # instead, which flips the minority branch to the recovered kernel.
            bicubic_alt_prob=(
                1.0 - float(cfg.get("bicubic_alt_prob", BICUBIC_ALT_PROB))
                if kernel == "bicubic_aa_off"
                else float(cfg.get("bicubic_alt_prob", BICUBIC_ALT_PROB))
            ),
            gauss_pre_down_prob=float(cfg.get("gauss_pre_down_prob", GAUSS_PRE_DOWN_PROB)),
            shot_speckle_pre_down_prob=float(
                cfg.get("shot_speckle_pre_down_prob", SHOT_SPECKLE_PRE_DOWN_PROB)
            ),
            post_order_swap_prob=float(cfg.get("post_order_swap_prob", POST_ORDER_SWAP_PROB)),
        )


def sample_noise_params(rng: np.random.Generator, cfg: DegradeConfig | None = None) -> NoiseParams:
    """Draw one (sigma, a, v) triple: a and v randomised by ``+/-cfg.randomise_frac``, sigma
    from ``U(cfg.gauss_sigma_range)``. Defaults are +/-120% and U(0, 0.065) (widened for
    upper-tail coverage; see NOISE_RANDOMISE_FRAC / GAUSS_SIGMA_RANGE)."""
    c = cfg or DegradeConfig()
    f = c.randomise_frac
    return NoiseParams(
        sigma=float(rng.uniform(c.gauss_sigma_range[0], c.gauss_sigma_range[1])),
        a=float(rng.uniform(c.a * (1.0 - f), c.a * (1.0 + f))),
        v=float(rng.uniform(c.v * (1.0 - f), c.v * (1.0 + f))),
    )


def noise_variance(x: np.ndarray, params: NoiseParams = FITTED_NOISE) -> np.ndarray:
    """``var(residual | x) = sigma^2 + a*max(x,0) + v*x^2``, the measured model.

    ``max(x, 0)`` on the shot term only: the sharpening kernel can push the clean LR
    slightly below zero, and photon count cannot. This floors a *variance*, never a pixel.
    """
    xf = np.asarray(x, dtype=np.float64)
    return params.sigma ** 2 + params.a * np.maximum(xf, 0.0) + params.v * xf * xf


# ======================================================================================
# DEGRADATION
# ======================================================================================
def downsample(
    gt: np.ndarray,
    rng: np.random.Generator | None = None,
    use_bicubic: bool | None = None,
    cfg: DegradeConfig | None = None,
) -> np.ndarray:
    """Downsample GT by exactly 2x. Returns the clean LR signal, no noise, never clipped.

    Args:
        gt: 2-D GT array, even sized.
        rng: used only to pick the resampler when ``use_bicubic`` is None.
        use_bicubic: force bicubic(antialias=False) (True) or the recovered kernel (False).
        cfg: supplies ``bicubic_alt_prob`` for the random choice.
    """
    c = cfg or DegradeConfig()
    if use_bicubic is None:
        use_bicubic = bool(rng.random() < c.bicubic_alt_prob) if rng is not None else False
    k = BICUBIC_AA_OFF_KERNEL_4X4 if use_bicubic else RECOVERED_KERNEL_4X4
    return conv_downsample_2x(gt, kernel=k)


def _shot_speckle_delta(x: np.ndarray, rng: np.random.Generator, p: NoiseParams) -> np.ndarray:
    """Shot (linear) + speckle (quadratic) noise delta, at whatever resolution ``x`` is.

    ``sqrt(a*max(x,0))*N(0,1) + x*sqrt(v)*N(0,1)`` -- the two signal-dependent terms of the
    measured three-parameter model. Factored out of ``add_noise`` so the order-permutation
    logic in ``degrade()`` can apply this same delta before OR after downsampling.
    """
    xf = np.asarray(x, dtype=np.float32)
    xf64 = xf.astype(np.float64)
    delta = np.zeros_like(xf, dtype=np.float32)
    if p.a > 0.0:
        shot_sd = np.sqrt(p.a * np.maximum(xf64, 0.0))
        delta = delta + (rng.standard_normal(xf.shape) * shot_sd).astype(np.float32)
    if p.v > 0.0:
        delta = delta + (rng.standard_normal(xf.shape) * (np.sqrt(p.v) * xf64)).astype(np.float32)
    return delta


def _gauss_delta(x: np.ndarray, rng: np.random.Generator, p: NoiseParams) -> np.ndarray:
    """Additive Gaussian delta, ``sigma*N(0,1)`` -- independent of ``x``'s value, shaped like it."""
    xf = np.asarray(x, dtype=np.float32)
    if p.sigma <= 0.0:
        return np.zeros_like(xf, dtype=np.float32)
    return (rng.standard_normal(xf.shape) * p.sigma).astype(np.float32)


def add_noise(
    lr_clean: np.ndarray,
    rng: np.random.Generator,
    params: NoiseParams | None = None,
    cfg: DegradeConfig | None = None,
) -> np.ndarray:
    """Apply the measured three-parameter noise AFTER downsampling -- the CANONICAL order.

    ``var(residual | x) = sigma^2 + a*max(x,0) + v*x^2`` with a and v randomised around the
    global fit and sigma drawn from ``U(0, gauss_sigma_range[1])`` including zero (ranges
    widened for upper-tail coverage -- see NOISE_RANDOMISE_FRAC / GAUSS_SIGMA_RANGE).

    The three terms are drawn independently and added explicitly rather than as a single
    Gaussian of the summed variance, so each mechanism stays visible:
      * shot / Poisson   ``sqrt(a*max(x,0)) * N(0,1)``    -- the linear term SPEC 6.4 omits
      * speckle          ``x * sqrt(v) * N(0,1)``         -- MATLAB imnoise('speckle') form
      * additive Gaussian ``sigma * N(0,1)``              -- the F3 hedge, usually small

    Draw order (shot, then speckle, then Gaussian) matches ``degrade()``'s canonical branch
    exactly, so this function's output is bit-identical to what it was before the order
    permutation was added, for a fixed rng stream and the canonical order.

    The result is NOT clipped (SPEC F5).
    """
    p = params if params is not None else sample_noise_params(rng, cfg)
    x = np.asarray(lr_clean, dtype=np.float32)
    return x + _shot_speckle_delta(x, rng, p) + _gauss_delta(x, rng, p)


def degrade(
    gt: np.ndarray,
    rng: np.random.Generator,
    cfg: DegradeConfig | None = None,
    params: NoiseParams | None = None,
    use_bicubic: bool | None = None,
) -> np.ndarray:
    """Full GT -> synthetic NoisyLR, with the degradation *order* randomised. Never clipped.

    KLA's brief requires the model to "handle them even when applied in any order" for the
    three degradations {downsample D, shot+speckle noise S, additive Gaussian noise G}. D2
    measured noise-after-downsampling (``D, S, G``) as the order the RELEASED data actually
    follows, so it stays the majority/modal case here -- but every one of the 3! = 6 orderings
    of {D, S, G} is reachable on a synthetic sample:

        D,S,G   canonical / modal (matches D2)
        D,G,S   swap the two post-down noise terms
        S,D,G   shot+speckle applied on the GT-resolution signal, then downsampled, then G
        G,D,S   additive Gaussian applied pre-down (blurred/attenuated by the kernel), then S
        S,G,D   both noise terms applied pre-down, S first
        G,S,D   both noise terms applied pre-down, G first

    ``S`` and ``G`` are each independently moved before ``D`` with probability
    ``shot_speckle_pre_down_prob`` / ``gauss_pre_down_prob``. If both move, or neither moves,
    a further coin flip (``PRE_ORDER_SWAP_PROB`` / ``post_order_swap_prob``) decides which of
    the two is applied first, so all 6 orderings have non-zero probability. Order is only
    randomised for the on-the-fly synthetic path (``params is None``); an explicit ``params=``
    caller (``degrade_fitted``, the V33 fidelity path) always gets the canonical order,
    deterministically, because that is what the fidelity comparison is measuring.

    When S or G is applied pre-down, it acts on the GT-resolution array using the GT pixel
    value as ``x`` -- an approximation (the measured model was fit on the LR resolution), but
    the intent here is reachability for robustness, not a claim that this is the true generative
    process; D2's evidence is exactly why the post-down order remains modal.
    """
    c = cfg or DegradeConfig()
    p = params if params is not None else sample_noise_params(rng, c)
    randomise_order = params is None
    src = np.asarray(gt, dtype=np.float32)

    g_pre = randomise_order and c.gauss_pre_down_prob > 0.0 and bool(
        rng.random() < c.gauss_pre_down_prob
    )
    s_pre = randomise_order and c.shot_speckle_pre_down_prob > 0.0 and bool(
        rng.random() < c.shot_speckle_pre_down_prob
    )

    pre_ops: list[str] = []
    post_ops: list[str] = []
    if g_pre and s_pre:
        pre_ops = ["S", "G"] if rng.random() < PRE_ORDER_SWAP_PROB else ["G", "S"]
    elif g_pre:
        pre_ops, post_ops = ["G"], ["S"]
    elif s_pre:
        pre_ops, post_ops = ["S"], ["G"]
    else:
        post_ops = (
            ["G", "S"] if randomise_order and rng.random() < c.post_order_swap_prob else ["S", "G"]
        )

    for op in pre_ops:
        src = src + (_gauss_delta(src, rng, p) if op == "G" else _shot_speckle_delta(src, rng, p))

    lr = downsample(src, rng=rng, use_bicubic=use_bicubic, cfg=c)

    for op in post_ops:
        lr = lr + (_gauss_delta(lr, rng, p) if op == "G" else _shot_speckle_delta(lr, rng, p))

    return lr


def degrade_fitted(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Re-degrade GT with the exact fitted parameters: recovered kernel, sigma=0, no
    randomisation. This is the V33 fidelity path, not the training path."""
    lr = conv_downsample_2x(gt, kernel=RECOVERED_KERNEL_4X4)
    return add_noise(lr, rng, params=FITTED_NOISE)


# ======================================================================================
# V33 -- DEGRADATION SIMULATOR FIDELITY
# ======================================================================================
def variance_vs_intensity(
    x: np.ndarray, resid: np.ndarray, nbins: int = 24, edges: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bin ``resid`` by ``x`` and return (bin_centres, variances, counts, edges).

    Quantile bins, matching scripts/fit_degradation.py so the numbers are comparable.
    Pass ``edges`` to reuse another curve's bins -- required for a like-for-like compare.
    """
    xf = np.asarray(x, dtype=np.float64).ravel()
    rf = np.asarray(resid, dtype=np.float64).ravel()
    if edges is None:
        edges = np.unique(np.quantile(xf, np.linspace(0.0, 1.0, nbins + 1)))
    idx = np.clip(np.searchsorted(edges, xf, side="right") - 1, 0, len(edges) - 2)
    centres, variances, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        c = int(m.sum())
        if c < 50:
            continue
        centres.append(float(xf[m].mean()))
        variances.append(float(rf[m].var()))
        counts.append(c)
    return (
        np.asarray(centres),
        np.asarray(variances),
        np.asarray(counts, dtype=np.float64),
        np.asarray(edges),
    )


def _read_name_list(path: Path) -> list[str]:
    """Read a committed one-name-per-line list, ignoring blanks and ``#`` comments."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def repo_root() -> Path:
    """Repo root, resolved from this file. Never from CWD (CLAUDE.md PD5)."""
    return Path(__file__).resolve().parents[1]


class _BinAccumulator:
    """Streaming per-bin (count, sum x, sum r, sum r^2) so the whole train split fits in RAM.

    Concatenating every residual would cost ~1.5 GB at 2800 pairs; the binned statistics
    cost O(nbins). Variance per bin is E[r^2] - E[r]^2, computed in float64.
    """

    def __init__(self, edges: np.ndarray, nseries: int = 2) -> None:
        self.edges = np.asarray(edges, dtype=np.float64)
        nb = len(self.edges) - 1
        self.count = np.zeros(nb, dtype=np.float64)
        self.sum_x = np.zeros(nb, dtype=np.float64)
        self.sum_r = np.zeros((nseries, nb), dtype=np.float64)
        self.sum_r2 = np.zeros((nseries, nb), dtype=np.float64)
        self.total_n = 0
        self.total_r = np.zeros(nseries, dtype=np.float64)
        self.total_r2 = np.zeros(nseries, dtype=np.float64)

    def add(self, x: np.ndarray, residuals: Sequence[np.ndarray]) -> None:
        xf = np.asarray(x, dtype=np.float64).ravel()
        idx = np.clip(np.searchsorted(self.edges, xf, side="right") - 1, 0, len(self.edges) - 2)
        nb = len(self.edges) - 1
        self.count += np.bincount(idx, minlength=nb)
        self.sum_x += np.bincount(idx, weights=xf, minlength=nb)
        self.total_n += xf.size
        for s, r in enumerate(residuals):
            rf = np.asarray(r, dtype=np.float64).ravel()
            self.sum_r[s] += np.bincount(idx, weights=rf, minlength=nb)
            self.sum_r2[s] += np.bincount(idx, weights=rf * rf, minlength=nb)
            self.total_r[s] += rf.sum()
            self.total_r2[s] += float(rf @ rf)

    def curves(self, min_count: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = self.count >= min_count
        c = self.sum_x[keep] / self.count[keep]
        mean = self.sum_r[:, keep] / self.count[keep]
        var = self.sum_r2[:, keep] / self.count[keep] - mean * mean
        return c, var, self.count[keep]

    def global_std(self) -> np.ndarray:
        mean = self.total_r / self.total_n
        return np.sqrt(self.total_r2 / self.total_n - mean * mean)

    def global_mean(self) -> np.ndarray:
        return self.total_r / self.total_n


def fidelity_report(
    data_root: str | Path,
    n: int | None = None,
    out_dir: str | Path | None = None,
    seed: int = 0,
    nbins: int = 24,
    names: Iterable[str] | None = None,
    make_figure: bool = True,
) -> dict[str, Any]:
    """V33: compare the synthetic residual's variance-vs-intensity curve to the real one.

    For every train pair outside the committed validation split:
        x        = conv_downsample_2x(GT)            # clean LR signal, recovered kernel
        r_real   = real NoisyLR - x
        r_synth  = degrade_fitted(GT) - x            # our simulator, fitted parameters
    Both residuals are binned by the SAME quantile edges of x, so the comparison is
    like-for-like. Agreement is reported as relative error per bin plus an R^2 of the
    synthetic against the real binned variance, against FIDELITY_TOLERANCE.

    File selection is **deterministic** -- the whole non-val train list, or, when ``n`` is
    given, an evenly strided subsample of it. It is deliberately not a random draw: the
    per-image noise level genuinely varies (docs/decisions.md D2 per-image table), so a
    random subsample moves the metric by more than the simulator's own error and would make
    the check flaky. ``seed`` therefore only controls the synthetic noise realisation, which
    moves the reported numbers by <0.003.

    Only ``train/`` is read, and only files NOT in configs/split_val.txt -- test_NoisyLR is
    never touched and no parameter is fitted here (SPEC F17).
    """
    root = Path(data_root)
    gt_dir = root / "train" / "GT"
    lr_dir = root / "train" / "NoisyLR"
    if not gt_dir.is_dir() or not lr_dir.is_dir():
        raise FileNotFoundError(f"expected {gt_dir} and {lr_dir}")

    val = set(_read_name_list(repo_root() / "configs" / "split_val.txt"))
    if names is None:
        pool = [p.name for p in sorted(gt_dir.glob("*.npy")) if p.name not in val]
        if n is not None and 0 < n < len(pool):
            step = len(pool) // n
            chosen = pool[::step][:n]
            sel_rule = f"every {step}-th of the {len(pool)} non-val train names, capped at n={n}"
        else:
            chosen = pool
            sel_rule = f"all {len(pool)} train names not in configs/split_val.txt"
    else:
        chosen = [str(x) for x in names]
        sel_rule = "explicit names= argument"
    if not chosen:
        raise RuntimeError("no train pairs left after excluding configs/split_val.txt")

    rng = np.random.default_rng(seed + 1)

    # Bin edges from a deterministic 64-file probe of the clean-LR intensity distribution.
    probe_step = max(1, len(chosen) // 64)
    probe = np.concatenate(
        [conv_downsample_2x(np.load(gt_dir / nm, allow_pickle=False)).ravel()
         for nm in chosen[::probe_step][:64]]
    ).astype(np.float64)
    edges = np.unique(np.quantile(probe, np.linspace(0.0, 1.0, nbins + 1)))
    edges[0] = -np.inf
    edges[-1] = np.inf
    del probe

    acc = _BinAccumulator(edges, nseries=2)
    for name in chosen:
        g = np.load(gt_dir / name, allow_pickle=False)
        real = np.load(lr_dir / name, allow_pickle=False)
        x = conv_downsample_2x(g)
        syn = degrade_fitted(g, rng)
        acc.add(x, [real.astype(np.float32) - x, syn - x])

    c_real, var, cnt = acc.curves()
    v_real, v_syn = var[0], var[1]
    g_std = acc.global_std()
    g_mean = acc.global_mean()

    rel = np.abs(v_syn - v_real) / np.maximum(v_real, 1e-12)
    hi = c_real >= 0.1
    ss_res = float(np.sum((v_syn - v_real) ** 2))
    ss_tot = float(np.sum((v_real - v_real.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    std_ratio = float(g_std[1] / g_std[0])

    # Reference point: SPEC 5.2's two-parameter model on the SAME bins. It is the answer to
    # 5.2 as literally posed and must never generate data; scoring it here is what turns
    # "our curve is close" into a quantified claim (docs/SPEC_ADDENDUM.md section 12).
    v_2par = SPEC_2PAR_SIGMA ** 2 + SPEC_2PAR_V * c_real ** 2
    rel_2par = np.abs(v_2par - v_real) / np.maximum(v_real, 1e-12)
    gain = float(rel_2par.mean() / rel.mean()) if rel.mean() > 0 else float("inf")
    gain_worst = float(rel_2par.max() / rel.max()) if rel.max() > 0 else float("inf")

    metrics = {
        "n_pairs": len(chosen),
        "n_pixels": int(acc.total_n),
        "n_bins": int(c_real.size),
        "mean_abs_rel_err": float(rel.mean()),
        "mean_abs_rel_err_x_ge_0p1": float(rel[hi].mean()) if hi.any() else float("nan"),
        "max_abs_rel_err": float(rel.max()),
        "argmax_rel_err_bin_centre": float(c_real[int(np.argmax(rel))]),
        "resid_std_real": float(g_std[0]),
        "resid_std_synth": float(g_std[1]),
        "resid_std_ratio": std_ratio,
        "binned_r2": float(r2),
        "resid_mean_real": float(g_mean[0]),
        "resid_mean_synth": float(g_mean[1]),
        "mean_abs_rel_err_spec_2par": float(rel_2par.mean()),
        "max_abs_rel_err_spec_2par": float(rel_2par.max()),
        "gain_over_spec_2par": gain,
        "gain_over_spec_2par_worst_bin": gain_worst,
        "synth_lr_clipped": False,
    }
    lo, up = FIDELITY_TOLERANCE["resid_std_ratio"]
    checks = {
        "mean_abs_rel_err": metrics["mean_abs_rel_err"] <= FIDELITY_TOLERANCE["mean_abs_rel_err"],
        "mean_abs_rel_err_x_ge_0p1": (
            metrics["mean_abs_rel_err_x_ge_0p1"] <= FIDELITY_TOLERANCE["mean_abs_rel_err_x_ge_0p1"]
        ),
        "max_abs_rel_err": metrics["max_abs_rel_err"] <= FIDELITY_TOLERANCE["max_abs_rel_err"],
        "resid_std_ratio": lo <= std_ratio <= up,
        "binned_r2": metrics["binned_r2"] >= FIDELITY_TOLERANCE["binned_r2_min"],
        "beats_spec_2par": gain >= FIDELITY_TOLERANCE["min_gain_over_spec_2par"],
        "beats_spec_2par_worst_bin": (
            gain_worst >= FIDELITY_TOLERANCE["min_gain_over_spec_2par_worst_bin"]
        ),
    }

    out = Path(out_dir) if out_dir is not None else repo_root() / "results" / "degrade_fidelity"
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "check": "V33",
        "params": {
            "sigma": FITTED_NOISE.sigma, "a": FITTED_NOISE.a, "v": FITTED_NOISE.v,
            "kernel": "recovered_4x4",
        },
        "tolerance": {k: (list(v) if isinstance(v, tuple) else v) for k, v in FIDELITY_TOLERANCE.items()},
        "metrics": metrics,
        "checks": checks,
        "pass": all(checks.values()),
        "curve": {
            "bin_centre": [float(c) for c in c_real],
            "var_real": [float(v) for v in v_real],
            "var_synth": [float(v) for v in v_syn],
            "var_spec_2par_rejected": [float(v) for v in v_2par],
            "count": [int(c) for c in cnt],
            "rel_err": [float(r) for r in rel],
            "rel_err_spec_2par_rejected": [float(r) for r in rel_2par],
        },
        "selection": {
            "rule": sel_rule,
            "n_requested": n,
            "n_used": len(chosen),
            "first": chosen[0],
            "last": chosen[-1],
            "val_excluded": len(val),
            "noise_seed": seed,
        },
        "data_root": str(root),
        "figure": str(out / "degrade_fidelity.png") if make_figure else None,
    }
    (out / "degrade_fidelity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if make_figure:
        _save_figure(out / "degrade_fidelity.png", c_real, v_real, v_syn, rel, metrics)
    return report


def _save_figure(
    path: Path,
    centres: np.ndarray,
    v_real: np.ndarray,
    v_syn: np.ndarray,
    rel: np.ndarray,
    metrics: Mapping[str, Any],
) -> None:
    """Save the V33 evidence figure. matplotlib is imported lazily: this module must stay
    import-light for DataLoader workers, and matplotlib is not an inference dependency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.linspace(max(centres.min(), 1e-3), centres.max(), 200)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax[0].loglog(centres, v_real, "o", ms=5, color="#1a1a1a", label="real NoisyLR residual")
    ax[0].loglog(centres, v_syn, "s", ms=4.5, mfc="none", color="#c1121f",
                 label="synthetic (src/degrade.py)")
    ax[0].loglog(xs, NOISE_A_FITTED * xs + NOISE_V_FITTED * xs ** 2, "-", lw=1.2, color="#c1121f",
                 label=r"fitted $a x + v x^2$  (a=%.6f, v=%.6f)" % (NOISE_A_FITTED, NOISE_V_FITTED))
    ax[0].loglog(xs, SPEC_2PAR_SIGMA ** 2 + SPEC_2PAR_V * xs ** 2, "--", lw=1.2, color="#4361ee",
                 label=r"SPEC 5.2 $\sigma^2 + v x^2$ (rejected: 12.5x too dark-noisy)")
    ax[0].set_xlabel("clean LR intensity  x  (recovered-kernel downsample of GT)")
    ax[0].set_ylabel(r"var(residual | x)")
    ax[0].set_title("V33 degradation fidelity: variance vs intensity")
    ax[0].grid(True, which="both", alpha=0.25)
    ax[0].legend(fontsize=7.5, loc="upper left")

    ax[1].semilogx(centres, 100.0 * rel, "o-", ms=4, color="#1a1a1a")
    ax[1].axhline(100.0 * float(metrics["mean_abs_rel_err"]), ls=":", color="#c1121f",
                  label="mean %.1f%%" % (100.0 * float(metrics["mean_abs_rel_err"])))
    ax[1].set_xlabel("bin centre  x")
    ax[1].set_ylabel("|var_synth - var_real| / var_real   [%]")
    ax[1].set_title(
        "per-bin relative error\nresid std: real %.6f / synth %.6f (ratio %.4f)"
        % (metrics["resid_std_real"], metrics["resid_std_synth"], metrics["resid_std_ratio"]),
        fontsize=10,
    )
    ax[1].grid(True, which="both", alpha=0.25)
    ax[1].legend(fontsize=8)

    fig.suptitle(
        "n=%d train pairs, %d px, %d bins. Synthetic LR is NOT clipped (SPEC F5)."
        % (metrics["n_pairs"], metrics["n_pixels"], metrics["n_bins"]),
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for the V33 evidence job."""
    ap = argparse.ArgumentParser(description="Degradation simulator fidelity (V33).")
    ap.add_argument("--data-root", required=True, help="dataset root containing train/GT and train/NoisyLR")
    ap.add_argument("--n", type=int, default=None,
                    help="cap the number of train pairs (evenly strided). Default: all non-val pairs")
    ap.add_argument("--out-dir", default=None, help="default: <repo>/results/degrade_fidelity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    rep = fidelity_report(
        args.data_root, n=args.n, out_dir=args.out_dir, seed=args.seed,
        nbins=args.nbins, make_figure=not args.no_figure,
    )
    payload = rep if args.verbose else {
        "pass": rep["pass"], "metrics": rep["metrics"], "checks": rep["checks"],
        "tolerance": rep["tolerance"], "figure": rep["figure"],
    }
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
