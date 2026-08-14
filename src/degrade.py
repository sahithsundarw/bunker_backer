"""Degradation simulator -- built to MEASUREMENTS, not to SPEC 6.4.

Binding parameterisation: docs/decisions.md D12, docs/SPEC_ADDENDUM.md section 12.

Measured facts this module must honour:
  * Downsample is a SHARPENING kernel, not a box. Use the recovered 4x4 kernel as a fixed
    conv; bicubic(antialias=False) is within 1.22e-05 of optimal and is kept as a minority
    randomisation alternative (docs/decisions.md D1).
  * Noise is applied AFTER downsampling -- residual autocorrelation is ~0 or slightly
    negative at lags (0,1),(1,0),(1,1) (docs/decisions.md D2).
  * There is NO additive Gaussian floor. The three-parameter fit gives sigma=0.000000,
    a=0.011253 (shot/linear), v=0.015745 (speckle/quadratic). SPEC 6.4's add_speckle
    implements only the quadratic term and over-noises darks by up to 12.5x.
  * Synthetic LR is NOT clipped to [0,1]. Real NoisyLR spans [-0.28, 2.16] (SPEC F5).

Owner: data-pipeline.
"""

from __future__ import annotations

import numpy as np

# Recovered 4x4 downsample kernel, offsets [-1,0,1,2] from 2i in both axes.
# Least-squares recovery over 3,125,000 equations, n=200 pairs. docs/decisions.md D1.
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

# Global three-parameter noise fit. docs/decisions.md D12.
NOISE_SIGMA_FITTED = 0.0        # additive Gaussian: fits to exactly zero
NOISE_A_FITTED = 0.011253       # shot / linear term
NOISE_V_FITTED = 0.015745       # speckle / quadratic term
NOISE_RANDOMISE_FRAC = 0.30     # randomise a and v by +/-30%

# Additive Gaussian is retained as a hedge even though it fits to zero: SPEC F3 names it a
# benchmark degradation and F7 warns test noise levels may vary. Sampling from zero upward
# is free when the true value is zero. docs/decisions.md D12.
GAUSS_SIGMA_RANGE: tuple[float, float] = (0.0, 0.02)

# Fraction of samples that use bicubic(antialias=False) instead of the recovered kernel,
# for randomisation diversity per SPEC 6.3. Keep it a minority.
BICUBIC_ALT_PROB = 0.25


def downsample(gt: np.ndarray, rng: np.random.Generator, use_bicubic: bool | None = None) -> np.ndarray:
    """Downsample GT by exactly 2x. Returns the clean LR signal, no noise."""
    raise NotImplementedError("downsample: not implemented yet")


def add_noise(lr_clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the measured three-parameter noise AFTER downsampling.

    var(residual | x) = sigma^2 + a*x + v*x^2, with a and v randomised +/-30% around the
    global fit and sigma drawn from U(0, 0.02) including zero.

    MUST NOT clip the result (SPEC F5).
    """
    raise NotImplementedError("add_noise: not implemented yet")


def degrade(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Full GT -> NoisyLR degradation: downsample, then noise. Never clipped."""
    raise NotImplementedError("degrade: not implemented yet")
