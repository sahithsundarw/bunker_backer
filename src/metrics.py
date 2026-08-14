"""Metric implementations with PINNED settings (SPEC 10, asserted by V31).

    psnr : data_range=1.0
    ssim : data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
    lpips: net='alex', grayscale -> repeat(1,3,1,1), scaled [0,1] -> [-1,1]

Deviating makes our numbers incomparable. State these settings in the deck.

V30: score the RELOADED on-disk artifacts, not in-memory tensors.

This module may import skimage/lpips freely -- it is not inference.py and not inside the
measured window.

Owner: loss-metrics.
"""

from __future__ import annotations

import numpy as np


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: float32 HxW in [0,1]; pred ALREADY clipped."""
    raise NotImplementedError("psnr: not implemented yet")


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Wang et al. (2004) reference settings -- see module docstring."""
    raise NotImplementedError("ssim: not implemented yet")


def lpips_score(pred: np.ndarray, gt: np.ndarray, device: str = "cuda") -> float:
    """AlexNet backbone, grayscale replicated to 3 channels, scaled to [-1,1]."""
    raise NotImplementedError("lpips_score: not implemented yet")
