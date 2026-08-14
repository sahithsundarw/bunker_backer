"""Loss functions: Charbonnier, SSIM/MS-SSIM, FFT, optional LPIPS (SPEC 8).

L_total = 1.0*Charbonnier + 0.15*(1-MS_SSIM) + 0.05*L_fft + 0.02*L_lpips

Compute the loss on the UNCLIPPED network output; clip only at save time. Clipping inside
the loss zero-grads saturated pixels (SPEC 8, SPEC 18 pitfall 3).

MS-SSIM needs >=161 px across its 5 scales. At 128x128 GT patches it will throw -- use
single-scale SSIM at 128 or raise the patch size to 192+.

No adversarial loss (SPEC 7.2): hallucinated structure is the worst failure in inspection.

Owner: loss-metrics.
"""

from __future__ import annotations

import torch


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """sqrt((pred-target)^2 + eps^2). Smooth L1; drives PSNR."""
    raise NotImplementedError("charbonnier: not implemented yet")


def fft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between FFT magnitudes. Penalises missing high-frequency content."""
    raise NotImplementedError("fft_loss: not implemented yet")
