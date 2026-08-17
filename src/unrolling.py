"""Algorithm-unrolling hybrid (Round 2 differentiation, Phase 4, stretch goal).

Cites Monga, V. et al. (2021), *Algorithm Unrolling: Interpretable, Efficient Deep Learning
for Signal and Image Processing*, IEEE Signal Processing Magazine -- the exact survey KLA's
own problem-statement material supplied (docs/SPEC.md section 14 references list).

The idea: unroll a fixed number of proximal-gradient steps for the regularised inverse
problem ``min_x 0.5||Kx - y||^2 + R(x)`` into a feed-forward network, replacing the hand-designed
proximal operator (denoiser) at each step with a small learned CNN. Most unrolling work has
to learn (or assume a generic form for) the forward operator ``K`` too, since the true
degradation is usually unknown. This project does not have that problem: `src/degrade.py`'s
``RECOVERED_KERNEL_4X4`` is an EMPIRICALLY MEASURED forward operator (least-squares recovery
over 3,125,000 equations, docs/decisions.md D1) -- it is plugged in directly below as a fixed,
non-trainable buffer, rather than re-learned. The unrolling is exact with respect to the
measured forward model, not approximate.

Owner: main session (Round 2 differentiation, not on the parallel-agent ownership map --
this is new, stretch-goal architecture work, not a change to the frozen NAFSR/UNetSR
contract in src/model.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NAFBlock, bilinear_upsample
from .degrade import RECOVERED_KERNEL_4X4

__all__ = ["UnrolledSR"]


class UnrolledSR(nn.Module):
    """T unrolled proximal-gradient steps against the measured x2 degradation kernel (D1).

    Each step:
        grad = K^T(K(x) - y)                # data-consistency gradient, K is FIXED (measured)
        z    = x - step_size_t * grad       # gradient step, step_size_t is LEARNED per step
        x    = z + denoiser_t(z)            # learned proximal operator (residual CNN)

    ``x`` lives at HR resolution throughout (initialised via a bilinear upsample of the input,
    matching NAFSR's own global-skip convention, SPEC 7.1). ``K``/``K^T`` are implemented as a
    fixed strided conv / transposed conv using the recovered kernel -- see ``_K``/``_KT`` for
    the exact padding convention, which mirrors ``src.degrade.conv_downsample_2x`` (edge-padded)
    up to the boundary-adjoint approximation stated there. That approximation is standard in
    the unrolling literature (the learned denoiser corrects for it during training) and is
    stated here rather than silently assumed exact.

    Only ``scale=2`` is supported: ``K``/``K^T`` are hardcoded for the measured x2 kernel.
    """

    def __init__(
        self,
        width: int = 32,
        num_steps: int = 6,
        denoiser_blocks: int = 2,
        scale: int = 2,
        in_ch: int = 1,
        out_ch: int = 1,
        share_denoiser: bool = True,
        step_size_init: float = 0.05,
    ) -> None:
        super().__init__()
        if int(scale) != 2:
            raise ValueError(
                f"UnrolledSR only supports scale=2 (the measured kernel is x2); got {scale}"
            )
        if width < 1 or num_steps < 1:
            raise ValueError(f"width and num_steps must be >= 1 (got {width}, {num_steps})")
        self.scale = 2
        self.num_steps = int(num_steps)
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.share_denoiser = bool(share_denoiser)

        kernel = torch.from_numpy(RECOVERED_KERNEL_4X4.astype("float32"))[None, None]  # (1,1,4,4)
        self.register_buffer("kernel", kernel)
        # One learned step size per unrolled iteration -- clamped positive in _step_size so
        # training cannot walk it to a sign flip (which would turn the gradient step into an
        # ascent step and diverge).
        self.raw_step_size = nn.Parameter(torch.full((self.num_steps,), float(step_size_init)))

        def make_denoiser() -> nn.Module:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, width, kernel_size=3, padding=1)]
            layers += [NAFBlock(width) for _ in range(int(denoiser_blocks))]
            layers.append(nn.Conv2d(width, out_ch, kernel_size=3, padding=1))
            return nn.Sequential(*layers)

        if self.share_denoiser:
            self.denoiser = make_denoiser()
        else:
            self.denoisers = nn.ModuleList([make_denoiser() for _ in range(self.num_steps)])

    def _denoiser(self, step: int) -> nn.Module:
        return self.denoiser if self.share_denoiser else self.denoisers[step]

    def _step_size(self, step: int) -> torch.Tensor:
        # Softplus keeps every step size strictly positive without a hard clamp (which would
        # zero the gradient once clamped) -- a smooth reparameterisation, not a projection.
        return F.softplus(self.raw_step_size[step])

    def _K(self, x: torch.Tensor) -> torch.Tensor:
        """Measured forward operator: HR -> LR. Mirrors conv_downsample_2x's edge padding."""
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(xp, self.kernel, stride=self.scale)

    def _KT(self, r: torch.Tensor) -> torch.Tensor:
        """Adjoint: LR -> HR. Zero-padding adjoint of the forward conv, cropped back to HR
        size -- the standard unrolling approximation for a replicate-padded forward op (the
        learned denoiser corrects the residual boundary error during training)."""
        out = F.conv_transpose2d(r, self.kernel, stride=self.scale)
        return out[..., 1:-1, 1:-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x  # the input IS the LR observation; keep the name distinct from the HR iterate
        est = bilinear_upsample(y, self.scale)
        for k in range(self.num_steps):
            grad = self._KT(self._K(est) - y)
            z = est - self._step_size(k) * grad
            est = z + self._denoiser(k)(z)
        return est
