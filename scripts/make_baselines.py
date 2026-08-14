#!/usr/bin/env python3
"""Generate the required baselines (SPEC 10).

1. Bicubic x2 of raw NoisyLR  -- measured floor: 23.4247 +/- 2.8319 dB PSNR,
   0.54284 SSIM on 200 held-out train pairs with clip-to-[0,1] (docs/decisions.md D3).
2. Classical denoise -> bicubic x2.
3. Small plain U-Net at the same training budget.

Owner: loss-metrics.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", default="results/metrics_summary.md")
    _ = ap.parse_args(argv)
    raise NotImplementedError("make_baselines.main: not implemented yet")


if __name__ == "__main__":
    sys.exit(main())
