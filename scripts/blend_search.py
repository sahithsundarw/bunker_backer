#!/usr/bin/env python3
"""Phase 3 (codex/residual-ls5-refinement): sweep alpha for the LS5/refined blend.

final = clip(alpha * refined_output + (1 - alpha) * LS5_output, 0, 1)

Reads two already-on-disk, already-clipped prediction directories (one per method, written by
scripts/make_baselines.py) and scores each blend with the pinned metrics from src/metrics.py.
This intermediate sweep does NOT round-trip each candidate through disk -- both inputs are
themselves reloaded .npy arrays, so this only skips re-saving derived blends that most alphas
will discard. The winning alpha's predictions are written to disk and re-scored through
scripts/evaluate.py (which does the full V30 disk round-trip, with LPIPS) as the number of
record -- see the printed follow-up command.

Usage:
    python scripts/blend_search.py --data_root /Users/shanmukhsai/Downloads \
        --ls5_dir results/baselines/final --refined_dir results/residual_experiments/r1_nb4/preds/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import psnr, ssim  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ls5_dir", required=True)
    ap.add_argument("--refined_dir", required=True)
    ap.add_argument("--split", default=str(_ROOT / "configs" / "split_val.txt"))
    ap.add_argument("--write_alpha", type=float, default=None,
                     help="Skip the sweep; write clip(a*refined+(1-a)*ls5,0,1) predictions to "
                          "<refined_dir>/../blend_alpha<a>/ for disk-based re-scoring.")
    ap.add_argument("--write_out", default=None,
                     help="Output dir for --write_alpha (default: sibling of refined_dir's parent).")
    args = ap.parse_args(argv)

    gt_dir = Path(args.data_root) / "train" / "GT"
    ls5_dir = Path(args.ls5_dir)
    refined_dir = Path(args.refined_dir)
    names = [ln.strip() for ln in Path(args.split).read_text().splitlines()
              if ln.strip() and not ln.startswith("#")]

    ls5 = [np.load(ls5_dir / n, allow_pickle=False).astype(np.float32) for n in names]
    ref = [np.load(refined_dir / n, allow_pickle=False).astype(np.float32) for n in names]

    if args.write_alpha is not None:
        a = float(args.write_alpha)
        out_dir = Path(args.write_out) if args.write_out else (
            refined_dir.parent.parent / f"blend_alpha{a:.2f}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for n, l, r in zip(names, ls5, ref):
            blend = np.clip(a * r + (1.0 - a) * l, 0.0, 1.0).astype(np.float32)
            np.save(out_dir / n, blend)
        print(f"wrote {len(names)} blended predictions (alpha={a:.2f}) to {out_dir}")
        return 0

    gts = [np.load(gt_dir / n, allow_pickle=False).astype(np.float32) for n in names]

    alphas = [round(0.05 * k, 2) for k in range(1, 21)]
    best = {"alpha": None, "psnr": float("-inf"), "ssim": None}
    print(f"{'alpha':>6} {'psnr_mean':>10} {'psnr_std':>9} {'ssim_mean':>10} {'ssim_std':>9}")
    for a in alphas:
        ps, ss = [], []
        for g, l, r in zip(gts, ls5, ref):
            blend = np.clip(a * r + (1.0 - a) * l, 0.0, 1.0).astype(np.float32)
            ps.append(psnr(blend, g))
            ss.append(ssim(blend, g))
        pm, pstd = float(np.mean(ps)), float(np.std(ps))
        sm, sstd = float(np.mean(ss)), float(np.std(ss))
        print(f"{a:6.2f} {pm:10.4f} {pstd:9.4f} {sm:10.5f} {sstd:9.5f}")
        if pm > best["psnr"]:
            best = {"alpha": a, "psnr": pm, "ssim": sm}

    print(f"\nbest alpha = {best['alpha']} (psnr={best['psnr']:.4f}, ssim={best['ssim']:.5f})")
    print("Write and re-score the winning alpha through the disk pipeline with:")
    print(f"  python scripts/blend_search.py --write_alpha {best['alpha']} "
          f"--data_root {args.data_root} --ls5_dir {args.ls5_dir} --refined_dir {args.refined_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
