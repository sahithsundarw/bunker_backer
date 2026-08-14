"""Renormalisation experiment.

GT is per-image min-max normalised to exactly [0,1] (all 3200 files attain both
endpoints). That raises the question of whether predictions should also be
per-image min-max renormalised before scoring. This measures it.

Model under test: parameter-free bicubic x2 upsample of the LR input. Parameter-free
matters -- nothing is fitted, so the comparison isolates the post-processing choice.

Variants:
  V1 raw    : prediction as produced, no post-processing
  V2 clip   : np.clip(pred, 0, 1)                      <- conservative default
  V3 renorm : (pred - pred.min()) / (pred.max() - pred.min())   -> exactly [0,1]

Validation split: the last 200 training pairs (indices 3000-3199), held out.

Usage:  py -3.12 scripts/renorm_experiment.py C:\\kla-data
"""

import argparse
import json
import os

import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_degradation import weight_matrix

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def psnr(a, b, data_range=1.0):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((data_range ** 2) / mse))


def ssim(a, b, data_range=1.0, sigma=1.5):
    """Standard Wang et al. SSIM with a Gaussian window."""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu_a = gaussian_filter(a, sigma, mode="reflect")
    mu_b = gaussian_filter(b, sigma, mode="reflect")
    maa = gaussian_filter(a * a, sigma, mode="reflect")
    mbb = gaussian_filter(b * b, sigma, mode="reflect")
    mab = gaussian_filter(a * b, sigma, mode="reflect")
    va = maa - mu_a * mu_a
    vb = mbb - mu_b * mu_b
    vab = mab - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * vab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)
    return float(np.mean(num / den))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"C:\kla-data")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    gt_dir = os.path.join(args.root, "train", "GT")
    lr_dir = os.path.join(args.root, "train", "NoisyLR")
    files = sorted(os.listdir(gt_dir))
    val = files[-args.n:]

    print("Renormalisation experiment")
    print("  validation split : last %d train pairs (%s .. %s), held out" % (len(val), val[0], val[-1]))
    print("  model            : bicubic x2 upsample, parameter-free")
    print()

    W = weight_matrix(128, 256, "cubic", antialias=False)

    rows = {"V1 raw": [], "V2 clip": [], "V3 renorm": []}
    ranges = []

    for f in val:
        g = np.load(os.path.join(gt_dir, f), allow_pickle=False).astype(np.float64)
        l = np.load(os.path.join(lr_dir, f), allow_pickle=False).astype(np.float64)
        pred = (W @ l) @ W.T

        v1 = pred
        v2 = np.clip(pred, 0.0, 1.0)
        lo, hi = pred.min(), pred.max()
        v3 = (pred - lo) / (hi - lo) if hi > lo else np.zeros_like(pred)
        ranges.append((lo, hi))

        for name, arr in (("V1 raw", v1), ("V2 clip", v2), ("V3 renorm", v3)):
            rows[name].append((psnr(arr, g), ssim(arr, g)))

    print("=" * 78)
    print("RESULTS  (n=%d validation pairs)" % len(val))
    print("=" * 78)
    print()
    print("%-12s %18s %18s" % ("variant", "PSNR dB (mean+-sd)", "SSIM (mean+-sd)"))
    print("-" * 52)
    stats = {}
    for name in ("V1 raw", "V2 clip", "V3 renorm"):
        arr = np.array(rows[name])
        p, s = arr[:, 0], arr[:, 1]
        stats[name] = {"psnr_mean": float(p.mean()), "psnr_sd": float(p.std()),
                       "ssim_mean": float(s.mean()), "ssim_sd": float(s.std())}
        print("%-12s   %8.4f +- %-6.4f   %8.5f +- %-7.5f" % (name, p.mean(), p.std(), s.mean(), s.std()))

    p1 = np.array(rows["V1 raw"])[:, 0]
    p2 = np.array(rows["V2 clip"])[:, 0]
    p3 = np.array(rows["V3 renorm"])[:, 0]
    s2 = np.array(rows["V2 clip"])[:, 1]
    s3 = np.array(rows["V3 renorm"])[:, 1]

    print()
    print("PAIRWISE, per image:")
    print("  clip   vs raw    : dPSNR mean %+0.4f dB   clip wins %d/%d" % ((p2 - p1).mean(), int((p2 > p1).sum()), len(val)))
    print("  renorm vs clip   : dPSNR mean %+0.4f dB   renorm wins %d/%d" % ((p3 - p2).mean(), int((p3 > p2).sum()), len(val)))
    print("  renorm vs clip   : dSSIM mean %+0.6f      renorm wins %d/%d" % ((s3 - s2).mean(), int((s3 > s2).sum()), len(val)))
    print()
    print("  renorm vs clip dPSNR: min %+0.4f  p5 %+0.4f  median %+0.4f  p95 %+0.4f  max %+0.4f"
          % ((p3 - p2).min(), np.percentile(p3 - p2, 5), np.median(p3 - p2),
             np.percentile(p3 - p2, 95), (p3 - p2).max()))

    r = np.array(ranges)
    print()
    print("Raw prediction range before post-processing:")
    print("  min : mean %+0.4f  worst %+0.4f   |  frac images with min < 0 : %.3f" % (r[:, 0].mean(), r[:, 0].min(), (r[:, 0] < 0).mean()))
    print("  max : mean %+0.4f  worst %+0.4f   |  frac images with max > 1 : %.3f" % (r[:, 1].mean(), r[:, 1].max(), (r[:, 1] > 1).mean()))

    best = max(stats, key=lambda k: stats[k]["psnr_mean"])
    print()
    print("BEST BY PSNR: %s" % best)
    print("BEST BY SSIM: %s" % max(stats, key=lambda k: stats[k]["ssim_mean"]))

    out = os.path.join(REPO_ROOT, "results", "eda")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "renorm_experiment.json"), "w") as fh:
        json.dump({"n": len(val), "split": [val[0], val[-1]], "stats": stats,
                   "renorm_minus_clip_psnr_mean": float((p3 - p2).mean()),
                   "renorm_wins_psnr": int((p3 > p2).sum()),
                   "clip_minus_raw_psnr_mean": float((p2 - p1).mean())}, fh, indent=2)
    print()
    print("saved %s" % os.path.abspath(os.path.join(out, "renorm_experiment.json")))


if __name__ == "__main__":
    main()
