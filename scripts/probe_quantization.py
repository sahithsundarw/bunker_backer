"""Follow-up probes: value quantisation grid, clipping, train-vs-test distribution.

python scripts/probe_quantization.py C:\\kla-data
"""

import os
import sys

import numpy as np


def grid_test(a, levels):
    """Fraction of values that sit on a k/levels grid (within float32 eps)."""
    x = a.astype(np.float64) * levels
    return float(np.mean(np.abs(x - np.round(x)) < 1e-3))


def folder_stats(d, files, n=300):
    step = max(1, len(files) // n)
    sample = files[::step][:n]
    mins, maxs, means, stds = [], [], [], []
    below0 = above1 = 0
    tot = 0
    for f in sample:
        a = np.load(os.path.join(d, f), allow_pickle=False)
        mins.append(a.min())
        maxs.append(a.max())
        means.append(a.mean())
        stds.append(a.std())
        below0 += int((a < 0).sum())
        above1 += int((a > 1).sum())
        tot += a.size
    return dict(
        n=len(sample),
        min=float(np.min(mins)),
        max=float(np.max(maxs)),
        mean=float(np.mean(means)),
        std=float(np.mean(stds)),
        frac_below_0=below0 / tot,
        frac_above_1=above1 / tot,
    )


def main(root):
    gt = os.path.join(root, "train", "GT")
    lr = os.path.join(root, "train", "NoisyLR")
    te = os.path.join(root, "test_NoisyLR")

    print("=" * 78)
    print("A. QUANTISATION GRID TEST  (fraction of values on a k/L grid)")
    print("=" * 78)
    for label, d in (("GT", gt), ("NoisyLR", lr), ("test_NoisyLR", te)):
        files = sorted(os.listdir(d))[:5]
        print()
        print("--- %s ---" % label)
        for f in files:
            a = np.load(os.path.join(d, f), allow_pickle=False)
            print(
                "  %-14s /255=%.4f  /1023=%.4f  /4095=%.4f  /65535=%.4f"
                % (f, grid_test(a, 255), grid_test(a, 1023), grid_test(a, 4095), grid_test(a, 65535))
            )

    print()
    print("=" * 78)
    print("B. RANGE / CLIPPING  (sampled n=300 per folder)")
    print("=" * 78)
    for label, d in (("GT", gt), ("NoisyLR", lr), ("test_NoisyLR", te)):
        files = sorted(os.listdir(d))
        s = folder_stats(d, files)
        print()
        print("%s (n=%d)" % (label, s["n"]))
        print("  min=%.6f  max=%.6f" % (s["min"], s["max"]))
        print("  mean=%.6f  std=%.6f" % (s["mean"], s["std"]))
        print("  frac pixels < 0 : %.6f" % s["frac_below_0"])
        print("  frac pixels > 1 : %.6f" % s["frac_above_1"])

    print()
    print("=" * 78)
    print("C. TRAIN-LR vs TEST-LR DISTRIBUTION (per-image mean, n=400 each)")
    print("=" * 78)
    for label, d in (("train/NoisyLR", lr), ("test_NoisyLR", te)):
        files = sorted(os.listdir(d))
        step = max(1, len(files) // 400)
        sample = files[::step][:400]
        m = np.array([np.load(os.path.join(d, f), allow_pickle=False).mean() for f in sample])
        s = np.array([np.load(os.path.join(d, f), allow_pickle=False).std() for f in sample])
        print()
        print("%s (n=%d)" % (label, len(sample)))
        print("  per-image mean : mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (m.mean(), m.std(), m.min(), m.max()))
        print("  per-image std  : mean=%.4f  std=%.4f  min=%.4f  max=%.4f" % (s.mean(), s.std(), s.min(), s.max()))

    print()
    print("=" * 78)
    print("D. FILENAME NAMESPACE COLLISION CHECK")
    print("=" * 78)
    gtf = set(os.listdir(gt))
    tef = set(os.listdir(te))
    inter = sorted(gtf & tef)
    print("names shared between train/GT and test_NoisyLR : %d" % len(inter))
    print("example shared names: %s" % inter[:5])
    if inter:
        f = inter[0]
        a = np.load(os.path.join(lr, f), allow_pickle=False)
        b = np.load(os.path.join(te, f), allow_pickle=False)
        print()
        print("Are they the SAME image? compare train/NoisyLR/%s vs test_NoisyLR/%s" % (f, f))
        print("  shapes      : %s vs %s" % (a.shape, b.shape))
        print("  array_equal : %s" % np.array_equal(a, b))
        print("  mean        : %.6f vs %.6f" % (a.mean(), b.mean()))
        print("  => identical filenames refer to DIFFERENT images." if not np.array_equal(a, b) else "  => identical content.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "C:\\kla-data"))
