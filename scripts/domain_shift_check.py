"""Quantify train-vs-test content shift, comparing LIKE WITH LIKE.

train/NoisyLR and test_NoisyLR are both noisy 128x128 inputs, so any difference is
content, not degradation. Measures spectral peakiness -- natural scenes have smooth
~1/f spectra; periodic man-made structure (facades, tiled floors, gratings) puts
sharp isolated peaks in the spectrum.

Usage: py -3.12 scripts/domain_shift_check.py C:\\kla-data
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def spectral_peakiness(a):
    """Ratio of the strongest off-DC spectral peak to the median off-DC magnitude.

    High => energy concentrated in a few discrete frequencies => periodic structure.
    """
    A = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
    h, w = A.shape
    cy, cx = h // 2, w // 2
    A[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0  # kill DC neighbourhood
    v = A[A > 0]
    if v.size == 0:
        return 0.0
    return float(v.max() / np.median(v))


def anisotropy(a):
    """Ratio of directional gradient energies; gratings/facades are anisotropic."""
    gy, gx = np.gradient(a)
    ey, ex = float((gy ** 2).mean()), float((gx ** 2).mean())
    return float(max(ey, ex) / max(min(ey, ex), 1e-12))


def describe(d, files, label, n=400):
    step = max(1, len(files) // n)
    sel = files[::step][:n]
    peaks, anis, grads = [], [], []
    for f in sel:
        a = np.load(os.path.join(d, f), allow_pickle=False).astype(np.float64)
        peaks.append(spectral_peakiness(a))
        anis.append(anisotropy(a))
        gy, gx = np.gradient(a)
        grads.append(float(np.sqrt(gy ** 2 + gx ** 2).mean()))
    peaks = np.array(peaks); anis = np.array(anis); grads = np.array(grads)
    print()
    print("%s  (n=%d)" % (label, len(sel)))
    print("  spectral peakiness : mean=%8.2f  median=%8.2f  p90=%8.2f  max=%9.2f"
          % (peaks.mean(), np.median(peaks), np.percentile(peaks, 90), peaks.max()))
    print("  gradient anisotropy: mean=%8.3f  median=%8.3f  p90=%8.3f"
          % (anis.mean(), np.median(anis), np.percentile(anis, 90)))
    print("  gradient magnitude : mean=%8.5f  median=%8.5f" % (grads.mean(), np.median(grads)))
    return peaks, anis, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"C:\kla-data")
    args = ap.parse_args()

    lr = os.path.join(args.root, "train", "NoisyLR")
    te = os.path.join(args.root, "test_NoisyLR")

    print("Train-vs-test content shift (both are noisy 128x128 inputs -- like for like)")
    p_tr, a_tr, g_tr = describe(lr, sorted(os.listdir(lr)), "train/NoisyLR")
    p_te, a_te, g_te = describe(te, sorted(os.listdir(te)), "test_NoisyLR")

    print()
    print("=" * 78)
    print("SHIFT")
    print("=" * 78)
    print("  spectral peakiness  median  train %8.2f  ->  test %8.2f   (x%.2f)"
          % (np.median(p_tr), np.median(p_te), np.median(p_te) / np.median(p_tr)))
    print("  spectral peakiness  p90     train %8.2f  ->  test %8.2f   (x%.2f)"
          % (np.percentile(p_tr, 90), np.percentile(p_te, 90),
             np.percentile(p_te, 90) / np.percentile(p_tr, 90)))
    print("  gradient anisotropy median  train %8.3f  ->  test %8.3f   (x%.2f)"
          % (np.median(a_tr), np.median(a_te), np.median(a_te) / np.median(a_tr)))

    # fraction of images that look strongly periodic
    thr = np.percentile(p_tr, 90)
    print()
    print("  Using train p90 peakiness (%.2f) as a 'strongly periodic' threshold:" % thr)
    print("    train images above it : %.1f%%  (10.0%% by construction)" % (100.0 * (p_tr > thr).mean()))
    print("    test  images above it : %.1f%%" % (100.0 * (p_te > thr).mean()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, np.percentile(np.concatenate([p_tr, p_te]), 99), 50)
    ax[0].hist(p_tr, bins=bins, alpha=0.6, density=True, label="train/NoisyLR")
    ax[0].hist(p_te, bins=bins, alpha=0.6, density=True, label="test_NoisyLR")
    ax[0].set_xlabel("spectral peakiness (max off-DC / median)")
    ax[0].set_ylabel("density"); ax[0].legend(); ax[0].grid(alpha=0.25)
    ax[0].set_title("Periodic-structure content")
    bins2 = np.linspace(0, np.percentile(np.concatenate([a_tr, a_te]), 99), 50)
    ax[1].hist(a_tr, bins=bins2, alpha=0.6, density=True, label="train/NoisyLR")
    ax[1].hist(a_te, bins=bins2, alpha=0.6, density=True, label="test_NoisyLR")
    ax[1].set_xlabel("gradient anisotropy"); ax[1].legend(); ax[1].grid(alpha=0.25)
    ax[1].set_title("Directional structure")
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "results", "eda", "domain_shift.png")
    fig.savefig(out, dpi=130)
    print()
    print("saved %s" % out)


if __name__ == "__main__":
    main()
