"""SPEC 5.4 visual audit -- mandatory, not optional.

Saves a grid of 12 triplets [NoisyLR nearest-upscaled 2x | GT | |difference|] to
results/eda/pairs_grid.png, plus an aliasing screen that ranks pairs by how much
high-frequency GT energy sits above the LR Nyquist limit. Aliased dense periodic
structure is genuinely unrecoverable and SPEC 14 (Slide 6) requires one honest
failure case -- this finds a candidate.

Usage: py -3.12 scripts/visual_audit.py C:\\kla-data
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def nearest_up2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def hf_energy_ratio(gt):
    """Fraction of GT spectral energy above the LR Nyquist limit.

    GT is 2x the LR grid, so anything beyond half the GT Nyquist cannot be
    represented in the LR image and will alias on decimation.
    """
    F = np.fft.fftshift(np.abs(np.fft.fft2(gt - gt.mean())) ** 2)
    h, w = F.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)
    total = F.sum()
    return float(F[r > 0.5].sum() / total) if total > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"C:\kla-data")
    ap.add_argument("--scan", type=int, default=400, help="pairs to screen for aliasing")
    args = ap.parse_args()

    gt_dir = os.path.join(args.root, "train", "GT")
    lr_dir = os.path.join(args.root, "train", "NoisyLR")
    files = sorted(os.listdir(gt_dir))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- aliasing screen -------------------------------------------------
    step = max(1, len(files) // args.scan)
    scan = files[::step][: args.scan]
    ratios = []
    for f in scan:
        g = np.load(os.path.join(gt_dir, f), allow_pickle=False).astype(np.float64)
        ratios.append((hf_energy_ratio(g), f))
    ratios.sort(reverse=True)

    print("SPEC 5.4 visual audit")
    print("  screened %d pairs for above-Nyquist GT energy" % len(scan))
    print()
    print("  WORST 8 (most at risk of unrecoverable aliasing at 2x decimation):")
    for r, f in ratios[:8]:
        print("    %-14s HF energy above LR Nyquist = %.4f" % (f, r))
    print()
    print("  BEST 5 (safest):")
    for r, f in ratios[-5:]:
        print("    %-14s HF energy above LR Nyquist = %.4f" % (f, r))
    rr = np.array([r for r, _ in ratios])
    print()
    print("  distribution: mean=%.4f sd=%.4f min=%.4f median=%.4f max=%.4f"
          % (rr.mean(), rr.std(), rr.min(), np.median(rr), rr.max()))

    # ---- 12 triplets: 9 spread across the range + the 3 worst ------------
    spread = [f for _, f in ratios[::max(1, len(ratios) // 9)]][:9]
    worst = [f for _, f in ratios[:3]]
    show = list(dict.fromkeys(spread + worst))[:12]

    fig, axes = plt.subplots(len(show), 3, figsize=(9.6, 3.2 * len(show)))
    for row, f in enumerate(show):
        g = np.load(os.path.join(gt_dir, f), allow_pickle=False).astype(np.float64)
        l = np.load(os.path.join(lr_dir, f), allow_pickle=False).astype(np.float64)
        up = nearest_up2(l)
        diff = np.abs(up - g)
        hf = dict((ff, r) for r, ff in ratios)[f]

        for col, (img, title) in enumerate((
            (up, "NoisyLR x2 nearest"),
            (g, "GT"),
            (diff, "|difference|"),
        )):
            ax = axes[row, col]
            vmin, vmax = (0, 1) if col < 2 else (0, float(diff.max()))
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=10)
        axes[row, 0].set_ylabel("%s\nHF=%.3f" % (f, hf), fontsize=7)

    fig.suptitle("SPEC 5.4 visual audit -- 12 pairs, last 3 rows are worst-aliasing candidates",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    out = os.path.join(REPO_ROOT, "results", "eda")
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, "pairs_grid.png")
    fig.savefig(png, dpi=110)
    print()
    print("saved %s" % png)

    # ---- aliasing demonstration on the single worst case -----------------
    fw = ratios[0][1]
    g = np.load(os.path.join(gt_dir, fw), allow_pickle=False).astype(np.float64)
    l = np.load(os.path.join(lr_dir, fw), allow_pickle=False).astype(np.float64)
    fig2, ax2 = plt.subplots(1, 4, figsize=(14, 3.8))
    ax2[0].imshow(g, cmap="gray", vmin=0, vmax=1); ax2[0].set_title("GT %s" % fw)
    ax2[1].imshow(nearest_up2(l), cmap="gray", vmin=0, vmax=1); ax2[1].set_title("NoisyLR x2")
    S = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean()))))
    ax2[2].imshow(S, cmap="magma"); ax2[2].set_title("log|FFT(GT)|")
    h = S.shape[0]
    ax2[2].add_patch(plt.Rectangle((h * 0.25, h * 0.25), h * 0.5, h * 0.5,
                                   fill=False, ec="cyan", lw=1.2))
    S2 = np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(l - l.mean()))))
    ax2[3].imshow(S2, cmap="magma"); ax2[3].set_title("log|FFT(NoisyLR)|")
    for a in ax2:
        a.set_xticks([]); a.set_yticks([])
    fig2.suptitle("Worst-case aliasing candidate: GT energy outside the cyan box cannot survive 2x decimation")
    fig2.tight_layout()
    png2 = os.path.join(out, "aliasing_failure_case.png")
    fig2.savefig(png2, dpi=130)
    print("saved %s" % png2)
    print()
    print("failure-case candidate for SPEC 14 Slide 6: %s (HF above Nyquist = %.4f)"
          % (fw, ratios[0][0]))


if __name__ == "__main__":
    main()
