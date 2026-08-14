"""Content audit: what IS this imagery?

SPEC 1 / F8 describe semiconductor inspection images (line/space arrays, contact
holes, dense periodic arrays). The 5.4 visual audit showed natural photographs.
This builds large contact sheets of train GT and of the test inputs so the content
domain can be judged from a big sample rather than 12 rows, and reports simple
statistics that separate natural imagery from SEM-style inspection imagery.

Usage: py -3.12 scripts/content_audit.py C:\\kla-data
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def contact_sheet(paths, cols, title, out_png, upscale=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.75))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.ravel()):
        ax.set_xticks([]); ax.set_yticks([])
        if k >= len(paths):
            ax.axis("off"); continue
        a = np.load(paths[k], allow_pickle=False).astype(np.float64)
        if upscale:
            a = np.repeat(np.repeat(a, 2, 0), 2, 1)
        ax.imshow(np.clip(a, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(os.path.basename(paths[k]), fontsize=5, pad=1.5)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_png, dpi=115)
    plt.close(fig)
    print("saved %s" % out_png)


def stats(paths, label, upscale=False):
    """Statistics that separate natural photos from SEM/inspection imagery."""
    sat_lo = sat_hi = 0
    tot = 0
    grads, ents, means = [], [], []
    for p in paths:
        a = np.load(p, allow_pickle=False).astype(np.float64)
        a = np.clip(a, 0, 1)
        sat_lo += int((a < 0.02).sum())
        sat_hi += int((a > 0.98).sum())
        tot += a.size
        gy, gx = np.gradient(a)
        grads.append(float(np.sqrt(gy ** 2 + gx ** 2).mean()))
        h, _ = np.histogram(a, bins=64, range=(0, 1), density=False)
        p_ = h / h.sum()
        p_ = p_[p_ > 0]
        ents.append(float(-(p_ * np.log2(p_)).sum()))
        means.append(float(a.mean()))
    print()
    print("%s  (n=%d)" % (label, len(paths)))
    print("  mean intensity          : %.4f  (sd %.4f)" % (np.mean(means), np.std(means)))
    print("  mean gradient magnitude : %.5f" % np.mean(grads))
    print("  histogram entropy /6 bit: %.3f of 6.000 max" % np.mean(ents))
    print("  frac pixels < 0.02      : %.4f" % (sat_lo / tot))
    print("  frac pixels > 0.98      : %.4f" % (sat_hi / tot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"C:\kla-data")
    args = ap.parse_args()

    gt_dir = os.path.join(args.root, "train", "GT")
    te_dir = os.path.join(args.root, "test_NoisyLR")
    out = os.path.join(REPO_ROOT, "results", "eda")
    os.makedirs(out, exist_ok=True)

    gt_files = sorted(os.listdir(gt_dir))
    te_files = sorted(os.listdir(te_dir))

    gt_sel = [os.path.join(gt_dir, gt_files[i]) for i in range(0, len(gt_files), max(1, len(gt_files) // 48))][:48]
    te_sel = [os.path.join(te_dir, te_files[i]) for i in range(0, len(te_files), max(1, len(te_files) // 48))][:48]

    print("Content audit -- is this semiconductor inspection imagery?")
    contact_sheet(gt_sel, 8, "train/GT  -- 48 samples spread across all 3200",
                  os.path.join(out, "content_train_gt.png"))
    contact_sheet(te_sel, 8, "test_NoisyLR -- 48 samples spread across all 400 (nearest x2 for display)",
                  os.path.join(out, "content_test_inputs.png"), upscale=True)

    stats(gt_sel, "train/GT")
    stats(te_sel, "test_NoisyLR")

    print()
    print("Reference points for interpretation:")
    print("  SEM / inspection imagery is typically dominated by a few discrete grey")
    print("  levels (substrate vs feature), giving LOW histogram entropy and strong")
    print("  bimodality. Natural photographs give high entropy and a broad histogram.")


if __name__ == "__main__":
    main()
