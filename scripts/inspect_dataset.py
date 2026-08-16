"""Dataset inspector for the KLA image-restoration data.

Substitute for the missing peek.py (not found anywhere on this machine).
Same CLI shape:  python scripts/inspect_dataset.py C:\\kla-data

Emits numeric evidence only. No inference, no guesses.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

from dataset_paths import default_dataset_root, resolve_test_input_dir

GT_DIR = ("train", "GT")
LR_DIR = ("train", "NoisyLR")


def hr(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def listdir_sorted(path):
    return sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))


def load_header(path):
    """Read shape/dtype without pulling the whole array into memory."""
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    return a.shape, a.dtype


def describe_array(path):
    a = np.load(path, allow_pickle=False)
    return {
        "shape": a.shape,
        "dtype": str(a.dtype),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "n_unique": int(np.unique(a).size),
        "C_contiguous": a.flags["C_CONTIGUOUS"],
        "nbytes": int(a.nbytes),
    }


def block_mean_2x(img):
    """Average-pool a 2D array by 2x2. Requires even dims."""
    h, w = img.shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    x = img[:h2, :w2].astype(np.float64)
    return x.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))


def ncc(a, b):
    """Normalised cross-correlation (Pearson) between two equal-shape arrays."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    da, db = np.sqrt((a * a).sum()), np.sqrt((b * b).sum())
    if da == 0 or db == 0:
        return float("nan")
    return float((a * b).sum() / (da * db))


def shift_scan(ref, mov, radius=3):
    """Correlate ref against mov over integer shifts. Returns (best_dy,dx,corr, corr_at_0)."""
    best = (None, None, -2.0)
    zero = float("nan")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r = ref[
                max(0, dy) : ref.shape[0] + min(0, dy),
                max(0, dx) : ref.shape[1] + min(0, dx),
            ]
            m = mov[
                max(0, -dy) : mov.shape[0] + min(0, -dy),
                max(0, -dx) : mov.shape[1] + min(0, -dx),
            ]
            c = ncc(r, m)
            if dy == 0 and dx == 0:
                zero = c
            if c > best[2]:
                best = (dy, dx, c)
    return best[0], best[1], best[2], zero


def main(root):
    root = os.path.abspath(root)
    gt_dir = os.path.join(root, *GT_DIR)
    lr_dir = os.path.join(root, *LR_DIR)
    try:
        test_dir = str(resolve_test_input_dir(root))
    except FileNotFoundError as exc:
        print(f"FATAL: {exc}")
        return 1

    for p in (gt_dir, lr_dir, test_dir):
        if not os.path.isdir(p):
            print("FATAL: missing directory %s" % p)
            return 1

    hr("0. ROOT + FOLDER INVENTORY")
    print("root = %s" % root)
    for label, p in (("train/GT", gt_dir), ("train/NoisyLR", lr_dir), ("test_NoisyLR", test_dir)):
        files = listdir_sorted(p)
        exts = Counter(os.path.splitext(f)[1] for f in files)
        print("%-16s n_files=%-6d extensions=%s" % (label, len(files), dict(exts)))

    gt_files = listdir_sorted(gt_dir)
    lr_files = listdir_sorted(lr_dir)
    test_files = listdir_sorted(test_dir)

    # ---------------------------------------------------------------- U2
    hr("U2. FILENAME PAIRING RULE")
    print("first 5 GT      : %s" % gt_files[:5])
    print("first 5 NoisyLR : %s" % lr_files[:5])
    print("first 5 test    : %s" % test_files[:5])
    gt_set, lr_set = set(gt_files), set(lr_files)
    print()
    print("identical-name test:")
    print("  |GT|                = %d" % len(gt_set))
    print("  |NoisyLR|           = %d" % len(lr_set))
    print("  |GT & NoisyLR|      = %d" % len(gt_set & lr_set))
    print("  GT-only  (first 5)  = %s" % sorted(gt_set - lr_set)[:5])
    print("  LR-only  (first 5)  = %s" % sorted(lr_set - gt_set)[:5])
    print("  exact name match    = %s" % (gt_set == lr_set))
    print()
    print("test-vs-train name overlap:")
    print("  |test & GT|         = %d" % len(set(test_files) & gt_set))
    print("  |test & NoisyLR|    = %d" % len(set(test_files) & lr_set))

    # ---------------------------------------------------------------- U1
    hr("U1. FORMAT / DTYPE / VALUE RANGE")
    print("container format: .npy  (NumPy binary; np.load, allow_pickle=False)")
    for label, d, files in (
        ("GT", gt_dir, gt_files),
        ("NoisyLR", lr_dir, lr_files),
        ("test_NoisyLR", test_dir, test_files),
    ):
        print()
        print("--- %s : per-file detail, first 3 files ---" % label)
        for f in files[:3]:
            info = describe_array(os.path.join(d, f))
            print(
                "  %-28s shape=%-14s dtype=%-8s min=%-8.3f max=%-9.3f mean=%-9.3f std=%-8.3f n_unique=%d"
                % (
                    f,
                    str(info["shape"]),
                    info["dtype"],
                    info["min"],
                    info["max"],
                    info["mean"],
                    info["std"],
                    info["n_unique"],
                )
            )

    # aggregate dtype / range / unique over a sample
    hr("U1b. AGGREGATE STATS OVER SAMPLE (n=200 per folder, deterministic stride)")
    for label, d, files in (
        ("GT", gt_dir, gt_files),
        ("NoisyLR", lr_dir, lr_files),
        ("test_NoisyLR", test_dir, test_files),
    ):
        step = max(1, len(files) // 200)
        sample = files[::step][:200]
        dts, mins, maxs, uniq, ndim = Counter(), [], [], [], Counter()
        for f in sample:
            a = np.load(os.path.join(d, f), allow_pickle=False)
            dts[str(a.dtype)] += 1
            mins.append(float(a.min()))
            maxs.append(float(a.max()))
            uniq.append(int(np.unique(a).size))
            ndim[a.ndim] += 1
        print()
        print("%s  (n_sampled=%d)" % (label, len(sample)))
        print("  dtypes            : %s" % dict(dts))
        print("  ndim              : %s" % dict(ndim))
        print("  global min        : %.6f" % min(mins))
        print("  global max        : %.6f" % max(maxs))
        print("  n_unique  min/max : %d / %d" % (min(uniq), max(uniq)))
        print("  n_unique <= 256   : %s  (True => 8-bit-valued)" % (max(uniq) <= 256))

    # ---------------------------------------------------------------- U3
    hr("U3. PAIR COUNT + RESOLUTION SPLIT (full scan, all files)")
    gt_shapes, lr_shapes, test_shapes = Counter(), Counter(), Counter()
    gt_shape_map = {}
    for f in gt_files:
        s, _ = load_header(os.path.join(gt_dir, f))
        gt_shapes[s] += 1
        gt_shape_map[f] = s
    lr_shape_map = {}
    for f in lr_files:
        s, _ = load_header(os.path.join(lr_dir, f))
        lr_shapes[s] += 1
        lr_shape_map[f] = s
    for f in test_files:
        s, _ = load_header(os.path.join(test_dir, f))
        test_shapes[s] += 1

    print("GT shape histogram           : %s" % dict(gt_shapes))
    print("NoisyLR shape histogram      : %s" % dict(lr_shapes))
    print("test_NoisyLR shape histogram : %s" % dict(test_shapes))
    print()
    n512 = sum(v for k, v in gt_shapes.items() if k[0] == 512)
    n256 = sum(v for k, v in gt_shapes.items() if k[0] == 256)
    print("GT with first-dim 512 : %d" % n512)
    print("GT with first-dim 256 : %d" % n256)
    print("GT other first-dims   : %d" % (len(gt_files) - n512 - n256))

    # ------------------------------------------------- 2x invariant, ALL pairs
    hr("2x INVARIANT: GT dims == 2 * LR dims, checked on EVERY pair")
    paired = sorted(gt_set & lr_set)
    print("pairs checked = %d" % len(paired))
    violations = []
    ratio_hist = Counter()
    for f in paired:
        g, l = gt_shape_map[f], lr_shape_map[f]
        ok = len(g) >= 2 and len(l) >= 2 and g[0] == 2 * l[0] and g[1] == 2 * l[1]
        ratio_hist[(g[0], l[0], g[1], l[1])] += 1
        if not ok:
            violations.append((f, g, l))
    print("(GT_h, LR_h, GT_w, LR_w) histogram : %s" % dict(ratio_hist))
    print("violations = %d" % len(violations))
    for v in violations[:20]:
        print("  VIOLATION %s GT=%s LR=%s" % v)
    if not violations:
        print("RESULT: every pair satisfies GT == 2x LR in both spatial dims.")

    # ---------------------------------------------------------------- U8
    hr("U8. PIXEL ALIGNMENT (GT block-mean-2x vs LR, integer shift scan +/-3)")
    print("Interpretation: best shift (0,0) => GT and LR are pixel-aligned.")
    print()
    step = max(1, len(paired) // 12)
    sample = paired[::step][:12]
    zero_wins = 0
    for f in sample:
        g = np.load(os.path.join(gt_dir, f), allow_pickle=False).astype(np.float64)
        l = np.load(os.path.join(lr_dir, f), allow_pickle=False).astype(np.float64)
        if g.ndim == 3:
            g = g.mean(axis=2)
        if l.ndim == 3:
            l = l.mean(axis=2)
        gd = block_mean_2x(g)
        h = min(gd.shape[0], l.shape[0])
        w = min(gd.shape[1], l.shape[1])
        dy, dx, best, zero = shift_scan(gd[:h, :w], l[:h, :w], radius=3)
        if (dy, dx) == (0, 0):
            zero_wins += 1
        print(
            "  %-28s best_shift=(%+d,%+d) corr_best=%.4f corr_at_(0,0)=%.4f"
            % (f, dy, dx, best, zero)
        )
    print()
    print("best shift == (0,0) for %d / %d sampled pairs" % (zero_wins, len(sample)))

    # ---------------------------------------------------------------- U9
    hr("U9. TEST INPUTS")
    print("test_NoisyLR present at : %s" % test_dir)
    print("n_files                 : %d" % len(test_files))
    print("GT for test set present : %s" % os.path.isdir(os.path.join(root, "test_GT")))

    hr("DONE")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", nargs="?", default=str(default_dataset_root()))
    cli = parser.parse_args()
    sys.exit(main(cli.root))
