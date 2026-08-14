"""Time-boxed attempt to identify the source dataset.

Hypothesis under test: 3200 = 800 x 4 and 400 = 100 x 4, i.e. K crops taken from each
source photograph, laid out consecutively. DIV2K has exactly 800 train and 100 validation
images, which would fit K=4.

If crops are consecutive, images 4k..4k+3 share a source photo and should be markedly more
similar to each other than to crops from other photos. Scanning K = 1..12 shows whether any
grouping is special, and which.

Usage: py -3.12 scripts/source_id.py C:\\kla-data
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def npy_header(path):
    with open(path, "rb") as fh:
        magic = fh.read(6)
        ver = fh.read(2)
        hlen = int.from_bytes(fh.read(2), "little")
        hdr = fh.read(hlen)
    return magic, tuple(ver), hlen, hdr


def feats(d, files):
    """Per-image descriptor: intensity histogram + simple texture stats."""
    H = []
    for f in files:
        a = np.load(os.path.join(d, f), allow_pickle=False).astype(np.float64)
        h, _ = np.histogram(np.clip(a, 0, 1), bins=32, range=(0, 1), density=True)
        gy, gx = np.gradient(a)
        g = float(np.sqrt(gy ** 2 + gx ** 2).mean())
        H.append(np.concatenate([h / (h.sum() + 1e-12), [g * 10, a.mean(), a.std()]]))
    return np.array(H)


def group_score(F, K):
    """Mean within-group distance / mean between-group distance, for consecutive groups of K.

    < 1 means consecutive runs of K are more alike than chance -- evidence of K crops
    per source image. Lower is stronger.
    """
    n = (len(F) // K) * K
    if n < 2 * K or K < 2:
        return float("nan")
    X = F[:n]
    G = np.arange(n) // K
    # pairwise L1 distances
    D = np.abs(X[:, None, :] - X[None, :, :]).sum(axis=2)
    same = G[:, None] == G[None, :]
    eye = np.eye(n, dtype=bool)
    within = D[same & ~eye]
    between = D[~same]
    if within.size == 0 or between.size == 0:
        return float("nan")
    return float(within.mean() / between.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=r"C:\kla-data")
    ap.add_argument("--n", type=int, default=480)
    args = ap.parse_args()

    gt = os.path.join(args.root, "train", "GT")
    te = os.path.join(args.root, "test_NoisyLR")

    print("=" * 78)
    print("A. .npy HEADER -- any embedded metadata?")
    print("=" * 78)
    for label, d in (("train/GT", gt), ("test_NoisyLR", te)):
        f = sorted(os.listdir(d))[0]
        magic, ver, hlen, hdr = npy_header(os.path.join(d, f))
        print("  %-14s %s  magic=%r ver=%s hlen=%d" % (label, f, magic, ver, hlen))
        print("                 header=%r" % hdr.strip())
    print()
    print("  .npy headers carry only descr/fortran_order/shape. No EXIF, no author, no")
    print("  original filename. Nothing identifying survives the conversion.")

    print()
    print("=" * 78)
    print("B. CONSECUTIVE-CROP GROUPING  (first %d train GT images)" % args.n)
    print("=" * 78)
    files = sorted(os.listdir(gt))[: args.n]
    F = feats(gt, files)
    print()
    print("  ratio = mean(within-group dist) / mean(between-group dist)")
    print("  ~1.00 => no grouping at that K.  Clearly <1 => K crops per source image.")
    print()
    print("  %4s %10s" % ("K", "ratio"))
    best = None
    for K in range(2, 13):
        r = group_score(F, K)
        flag = ""
        if np.isfinite(r):
            if best is None or r < best[1]:
                best = (K, r)
        print("  %4d %10.4f%s" % (K, r, flag))
    print()
    if best is not None:
        print("  strongest grouping: K=%d at ratio %.4f" % best)

    print()
    print("=" * 78)
    print("C. SAME TEST ON test_NoisyLR (first %d)" % min(args.n, 400))
    print("=" * 78)
    tfiles = sorted(os.listdir(te))[: min(args.n, 400)]
    FT = feats(te, tfiles)
    print()
    print("  %4s %10s" % ("K", "ratio"))
    bestt = None
    for K in range(2, 13):
        r = group_score(FT, K)
        if np.isfinite(r) and (bestt is None or r < bestt[1]):
            bestt = (K, r)
        print("  %4d %10.4f" % (K, r))
    print()
    if bestt is not None:
        print("  strongest grouping: K=%d at ratio %.4f" % bestt)

    print()
    print("=" * 78)
    print("D. ARITHMETIC AGAINST KNOWN CORPORA")
    print("=" * 78)
    print("  observed: 3200 train pairs, 400 test inputs, all GT 256x256")
    print()
    for name, ntr, nva in (
        ("DIV2K (800 train / 100 val)", 800, 100),
        ("Flickr2K (2650)", 2650, 0),
        ("DF2K = DIV2K800 + Flickr2K", 3450, 0),
        ("BSD500 (400 train+val / 100 test)", 400, 100),
        ("Waterloo Exploration (4744)", 4744, 0),
    ):
        s = ""
        if ntr and 3200 % ntr == 0:
            s += "  3200 = %d x %d" % (ntr, 3200 // ntr)
        if nva and 400 % nva == 0:
            s += "   400 = %d x %d" % (nva, 400 // nva)
        print("  %-36s%s" % (name, s if s else "  no integer fit"))


if __name__ == "__main__":
    main()
