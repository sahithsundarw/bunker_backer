#!/usr/bin/env python3
"""Generate the real-SEM OOD robustness set (Round 2 Phase 3, docs/decisions.md D53).

Mirrors results/eda/proxy_ood/'s generation method exactly (docs/dataset_findings.md
"Generation method"): degrade_fitted(gt, rng) only -- no randomisation -- and GT is
per-image min-max normalised to [0,1] to match the real-GT convention (U1).

Source: Zenodo record 17315241, "Scanning Electron Microscopy (SEM) Dataset of
Additively Manufactured Ni-WC Metal Matrix Composites for Semantic Segmentation",
CC-BY 4.0, https://zenodo.org/records/17315241 . Download `AugmentedImages.zip` from
that record and pass its extracted path via --sem_root.

One real SEM tile per unique underlying crop is used (the source ships ~9 augmented
variants per tile; using all of them would represent near-duplicate copies as if they
were independent samples). The HorizontalFlip variant is picked for each tile
deterministically -- a lossless mirror of real sensor pixels, not a synthetic warp,
unlike ElasticTransform/GridDistortion which are also present in the source.

Usage:
    py -3.12 scripts/gen_real_sem_ood.py --sem_root <path>/AugmentedImages --data_root C:\\kla-data

Owner: main session (Round 2 differentiation, not on the parallel-agent ownership map).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.degrade import degrade_fitted  # noqa: E402

OUT_ROOT = REPO_ROOT / "results" / "eda" / "real_sem_ood"
CROP_SIZE = 256
SEED_BASE = 20260817

# Non-overlapping crop-offset layouts, keyed by --crops_per_tile. 1 == today's exact
# behaviour (single centre crop, computed the same way as before -- not hardcoded here,
# see below). 4 == the four quadrants of a tile whose side is an exact multiple of
# 2*CROP_SIZE (verified per-tile at runtime, never assumed).
SUPPORTED_CROPS_PER_TILE = (1, 4)


def _crop_offsets(h: int, w: int, crops_per_tile: int) -> list[tuple[int, int]]:
    """Return a list of (row, col) top-left offsets for non-overlapping CROP_SIZE crops.

    crops_per_tile == 1: single centre crop -- byte-identical to the script's original
    (pre-extension) behaviour.
    crops_per_tile == 4: the four quadrants tiling the whole tile with zero overlap and
    zero gap, requiring h == w == 2 * CROP_SIZE exactly (checked, not assumed).
    """
    if crops_per_tile == 1:
        oy, ox = (h - CROP_SIZE) // 2, (w - CROP_SIZE) // 2
        return [(oy, ox)]
    if crops_per_tile == 4:
        if h != 2 * CROP_SIZE or w != 2 * CROP_SIZE:
            raise SystemExit(
                f"--crops_per_tile 4 requires each tile to be exactly "
                f"{2 * CROP_SIZE}x{2 * CROP_SIZE} so it tiles into 4 non-overlapping "
                f"{CROP_SIZE}x{CROP_SIZE} crops with no gap/overlap; got {h}x{w}."
            )
        return [(0, 0), (0, CROP_SIZE), (CROP_SIZE, 0), (CROP_SIZE, CROP_SIZE)]
    raise SystemExit(
        f"--crops_per_tile {crops_per_tile} is not supported (only "
        f"{SUPPORTED_CROPS_PER_TILE} have a defined non-overlapping tiling)."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sem_root", required=True,
                    help="path to the extracted AugmentedImages/ directory (Zenodo 17315241)")
    ap.add_argument("--data_root", default=None,
                    help="dataset root (contains train/GT, train/NoisyLR, test_NoisyLR) for "
                         "the disjointness check; skipped if not given")
    ap.add_argument("--out_root", default=str(OUT_ROOT))
    ap.add_argument("--crops_per_tile", type=int, default=1,
                    help="non-overlapping 256x256 crops per 512x512 source tile. Default 1 "
                         "(single centre crop) reproduces the existing 45-image set exactly. "
                         "4 takes the four quadrants of each tile (n=180 for this source), "
                         "using genuinely distinct real sensor pixels -- no synthetic "
                         "augmentation. See docs/decisions.md D53 and the follow-up entry.")
    args = ap.parse_args(argv)
    if args.crops_per_tile not in SUPPORTED_CROPS_PER_TILE:
        raise SystemExit(
            f"--crops_per_tile {args.crops_per_tile} not supported; choose from "
            f"{SUPPORTED_CROPS_PER_TILE}."
        )

    sem_root = Path(args.sem_root)
    out_root = Path(args.out_root)
    gt_dir = out_root / "GT"
    lr_dir = out_root / "NoisyLR"
    gt_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(sem_root.glob("*_HorizontalFlip.bmp"))
    if not files:
        raise SystemExit(f"no *_HorizontalFlip.bmp files found under {sem_root}")
    bases = sorted({re.match(r"(sem\d+_x\d+_y\d+)_", f.name).group(1) for f in files})
    print(f"unique base tiles with HorizontalFlip variant: {len(bases)}")
    print(f"crops_per_tile: {args.crops_per_tile}  -> expected n = "
          f"{len(bases) * args.crops_per_tile}")

    manifest = []
    membership = []
    idx = 0
    for base in bases:
        src = sem_root / f"{base}_HorizontalFlip.bmp"
        im = Image.open(src).convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        h, w = arr.shape
        if h < CROP_SIZE or w < CROP_SIZE:
            raise SystemExit(f"{src}: {arr.shape} smaller than {CROP_SIZE}x{CROP_SIZE}")
        offsets = _crop_offsets(h, w, args.crops_per_tile)

        for oy, ox in offsets:
            crop = arr[oy:oy + CROP_SIZE, ox:ox + CROP_SIZE]
            lo, hi = float(crop.min()), float(crop.max())
            gt = ((crop - lo) / (hi - lo) if hi > lo else crop).astype(np.float32)

            rng = np.random.default_rng([SEED_BASE, idx])
            lr = degrade_fitted(gt, rng).astype(np.float32)

            name = f"realsem_{idx:06d}.npy"
            np.save(gt_dir / name, gt)
            np.save(lr_dir / name, lr)
            manifest.append({
                "file": name, "source_tile": base, "source_file": src.name,
                "crop_offset_row": int(oy), "crop_offset_col": int(ox),
                "gt_min": float(gt.min()), "gt_max": float(gt.max()),
                "lr_min": float(lr.min()), "lr_max": float(lr.max()),
            })
            membership.append(name)
            idx += 1

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# real_sem_ood membership list -- one filename per line.",
             "# Source: Zenodo record 17315241, CC-BY 4.0. See docs/decisions.md D53.",
             *membership]
    (out_root / "membership_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    gt_mins = [m["gt_min"] for m in manifest]
    gt_maxs = [m["gt_max"] for m in manifest]
    lr_all = np.concatenate([np.load(lr_dir / m["file"]).ravel() for m in manifest])
    print(f"n = {len(manifest)}")
    print(f"gt_min==0.0 for {sum(1 for v in gt_mins if v == 0.0)}/{len(manifest)}")
    print(f"gt_max==1.0 for {sum(1 for v in gt_maxs if v == 1.0)}/{len(manifest)}")
    print(f"LR range: [{lr_all.min():.4f}, {lr_all.max():.4f}]")
    print(f"frac LR < 0: {(lr_all < 0).mean():.4f}   frac LR > 1: {(lr_all > 1).mean():.4f}")

    check: dict[str, object] = {"n_real_sem_ood": len(membership)}
    if args.data_root:
        kla = Path(args.data_root)
        train_gt = {p.name for p in (kla / "train" / "GT").glob("*.npy")}
        train_lr = {p.name for p in (kla / "train" / "NoisyLR").glob("*.npy")}
        test_ = {p.name for p in (kla / "test_NoisyLR").glob("*.npy")}
        names = set(membership)
        check.update({
            "n_train_gt": len(train_gt), "n_train_lr": len(train_lr), "n_test": len(test_),
            "intersection_with_train_gt": sorted(names & train_gt),
            "intersection_with_train_lr": sorted(names & train_lr),
            "intersection_with_test": sorted(names & test_),
        })
        check["disjoint_from_train_gt"] = len(check["intersection_with_train_gt"]) == 0
        check["disjoint_from_train_lr"] = len(check["intersection_with_train_lr"]) == 0
        check["disjoint_from_test"] = len(check["intersection_with_test"]) == 0
    (out_root / "membership_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(json.dumps(check, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
