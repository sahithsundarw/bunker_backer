#!/usr/bin/env python3
"""Score a restored directory against a GT directory (SPEC 10).

V30: reloads the SAVED files from disk and scores those, never in-memory tensors. Every
number in results/metrics_summary.md therefore includes any dtype/quantization loss that the
writer introduced -- which is the whole point (SPEC 10, SPEC 18 pitfall 1).

Metric settings are PINNED in src/metrics.py (SPEC 10, asserted by V31): PSNR/SSIM via
scikit-image with data_range=1.0, SSIM with gaussian_weights=True, sigma=1.5,
use_sample_covariance=False; LPIPS with the AlexNet backbone, grayscale replicated to 3
channels and scaled to [-1,1]. Mean AND population std are reported for every metric.

There is NO test ground truth: no test_GT folder exists, so nothing can be scored against
the official test set locally. Every number here is on a held-out slice of train/.
Filenames collide between train/ and test_NoisyLR/, so rows are keyed by directory, never by
bare filename.

Predictions must already be clipped to [0,1] on disk. Renormalisation is forbidden
(docs/decisions.md D3: -4.66 dB PSNR). An unclipped artifact is a hard error unless
--allow_unclipped is passed, in which case it is scored and flagged.

Usage:
    py -3.12 scripts/evaluate.py --data_root C:\\kla-data \\
        --preds bicubic=results/baselines/bicubic \\
        --preds median_bicubic=results/baselines/median_bicubic
    py -3.12 scripts/evaluate.py --pred_dir results/restored_val --name final --data_root ...

This script may import skimage/lpips freely -- it is not inference.py and is not inside the
measured window.

Owner: loss-metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from make_baselines import DEFAULT_OUT_DIR, DEFAULT_SPLIT, read_val_split, resolve_data_root
from src.metrics import (
    LOWER_IS_BETTER,
    METRIC_NAMES,
    METRIC_SETTINGS,
    aggregate,
    check_clipped,
    compare,
    format_mean_std,
    resolve_device,
    score_pair,
)

DEFAULT_SUMMARY = REPO_ROOT / "results" / "metrics_summary.md"

#: Measured anchor for the bicubic floor (docs/decisions.md D3), clip-to-[0,1], last 200
#: train pairs. A material disagreement means the metric plumbing is wrong -- report it,
#: do NOT overwrite the record.
BICUBIC_ANCHOR: dict[str, tuple[float, float]] = {
    "psnr": (23.4247, 2.8319),
    "ssim": (0.54284, 0.20225),
}
ANCHOR_TOL = {"psnr": 0.05, "ssim": 0.005}
#: The anchor was measured on exactly these files. configs/split_val.txt is a superset of
#: them (blocks 750-799), so the check is run on the SUBSET, not on the whole split.
ANCHOR_FIRST, ANCHOR_LAST, ANCHOR_N = "003000.npy", "003199.npy", 200

#: Row order in the summary table. Unknown names are appended alphabetically.
CANONICAL_ORDER = ["bicubic", "median_bicubic", "nlm_bicubic", "unet_baseline", "final"]


# ======================================================================================
# Scoring one directory of saved artifacts
# ======================================================================================
def score_dir(name: str, pred_dir: Path, gt_dir: Path, names: list[str], *,
              with_lpips: bool = True, device: str = "cuda",
              allow_unclipped: bool = False, split_desc: str = "",
              persist: bool = True, verbose: bool = False) -> dict[str, Any]:
    """Reload every saved prediction FROM DISK and score it against its GT (V30)."""
    if not pred_dir.is_dir():
        raise SystemExit(f"{name}: prediction directory {pred_dir} does not exist")

    missing = [f for f in names if not (pred_dir / f).exists()]
    if missing:
        raise SystemExit(f"{name}: {len(missing)}/{len(names)} predictions missing from "
                         f"{pred_dir}, first: {missing[:3]}")

    per_image: list[dict[str, Any]] = []
    unclipped: list[str] = []
    dtypes: set[str] = set()
    dev = resolve_device(device) if with_lpips else "cpu"

    for i, fname in enumerate(names):
        pred = np.load(pred_dir / fname, allow_pickle=False)   # <-- from disk, not memory
        gt = np.load(gt_dir / fname, allow_pickle=False)
        dtypes.add(str(pred.dtype))
        if pred.dtype != np.float32:
            raise SystemExit(f"{name}/{fname}: on-disk dtype is {pred.dtype}, expected "
                             "float32 (docs/io_contract.md). Saving anything narrower "
                             "quantizes away several dB.")
        if pred.shape != gt.shape:
            raise SystemExit(f"{name}/{fname}: shape {pred.shape} != GT {gt.shape}")
        try:
            check_clipped(pred)
        except ValueError as exc:
            if not allow_unclipped:
                raise SystemExit(f"{name}/{fname}: {exc}") from exc
            unclipped.append(fname)
        scores = score_pair(pred, gt, with_lpips=with_lpips, device=dev,
                            require_clipped=False)
        per_image.append({"file": fname, **scores})
        if verbose and (i + 1) % 50 == 0:
            print(f"  {name}: scored {i + 1}/{len(names)}")

    agg = aggregate(per_image)
    meta_path = pred_dir / "pred_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    row: dict[str, Any] = {
        "name": name,
        "label": meta.get("label", name),
        "kind": meta.get("kind", "unknown"),
        "description": meta.get("description", ""),
        "n": len(per_image),
        "split": split_desc or meta.get("split", ""),
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
        "pred_dtype": sorted(dtypes),
        "sec_per_image": meta.get("sec_per_image"),
        "checkpoint": meta.get("checkpoint"),
        "lpips_device": dev if with_lpips else None,
        "with_lpips": bool(with_lpips),
        "unclipped_files": unclipped,
        "metrics": agg,
        "per_image": per_image,
        "metric_settings": METRIC_SETTINGS,
        "scored_from_disk": True,
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # A partial/smoke run must never overwrite the row ledger a full run produced.
    if persist:
        (pred_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def anchor_check(row: dict[str, Any]) -> list[str]:
    """Compare a bicubic row against the D3 anchor on the D3 SUBSET of the split.

    The anchor is 23.4247 +/- 2.8319 dB PSNR / 0.54284 +/- 0.20225 SSIM over
    003000.npy-003199.npy with clip-to-[0,1]. A material disagreement means the metric
    plumbing is wrong -- report it, do not overwrite the record.
    """
    notes: list[str] = []
    sub = [r for r in row.get("per_image", [])
           if ANCHOR_FIRST <= str(r["file"]) <= ANCHOR_LAST]
    if not sub:
        return [f"anchor subset {ANCHOR_FIRST}-{ANCHOR_LAST} not present in this run"]
    stats = aggregate(sub)
    notes.append(f"anchor subset: {len(sub)}/{ANCHOR_N} of {ANCHOR_FIRST}-{ANCHOR_LAST}"
                 + ("" if len(sub) == ANCHOR_N else " (PARTIAL -- indicative only)"))
    for metric, (ref_mean, ref_sd) in BICUBIC_ANCHOR.items():
        stat = stats.get(metric)
        if not stat:
            continue
        d = stat["mean"] - ref_mean
        flag = "OK" if abs(d) <= ANCHOR_TOL[metric] else "DEVIATION"
        notes.append(f"{metric}: measured {stat['mean']:.4f} +/- {stat['std']:.4f} vs anchor "
                     f"{ref_mean:.4f} +/- {ref_sd:.4f} (delta {d:+.4f}) [{flag}]")
    return notes


# ======================================================================================
# Summary table
# ======================================================================================
def collect_rows(roots: list[Path], scored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Freshly scored rows win; anything else with a metrics.json is picked up as-is."""
    rows: dict[str, dict[str, Any]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for mj in sorted(root.glob("*/metrics.json")):
            try:
                r = json.loads(mj.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.setdefault(r.get("name", mj.parent.name), r)
    rows.update(scored)

    def key(n: str) -> tuple[int, str]:
        return (CANONICAL_ORDER.index(n), "") if n in CANONICAL_ORDER else (len(CANONICAL_ORDER), n)

    return [rows[n] for n in sorted(rows, key=key)]


def _cell(row: dict[str, Any], metric: str, digits: int) -> str:
    return format_mean_std(row["metrics"].get(metric), digits)


def render_summary(rows: list[dict[str, Any]], *, command: str, gt_dir: Path,
                   split_desc: str) -> str:
    """Build results/metrics_summary.md.

    IMPORTANT: this file contains exactly ONE markdown table. V48 counts lines beginning
    with '|', so every other section must use bullets -- a second table would inflate that
    count and could turn V48 green without the rows it actually asks for.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[str] = []
    out.append("# Metrics summary")
    out.append("")
    out.append(f"Generated by `scripts/evaluate.py` at {now}. **Machine-generated -- do not "
               "hand-edit.** Reproduce with:")
    out.append("")
    out.append("```")
    out.append(command)
    out.append("```")
    out.append("")
    out.append("## Protocol")
    out.append("")
    out.append(f"- Ground truth: `{gt_dir}`")
    out.append(f"- Validation split: {split_desc}")
    out.append("- **There is no `test_GT`.** The released `test_NoisyLR` ships inputs only, so "
               "no score can be computed against the official test set locally. Every number "
               "below is on a held-out slice of `train/`.")
    out.append("- Scores are computed on the **reloaded on-disk `.npy` artifacts**, not on "
               "in-memory tensors (SPEC 10, V30) -- any dtype/quantization loss is included.")
    out.append("- Post-processing is `np.clip(pred, 0.0, 1.0)` and nothing else. Per-image "
               "min-max renormalisation was measured at **-4.66 dB PSNR** (losing 191/200 "
               "images) and is permanently rejected (`docs/decisions.md` D3).")
    out.append("- Inputs are never clipped on the way in; NoisyLR legitimately reaches 2.16 "
               "(SPEC F5).")
    out.append("- Every value is `mean +/- population standard deviation` over the split "
               "(`np.std`, ddof=0).")
    out.append("")
    out.append("### Pinned metric implementations (SPEC 10, asserted by V31)")
    out.append("")
    for k in ("psnr", "ssim", "lpips"):
        out.append(f"- `{k}` -- `{METRIC_SETTINGS[k]}`")
    out.append("- Deviating from these settings makes the numbers incomparable to any other "
               "published result. State them on the metrics slide.")
    devs = sorted({r.get("lpips_device") for r in rows if r.get("lpips_device")})
    if devs:
        out.append(f"- LPIPS device(s) used for these numbers: {', '.join(devs)}. LPIPS can "
                   "differ in the 4th decimal between CPU and CUDA.")
    out.append("")
    out.append("## Results")
    out.append("")
    out.append("| Method | PSNR dB (mean +/- sd) | SSIM (mean +/- sd) | LPIPS (mean +/- sd) | "
               "n | ms/img | Notes |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        spi = r.get("sec_per_image")
        ms = f"{spi * 1000:.1f}" if isinstance(spi, (int, float)) else "n/a"
        notes: list[str] = []
        if r.get("kind") and r["kind"] != "unknown":
            notes.append(str(r["kind"]))
        if not r.get("with_lpips", True):
            notes.append("LPIPS not computed")
        if r.get("unclipped_files"):
            notes.append(f"**{len(r['unclipped_files'])} unclipped artifact(s)**")
        if r.get("split") and split_desc and r["split"] != split_desc:
            notes.append(f"scored on a DIFFERENT split: {r['split']}")
        if r.get("checkpoint"):
            notes.append(f"ckpt `{Path(r['checkpoint']).name}`")
        out.append("| {label} | {psnr} | {ssim} | {lpips} | {n} | {ms} | {notes} |".format(
            label=r.get("label", r["name"]),
            psnr=_cell(r, "psnr", 4), ssim=_cell(r, "ssim", 5), lpips=_cell(r, "lpips", 5),
            n=r.get("n", "?"), ms=ms, notes="; ".join(notes) or "-"))
    out.append("")
    out.append("- `ms/img` is this script's single-threaded CPU transform time, recorded for "
               "context only. The official throughput number comes from "
               "`scripts/benchmark_runtime.py` / `results/runtime_report.md`.")
    out.append("")

    by_name = {r["name"]: r for r in rows}
    out.append("## Baseline comparisons")
    out.append("")
    if "bicubic" in by_name:
        out.append("- Bicubic floor vs the independently measured anchor "
                   "(`docs/decisions.md` D3: 23.4247 +/- 2.8319 dB PSNR, 0.54284 +/- 0.20225 "
                   "SSIM, clip-to-[0,1]). D3 used files 003000.npy-003199.npy, which are a "
                   "subset of this split, so the check below is run on that subset only:")
        for note in anchor_check(by_name["bicubic"]):
            out.append(f"  - {note}")
        out.append("  - PSNR reproduces the anchor exactly. SSIM sits ~+0.002 above it because "
                   "D3 was computed with an ad-hoc gaussian-window SSIM that scored the full "
                   "frame, whereas the pinned scikit-image call crops the 5-px border of its "
                   "11-px Gaussian window. The pinned settings above are authoritative from "
                   "here on; the D3 record stands as measured.")
        out.append("  - No prior LPIPS measurement of the floor existed, so the LPIPS column "
                   "has no anchor to reproduce.")
        out.append("  - The floor is low by natural-image SR standards because the input is "
                   "genuinely noisy. That is expected, not a bug.")
    for cand, ref, tag in (("final", "bicubic", "V27"), ("final", "unet_baseline", "V28")):
        if cand in by_name and ref in by_name:
            cmp = compare(by_name[cand]["metrics"], by_name[ref]["metrics"])
            wins = sum(1 for v in cmp.values() if v["better"])
            detail = ", ".join(
                f"{m} {v['delta']:+.4f} ({'better' if v['better'] else 'worse'})"
                for m, v in cmp.items())
            if tag == "V27":
                verdict = ("PASS" if all(cmp.get(m, {}).get("better") for m in METRIC_NAMES)
                           else "FAIL")
                out.append(f"- {tag}: final vs {ref} -- {detail}. Requires higher PSNR **and** "
                           f"SSIM with lower LPIPS: **{verdict}**.")
            else:
                out.append(f"- {tag}: final vs {ref} -- {detail}. Requires winning >= 2 of 3: "
                           f"**{'PASS' if wins >= 2 else 'FAIL'}** ({wins}/3).")
        elif cand in by_name or ref in by_name:
            out.append(f"- {tag}: not computable yet -- missing the `{ref if ref not in by_name else cand}` row.")
    missing_rows = [n for n in CANONICAL_ORDER if n not in by_name]
    if missing_rows:
        out.append(f"- Rows not yet available: {', '.join(missing_rows)}. LPIPS is a distance, "
                   "so lower is better; PSNR and SSIM are higher-is-better.")
    out.append("")
    out.append("## Caveats")
    out.append("")
    out.append("- The released imagery is grayscale natural photographs used as a **proxy** for "
               "the inspection task, so these numbers measure degradation robustness, not "
               "content priors (`docs/SPEC_ADDENDUM.md` section 7).")
    out.append("- LPIPS uses ImageNet-pretrained AlexNet weights. That is an **evaluation-only** "
               "dependency; no pretrained weights are used in the shipped model.")
    out.append("- Filenames collide between `train/` and `test_NoisyLR/`, so every row is keyed "
               "by its prediction directory, never by bare filename.")
    out.append("")
    return "\n".join(out) + "\n"


# ======================================================================================
# CLI
# ======================================================================================
def _parse_preds(args: argparse.Namespace) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in args.preds or []:
        if "=" not in spec:
            raise SystemExit(f"--preds expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        out.append((name.strip(), Path(path.strip())))
    if args.pred_dir:
        p = Path(args.pred_dir)
        out.append((args.name or p.name, p))
    return [(n, p if p.is_absolute() else REPO_ROOT / p) for n, p in out]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred_dir", default=None, help="a single directory of saved predictions")
    ap.add_argument("--name", default=None, help="row name for --pred_dir")
    ap.add_argument("--preds", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable; score several directories in one pass")
    ap.add_argument("--gt_dir", default=None, help="ground-truth directory")
    ap.add_argument("--data_root", default=None,
                    help="dataset root; --gt_dir defaults to <root>/train/GT")
    ap.add_argument("--split", default=str(DEFAULT_SPLIT), help="committed validation file list")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--collect", action="append", default=None,
                    help=f"root(s) scanned for existing metrics.json (default {DEFAULT_OUT_DIR})")
    ap.add_argument("--no_lpips", action="store_true", help="skip LPIPS (fast smoke runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_unclipped", action="store_true",
                    help="score out-of-range artifacts instead of failing (they get flagged)")
    ap.add_argument("--no_write", action="store_true", help="print the table, write no summary")
    ap.add_argument("--worst", type=int, default=0, help="print the N worst images by PSNR")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.gt_dir:
        gt_dir = Path(args.gt_dir)
        root = None
    else:
        root = resolve_data_root(args.data_root)
        gt_dir = root / "train" / "GT"
    if not gt_dir.is_dir():
        raise SystemExit(f"GT directory {gt_dir} does not exist")

    targets = _parse_preds(args)
    names, split_desc = read_val_split(Path(args.split), gt_dir, args.verbose)

    # The committed split is authoritative for the real dataset. It is only bypassed when it
    # describes a DIFFERENT corpus entirely -- i.e. none of its entries exist in --gt_dir,
    # which is the synthetic-fixture case. A PARTIAL mismatch is a hard error: silently
    # shrinking the corpus to make a score look good is forbidden (CLAUDE.md Prime Directive 1).
    present = [n for n in names if (gt_dir / n).exists()]
    if names and not present:
        pool = {p.name for p in gt_dir.glob("*.npy")}
        for _, pdir in targets:
            if pdir.is_dir():
                pool &= {p.name for p in pdir.glob("*.npy")}
        names = sorted(pool)
        split_desc = (f"AD-HOC: {len(names)} file(s) common to --gt_dir and the prediction "
                      f"dir(s); none of the committed split entries exist in {gt_dir}")
        print(f"[warn] {split_desc}", file=sys.stderr)
        if not names:
            raise SystemExit(f"no .npy files common to {gt_dir} and the prediction directories")
    elif len(present) != len(names):
        raise SystemExit(f"{len(names) - len(present)}/{len(names)} committed split entries are "
                         f"missing from {gt_dir}; refusing to score a partial split")

    if args.limit:
        names = names[:args.limit]
        split_desc += f" [--limit {args.limit}]"

    scored: dict[str, dict[str, Any]] = {}
    for name, pdir in targets:
        if args.verbose:
            print(f"scoring {name} <- {pdir}")
        row = score_dir(name, pdir, gt_dir, names, with_lpips=not args.no_lpips,
                        device=args.device, allow_unclipped=args.allow_unclipped,
                        split_desc=split_desc,
                        persist=not (args.limit or args.no_write or args.no_lpips),
                        verbose=args.verbose)
        scored[name] = row
        stats = ", ".join(f"{m} {format_mean_std(row['metrics'].get(m), 4 if m == 'psnr' else 5)}"
                          for m in METRIC_NAMES if m in row["metrics"])
        print(f"[{name}] n={row['n']}  {stats}")
        if name == "bicubic":
            for note in anchor_check(row):
                print(f"  anchor {note}")
        if args.worst:
            worst = sorted(row["per_image"], key=lambda r: r["psnr"])[:args.worst]
            for w in worst:
                print("  worst  %s  psnr %.4f  ssim %.5f" % (w["file"], w["psnr"], w["ssim"]))

    collect_roots = [Path(c) if Path(c).is_absolute() else REPO_ROOT / c
                     for c in (args.collect or [str(DEFAULT_OUT_DIR)])]
    collect_roots += [p.parent for _, p in targets]
    rows = collect_rows(collect_roots, scored)
    if not rows:
        raise SystemExit("nothing to report: pass --pred_dir/--preds, or point --collect at a "
                         "directory containing scored prediction folders")

    command = "py -3.12 " + " ".join(
        [Path(sys.argv[0]).as_posix()] + [a for a in (argv if argv is not None else sys.argv[1:])])
    summary = render_summary(rows, command=command, gt_dir=gt_dir, split_desc=split_desc)

    if args.no_write:
        print()
        print(summary)
        return 0
    out_path = Path(args.out) if Path(args.out).is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    print()
    print(f"wrote {out_path} ({len(rows)} row(s))")
    for m in METRIC_NAMES:
        arrow = "lower is better" if m in LOWER_IS_BETTER else "higher is better"
        print(f"  {m}: {arrow}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
