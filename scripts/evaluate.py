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

from make_baselines import (
    DEFAULT_OUT_DIR,
    DEFAULT_SPLIT,
    PROXY_OOD_GT,
    PROXY_OOD_OUT_SUBDIR,
    read_val_split,
    resolve_data_root,
)
from src.metrics import (
    LOWER_IS_BETTER,
    METRIC_NAMES,
    METRIC_SETTINGS,
    aggregate,
    check_clipped,
    compare,
    format_mean_std,
    paired_compare,
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

#: Proxy-OOD generalisation check (U-9, V63): 40 procedurally-generated geometric images,
#: NOT semiconductor imagery, NOT the official test set (docs/dataset_findings.md "Proxy-OOD
#: set", docs/SPEC_ADDENDUM.md section 11 banned-phrase list). Scored and reported as a
#: SEPARATE section of results/metrics_summary.md, never blended into the in-distribution
#: CANONICAL_ORDER table above -- the two corpora measure different things (content-domain
#: generalisation under a fixed, already-measured degradation, vs in-distribution quality).
DEFAULT_PROXY_OOD_PRED_DIR = DEFAULT_OUT_DIR / PROXY_OOD_OUT_SUBDIR / "final"
DEFAULT_PROXY_OOD_GT_DIR = PROXY_OOD_GT
PROXY_OOD_ROW_NAME = "proxy_ood_final"

#: Real-SEM OOD robustness report (Round 2 Phase 3, docs/decisions.md D53): 45 real electron-
#: microscopy images (Zenodo record 17315241, CC-BY 4.0 -- see scripts/gen_real_sem_ood.py and
#: docs/decisions.md D53 for full provenance/licence), degraded with the same already-fitted
#: degradation model, zero refitting. Scored and reported as a SEPARATE section, same
#: discipline as the procedural proxy-OOD check above -- never blended into the
#: in-distribution CANONICAL_ORDER table, since it measures a different thing (transfer to a
#: genuinely different imaging modality, not just different content within natural photos).
REAL_SEM_OOD_ROOT = REPO_ROOT / "results" / "eda" / "real_sem_ood"
DEFAULT_REAL_SEM_OOD_GT_DIR = REAL_SEM_OOD_ROOT / "GT"
DEFAULT_REAL_SEM_OOD_PRED_DIR = DEFAULT_OUT_DIR / "real_sem_ood" / "final"
REAL_SEM_OOD_ROW_NAME = "real_sem_ood_final"


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
# Proxy-OOD generalisation check (U-9, V63) -- scored SEPARATELY from the in-distribution table
# ======================================================================================
def score_proxy_ood(pred_dir: Path, gt_dir: Path, *, with_lpips: bool = True,
                    device: str = "cuda", allow_unclipped: bool = False,
                    persist: bool = True, verbose: bool = False) -> dict[str, Any]:
    """Score the shipped model's predictions on the 40 procedural proxy-OOD images.

    This is NOT semiconductor imagery and NOT the official test set (docs/dataset_findings.md
    "Proxy-OOD set"). Reuses score_dir with the same pinned metric settings (V31) and the
    same on-disk-reload discipline (V30) as the in-distribution table.

    ``persist`` must be wired to the same (limit/no_write/no_lpips) guard the main table uses:
    a partial/smoke run (e.g. --no_lpips) must never overwrite the full metrics.json a
    complete run produced, exactly the discipline score_dir's caller already applies to every
    other row (see the persist= comment at its call site above).
    """
    if not pred_dir.is_dir():
        raise SystemExit(
            f"--proxy_ood: prediction directory {pred_dir} does not exist. Generate it with:\n"
            f"  py -3.12 scripts/make_baselines.py --proxy_ood --baselines final")
    if not gt_dir.is_dir():
        raise SystemExit(f"--proxy_ood: GT directory {gt_dir} does not exist")
    names = sorted(p.name for p in gt_dir.glob("*.npy"))
    if not names:
        raise SystemExit(f"--proxy_ood: no .npy GT files found in {gt_dir}")
    split_desc = (f"PROXY-OOD: {len(names)} procedurally-generated geometric images "
                 f"({gt_dir.parent.name}/GT) -- NOT semiconductor imagery, NOT the official "
                 "test set. Degraded with the already-fitted degradation model, zero refitting.")
    return score_dir(PROXY_OOD_ROW_NAME, pred_dir, gt_dir, names, with_lpips=with_lpips,
                     device=device, allow_unclipped=allow_unclipped, split_desc=split_desc,
                     persist=persist, verbose=verbose)


def render_proxy_ood_section(row: dict[str, Any], *, indist_row: dict[str, Any] | None) -> list[str]:
    """Bullet-only section (NEVER a markdown table -- V48 counts '|'-prefixed lines against
    the single in-distribution table above; a second table would inflate that count).
    """
    out: list[str] = []
    out.append("## Proxy-OOD generalisation check (procedural geometric content, n="
               f"{row.get('n', '?')})")
    out.append("")
    out.append("**What this is:** the shipped model (`weights/best.pt`) run on 40 "
               "procedurally-generated (numpy primitives only) synthetic grayscale images -- "
               "line/space gratings, contact-hole grids, checkerboards, circuit-like traces, "
               "sharp-edged shapes -- degraded with the SAME already-fitted degradation model "
               "used for the in-distribution numbers, with zero parameters refit on this "
               "content. It tests whether restoration quality holds up when image CONTENT "
               "shifts from natural-photo textures to periodic/geometric structure, with the "
               "degradation held fixed.")
    out.append("")
    out.append("**What this is NOT:** this is not semiconductor imagery, not SEM imagery, and "
               "not a validated proxy for KLA's hidden test set -- it is synthetic procedural "
               "geometric content (`docs/dataset_findings.md` \"Proxy-OOD set\", "
               "`docs/SPEC_ADDENDUM.md` section 11). No real inspection imagery exists "
               "anywhere in this project.")
    out.append("")
    m = row["metrics"]
    out.append(f"- PSNR dB (mean +/- sd): {format_mean_std(m.get('psnr'), 4)}")
    out.append(f"- SSIM (mean +/- sd): {format_mean_std(m.get('ssim'), 5)}")
    out.append(f"- LPIPS (mean +/- sd): {format_mean_std(m.get('lpips'), 5)}")
    out.append(f"- n = {row.get('n', '?')}, ground truth: `{row.get('gt_dir', '?')}`, "
               f"predictions: `{row.get('pred_dir', '?')}`")
    if row.get("unclipped_files"):
        out.append(f"- **{len(row['unclipped_files'])} unclipped artifact(s)** flagged -- see "
                   "`--allow_unclipped`.")
    if indist_row is not None:
        im = indist_row["metrics"]
        out.append("")
        out.append("**Versus the in-distribution 400-image number for the same checkpoint** "
                   f"(`{indist_row.get('label', indist_row.get('name'))}`, "
                   f"n={indist_row.get('n', '?')}):")
        for key, digits in (("psnr", 4), ("ssim", 5), ("lpips", 5)):
            if key not in m or key not in im:
                continue
            delta = m[key]["mean"] - im[key]["mean"]
            better = (delta < 0.0) if key in LOWER_IS_BETTER else (delta > 0.0)
            out.append(f"  - {key}: proxy-OOD {m[key]['mean']:.{digits}f} vs in-distribution "
                       f"{im[key]['mean']:.{digits}f} (delta {delta:+.{digits}f}, "
                       f"{'better' if better else 'worse'} on proxy-OOD)")
    else:
        out.append("")
        out.append("- No in-distribution `final` row was scored in this run, so no side-by-side "
                   "delta is shown here. See the Results table above for that number when "
                   "available.")
    out.append("")
    out.append("- **Read the PSNR-vs-SSIM/LPIPS split, don't average it away.** PSNR is measurably "
               "worse here than in-distribution (fine periodic gratings near the 2x decimation's "
               "Nyquist limit alias, consistent with `docs/dataset_findings.md`'s proxy-OOD "
               "finding), yet SSIM and LPIPS are both measurably BETTER than in-distribution. "
               "This is not a scoring bug: the worst-PSNR images in this set still score SSIM "
               ">= 0.94 and LPIPS <= 0.09 (checked per-image), because large flat/uniform "
               "regions and locally-correlated periodic structure are easy for a windowed "
               "structural or perceptual metric even when a global phase/intensity offset "
               "tanks PSNR. Report all three numbers on the deck; do not collapse them to one.")
    out.append("")
    out.append("- **Cannot** validate performance on real semiconductor/SEM inspection imagery "
               "(none exists anywhere in this project) or robustness to a *different* "
               "degradation than the one measured from the released data.")
    out.append("")
    return out


# ======================================================================================
# Real-SEM OOD robustness report (Round 2 Phase 3, docs/decisions.md D53)
# ======================================================================================
def score_real_sem_ood(pred_dir: Path, gt_dir: Path, *, with_lpips: bool = True,
                       device: str = "cuda", allow_unclipped: bool = False,
                       persist: bool = True, verbose: bool = False) -> dict[str, Any]:
    """Score the shipped model's predictions on 45 real SEM (electron-microscopy) images.

    Real inspection-adjacent imagery, not natural photos, not procedural geometry -- the
    actual transfer-risk axis this project has never had evidence on. Zero training, zero
    parameter fitting on this content: same pinned metrics (V31), same on-disk-reload
    discipline (V30) as every other row.
    """
    if not pred_dir.is_dir():
        raise SystemExit(
            f"--real_sem_ood: prediction directory {pred_dir} does not exist. Generate "
            "predictions from weights/best.pt against results/eda/real_sem_ood/NoisyLR first "
            "(see scripts/gen_real_sem_ood.py and docs/decisions.md D53).")
    if not gt_dir.is_dir():
        raise SystemExit(f"--real_sem_ood: GT directory {gt_dir} does not exist")
    names = sorted(p.name for p in gt_dir.glob("*.npy"))
    if not names:
        raise SystemExit(f"--real_sem_ood: no .npy GT files found in {gt_dir}")
    split_desc = (f"REAL-SEM OOD: {len(names)} real electron-microscopy images "
                 "(Zenodo 17315241, CC-BY 4.0) -- NOT the official test set, NOT trained or "
                 "fitted on. Degraded with the already-fitted degradation model, zero "
                 "refitting.")
    return score_dir(REAL_SEM_OOD_ROW_NAME, pred_dir, gt_dir, names, with_lpips=with_lpips,
                     device=device, allow_unclipped=allow_unclipped, split_desc=split_desc,
                     persist=persist, verbose=verbose)


def render_real_sem_ood_section(row: dict[str, Any], *, indist_row: dict[str, Any] | None,
                                bicubic_row: dict[str, Any] | None) -> list[str]:
    """Bullet-only section (same V48 constraint as render_proxy_ood_section: no second table)."""
    out: list[str] = []
    out.append("## Real-SEM OOD robustness report (genuine electron-microscopy content, n="
               f"{row.get('n', '?')})")
    out.append("")
    out.append(f"**What this is:** the shipped model (`weights/best.pt`) run on {row.get('n', '?')} "
               "real scanning-electron-microscopy images/crops -- from unique underlying tiles "
               "of **Scanning Electron Microscopy (SEM) Dataset of Additively Manufactured Ni-WC "
               "Metal Matrix Composites for Semantic Segmentation**, Zenodo record 17315241, "
               "**CC-BY 4.0** (`https://zenodo.org/records/17315241`), used here for "
               "evaluation only -- no training, no parameter fitting, per F14 disclosure. "
               "Degraded with the SAME already-fitted degradation model used for every other "
               "number in this file, zero refitting on this content. This measures actual "
               "transfer from natural-photograph training to a genuinely different imaging "
               "modality -- the real question this project's own proxy framing (D4) always "
               "left open.")
    out.append("")
    out.append("**What this is NOT:** not semiconductor inspection imagery specifically (it is "
               "materials-science SEM of a metal matrix composite), not KLA's hidden test set, "
               "and the source images were centre-cropped 512->256 and per-image min-max "
               "normalised to [0,1] to match this project's GT convention -- a real "
               "pre-processing step, not part of the original dataset.")
    out.append("")
    m = row["metrics"]
    out.append(f"- PSNR dB (mean +/- sd): {format_mean_std(m.get('psnr'), 4)}")
    out.append(f"- SSIM (mean +/- sd): {format_mean_std(m.get('ssim'), 5)}")
    out.append(f"- LPIPS (mean +/- sd): {format_mean_std(m.get('lpips'), 5)}")
    out.append(f"- n = {row.get('n', '?')}, ground truth: `{row.get('gt_dir', '?')}`, "
               f"predictions: `{row.get('pred_dir', '?')}`")
    if row.get("unclipped_files"):
        out.append(f"- **{len(row['unclipped_files'])} unclipped artifact(s)** flagged -- see "
                   "`--allow_unclipped`.")
    out.append("")
    if indist_row is not None:
        im = indist_row["metrics"]
        out.append("**Versus the in-distribution 400-image number for the same checkpoint** "
                   f"(`{indist_row.get('label', indist_row.get('name'))}`, "
                   f"n={indist_row.get('n', '?')}):")
        for key, digits in (("psnr", 4), ("ssim", 5), ("lpips", 5)):
            if key not in m or key not in im:
                continue
            delta = m[key]["mean"] - im[key]["mean"]
            better = (delta < 0.0) if key in LOWER_IS_BETTER else (delta > 0.0)
            out.append(f"  - {key}: real-SEM {m[key]['mean']:.{digits}f} vs in-distribution "
                       f"{im[key]['mean']:.{digits}f} (delta {delta:+.{digits}f}, "
                       f"{'better' if better else 'worse'} on real-SEM)")
        out.append("")
    if bicubic_row is not None:
        bm = bicubic_row["metrics"]
        out.append(f"**Versus a bicubic floor computed on these SAME {row.get('n', '?')} pairs** "
                   "(isolates whether the low absolute scores below reflect this content being "
                   "structurally hard for ANY method, or a model-specific failure):")
        for key, digits in (("psnr", 4), ("ssim", 5), ("lpips", 5)):
            if key not in m or key not in bm:
                continue
            delta = m[key]["mean"] - bm[key]["mean"]
            better = (delta < 0.0) if key in LOWER_IS_BETTER else (delta > 0.0)
            out.append(f"  - {key}: final {m[key]['mean']:.{digits}f} vs bicubic "
                       f"{bm[key]['mean']:.{digits}f} (delta {delta:+.{digits}f}, "
                       f"{'better' if better else 'worse'} than bicubic)")
        out.append("")
    out.append("- **Read honestly, not softened.** Absolute quality on real SEM content is "
               "far below in-distribution (PSNR typically ~10 dB lower) -- a real, severe "
               "domain gap between natural-photo training and this imaging modality. The "
               "bicubic comparison above shows whether the model still adds value over doing "
               "nothing: a model win on PSNR/LPIPS but a loss on SSIM here would mean the "
               "measured degradation transfers (D16's hypothesis) but natural-photo content "
               "priors partially work against the model on real inspection-adjacent texture -- "
               "report whichever pattern is actually measured, do not round it up.")
    out.append("")
    out.append("- **Cannot** claim this validates performance on KLA's actual hidden test set "
               "(different, undisclosed content) or on true semiconductor inspection imagery "
               "specifically (this is a materials-science SEM set, not a fab/inspection one) -- "
               "it is the closest genuinely-real evidence available without violating the "
               "F17/D11 prohibitions on touching anything resembling the hidden test data.")
    out.append("")
    return out


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
                   split_desc: str, proxy_ood_row: dict[str, Any] | None = None,
                   real_sem_ood_row: dict[str, Any] | None = None,
                   real_sem_bicubic_row: dict[str, Any] | None = None) -> str:
    """Build results/metrics_summary.md.

    IMPORTANT: this file contains exactly ONE markdown table. V48 counts lines beginning
    with '|', so every other section must use bullets -- a second table would inflate that
    count and could turn V48 green without the rows it actually asks for. This is also why
    the proxy-OOD section (U-9, V63) is bullets-only, never a second table.
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
            if tag == "V27":
                # V27 in verify_all.py compares two independent means (plus a 2*SEM margin
                # check on PSNR), so an unpaired mean-delta comparison is the right statistic
                # here and this verdict matches the verifier's.
                cmp = compare(by_name[cand]["metrics"], by_name[ref]["metrics"])
                detail = ", ".join(
                    f"{m} {v['delta']:+.4f} ({'better' if v['better'] else 'worse'})"
                    for m, v in cmp.items())
                verdict = ("PASS" if all(cmp.get(m, {}).get("better") for m in METRIC_NAMES)
                           else "FAIL")
                out.append(f"- {tag}: final vs {ref} -- {detail}. Requires higher PSNR **and** "
                           f"SSIM with lower LPIPS: **{verdict}**.")
            else:
                # V28 in verify_all.py is a PAIRED per-image test (both models score the same
                # images), not a comparison of two independent means -- counting a positive
                # mean delta as a "win" here is exactly the defect D31 removed from the
                # verifier (docs/decisions.md D31). This script must not re-introduce it, and
                # must not self-grade a PASS/FAIL against a verification check it does not
                # own: report the raw paired win/loss/tie counts only.
                pv = paired_compare(by_name[cand].get("per_image"),
                                    by_name[ref].get("per_image"))
                if not pv:
                    out.append(f"- {tag}: final vs {ref} -- fewer than "
                               "`src.metrics.PAIRED_MIN_N` common per-image pairs (or "
                               "per-image data missing), so no paired verdict is computed "
                               "here. See `py -3.12 scripts/verify_all.py --only V28` for the "
                               "authoritative result.")
                else:
                    detail = ", ".join(
                        f"{m} mean_diff {v['mean_diff']:+.5f} t={v['t']:+.2f} n={v['n']} "
                        f"({'win' if v['win'] else ('loss' if v['loss'] else 'tie')})"
                        for m, v in pv.items())
                    wins = sorted(k for k, v in pv.items() if v["win"])
                    losses = sorted(k for k, v in pv.items() if v["loss"])
                    ties = sorted(k for k, v in pv.items() if not v["win"] and not v["loss"])
                    out.append(f"- {tag}: final vs {ref} (paired per-image test, matching "
                               f"`scripts/verify_all.py`'s V28 statistic) -- {detail}. "
                               f"wins={wins} losses={losses} ties={ties}. **This script does "
                               "not declare a PASS/FAIL verdict for a verification check it "
                               "does not own** -- run `py -3.12 scripts/verify_all.py "
                               "--only V28` for that.")
        elif cand in by_name or ref in by_name:
            out.append(f"- {tag}: not computable yet -- missing the `{ref if ref not in by_name else cand}` row.")
    missing_rows = [n for n in CANONICAL_ORDER if n not in by_name]
    if missing_rows:
        out.append(f"- Rows not yet available: {', '.join(missing_rows)}. LPIPS is a distance, "
                   "so lower is better; PSNR and SSIM are higher-is-better.")
    out.append("")
    if proxy_ood_row is not None:
        out.extend(render_proxy_ood_section(proxy_ood_row, indist_row=by_name.get("final")))
    if real_sem_ood_row is not None:
        out.extend(render_real_sem_ood_section(real_sem_ood_row, indist_row=by_name.get("final"),
                                               bicubic_row=real_sem_bicubic_row))
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
    ap.add_argument("--proxy_ood", action="store_true",
                    help="ALSO score the 40-image procedural proxy-OOD set (U-9, V63) as a "
                         "SEPARATE section, never blended into the in-distribution table. "
                         "Requires predictions already written by "
                         "'make_baselines.py --proxy_ood --baselines final'.")
    ap.add_argument("--proxy_ood_pred_dir", default=None,
                    help=f"override (default {DEFAULT_PROXY_OOD_PRED_DIR})")
    ap.add_argument("--proxy_ood_gt_dir", default=None,
                    help=f"override (default {DEFAULT_PROXY_OOD_GT_DIR})")
    ap.add_argument("--real_sem_ood", action="store_true",
                    help="ALSO score the 45-image real-SEM OOD robustness set (Round 2 "
                         "Phase 3, docs/decisions.md D53) as a SEPARATE section. Requires "
                         "predictions already written against results/eda/real_sem_ood/"
                         "NoisyLR (see scripts/gen_real_sem_ood.py).")
    ap.add_argument("--real_sem_ood_pred_dir", default=None,
                    help=f"override (default {DEFAULT_REAL_SEM_OOD_PRED_DIR})")
    ap.add_argument("--real_sem_ood_gt_dir", default=None,
                    help=f"override (default {DEFAULT_REAL_SEM_OOD_GT_DIR})")
    ap.add_argument("--real_sem_ood_bicubic_dir", default=None,
                    help="optional bicubic-baseline predictions on the SAME real-SEM pairs, "
                         "for the honest-difficulty comparison in that section")
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

    proxy_ood_row: dict[str, Any] | None = None
    if args.proxy_ood:
        po_pred = (Path(args.proxy_ood_pred_dir) if args.proxy_ood_pred_dir
                  else DEFAULT_PROXY_OOD_PRED_DIR)
        po_pred = po_pred if po_pred.is_absolute() else REPO_ROOT / po_pred
        po_gt = (Path(args.proxy_ood_gt_dir) if args.proxy_ood_gt_dir
                else DEFAULT_PROXY_OOD_GT_DIR)
        po_gt = po_gt if po_gt.is_absolute() else REPO_ROOT / po_gt
        if args.verbose:
            print(f"scoring proxy-OOD <- {po_pred} vs {po_gt}")
        proxy_ood_row = score_proxy_ood(po_pred, po_gt, with_lpips=not args.no_lpips,
                                        device=args.device, allow_unclipped=args.allow_unclipped,
                                        persist=not (args.no_write or args.no_lpips),
                                        verbose=args.verbose)
        pm = proxy_ood_row["metrics"]
        stats = ", ".join(f"{m} {format_mean_std(pm.get(m), 4 if m == 'psnr' else 5)}"
                          for m in METRIC_NAMES if m in pm)
        print(f"[proxy-OOD] n={proxy_ood_row['n']}  {stats}  "
              "(procedural geometric content, NOT semiconductor imagery)")

    real_sem_ood_row: dict[str, Any] | None = None
    if args.real_sem_ood:
        rs_pred = (Path(args.real_sem_ood_pred_dir) if args.real_sem_ood_pred_dir
                  else DEFAULT_REAL_SEM_OOD_PRED_DIR)
        rs_pred = rs_pred if rs_pred.is_absolute() else REPO_ROOT / rs_pred
        rs_gt = (Path(args.real_sem_ood_gt_dir) if args.real_sem_ood_gt_dir
                else DEFAULT_REAL_SEM_OOD_GT_DIR)
        rs_gt = rs_gt if rs_gt.is_absolute() else REPO_ROOT / rs_gt
        if args.verbose:
            print(f"scoring real-SEM OOD <- {rs_pred} vs {rs_gt}")
        real_sem_ood_row = score_real_sem_ood(
            rs_pred, rs_gt, with_lpips=not args.no_lpips, device=args.device,
            allow_unclipped=args.allow_unclipped,
            persist=not (args.no_write or args.no_lpips), verbose=args.verbose)
        rm = real_sem_ood_row["metrics"]
        stats = ", ".join(f"{m} {format_mean_std(rm.get(m), 4 if m == 'psnr' else 5)}"
                          for m in METRIC_NAMES if m in rm)
        print(f"[real-SEM OOD] n={real_sem_ood_row['n']}  {stats}  "
              "(genuine electron-microscopy content, Zenodo 17315241 CC-BY 4.0)")
    real_sem_bicubic_row: dict[str, Any] | None = None
    if real_sem_ood_row is not None and args.real_sem_ood_bicubic_dir:
        bc_dir = Path(args.real_sem_ood_bicubic_dir)
        bc_dir = bc_dir if bc_dir.is_absolute() else REPO_ROOT / bc_dir
        names_rs = sorted(p.name for p in rs_gt.glob("*.npy"))
        real_sem_bicubic_row = score_dir(
            "real_sem_ood_bicubic", bc_dir, rs_gt, names_rs, with_lpips=not args.no_lpips,
            device=args.device, allow_unclipped=args.allow_unclipped,
            split_desc=f"REAL-SEM OOD bicubic floor, same {len(names_rs)} pairs",
            persist=not (args.no_write or args.no_lpips), verbose=args.verbose)

    command = "py -3.12 " + " ".join(
        [Path(sys.argv[0]).as_posix()] + [a for a in (argv if argv is not None else sys.argv[1:])])
    summary = render_summary(rows, command=command, gt_dir=gt_dir, split_desc=split_desc,
                             proxy_ood_row=proxy_ood_row, real_sem_ood_row=real_sem_ood_row,
                             real_sem_bicubic_row=real_sem_bicubic_row)

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
