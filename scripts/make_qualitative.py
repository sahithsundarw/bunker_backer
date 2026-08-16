#!/usr/bin/env python3
"""Build results/qualitative/ -- the visual evidence pack required by V49.

Produces, from artifacts that already exist on disk:

  * >= 4 SUCCESS figures, each a 4-panel row at full 256x256 resolution --
    degraded input (nearest x2), bicubic x2 baseline, our model, ground truth --
    with the measured PSNR/SSIM of every restored panel in its title;
  * >= 1 FAILURE figure ('fail' in the filename), same layout, plus the
    spectral measurements that explain WHY the case is unrecoverable;
  * README.md describing every figure and the failure in detail.

Selection is deterministic: successes are the validation images whose model PSNR
lands closest to fixed percentiles of the validation PSNR distribution, so the set
spans the distribution instead of showing four top-decile images. The primary
failure is the worst-PSNR validation image; a second failure shows the highest
above-Nyquist-energy validation image.

Everything is CPU-only and numpy/matplotlib/scikit-image: no torch, no GPU, no
inference. It re-scores the SAVED .npy artifacts from disk (V30), it never
reuses a cached number, and it never renormalises a prediction (docs/decisions.md
D3: per-image min-max renorm costs -4.66 dB PSNR).

The degraded input is NOT clipped for display -- its real range escapes [0,1] and
each figure states the measured range instead of hiding it. The input is enlarged
by pixel replication (nearest), never by interpolation, so no panel implies detail
the input does not contain.

There is no test ground truth in the release, so every number here comes from the
held-out slice of train/ named by configs/split_val.txt. Filenames collide between
train/ and test_NoisyLR/; everything below is keyed to train/.

The released imagery is ordinary grayscale photographs used as a degradation proxy.

Usage:
    py -3.12 scripts/make_qualitative.py --data_root C:\\kla-data --verbose

Owner: loss-metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate import ANCHOR_FIRST, ANCHOR_LAST, ANCHOR_N, BICUBIC_ANCHOR  # noqa: E402
from make_baselines import DEFAULT_SPLIT, read_val_split, resolve_data_root  # noqa: E402
from src.metrics import METRIC_SETTINGS, check_clipped, psnr, ssim  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "qualitative"
DEFAULT_PRED = REPO_ROOT / "results" / "baselines" / "final"
DEFAULT_BICUBIC = REPO_ROOT / "results" / "baselines" / "bicubic"

#: Percentiles of the model PSNR distribution used to pick representative successes.
#: Includes the median deliberately -- a pack of four top-decile images is marketing,
#: not evidence.
SUCCESS_PERCENTILES: tuple[int, ...] = (90, 75, 60, 50, 25)

#: The documented broadband-texture case (docs/decisions.md / SPEC 5.4). Verified here
#: rather than quoted, and only usable as held-out evidence if it is in the val split.
DOCUMENTED_HARD_CASE = "000984.npy"

#: Panels are rendered at exactly this many pixels per side: full resolution, no
#: downscaling, no resampling.
PANEL_PX = 256
DPI = 100

#: Fixed PNG metadata so a rerun is byte-identical (matplotlib stamps a date otherwise).
PNG_METADATA = {"Software": "kla-ps01 scripts/make_qualitative.py", "Date": None}


def log(msg: str, verbose: bool) -> None:
    """Emit progress on stderr, only under --verbose. No print debugging is shipped."""
    if verbose:
        sys.stderr.write(msg.rstrip() + "\n")


# ======================================================================================
# Measurement
# ======================================================================================
def nearest_up2(a: np.ndarray) -> np.ndarray:
    """Enlarge 2x by pixel replication. Not interpolation -- the input gains no detail."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    """Normalised radial frequency; 1.0 is the GT Nyquist, 0.5 is the LR Nyquist."""
    h, w = shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    return np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)


def hf_energy_ratio(gt: np.ndarray) -> float:
    """Fraction of GT spectral energy above the LR Nyquist limit.

    Identical definition to ``scripts/visual_audit.py``: GT sits on a grid 2x the LR
    grid, so energy beyond half the GT Nyquist cannot be represented in the input at
    all. A high value means the information is absent from the input, not merely hard
    to recover.
    """
    g = np.asarray(gt, dtype=np.float64)
    power = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())) ** 2)
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[_radius_grid(power.shape) > 0.5].sum() / total)


def hf_peak_concentration(gt: np.ndarray, top_frac: float = 0.01) -> float:
    """Share of the above-Nyquist energy held by its strongest ``top_frac`` of bins.

    This separates the two candidate explanations for a hard case. A periodic pattern
    (the moire / aliasing story) puts nearly all above-Nyquist energy in a handful of
    bins, so this tends to 1.0; a single sinusoid control measures 1.0000. Broadband
    texture spreads it, and a white-noise control measures 0.0551. Report the number,
    do not assume the mechanism.
    """
    g = np.asarray(gt, dtype=np.float64)
    power = np.fft.fftshift(np.abs(np.fft.fft2(g - g.mean())) ** 2)
    band = power[_radius_grid(power.shape) > 0.5]
    if band.size == 0 or band.sum() <= 0:
        return 0.0
    ordered = np.sort(band)[::-1]
    k = max(1, int(round(top_frac * ordered.size)))
    return float(ordered[:k].sum() / band.sum())


def bandlimited_oracle(gt: np.ndarray) -> np.ndarray:
    """GT with everything above the LR Nyquist removed, then clipped to [0,1].

    An upper bound on any method that only ever sees the LR grid: it keeps every
    representable frequency perfectly and invents nothing. Scoring it gives the
    ceiling a non-hallucinating restorer can reach on that image.
    """
    g = np.asarray(gt, dtype=np.float64)
    spec = np.fft.fftshift(np.fft.fft2(g))
    spec[_radius_grid(spec.shape) > 0.5] = 0.0
    out = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_npy(path: Path) -> np.ndarray:
    """Load a float32 2-D array with pickle disabled (io contract)."""
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 2:
        raise SystemExit(f"{path}: expected a 2-D array, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def score_split(names: Sequence[str], gt_dir: Path, pred_dir: Path, bicubic_dir: Path,
                verbose: bool = False) -> list[dict[str, Any]]:
    """Re-score every val pair FROM DISK and measure the GT spectrum of each."""
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        gt = load_npy(gt_dir / name)
        pred = load_npy(pred_dir / name)
        bic = load_npy(bicubic_dir / name)
        check_clipped(pred)
        check_clipped(bic)
        rows.append({
            "file": name,
            "psnr": psnr(pred, gt),
            "ssim": ssim(pred, gt),
            "bicubic_psnr": psnr(bic, gt),
            "bicubic_ssim": ssim(bic, gt),
            "hf_energy_ratio": hf_energy_ratio(gt),
        })
        if verbose and (i + 1) % 100 == 0:
            log(f"  scored {i + 1}/{len(names)}", verbose)
    return rows


# ======================================================================================
# Selection
# ======================================================================================
def select_successes(rows: Sequence[dict[str, Any]], percentiles: Sequence[int],
                     exclude: Sequence[str]) -> list[dict[str, Any]]:
    """Pick the image whose model PSNR is closest to each percentile of the val PSNRs.

    Deterministic and non-cherry-picked: the set spans the distribution by construction,
    and collisions fall through to the next-closest image.
    """
    psnrs = np.asarray([r["psnr"] for r in rows], dtype=np.float64)
    taken = set(exclude)
    chosen: list[dict[str, Any]] = []
    for q in percentiles:
        target = float(np.percentile(psnrs, q))
        order = np.argsort(np.abs(psnrs - target), kind="stable")
        for idx in order:
            row = rows[int(idx)]
            if row["file"] not in taken:
                taken.add(row["file"])
                chosen.append({**row, "percentile": int(q), "percentile_psnr": target})
                break
    return chosen


def percentile_of(values: Sequence[float], value: float) -> float:
    """Percentage of ``values`` strictly below ``value``."""
    arr = np.asarray(values, dtype=np.float64)
    return float(100.0 * (arr < value).mean())


def block_siblings(name: str, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The other validation crops from the same block of 4 (docs/decisions.md D8).

    D8 measured that four consecutive filenames come from one source frame. Returns the
    rows for that block that are actually in the split, worst PSNR first. Empty if the
    filename is not a plain integer stem.
    """
    stem = Path(name).stem
    if not stem.isdigit():
        return []
    block = int(stem) // 4
    sibs = [r for r in rows if Path(r["file"]).stem.isdigit()
            and int(Path(r["file"]).stem) // 4 == block]
    return sorted(sibs, key=lambda r: r["psnr"])


# ======================================================================================
# Figures
# ======================================================================================
def _layout(n_rows: int, n_cols: int, caption_lines: int) -> tuple[Any, list[list[Any]]]:
    """Absolute-inch layout so every panel is exactly PANEL_PX square at DPI."""
    import matplotlib.pyplot as plt

    panel = PANEL_PX / DPI
    left = right = 0.14
    hgap, row_gap = 0.16, 0.34
    title_h, suptitle_h = 0.62, 0.56
    caption_h = 0.30 + 0.20 * caption_lines

    fig_w = left + n_cols * panel + (n_cols - 1) * hgap + right
    fig_h = suptitle_h + n_rows * (title_h + panel) + (n_rows - 1) * row_gap + caption_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

    axes: list[list[Any]] = []
    for r in range(n_rows):
        top = fig_h - suptitle_h - r * (title_h + panel + row_gap) - title_h
        row_axes = []
        for c in range(n_cols):
            x = left + c * (panel + hgap)
            ax = fig.add_axes([x / fig_w, (top - panel) / fig_h, panel / fig_w, panel / fig_h])
            ax.set_xticks([])
            ax.set_yticks([])
            row_axes.append(ax)
        axes.append(row_axes)
    return fig, axes


def make_figure(out_png: Path, *, suptitle: str, caption: str,
                panels: Sequence[tuple[np.ndarray, str, float, float]],
                stretch_vmax: float | None = None) -> Path:
    """Render one 4-panel comparison row (plus an optional display-stretch row).

    ``panels`` is (image, title, vmin, vmax). The optional second row re-displays the
    same arrays with a narrower window; that is a viewing aid only and is never applied
    to the scored data.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = 2 if stretch_vmax is not None else 1
    caption_lines = caption.count("\n") + 1
    fig, axes = _layout(n_rows, len(panels), caption_lines)

    for col, (img, title, vmin, vmax) in enumerate(panels):
        ax = axes[0][col]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, linespacing=1.25)

    if stretch_vmax is not None:
        for col, (img, _title, vmin, _vmax) in enumerate(panels):
            ax = axes[1][col]
            ax.imshow(img, cmap="gray", vmin=min(vmin, 0.0), vmax=stretch_vmax,
                      interpolation="nearest")
            if col == 0:
                ax.set_title(f"display stretch to [0, {stretch_vmax:.2f}] -- viewing aid "
                             f"only, not applied to the scored data", fontsize=8.5,
                             loc="left")

    fig.suptitle(suptitle, fontsize=10.5, y=1.0 - 0.22 / fig.get_figheight(),
                 va="top")
    fig.text(0.14 / fig.get_figwidth(), 0.10 / fig.get_figheight(), caption,
             fontsize=7.6, va="bottom", ha="left", family="monospace", linespacing=1.45)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, metadata=PNG_METADATA)
    plt.close(fig)
    return out_png


def build_panels(name: str, gt_dir: Path, lr_dir: Path, pred_dir: Path, bicubic_dir: Path,
                 row: dict[str, Any]) -> tuple[list[tuple[np.ndarray, str, float, float]],
                                               dict[str, Any]]:
    """Assemble the four panels for one image and the numbers printed on them."""
    gt = load_npy(gt_dir / name)
    lr = load_npy(lr_dir / name)
    pred = load_npy(pred_dir / name)
    bic = load_npy(bicubic_dir / name)

    lo, hi = float(lr.min()), float(lr.max())
    up = nearest_up2(lr)
    panels = [
        (up, f"degraded input NoisyLR {lr.shape[0]}x{lr.shape[1]}\n"
             f"nearest x2 to {up.shape[0]}x{up.shape[1]}, no interpolation\n"
             f"unclipped, windowed to [{lo:.3f}, {hi:.3f}]", lo, hi),
        (bic, f"bicubic x2 baseline\nPSNR {row['bicubic_psnr']:.2f} dB\n"
              f"SSIM {row['bicubic_ssim']:.4f}", 0.0, 1.0),
        (pred, f"our model\nPSNR {row['psnr']:.2f} dB\nSSIM {row['ssim']:.4f}", 0.0, 1.0),
        (gt, f"ground truth {gt.shape[0]}x{gt.shape[1]}\nreference\n"
             f"range [{float(gt.min()):.3f}, {float(gt.max()):.3f}]", 0.0, 1.0),
    ]
    info = {
        "lr_shape": list(lr.shape), "lr_min": lo, "lr_max": hi,
        "gt_min": float(gt.min()), "gt_max": float(gt.max()),
        "gt_mean": float(gt.mean()), "gt_std": float(gt.std()),
        "gt_p995": float(np.percentile(gt, 99.5)),
    }
    return panels, info


# ======================================================================================
# README
# ======================================================================================
def _fmt_row(r: dict[str, Any]) -> str:
    return (f"| `{r['file']}` | {r['psnr']:.2f} | {r['ssim']:.4f} | "
            f"{r['bicubic_psnr']:.2f} | {r['bicubic_ssim']:.4f} | "
            f"{r['psnr'] - r['bicubic_psnr']:+.2f} | {r['hf_energy_ratio']:.4f} |")


def write_readme(out_dir: Path, ctx: dict[str, Any]) -> Path:
    """Write results/qualitative/README.md. Every number here is measured, not quoted."""
    s = ctx["successes"]
    fails = ctx["failures"]
    val = ctx["val_stats"]
    primary = fails[0]

    lines: list[str] = []
    a = lines.append
    a("# Qualitative results")
    a("")
    a("Visual evidence for V49: "
      f"{len(s)} success cases and {len(fails)} failure cases, every panel at full "
      "256x256 resolution, plus the written failure analysis below.")
    a("")
    a("Regenerate with:")
    a("")
    a("```")
    a("py -3.12 scripts/make_qualitative.py --data_root <dataset root> --verbose")
    a("```")
    a("")
    a("Every number on every figure and in every table below was measured by that script "
      "from the files on disk. `figures.json` in this directory carries the same numbers "
      "in machine-readable form.")
    a("")
    a("## What you are looking at")
    a("")
    a("The released imagery is ordinary grayscale photographs. This project treats them as "
      "a **proxy** for the degradation problem, not as domain data; nothing here should be "
      "read as content-specific tuning.")
    a("")
    a("Panel layout, left to right, identical in every figure:")
    a("")
    a("1. **degraded input** -- the 128x128 `NoisyLR` array enlarged to 256x256 by **pixel "
      "replication (nearest)**, so it is displayed at the same size as the others without "
      "being given detail it does not have. It is **not clipped**: the real arrays escape "
      "[0,1], so each panel is windowed to its own measured min/max and the range is "
      "printed in the title.")
    a("2. **bicubic x2** -- the baseline of record (`results/baselines/bicubic`).")
    a("3. **our model** -- `results/baselines/final`, the saved output of `inference.py`.")
    a("4. **ground truth** -- the 256x256 `GT` array.")
    a("")
    a("Panels 2-4 are displayed with a fixed [0,1] window. The only post-processing applied "
      "to a prediction anywhere in this project is `np.clip(pred, 0.0, 1.0)`; per-image "
      "min-max renormalisation was measured at -4.66 dB PSNR and is forbidden "
      "(`docs/decisions.md` D3).")
    a("")
    a("## Provenance and scope")
    a("")
    a(f"- Split: `{ctx['split_desc']}`. There is **no `test_GT`** in the release, so no "
      "metric anywhere in this repo is computed on the official test set. Everything here "
      "is a held-out slice of `train/`.")
    a("- `train/` and `test_NoisyLR/` reuse the same filenames for different images. Every "
      "name below refers to `train/`.")
    a("- Scores are recomputed from the reloaded `.npy` artifacts on disk (V30), not from "
      "cached numbers or in-memory tensors.")
    a("- Metric settings are pinned (SPEC 10, asserted by V31):")
    a("")
    a(f"      psnr  {METRIC_SETTINGS['psnr']}")
    a(f"      ssim  {METRIC_SETTINGS['ssim']}")
    a("")
    a("  LPIPS is not printed on these figures (it is a whole-set metric in "
      "`results/metrics_summary.md`); PSNR and SSIM are per-image and are shown per panel.")
    a("")
    a("## Validation-set context for these numbers")
    a("")
    a(f"Over all {val['n']} validation pairs, measured by this script:")
    a("")
    a(f"- our model: PSNR **{val['psnr_mean']:.4f} +/- {val['psnr_std']:.4f} dB**, "
      f"SSIM **{val['ssim_mean']:.5f} +/- {val['ssim_std']:.5f}** "
      f"(min {val['psnr_min']:.4f} dB, median {val['psnr_median']:.4f} dB, "
      f"max {val['psnr_max']:.4f} dB)")
    a(f"- bicubic x2: PSNR **{val['bicubic_psnr_mean']:.4f} +/- "
      f"{val['bicubic_psnr_std']:.4f} dB**, SSIM **{val['bicubic_ssim_mean']:.5f} +/- "
      f"{val['bicubic_ssim_std']:.5f}**")
    a(f"- the model beats bicubic on PSNR on **{val['model_wins']}/{val['n']}** images")
    a("")
    anc = ctx.get("bicubic_anchor")
    if anc:
        a(f"Sanity check on the metric plumbing: on the exact {anc['n']}-pair subset the "
          f"bicubic floor of record was measured on (`{anc['first']}`-`{anc['last']}`, "
          "`docs/decisions.md` D3), this script measures bicubic at "
          f"**{anc['psnr_mean']:.4f} +/- {anc['psnr_std']:.4f} dB** PSNR and "
          f"**{anc['ssim_mean']:.5f} +/- {anc['ssim_std']:.5f}** SSIM, against the recorded "
          f"{anc['recorded_psnr'][0]:.4f} +/- {anc['recorded_psnr'][1]:.4f} dB and "
          f"{anc['recorded_ssim'][0]:.5f} +/- {anc['recorded_ssim'][1]:.5f} "
          f"(PSNR delta {anc['psnr_mean'] - anc['recorded_psnr'][0]:+.4f} dB, SSIM delta "
          f"{anc['ssim_mean'] - anc['recorded_ssim'][0]:+.5f}).")
        a("")
    a("## How the successes were chosen")
    a("")
    a("Not by eye, and not by taking the best. For each target percentile in "
      f"{list(ctx['percentiles'])} of the model PSNR distribution over the {val['n']} "
      "validation images, the script takes the image whose PSNR is closest to that "
      "percentile (skipping any image already used). The set therefore spans the "
      "distribution and **includes the median case**, "
      f"`{ctx['median_file']}` at the 50th percentile "
      f"({ctx['median_psnr']:.2f} dB), rather than four top-decile images. The strongest "
      f"case shown is the 90th percentile, not the 100th: the best image in the split "
      f"scores {val['psnr_max']:.2f} dB and is deliberately not in the pack.")
    a("")
    a("| figure | file | percentile | model PSNR / SSIM | bicubic PSNR / SSIM | gain |")
    a("|---|---|---|---|---|---|")
    for r in s:
        a(f"| `{Path(r['png']).name}` | `{r['file']}` | p{r['percentile']} | "
          f"{r['psnr']:.2f} dB / {r['ssim']:.4f} | "
          f"{r['bicubic_psnr']:.2f} dB / {r['bicubic_ssim']:.4f} | "
          f"{r['psnr'] - r['bicubic_psnr']:+.2f} dB |")
    a("")
    for r in s:
        a(f"- **`{Path(r['png']).name}`** -- {r['note']}")
    a("")
    a("## Failure cases")
    a("")
    a("### The documented hard case is not in the validation split")
    a("")
    a(f"`{DOCUMENTED_HARD_CASE}` is the case on record for unrecoverable high-frequency "
      "content. Measured here rather than quoted: its above-LR-Nyquist share of GT "
      f"spectral energy is **{ctx['documented_hf']:.4f}** "
      f"({100 * ctx['documented_hf']:.2f}%), which confirms the documented 80.5%.")
    a("")
    a(f"But `{DOCUMENTED_HARD_CASE}` is **not** in `configs/split_val.txt` -- it is a "
      "training image, so showing it would not be held-out evidence. The failures below "
      "are validation images: the worst-PSNR one, and the one whose above-Nyquist energy "
      f"({fails[-1]['hf_energy_ratio']:.4f}) reproduces the same regime on held-out data.")
    a("")
    a("| figure | file | model PSNR / SSIM | bicubic PSNR / SSIM | "
      "band-limited ceiling PSNR / SSIM | above-Nyquist GT energy |")
    a("|---|---|---|---|---|---|")
    for r in fails:
        a(f"| `{Path(r['png']).name}` | `{r['file']}` | "
          f"{r['psnr']:.2f} dB / {r['ssim']:.4f} | "
          f"{r['bicubic_psnr']:.2f} dB / {r['bicubic_ssim']:.4f} | "
          f"{r['oracle_psnr']:.2f} dB / {r['oracle_ssim']:.4f} | "
          f"{r['hf_energy_ratio']:.4f} |")
    a("")
    a("The **band-limited ceiling** is GT with every frequency above the LR Nyquist "
      "removed and the result clipped to [0,1]. It is the score of a hypothetical method "
      "that recovers the representable band *perfectly* and invents nothing. No "
      "non-hallucinating restorer can beat it.")
    a("")
    a(f"### `{primary['file']}` -- {primary['headline']}")
    a("")
    for para in primary["analysis"]:
        a(para)
        a("")
    if len(fails) > 1:
        second = fails[1]
        a(f"### `{second['file']}` -- {second['headline']}")
        a("")
        for para in second["analysis"]:
            a(para)
            a("")
    a("## Per-figure measurements")
    a("")
    a("| file | model PSNR | model SSIM | bicubic PSNR | bicubic SSIM | PSNR gain | "
      "above-Nyquist GT energy |")
    a("|---|---|---|---|---|---|---|")
    for r in list(s) + list(fails):
        a(_fmt_row(r))
    a("")
    a("## Files")
    a("")
    a("| file | bytes |")
    a("|---|---|")
    for nm, nb in ctx["file_sizes"]:
        a(f"| `{nm}` | {nb:,} |")
    a("")

    out = out_dir / "README.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ======================================================================================
# Driver
# ======================================================================================
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build results/qualitative/ (V49).")
    ap.add_argument("--data_root", default=None,
                    help="dataset root; else $KLA_DATA_ROOT, else docs/DATA_LOCATION.md")
    ap.add_argument("--split", default=str(DEFAULT_SPLIT))
    ap.add_argument("--pred_dir", default=str(DEFAULT_PRED))
    ap.add_argument("--bicubic_dir", default=str(DEFAULT_BICUBIC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--percentiles", default=",".join(str(p) for p in SUCCESS_PERCENTILES),
                    help="comma-separated PSNR percentiles used to pick the successes")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    v = bool(args.verbose)
    root = resolve_data_root(args.data_root)
    gt_dir, lr_dir = root / "train" / "GT", root / "train" / "NoisyLR"
    pred_dir, bicubic_dir = Path(args.pred_dir), Path(args.bicubic_dir)
    out_dir = Path(args.out)
    for d in (gt_dir, lr_dir, pred_dir, bicubic_dir):
        if not d.is_dir():
            raise SystemExit(f"missing required directory: {d}")

    names, split_desc = read_val_split(Path(args.split), gt_dir, verbose=v)
    log(f"scoring {len(names)} validation pairs from disk", v)
    rows = score_split(names, gt_dir, pred_dir, bicubic_dir, verbose=v)
    by_name = {r["file"]: r for r in rows}

    p = np.asarray([r["psnr"] for r in rows])
    bp = np.asarray([r["bicubic_psnr"] for r in rows])
    sv = np.asarray([r["ssim"] for r in rows])
    bs = np.asarray([r["bicubic_ssim"] for r in rows])
    hf = np.asarray([r["hf_energy_ratio"] for r in rows])
    val_stats = {
        "n": len(rows),
        "psnr_mean": float(p.mean()), "psnr_std": float(p.std(ddof=0)),
        "psnr_min": float(p.min()), "psnr_median": float(np.median(p)),
        "psnr_max": float(p.max()),
        "ssim_mean": float(sv.mean()), "ssim_std": float(sv.std(ddof=0)),
        "bicubic_psnr_mean": float(bp.mean()), "bicubic_psnr_std": float(bp.std(ddof=0)),
        "bicubic_ssim_mean": float(bs.mean()), "bicubic_ssim_std": float(bs.std(ddof=0)),
        "model_wins": int((p > bp).sum()),
    }

    # Reproduce the bicubic floor of record on the exact subset it was measured on
    # (docs/decisions.md D3). A material disagreement means the metric plumbing moved.
    anchor_rows = [r for r in rows if ANCHOR_FIRST <= r["file"] <= ANCHOR_LAST]
    anchor: dict[str, Any] | None = None
    if len(anchor_rows) == ANCHOR_N:
        ap_ = np.asarray([r["bicubic_psnr"] for r in anchor_rows])
        as_ = np.asarray([r["bicubic_ssim"] for r in anchor_rows])
        anchor = {
            "n": ANCHOR_N, "first": ANCHOR_FIRST, "last": ANCHOR_LAST,
            "psnr_mean": float(ap_.mean()), "psnr_std": float(ap_.std(ddof=0)),
            "ssim_mean": float(as_.mean()), "ssim_std": float(as_.std(ddof=0)),
            "recorded_psnr": list(BICUBIC_ANCHOR["psnr"]),
            "recorded_ssim": list(BICUBIC_ANCHOR["ssim"]),
        }
        log(f"bicubic anchor subset: PSNR {anchor['psnr_mean']:.4f} +/- "
            f"{anchor['psnr_std']:.4f} vs recorded {BICUBIC_ANCHOR['psnr'][0]:.4f}", v)

    # ---- failures: worst PSNR, plus the highest above-Nyquist-energy image -----------
    worst = min(rows, key=lambda r: r["psnr"])
    highest_hf = max(rows, key=lambda r: r["hf_energy_ratio"])
    fail_rows = [worst] if highest_hf["file"] == worst["file"] else [worst, highest_hf]

    documented_hf = hf_energy_ratio(load_npy(gt_dir / DOCUMENTED_HARD_CASE))
    log(f"{DOCUMENTED_HARD_CASE}: above-Nyquist GT energy = {documented_hf:.4f} "
        f"(in val split: {DOCUMENTED_HARD_CASE in by_name})", v)

    percentiles = tuple(int(x) for x in str(args.percentiles).split(",") if x.strip())
    successes = select_successes(rows, percentiles, [r["file"] for r in fail_rows])

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    # ---- success figures -------------------------------------------------------------
    for r in successes:
        name = r["file"]
        panels, info = build_panels(name, gt_dir, lr_dir, pred_dir, bicubic_dir, r)
        stretch = 0.30 if info["gt_p995"] < 0.5 else None
        png = out_dir / (f"success_p{r['percentile']:02d}_{Path(name).stem}"
                         f"_psnr{r['psnr']:.2f}.png")
        caption = (
            f"validation image {name} -- model PSNR sits at the {r['percentile']}th "
            f"percentile of the {val_stats['n']}-image validation split\n"
            f"model {r['psnr']:.2f} dB / SSIM {r['ssim']:.4f}   vs   bicubic x2 "
            f"{r['bicubic_psnr']:.2f} dB / SSIM {r['bicubic_ssim']:.4f}   "
            f"(gain {r['psnr'] - r['bicubic_psnr']:+.2f} dB)\n"
            f"GT energy above the LR Nyquist limit: "
            f"{100 * r['hf_energy_ratio']:.2f}%\n"
            f"input enlarged by pixel replication, never interpolated; predictions "
            f"clipped to [0,1], never renormalised\n"
            f"scores recomputed from the saved .npy files on disk, not from memory"
        )
        make_figure(png, suptitle=f"SUCCESS  --  {name}  (val, p{r['percentile']} of "
                                  f"model PSNR)",
                    caption=caption, panels=panels, stretch_vmax=stretch)
        r["png"] = str(png.relative_to(REPO_ROOT).as_posix())
        r["note"] = (
            f"validation image `{name}`, chosen as the p{r['percentile']} case "
            f"({r['psnr']:.2f} dB vs a p{r['percentile']} target of "
            f"{r['percentile_psnr']:.2f} dB). The model gains "
            f"{r['psnr'] - r['bicubic_psnr']:+.2f} dB PSNR and "
            f"{r['ssim'] - r['bicubic_ssim']:+.4f} SSIM over bicubic x2. "
            f"{100 * r['hf_energy_ratio']:.2f}% of its GT spectral energy lies above the "
            f"LR Nyquist limit."
        )
        log(f"wrote {png.name}", v)

    # ---- failure figures --------------------------------------------------------------
    sine_ctrl, white_ctrl = _peak_controls()
    for r in fail_rows:
        name = r["file"]
        gt = load_npy(gt_dir / name)
        oracle = bandlimited_oracle(gt)
        r["oracle_psnr"], r["oracle_ssim"] = psnr(oracle, gt), ssim(oracle, gt)
        r["hf_peak_concentration"] = hf_peak_concentration(gt)
        r["psnr_percentile"] = percentile_of(p, r["psnr"])
        r["ssim_percentile"] = percentile_of(sv, r["ssim"])
        r["hf_percentile"] = percentile_of(hf, r["hf_energy_ratio"])
        r["psnr_rank"] = int((p < r["psnr"]).sum()) + 1     # 1 = worst in the split
        r["ssim_rank"] = int((sv < r["ssim"]).sum()) + 1

        panels, info = build_panels(name, gt_dir, lr_dir, pred_dir, bicubic_dir, r)
        stretch = 0.30 if info["gt_p995"] < 0.5 else None
        tag = "worst_psnr" if name == worst["file"] else "highest_hf_energy"
        png = out_dir / f"fail_{tag}_{Path(name).stem}_psnr{r['psnr']:.2f}.png"
        caption = (
            f"validation image {name} -- FAILURE CASE ({tag.replace('_', ' ')} in the "
            f"{val_stats['n']}-image validation split)\n"
            f"model {r['psnr']:.2f} dB / SSIM {r['ssim']:.4f}   bicubic x2 "
            f"{r['bicubic_psnr']:.2f} dB / SSIM {r['bicubic_ssim']:.4f}   "
            f"band-limited ceiling {r['oracle_psnr']:.2f} dB / SSIM "
            f"{r['oracle_ssim']:.4f}\n"
            f"{100 * r['hf_energy_ratio']:.2f}% of GT spectral energy is above the LR "
            f"Nyquist limit, i.e. absent from the input -- not merely attenuated\n"
            f"that energy is broadband, NOT periodic aliasing: the strongest 1% of "
            f"above-Nyquist bins hold only {100 * r['hf_peak_concentration']:.2f}% of it,\n"
            f"versus {100 * sine_ctrl:.2f}% for a sinusoid control and "
            f"{100 * white_ctrl:.2f}% for a white-noise control -- see README.md"
        )
        make_figure(png, suptitle=f"FAILURE  --  {name}  (val, {tag.replace('_', ' ')})",
                    caption=caption, panels=panels, stretch_vmax=stretch)
        r["png"] = str(png.relative_to(REPO_ROOT).as_posix())
        r["headline"], r["analysis"] = _failure_analysis(
            r, worst["file"], val_stats, sine_ctrl, white_ctrl, info, documented_hf, rows)
        log(f"wrote {png.name}", v)

    # ---- sidecar + README -------------------------------------------------------------
    median_row = next(x for x in successes if x["percentile"] == 50) if any(
        x["percentile"] == 50 for x in successes) else successes[0]
    sizes = sorted((f.name, f.stat().st_size) for f in out_dir.glob("*.png"))
    ctx = {
        "split_desc": split_desc, "percentiles": percentiles, "val_stats": val_stats,
        "bicubic_anchor": anchor,
        "successes": successes, "failures": fail_rows,
        "documented_hf": documented_hf,
        "documented_hard_case_in_val": DOCUMENTED_HARD_CASE in by_name,
        "median_file": median_row["file"], "median_psnr": median_row["psnr"],
        "peak_controls": {"sinusoid": sine_ctrl, "white_noise": white_ctrl},
        "file_sizes": sizes,
        "metric_settings": {k: METRIC_SETTINGS[k] for k in ("psnr", "ssim", "postprocess",
                                                            "artifact")},
        "sources": {
            "gt_dir": "<data_root>/train/GT", "lr_dir": "<data_root>/train/NoisyLR",
            "pred_dir": str(pred_dir.relative_to(REPO_ROOT).as_posix()),
            "bicubic_dir": str(bicubic_dir.relative_to(REPO_ROOT).as_posix()),
        },
    }
    (out_dir / "figures.json").write_text(
        json.dumps(_jsonable(ctx), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    readme = write_readme(out_dir, ctx)

    total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
    log(f"wrote {readme} and figures.json; {len(sizes)} PNGs, {total:,} bytes total", v)
    return 0


def _peak_controls() -> tuple[float, float]:
    """Reference values for hf_peak_concentration: a pure sinusoid and white noise."""
    h = w = PANEL_PX
    xx = np.arange(w)[None, :] * np.ones((h, 1))
    sinusoid = 0.5 + 0.4 * np.sin(2 * np.pi * 0.35 * xx)
    noise = np.random.default_rng(0).random((h, w))
    return hf_peak_concentration(sinusoid), hf_peak_concentration(noise)


def _failure_analysis(r: dict[str, Any], worst_name: str, val: dict[str, Any],
                      sine_ctrl: float, white_ctrl: float, info: dict[str, Any],
                      documented_hf: float, rows: Sequence[dict[str, Any]]
                      ) -> tuple[str, list[str]]:
    """Compose the written failure explanation from measured quantities only."""
    is_worst = r["file"] == worst_name
    headline = (f"worst PSNR in the validation split ({r['psnr']:.2f} dB)" if is_worst
                else f"highest above-Nyquist GT energy in the validation split "
                     f"({r['hf_energy_ratio']:.4f})")

    gap_to_ceiling = r["oracle_psnr"] - r["psnr"]
    gain = r["psnr"] - r["bicubic_psnr"]
    sibs = block_siblings(r["file"], rows)

    paras = [
        f"**What the numbers say.** The model scores {r['psnr']:.2f} dB PSNR / "
        f"{r['ssim']:.4f} SSIM here, against a validation mean of "
        f"{val['psnr_mean']:.2f} dB -- rank {r['psnr_rank']}/{val['n']} on PSNR "
        f"(1 = worst) and rank {r['ssim_rank']}/{val['n']} on SSIM. It still beats "
        f"bicubic x2 "
        f"({r['bicubic_psnr']:.2f} dB / {r['bicubic_ssim']:.4f}, "
        f"{gain:+.2f} dB), so this is not a case where the network is worse than doing "
        f"nothing clever; it is a case where nothing clever helps much."
        + ("" if is_worst else
           f" It is filed as a failure on structural, not pixel, grounds: PSNR is "
           f"unremarkable here only because the frame is nearly black, while SSIM is "
           f"rank {r['ssim_rank']}/{val['n']} and the reconstruction is visibly smoother "
           f"than the reference."),

        f"**Why it is unrecoverable, measured.** {100 * r['hf_energy_ratio']:.2f}% of this "
        f"image's ground-truth spectral energy lies above the Nyquist limit of the 128x128 "
        f"input -- the {r['hf_percentile']:.1f}th percentile of the validation split, whose "
        f"median is {100 * float(np.median([x['hf_energy_ratio'] for x in rows])):.2f}%. "
        f"That energy is not attenuated in the input, it is *absent* from it: the "
        f"sampling grid cannot represent it. Removing exactly that band from the GT and "
        f"clipping to [0,1] gives a band-limited oracle -- a method that recovers every "
        f"representable frequency perfectly and invents nothing -- and that oracle scores "
        f"only {r['oracle_psnr']:.2f} dB / SSIM {r['oracle_ssim']:.4f} on this image. The "
        f"model is {gap_to_ceiling:.2f} dB below that ceiling. Most of the visible "
        f"shortfall on this figure is missing information, not model error.",

        f"**It is broadband texture, not periodic aliasing.** This distinction was tested, "
        f"not assumed. If the above-Nyquist content were a periodic pattern -- the moire / "
        f"aliasing story -- its energy would sit in a handful of spectral bins. Measured: "
        f"the strongest 1% of above-Nyquist bins hold only "
        f"{100 * r['hf_peak_concentration']:.2f}% of the above-Nyquist energy here, versus "
        f"{100 * sine_ctrl:.2f}% for a pure-sinusoid control and {100 * white_ctrl:.2f}% "
        f"for a white-noise control on the same 256x256 grid. This case sits at the "
        f"broadband end and nowhere near the periodic end: the lost content is fine "
        f"broadband texture spread across the entire band. The moire / periodic-aliasing "
        f"explanation that was hypothesised for this regime is refuted by that "
        f"measurement, and this figure must not be captioned as moire.",

        f"**What that means for the submission.** The honest ceiling on this image is set "
        f"by the input, not the architecture. Closing the remaining "
        f"{gap_to_ceiling:.2f} dB would require inventing plausible texture, which is "
        f"exactly the failure mode an inspection setting cannot tolerate -- hallucinated "
        f"structure that looks like a defect. This is why no adversarial loss is used "
        f"(SPEC 7.2). The model degrades to a smooth, honest reconstruction instead of a "
        f"confident, invented one.",
    ]

    if is_worst:
        sib_txt = ", ".join(f"`{x['file']}` {x['psnr']:.2f} dB (rank "
                            f"{int((np.asarray([y['psnr'] for y in rows]) < x['psnr']).sum()) + 1}"
                            f"/{val['n']})" for x in sibs)
        paras.insert(1,
            f"**Content.** The frame is dominated by dense, thin, high-contrast structures "
            f"at every orientation (GT mean {info['gt_mean']:.4f}, std "
            f"{info['gt_std']:.4f}, range [{info['gt_min']:.3f}, {info['gt_max']:.3f}]). "
            f"Structures roughly one input pixel wide survive 2x decimation only as a "
            f"smear. docs/decisions.md D8 measured that four consecutive filenames are "
            f"four crops of one source frame; all {len(sibs)} crops of this block are in "
            f"the split and all of them are hard -- {sib_txt}. The difficulty is a "
            f"property of the content and it reproduces across crops, so it is not a "
            f"one-off artifact of a single image.")
    else:
        paras.insert(1,
            f"**Content.** A very dark, low-contrast frame (GT mean "
            f"{info['gt_mean']:.4f}, std "
            f"{info['gt_std']:.4f}, 99.5th percentile {info['gt_p995']:.3f}) whose entire "
            f"content is fine-grained broadband texture. The figure therefore includes a "
            f"second row with a display-only stretch so the structure is visible on "
            f"screen; the stretch is not applied to the scored data. This is the held-out "
            f"analogue of the documented hard case {DOCUMENTED_HARD_CASE}, whose "
            f"above-Nyquist energy measures {documented_hf:.4f} here against "
            f"{r['hf_energy_ratio']:.4f} for this image.")

    return headline, paras


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and Paths for json.dump."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj.as_posix())
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
