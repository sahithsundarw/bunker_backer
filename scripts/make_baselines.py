#!/usr/bin/env python3
"""Generate the required baselines (SPEC 10).

1. Bicubic x2 of raw NoisyLR  -- measured floor: 23.4247 +/- 2.8319 dB PSNR,
   0.54284 SSIM on 200 held-out train pairs with clip-to-[0,1] (docs/decisions.md D3).
2. Classical denoise -> bicubic x2.
3. Small plain U-Net at the same training budget.

Plus, optionally, the final model, so the whole SPEC 10 table comes from one command.

This script only WRITES PREDICTIONS to disk (float32 .npy, clipped to [0,1], input filename
preserved). Scoring is scripts/evaluate.py's job, and it scores the files this script wrote,
never an in-memory array (V30).

Post-processing is ``np.clip(pred, 0.0, 1.0)`` and nothing else. Per-image min-max
renormalisation was measured at -4.66 dB PSNR and is permanently rejected (docs/decisions.md
D3). The bicubic kernel is implemented here from scratch (Catmull-Rom a=-0.5, antialias off,
replicate edges, renormalised weights) so the number does not depend on any library's
undocumented resampling behaviour -- the same construction that produced the D3 anchor.

Validation split: a held-out slice of train/. There is NO test ground truth (no test_GT
folder exists), so no score can be computed against the official test set locally.

Usage:
    py -3.12 scripts/make_baselines.py --data_root C:\\kla-data
    py -3.12 scripts/make_baselines.py --baselines bicubic --limit 20 --verbose

Owner: loss-metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT_DIR = REPO_ROOT / "results" / "baselines"
DEFAULT_SPLIT = REPO_ROOT / "configs" / "split_val.txt"
FALLBACK_VAL_N = 200          # docs/decisions.md D3 used the last 200 train pairs
SCALE = 2


# ======================================================================================
# Paths and split
# ======================================================================================
def resolve_data_root(arg: str | None) -> Path:
    """--data_root, else $KLA_DATA_ROOT, else the root documented in docs/DATA_LOCATION.md.

    No dataset path is hard-coded in this file (CLAUDE.md Prime Directive 5).
    """
    if arg:
        root = Path(arg).expanduser()
    elif os.environ.get("KLA_DATA_ROOT"):
        root = Path(os.environ["KLA_DATA_ROOT"]).expanduser()
    else:
        doc = REPO_ROOT / "docs" / "DATA_LOCATION.md"
        cand: Path | None = None
        if doc.exists():
            m = re.search(r"^```\s*\n(.+?)\n```", doc.read_text(encoding="utf-8"),
                          re.DOTALL | re.MULTILINE)
            if m:
                cand = Path(m.group(1).strip())
        if cand is None:
            raise SystemExit("could not determine the dataset root: pass --data_root or set "
                             "KLA_DATA_ROOT (see docs/DATA_LOCATION.md)")
        root = cand
    if not (root / "train" / "GT").is_dir():
        raise SystemExit(f"dataset root {root} has no train/GT directory")
    return root


def read_val_split(split_path: Path, gt_dir: Path, verbose: bool = False) -> tuple[list[str], str]:
    """Return (filenames, description). Names are relative to <data_root>/train/.

    The committed list is authoritative. If it holds no entries yet, fall back to the last
    ``FALLBACK_VAL_N`` train pairs -- exactly the slice used for the docs/decisions.md D3
    anchor -- and say so loudly, because the split identity is part of every reported number.
    """
    names: list[str] = []
    if split_path.exists():
        for line in split_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                names.append(Path(s).name)
    if names:
        return names, f"{split_path.relative_to(REPO_ROOT).as_posix()} ({len(names)} pairs)"

    all_gt = sorted(p.name for p in gt_dir.glob("*.npy"))
    names = all_gt[-FALLBACK_VAL_N:]
    desc = (f"FALLBACK last {len(names)} train pairs ({names[0]}-{names[-1]}) -- "
            f"{split_path.name} has no entries yet")
    print(f"[warn] {desc}", file=sys.stderr)
    return names, desc


# ======================================================================================
# Resampling -- Catmull-Rom bicubic, implemented from scratch
# ======================================================================================
def _catmull_rom(x: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Keys bicubic with a = -0.5 (the OpenCV/PIL convention)."""
    ax = np.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    out = np.zeros_like(ax)
    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)
    out[m1] = (a + 2.0) * ax3[m1] - (a + 3.0) * ax2[m1] + 1.0
    out[m2] = a * ax3[m2] - 5.0 * a * ax2[m2] + 8.0 * a * ax[m2] - 4.0 * a
    return out


def cubic_weight_matrix(in_len: int, out_len: int) -> np.ndarray:
    """Dense (out_len x in_len) bicubic weights: centre-aligned, replicate-padded, renormalised.

    Antialias is OFF, which is correct for UPsampling and matches the kernel recovered from
    the data in docs/decisions.md D1.
    """
    scale = in_len / out_len
    centers = (np.arange(out_len) + 0.5) * scale - 0.5
    radius = 2.0
    W = np.zeros((out_len, in_len), dtype=np.float64)
    for i, c in enumerate(centers):
        lo = int(np.floor(c - radius + 0.5))
        hi = int(np.ceil(c + radius - 0.5))
        idx = np.arange(lo, hi + 1)
        w = _catmull_rom(idx - c)
        np.add.at(W[i], np.clip(idx, 0, in_len - 1), w)
        s = W[i].sum()
        if s != 0:
            W[i] /= s
    return W


_W_CACHE: dict[tuple[int, int], np.ndarray] = {}


def bicubic_upsample(img: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Separable bicubic x``scale`` upsample in float64. No clipping here."""
    h, w = img.shape
    key_r, key_c = (h, h * scale), (w, w * scale)
    for k in (key_r, key_c):
        if k not in _W_CACHE:
            _W_CACHE[k] = cubic_weight_matrix(k[0], k[1])
    return (_W_CACHE[key_r] @ img.astype(np.float64)) @ _W_CACHE[key_c].T


# ======================================================================================
# Classical denoisers (applied at LR, before upsampling)
# ======================================================================================
def median3(img: np.ndarray) -> np.ndarray:
    """3x3 median filter, reflect-padded. Pure numpy -- no scipy, no cv2."""
    p = np.pad(img.astype(np.float64), 1, mode="reflect")
    h, w = img.shape
    stack = np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3)], axis=0)
    return np.median(stack, axis=0)


def estimate_sigma_immerkaer(img: np.ndarray) -> float:
    """Immerkaer (1996) fast noise sigma estimate: MAD-free, pure numpy, no PyWavelets.

    sigma = sqrt(pi/2) / (6 (W-2)(H-2)) * sum |img * L|, L = [[1,-2,1],[-2,4,-2],[1,-2,1]].
    A single global sigma is crude here -- the noise is signal-dependent
    (var = s^2 + v*x^2, docs/decisions.md D2) -- but this is a deliberately classical baseline.
    """
    a = np.asarray(img, dtype=np.float64)
    lap = (4.0 * a[1:-1, 1:-1]
           - 2.0 * (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:])
           + (a[:-2, :-2] + a[:-2, 2:] + a[2:, :-2] + a[2:, 2:]))
    h, w = a.shape
    denom = 6.0 * max(1, (w - 2)) * max(1, (h - 2))
    return float(np.sqrt(np.pi / 2.0) * np.abs(lap).sum() / denom)


def nlm_denoise(img: np.ndarray) -> np.ndarray:
    """Non-local means with an estimated global sigma -- the BM3D-style slot in SPEC 10.

    NLM is the closest patch-matching denoiser available from the already-pinned
    dependencies. Wavelet (BayesShrink) would need PyWavelets, and a new dependency for a
    baseline is not worth it. skimage lives outside the measured inference window, so
    importing it here is free.
    """
    from skimage.restoration import denoise_nl_means

    a = img.astype(np.float64)
    sigma = estimate_sigma_immerkaer(a)
    return np.asarray(
        denoise_nl_means(a, h=1.15 * sigma, sigma=sigma, patch_size=5, patch_distance=6,
                         fast_mode=True, channel_axis=None),
        dtype=np.float64,
    )


# ======================================================================================
# Baseline definitions
# ======================================================================================
class Baseline:
    """A named prediction producer that writes clipped float32 .npy files."""

    def __init__(self, name: str, label: str, description: str,
                 fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self.name = name
        self.label = label
        self.description = description
        self.fn = fn


CLASSICAL_BASELINES: dict[str, Baseline] = {
    "bicubic": Baseline(
        "bicubic", "Bicubic x2 (raw NoisyLR)",
        "Catmull-Rom bicubic x2 of the raw noisy input, no denoising. The floor.",
        lambda a: bicubic_upsample(a),
    ),
    "median_bicubic": Baseline(
        "median_bicubic", "Median 3x3 -> bicubic x2",
        "Classical pipeline: 3x3 median denoise at LR, then bicubic x2.",
        lambda a: bicubic_upsample(median3(a)),
    ),
    "nlm_bicubic": Baseline(
        "nlm_bicubic", "Non-local means -> bicubic x2",
        "Classical pipeline: non-local-means denoise at LR, then bicubic x2 (BM3D-style).",
        lambda a: bicubic_upsample(nlm_denoise(a)),
    ),
}

#: Learned baselines need a checkpoint; they skip cleanly until one exists.
LEARNED_BASELINES: dict[str, tuple[str, str, str]] = {
    "unet_baseline": ("U-Net baseline", "weights/baseline_unet.pt",
                      "Small plain U-Net, same training budget. The learned baseline."),
    "final": ("Final model", "weights/best.pt",
              "The submitted model. Canonical end-to-end numbers come from running "
              "inference.py on the val inputs and scoring that directory."),
}

ALL_BASELINES = list(CLASSICAL_BASELINES) + list(LEARNED_BASELINES)


# ======================================================================================
# Writers
# ======================================================================================
def save_prediction(out_dir: Path, basename: str, arr: np.ndarray) -> None:
    """docs/io_contract.md reference writer: clip to [0,1], float32, filename verbatim."""
    if arr.ndim != 2:
        raise ValueError(f"{basename}: prediction must be 2-D, got {arr.shape}")
    out = np.clip(arr, 0.0, 1.0).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / basename, out)


def write_meta(out_dir: Path, **kw: Any) -> None:
    (out_dir / "pred_meta.json").write_text(json.dumps(kw, indent=2), encoding="utf-8")


def run_classical(bl: Baseline, names: list[str], lr_dir: Path, out_dir: Path,
                  split_desc: str, verbose: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t_total = 0.0
    for i, name in enumerate(names):
        lr = np.load(lr_dir / name, allow_pickle=False)   # never clip the INPUT (SPEC F5)
        t0 = time.perf_counter()
        pred = bl.fn(lr)
        t_total += time.perf_counter() - t0
        save_prediction(out_dir, name, pred)
        if verbose and (i + 1) % 50 == 0:
            print(f"  {bl.name}: {i + 1}/{len(names)}")
    meta = {
        "name": bl.name, "label": bl.label, "description": bl.description,
        "kind": "classical", "n": len(names), "split": split_desc,
        "input_dir": str(lr_dir), "sec_per_image": t_total / max(1, len(names)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "postprocess": "np.clip(pred, 0.0, 1.0) only -- no renormalisation (D3)",
    }
    write_meta(out_dir, **meta)
    return {"status": "ok", **meta}


def run_learned(name: str, names: list[str], lr_dir: Path, out_dir: Path, split_desc: str,
                ckpt_path: Path, device: str, verbose: bool = False) -> dict[str, Any]:
    """Run a checkpointed model. Skips cleanly (never crashes) if the pieces are missing."""
    label, _, description = LEARNED_BASELINES[name]
    if not ckpt_path.exists():
        return {"status": "skipped", "name": name, "label": label,
                "reason": f"checkpoint {ckpt_path} does not exist yet"}
    try:
        import torch

        from src.model import build_model
    except Exception as exc:                       # noqa: BLE001 - report, never crash
        return {"status": "skipped", "name": name, "label": label,
                "reason": f"src.model unavailable ({type(exc).__name__}: {exc})"}

    try:
        # weights_only=True: never unpickle arbitrary objects out of a checkpoint. The
        # embedded config (SPEC 9) is plain dict/str/float, which this mode allows.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
        model = build_model(model_cfg)
        state = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        dev = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        model = model.to(dev).eval()
    except Exception as exc:                       # noqa: BLE001
        return {"status": "skipped", "name": name, "label": label,
                "reason": f"could not load {ckpt_path.name} ({type(exc).__name__}: {exc})"}

    out_dir.mkdir(parents=True, exist_ok=True)
    t_total = 0.0
    with torch.no_grad():
        for i, fname in enumerate(names):
            lr = np.load(lr_dir / fname, allow_pickle=False)
            x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(dev)
            t0 = time.perf_counter()
            y = model(x)
            t_total += time.perf_counter() - t0
            save_prediction(out_dir, fname, y[0, 0].float().cpu().numpy())
            if verbose and (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1}/{len(names)}")

    meta = {
        "name": name, "label": label, "description": description, "kind": "learned",
        "n": len(names), "split": split_desc, "input_dir": str(lr_dir),
        "checkpoint": str(ckpt_path), "device": dev,
        "sec_per_image": t_total / max(1, len(names)),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "postprocess": "np.clip(pred, 0.0, 1.0) only -- no renormalisation (D3)",
    }
    write_meta(out_dir, **meta)
    return {"status": "ok", **meta}


# ======================================================================================
# CLI
# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", default=None,
                    help="dataset root (else $KLA_DATA_ROOT, else docs/DATA_LOCATION.md)")
    ap.add_argument("--out", default=str(DEFAULT_OUT_DIR),
                    help="directory that receives one subdirectory of predictions per baseline")
    ap.add_argument("--split", default=str(DEFAULT_SPLIT),
                    help="committed validation file list (never regenerated at runtime)")
    ap.add_argument("--baselines", default=",".join(ALL_BASELINES),
                    help=f"comma-separated subset of: {', '.join(ALL_BASELINES)}")
    ap.add_argument("--unet_ckpt", default=None, help="override weights/baseline_unet.pt")
    ap.add_argument("--final_ckpt", default=None, help="override weights/best.pt")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N val pairs (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    root = resolve_data_root(args.data_root)
    gt_dir = root / "train" / "GT"
    lr_dir = root / "train" / "NoisyLR"
    names, split_desc = read_val_split(Path(args.split), gt_dir, args.verbose)
    if args.limit:
        names = names[:args.limit]
        split_desc += f" [--limit {args.limit}]"

    missing = [n for n in names if not (lr_dir / n).exists() or not (gt_dir / n).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} split entries missing from the dataset, "
                         f"first: {missing[:3]}")

    requested = [b.strip() for b in args.baselines.split(",") if b.strip()]
    unknown = [b for b in requested if b not in ALL_BASELINES]
    if unknown:
        raise SystemExit(f"unknown baseline(s): {unknown}; choose from {ALL_BASELINES}")

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root

    print(f"dataset root : {root}")
    print(f"val split    : {split_desc}")
    print(f"pred root    : {out_root}")
    print("post-process : np.clip(pred, 0.0, 1.0) only -- never renormalise (D3)")
    print("NOTE         : no test_GT exists; every number is on a held-out slice of train/.")
    print()

    results: list[dict[str, Any]] = []
    for name in requested:
        out_dir = out_root / name
        t0 = time.perf_counter()
        if name in CLASSICAL_BASELINES:
            try:
                res = run_classical(CLASSICAL_BASELINES[name], names, lr_dir, out_dir,
                                    split_desc, args.verbose)
            except Exception as exc:               # noqa: BLE001 - one baseline must not kill the rest
                res = {"status": "skipped", "name": name,
                       "label": CLASSICAL_BASELINES[name].label,
                       "reason": f"{type(exc).__name__}: {exc}"}
        else:
            override = args.unet_ckpt if name == "unet_baseline" else args.final_ckpt
            ckpt = Path(override) if override else REPO_ROOT / LEARNED_BASELINES[name][1]
            res = run_learned(name, names, lr_dir, out_dir, split_desc, ckpt,
                              args.device, args.verbose)
        dt = time.perf_counter() - t0
        results.append(res)
        if res["status"] == "ok":
            print(f"[ok]   {name:16s} {res['n']} predictions -> "
                  f"{out_dir.relative_to(REPO_ROOT).as_posix()}  "
                  f"({dt:.1f}s wall, {res['sec_per_image'] * 1000:.1f} ms/img transform)")
        else:
            print(f"[skip] {name:16s} {res['reason']}")

    ok = [r for r in results if r["status"] == "ok"]
    print()
    print(f"{len(ok)}/{len(results)} baselines produced. Score them with:")
    print("  py -3.12 scripts/evaluate.py --data_root <root> "
          + " ".join(f"--preds {r['name']}={Path(args.out).as_posix()}/{r['name']}" for r in ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
