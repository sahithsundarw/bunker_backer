"""Generate before/after qualitative panels for the shipped checkpoint (submission evidence).

Not part of inference.py or scripts/verify_all.py -- a one-off evidence-generation script for
results/qualitative/. Uses matplotlib freely; that is fine here (it is banned only from
inference.py's module-level imports per CLAUDE.md STYLE).

Validation panels: NoisyLR | Restored | GT | |Restored-GT| error map, titled with PSNR/SSIM
scored with the SAME pinned settings as scripts/evaluate.py / src/metrics.py.

Final-test panels: NoisyLR | Restored only, explicitly labelled "no GT / no metric" -- the
released test set has no ground truth (SPEC, README "Result summary").

A separate, optional "D5 case" panel reproduces docs/decisions.md D5's documented honest
failure case (000984.npy): it is NOT a member of configs/split_val.txt, so its restoration is
produced with a live `inference.py --require_weights` subprocess call against a scratch
input/output directory -- the real production entry point, not a re-implemented forward pass
-- and its PSNR/SSIM is reported as illustrative only, never folded into the 400-pair mean.

Usage:
    .venv-mac/bin/python scripts/make_qualitative_examples.py \\
        --data_root /path/to/dataset \\
        --checkpoint weights/best.pt

Owner: main session (submission-evidence utility, not on the parallel-agent ownership map).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_paths import resolve_test_input_dir

DEFAULT_CKPT_PATH = REPO_ROOT / "weights/best.pt"

VAL_EXAMPLES = [
    ("001323.npy", "best"),
    ("000960.npy", "strong"),
    ("003167.npy", "typical_near_mean"),
    ("001682.npy", "typical_near_mean"),
    ("002041.npy", "weak_worst_in_split"),
    ("000900.npy", "weak_lowest_ssim_in_worst8"),
]
TEST_EXAMPLES = ["000000.npy", "000099.npy", "000199.npy", "000299.npy", "000399.npy"]
D5_FILE = "000984.npy"


def psnr_ssim(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    p = sk_psnr(gt, pred, data_range=1.0)
    s = sk_ssim(gt, pred, data_range=1.0, gaussian_weights=True, sigma=1.5,
                use_sample_covariance=False)
    return float(p), float(s)


def show(ax, img, title, vmin=0.0, vmax=1.0, cmap="gray"):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return im


def val_panel(plt, out_dir, fname, tag, lr_dir, gt_dir, pred_dir, header, subtitle_extra,
              out_prefix, checkpoint_sha):
    lr = np.load(lr_dir / fname, allow_pickle=False)
    gt = np.load(gt_dir / fname, allow_pickle=False)
    pred = np.load(pred_dir / fname, allow_pickle=False)
    psnr, ssim = psnr_ssim(gt, pred)
    err = np.abs(pred - gt)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    show(axes[0], np.clip(lr, 0.0, 1.0), f"NoisyLR {lr.shape}\n(display-clipped; raw range "
         f"[{lr.min():.3f}, {lr.max():.3f}])")
    show(axes[1], pred, f"Restored {pred.shape}\n(model output, clipped [0,1] on disk)")
    show(axes[2], gt, f"GT {gt.shape}")
    im = show(axes[3], err, f"|Restored - GT|\nmax err {err.max():.3f}", vmin=0.0,
              vmax=max(0.05, float(err.max())), cmap="inferno")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{header} | {fname} | PSNR {psnr:.2f} dB | SSIM {ssim:.4f} | "
        f"checkpoint {checkpoint_sha[:12]}...{subtitle_extra}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90 if subtitle_extra else 0.93])
    out_path = out_dir / f"{out_prefix}_{fname.replace('.npy', '')}_{tag}_psnr{psnr:.2f}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"file": fname, "tag": tag, "psnr": round(psnr, 4), "ssim": round(ssim, 4),
            "png": out_path.name}


def test_panel(plt, out_dir, fname, lr_dir, pred_dir, checkpoint_sha):
    lr = np.load(lr_dir / fname, allow_pickle=False)
    pred = np.load(pred_dir / fname, allow_pickle=False)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
    show(axes[0], np.clip(lr, 0.0, 1.0), f"NoisyLR {lr.shape}\n(display-clipped; raw range "
         f"[{lr.min():.3f}, {lr.max():.3f}])")
    show(axes[1], pred, f"Restored {pred.shape}\n(model output, clipped [0,1] on disk)")

    fig.suptitle(
        f"FINAL TEST (released test_NoisyLR) | {fname}\n"
        f"NO GROUND TRUTH -- no PSNR/SSIM/LPIPS computed or claimed | "
        f"checkpoint {checkpoint_sha[:12]}...",
        fontsize=10, color="firebrick",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    out_path = out_dir / f"finaltest_{fname.replace('.npy', '')}_no_gt.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"file": fname, "png": out_path.name}


def make_predictions(lr_dir: Path, filenames: list[str], checkpoint: Path, scratch: Path,
                     label: str) -> Path:
    """Restore selected files with the strict submission inference path."""
    tmp_in = scratch / f"{label}-in"
    tmp_out = scratch / f"{label}-out"
    tmp_in.mkdir(parents=True)
    for filename in filenames:
        shutil.copy(lr_dir / filename, tmp_in / filename)
    cmd = [sys.executable, str(REPO_ROOT / "inference.py"),
           "--input_dir", str(tmp_in), "--output_dir", str(tmp_out),
           "--weights", str(checkpoint), "--require_weights",
           "--device", "cpu", "--precision", "fp32", "--verbose"]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    return tmp_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", required=True,
                     help="dir containing train/GT and train/NoisyLR (val split lives here)")
    ap.add_argument("--final_test_lr_dir", default=None,
                     help="dir with the released test_NoisyLR inputs (default: "
                          "<data_root>/NoisyLR)")
    ap.add_argument("--final_test_pred_dir", default=None,
                    help="optional existing final-test restorations; selected files are "
                         "generated with strict inference when omitted")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT_PATH),
                    help="checkpoint used to generate and label examples")
    ap.add_argument("--val_pred_dir", default=None,
                    help="optional existing validation predictions; selected files are "
                         "generated with strict inference when omitted")
    ap.add_argument("--out_dir", default=str(REPO_ROOT / "results" / "qualitative"))
    ap.add_argument("--skip_d5", action="store_true",
                     help="skip the D5 out-of-split failure-case panel (runs inference.py)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_root = Path(args.data_root)
    val_gt_dir = data_root / "train" / "GT"
    val_lr_dir = data_root / "train" / "NoisyLR"
    test_lr_dir = (Path(args.final_test_lr_dir) if args.final_test_lr_dir else
                   resolve_test_input_dir(data_root))
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    import torch
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_metrics = checkpoint_data.get("metrics", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("val_*.png", "finaltest_*.png", "failurecase_D5_*.png"):
        for stale in out_dir.glob(pattern):
            stale.unlink()

    with tempfile.TemporaryDirectory(prefix="kla-qualitative-") as tmp:
        scratch = Path(tmp)
        val_pred_dir = (Path(args.val_pred_dir) if args.val_pred_dir else
                        make_predictions(val_lr_dir, [f for f, _ in VAL_EXAMPLES], checkpoint,
                                         scratch, "validation"))
        test_pred_dir = (Path(args.final_test_pred_dir) if args.final_test_pred_dir else
                         make_predictions(test_lr_dir, TEST_EXAMPLES, checkpoint, scratch,
                                          "final-test"))
        val_records = [
            val_panel(plt, out_dir, f, tag, val_lr_dir, val_gt_dir, val_pred_dir,
                      "Validation split", "", "val", checkpoint_sha)
            for f, tag in VAL_EXAMPLES
        ]
        test_records = [
            test_panel(plt, out_dir, f, test_lr_dir, test_pred_dir, checkpoint_sha)
            for f in TEST_EXAMPLES
        ]

        d5_record = None
        if not args.skip_d5:
            d5_pred_dir = make_predictions(val_lr_dir, [D5_FILE], checkpoint, scratch, "d5")
            d5_record = val_panel(
                plt, out_dir, D5_FILE, "D5_documented_aliasing_broadband_texture_case",
                val_lr_dir, val_gt_dir, d5_pred_dir,
                "D5 out-of-split case (NOT in configs/split_val.txt)",
                "\n80.5% of GT spectral energy is above the LR Nyquist limit "
                "(docs/decisions.md D5) -- provably unrecoverable, broadband texture, "
                "not periodic aliasing",
                "failurecase_D5", checkpoint_sha,
            )

    import json
    try:
        checkpoint_display = checkpoint.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        checkpoint_display = str(checkpoint.resolve())
    manifest = {
        "checkpoint": checkpoint_display,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_metrics": checkpoint_metrics,
        "validation_examples": val_records,
        "d5_documented_failure_case": d5_record,
        "final_test_examples": test_records,
        "note": ("final_test_examples have no ground truth; no metric is computed for them. "
                 "d5_documented_failure_case is NOT part of the 400-pair scored validation "
                 "split; its PSNR/SSIM is a single-file illustrative measurement only, not "
                 "part of the checkpoint's reported validation mean."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for r in val_records:
        print("VAL ", r)
    if d5_record:
        print("D5  ", d5_record)
    for r in test_records:
        print("TEST", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
