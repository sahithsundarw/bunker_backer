"""Plan Phase 1(d): where does the shipped model actually fail on real-SEM OOD -- edges, flat
regions, or periodic/fine structure?

Follows scripts/content_stats_sem_vs_natural.py's finding (edge density z=+3.6, local
contrast z=+2.5, flatter spectral slope z=+2.0 -- real-SEM content is dense, high-frequency
micro-texture, statistically unlike the natural-photo training content). This script checks
whether the model's actual per-pixel error CONCENTRATES on that same fine structure, which
would directly connect the content-statistics finding to the measured quality gap rather than
leaving them as two separate facts.

Runs weights/best.pt over all 45 real-SEM pairs, computes the per-pixel |pred-gt| error map,
and correlates it against per-pixel local image statistics computed on the SAME images: local
gradient magnitude (proxy for "near an edge"), local variance (proxy for "textured, not
flat"). No training, no fitting -- pure evaluation of the already-shipped checkpoint.

Usage:
    py -3.12 scripts/sem_error_localisation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import build_model  # noqa: E402

CKPT_PATH = ROOT / "weights" / "best.pt"
SEM_DIR = ROOT / "results" / "eda" / "real_sem_ood"
WIN = 7  # local-statistics window


def load_model(device: str) -> torch.nn.Module:
    ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    m = build_model(cfg.get("model", cfg))
    m.load_state_dict(ck.get("ema") or ck.get("model"), strict=True)
    return m.to(device).eval()


def infer(model: torch.nn.Module, lr: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        y = model(x)
    return np.clip(y[0, 0].float().cpu().numpy(), 0.0, 1.0)


def local_gradient_magnitude(a: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(a)
    return np.sqrt(gy ** 2 + gx ** 2)


def local_variance_map(a: np.ndarray, win: int = WIN) -> np.ndarray:
    """Local variance via a uniform box filter (mean of squares minus square of mean),
    same output shape as the input (edge-padded)."""
    from scipy.ndimage import uniform_filter
    m = uniform_filter(a, size=win, mode="reflect")
    m2 = uniform_filter(a * a, size=win, mode="reflect")
    return np.maximum(m2 - m * m, 0.0)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    gt_dir, lr_dir = SEM_DIR / "GT", SEM_DIR / "NoisyLR"
    files = sorted(gt_dir.glob("*.npy"))
    if not files:
        print(f"ERROR: no files under {gt_dir}", file=sys.stderr)
        return 2

    all_err, all_grad, all_var = [], [], []
    per_image: list[dict[str, Any]] = []
    t0 = time.time()
    for f in files:
        gt = np.load(f, allow_pickle=False).astype(np.float64)
        lr = np.load(lr_dir / f.name, allow_pickle=False)
        pred = infer(model, lr, device).astype(np.float64)
        err = np.abs(pred - gt)
        grad = local_gradient_magnitude(gt)
        var = local_variance_map(gt)
        all_err.append(err.ravel())
        all_grad.append(grad.ravel())
        all_var.append(var.ravel())
        per_image.append({"file": f.name, "mean_abs_err": float(err.mean()),
                          "mean_local_grad": float(grad.mean()),
                          "mean_local_var": float(var.mean())})

    err_v = np.concatenate(all_err)
    grad_v = np.concatenate(all_grad)
    var_v = np.concatenate(all_var)

    def pearson(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    r_grad = pearson(err_v, grad_v)
    r_var = pearson(err_v, var_v)

    # Flat vs edge/textured region comparison: split by median of each stat, compare mean err
    grad_med, var_med = np.median(grad_v), np.median(var_v)
    err_flat_by_grad = float(err_v[grad_v <= grad_med].mean())
    err_edge_by_grad = float(err_v[grad_v > grad_med].mean())
    err_flat_by_var = float(err_v[var_v <= var_med].mean())
    err_textured_by_var = float(err_v[var_v > var_med].mean())

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "checkpoint": str(CKPT_PATH),
        "n_images": len(files),
        "n_pixels_pooled": int(err_v.size),
        "pearson_err_vs_local_gradient": r_grad,
        "pearson_err_vs_local_variance": r_var,
        "mean_abs_err_below_median_gradient_ie_flatter_regions": err_flat_by_grad,
        "mean_abs_err_above_median_gradient_ie_edgier_regions": err_edge_by_grad,
        "mean_abs_err_below_median_variance_ie_flat_regions": err_flat_by_var,
        "mean_abs_err_above_median_variance_ie_textured_regions": err_textured_by_var,
        "edge_vs_flat_ratio": err_edge_by_grad / max(err_flat_by_grad, 1e-9),
        "textured_vs_flat_ratio": err_textured_by_var / max(err_flat_by_var, 1e-9),
        "per_image": per_image,
        "wall_clock_s": round(time.time() - t0, 2),
    }
    out_path = ROOT / "results" / "eda" / "sem_error_localisation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"pearson(err, local_gradient) = {r_grad:.4f}")
    print(f"pearson(err, local_variance) = {r_var:.4f}")
    print(f"mean |err|: flatter half={err_flat_by_grad:.5f}  edgier half={err_edge_by_grad:.5f}"
          f"  ratio={report['edge_vs_flat_ratio']:.2f}x")
    print(f"mean |err|: flat half={err_flat_by_var:.5f}  textured half={err_textured_by_var:.5f}"
          f"  ratio={report['textured_vs_flat_ratio']:.2f}x")

    # A couple of example error-map panels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = sorted(per_image, key=lambda r: -r["mean_abs_err"])[:2]
    fig, axes = plt.subplots(len(worst), 4, figsize=(16, 4 * len(worst)))
    if len(worst) == 1:
        axes = axes[None, :]
    for row, rec in enumerate(worst):
        gt = np.load(gt_dir / rec["file"], allow_pickle=False).astype(np.float64)
        lr = np.load(lr_dir / rec["file"], allow_pickle=False)
        pred = infer(model, lr, device).astype(np.float64)
        err = np.abs(pred - gt)
        for col, (img, title, cmap) in enumerate([
            (np.clip(lr, 0, 1), f"NoisyLR {rec['file']}", "gray"),
            (pred, "Restored", "gray"),
            (gt, "GT", "gray"),
            (err, f"|err| mean={rec['mean_abs_err']:.4f}", "inferno"),
        ]):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1 if col < 3 else err.max())
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig_path = ROOT / "results" / "eda" / "sem_error_localisation.png"
    fig.savefig(fig_path, dpi=130)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
