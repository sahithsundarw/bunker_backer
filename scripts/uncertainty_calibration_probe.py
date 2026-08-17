#!/usr/bin/env python3
"""Uncertainty calibration check (plan PRIORITY 1, P1.2): does the predicted log-variance
actually correlate with real squared error, or is it just present without being useful?

Runs an uncertainty-enabled checkpoint on the committed validation split
(configs/split_val.txt -- the same split its own val_psnr/val_ssim/val_lpips were measured
against during training) with ``return_uncertainty=True``, and correlates the predicted
``exp(log_var)`` against the actual per-pixel squared error ``(pred - gt)^2``, both at
per-image-mean granularity (n = number of val images) and pooled per-pixel (a random
subsample, since every pixel would be tens of millions of points).

Usage:
    python scripts/uncertainty_calibration_probe.py --checkpoint <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = REPO_ROOT / "results" / "eda" / "uncertainty_calibration.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--pixel_subsample_per_image", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from src.dataset import resolve_data_root, train_val_names
    from src.io_utils import load_array
    from src.model import build_model

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_cfg = ck["config"].get("model", ck["config"])
    if not bool(model_cfg.get("uncertainty", False)):
        print(f"checkpoint config has uncertainty={model_cfg.get('uncertainty')} -- this "
              f"probe requires an uncertainty-enabled checkpoint", file=sys.stderr)
        return 2
    net = build_model(model_cfg)
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    if not getattr(net, "has_uncertainty", False):
        print("build_model produced has_uncertainty=False despite uncertainty=True in "
              "config -- contract violation", file=sys.stderr)
        return 2

    root = resolve_data_root(args.data_root)
    _, val_names = train_val_names(root)
    rng = np.random.default_rng(args.seed)

    per_image_mean_expvar: list[float] = []
    per_image_mean_sqerr: list[float] = []
    pooled_expvar: list[float] = []
    pooled_sqerr: list[float] = []

    with torch.no_grad():
        for name in val_names:
            gt = load_array(root / "train" / "GT" / name)
            lr = load_array(root / "train" / "NoisyLR" / name)
            lr_t = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None]
            gt_t = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32))[None, None]
            pred, log_var = net(lr_t, return_uncertainty=True)
            sqerr = (pred - gt_t) ** 2
            expvar = torch.exp(log_var)

            per_image_mean_expvar.append(float(expvar.mean().item()))
            per_image_mean_sqerr.append(float(sqerr.mean().item()))

            flat_e = expvar.flatten().numpy()
            flat_s = sqerr.flatten().numpy()
            k = min(args.pixel_subsample_per_image, flat_e.size)
            idx = rng.choice(flat_e.size, size=k, replace=False)
            pooled_expvar.extend(flat_e[idx].tolist())
            pooled_sqerr.extend(flat_s[idx].tolist())

    per_image_corr = float(np.corrcoef(per_image_mean_expvar, per_image_mean_sqerr)[0, 1])
    pooled_corr = float(np.corrcoef(pooled_expvar, pooled_sqerr)[0, 1])
    # Spearman (rank) correlation is more appropriate for a calibration claim than Pearson --
    # a well-calibrated uncertainty need not be LINEARLY related to squared error, only
    # monotonically related (larger predicted variance -> larger typical error). Computed
    # directly via rank transform, no scipy dependency.
    def _spearman(a: list[float], b: list[float]) -> float:
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    per_image_spearman = _spearman(per_image_mean_expvar, per_image_mean_sqerr)
    pooled_spearman = _spearman(pooled_expvar, pooled_sqerr)

    result = {
        "checkpoint": str(args.checkpoint),
        "n_images": len(val_names),
        "n_pooled_pixels": len(pooled_expvar),
        "per_image_pearson_r": per_image_corr,
        "per_image_spearman_r": per_image_spearman,
        "pooled_pixel_pearson_r": pooled_corr,
        "pooled_pixel_spearman_r": pooled_spearman,
        "mean_exp_log_var": float(np.mean(per_image_mean_expvar)),
        "mean_sq_err": float(np.mean(per_image_mean_sqerr)),
    }
    print(json.dumps(result, indent=2))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
