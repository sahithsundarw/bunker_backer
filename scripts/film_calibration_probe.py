#!/usr/bin/env python3
"""FiLM calibration probe (plan PRIORITY 1, P1.1): does the NoiseEstimator's embedding
actually track the TRUE sampled noise level, or is it just present without doing anything
useful?

For N synthetic degraded pairs with a KNOWN sampled (sigma, a, v) noise triple
(src/degrade.py::sample_noise_params), this measures the Pearson correlation between the
FiLM conditioning embedding's L2 norm (a scalar summary of a 16-dim vector -- the raw vector
has no natural scalar ordering, but its norm is a reasonable "how much is this embedding
saying" summary) and a true noise-level scalar computed the same way the noise model itself
defines variance: ``sqrt(sigma^2 + a*x + v*x^2)`` at a representative x=1.0
(src/degrade.py::noise_variance's own formula, not a re-derivation).

No training here -- pure forward passes through an already-trained FiLM-enabled checkpoint
(the Pareto-sweep's config e, docs/decisions.md D55, which has film_dim=16/uncertainty=True).

Usage:
    python scripts/film_calibration_probe.py --checkpoint <path to a film_dim>0 checkpoint>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_PATH = REPO_ROOT / "results" / "eda" / "film_calibration.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=300, help="number of synthetic samples")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--data_root", default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch

    from src.dataset import resolve_data_root, train_val_names
    from src.degrade import DegradeConfig, degrade, sample_noise_params, noise_variance
    from src.io_utils import load_array
    from src.model import build_model

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_cfg = ck["config"].get("model", ck["config"])
    if int(model_cfg.get("film_dim", 0)) <= 0:
        print(f"checkpoint config has film_dim={model_cfg.get('film_dim')} -- this probe "
              f"requires a FiLM-enabled checkpoint", file=sys.stderr)
        return 2
    net = build_model(model_cfg)
    net.load_state_dict(ck["model"], strict=True)
    net.eval()
    if net.noise_est is None:
        print("build_model produced a model with noise_est=None despite film_dim>0 -- "
              "contract violation", file=sys.stderr)
        return 2

    root = resolve_data_root(args.data_root)
    train_names, _ = train_val_names(root)
    rng = np.random.default_rng(args.seed)
    dcfg = DegradeConfig()

    embed_norms: list[float] = []
    embed_vecs: list[np.ndarray] = []
    true_noise: list[float] = []
    with torch.no_grad():
        for i in range(args.n):
            name = train_names[i % len(train_names)]
            gt = load_array(root / "train" / "GT" / name)
            p = sample_noise_params(rng, dcfg)
            lr = degrade(gt, rng, dcfg, params=p)
            # randomise_frac widens (a, v) by +/-120%, so a randomly-drawn triple can imply a
            # negative variance at this formula's face value -- clamped at 0 before sqrt, same
            # as any real "how much noise was added" question must be (a variance cannot be
            # negative; this is a property of the randomisation range, not of the physical
            # noise itself).
            var = max(float(noise_variance(np.array([1.0], dtype=np.float32), p)[0]), 0.0)
            true_sigma = float(np.sqrt(var))

            lr_t = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None]
            cond = net.noise_est(lr_t)
            vec = cond.squeeze(0).numpy()
            embed_norms.append(float(np.linalg.norm(vec)))
            embed_vecs.append(vec)
            true_noise.append(true_sigma)

    embed_arr = np.array(embed_norms)
    embed_mat = np.stack(embed_vecs)          # (n, film_dim)
    true_arr = np.array(true_noise)
    corr_norm = float(np.corrcoef(embed_arr, true_arr)[0, 1])

    # Fairer test than the raw L2 norm: noise information could be linearly decodable along
    # SOME direction of the embedding even if the norm (which mixes in content-dependent
    # variation from every other direction) shows no correlation. Per-dimension correlation,
    # plus a held-out linear probe (OLS on a train split, R^2 on a disjoint val split) checks
    # that directly rather than assuming the norm is the only fair summary.
    per_dim_corr = [float(np.corrcoef(embed_mat[:, d], true_arr)[0, 1])
                    for d in range(embed_mat.shape[1])]
    max_abs_per_dim_corr = float(np.max(np.abs(per_dim_corr)))

    n_train = int(0.7 * args.n)
    x_tr, x_val = embed_mat[:n_train], embed_mat[n_train:]
    y_tr, y_val = true_arr[:n_train], true_arr[n_train:]
    x_tr1 = np.concatenate([x_tr, np.ones((len(x_tr), 1))], axis=1)
    x_val1 = np.concatenate([x_val, np.ones((len(x_val), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(x_tr1, y_tr, rcond=None)
    pred_val = x_val1 @ coef
    ss_res = float(np.sum((y_val - pred_val) ** 2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    held_out_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    result = {
        "checkpoint": str(args.checkpoint),
        "n": args.n,
        "pearson_r_embed_norm_vs_true_noise_std": corr_norm,
        "max_abs_pearson_r_single_dim_vs_true_noise_std": max_abs_per_dim_corr,
        "held_out_linear_probe_r2": held_out_r2,
        "embed_norm_mean": float(embed_arr.mean()),
        "embed_norm_std": float(embed_arr.std()),
        "true_noise_std_mean": float(true_arr.mean()),
        "true_noise_std_range": [float(true_arr.min()), float(true_arr.max())],
    }
    print(json.dumps(result, indent=2))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
