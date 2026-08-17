"""Plan Phase 2, item 1: weight-space interpolation between the shipped checkpoint and D67's
fine-tune, at zero training cost.

D67's fine-tune and the shipped checkpoint fail in OPPOSITE directions: the fine-tune wins big
in-distribution but breaks proxy-OOD and doesn't fix real-SEM OOD; the shipped checkpoint is
the reverse. Both already exist as trained weights. Linearly interpolating between them in
weight space (`theta = (1-alpha)*base + alpha*finetuned`) is a well-known cheap way to trade
between two fine-tuned solutions' basins without any further training -- this script sweeps
alpha and scores all three evaluation sets (in-distribution val, proxy-OOD, real-SEM OOD) at
each point.

No training, no fitting -- pure evaluation, weight averaging only. F17 untouched.

Decision rule, fixed before running (see module-level DECISION_RULE below): a ship-worthy
alpha must win real-SEM OOD on a PAIRED test against alpha=0 AND stay within ~0.15 dB PSNR of
alpha=0 in-distribution AND not lose proxy-OOD on a paired test.

Usage:
    py -3.12 scripts/weight_interpolate_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import resolve_data_root, train_val_names  # noqa: E402
from src.metrics import paired_compare, score_pair  # noqa: E402
from src.model import build_model  # noqa: E402

BASE_CKPT = ROOT / "weights" / "best.pt"
FINETUNE_HUB_PATH = "20260817T101639Z-finetune_ood_wide-s42/step_00104000/best_snap_it104000.pt"
FINETUNE_REPO = "Team-Ceciroleo67/kla-ps01-checkpoints"
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

DECISION_RULE = (
    "ship-worthy alpha must: (1) win real-SEM OOD on a paired test vs alpha=0, AND "
    "(2) stay within ~0.15 dB PSNR of alpha=0 in-distribution, AND "
    "(3) not lose proxy-OOD on a paired test vs alpha=0. Fixed before running."
)


def _load_state(path) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})
    state = ck.get("ema") or ck.get("model")
    return cfg.get("model", cfg), state


def interpolate(state_a: dict, state_b: dict, alpha: float) -> dict:
    out = {}
    for k in state_a:
        a, b = state_a[k], state_b[k]
        if a.dtype.is_floating_point:
            out[k] = (1.0 - alpha) * a.float() + alpha * b.float()
        else:
            out[k] = a if alpha < 0.5 else b  # non-float buffers (e.g. counters): pick a side
    return out


def infer(model: torch.nn.Module, lr: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        y = model(x)
    return np.clip(y[0, 0].float().cpu().numpy(), 0.0, 1.0)


def score_set(model, device, lr_dir, gt_dir, names) -> list[dict[str, Any]]:
    out = []
    for n in names:
        lr = np.load(lr_dir / n, allow_pickle=False)
        gt = np.load(gt_dir / n, allow_pickle=False)
        pred = infer(model, lr, device)
        out.append({"file": n, **score_pair(pred, gt, with_lpips=True, device=device)})
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg, state_a = _load_state(BASE_CKPT)
    ft_path = hf_hub_download(FINETUNE_REPO, FINETUNE_HUB_PATH, repo_type="model")
    _, state_b = _load_state(ft_path)
    if set(state_a) != set(state_b):
        print("ERROR: state dict key mismatch, cannot interpolate", file=sys.stderr)
        return 2

    root = resolve_data_root(None)
    _, names = train_val_names(root)
    lr_dir, gt_dir = root / "train" / "NoisyLR", root / "train" / "GT"

    from pathlib import Path as P
    sets = {
        "proxy_ood": (P("results/eda/proxy_ood/NoisyLR"), P("results/eda/proxy_ood/GT")),
        "real_sem_ood": (P("results/eda/real_sem_ood/NoisyLR"), P("results/eda/real_sem_ood/GT")),
    }
    set_names = {}
    for k, (ld, gd) in sets.items():
        set_names[k] = sorted(p.name for p in ld.glob("*.npy"))

    t0 = time.time()
    results: dict[float, dict[str, Any]] = {}
    alpha0_scores: dict[str, list[dict[str, Any]]] = {}
    for alpha in ALPHAS:
        print(f"=== alpha={alpha} ===")
        state = interpolate(state_a, state_b, alpha)
        model = build_model(model_cfg)
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()

        val_scores = score_set(model, device, lr_dir, gt_dir, names)
        entry: dict[str, Any] = {
            "val_psnr": float(np.mean([r["psnr"] for r in val_scores])),
            "val_ssim": float(np.mean([r["ssim"] for r in val_scores])),
            "val_lpips": float(np.mean([r["lpips"] for r in val_scores])),
        }
        if alpha == 0.0:
            alpha0_scores["val"] = val_scores
        else:
            entry["val_paired_vs_alpha0"] = paired_compare(val_scores, alpha0_scores["val"])

        for setname, (ld, gd) in sets.items():
            sc = score_set(model, device, ld, gd, set_names[setname])
            entry[f"{setname}_psnr"] = float(np.mean([r["psnr"] for r in sc]))
            entry[f"{setname}_ssim"] = float(np.mean([r["ssim"] for r in sc]))
            entry[f"{setname}_lpips"] = float(np.mean([r["lpips"] for r in sc]))
            if alpha == 0.0:
                alpha0_scores[setname] = sc
            else:
                entry[f"{setname}_paired_vs_alpha0"] = paired_compare(sc, alpha0_scores[setname])

        results[alpha] = entry
        print(f"  val psnr={entry['val_psnr']:.4f} ssim={entry['val_ssim']:.5f} "
              f"lpips={entry['val_lpips']:.5f}")
        print(f"  real_sem_ood psnr={entry['real_sem_ood_psnr']:.4f} "
              f"ssim={entry['real_sem_ood_ssim']:.5f} lpips={entry['real_sem_ood_lpips']:.5f}")
        print(f"  proxy_ood psnr={entry['proxy_ood_psnr']:.4f} "
              f"ssim={entry['proxy_ood_ssim']:.5f} lpips={entry['proxy_ood_lpips']:.5f}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "base_checkpoint": str(BASE_CKPT),
        "finetune_checkpoint": FINETUNE_HUB_PATH,
        "decision_rule": DECISION_RULE,
        "alphas": ALPHAS,
        "results": results,
        "wall_clock_s": round(time.time() - t0, 2),
    }
    out_path = ROOT / "results" / "eda" / "weight_interpolate_sweep.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, setname, title in zip(
        axes, ["val", "proxy_ood", "real_sem_ood"],
        ["In-distribution (val)", "Proxy-OOD", "Real-SEM OOD"],
    ):
        psnr = [results[a][f"{setname}_psnr"] if setname != "val" else results[a]["val_psnr"]
                for a in ALPHAS]
        ax.plot(ALPHAS, psnr, marker="o")
        ax.set_xlabel("alpha (0=shipped, 1=fine-tuned)")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = ROOT / "results" / "eda" / "weight_interpolate_sweep.png"
    fig.savefig(fig_path, dpi=120)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
