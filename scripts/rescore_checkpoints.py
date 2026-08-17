"""Phase B3: free re-score of every checkpoint the Round 2 sweep/long-run/fine-tune pushed to
the HF Hub, under PSNR-only, SSIM-only, LPIPS-only and one disclosed blended criterion.

Why this exists: `train.py`'s in-loop checkpoint-save condition is hardcoded to
`val_psnr > best_psnr` (README's "What metric selects the 'best' checkpoint" section). KLA
scores an undisclosed PSNR+SSIM+LPIPS blend. Every "new best" checkpoint the long run pushed
along the way, plus the 6 Pareto-sweep finals, already exist on the Hub at zero additional
training cost -- this re-scores all of them under one harness to check whether a DIFFERENT
checkpoint the training run already produced would have been a better pick than the one
`save_best_on: psnr` actually chose.

No training, no fitting (F17 untouched) -- pure evaluation, one forward pass per checkpoint
per val image, all on the already-committed 400-pair val split.

Blended-criterion caveat, stated once and disclosed everywhere this script's output is used:
KLA's actual metric weights are undisclosed (SPEC), so no blend here can claim to reproduce
them. `--blend` picks a disclosed, defensible choice (equal-weighted z-score by default) and
prints it explicitly -- this is one reasonable candidate, not a guess at the real answer.

Usage:
    python scripts/rescore_checkpoints.py --repo_id Team-Ceciroleo67/kla-ps01-checkpoints \
        --pattern "20260816T211258Z-long_run_e-s42/*" --val_n 400
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import resolve_data_root, train_val_names  # noqa: E402
from src.metrics import score_pair  # noqa: E402
from src.model import build_model  # noqa: E402

DEFAULT_REPO = "Team-Ceciroleo67/kla-ps01-checkpoints"


def _load_model(ckpt_path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model = build_model(model_cfg)
    state = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    meta = {"iter": ckpt.get("iter") if isinstance(ckpt, dict) else None,
            "embedded_metrics": ckpt.get("metrics") if isinstance(ckpt, dict) else None,
            "config": model_cfg}
    return model, meta


def _infer(model: torch.nn.Module, lr: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        y = model(x)
    return np.clip(y[0, 0].float().cpu().numpy(), 0.0, 1.0)


def score_checkpoint(ckpt_path: Path, device: str, lr_dir: Path, gt_dir: Path,
                     names: list[str], with_lpips: bool) -> dict[str, Any]:
    model, meta = _load_model(ckpt_path, device)
    psnrs, ssims, lpipss = [], [], []
    for name in names:
        lr = np.load(lr_dir / name, allow_pickle=False)
        gt = np.load(gt_dir / name, allow_pickle=False)
        pred = _infer(model, lr, device)
        s = score_pair(pred, gt, with_lpips=with_lpips, device=device)
        psnrs.append(s["psnr"])
        ssims.append(s["ssim"])
        if with_lpips:
            lpipss.append(s["lpips"])
    out = {
        "n": len(names),
        "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
        "meta": meta,
    }
    if with_lpips:
        out["lpips_mean"] = float(np.mean(lpipss))
        out["lpips_std"] = float(np.std(lpipss))
    return out


def zscore_blend(rows: list[dict[str, Any]]) -> list[float]:
    """Equal-weighted z-score blend across PSNR (higher better), SSIM (higher better), LPIPS
    (lower better -- negated before z-scoring). Disclosed, not KLA's real undisclosed weights
    (see module docstring)."""
    def z(vals: list[float], higher_better: bool) -> np.ndarray:
        a = np.array(vals, dtype=np.float64)
        if not higher_better:
            a = -a
        mu, sd = a.mean(), a.std()
        return (a - mu) / sd if sd > 0 else np.zeros_like(a)

    psnr_z = z([r["psnr_mean"] for r in rows], True)
    ssim_z = z([r["ssim_mean"] for r in rows], True)
    lpips_z = z([r.get("lpips_mean", 0.0) for r in rows], False) if all(
        "lpips_mean" in r for r in rows) else np.zeros(len(rows))
    return list((psnr_z + ssim_z + lpips_z) / 3.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo_id", default=DEFAULT_REPO)
    ap.add_argument("--pattern", default="*", help="fnmatch glob against Hub repo file paths")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val_n", type=int, default=400)
    ap.add_argument("--with_lpips", action="store_true", default=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "eda" / "rescore_checkpoints.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    from huggingface_hub import hf_hub_download, list_repo_files

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    all_files = list_repo_files(args.repo_id, repo_type="model")
    matches = sorted(f for f in all_files if f.endswith(".pt") and fnmatch.fnmatch(f, args.pattern))
    if not matches:
        print(f"no files under {args.repo_id!r} match pattern {args.pattern!r}", file=sys.stderr)
        return 2

    data_root = resolve_data_root(args.data_root)
    _, names = train_val_names(data_root)
    if args.val_n and args.val_n < len(names):
        names = names[: args.val_n]
    lr_dir, gt_dir = data_root / "train" / "NoisyLR", data_root / "train" / "GT"

    t_start = time.time()
    rows: list[dict[str, Any]] = []
    for i, rel_path in enumerate(matches):
        if args.verbose:
            print(f"[{i + 1}/{len(matches)}] {rel_path} ...")
        local = Path(hf_hub_download(args.repo_id, rel_path, repo_type="model"))
        try:
            row = score_checkpoint(local, device, lr_dir, gt_dir, names,
                                   with_lpips=bool(args.with_lpips))
        except Exception as exc:  # noqa: BLE001 -- one bad checkpoint must not kill the sweep
            print(f"  SKIP {rel_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        row["hub_path"] = rel_path
        rows.append(row)
        if args.verbose:
            lp = f" lpips={row.get('lpips_mean', float('nan')):.5f}" if "lpips_mean" in row else ""
            print(f"  psnr={row['psnr_mean']:.4f} ssim={row['ssim_mean']:.5f}{lp}")

    if not rows:
        print("nothing scored -- all candidates failed to load", file=sys.stderr)
        return 2

    blend = zscore_blend(rows)
    for r, b in zip(rows, blend):
        r["blend_score"] = b

    def _best(key: str, higher_better: bool = True) -> dict[str, Any]:
        return max(rows, key=lambda r: r[key] if higher_better else -r[key])

    winners = {
        "psnr_only": _best("psnr_mean")["hub_path"],
        "ssim_only": _best("ssim_mean")["hub_path"],
        "lpips_only": (min(rows, key=lambda r: r.get("lpips_mean", float("inf")))["hub_path"]
                       if all("lpips_mean" in r for r in rows) else None),
        "blended_equal_zscore": _best("blend_score")["hub_path"],
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "repo_id": args.repo_id,
        "pattern": args.pattern,
        "n_val": len(names),
        "blend_method": ("EQUAL-WEIGHTED z-score across PSNR/SSIM/(negated)LPIPS -- a "
                         "disclosed, defensible choice, NOT KLA's real undisclosed weights"),
        "candidates": rows,
        "winners_by_criterion": winners,
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"scored {len(rows)}/{len(matches)} checkpoints")
    for k, v in winners.items():
        print(f"  winner by {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
