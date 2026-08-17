"""H0.2 -- is there a train/test scale gap for the shipped checkpoint?

NAFSR trains on 64px LR patches (-> 128px GT crops, configs/long_run_e.yaml lr_patch=64) but
val/test inference runs on full 128px LR images (-> 256px), and the synthetic fixture exercises
256px LR (-> 512px). Super-resolution CNNs sometimes underperform when the inference input size
differs from the training patch size, because zero-padding boundary effects and any global
receptive-field statistics the network implicitly learned no longer match. This script measures
that directly rather than assuming it away.

Method, per image:
  (a) "full"  -- run the WHOLE LR image through the model (the real inference.py path), then
                 crop the CENTRE region of the output that corresponds to the training patch
                 size (64 LR px -> 128 pred px, centred).
  (b) "crop"  -- crop the CENTRE 64x64 LR region FIRST, run the model on just that crop (the
                 exact size the model was trained on), producing a 128x128 prediction directly.
  Score both centre-region predictions against the identically-cropped GT centre, then compare
  (a) vs (b) with a PAIRED test (src.metrics.paired_compare). A significant, consistent gap
  between the two conditions -- same pixels, same GT, only the surrounding context differs --
  is direct evidence of a train/test scale mismatch. No such gap means the network generalises
  across input size cleanly (plausible for a fully-convolutional, zero-init-FiLM, no-attention
  architecture with no global pooling in the trunk).

Repeats the same experiment on the 256->512 synthetic fixture (crop 128 LR px, matching nothing
about the LR patch training size but exercising the SAME question at the other released
resolution) for extra evidence, if `results/eda/synthetic_512_fixture/` exists locally --
otherwise that half is skipped and reported as skipped, not fabricated.

No training, no fitting (F17 untouched) -- pure evaluation of the already-shipped checkpoint.

Writes results/eda/scale_gap_probe.json.
"""

from __future__ import annotations

import argparse
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
from src.metrics import paired_compare, score_pair  # noqa: E402
from src.model import build_model  # noqa: E402

CKPT_PATH = ROOT / "weights" / "best.pt"
TRAIN_LR_PATCH = 64  # configs/long_run_e.yaml data.lr_patch -- the training input size


def _load_model(device: str) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", cfg)
    model = build_model(model_cfg)
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    scale = int(model_cfg.get("scale", 2))
    return model, scale


def _infer(model: torch.nn.Module, lr: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        y = model(x)
    return np.clip(y[0, 0].float().cpu().numpy(), 0.0, 1.0)


def _centre_slice(size: int, patch: int) -> slice:
    start = (size - patch) // 2
    return slice(start, start + patch)


def compare_full_vs_crop(model: torch.nn.Module, scale: int, lr: np.ndarray, gt: np.ndarray,
                          lr_patch: int, device: str, with_lpips: bool) -> dict[str, Any] | None:
    """One image: score the centre region under 'full-image inference' vs 'patch-size crop
    inference'. Returns per-image score dicts under both conditions, or None if the image is
    smaller than the patch (can't crop)."""
    h, w = lr.shape
    if h < lr_patch or w < lr_patch:
        return None
    r_lr, c_lr = _centre_slice(h, lr_patch), _centre_slice(w, lr_patch)
    pred_patch = lr_patch * scale

    # (a) full-image inference, then crop the centre of the OUTPUT
    pred_full = _infer(model, lr, device)
    r_pred = slice(r_lr.start * scale, r_lr.start * scale + pred_patch)
    c_pred = slice(c_lr.start * scale, c_lr.start * scale + pred_patch)
    centre_from_full = pred_full[r_pred, c_pred]

    # (b) crop the LR centre FIRST (exact training patch size), then infer
    lr_crop = lr[r_lr, c_lr]
    centre_from_crop = _infer(model, lr_crop, device)

    gt_centre = gt[r_pred, c_pred]
    if centre_from_full.shape != gt_centre.shape or centre_from_crop.shape != gt_centre.shape:
        return None

    return {
        "full": score_pair(centre_from_full, gt_centre, with_lpips=with_lpips, device=device),
        "crop": score_pair(centre_from_crop, gt_centre, with_lpips=with_lpips, device=device),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=400, help="0 = full split")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t_start = time.time()

    model, scale = _load_model(device)

    data_root = resolve_data_root(args.data_root)
    _, names = train_val_names(data_root)
    if args.n and args.n < len(names):
        names = names[: args.n]
    lr_dir, gt_dir = data_root / "train" / "NoisyLR", data_root / "train" / "GT"

    full_scores, crop_scores = [], []
    skipped = 0
    for i, name in enumerate(names):
        lr = np.load(lr_dir / name, allow_pickle=False)
        gt = np.load(gt_dir / name, allow_pickle=False)
        res = compare_full_vs_crop(model, scale, lr, gt, TRAIN_LR_PATCH, device, with_lpips=True)
        if res is None:
            skipped += 1
            continue
        full_scores.append({"file": name, **res["full"]})
        crop_scores.append({"file": name, **res["crop"]})
        if args.verbose and (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)}")

    paired = paired_compare(full_scores, crop_scores)  # "full" is the candidate, "crop" the ref

    def _mean(scores: list[dict[str, Any]], key: str) -> float:
        vals = [s[key] for s in scores if key in s]
        return float(np.mean(vals)) if vals else float("nan")

    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "checkpoint": str(CKPT_PATH),
        "train_lr_patch": TRAIN_LR_PATCH,
        "n_scored": len(full_scores),
        "n_skipped_too_small": skipped,
        "means": {
            "full_image_inference": {m: _mean(full_scores, m) for m in ("psnr", "ssim", "lpips")},
            "patch_crop_inference": {m: _mean(crop_scores, m) for m in ("psnr", "ssim", "lpips")},
        },
        "paired_full_vs_crop": paired,
    }

    # A real, consistent, significant gap = "full" LOSES to "crop" on multiple metrics.
    full_loses = [m for m in ("psnr", "ssim", "lpips") if paired.get(m, {}).get("loss")]
    if len(full_loses) >= 2:
        verdict = "scale_gap_present"
    elif full_loses:
        verdict = "scale_gap_weak"
    else:
        verdict = "no_scale_gap"
    report["verdict"] = verdict
    report["verdict_rule"] = ("scale_gap_present = full-image inference paired-loses to "
                               "patch-size-crop inference on >=2/3 metrics; scale_gap_weak = "
                               "on exactly 1; no_scale_gap = on 0")
    report["wall_clock_s"] = round(time.time() - t_start, 2)

    out_path = ROOT / "results" / "eda" / "scale_gap_probe.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"VERDICT: {verdict} (n={len(full_scores)}, skipped={skipped})")
    for m, v in paired.items():
        print(f"  {m}: mean_diff(full-crop)={v['mean_diff']:+.5f} t={v['t']:+.2f} "
              f"win={v['win']} loss={v['loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
