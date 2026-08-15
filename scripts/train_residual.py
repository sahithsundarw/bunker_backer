#!/usr/bin/env python3
"""Phase 2 (codex/residual-ls5-refinement): train a residual body on top of frozen LS-5.

The shipped ``weights/best.pt`` embeds a closed-form ridge-regularised 5x5 linear filter
into the ``stem`` and ``head.expand``/``head.project`` weights of a NAFSR carrier, with the
``body``/``body_tail`` zeroed to an exact identity (see ``train.py::_embed_linear_residual``).
Because ``head.expand`` is linear, the NAFSR forward is exactly

    output = LS5_output + project(shuffle(expand(body_tail(body(feat)) - feat_contribution)))

i.e. the body pathway is an additive residual on top of the closed-form LS-5 output. This
script builds a *fresh* NAFSR (independently sized: narrower body depth so it trains in
reasonable wall-clock time on this machine's MPS backend), copies the frozen LS-5
stem/head weights from an existing checkpoint into it, freezes exactly those tensors, and
trains only body/body_tail (with a small layerscale/init so training starts close to the
LS-5 baseline rather than perturbing it) on the training split only.

This never touches ``weights/best.pt``: it writes a new, independently-scored checkpoint.
Promoting it to ``weights/best.pt`` is a decision made after Phase 3/4 evaluation, not here.

Usage:
    python scripts/train_residual.py --data_root /Users/shanmukhsai/Downloads \
        --base_checkpoint weights/best.pt --out results/residual_experiments/r1/model.pt \
        --num_blocks 4 --iters 1500 --device mps --tag phase2-r1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import build_datasets  # noqa: E402
from src.losses import build_loss  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.utils import (  # noqa: E402
    EMA,
    append_experiment,
    cosine_warmup_lr,
    format_hms,
    git_sha,
    load_config,
    resolve_device,
    save_checkpoint,
    seed_everything,
)

sys.path.insert(0, str(_ROOT))
import train as _train  # noqa: E402  (reuse validate(), never re-implement metric scoring)

LEDGER = _ROOT / "results" / "experiments.csv"

#: Tensors that carry the fitted LS-5 filter (SPEC 9). Shapes depend only on width/scale/
#: in_ch/out_ch -- never on num_blocks -- so they transplant cleanly into a shallower body.
FROZEN_KEYS = (
    "stem.weight", "stem.bias",
    "head.expand.weight", "head.expand.bias",
    "head.project.weight", "head.project.bias",
)


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_ROOT / "configs" / "final.yaml"),
                    help="base config; only its data/loss sections are reused")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--base_checkpoint", default=str(_ROOT / "weights" / "best.pt"),
                    help="frozen LS-5 checkpoint whose stem/head weights are transplanted")
    ap.add_argument("--out", required=True, help="output checkpoint path for this experiment")
    ap.add_argument("--width", type=int, default=48,
                    help="must equal the base checkpoint's width -- stem/head shapes must match")
    ap.add_argument("--num_blocks", type=int, default=4)
    ap.add_argument("--layerscale_init", type=float, default=0.02,
                    help="small on purpose: body starts near-inert so training begins close "
                         "to the frozen LS-5 output instead of perturbing it (blocks.py notes "
                         "zero-init stalls gradient into the branch until beta/gamma move)")
    ap.add_argument("--body_tail_init_scale", type=float, default=0.02,
                    help="extra shrink on body_tail's default init for the same reason")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--val_limit", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--min_lr", type=float, default=1.0e-6)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_ledger", action="store_true")
    ap.add_argument("--tag", default="phase2-residual-ls5")
    ap.add_argument("--verbose", action="store_true")
    return ap


def _repo_relative(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def build_residual_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Fresh shallow NAFSR with the frozen LS-5 stem/head transplanted in."""
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    base_sd = base_ckpt.get("ema") or base_ckpt["model"]
    base_model_cfg = base_ckpt["config"].get("model", base_ckpt["config"])
    if int(base_model_cfg.get("width", 48)) != int(args.width):
        raise ValueError(
            f"--width {args.width} must match the base checkpoint's width "
            f"{base_model_cfg.get('width')} -- stem/head tensors must be shape-compatible"
        )
    if int(base_model_cfg.get("scale", 2)) != 2:
        raise ValueError("base checkpoint scale must be 2")

    model_cfg = {
        "name": "NAFSR",
        "width": int(args.width),
        "num_blocks": int(args.num_blocks),
        "scale": 2,
        "in_ch": 1,
        "out_ch": 1,
        "dw_expand": int(base_model_cfg.get("dw_expand", 2)),
        "ffn_expand": int(base_model_cfg.get("ffn_expand", 2)),
        "layerscale_init": float(args.layerscale_init),
        "padding_mode": str(base_model_cfg.get("padding_mode", "replicate")),
    }
    model = build_model({"model": model_cfg})

    params = dict(model.named_parameters())
    missing = [k for k in FROZEN_KEYS if k not in params]
    if missing:
        raise KeyError(f"expected frozen tensors not found in fresh model: {missing}")
    with torch.no_grad():
        for k in FROZEN_KEYS:
            params[k].copy_(base_sd[k])
            params[k].requires_grad_(False)
        # Shrink the trainable body_tail's default init so the untrained residual starts
        # near zero (SPEC 9's "final = LS5 + residual" behaviour, not a random perturbation).
        model.body_tail.weight.mul_(float(args.body_tail_init_scale))
        model.body_tail.bias.zero_()

    frozen_n = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "model_cfg": model_cfg,
        "base_checkpoint": _repo_relative(Path(args.base_checkpoint)),
        "base_val_psnr": base_ckpt.get("metrics", {}).get("val_psnr"),
        "frozen_params": frozen_n,
        "trainable_params": trainable_n,
    }
    return model, info


def sanity_check_init(model: torch.nn.Module, val_ds, device: torch.device) -> dict[str, Any]:
    """Confirm the untrained residual model starts close to (not equal to) the LS-5 output."""
    model.eval()
    v = _train.validate(model, val_ds, device, amp="fp32", channels_last=False,
                        limit=40, with_lpips=False)
    model.train()
    return v


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    seed = seed_everything(args.seed)
    device = resolve_device(args.device)
    _log(f"device={device}")

    base_cfg = load_config(args.config)
    root = _train.resolve_data_root(args.data_root)

    model, build_info = build_residual_model(args)
    model = model.to(device)
    _log(json.dumps(build_info, indent=2))

    dcfg = _train._data_config(base_cfg, seed, preload=None)
    train_ds, val_ds = build_datasets(root, dcfg)
    _log(f"train_n={len(train_ds)} val_n={len(val_ds)}")

    init_probe = sanity_check_init(model, val_ds, device)
    _log(f"[init sanity] untrained residual model: psnr={init_probe['psnr']:.4f} "
         f"ssim={init_probe['ssim']:.5f} (n={init_probe['n']}) -- should be close to but not "
         f"exactly the LS-5 baseline ({build_info.get('base_val_psnr')})")

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=int(args.workers), generator=gen, pin_memory=False,
                        persistent_workers=False)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.9), weight_decay=0.0)
    ema = EMA(model, decay=0.999)
    crit = build_loss(_train._section(base_cfg, "loss"))

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    sha = git_sha()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-residual-ls5-s{seed}"

    full_cfg = dict(base_cfg)
    full_cfg["model"] = build_info["model_cfg"]

    best = {"psnr": float("-inf"), "ssim": float("nan"), "iter": -1}
    t0 = time.perf_counter()
    it = 0
    epoch = 0
    model.train()
    shadow, _ = build_residual_model(args)
    shadow = shadow.to(device)

    while it < args.iters:
        train_ds.set_epoch(epoch)
        for batch in loader:
            if it >= args.iters:
                break
            lr_b = batch["lr"].to(device, non_blocking=True)
            gt_b = batch["gt"].to(device, non_blocking=True)

            cur_lr = cosine_warmup_lr(it, args.lr, args.min_lr, args.warmup_iters, args.iters)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            pred = model(lr_b)
            loss, logs = crit(pred.float(), gt_b, progress=it / max(1, args.iters))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            opt.step()
            ema.update(model)
            it += 1

            if args.verbose or it % 50 == 0 or it == 1:
                el = time.perf_counter() - t0
                ips = it / max(el, 1e-9)
                eta = (args.iters - it) / max(ips, 1e-9)
                _log(f"it {it}/{args.iters} loss {logs['total']:.5f} lr {cur_lr:.3e} "
                     f"{ips:.2f} it/s elapsed {format_hms(el)} eta {format_hms(eta)}")

            if args.val_every and (it % args.val_every == 0 or it == args.iters):
                ema.copy_to(shadow)
                v = _train.validate(shadow, val_ds, device, amp="fp32", channels_last=False,
                                    limit=args.val_limit)
                _log(f"  [val ema] it {it} psnr {v['psnr']:.4f} ssim {v['ssim']:.5f} (n={v['n']})")
                if v["psnr"] > best["psnr"]:
                    best = {"psnr": v["psnr"], "ssim": v["ssim"], "iter": it}
                    save_checkpoint(out_path, model=model, ema=ema, config=full_cfg,
                                    iteration=it,
                                    metrics={"val_psnr": v["psnr"], "val_ssim": v["ssim"],
                                             "val_n": v["n"], "selection": "ema/psnr",
                                             "split": "configs/split_val.txt",
                                             "training_mode": "residual_ls5_phase2",
                                             **build_info},
                                    git=sha)
                    _log(f"  [ckpt] new best {v['psnr']:.4f} dB -> {out_path}")
        epoch += 1

    wall = time.perf_counter() - t0
    if best["iter"] < 0:
        _log("no validation improvement was recorded; nothing was saved")
        return 2

    ck = torch.load(out_path, map_location="cpu", weights_only=True)
    final_model = build_model(ck["config"]).to(device)
    final_model.load_state_dict(ck["ema"] or ck["model"], strict=True)
    full = _train.validate(final_model, val_ds, device, amp="fp32", channels_last=False,
                           limit=None, with_lpips=False)
    _log(f"[final val, EMA, full committed split, in-memory scoring only] "
         f"psnr {full['psnr']:.4f} +/- {full['psnr_std']:.4f}  ssim {full['ssim']:.5f} "
         f"+/- {full['ssim_std']:.4f}  n={full['n']}")
    _log("NOTE: this in-memory number is provisional. The authoritative Phase 2 metric is "
         "produced by make_baselines.py + evaluate.py from files written to disk (V30).")

    if not args.no_ledger:
        append_experiment(LEDGER, {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": sha,
            "config": _repo_relative(Path(args.config)),
            "seed": seed,
            "model": "NAFSR",
            "params": build_info["frozen_params"] + build_info["trainable_params"],
            "iters": it,
            "batch_size": args.batch_size,
            "lr_patch": dcfg.lr_patch,
            "structural_kind": "",
            "best_iter": int(best["iter"]),
            "best_psnr": round(float(best["psnr"]), 4),
            "best_ssim": round(float(best["ssim"]), 5),
            "best_lpips": "",
            "val_n": int(args.val_limit),
            "weights_used": "ema",
            "wall_clock_s": round(wall, 1),
            "wall_clock_hms": format_hms(wall),
            "checkpoint": _repo_relative(out_path),
            "device": str(device),
            "notes": args.tag,
        })
        _log(f"[ledger] appended {run_id} to {LEDGER}")

    report = {
        "run_id": run_id, "checkpoint": str(out_path), "git": sha, "seed": seed,
        "best": best, "build_info": build_info, "wall_clock_s": round(wall, 1),
    }
    _log(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
