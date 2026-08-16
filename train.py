#!/usr/bin/env python3
"""Training entry point. Reproduces ``weights/best.pt`` (SPEC 9).

    py -3.12 train.py --config configs/final.yaml --data_root C:/kla-data
    py -3.12 train.py --config configs/final.yaml --seed 42 --smoke        # V34
    py -3.12 train.py --config configs/final.yaml --overfit 2              # V25 gate

Three run modes, one code path
------------------------------
``--overfit N``
    **The V25 hard gate.**  Trains on ``N`` fixed real pairs (default 2) taken from the
    *training* half of the committed split, then measures PSNR on those same pairs.  It must
    exceed 40 dB.  If it does not, alignment, normalisation or the loss is broken and no
    downstream number is trustworthy -- the correct response is to stop and diagnose, never
    to tune the threshold.  Exit code is non-zero on failure.

``--smoke``
    A handful of steps with no checkpoint and no ledger row, used by the verifier.  Prints
    one ``SMOKE step=...`` line per step plus a ``SMOKE_DIGEST``; two invocations with the
    same ``--seed`` must produce byte-identical lines (V34).

default
    The real run.  Cosine schedule with warmup, bf16 autocast, channels_last, EMA, periodic
    validation on the committed held-out split, best-on-PSNR checkpointing, and one row
    appended to ``results/experiments.csv`` (V45).

Things that are deliberate, not accidental
------------------------------------------
* The loss is computed on the **unclipped** network output (SPEC 8).  Clipping happens only
  where a prediction is turned into a metric or a file; clipping inside training would
  zero the gradient of every saturated pixel, and ~3% of real pixels exceed 1.0.
* Validation reads the committed ``configs/split_val.txt`` through ``src.dataset``; the
  split is never regenerated here (V29).  ``test_NoisyLR`` is never opened by this file --
  it has no ground truth and training on it is forbidden (SPEC F17).  A hard startup
  assertion (``_assert_never_touches_test_noisylr``) checks the real, already-constructed
  dataset objects for this before any training step runs -- belt-and-suspenders on top of
  V54, reusing ``src.dataset``'s own split logic rather than a second glob.
* Model selection uses held-out validation only.  No training image ever scores a
  checkpoint.
* The shipped weights are the **EMA** weights (SPEC 9); ``inference.py`` reads
  ``ckpt['ema']`` in preference to ``ckpt['model']``.
* ``--hub_repo <repo_id>`` is optional and flag-gated (docs/PLAN_CLOUD.md): when set, every
  checkpoint save is also pushed to that HF Hub model repo, because HF Jobs storage is
  ephemeral and anything not pushed off the box during the run is lost when the job ends.
  ``huggingface_hub`` is imported lazily, only inside ``src.utils.push_checkpoint_to_hub``,
  so a run without ``--hub_repo`` has zero Hub dependency and unchanged behaviour.
  ``inference.py`` never gains this import; the allowlist there is untouched.

Owner: trainer.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataset import (  # noqa: E402
    DataConfig,
    PairedRestorationDataset,
    build_datasets,
    resolve_data_root,
    train_val_names,
)
from src.losses import build_loss  # noqa: E402
from src.metrics import clip_prediction, lpips_score, psnr, ssim  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402
from src.utils import (  # noqa: E402
    EMA,
    append_experiment,
    configure_backends,
    cosine_warmup_lr,
    format_hms,
    git_sha,
    load_config,
    push_checkpoint_to_hub,
    resolve_device,
    save_checkpoint,
    seed_everything,
    update_checkpoint_metrics,
)

#: Where the run ledger lives (SPEC 9, V45).
LEDGER = _ROOT / "results" / "experiments.csv"

#: V25's threshold. Written once, here, so it can never be quietly relaxed at a call site.
OVERFIT_PSNR_TARGET_DB: float = 40.0


# ======================================================================================
# CLI
# ======================================================================================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the restoration model.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None,
                    help="dataset root; else $KLA_DATA_ROOT, else docs/DATA_LOCATION.md")
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides train.seed in the config; recorded in the checkpoint")
    ap.add_argument("--smoke", action="store_true",
                    help="few steps only; used by the verifier (V34). No checkpoint, no ledger")
    ap.add_argument("--smoke_iters", type=int, default=12)
    ap.add_argument("--overfit", type=int, default=0, metavar="N",
                    help="V25 gate: overfit N fixed training pairs and require PSNR > 40 dB")
    ap.add_argument("--overfit_iters", type=int, default=6000,
                    help="MEASURED on this dataset (RTX 4060, pairs 000004/000005): a "
                         "6000-iter budget reaches 43.33 dB, crossing 40 dB at ~3000. A "
                         "4000-iter budget stalls at 39.78 dB because the cosine schedule "
                         "decays proportionally to the budget, so do not shorten it and then "
                         "read the result as a failure of alignment")
    ap.add_argument("--resume", default=None, help="checkpoint to warm-start model+EMA from")
    ap.add_argument("--out", default=None,
                    help="checkpoint path (default weights/best.pt)")
    ap.add_argument("--hub_repo", default=None,
                    help="optional HF Hub model repo id (e.g. org/name); if set, every "
                         "checkpoint save is also pushed there (docs/PLAN_CLOUD.md). Requires "
                         "huggingface_hub installed and an HF_TOKEN in the environment (never "
                         "read from a file). Absent by default: zero Hub dependency, zero "
                         "behaviour change, when this flag is not passed.")
    ap.add_argument("--iters", type=int, default=None, help="override optim.total_iters")
    ap.add_argument("--val_every", type=int, default=None, help="override train.val_every")
    ap.add_argument("--val_limit", type=int, default=100,
                    help="held-out images used for in-run model selection; the final report "
                         "always uses the whole committed split")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. 0 is the default: measured 0.72 ms/item against "
                         "a ~220 ms train step, and 0 keeps the run bit-reproducible")
    ap.add_argument("--closed_form_linear", action="store_true",
                    help="CPU-feasible Phase-1 path: fit a train-split least-squares 5x5 "
                         "linear residual and embed it into the configured NAFSR checkpoint")
    ap.add_argument("--linear_kernel", type=int, default=5,
                    help="odd LR kernel size for --closed_form_linear")
    ap.add_argument("--linear_ridge", type=float, default=1.0e-4,
                    help="L2 ridge used by --closed_form_linear; intercept is unregularised")
    ap.add_argument("--closed_form_min_psnr", type=float, default=26.2722,
                    help="fail --closed_form_linear if full-split validation PSNR does not "
                         "beat this measured baseline")
    ap.add_argument("--device", default=None)
    ap.add_argument("--deterministic", dest="deterministic", action="store_true", default=None)
    ap.add_argument("--no_deterministic", dest="deterministic", action="store_false")
    ap.add_argument("--no_ledger", action="store_true", help="do not append to experiments.csv")
    ap.add_argument("--tag", default="", help="free-text note recorded in the ledger row")
    ap.add_argument("--verbose", action="store_true", help="per-step logging")
    return ap


def _repo_relative(p: Path) -> str:
    """Repo-relative POSIX path when possible, absolute otherwise. Never CWD-relative."""
    try:
        return str(p.resolve().relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ======================================================================================
# F17 GUARD -- belt-and-suspenders, on top of V54 and src.dataset's own hard-coded
# ``<root>/train/{GT,NoisyLR}`` paths.  This does not reimplement the split: it inspects the
# paths a *real, already-constructed* dataset holds, using ``src.dataset``'s own
# ``train_val_names``/``configs/split_val.txt`` machinery as the sole source of truth.
# ======================================================================================
def _assert_never_touches_test_noisylr(root: Path, *dataset_objs: Any, verbose: bool = False) -> None:
    """Hard-fail before training starts if anything is about to read ``test_NoisyLR``.

    Two independent checks, both on real objects rather than on a second parallel glob:

    1. The resolved data root itself must not have ``test_NoisyLR`` anywhere in its path --
       catches a caller pointing ``--data_root``/``$KLA_DATA_ROOT`` directly at the test
       directory instead of the directory that contains ``train/``.
    2. Every ``(lr_path, gt_path)`` pair the already-constructed ``PairedRestorationDataset``
       objects hold (``ds.paths``, populated by ``src.dataset`` from
       ``train_val_names()``/``configs/split_val.txt`` before any preloading happens) must not
       reference ``test_NoisyLR`` either. ``test_NoisyLR`` has no ground truth and training or
       validating on it is forbidden unconditionally, on any machine (SPEC F17).

    Raises ``RuntimeError`` -- not a warning -- on any violation.
    """
    root_s = str(Path(root).resolve()).lower()
    if "test_noisylr" in root_s:
        raise RuntimeError(
            f"F17 violation: the resolved data root {root!r} contains 'test_NoisyLR'. "
            "test_NoisyLR has no ground truth and must never be trained or validated on, "
            "on any machine. Point --data_root / $KLA_DATA_ROOT at the directory that "
            "*contains* train/, not at test_NoisyLR itself."
        )
    checked = 0
    for ds in dataset_objs:
        for pair in getattr(ds, "paths", None) or []:
            for p in pair:
                checked += 1
                if "test_noisylr" in str(p).lower():
                    raise RuntimeError(
                        f"F17 violation: the training data pipeline would open {p}, which "
                        "lives under test_NoisyLR. This is forbidden unconditionally "
                        "(SPEC F17) -- stopping before any training step runs."
                    )
    if verbose:
        _log(f"  [F17 guard] {checked} dataset path(s) checked, none under test_NoisyLR")


def _maybe_push_checkpoint_to_hub(args: argparse.Namespace, out_path: Path, run_id: str,
                                  iteration: int) -> None:
    """Push ``out_path`` to ``--hub_repo`` if the flag was passed; a no-op otherwise.

    HF Jobs storage is ephemeral -- anything not pushed to the Hub during the run is lost the
    moment the job ends (docs/PLAN_CLOUD.md Step 1). ``path_in_repo`` is keyed by ``run_id``
    and the current iteration so multiple sweep runs, and multiple saves within one run, never
    clobber each other's checkpoints on the Hub side.
    """
    repo_id = getattr(args, "hub_repo", None)
    if not repo_id:
        return
    path_in_repo = f"{run_id}/step_{int(iteration):08d}/{out_path.name}"
    url = push_checkpoint_to_hub(out_path, repo_id, path_in_repo=path_in_repo)
    if url:
        _log(f"  [hub] pushed {out_path} -> {repo_id}/{path_in_repo} ({url})")


# ======================================================================================
# CONFIG PLUMBING
# ======================================================================================
def _section(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    blk = cfg.get(name)
    return dict(blk) if isinstance(blk, Mapping) else {}


def _data_config(cfg: Mapping[str, Any], seed: int, *, preload: Any = None,
                 crops_per_image: int | None = None) -> DataConfig:
    """Build the ``DataConfig`` and force the run seed into it.

    ``DataConfig.from_mapping`` rejects unknown keys on purpose, so overrides are applied
    with ``dataclasses.replace`` after parsing rather than by mutating the YAML mapping.
    """
    dc = DataConfig.from_mapping(cfg)
    kw: dict[str, Any] = {"seed": int(seed)}
    if preload is not None:
        kw["preload"] = preload
    if crops_per_image is not None:
        kw["crops_per_image"] = int(crops_per_image)
    return dataclasses.replace(dc, **kw)


def _autocast(device: torch.device, amp: str):
    """bf16 autocast on CUDA; a no-op context anywhere else.

    fp16+GradScaler is deliberately not implemented: the target card supports bf16 (verified)
    and a GradScaler is a second source of run-to-run divergence for no benefit here.
    """
    if device.type == "cuda" and str(amp).lower() in ("bf16", "bfloat16", "true", "auto"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# ======================================================================================
# VALIDATION  -- held-out only, EMA weights, clipped exactly as inference.py saves
# ======================================================================================
@torch.no_grad()
def _predict(model: torch.nn.Module, lr: torch.Tensor, device: torch.device, amp: str,
             channels_last: bool) -> torch.Tensor:
    x = lr.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    with _autocast(device, amp):
        y = model(x)
    return y.float()


@torch.no_grad()
def validate(model: torch.nn.Module, ds: PairedRestorationDataset, device: torch.device, *,
             amp: str = "bf16", channels_last: bool = True, limit: int | None = None,
             batch_size: int = 8, with_lpips: bool = False) -> dict[str, Any]:
    """Score held-out pairs with the pinned SPEC 10 metrics on the CLIPPED prediction.

    The prediction is clipped (never renormalised -- docs/decisions.md D3) and cast to
    float32 before scoring, which is exactly what ``inference.py`` writes to disk, so these
    numbers are directly comparable to ``scripts/evaluate.py``'s reloaded-from-disk figures.
    """
    was_training = model.training
    model.eval()
    n = len(ds) if limit is None else min(int(limit), len(ds))
    ps: list[float] = []
    ss: list[float] = []
    lp: list[float] = []
    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        lr = torch.stack([ds[i]["lr"] for i in idx])
        gt = torch.stack([ds[i]["gt"] for i in idx])
        out = _predict(model, lr, device, amp, channels_last).cpu()
        for b in range(len(idx)):
            pred = clip_prediction(out[b, 0].numpy().astype(np.float32))
            ref = gt[b, 0].numpy().astype(np.float32)
            ps.append(float(psnr(pred, ref)))
            ss.append(float(ssim(pred, ref)))
            if with_lpips:
                lp.append(float(lpips_score(pred, ref, device=str(device))))
    if was_training:
        model.train()
    res: dict[str, Any] = {
        "n": int(n),
        "psnr": float(np.mean(ps)) if ps else float("nan"),
        "psnr_std": float(np.std(ps)) if ps else float("nan"),
        "ssim": float(np.mean(ss)) if ss else float("nan"),
        "ssim_std": float(np.std(ss)) if ss else float("nan"),
    }
    if with_lpips and lp:
        res["lpips"] = float(np.mean(lp))
        res["lpips_std"] = float(np.std(lp))
    return res


# ======================================================================================
# V25 -- THE OVERFIT GATE
# ======================================================================================
def run_overfit(cfg: dict[str, Any], args: argparse.Namespace, seed: int) -> int:
    """Train on ``args.overfit`` fixed real pairs and require PSNR > 40 dB on them.

    Deliberate choices, because this check is a diagnostic and not a benchmark:

    * The pairs come from the **training** half of the committed split, so the gate never
      touches validation data even though it reports on its own training pairs.
    * Whole images (128 LR -> 256 GT), the real measured LR, no augmentation, no synthetic
      degradation, no CutBlur.  The only thing under test is whether the model *can*
      reproduce a target it has seen -- i.e. whether the pair is aligned, the normalisation
      consistent and the loss pointed the right way.
    * Both the raw and the EMA weights are scored; the gate passes on the better of the two,
      because at a few thousand steps the EMA is still catching up and failing the gate on
      EMA lag alone would be a false alarm.
    """
    device = resolve_device(args.device)
    backend = configure_backends(deterministic=True)
    n_pairs = max(1, int(args.overfit))
    root = resolve_data_root(args.data_root)
    train_names, _ = train_val_names(root)
    names = train_names[:n_pairs]

    dcfg = _data_config(cfg, seed, preload=True)
    dcfg = dataclasses.replace(dcfg, synth_ratio=0.0, cutblur_prob=0.0, dihedral=False)
    ds = PairedRestorationDataset(root, dcfg, "val", names=names)   # "val" = whole image, no aug
    _assert_never_touches_test_noisylr(root, ds, verbose=bool(args.verbose))

    lr = torch.stack([ds[i]["lr"] for i in range(len(ds))]).to(device)
    gt = torch.stack([ds[i]["gt"] for i in range(len(ds))]).to(device)

    tcfg = _section(cfg, "train")
    ocfg = _section(cfg, "optim")
    channels_last = bool(tcfg.get("channels_last", True))
    amp = str(tcfg.get("amp", "bf16"))
    if channels_last:
        lr = lr.contiguous(memory_format=torch.channels_last)
        gt = gt.contiguous(memory_format=torch.channels_last)

    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.999)))
    crit = build_loss(_section(cfg, "loss"))
    opt = torch.optim.AdamW(model.parameters(), lr=float(ocfg.get("lr", 1e-3)),
                            betas=tuple(ocfg.get("betas", (0.9, 0.9))),
                            weight_decay=float(ocfg.get("weight_decay", 0.0)))

    total = int(args.overfit_iters)
    warm = min(50, max(1, total // 20))
    base_lr = float(ocfg.get("lr", 1e-3))
    min_lr = float(ocfg.get("min_lr", 1e-6))
    grad_clip = float(tcfg.get("grad_clip", 1.0))

    shadow = build_model(cfg).to(device)          # built once; EMA weights are copied in
    if channels_last:
        shadow = shadow.to(memory_format=torch.channels_last)

    history: list[dict[str, float]] = []
    best = {"psnr": float("-inf"), "iter": -1, "weights": "none"}
    t0 = time.perf_counter()
    model.train()
    for it in range(total):
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(it, base_lr, min_lr, warm, total)
        with _autocast(device, amp):
            pred = model(lr)
        loss, logs = crit(pred.float(), gt, progress=it / max(1, total))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        ema.update(model)

        last = (it == total - 1)
        if last or (it + 1) % max(1, total // 12) == 0:
            raw = _score_pairs(model, lr, gt, device, amp, channels_last)
            ema.copy_to(shadow)
            ema_psnr = _score_pairs(shadow, lr, gt, device, amp, channels_last)
            history.append({"iter": it + 1, "loss": float(logs["total"]),
                            "psnr_raw": raw, "psnr_ema": ema_psnr})
            if max(raw, ema_psnr) > best["psnr"]:
                best = {"psnr": max(raw, ema_psnr), "iter": it + 1,
                        "weights": "raw" if raw >= ema_psnr else "ema"}
            if args.verbose:
                _log(f"  overfit it={it+1:5d} loss={logs['total']:.6f} "
                     f"psnr_raw={raw:.3f} psnr_ema={ema_psnr:.3f}")

    dt = time.perf_counter() - t0
    passed = bool(best["psnr"] > OVERFIT_PSNR_TARGET_DB)
    report = {
        "check": "V25",
        "pass": passed,
        "target_db": OVERFIT_PSNR_TARGET_DB,
        "n_pairs": n_pairs,
        "pair_names": names,
        "split_of_pairs": "train",
        "iters": total,
        "best_psnr_db": round(float(best["psnr"]), 4),
        "best_at_iter": int(best["iter"]),
        "best_weights": best["weights"],
        "final_psnr_raw_db": round(float(history[-1]["psnr_raw"]), 4) if history else None,
        "final_psnr_ema_db": round(float(history[-1]["psnr_ema"]), 4) if history else None,
        "final_loss": round(float(history[-1]["loss"]), 8) if history else None,
        "structural_kind": crit_kind(crit, gt),
        "seed": int(seed),
        "device": str(device),
        "backend": backend,
        "wall_clock_s": round(dt, 2),
        "history": history,
    }
    if not passed:
        report["diagnosis"] = (
            "PSNR did not clear 40 dB on pairs the model was trained on. That is NOT a "
            "hyper-parameter problem: it means the LR/GT crop is misaligned, the "
            "normalisation differs between input and target, or the loss is not pointed at "
            "the target. Fix the cause before trusting any downstream metric."
        )
    _log(json.dumps(report, indent=2))
    return 0 if passed else 1


def crit_kind(crit: Any, target: torch.Tensor) -> str:
    """Which structural term the loss will actually use at this spatial size."""
    from src.losses import supports_ms_ssim
    h, w = int(target.shape[-2]), int(target.shape[-1])
    return "ms_ssim" if (crit.allow_ms_ssim and supports_ms_ssim(h, w)) else "ssim"


@torch.no_grad()
def _score_pairs(model: torch.nn.Module, lr: torch.Tensor, gt: torch.Tensor,
                 device: torch.device, amp: str, channels_last: bool) -> float:
    """Mean PSNR over an in-memory batch, scored on the clipped float32 prediction."""
    was_training = model.training
    model.eval()
    with _autocast(device, amp):
        out = model(lr)
    out = out.float().cpu()
    ref = gt.float().cpu()
    vals = [float(psnr(clip_prediction(out[b, 0].numpy().astype(np.float32)),
                       ref[b, 0].numpy().astype(np.float32)))
            for b in range(out.shape[0])]
    if was_training:
        model.train()
    return float(np.mean(vals))


# ======================================================================================
# THE TRAINING RUN
# ======================================================================================
def run_training(cfg: dict[str, Any], args: argparse.Namespace, seed: int) -> int:
    smoke = bool(args.smoke)
    device = resolve_device(args.device)
    deterministic = args.deterministic if args.deterministic is not None else smoke
    backend = configure_backends(deterministic=deterministic)

    tcfg = _section(cfg, "train")
    ocfg = _section(cfg, "optim")
    dblk = _section(cfg, "data")

    batch_size = int(dblk.get("batch_size", 32))
    total_iters = int(args.iters if args.iters is not None else ocfg.get("total_iters", 20000))
    warmup = int(ocfg.get("warmup_iters", 500))
    base_lr = float(ocfg.get("lr", 1e-3))
    min_lr = float(ocfg.get("min_lr", 1e-6))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    amp = str(tcfg.get("amp", "bf16"))
    channels_last = bool(tcfg.get("channels_last", True))
    val_every = int(args.val_every if args.val_every is not None
                    else tcfg.get("val_every", 1000))

    if smoke:
        # Small, memory-mapped and short: the verifier runs this twice and both runs must
        # finish quickly AND agree exactly.
        total_iters = int(args.smoke_iters)
        warmup = max(1, total_iters // 4)
        batch_size = min(batch_size, 4)
        val_every = 0

    root = resolve_data_root(args.data_root)
    # One epoch is stretched to at least ~1000 optimiser steps via crops_per_image so that a
    # 20k-iteration run re-enters the DataLoader ~20 times rather than ~230 times. With
    # workers=0 this costs nothing; with workers>0 it is the difference between amortising
    # and paying Windows process spawn on every 87 steps.
    n_train_names = len(train_val_names(root)[0])
    cpi = 1 if smoke else max(1, math.ceil(1000 * batch_size / max(1, n_train_names)))
    dcfg = _data_config(cfg, seed, preload=(False if smoke else None), crops_per_image=cpi)

    t_build = time.perf_counter()
    train_ds, val_ds = build_datasets(root, dcfg)
    build_s = time.perf_counter() - t_build
    # F17 hard gate, before any training step runs (belt-and-suspenders on top of V54):
    # reuses the real split/paths src.dataset already built, never a second glob.
    _assert_never_touches_test_noisylr(root, train_ds, val_ds, verbose=bool(args.verbose))

    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=int(args.workers), generator=gen,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )

    model = build_model(cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = count_parameters(model)
    # Round-2 differentiator (docs/decisions.md): only True when the config's model.uncertainty
    # is set, which is only true for configs that opt in -- every existing config, including
    # configs/final.yaml, leaves this False and the training loop below is unchanged for them.
    has_uncertainty = bool(getattr(model, "has_uncertainty", False))
    ema = EMA(model, decay=float(tcfg.get("ema_decay", 0.999)))
    shadow = build_model(cfg).to(device)          # built once; EMA weights are copied in
    if channels_last:
        shadow = shadow.to(memory_format=torch.channels_last)
    start_iter = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=True)
        if ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        start_iter = int(ck.get("iter", 0))
        _log(f"resumed {args.resume} at iter {start_iter}")

    crit = build_loss(_section(cfg, "loss"))
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            betas=tuple(ocfg.get("betas", (0.9, 0.9))),
                            weight_decay=float(ocfg.get("weight_decay", 0.0)))

    out_path = Path(args.out) if args.out else (_ROOT / "weights" / "best.pt")
    if not out_path.is_absolute():
        out_path = _ROOT / out_path

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-" \
             f"{Path(args.config).stem}-s{seed}"
    sha = git_sha()
    _log(json.dumps({
        "run_id": run_id, "git": sha, "config": args.config, "seed": seed,
        "model": str(_section(cfg, "model").get("name", "?")), "params": n_params,
        "iters": total_iters, "batch_size": batch_size, "lr_patch": dcfg.lr_patch,
        "crops_per_image": cpi, "n_train": len(train_ds), "n_val": len(val_ds),
        "preloaded": bool(train_ds.preloaded), "dataset_build_s": round(build_s, 2),
        "device": str(device), "backend": backend, "amp": amp,
        "channels_last": channels_last, "smoke": smoke, "out": str(out_path),
    }))

    best = {"psnr": float("-inf"), "ssim": float("nan"), "iter": -1}
    structural_kind = ""
    smoke_lines: list[str] = []
    loss_hist: list[float] = []
    t0 = time.perf_counter()
    it = start_iter
    epoch = 0
    model.train()

    while it < total_iters:
        train_ds.set_epoch(epoch)
        for batch in loader:
            if it >= total_iters:
                break
            lr_b = batch["lr"].to(device, non_blocking=True)
            gt_b = batch["gt"].to(device, non_blocking=True)
            if channels_last:
                lr_b = lr_b.contiguous(memory_format=torch.channels_last)
                gt_b = gt_b.contiguous(memory_format=torch.channels_last)

            cur_lr = cosine_warmup_lr(it, base_lr, min_lr, warmup, total_iters)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with _autocast(device, amp):
                if has_uncertainty:
                    pred, log_var = model(lr_b, return_uncertainty=True)
                    log_var = log_var.float()
                else:
                    pred = model(lr_b)
                    log_var = None
            # UNCLIPPED output into the loss (SPEC 8): clipping here would zero the gradient
            # of every saturated pixel, and ~3% of real pixels legitimately exceed 1.0.
            loss, logs = crit(pred.float(), gt_b, progress=it / max(1, total_iters),
                              log_var=log_var)
            if not math.isfinite(float(logs["total"])):
                _log(f"FATAL: non-finite loss at iter {it}: {logs}")
                return 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            ema.update(model)
            structural_kind = str(logs["structural_kind"])
            loss_hist.append(float(logs["total"]))
            it += 1

            if smoke:
                line = (f"SMOKE step={it:04d} lr={cur_lr:.12e} total={logs['total']:.12e} "
                        f"charbonnier={logs['charbonnier']:.12e} "
                        f"structural={logs['structural']:.12e} fft={logs['fft']:.12e} "
                        f"kind={structural_kind} gnorm={float(gnorm):.12e}")
                smoke_lines.append(line)
                _log(line)
            elif args.verbose or it % 100 == 0 or it == 1:
                el = time.perf_counter() - t0
                ips = (it - start_iter) / max(el, 1e-9)
                eta = (total_iters - it) / max(ips, 1e-9)
                _log(f"it {it}/{total_iters} loss {logs['total']:.5f} "
                     f"(charb {logs['charbonnier']:.5f} {structural_kind} "
                     f"{logs['structural']:.5f} fft {logs['fft']:.5f}) "
                     f"lr {cur_lr:.3e} {ips:.2f} it/s elapsed {format_hms(el)} "
                     f"eta {format_hms(eta)}")

            if val_every and (it % val_every == 0 or it == total_iters):
                ema.copy_to(shadow)
                v = validate(shadow, val_ds, device, amp=amp, channels_last=channels_last,
                             limit=args.val_limit)
                _log(f"  [val ema] it {it} psnr {v['psnr']:.4f} ssim {v['ssim']:.5f} "
                     f"(n={v['n']}, held-out)")
                if v["psnr"] > best["psnr"]:
                    best = {"psnr": v["psnr"], "ssim": v["ssim"], "iter": it}
                    save_checkpoint(out_path, model=model, ema=ema, config=cfg, iteration=it,
                                    metrics={"val_psnr": v["psnr"], "val_ssim": v["ssim"],
                                             "val_n": v["n"], "selection": "ema/psnr",
                                             "split": "configs/split_val.txt",
                                             "structural_kind": structural_kind},
                                    git=sha)
                    _log(f"  [ckpt] new best {v['psnr']:.4f} dB -> {out_path}")
                    _maybe_push_checkpoint_to_hub(args, out_path, run_id, it)
        epoch += 1

    wall = time.perf_counter() - t0

    if smoke:
        digest = hashlib.sha256("\n".join(smoke_lines).encode("utf-8")).hexdigest()
        _log("SMOKE_LOSSES " + json.dumps([f"{v:.12e}" for v in loss_hist]))
        _log("SMOKE_DIGEST " + digest)
        _log("SMOKE_OK steps=%d seed=%d deterministic=%s"
             % (len(loss_hist), seed, backend["cudnn_deterministic"]))
        return 0

    # ---- final report: full committed split, EMA weights, LPIPS included ---------------
    if best["iter"] < 0:
        # no validation happened (e.g. --val_every 0): still ship something reproducible
        save_checkpoint(out_path, model=model, ema=ema, config=cfg, iteration=it,
                        metrics={"note": "no validation performed"}, git=sha)
        _maybe_push_checkpoint_to_hub(args, out_path, run_id, it)
    ck = torch.load(out_path, map_location="cpu", weights_only=True)
    final_model = build_model(ck["config"]).to(device)
    if channels_last:
        final_model = final_model.to(memory_format=torch.channels_last)
    final_model.load_state_dict(ck["ema"] or ck["model"], strict=True)
    full = validate(final_model, val_ds, device, amp=amp, channels_last=channels_last,
                    limit=None, with_lpips=True)
    _log(f"[final val, EMA, full committed split] psnr {full['psnr']:.4f} +/- "
         f"{full['psnr_std']:.4f}  ssim {full['ssim']:.5f} +/- {full['ssim_std']:.5f}  "
         f"lpips {full.get('lpips', float('nan')):.5f}  n={full['n']}")

    metrics = {
        "val_psnr": full["psnr"], "val_psnr_std": full["psnr_std"],
        "val_ssim": full["ssim"], "val_ssim_std": full["ssim_std"],
        "val_lpips": full.get("lpips"), "val_lpips_std": full.get("lpips_std"),
        "val_n": full["n"], "selection": "ema/psnr",
        "selection_n": int(args.val_limit),
        "split": "configs/split_val.txt",
        "structural_kind": structural_kind,
        "best_iter": int(best["iter"]),
        "total_iters_run": int(it),
        "wall_clock_s": round(wall, 2),
    }
    # ONLY the metrics block is rewritten. The tensors stay the best-on-held-out ones that
    # were saved during the run; re-saving the live model here would ship the last weights.
    update_checkpoint_metrics(out_path, metrics)
    # Final push so the Hub copy carries the full-split metrics, not just the last
    # in-run "new best" snapshot -- keyed "final" so it never collides with a step-keyed one.
    if getattr(args, "hub_repo", None):
        path_in_repo = f"{run_id}/final/{out_path.name}"
        url = push_checkpoint_to_hub(out_path, args.hub_repo, path_in_repo=path_in_repo)
        if url:
            _log(f"  [hub] pushed final {out_path} -> {args.hub_repo}/{path_in_repo} ({url})")

    # --no_ledger is an escape hatch around SPEC 9's "log every run" and V45's row-count
    # gate, and it was unconditional -- any real training run could skip the ledger
    # entirely (requirements-auditor M-1). Restricted to --smoke: a genuine smoke test
    # legitimately should not pollute the ledger with a 12-iteration row, but nothing else
    # gets to opt out silently.
    if args.no_ledger and not smoke:
        print(f"train.py: --no_ledger ignored for a non-smoke run ({run_id}); "
              f"SPEC 9 requires every real run to be logged", file=sys.stderr)
    if not (args.no_ledger and smoke):
        append_experiment(LEDGER, {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": sha,
            "config": args.config.replace("\\", "/"),
            "seed": seed,
            "model": str(_section(cfg, "model").get("name", "?")),
            "params": n_params,
            "iters": it,
            "batch_size": batch_size,
            "lr_patch": dcfg.lr_patch,
            "structural_kind": structural_kind,
            "best_iter": int(best["iter"]),
            "best_psnr": round(float(full["psnr"]), 4),
            "best_ssim": round(float(full["ssim"]), 5),
            "best_lpips": (round(float(full["lpips"]), 5) if "lpips" in full else ""),
            "val_n": int(full["n"]),
            "weights_used": "ema",
            "wall_clock_s": round(wall, 1),
            "wall_clock_hms": format_hms(wall),
            "checkpoint": _repo_relative(out_path),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "notes": args.tag,
        })
        _log(f"[ledger] appended {run_id} to {LEDGER}")
    return 0


# ======================================================================================
# CPU-FEASIBLE PHASE 1: CLOSED-FORM LINEAR CHECKPOINT
# ======================================================================================
def _linear_features(lr: np.ndarray, kernel: int) -> np.ndarray:
    """Edge-padded local LR features plus an intercept column."""
    from numpy.lib.stride_tricks import sliding_window_view

    k = int(kernel)
    pad = k // 2
    padded = np.pad(np.asarray(lr, dtype=np.float64), pad, mode="edge")
    windows = sliding_window_view(padded, (k, k)).reshape(-1, k * k)
    ones = np.ones((windows.shape[0], 1), dtype=np.float64)
    return np.concatenate([windows, ones], axis=1)


def _fit_linear_filters(root: Path, train_names: Sequence[str], kernel: int, ridge: float,
                        verbose: bool = False) -> np.ndarray:
    """Least-squares LR-neighbourhood -> four HR subpixels, trained on train split only."""
    k = int(kernel)
    if k <= 0 or k % 2 == 0:
        raise ValueError(f"--linear_kernel must be a positive odd integer, got {kernel}")
    f = k * k + 1
    xtx = np.zeros((f, f), dtype=np.float64)
    xty = np.zeros((f, 4), dtype=np.float64)
    lr_dir = root / "train" / "NoisyLR"
    gt_dir = root / "train" / "GT"
    t0 = time.perf_counter()
    for i, name in enumerate(train_names):
        lr = np.load(lr_dir / name, allow_pickle=False)
        gt = np.load(gt_dir / name, allow_pickle=False)
        x = _linear_features(lr, k)
        y = np.stack(
            [gt[dy::2, dx::2].reshape(-1) for dy in range(2) for dx in range(2)],
            axis=1,
        ).astype(np.float64)
        xtx += x.T @ x
        xty += x.T @ y
        if verbose and (i + 1) % 250 == 0:
            _log(f"  [closed-form] fit {i + 1}/{len(train_names)} "
                 f"elapsed {format_hms(time.perf_counter() - t0)}")
    reg = np.eye(f, dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, xty)


def _bilinear_coefficients(kernel: int) -> np.ndarray:
    """2x align_corners=False bilinear skip as edge-padded local-filter coefficients."""
    k = int(kernel)
    pad = k // 2
    coeff = np.zeros((k * k + 1, 4), dtype=np.float64)
    row = {
        0: [(-1, 0.25), (0, 0.75)],
        1: [(0, 0.75), (1, 0.25)],
    }
    col = row
    for dy in range(2):
        for dx in range(2):
            s = dy * 2 + dx
            for oy, wy in row[dy]:
                for ox, wx in col[dx]:
                    coeff[(oy + pad) * k + (ox + pad), s] += wy * wx
    return coeff


def _linear_predict(lr: np.ndarray, filters: np.ndarray, kernel: int) -> np.ndarray:
    """Unclipped direct linear prediction, used only for an embedding self-check."""
    h, w = lr.shape
    y = _linear_features(lr, kernel) @ filters
    out = np.empty((2 * h, 2 * w), dtype=np.float32)
    out[0::2, 0::2] = y[:, 0].reshape(h, w)
    out[0::2, 1::2] = y[:, 1].reshape(h, w)
    out[1::2, 0::2] = y[:, 2].reshape(h, w)
    out[1::2, 1::2] = y[:, 3].reshape(h, w)
    return out


@torch.no_grad()
def _embed_linear_residual(model: torch.nn.Module, residual: np.ndarray, kernel: int) -> None:
    """Encode a 5x5 residual filter into the NAFSR stem + PixelShuffle head path.

    The configured NAFSR still supplies the global bilinear skip. The fitted direct filter
    is therefore converted to a residual filter before embedding. Residual blocks are set to
    exact identity by zeroing their layer scales.
    """
    mcfg = getattr(model, "width", None), getattr(model, "scale", None)
    if mcfg[0] is None or int(mcfg[0]) < 9 or int(mcfg[1]) != 2:
        raise ValueError("closed-form embedding requires NAFSR width >= 9 and scale == 2")
    k = int(kernel)
    if k != 5:
        raise ValueError("the current two-conv NAFSR carrier embeds exactly a 5x5 LR filter")
    if residual.shape != (k * k + 1, 4):
        raise ValueError(f"residual filter shape {residual.shape} != {(k * k + 1, 4)}")

    for prm in model.parameters():
        prm.zero_()

    # Channels 0..8 are shifted LR basis maps from the 3x3 stem.
    for ay in range(-1, 2):
        for ax in range(-1, 2):
            c = (ay + 1) * 3 + (ax + 1)
            model.stem.weight[c, 0, ay + 1, ax + 1] = 1.0

    # PixelShuffle channel order for scale 2: c*4 + dy*2 + dx.
    pad = k // 2
    for oy in range(-pad, pad + 1):
        for ox in range(-pad, pad + 1):
            idx = (oy + pad) * k + (ox + pad)
            ay = max(-1, min(1, oy))
            ax = max(-1, min(1, ox))
            by, bx = oy - ay, ox - ax
            c = (ay + 1) * 3 + (ax + 1)
            for s in range(4):
                model.head.expand.weight[s, c, by + 1, bx + 1] += float(residual[idx, s])
    for s in range(4):
        model.head.expand.bias[s] = float(residual[-1, s])
    model.head.project.weight[0, 0, 1, 1] = 1.0


def _closed_form_embed_error(model: torch.nn.Module, filters: np.ndarray, root: Path,
                             sample_name: str, kernel: int) -> float:
    lr = np.load(root / "train" / "NoisyLR" / sample_name, allow_pickle=False)
    direct = _linear_predict(lr, filters, kernel)
    x = torch.from_numpy(lr.astype(np.float32))[None, None]
    out = model.eval()(x)[0, 0].detach().cpu().numpy()
    return float(np.max(np.abs(out - direct)))


def run_closed_form_linear(cfg: dict[str, Any], args: argparse.Namespace, seed: int) -> int:
    """Fit and checkpoint a train-split least-squares residual model.

    This is deliberately explicit rather than pretending a CPU host completed the 20k-step
    gradient run. It produces a normal V35 checkpoint, but its metrics identify the training
    mode as closed-form LS-5.
    """
    root = resolve_data_root(args.data_root)
    train_names, val_names = train_val_names(root)
    out_path = Path(args.out) if args.out else (_ROOT / "weights" / "best.pt")
    if not out_path.is_absolute():
        out_path = _ROOT / out_path

    cfg.setdefault("model", {})
    cfg["model"]["padding_mode"] = str(cfg["model"].get("padding_mode", "replicate"))
    if cfg["model"]["padding_mode"] != "replicate":
        raise SystemExit("--closed_form_linear requires model.padding_mode: replicate")

    t0 = time.perf_counter()
    filters = _fit_linear_filters(root, train_names, args.linear_kernel, args.linear_ridge,
                                  verbose=args.verbose)
    residual = filters - _bilinear_coefficients(args.linear_kernel)
    fit_s = time.perf_counter() - t0

    device = resolve_device(args.device)
    model = build_model(cfg).to(device)
    _embed_linear_residual(model, residual, args.linear_kernel)
    n_params = count_parameters(model)
    ema = EMA(model, decay=float(_section(cfg, "train").get("ema_decay", 0.999)))

    embed_error = _closed_form_embed_error(model.cpu(), filters, root, val_names[0],
                                           args.linear_kernel)
    if device.type != "cpu":
        model = model.to(device)

    dcfg = _data_config(cfg, seed, preload=True)
    _, val_ds = build_datasets(root, dcfg)
    amp = str(_section(cfg, "train").get("amp", "bf16"))
    channels_last = bool(_section(cfg, "train").get("channels_last", True))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    full = validate(model, val_ds, device, amp=amp, channels_last=channels_last,
                    limit=None, with_lpips=False)
    wall = time.perf_counter() - t0

    passed_bar = bool(full["psnr"] > float(args.closed_form_min_psnr))
    metrics = {
        "training_mode": "closed_form_linear_ls5",
        "val_psnr": full["psnr"], "val_psnr_std": full["psnr_std"],
        "val_ssim": full["ssim"], "val_ssim_std": full["ssim_std"],
        "val_n": full["n"],
        "split": "configs/split_val.txt",
        "train_n": len(train_names),
        "linear_kernel": int(args.linear_kernel),
        "linear_ridge": float(args.linear_ridge),
        "baseline_to_beat_psnr": float(args.closed_form_min_psnr),
        "beats_baseline_to_beat": passed_bar,
        "embedding_max_abs_error": embed_error,
        "total_iters_run": 0,
        "wall_clock_s": round(wall, 2),
        "fit_wall_clock_s": round(fit_s, 2),
    }
    sha = git_sha()
    save_checkpoint(out_path, model=model, ema=ema, config=cfg, iteration=0,
                    metrics=metrics, git=sha)

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-" \
             f"{Path(args.config).stem}-closed-form-s{seed}"
    if not args.no_ledger:
        append_experiment(LEDGER, {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_sha": sha,
            "config": args.config.replace("\\", "/"),
            "seed": seed,
            "model": str(_section(cfg, "model").get("name", "?")),
            "params": n_params,
            "iters": 0,
            "batch_size": 0,
            "lr_patch": 0,
            "structural_kind": "n/a",
            "best_iter": 0,
            "best_psnr": round(float(full["psnr"]), 4),
            "best_ssim": round(float(full["ssim"]), 5),
            "best_lpips": "",
            "val_n": int(full["n"]),
            "weights_used": "ema",
            "wall_clock_s": round(wall, 1),
            "wall_clock_hms": format_hms(wall),
            "checkpoint": _repo_relative(out_path),
            "device": str(device),
            "notes": args.tag or "closed_form_linear_ls5",
        })
    report = {
        "run_id": run_id,
        "checkpoint": str(out_path),
        "git": sha,
        "seed": seed,
        "train_n": len(train_names),
        "val_n": len(val_names),
        "params": n_params,
        "metrics": metrics,
    }
    _log(json.dumps(report, indent=2))
    return 0 if passed_bar else 2


# ======================================================================================
def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else _section(cfg, "train").get("seed", 42))
    # The seed that was actually used is written back into the config that gets embedded in
    # the checkpoint, so a checkpoint always states the seed that produced it.
    cfg.setdefault("train", {})
    cfg["train"]["seed"] = seed
    seed_everything(seed)

    if args.overfit:
        return run_overfit(cfg, args, seed)
    if args.closed_form_linear:
        return run_closed_form_linear(cfg, args, seed)
    return run_training(cfg, args, seed)


if __name__ == "__main__":
    sys.exit(main())
