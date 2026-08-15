#!/usr/bin/env python3
"""CPU-feasible backtracking search over adaptive supervised local restorers.

Every trial fits on the training side of the committed split, saves a normal checkpoint,
writes all 400 validation predictions to disk, and reloads those files for pinned
PSNR/SSIM/LPIPS evaluation. The incumbent checkpoint is replaced only after that persisted
evaluation reports a strict PSNR improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate import score_dir  # noqa: E402
from src.dataset import train_val_names  # noqa: E402
from src.io_utils import save_array  # noqa: E402
from src.model import AdaptiveLinearSR, build_model, count_parameters  # noqa: E402
from src.utils import git_sha, save_checkpoint, update_checkpoint_metrics  # noqa: E402


BASELINE = {
    "experiment_id": "E000-ls5",
    "parent_experiment_id": "",
    "config_diff": "closed_form_linear_ls5; kernel=5; global filter",
    "command": (
        "python train.py --config configs/final.yaml --data_root "
        "/Users/shanmukhsai/Downloads --seed 42 --closed_form_linear "
        "--out weights/best.pt"
    ),
    "checkpoint_path": "weights/best.pt",
    "validation_psnr": 26.327676082383558,
    "validation_ssim": 0.6599890989995291,
    "validation_lpips": 0.39992,
    "runtime_s": 24.9,
    "reason": "incumbent parent",
}

LEDGER_COLUMNS = [
    "experiment_id",
    "parent_experiment_id",
    "config_diff",
    "command",
    "checkpoint_path",
    "validation_psnr",
    "validation_ssim",
    "validation_lpips",
    "runtime_s",
    "reason",
]


@dataclass(frozen=True)
class Trial:
    experiment_id: str
    kernel: int
    gate_window: int
    intensity_bins: int
    std_bins: int
    ridge: float = 1.0e-4

    def model_config(self) -> dict[str, Any]:
        return {
            "name": "AdaptiveLinearSR",
            "kernel_size": self.kernel,
            "gate_window": self.gate_window,
            "intensity_bins": self.intensity_bins,
            "std_bins": self.std_bins,
            "scale": 2,
            "in_ch": 1,
            "out_ch": 1,
        }

    def diff(self) -> str:
        return json.dumps(
            {
                "model": self.model_config(),
                "fit": {"ridge": self.ridge, "train_partition_only": True},
            },
            sort_keys=True,
        )


TRIALS = [
    Trial("E002-raw-i2", 5, 1, 2, 1),
    Trial("E003-raw-i4", 5, 1, 4, 1),
    Trial("E004-raw-i8", 5, 1, 8, 1),
    Trial("E005-mean3-i8", 5, 3, 8, 1),
    Trial("E006-mean3-i8-s2", 5, 3, 8, 2),
    Trial("E007-mean3-i8-s4", 5, 3, 8, 4),
    Trial("E008-mean3-i8-s8", 5, 3, 8, 8),
    Trial("E009-mean3-i16-s8", 5, 3, 16, 8),
    Trial("E010-mean3-i16-s16", 5, 3, 16, 16),
    Trial("E011-mean3-i32-s8", 5, 3, 32, 8),
    Trial("E012-k7-mean3-i16-s8", 7, 3, 16, 8),
]


def _gate_maps(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(arr, dtype=np.float32)
    if int(window) == 1:
        return a, np.zeros_like(a)
    mean = uniform_filter(a, int(window), mode="nearest")
    mean_sq = uniform_filter(a * a, int(window), mode="nearest")
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)
    return mean, std


def _features(arr: np.ndarray, kernel: int) -> np.ndarray:
    p = int(kernel) // 2
    padded = np.pad(np.asarray(arr, dtype=np.float64), p, mode="edge")
    windows = sliding_window_view(padded, (kernel, kernel)).reshape(-1, kernel * kernel)
    return np.concatenate([windows, np.ones((windows.shape[0], 1), dtype=np.float64)], axis=1)


def _thresholds(
    root: Path,
    names: list[str],
    trial: Trial,
    sample_stride: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    lr_dir = root / "train" / "NoisyLR"
    for name in names:
        mean, std = _gate_maps(np.load(lr_dir / name, allow_pickle=False), trial.gate_window)
        means.append(mean[::sample_stride, ::sample_stride].reshape(-1))
        stds.append(std[::sample_stride, ::sample_stride].reshape(-1))
    mean_sample = np.concatenate(means)
    std_sample = np.concatenate(stds)
    iq = np.arange(1, trial.intensity_bins, dtype=np.float64) / trial.intensity_bins
    intensity = np.quantile(mean_sample, iq).astype(np.float32)
    if trial.std_bins == 1:
        return intensity, np.empty((trial.intensity_bins, 0), dtype=np.float32)
    intensity_index = np.digitize(mean_sample, intensity)
    sq = np.arange(1, trial.std_bins, dtype=np.float64) / trial.std_bins
    std_thresholds = np.stack(
        [np.quantile(std_sample[intensity_index == i], sq) for i in range(trial.intensity_bins)]
    ).astype(np.float32)
    return intensity, std_thresholds


def _categories(
    arr: np.ndarray,
    trial: Trial,
    intensity_thresholds: np.ndarray,
    std_thresholds: np.ndarray,
) -> np.ndarray:
    mean, std = _gate_maps(arr, trial.gate_window)
    intensity = np.digitize(mean.reshape(-1), intensity_thresholds)
    if trial.std_bins == 1:
        texture = np.zeros_like(intensity)
    else:
        texture = np.empty_like(intensity)
        flat_std = std.reshape(-1)
        for i in range(trial.intensity_bins):
            mask = intensity == i
            texture[mask] = np.digitize(flat_std[mask], std_thresholds[i])
    return intensity * trial.std_bins + texture


def fit_trial(
    root: Path,
    train_names: list[str],
    trial: Trial,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intensity_thresholds, std_thresholds = _thresholds(root, train_names, trial)
    n_filters = trial.intensity_bins * trial.std_bins
    n_features = trial.kernel * trial.kernel + 1
    xtx = np.zeros((n_filters, n_features, n_features), dtype=np.float64)
    xty = np.zeros((n_filters, n_features, 4), dtype=np.float64)
    lr_dir = root / "train" / "NoisyLR"
    gt_dir = root / "train" / "GT"
    for index, name in enumerate(train_names):
        lr = np.load(lr_dir / name, allow_pickle=False)
        gt = np.load(gt_dir / name, allow_pickle=False)
        x = _features(lr, trial.kernel)
        y = np.stack(
            [gt[dy::2, dx::2].reshape(-1) for dy in range(2) for dx in range(2)],
            axis=1,
        ).astype(np.float64)
        category = _categories(lr, trial, intensity_thresholds, std_thresholds)
        for c in np.unique(category):
            mask = category == c
            selected = x[mask]
            xtx[c] += selected.T @ selected
            xty[c] += selected.T @ y[mask]
        if (index + 1) % 700 == 0:
            print(f"  {trial.experiment_id}: fit {index + 1}/{len(train_names)}", flush=True)
    filters = np.empty((n_filters, n_features, 4), dtype=np.float64)
    for c in range(n_filters):
        regularizer = np.eye(n_features, dtype=np.float64) * trial.ridge
        regularizer[-1, -1] = 0.0
        filters[c] = np.linalg.solve(xtx[c] + regularizer, xty[c])
    return filters.astype(np.float32), intensity_thresholds, std_thresholds


def build_fitted_model(
    trial: Trial,
    filters: np.ndarray,
    intensity_thresholds: np.ndarray,
    std_thresholds: np.ndarray,
) -> AdaptiveLinearSR:
    model = build_model(trial.model_config())
    if not isinstance(model, AdaptiveLinearSR):
        raise TypeError(f"expected AdaptiveLinearSR, got {type(model).__name__}")
    model.set_fitted_state(
        torch.from_numpy(filters),
        torch.from_numpy(intensity_thresholds),
        torch.from_numpy(std_thresholds),
    )
    return model.eval()


def write_predictions(
    model: torch.nn.Module,
    root: Path,
    names: list[str],
    out_dir: Path,
    *,
    tta: bool = False,
    batch_size: int = 4,
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    lr_dir = root / "train" / "NoisyLR"
    started = time.perf_counter()
    with torch.inference_mode():
        for first in range(0, len(names), batch_size):
            batch_names = names[first:first + batch_size]
            arrays = [np.load(lr_dir / name, allow_pickle=False).astype(np.float32) for name in batch_names]
            x = torch.from_numpy(np.stack(arrays)[:, None])
            if tta:
                total: torch.Tensor | None = None
                for k in range(4):
                    for flip in (False, True):
                        u = torch.rot90(x, k, (-2, -1))
                        if flip:
                            u = torch.flip(u, (-1,))
                        pred = model(u).float()
                        if flip:
                            pred = torch.flip(pred, (-1,))
                        pred = torch.rot90(pred, -k, (-2, -1))
                        total = pred if total is None else total + pred
                assert total is not None
                y = total / 8.0
            else:
                y = model(x).float()
            for name, pred in zip(batch_names, y[:, 0].cpu().numpy()):
                save_array(out_dir / name, pred)
    return time.perf_counter() - started


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_root", default="runs/backtrack_search")
    parser.add_argument("--target", type=float, default=29.0)
    parser.add_argument("--max_trials", type=int, default=len(TRIALS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.device != "cpu":
        raise SystemExit("the adaptive closed-form search is CPU-only")

    root = Path(args.data_root)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    train_names, val_names = train_val_names(root)
    gt_dir = root / "train" / "GT"
    command_base = (
        f"python scripts/backtrack_search.py --data_root {root} --out_root "
        f"{Path(args.out_root)} --target {args.target} --max_trials {args.max_trials} --device cpu"
    )

    rows: list[dict[str, Any]] = [dict(BASELINE)]
    rows.append({
        "experiment_id": "E001-nafsr-20k-budget",
        "parent_experiment_id": "E000-ls5",
        "config_diff": "configs/final.yaml; requested 20000 gradient iterations",
        "command": (
            "python train.py --config configs/final.yaml --data_root "
            f"{root} --seed 42 --iters 20000 --device cpu"
        ),
        "checkpoint_path": "",
        "validation_psnr": "",
        "validation_ssim": "",
        "validation_lpips": "",
        "runtime_s": 51.10,
        "reason": (
            "backtrack before full run: measured configured batch-32 step took 39 s, "
            "projecting about 216 hours for 20k; CUDA unavailable and MPS unavailable"
        ),
    })
    best_id = str(BASELINE["experiment_id"])
    best_psnr = float(BASELINE["validation_psnr"])
    best_metrics = {
        "psnr": float(BASELINE["validation_psnr"]),
        "ssim": float(BASELINE["validation_ssim"]),
        "lpips": float(BASELINE["validation_lpips"]),
    }
    best_config: dict[str, Any] = {
        "model": {"name": "NAFSR"},
        "inference": {"tta": False},
    }
    best_checkpoint = REPO_ROOT / "weights" / "best.pt"
    best_predictions: Path | None = None
    misses = 0
    completed = 0

    for trial in TRIALS[:max(0, args.max_trials)]:
        if best_psnr >= args.target or misses >= 10:
            break
        parent = best_id
        trial_started = time.perf_counter()
        filters, intensity_thresholds, std_thresholds = fit_trial(root, train_names, trial)
        model = build_fitted_model(trial, filters, intensity_thresholds, std_thresholds)
        trial_dir = out_root / trial.experiment_id
        checkpoint = trial_dir / "checkpoint.pt"
        config = {
            "model": trial.model_config(),
            "fit": {
                "method": "train-split binned local least squares",
                "ridge": trial.ridge,
                "seed": 42,
                "train_n": len(train_names),
                "split": "configs/split_val.txt",
            },
            "inference": {"tta": False},
        }
        save_checkpoint(
            checkpoint,
            model=model,
            ema=None,
            config=config,
            iteration=0,
            metrics={"status": "awaiting saved-output evaluation"},
            git=git_sha(),
        )
        pred_dir = trial_dir / "predictions"
        inference_s = write_predictions(model, root, val_names, pred_dir)
        (pred_dir / "pred_meta.json").write_text(
            json.dumps({
                "label": trial.experiment_id,
                "kind": "learned",
                "checkpoint": str(checkpoint),
                "sec_per_image": inference_s / len(val_names),
                "split": "configs/split_val.txt (400 pairs)",
            }, indent=2),
            encoding="utf-8",
        )
        scored = score_dir(
            trial.experiment_id,
            pred_dir,
            gt_dir,
            val_names,
            with_lpips=True,
            device="cpu",
            split_desc="configs/split_val.txt (400 pairs)",
            persist=True,
            verbose=False,
        )
        metrics = {name: float(scored["metrics"][name]["mean"]) for name in ("psnr", "ssim", "lpips")}
        metrics.update({
            "psnr_std": float(scored["metrics"]["psnr"]["std"]),
            "ssim_std": float(scored["metrics"]["ssim"]["std"]),
            "lpips_std": float(scored["metrics"]["lpips"]["std"]),
            "val_n": len(val_names),
            "scored_from_disk": True,
            "training_mode": "adaptive_local_least_squares",
        })
        update_checkpoint_metrics(checkpoint, metrics)
        runtime = time.perf_counter() - trial_started
        improved = metrics["psnr"] > best_psnr
        if improved:
            delta = metrics["psnr"] - best_psnr
            best_id = trial.experiment_id
            best_psnr = metrics["psnr"]
            best_metrics = {k: metrics[k] for k in ("psnr", "ssim", "lpips")}
            best_config = config
            best_predictions = pred_dir
            shutil.copy2(checkpoint, best_checkpoint)
            reason = f"improved parent by {delta:+.6f} dB; promote and expand"
            misses = 0
        else:
            reason = f"no PSNR improvement; backtrack to {best_id} at {best_psnr:.6f} dB"
            misses += 1
        rows.append({
            "experiment_id": trial.experiment_id,
            "parent_experiment_id": parent,
            "config_diff": trial.diff(),
            "command": command_base + f"  # node {trial.experiment_id}",
            "checkpoint_path": str(checkpoint.relative_to(REPO_ROOT)),
            "validation_psnr": f"{metrics['psnr']:.8f}",
            "validation_ssim": f"{metrics['ssim']:.8f}",
            "validation_lpips": f"{metrics['lpips']:.8f}",
            "runtime_s": f"{runtime:.2f}",
            "reason": reason,
        })
        write_ledger(out_root / "experiments.csv", rows)
        completed += 1
        print(
            f"[{trial.experiment_id}] PSNR {metrics['psnr']:.6f} SSIM {metrics['ssim']:.6f} "
            f"LPIPS {metrics['lpips']:.6f} -> {reason}",
            flush=True,
        )

    # Phase F: score the incumbent with the repository's 8-way dihedral self-ensemble.
    if best_psnr < args.target and best_predictions is not None and misses < 10:
        parent = best_id
        started = time.perf_counter()
        checkpoint_data = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        model = build_model(checkpoint_data["config"])
        model.load_state_dict(checkpoint_data.get("ema") or checkpoint_data["model"], strict=True)
        model.eval()
        pred_dir = out_root / "F001-tta" / "predictions"
        inference_s = write_predictions(model, root, val_names, pred_dir, tta=True, batch_size=2)
        (pred_dir / "pred_meta.json").write_text(
            json.dumps({
                "label": "F001-tta",
                "kind": "learned",
                "checkpoint": "weights/best.pt",
                "sec_per_image": inference_s / len(val_names),
                "split": "configs/split_val.txt (400 pairs)",
            }, indent=2),
            encoding="utf-8",
        )
        scored = score_dir(
            "F001-tta", pred_dir, gt_dir, val_names, with_lpips=True, device="cpu",
            split_desc="configs/split_val.txt (400 pairs)", persist=True, verbose=False,
        )
        metrics = {name: float(scored["metrics"][name]["mean"]) for name in ("psnr", "ssim", "lpips")}
        runtime = time.perf_counter() - started
        if metrics["psnr"] > best_psnr:
            delta = metrics["psnr"] - best_psnr
            best_id = "F001-tta"
            best_psnr = metrics["psnr"]
            best_metrics = metrics
            best_config["inference"] = {"tta": True}
            best_predictions = pred_dir
            reason = f"improved parent by {delta:+.6f} dB; keep TTA"
            misses = 0
            checkpoint_metrics = dict(checkpoint_data.get("metrics", {}))
            checkpoint_metrics.update({
                "selected_val_psnr": metrics["psnr"],
                "selected_val_ssim": metrics["ssim"],
                "selected_val_lpips": metrics["lpips"],
                "selected_inference_tta": True,
                "scored_from_disk": True,
            })
            update_checkpoint_metrics(best_checkpoint, checkpoint_metrics)
        else:
            reason = f"no PSNR improvement; backtrack to {best_id} at {best_psnr:.6f} dB"
            misses += 1
        rows.append({
            "experiment_id": "F001-tta",
            "parent_experiment_id": parent,
            "config_diff": json.dumps({"inference": {"tta": True}}, sort_keys=True),
            "command": "python inference.py --tta --require_weights ...",
            "checkpoint_path": "weights/best.pt",
            "validation_psnr": f"{metrics['psnr']:.8f}",
            "validation_ssim": f"{metrics['ssim']:.8f}",
            "validation_lpips": f"{metrics['lpips']:.8f}",
            "runtime_s": f"{runtime:.2f}",
            "reason": reason,
        })
        print(
            f"[F001-tta] PSNR {metrics['psnr']:.6f} SSIM {metrics['ssim']:.6f} "
            f"LPIPS {metrics['lpips']:.6f} -> {reason}",
            flush=True,
        )

    write_ledger(REPO_ROOT / "results" / "backtrack_experiments.csv", rows)
    best_config["selection"] = {
        "experiment_id": best_id,
        "validation_psnr": best_metrics["psnr"],
        "validation_ssim": best_metrics["ssim"],
        "validation_lpips": best_metrics["lpips"],
        "target_psnr": args.target,
        "target_reached": bool(best_psnr >= args.target),
    }
    (REPO_ROOT / "configs" / "backtrack_best.yaml").write_text(
        yaml.safe_dump(best_config, sort_keys=False), encoding="utf-8"
    )
    summary = [
        "# Backtracking search summary",
        "",
        f"- Best experiment: `{best_id}`",
        f"- Best validation PSNR: **{best_metrics['psnr']:.6f} dB**",
        f"- Best validation SSIM: **{best_metrics['ssim']:.6f}**",
        f"- Best validation LPIPS: **{best_metrics['lpips']:.6f}**",
        f"- Target: {args.target:.1f} dB",
        f"- Target reached: **{'yes' if best_psnr >= args.target else 'no'}**",
        f"- Completed adaptive trials: {completed}",
        "- Validation protocol: 400 committed validation names; predictions saved and reloaded from disk.",
        "- Data isolation: filters fitted on 2,800 training names only; final input-only data was not opened.",
        "",
        "## Budget decision",
        "",
        "The configured NAFSR batch-32 step took 39 seconds on the available CPU. With CUDA and",
        "MPS both unavailable, 20,000 iterations project to about 216 hours before periodic",
        "validation. Neural length, loss, and larger-architecture branches therefore exhausted the",
        "local compute budget before a full trial. The CPU-feasible adaptive search was completed",
        "instead; exact parentage, commands, metrics, runtimes, and backtracks are in",
        "`results/backtrack_experiments.csv`.",
        "",
    ]
    (REPO_ROOT / "results" / "backtrack_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    if best_predictions is not None:
        (out_root / "best_predictions.txt").write_text(str(best_predictions), encoding="utf-8")
    print(json.dumps({
        "best_experiment": best_id,
        "best_metrics": best_metrics,
        "target_reached": best_psnr >= args.target,
        "best_checkpoint": str(best_checkpoint),
        "best_config": str(REPO_ROOT / "configs" / "backtrack_best.yaml"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
