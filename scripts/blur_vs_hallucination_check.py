"""Plan Phase 4: does the model fail by BLURRING or by HALLUCINATING structure that isn't
there, on its worst-case images?

Hallucination is the disqualifying failure mode in an inspection context (README's own stated
design principle, "no adversarial loss -- hallucinating a structure that is not there is the
worst possible failure"). This checks it with evidence rather than asserting it from the
architecture choice (no-GAN) alone.

Method, per image: compare the prediction's high-frequency energy (the band ABOVE the LR
input's Nyquist limit after 2x decimation -- content the LR input cannot possibly have
supplied) against GT's energy in that same band, AND the spatial cross-correlation between
the two band-passed images in that band.
  - Pred energy < GT energy in that band, with low-to-moderate cross-correlation: BLURRING
    (the safe failure -- the model is conservative, not inventing detail).
  - Pred energy >= GT energy in that band: only a hallucination concern if cross-correlation
    is also low (energy present but spatially uncorrelated with the true high-frequency
    content = invented pattern, not recovered detail).

No training, no fitting -- pure evaluation of the shipped checkpoint on already-existing
fixtures (D5's documented failure case + the worst real-SEM images).

Usage:
    py -3.12 scripts/blur_vs_hallucination_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model import build_model  # noqa: E402

CKPT_PATH = ROOT / "weights" / "best.pt"
SCALE = 2


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


def high_freq_band_mask(h: int, w: int, scale: int) -> np.ndarray:
    """Boolean mask over an (h, w) FFT-shifted spectrum selecting frequencies ABOVE the LR
    input's Nyquist limit -- content the LR input, at 1/scale the linear resolution, cannot
    have supplied. The LR Nyquist maps to radius (min(h,w)/2) / scale in the HR spectrum."""
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    lr_nyquist_radius = (min(h, w) / 2.0) / scale
    return r > lr_nyquist_radius


def band_energy_and_correlation(pred: np.ndarray, gt: np.ndarray, scale: int) -> dict[str, float]:
    h, w = gt.shape
    mask = high_freq_band_mask(h, w, scale)

    Fp = np.fft.fftshift(np.fft.fft2(pred))
    Fg = np.fft.fftshift(np.fft.fft2(gt))

    e_pred = float((np.abs(Fp[mask]) ** 2).sum())
    e_gt = float((np.abs(Fg[mask]) ** 2).sum())

    # Band-pass by zeroing everything outside the high-freq mask, then inverse FFT to get the
    # spatial high-frequency component of each image, and correlate those spatially.
    Fp_band = np.zeros_like(Fp)
    Fp_band[mask] = Fp[mask]
    Fg_band = np.zeros_like(Fg)
    Fg_band[mask] = Fg[mask]
    pred_band = np.real(np.fft.ifft2(np.fft.ifftshift(Fp_band)))
    gt_band = np.real(np.fft.ifft2(np.fft.ifftshift(Fg_band)))
    corr = float(np.corrcoef(pred_band.ravel(), gt_band.ravel())[0, 1])

    return {
        "high_freq_energy_pred": e_pred,
        "high_freq_energy_gt": e_gt,
        "energy_ratio_pred_over_gt": e_pred / max(e_gt, 1e-12),
        "high_freq_spatial_correlation": corr,
    }


def run_d5_case(model: torch.nn.Module, device: str) -> dict[str, Any]:
    """D5's documented failure case (000984.npy) is NOT in configs/split_val.txt -- reproduce
    its restoration via a live inference.py subprocess (the real production path), matching
    the precedent scripts/make_qualitative_examples.py already established for this exact
    file."""
    from src.dataset import resolve_data_root

    root = resolve_data_root(None)
    src = root / "train" / "NoisyLR" / "000984.npy"
    gt_path = root / "train" / "GT" / "000984.npy"
    if not src.exists():
        return {"error": f"{src} not found"}
    with tempfile.TemporaryDirectory() as tdin, tempfile.TemporaryDirectory() as tdout:
        tdin_p, tdout_p = Path(tdin), Path(tdout)
        import shutil
        shutil.copy(src, tdin_p / "000984.npy")
        subprocess.run([sys.executable, str(ROOT / "inference.py"),
                        "--input_dir", str(tdin_p), "--output_dir", str(tdout_p),
                        "--require_weights"], check=True, capture_output=True)
        pred = np.load(tdout_p / "000984.npy", allow_pickle=False).astype(np.float64)
    gt = np.load(gt_path, allow_pickle=False).astype(np.float64)
    stats = band_energy_and_correlation(pred, gt, SCALE)
    stats["file"] = "000984.npy (D5)"
    return stats


def run_worst_sem_cases(model: torch.nn.Module, device: str, n: int = 3) -> list[dict[str, Any]]:
    sem_dir = ROOT / "results" / "eda" / "real_sem_ood"
    loc_path = ROOT / "results" / "eda" / "sem_error_localisation.json"
    if not loc_path.exists():
        print("WARNING: run scripts/sem_error_localisation.py first for a principled "
              "worst-case selection; falling back to first N files", file=sys.stderr)
        files = sorted((sem_dir / "GT").glob("*.npy"))[:n]
    else:
        loc = json.loads(loc_path.read_text(encoding="utf-8"))
        worst = sorted(loc["per_image"], key=lambda r: -r["mean_abs_err"])[:n]
        files = [sem_dir / "GT" / r["file"] for r in worst]

    out = []
    for f in files:
        gt = np.load(f, allow_pickle=False).astype(np.float64)
        lr = np.load(sem_dir / "NoisyLR" / f.name, allow_pickle=False)
        pred = infer(model, lr, device).astype(np.float64)
        stats = band_energy_and_correlation(pred, gt, SCALE)
        stats["file"] = f.name
        out.append(stats)
    return out


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    t0 = time.time()
    d5 = run_d5_case(model, device)
    sem_worst = run_worst_sem_cases(model, device, n=3)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "checkpoint": str(CKPT_PATH),
        "method": ("high-frequency band = FFT radius above the LR input's Nyquist limit "
                   "after 2x decimation -- content the LR input cannot have supplied. "
                   "energy_ratio < 1 means the model under-shoots (blurs); a ratio >= 1 is "
                   "only a hallucination concern if high_freq_spatial_correlation is also "
                   "low (energy present but spatially uncorrelated with true content)."),
        "d5_case": d5,
        "worst_real_sem_cases": sem_worst,
        "wall_clock_s": round(time.time() - t0, 2),
    }
    out_path = ROOT / "results" / "eda" / "blur_vs_hallucination_check.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"D5 case: energy_ratio={d5.get('energy_ratio_pred_over_gt', float('nan')):.4f} "
          f"corr={d5.get('high_freq_spatial_correlation', float('nan')):.4f}")
    for r in sem_worst:
        print(f"{r['file']}: energy_ratio={r['energy_ratio_pred_over_gt']:.4f} "
              f"corr={r['high_freq_spatial_correlation']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
