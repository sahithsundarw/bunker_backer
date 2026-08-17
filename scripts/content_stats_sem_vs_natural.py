"""Plan Phase 1(a): how far does real-SEM content sit outside the natural-photo training
distribution, on several independent content-statistics axes?

D67's fine-tune widened degradation-parameter randomisation and that neither fixed real-SEM
OOD nor helped elsewhere -- evidence the gap is not a degradation-coverage problem. This
script tests the remaining live hypothesis directly: is real-SEM content simply statistically
different from the natural photographs this project trains on?

Extends scripts/domain_shift_check.py's existing spectral-peakiness / gradient-anisotropy
comparison (which only ever compared natural photos against other natural photos --
train/NoisyLR vs test_NoisyLR) to add: intensity-histogram bimodality (Sarle's coefficient),
spectral slope (radially-averaged log-log power-spectrum fit), edge density (fraction of
pixels above a fixed absolute gradient threshold -- meaningful here because BOTH sets are
independently confirmed per-image min-max normalised to the same [0,1] convention, U1;
scripts/gen_real_sem_ood.py:79-81), local contrast (windowed std), and intensity-histogram
entropy (a disclosed simplification of full local-window texture entropy, given the time
budget -- global per-image histogram entropy, not a GLCM/patch-based measure).

Compares natural-photo GT (train/GT, sampled) against real-SEM GT
(results/eda/real_sem_ood/GT, all 45) -- GT, not NoisyLR, since the question is about
CONTENT, and GT is the cleanest signal for that (matches domain_shift_check.py's own
like-with-like principle, adapted: both sides here are GT, not degraded).

Usage:
    py -3.12 scripts/content_stats_sem_vs_natural.py --data_root C:\\kla-data --n_natural 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EDGE_THRESHOLD = 0.10  # absolute gradient magnitude; both sets share the [0,1] U1 convention


def spectral_peakiness(a: np.ndarray) -> float:
    """Verbatim from scripts/domain_shift_check.py -- ratio of the strongest off-DC spectral
    peak to the median off-DC magnitude. High => periodic structure."""
    A = np.abs(np.fft.fftshift(np.fft.fft2(a - a.mean())))
    h, w = A.shape
    cy, cx = h // 2, w // 2
    A[cy - 2:cy + 3, cx - 2:cx + 3] = 0.0
    v = A[A > 0]
    return float(v.max() / np.median(v)) if v.size else 0.0


def gradient_anisotropy(a: np.ndarray) -> float:
    """Verbatim from scripts/domain_shift_check.py."""
    gy, gx = np.gradient(a)
    ey, ex = float((gy ** 2).mean()), float((gx ** 2).mean())
    return float(max(ey, ex) / max(min(ey, ex), 1e-12))


def bimodality_coefficient(a: np.ndarray) -> float:
    """Sarle's bimodality coefficient (Pearson-kurtosis form): (skew^2 + 1) / kurtosis.
    > ~0.555 (a uniform distribution's value) is commonly read as evidence of bimodality;
    higher = more bimodal, lower = more unimodal/peaked."""
    x = a.ravel().astype(np.float64)
    mu, sd = x.mean(), x.std()
    if sd < 1e-12:
        return 0.0
    z = (x - mu) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())  # Pearson kurtosis (not excess); normal ~= 3
    return float((skew ** 2 + 1.0) / max(kurt, 1e-6))


def spectral_slope(a: np.ndarray) -> float:
    """Radially-averaged log-log power-spectrum slope. Natural scenes are close to 1/f^2 in
    power (slope ~ -2); sharper, more structured/periodic content deviates from this."""
    h, w = a.shape
    F = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    P = np.abs(F) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
    rmax = min(cy, cx)
    radial = np.bincount(r.ravel(), weights=P.ravel(), minlength=rmax + 1)[1:rmax]
    counts = np.bincount(r.ravel(), minlength=rmax + 1)[1:rmax]
    radial_mean = radial / np.maximum(counts, 1)
    freqs = np.arange(1, len(radial_mean) + 1, dtype=np.float64)
    mask = radial_mean > 0
    if mask.sum() < 4:
        return 0.0
    slope, _ = np.polyfit(np.log(freqs[mask]), np.log(radial_mean[mask]), 1)
    return float(slope)


def edge_density(a: np.ndarray, threshold: float = EDGE_THRESHOLD) -> float:
    gy, gx = np.gradient(a)
    mag = np.sqrt(gy ** 2 + gx ** 2)
    return float((mag > threshold).mean())


def local_contrast(a: np.ndarray, win: int = 7) -> float:
    """Mean local standard deviation over non-overlapping win x win windows."""
    h, w = a.shape
    h2, w2 = (h // win) * win, (w // win) * win
    blocks = a[:h2, :w2].reshape(h2 // win, win, w2 // win, win)
    stds = blocks.std(axis=(1, 3))
    return float(stds.mean())


def intensity_entropy(a: np.ndarray, bins: int = 256) -> float:
    """Shannon entropy (bits) of the per-image intensity histogram -- a disclosed
    simplification of full local-window texture entropy (see module docstring)."""
    hist, _ = np.histogram(a, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


METRICS = {
    "spectral_peakiness": spectral_peakiness,
    "gradient_anisotropy": gradient_anisotropy,
    "bimodality_coefficient": bimodality_coefficient,
    "spectral_slope": spectral_slope,
    "edge_density": edge_density,
    "local_contrast": local_contrast,
    "intensity_entropy": intensity_entropy,
}


def describe(files: list[Path]) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {k: [] for k in METRICS}
    for f in files:
        a = np.load(f, allow_pickle=False).astype(np.float64)
        for name, fn in METRICS.items():
            out[name].append(fn(a))
    return {k: np.array(v) for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--n_natural", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args(argv)

    from src.dataset import resolve_data_root

    root = resolve_data_root(args.data_root)
    natural_dir = root / "train" / "GT"
    sem_dir = ROOT / "results" / "eda" / "real_sem_ood" / "GT"
    if not sem_dir.is_dir():
        print(f"ERROR: {sem_dir} not found -- run scripts/gen_real_sem_ood.py first",
              file=sys.stderr)
        return 2

    all_natural = sorted(natural_dir.glob("*.npy"))
    rng = np.random.default_rng(args.seed)
    natural_sample = [all_natural[i] for i in
                      sorted(rng.choice(len(all_natural), size=min(args.n_natural,
                                                                    len(all_natural)),
                                        replace=False))]
    sem_files = sorted(sem_dir.glob("*.npy"))

    print(f"natural GT: n={len(natural_sample)} (sampled from {len(all_natural)}), "
          f"real-SEM GT: n={len(sem_files)}")
    nat = describe(natural_sample)
    sem = describe(sem_files)

    report: dict[str, Any] = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "n_natural": len(natural_sample), "n_sem": len(sem_files),
        "edge_threshold": EDGE_THRESHOLD,
        "metrics": {},
    }
    print(f"\n{'metric':<24}{'natural mean':>14}{'natural std':>14}{'SEM mean':>14}"
          f"{'SEM std':>14}{'SEM z-score':>14}")
    for name in METRICS:
        n_mean, n_std = float(nat[name].mean()), float(nat[name].std())
        s_mean, s_std = float(sem[name].mean()), float(sem[name].std())
        z = (s_mean - n_mean) / n_std if n_std > 1e-12 else float("nan")
        print(f"{name:<24}{n_mean:>14.4f}{n_std:>14.4f}{s_mean:>14.4f}{s_std:>14.4f}{z:>14.2f}")
        report["metrics"][name] = {
            "natural_mean": n_mean, "natural_std": n_std,
            "sem_mean": s_mean, "sem_std": s_std,
            "sem_zscore_vs_natural": z,
        }

    out_path = ROOT / "results" / "eda" / "content_stats_sem_vs_natural.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # Plot: one panel per metric, natural histogram vs SEM value(s) overlaid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()
    for i, name in enumerate(METRICS):
        ax = axes[i]
        ax.hist(nat[name], bins=30, alpha=0.6, density=True, label="natural GT")
        ax.hist(sem[name], bins=30, alpha=0.6, density=True, label="real-SEM GT")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    fig.tight_layout()
    fig_path = ROOT / "results" / "eda" / "content_stats_sem_vs_natural.png"
    fig.savefig(fig_path, dpi=120)
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
