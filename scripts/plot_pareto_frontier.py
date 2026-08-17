#!/usr/bin/env python3
"""Render the Pareto sweep frontier (docs/decisions.md D55) as an actual figure, not just a
markdown table -- SPEC section 19 backlog item #2 explicitly asks to "present the curve" for
the scaling sweep (plan PRIORITY 1, P1.3).

Reads results/eda/sweep_results.json, produced by scripts/fetch_sweep_metrics.py from the
sweep checkpoints' own embedded config+metrics (never hand-typed numbers) -- so this figure is
fully re-derivable from a committed script + a committed data file, not from prose.

Usage:
    python scripts/plot_pareto_frontier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_PATH = REPO_ROOT / "results" / "eda" / "sweep_results.json"
OUT_PATH = REPO_ROOT / "results" / "eda" / "pareto_frontier.png"


def main() -> int:
    if not IN_PATH.exists():
        print(f"{IN_PATH} does not exist -- run scripts/fetch_sweep_metrics.py first",
              file=sys.stderr)
        return 1
    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    rows = sorted(data["results"], key=lambda r: r["params"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params_m = [r["params"] / 1e6 for r in rows]
    psnr = [r["val_psnr"] for r in rows]
    ssim = [r["val_ssim"] for r in rows]
    lpips = [r["val_lpips"] for r in rows]
    labels = [f"{r['name']}\n{r['width']}x{r['num_blocks']}" for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    chosen = "sweep_e"  # docs/decisions.md D55: user's explicit choice for the long run

    for ax, ys, title, higher_better in (
        (axes[0], psnr, "PSNR (dB)", True),
        (axes[1], ssim, "SSIM", True),
        (axes[2], lpips, "LPIPS", False),
    ):
        ax.plot(params_m, ys, "o-", color="#1f77b4", zorder=2)
        for x, y, r in zip(params_m, ys, rows):
            marker_color = "#d62728" if r["name"] == chosen else "#1f77b4"
            ax.scatter([x], [y], color=marker_color, zorder=3, s=60)
            ax.annotate(r["name"].replace("sweep_", ""), (x, y),
                       textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
        ax.set_xlabel("Parameters (M)")
        ax.set_title(f"{title} {'(higher better)' if higher_better else '(lower better)'}")
        ax.grid(alpha=0.3)

    fig.suptitle("Pareto sweep: quality vs. parameter count\n"
                 "(6 FiLM-enabled NAFSR configs, HF Jobs A100-large; red = config chosen "
                 "for the long run, docs/decisions.md D55)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
