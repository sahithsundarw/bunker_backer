#!/usr/bin/env python3
"""Fetch the 6 Pareto-sweep checkpoints' embedded config+metrics from the private HF Hub
checkpoint repo and write results/eda/sweep_results.json -- a committed, re-derivable record
of the numbers docs/decisions.md D55 reports, so a plot (scripts/plot_pareto_frontier.py) can
be produced FROM data, not retyped from prose (CLAUDE.md: "no number enters a doc unless a
repo script produced it").

Each checkpoint dict has exactly the V35 keys (src/utils.py::save_checkpoint): model, ema,
config, iter, metrics, git -- config/metrics are read directly, model/ema weights are
discarded after computing the parameter count via build_model (does not require CUDA).

Usage:
    HF_TOKEN=... python scripts/fetch_sweep_metrics.py
    HF_TOKEN=... python scripts/fetch_sweep_metrics.py --repo_id Team-Ceciroleo67/kla-ps01-checkpoints
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SWEEP_LABELS = ["a", "b", "c", "d", "e", "f"]
OUT_PATH = REPO_ROOT / "results" / "eda" / "sweep_results.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo_id", default="Team-Ceciroleo67/kla-ps01-checkpoints")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set in environment -- refusing to guess a token from a file "
              "(project rule: token lives only in the process environment).", file=sys.stderr)
        return 2

    from huggingface_hub import HfApi, hf_hub_download
    import torch

    from src.model import build_model, count_parameters

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=args.repo_id, repo_type="model")
    # Match "<run_id>-sweep_X_...-s<seed>/final/sweep_X_....pt" for each of the 6 sweep
    # labels -- run_id/config-suffix are UTC-timestamp/width-depth strings this script does
    # not need to guess, it just lists what actually exists and keys on the label letter.
    by_label: dict[str, str] = {}
    for f in files:
        m = re.match(r"^[^/]*-sweep_(?P<label>[a-f])_[^/]+/final/sweep_[a-f][^/]+\.pt$", f)
        if m:
            label = m.group("label")
            # keep the lexicographically LAST path per label (most recent timestamp) in case
            # a config was re-run
            if label not in by_label or f > by_label[label]:
                by_label[label] = f

    missing = [n for n in SWEEP_LABELS if n not in by_label]
    if missing:
        print(f"missing sweep checkpoints on {args.repo_id}: {missing} "
              f"(found labels: {sorted(by_label)})", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="kla_sweep_fetch_") as tmp:
        for label in SWEEP_LABELS:
            path_in_repo = by_label[label]
            name = f"sweep_{label}"
            if args.verbose:
                print(f"fetching {path_in_repo} ...", flush=True)
            local = hf_hub_download(repo_id=args.repo_id, repo_type="model",
                                    filename=path_in_repo, token=token,
                                    local_dir=tmp)
            ck = torch.load(local, map_location="cpu", weights_only=True)
            cfg = ck["config"]
            metrics = ck["metrics"]
            model_cfg = cfg.get("model", cfg)
            try:
                net = build_model(model_cfg)
                params = count_parameters(net)
            except Exception as exc:  # noqa: BLE001
                print(f"{name}: could not build model to count params: {exc}", file=sys.stderr)
                params = None
            results.append({
                "name": name,
                "path_in_repo": path_in_repo,
                "width": model_cfg.get("width"),
                "num_blocks": model_cfg.get("num_blocks"),
                "params": params,
                "val_psnr": metrics.get("val_psnr"),
                "val_ssim": metrics.get("val_ssim"),
                "val_lpips": metrics.get("val_lpips"),
                "iter": ck.get("iter"),
                "git": ck.get("git"),
            })
            if args.verbose:
                print(f"  {name}: params={params} psnr={metrics.get('val_psnr')} "
                      f"ssim={metrics.get('val_ssim')} lpips={metrics.get('val_lpips')}",
                      flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"repo_id": args.repo_id, "results": results}, indent=2),
                        encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(results)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
