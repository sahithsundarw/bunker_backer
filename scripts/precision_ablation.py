"""Phase B1: price the V22 bf16/fp32 trade-off in dB/SSIM/LPIPS, not just measure the raw
pixel divergence V22 already reports.

D61 accepted V22 (bf16 vs fp32 output divergence: mean 1.85e-3, max 2.65e-2) as a "disclosed
trade-off ... for throughput" without ever measuring what bf16 actually COSTS in restoration
quality, or what fp32 actually COSTS in throughput. This prices both sides with the real
`inference.py` forward path (not a re-implemented one), on the full 400-pair val split, with a
PAIRED comparison (src.metrics.paired_compare) so a small mean shift cannot be over- or
under-read.

Decision rule, fixed BEFORE running this (not adjusted after seeing the result): if fp32 wins
any metric with paired significance AND costs less than ~15% throughput (per
scripts/benchmark_runtime.py, run separately), switch `inference.py --precision auto`'s CUDA
default from bf16 to fp32. Otherwise keep bf16 and record the measured price of the trade-off.
This script only measures; it does not itself flip the default -- that is a separate, disclosed
follow-up edit if the rule says to.

No training, no fitting -- pure evaluation of the already-shipped checkpoint.
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


def _load_model(device: str) -> torch.nn.Module:
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})
    model = build_model(cfg.get("model", cfg))
    state = ckpt.get("ema") or ckpt.get("model")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _infer(model: torch.nn.Module, lr: np.ndarray, device: str, precision: str) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(device)
    if precision == "fp32":
        with torch.no_grad():
            y = model(x)
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
            y = model(x)
    return np.clip(y[0, 0].float().cpu().numpy(), 0.0, 1.0)


def score_precision(model: torch.nn.Module, device: str, precision: str, lr_dir: Path,
                    gt_dir: Path, names: list[str], with_lpips: bool) -> list[dict[str, Any]]:
    out = []
    for name in names:
        lr = np.load(lr_dir / name, allow_pickle=False)
        gt = np.load(gt_dir / name, allow_pickle=False)
        pred = _infer(model, lr, device, precision)
        s = score_pair(pred, gt, with_lpips=with_lpips, device=device)
        out.append({"file": name, **s})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device -- bf16 autocast on CPU is not representative of the "
              "real inference.py path, which only uses bf16 on CUDA. Results from a CPU run "
              "should not be used to make the switch decision.", file=sys.stderr)

    t_start = time.time()
    model = _load_model(device)
    data_root = resolve_data_root(args.data_root)
    _, names = train_val_names(data_root)
    if args.n and args.n < len(names):
        names = names[: args.n]
    lr_dir, gt_dir = data_root / "train" / "NoisyLR", data_root / "train" / "GT"

    results: dict[str, list[dict[str, Any]]] = {}
    for precision in ("bf16", "fp32"):
        if args.verbose:
            print(f"scoring {len(names)} images at precision={precision} ...")
        results[precision] = score_precision(model, device, precision, lr_dir, gt_dir, names,
                                              with_lpips=True)

    paired = paired_compare(results["fp32"], results["bf16"])  # fp32 = candidate, bf16 = ref

    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([r[key] for r in rows]))

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "checkpoint": str(CKPT_PATH),
        "n": len(names),
        "means": {
            p: {m: _mean(results[p], m) for m in ("psnr", "ssim", "lpips")}
            for p in ("bf16", "fp32")
        },
        "paired_fp32_vs_bf16": paired,
        "decision_rule": ("fixed before running: switch inference.py's CUDA default to fp32 "
                          "only if fp32 wins ANY metric with paired significance AND the "
                          "separately-measured throughput cost is <15% (see "
                          "scripts/benchmark_runtime.py --precision fp32 vs --precision bf16)"),
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    out_path = ROOT / "results" / "eda" / "precision_ablation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    for m, v in paired.items():
        print(f"  {m}: fp32 mean={report['means']['fp32'][m]:.5f} "
              f"bf16 mean={report['means']['bf16'][m]:.5f} "
              f"diff={v['mean_diff']:+.5f} t={v['t']:+.2f} win={v['win']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
