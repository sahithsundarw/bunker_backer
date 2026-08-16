#!/usr/bin/env python3
"""Phase 2 (Round 2 differentiation, docs/decisions.md D54): PyTorch-native INT8 static
quantization experiment.

No TensorRT (user's explicit choice). torch.ao.quantization's classic static-quantization
backends (fbgemm/onednn/qnnpack) are CPU-only by design -- this measures the CPU-fallback
path SPEC requires (--device cpu must not crash), not the scored GPU/H100 axis, which already
uses bf16 (the fast, measured-effective lever, D21).

Calibrates on real TRAINING-split images only (never test_NoisyLR, F17 discipline). Result
recorded in docs/decisions.md D54: INT8 measured 2.06x SLOWER than fp32 on this architecture,
not faster, for a small quality cost -- a genuine negative result, reported honestly, and
consistent with D21's prior finding that this model is memory-bandwidth-bound, not
compute-bound (the win INT8 offers) at this parameter/spatial scale.

Usage:
    py -3.12 scripts/quantize_experiment.py --data_root C:\\kla-data

Owner: main session (Round 2 differentiation, not on the parallel-agent ownership map).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.blocks import SCA  # noqa: E402
from src.metrics import clip_prediction, psnr as m_psnr, ssim as m_ssim  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402


def score(model: torch.nn.Module, names: list[str], lr_dir: Path, gt_dir: Path,
         tag: str, verbose: bool) -> tuple[float, float, float]:
    psnrs, ssims = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for name in names:
            x = torch.from_numpy(np.load(lr_dir / name))[None, None]
            y = model(x)
            pred = clip_prediction(y[0, 0].numpy().astype(np.float32))
            gt = np.load(gt_dir / name).astype(np.float32)
            psnrs.append(m_psnr(pred, gt))
            ssims.append(m_ssim(pred, gt))
    dt = time.perf_counter() - t0
    if verbose:
        print(f"[{tag}] n={len(names)} psnr={np.mean(psnrs):.4f}+/-{np.std(psnrs):.4f} "
              f"ssim={np.mean(ssims):.5f} wall={dt:.2f}s ms/img={1000 * dt / len(names):.3f}")
    return float(np.mean(psnrs)), float(np.mean(ssims)), dt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data_root", required=True,
                    help="dataset root (contains train/GT, train/NoisyLR)")
    ap.add_argument("--weights", default=str(REPO_ROOT / "weights" / "best.pt"))
    ap.add_argument("--calib_n", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    gt_dir, lr_dir = root / "train" / "GT", root / "train" / "NoisyLR"

    ck = torch.load(args.weights, map_location="cpu", weights_only=True)
    fp32_model = build_model(ck["config"]).eval()
    fp32_model.load_state_dict(ck["ema"])

    calib_files = sorted(lr_dir.glob("*.npy"))[:args.calib_n]
    calib_x = torch.stack([torch.from_numpy(np.load(f))[None] for f in calib_files])

    engine = "onednn" if "onednn" in torch.backends.quantized.supported_engines else "fbgemm"
    torch.backends.quantized.engine = engine
    print(f"quantized engine: {engine} (supported: {torch.backends.quantized.supported_engines})")

    qconfig_mapping = get_default_qconfig_mapping(engine)
    example_inputs = (calib_x[:1],)
    # SCA.forward does device.type-dependent autocast control flow (the V22 fp32-precision
    # fix, src/blocks.py) -- FX symbolic tracing can't trace a Python-level attribute read
    # off a traced tensor proxy. Treat SCA as an opaque leaf module; its own 1x1 conv stays
    # fp32 in the quantized model as a result -- a stated limitation, not a silent gap.
    prepare_config = PrepareCustomConfig().set_non_traceable_module_classes([SCA])
    prepared = prepare_fx(fp32_model, qconfig_mapping, example_inputs,
                          prepare_custom_config=prepare_config)
    with torch.no_grad():
        for i in range(calib_x.shape[0]):
            prepared(calib_x[i:i + 1])
    quantized_model = convert_fx(prepared)
    print("quantization convert done")

    split_file = REPO_ROOT / "configs" / "split_val.txt"
    names = [ln.strip() for ln in split_file.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    print(f"n val = {len(names)}")

    fp32_psnr, fp32_ssim, fp32_dt = score(fp32_model, names, lr_dir, gt_dir, "fp32-cpu", True)
    int8_psnr, int8_ssim, int8_dt = score(quantized_model, names, lr_dir, gt_dir, "int8-cpu", True)

    print()
    print(f"delta psnr: {int8_psnr - fp32_psnr:+.4f} dB   delta ssim: {int8_ssim - fp32_ssim:+.5f}")
    print(f"speedup: {fp32_dt / int8_dt:.3f}x  ({'faster' if int8_dt < fp32_dt else 'SLOWER'})")
    print(f"fp32 params: {count_parameters(fp32_model):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
