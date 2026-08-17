#!/usr/bin/env python3
"""FP8 measurement (plan PRIORITY 1, P1.4): complete the quantization story alongside the
already-measured INT8 result (docs/decisions.md D54), rather than leaving FP8 mentioned but
unmeasured.

Two things are measured, honestly, in order:

1.  Whether the shipped architecture can even RUN in FP8 end to end. NAFSR is convolutional
    (3x3 depthwise + 1x1 pointwise convs via nn.Conv2d), not attention/matmul-only. PyTorch's
    native cuDNN conv backend does not implement an FP8 kernel -- this is checked directly by
    calling ``F.conv2d`` on ``float8_e4m3fn`` tensors and recording the exact error, not
    assumed from documentation.
2.  A bounded GEMM-level proxy: the model's own 1x1-pointwise-conv shapes (channels=64,
    dw_expand=2/ffn_expand=2 -> 64<->128, src/blocks.py NAFBlock) reshaped to matmul form
    (M=tokens, K=in_channels, N=out_channels) and timed via ``torch._scaled_mm`` (FP8, the only
    native PyTorch FP8 compute primitive that DOES work on this hardware) against the same
    shape in bf16 (the model's actual default precision, inference.py resolve_precision), on
    the SAME GPU this project already benchmarks on. This characterises whether the underlying
    compute primitive would even help if a custom FP8 conv kernel existed -- consistent with
    D21/D54's memory-bandwidth-bound finding, no benefit is expected, and this either confirms
    or contradicts that with a number rather than an assumption.

Explicitly scoped per the user's Round 2 clarification: PyTorch-native only, no custom
Triton/CUTLASS kernels, no TensorRT/ONNX. If (1) fails (it does, see below), a native
end-to-end FP8 path for this architecture does not exist within that scope -- stated as a
finding, not routed around.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "eda" / "fp8_probe.json"


def _time_gemm(fn, warmup: int = 10, iters: int = 50) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print("no CUDA device available -- FP8 tensor cores require CUDA", file=sys.stderr)
        return 1
    dev_name = torch.cuda.get_device_name(0)
    dev_cap = torch.cuda.get_device_capability(0)
    print(f"device: {dev_name} (compute capability {dev_cap})")

    result: dict = {"device": dev_name, "compute_capability": list(dev_cap)}

    # --- Part 1: end-to-end conv2d in FP8 -- does the native kernel exist at all? ----------
    x = torch.randn(1, 3, 16, 16, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(3, 3, 3, 3, device="cuda").to(torch.float8_e4m3fn)
    try:
        F.conv2d(x, w, padding=1)
        result["conv2d_fp8_supported"] = True
        result["conv2d_fp8_error"] = None
    except Exception as exc:  # noqa: BLE001
        result["conv2d_fp8_supported"] = False
        result["conv2d_fp8_error"] = f"{type(exc).__name__}: {exc}"
        print(f"Part 1: native cuDNN conv2d does NOT support FP8 inputs -- "
              f"{result['conv2d_fp8_error']}")

    # --- Part 2: bounded GEMM-level proxy at the model's own pointwise-conv shapes ---------
    # NAFBlock (src/blocks.py, width=64, dw_expand=2, ffn_expand=2): pointwise convs are
    # literally 1x1, i.e. exact GEMMs once channels are the trailing dim. M = tokens in a
    # representative 256x256 feature map (this project's stated eval resolution range).
    torch.manual_seed(0)
    M = 256 * 256
    shapes = [("64->128 (FFN expand)", 64, 128), ("128->64 (FFN project)", 128, 64)]
    gemm_results = []
    for label, k, n in shapes:
        a_bf16 = torch.randn(M, k, device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        t_bf16 = _time_gemm(lambda: torch.matmul(a_bf16, b_bf16))

        a_fp8 = torch.randn(M, k, device="cuda").to(torch.float8_e4m3fn)
        # cuBLASLt's FP8 GEMM requires B in column-major layout.
        b_fp8 = torch.randn(k, n, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
        scale_a = torch.tensor(1.0, device="cuda")
        scale_b = torch.tensor(1.0, device="cuda")
        try:
            t_fp8 = _time_gemm(lambda: torch._scaled_mm(
                a_fp8, b_fp8, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16))
            speedup = t_bf16 / t_fp8
            print(f"Part 2 [{label}] M={M} K={k} N={n}: bf16={t_bf16*1e3:.4f} ms  "
                  f"fp8={t_fp8*1e3:.4f} ms  speedup={speedup:.3f}x")
            gemm_results.append({"label": label, "M": M, "K": k, "N": n,
                                 "bf16_ms": t_bf16 * 1e3, "fp8_ms": t_fp8 * 1e3,
                                 "speedup": speedup, "error": None})
        except Exception as exc:  # noqa: BLE001
            print(f"Part 2 [{label}]: fp8 scaled_mm failed: {type(exc).__name__}: {exc}")
            gemm_results.append({"label": label, "M": M, "K": k, "N": n,
                                 "bf16_ms": t_bf16 * 1e3, "fp8_ms": None,
                                 "speedup": None, "error": f"{type(exc).__name__}: {exc}"})
    result["gemm_probe"] = gemm_results

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
