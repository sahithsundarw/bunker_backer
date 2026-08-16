# Runtime report

## Runtime headline

**Local Mac CPU external-process benchmark: 400 images in 71.72 s (5.6 img/s), batch size 32,
fp32.** This is the repository's single headline runtime measurement. It was measured on
2026-08-16 by `scripts/benchmark_runtime.py`, which timed process creation through exit.

## Headline measurement

| Field | Value |
|---|---|
| Device | Apple Silicon Mac CPU (`arm64`), macOS 26.6.1 |
| Runtime label | **Local Mac CPU external-process benchmark** |
| Timing scope | external subprocess timer, process creation through exit |
| Image count | 400 |
| Total end-to-end wall-clock | 71.72 s (one measured run) |
| Throughput | 5.6 img/s |
| Batch size | 32 |
| Precision | fp32 |
| Torch | 2.13.0 |
| Main pipeline, including model load, IO, compute, transfers, and writes | 70.34 s |
| Process startup/import overhead | 1.38 s |
| Checkpoint | `weights/best.pt` |
| Checkpoint SHA256 | `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7` |
| Checkpoint size | 2,068,091 bytes (1.97 MiB) |
| Total parameter count | 246,529 (84,049 frozen + 162,480 trainable) |
| Trainable parameter count | 162,480 |

Benchmark command:

```bash
python scripts/benchmark_runtime.py \
  --input_dir /Users/shanmukhsai/Downloads/NoisyLR \
  --out /tmp/kla-runtime-full-20260816.md \
  --repeats 1 \
  --device cpu \
  --precision fp32
```

Raw harness output:

```
repeat 1/1: 71.72s external, 70.34s main pipeline
```

The external timing window includes interpreter startup, imports, checkpoint loading, input
reads, CPU-to-device transfers, model execution, device-to-CPU transfers, clipping, and output
writes. The main-pipeline number is parsed from `inference.py` only to estimate startup/import
overhead; it is not substituted for the external headline.

## Other explicitly labeled runs

**Release-output generation / local Mac CPU:** the earlier run that produced the archived
outputs reported 56.73 s (7.1 img/s) from `inference.py`'s internal `main()` timer. It excludes
interpreter and module-import startup and is provenance for those output bytes, not the runtime
headline.

**Linux/CUDA fresh clone:** V04/V46 passed in a `python:3.12-slim` container with the pinned
CUDA 12.8 packages. That was a compatibility check; no Linux/CUDA or H100 runtime was measured.

The final test set has no ground truth. Runtime and throughput are reported for it; PSNR, SSIM,
and LPIPS are not.
