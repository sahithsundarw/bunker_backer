# Runtime report

## Runtime headline

**Local Mac CPU external-process benchmark: 400 images in 106.43 s (3.8 img/s), batch size
32, fp32.** This is the repository's single headline measurement. It was generated on
2026-08-16 by `scripts/benchmark_runtime.py` after final output-staging hardening.

| Field | Value |
|---|---|
| Runtime label | **Local Mac CPU external-process benchmark** |
| Device | macOS 26.6.1 arm64 / inference device `cpu` |
| End-to-end wall clock | 106.43 s, one measured run |
| Throughput | 3.8 img/s |
| Images | 400 |
| Batch / precision | 32 / fp32 |
| Torch | 2.13.0 |
| Main pipeline | 105.09 s |
| Process startup and imports | 1.34 s |
| Checkpoint | `weights/best.pt` |
| Checkpoint SHA256 | `cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054` |
| Checkpoint size | 3,291,621 bytes |
| Model | NAFSR, 388,225 parameters |

The external timer starts immediately before process creation and ends after process exit. It
includes interpreter startup, imports, checkpoint load, disk reads, preprocessing, CPU/device
movement, model execution, post-processing, atomic writes, staging reconciliation, and final
output validation. The main-pipeline number comes from `inference.py` only to show startup
overhead; it is not substituted for the external headline.

**Total end-to-end wall-clock** is the headline value above. The 105.09 s versus 1.34 s rows
provide the required **startup-vs-compute breakdown**; model load and IO remain inside the main
pipeline rather than being mislabeled as pure compute.

Reproduce on the measured dataset:

```bash
python scripts/benchmark_runtime.py \
  --input_dir /Users/shanmukhsai/Downloads/NoisyLR \
  --out /tmp/kla-runtime.md \
  --repeats 1 \
  --device cpu \
  --precision fp32
```

Raw harness output: `repeat 1/1: 106.43s external, 105.09s main pipeline`.

## Other Labeled Runs

**Release-output generation on NVIDIA RTX 4060 Laptop CUDA:** the historical run that
produced the published 400-file archive reported 20.09 s (19.9 img/s), batch size 32, bf16,
from `inference.py`'s internal main-pipeline timer. It is provenance for those release bytes,
not the runtime headline and not an external-process benchmark.

**Linux/CUDA fresh clone:** dependency installation, checkpoint loading, and inference
compatibility passed in the recorded CUDA 12.8 environment. No Linux/CUDA wall-clock runtime
was measured in the final hardening run.

**Not claimed:** no H100 or competition-server runtime has been measured. CUDA timing in
`scripts/make_baselines.py` synchronizes immediately before and after each model forward;
CPU timing is separately labeled as CPU wall clock.

The released final test set has no ground truth. Runtime is reported for it; no final-test
PSNR, SSIM, or LPIPS is claimed.
