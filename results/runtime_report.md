# Runtime report

## Measured Local CPU Run

| Field | Value |
|---|---|
| Device | Apple Silicon Mac CPU (`arm64`), macOS 26.6.1 |
| Runtime scope | **Local CPU final-output generation; not an official H100 or CUDA measurement** |
| Image count | 400 |
| Total local generation wall-clock | 56.73 s |
| Throughput | 7.1 img/s |
| Batch size | 32 |
| Precision | fp32 |
| Torch | 2.13.0 |
| Checkpoint | `weights/best.pt` |
| Checkpoint SHA256 | `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7` |
| Checkpoint size | 2,068,091 bytes (1.97 MiB) |
| Total parameter count | 246,529 (84,049 frozen + 162,480 trainable) |
| Trainable parameter count | 162,480 |

Command:

```bash
python inference.py \
  --input_dir /Users/shanmukhsai/Downloads/NoisyLR \
  --output_dir /tmp/semicon_final_outputs_28db \
  --weights weights/best.pt \
  --require_weights \
  --batch_size 32 \
  --device cpu \
  --precision fp32 \
  --verbose
```

Raw output of the timed run:

```
loaded weights/best.pt (ema weights)
restored 400/400 in 56.73s (7.1 img/s) | device=cpu precision=fp32 batch=32 shapes=[(128, 128)] weights=best unreadable=0 write_errors=0
```

The reported timing is the measured local generation run supplied with the final output
artifact. A startup-vs-compute breakdown was not separately measured: `scripts/benchmark_runtime.py`
(the external-process harness intended to produce that breakdown per SPEC 11.4 step 7) is not
yet implemented — see `docs/STATE.md` and `docs/AUDIT_20260815.md` for V37/V38/V39. This number
must not be relabelled as an official end-to-end benchmark, H100 result, CUDA result, or
competition-server runtime; it is a local Mac CPU measurement only.

This checkpoint (`r2_nb8_psnrloss`, 28.0394 dB) is smaller and slower per-image than the earlier
LS-5-only checkpoint (`d5807dab...`, 3.14 MiB, previously measured at 31.8 img/s on the same
machine): the residual NAFSR body adds compute per forward pass in exchange for the PSNR/SSIM/
LPIPS improvement documented in `weights/README.md` and `docs/decisions.md` D28/D29.

The final test set has no ground truth. Runtime and throughput are reported for it; PSNR, SSIM,
and LPIPS are not.
