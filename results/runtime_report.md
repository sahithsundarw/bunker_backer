# Runtime report

## Measured Local CPU Run

| Field | Value |
|---|---|
| Device | Apple Silicon Mac CPU (`arm64`), macOS 26.6.1 |
| Runtime scope | Local CPU final-output generation; not an official H100 or CUDA measurement |
| Image count | 400 |
| Total local generation wall-clock | 12.57 s |
| Throughput | 31.8 img/s |
| Batch size | 32 |
| Precision | fp32 |
| Torch | 2.13.0 |
| Checkpoint | `weights/best.pt` |
| Checkpoint SHA256 | `d5807dabad37b251f25d066838da9e3f73c164ec37ee777505a80e23cad9e90d` |
| Checkpoint size | 3,288,933 bytes (3.14 MiB) |
| Trainable parameter count | 388,225 |

Command:

```bash
python inference.py \
  --input_dir /Users/shanmukhsai/Downloads/NoisyLR \
  --output_dir /tmp/semicon_final_outputs \
  --weights weights/best.pt \
  --require_weights \
  --batch_size 32 \
  --device cpu \
  --precision fp32 \
  --verbose
```

The reported timing is the measured local generation run supplied with the final output
artifact. A startup-vs-compute breakdown was not separately measured. The repository's
external-process benchmark harness is not yet implemented, so this number must not be relabelled
as an official end-to-end benchmark, H100 result, CUDA result, or competition-server runtime.

The final test set has no ground truth. Runtime and throughput are reported for it; PSNR, SSIM,
and LPIPS are not.
