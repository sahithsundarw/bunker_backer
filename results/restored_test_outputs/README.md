# Final restored-output evidence

This directory contains the committed manifests for the 400 restored final-test outputs. The
raw `.npy` files are packaged separately as `semicon_final_outputs.zip`; the 90,992,260-byte
archive is intentionally not committed to Git and must be uploaded as a GitHub Release or
submission asset.

## Provenance

| Field | Measured value |
|---|---|
| Inputs | `/Users/shanmukhsai/Downloads/NoisyLR` |
| Outputs | 400 `.npy` files in `/tmp/semicon_final_outputs` |
| Output contract | `(256, 256)`, `float32`, finite, clipped to `[0, 1]` |
| Checkpoint | `weights/best.pt` |
| Checkpoint SHA256 | `d5807dabad37b251f25d066838da9e3f73c164ec37ee777505a80e23cad9e90d` |
| Archive | `semicon_final_outputs.zip` |
| Archive SHA256 | `17fdeba4b8d19b5b4ddbe0b3d430851b9385200c8922964b354c70ba108d5eed` |
| Archive size | 90,992,260 bytes |
| Per-file manifest | `manifest.csv` (400 rows) |

Generation command:

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

`--require_weights` is mandatory evidence: a missing or invalid checkpoint must fail instead
of silently producing bicubic fallback outputs.

## Validation

`manifest.csv` records each filename, file SHA256, shape, dtype, minimum, maximum, finite
status, and whether the matching input exists. All 400 rows were validated from files reloaded
from disk:

- output filenames exactly match the 400 input filenames;
- every shape is `(256, 256)` and every dtype is `float32`;
- every value is finite and inside `[0, 1]`;
- every output has a matching released input.

`manifest.json` records archive-level provenance. Its `release_url` remains empty until the
archive is uploaded; filling that field with the public asset URL is the remaining manual
submission step.

## No Final-Test Metrics

The released final test set contains inputs only and has no ground truth. Therefore no PSNR,
SSIM, or LPIPS is computed or reported for these 400 outputs. The model metrics elsewhere in
the repository are measured only on the committed held-out validation split of `train/`.

## Release Step

Upload `/tmp/semicon_final_outputs.zip` as a release/submission asset, verify the downloaded
bytes reproduce SHA256
`17fdeba4b8d19b5b4ddbe0b3d430851b9385200c8922964b354c70ba108d5eed`, then add the public
asset URL to `manifest.json`. Do not add the archive itself to Git.
