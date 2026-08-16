# Final submission checklist

Current as of 2026-08-16. Re-run the verifier command before release; a generated local JSON
report is not the authority.

## Required repository items

- [x] `inference.py` has exactly two required arguments: `--input_dir`, `--output_dir`.
- [x] `train.py`, `configs/final.yaml`, `src/`, and pinned `requirements.txt` are tracked.
- [x] `weights/best.pt` is tracked and loads without manual placement.
- [x] `results/experiments.csv` is tracked.
- [x] Validation metrics, qualitative success/failure cases, and runtime evidence are tracked.
- [x] 400 final outputs are published with a real URL plus archive/per-file checksums.

## Checkpoint

- Tracked SHA256:
  `cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054`.
- NAFSR width 48, 16 blocks, 388,225 parameters, EMA at iteration 20,000.
- Canonical config equals the checkpoint's embedded config exactly.
- Model and EMA tensors are byte-identical to public release checkpoint SHA
  `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313`.
- Source provenance preserves the original dirty marker and pins exact source identities.

## Validation quality

Full fixed 400-pair split, reloaded CPU-fp32 outputs:

- PSNR: 28.7864 dB
- SSIM: 0.78286
- LPIPS: 0.25323
- U-Net comparison: 28.8808 / 0.78273 / 0.26525

The selected NAFSR fixes the prior all-metric regression, significantly improves LPIPS over
U-Net, ties SSIM, and trades 0.0944 dB PSNR while using 7.65x fewer parameters. D41 records the
honest paired result and selection rationale.

## Published final outputs

- URL: https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip
- SHA256: `b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1`
- Size: 90,990,452 bytes
- Files: 400, flat `.npy` archive
- `manifest.csv` checkout SHA256:
  `30e8a921b7b55c365ca337da6ca73732e4fd0779ea940eebc7725efd76ce1ba6`
- `sha256sums.txt`: 400 per-file digests

The final test set has no GT. No final-test PSNR, SSIM, or LPIPS is claimed.

## Runtime

Headline: local Mac CPU external-process benchmark, 400 images in 106.43 s (3.8 img/s),
batch 32, fp32. Replace the number only with a newly generated end-to-end benchmark and update
all four runtime records together.

The 20.09 s RTX 4060 CUDA number is labeled only as historical release-output generation.
No final H100 runtime is claimed.

## Data-dependent checks

Set `KLA_DATA_ROOT` to a root containing `train/GT`, `train/NoisyLR`, and `NoisyLR` (or
historical `test_NoisyLR`). On the measured Mac the root is
`/Users/shanmukhsai/Downloads`. Final-test inputs are inference-only and have no GT.

## Final commands

```bash
python inference.py --input_dir sample_inputs --output_dir /tmp/kla-final-check
python inference.py --input_dir sample_inputs --output_dir /tmp/kla-final-required --require_weights
python -m unittest discover -s tests -v
KLA_DATA_ROOT=/path/to/dataset python scripts/verify_all.py --strict --fresh-clone
git diff --check
git status --short
```
