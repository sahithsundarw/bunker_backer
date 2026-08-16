# Published restored test outputs

This directory tracks the release manifest and per-file hashes. The 400 `.npy` output bytes
are published as a GitHub Release asset because the 91 MB archive exceeds the repository's
tracked-artifact budget.

## Status

The artifact is published and publicly downloadable:

- Asset: `restored_test_outputs.zip`
- URL: https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip
- SHA256: `b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1`
- Size: 90,990,452 bytes
- Layout: 400 flat `.npy` files at the archive root

A fresh public download returned HTTP 200 with the recorded size and digest. All 400 extracted
files were reloaded and checked against the released inputs. Each is float32, 2-D, `(256,256)`,
finite, clipped to `[0,1]`, and named exactly like its matching `(128,128)` input.

Verify the archive:

```bash
curl -L -o restored_test_outputs.zip https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip
sha256sum restored_test_outputs.zip
unzip restored_test_outputs.zip -d restored_test_outputs
cd restored_test_outputs && sha256sum -c /path/to/repository/results/restored_test_outputs/sha256sums.txt
```

The archive digest must be
`b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1`.

## Provenance

| Field | Value |
|---|---|
| Inputs | 400 released final-test arrays, `000000.npy` through `000399.npy` |
| Ground truth | None; final-test GT is withheld |
| Reproduction command | `python inference.py --input_dir <data-root>/NoisyLR --output_dir <out> --require_weights --verbose` |
| Checkpoint | EMA weights, tracked SHA `cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054` |
| Current tracked checkpoint | SHA `cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054` |
| Output cross-check | Public vs tracked-checkpoint CPU-fp32 reproduction: 65.72 dB mean inter-output PSNR, 54.90 dB minimum; cross-device byte identity is not claimed |
| Reproduction source | Git `659fa26024567c151eeb7d5efda1489398926403`, clean tree |

Normal submission inference requires the tracked checkpoint. Bicubic output is available only
with the explicit `--allow_bicubic_fallback` demo flag. The producing command includes
`--require_weights`, so the published outputs cannot be an accidental fallback.

## Manifests

- `manifest.json` is the machine-readable archive, checkpoint, runtime, and output-contract
  record used by verifier V56.
- `manifest.csv` contains one measured row per extracted output. It is generated with LF line
  endings by `scripts/build_output_manifest.py`; its checkout SHA256 is
  `30e8a921b7b55c365ca337da6ca73732e4fd0779ea940eebc7725efd76ce1ba6`.
- `sha256sums.txt` contains the 400 canonical per-file hashes.

Rebuild the two per-file manifests after extracting the archive:

```bash
python scripts/build_output_manifest.py \
  --input_dir /path/to/released/NoisyLR \
  --output_dir /path/to/extracted/restored_test_outputs \
  --manifest_dir results/restored_test_outputs
```

## Runtime Labels

The repository headline is the **local Mac CPU external-process benchmark: 400 images in
106.43 s (3.8 img/s), batch size 32, fp32**. The published bytes were generated in a separate
historical **prior release-output generation** run on an RTX 4060 Laptop GPU: 20.09 s from the
internal pipeline timer. That timing is not attributed to the replacement archive bytes.
Linux/CUDA fresh-clone compatibility was checked, but no final
Linux/CUDA runtime was measured. These are intentionally distinct measurements; see
`results/runtime_report.md`.

No PSNR, SSIM, or LPIPS is computed or claimed for the 400 final-test outputs because no GT
exists. The validation metrics elsewhere in the repository are from the held-out 400-pair
split of `train/`.
