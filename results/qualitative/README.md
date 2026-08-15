# Qualitative results

Visual evidence for V49: 5 success cases and 2 failure cases, every panel at full 256x256 resolution, plus the written failure analysis below.

Regenerate with:

```
py -3.12 scripts/make_qualitative.py --data_root <dataset root> --verbose
```

Every number on every figure and in every table below was measured by that script from the files on disk. `figures.json` in this directory carries the same numbers in machine-readable form.

## What you are looking at

The released imagery is ordinary grayscale photographs. This project treats them as a **proxy** for the degradation problem, not as domain data; nothing here should be read as content-specific tuning.

Panel layout, left to right, identical in every figure:

1. **degraded input** -- the 128x128 `NoisyLR` array enlarged to 256x256 by **pixel replication (nearest)**, so it is displayed at the same size as the others without being given detail it does not have. It is **not clipped**: the real arrays escape [0,1], so each panel is windowed to its own measured min/max and the range is printed in the title.
2. **bicubic x2** -- the baseline of record (`results/baselines/bicubic`).
3. **our model** -- `results/baselines/final`, the saved output of `inference.py`.
4. **ground truth** -- the 256x256 `GT` array.

Panels 2-4 are displayed with a fixed [0,1] window. The only post-processing applied to a prediction anywhere in this project is `np.clip(pred, 0.0, 1.0)`; per-image min-max renormalisation was measured at -4.66 dB PSNR and is forbidden (`docs/decisions.md` D3).

## Provenance and scope

- Split: `configs/split_val.txt (400 pairs)`. There is **no `test_GT`** in the release, so no metric anywhere in this repo is computed on the official test set. Everything here is a held-out slice of `train/`.
- `train/` and `test_NoisyLR/` reuse the same filenames for different images. Every name below refers to `train/`.
- Scores are recomputed from the reloaded `.npy` artifacts on disk (V30), not from cached numbers or in-memory tensors.
- Metric settings are pinned (SPEC 10, asserted by V31):

      psnr  skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=1.0)
      ssim  skimage.metrics.structural_similarity(gt, pred, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

  LPIPS is not printed on these figures (it is a whole-set metric in `results/metrics_summary.md`); PSNR and SSIM are per-image and are shown per panel.

## Validation-set context for these numbers

Over all 400 validation pairs, measured by this script:

- our model: PSNR **28.7865 +/- 4.5329 dB**, SSIM **0.78287 +/- 0.14169** (min 17.2272 dB, median 28.8223 dB, max 40.4185 dB)
- bicubic x2: PSNR **23.6524 +/- 3.0236 dB**, SSIM **0.54775 +/- 0.19197**
- the model beats bicubic on PSNR on **400/400** images

Sanity check on the metric plumbing: on the exact 200-pair subset the bicubic floor of record was measured on (`003000.npy`-`003199.npy`, `docs/decisions.md` D3), this script measures bicubic at **23.4247 +/- 2.8319 dB** PSNR and **0.54462 +/- 0.20392** SSIM, against the recorded 23.4247 +/- 2.8319 dB and 0.54284 +/- 0.20225 (PSNR delta +0.0000 dB, SSIM delta +0.00178).

## How the successes were chosen

Not by eye, and not by taking the best. For each target percentile in [90, 75, 60, 50, 25] of the model PSNR distribution over the 400 validation images, the script takes the image whose PSNR is closest to that percentile (skipping any image already used). The set therefore spans the distribution and **includes the median case**, `001143.npy` at the 50th percentile (28.82 dB), rather than four top-decile images. The strongest case shown is the 90th percentile, not the 100th: the best image in the split scores 40.42 dB and is deliberately not in the pack.

| figure | file | percentile | model PSNR / SSIM | bicubic PSNR / SSIM | gain |
|---|---|---|---|---|---|
| `success_p90_001322_psnr35.18.png` | `001322.npy` | p90 | 35.18 dB / 0.9063 | 31.93 dB / 0.8643 | +3.25 dB |
| `success_p75_003061_psnr31.40.png` | `003061.npy` | p75 | 31.40 dB / 0.9326 | 21.03 dB / 0.4197 | +10.37 dB |
| `success_p60_003154_psnr29.83.png` | `003154.npy` | p60 | 29.83 dB / 0.8913 | 26.23 dB / 0.7840 | +3.59 dB |
| `success_p50_001143_psnr28.82.png` | `001143.npy` | p50 | 28.82 dB / 0.6667 | 22.39 dB / 0.3599 | +6.43 dB |
| `success_p25_003094_psnr25.33.png` | `003094.npy` | p25 | 25.33 dB / 0.7206 | 21.03 dB / 0.4810 | +4.30 dB |

- **`success_p90_001322_psnr35.18.png`** -- validation image `001322.npy`, chosen as the p90 case (35.18 dB vs a p90 target of 35.21 dB). The model gains +3.25 dB PSNR and +0.0420 SSIM over bicubic x2. 0.76% of its GT spectral energy lies above the LR Nyquist limit.
- **`success_p75_003061_psnr31.40.png`** -- validation image `003061.npy`, chosen as the p75 case (31.40 dB vs a p75 target of 31.42 dB). The model gains +10.37 dB PSNR and +0.5129 SSIM over bicubic x2. 0.28% of its GT spectral energy lies above the LR Nyquist limit.
- **`success_p60_003154_psnr29.83.png`** -- validation image `003154.npy`, chosen as the p60 case (29.83 dB vs a p60 target of 29.83 dB). The model gains +3.59 dB PSNR and +0.1073 SSIM over bicubic x2. 0.77% of its GT spectral energy lies above the LR Nyquist limit.
- **`success_p50_001143_psnr28.82.png`** -- validation image `001143.npy`, chosen as the p50 case (28.82 dB vs a p50 target of 28.82 dB). The model gains +6.43 dB PSNR and +0.3068 SSIM over bicubic x2. 1.92% of its GT spectral energy lies above the LR Nyquist limit.
- **`success_p25_003094_psnr25.33.png`** -- validation image `003094.npy`, chosen as the p25 case (25.33 dB vs a p25 target of 25.33 dB). The model gains +4.30 dB PSNR and +0.2396 SSIM over bicubic x2. 5.99% of its GT spectral energy lies above the LR Nyquist limit.

## Failure cases

### The documented hard case is not in the validation split

`000984.npy` is the case on record for unrecoverable high-frequency content. Measured here rather than quoted: its above-LR-Nyquist share of GT spectral energy is **0.8046** (80.46%), which confirms the documented 80.5%.

But `000984.npy` is **not** in `configs/split_val.txt` -- it is a training image, so showing it would not be held-out evidence. The failures below are validation images: the worst-PSNR one, and the one whose above-Nyquist energy (0.8004) reproduces the same regime on held-out data.

| figure | file | model PSNR / SSIM | bicubic PSNR / SSIM | band-limited ceiling PSNR / SSIM | above-Nyquist GT energy |
|---|---|---|---|---|---|
| `fail_worst_psnr_002041_psnr17.23.png` | `002041.npy` | 17.23 dB / 0.5188 | 16.46 dB / 0.4896 | 19.46 dB / 0.6723 | 0.2663 |
| `fail_highest_hf_energy_000300_psnr26.65.png` | `000300.npy` | 26.65 dB / 0.4605 | 25.91 dB / 0.4259 | 27.31 dB / 0.4737 | 0.8004 |

The **band-limited ceiling** is GT with every frequency above the LR Nyquist removed and the result clipped to [0,1]. It is the score of a hypothetical method that recovers the representable band *perfectly* and invents nothing. No non-hallucinating restorer can beat it.

### `002041.npy` -- worst PSNR in the validation split (17.23 dB)

**What the numbers say.** The model scores 17.23 dB PSNR / 0.5188 SSIM here, against a validation mean of 28.79 dB -- rank 1/400 on PSNR (1 = worst) and rank 23/400 on SSIM. It still beats bicubic x2 (16.46 dB / 0.4896, +0.77 dB), so this is not a case where the network is worse than doing nothing clever; it is a case where nothing clever helps much.

**Content.** The frame is dominated by dense, thin, high-contrast structures at every orientation (GT mean 0.7506, std 0.2078, range [0.000, 1.000]). Structures roughly one input pixel wide survive 2x decimation only as a smear. docs/decisions.md D8 measured that four consecutive filenames are four crops of one source frame; all 4 crops of this block are in the split and all of them are hard -- `002041.npy` 17.23 dB (rank 1/400), `002043.npy` 17.36 dB (rank 2/400), `002042.npy` 17.87 dB (rank 3/400), `002040.npy` 19.91 dB (rank 6/400). The difficulty is a property of the content and it reproduces across crops, so it is not a one-off artifact of a single image.

**Why it is unrecoverable, measured.** 26.63% of this image's ground-truth spectral energy lies above the Nyquist limit of the 128x128 input -- the 97.8th percentile of the validation split, whose median is 2.10%. That energy is not attenuated in the input, it is *absent* from it: the sampling grid cannot represent it. Removing exactly that band from the GT and clipping to [0,1] gives a band-limited oracle -- a method that recovers every representable frequency perfectly and invents nothing -- and that oracle scores only 19.46 dB / SSIM 0.6723 on this image. The model is 2.23 dB below that ceiling. Most of the visible shortfall on this figure is missing information, not model error.

**It is broadband texture, not periodic aliasing.** This distinction was tested, not assumed. If the above-Nyquist content were a periodic pattern -- the moire / aliasing story -- its energy would sit in a handful of spectral bins. Measured: the strongest 1% of above-Nyquist bins hold only 9.57% of the above-Nyquist energy here, versus 100.00% for a pure-sinusoid control and 5.51% for a white-noise control on the same 256x256 grid. This case sits at the broadband end and nowhere near the periodic end: the lost content is fine broadband texture spread across the entire band. The moire / periodic-aliasing explanation that was hypothesised for this regime is refuted by that measurement, and this figure must not be captioned as moire.

**What that means for the submission.** The honest ceiling on this image is set by the input, not the architecture. Closing the remaining 2.23 dB would require inventing plausible texture, which is exactly the failure mode an inspection setting cannot tolerate -- hallucinated structure that looks like a defect. This is why no adversarial loss is used (SPEC 7.2). The model degrades to a smooth, honest reconstruction instead of a confident, invented one.

### `000300.npy` -- highest above-Nyquist GT energy in the validation split (0.8004)

**What the numbers say.** The model scores 26.65 dB PSNR / 0.4605 SSIM here, against a validation mean of 28.79 dB -- rank 141/400 on PSNR (1 = worst) and rank 17/400 on SSIM. It still beats bicubic x2 (25.91 dB / 0.4259, +0.74 dB), so this is not a case where the network is worse than doing nothing clever; it is a case where nothing clever helps much. It is filed as a failure on structural, not pixel, grounds: PSNR is unremarkable here only because the frame is nearly black, while SSIM is rank 17/400 and the reconstruction is visibly smoother than the reference.

**Content.** A very dark, low-contrast frame (GT mean 0.0825, std 0.0482, 99.5th percentile 0.164) whose entire content is fine-grained broadband texture. The figure therefore includes a second row with a display-only stretch so the structure is visible on screen; the stretch is not applied to the scored data. This is the held-out analogue of the documented hard case 000984.npy, whose above-Nyquist energy measures 0.8046 here against 0.8004 for this image.

**Why it is unrecoverable, measured.** 80.04% of this image's ground-truth spectral energy lies above the Nyquist limit of the 128x128 input -- the 99.8th percentile of the validation split, whose median is 2.10%. That energy is not attenuated in the input, it is *absent* from it: the sampling grid cannot represent it. Removing exactly that band from the GT and clipping to [0,1] gives a band-limited oracle -- a method that recovers every representable frequency perfectly and invents nothing -- and that oracle scores only 27.31 dB / SSIM 0.4737 on this image. The model is 0.67 dB below that ceiling. Most of the visible shortfall on this figure is missing information, not model error.

**It is broadband texture, not periodic aliasing.** This distinction was tested, not assumed. If the above-Nyquist content were a periodic pattern -- the moire / aliasing story -- its energy would sit in a handful of spectral bins. Measured: the strongest 1% of above-Nyquist bins hold only 5.66% of the above-Nyquist energy here, versus 100.00% for a pure-sinusoid control and 5.51% for a white-noise control on the same 256x256 grid. This case sits at the broadband end and nowhere near the periodic end: the lost content is fine broadband texture spread across the entire band. The moire / periodic-aliasing explanation that was hypothesised for this regime is refuted by that measurement, and this figure must not be captioned as moire.

**What that means for the submission.** The honest ceiling on this image is set by the input, not the architecture. Closing the remaining 0.67 dB would require inventing plausible texture, which is exactly the failure mode an inspection setting cannot tolerate -- hallucinated structure that looks like a defect. This is why no adversarial loss is used (SPEC 7.2). The model degrades to a smooth, honest reconstruction instead of a confident, invented one.

## Per-figure measurements

| file | model PSNR | model SSIM | bicubic PSNR | bicubic SSIM | PSNR gain | above-Nyquist GT energy |
|---|---|---|---|---|---|---|
| `001322.npy` | 35.18 | 0.9063 | 31.93 | 0.8643 | +3.25 | 0.0076 |
| `003061.npy` | 31.40 | 0.9326 | 21.03 | 0.4197 | +10.37 | 0.0028 |
| `003154.npy` | 29.83 | 0.8913 | 26.23 | 0.7840 | +3.59 | 0.0077 |
| `001143.npy` | 28.82 | 0.6667 | 22.39 | 0.3599 | +6.43 | 0.0192 |
| `003094.npy` | 25.33 | 0.7206 | 21.03 | 0.4810 | +4.30 | 0.0599 |
| `002041.npy` | 17.23 | 0.5188 | 16.46 | 0.4896 | +0.77 | 0.2663 |
| `000300.npy` | 26.65 | 0.4605 | 25.91 | 0.4259 | +0.74 | 0.8004 |

## Files

| file | bytes |
|---|---|
| `fail_highest_hf_energy_000300_psnr26.65.png` | 568,384 |
| `fail_worst_psnr_002041_psnr17.23.png` | 326,213 |
| `success_p25_003094_psnr25.33.png` | 276,564 |
| `success_p50_001143_psnr28.82.png` | 248,743 |
| `success_p60_003154_psnr29.83.png` | 249,008 |
| `success_p75_003061_psnr31.40.png` | 246,779 |
| `success_p90_001322_psnr35.18.png` | 188,305 |

