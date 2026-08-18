# KLA PS01 Deck Content — for manual PPT assembly

Team: **bunker_backer**. Give this file to whoever is building the PPT — every slide's text
and image is below, in order. Image paths are absolute; open the folder in Windows Explorer
or click through in a code/file viewer.

Repo root: `C:\Users\sahit\OneDrive\Desktop\semi`

---

## Slide 1 — Team Details
*(No image)*

```
Team name: bunker_backer

Members:
  S Sahith Somasundar — modelling
  Gara Shanmukh Sai — data & augmentation
  Navadeep Saran Y — inference optimization
  Vanisha Nadimpalli — evaluation & docs

College: VNR Vignana Jyothi Institute of Engineering and Technology
Contact: navadeep.saran@gmail.com
```

---

## Slide 2 — Problem Statement Addressed
*(No image)*

```
AI-Based Restoration of Degraded Images for Semiconductor Inspection (KLA PS01).
A single lost pixel or noisy region in an inspection image can hide a real defect
and cost a die — restoration quality is a correctness problem, not a cosmetic one.

Three degradation mechanisms, applied in combination:
  1. Speckle noise (signal-dependent, multiplicative-like variance term)
  2. Additive/shot noise (signal-dependent linear term)
  3. Downsampling (×2 decimation, GT/LR pair)

The pipeline must invert all three jointly, in whatever order they were applied,
and generalise to content the training set never showed it.
```

---

## Slide 3 — Idea Description

**Image:**
`C:\Users\sahit\OneDrive\Desktop\semi\results\eda\noise_variance_vs_intensity.png`

```
The released dataset is 3200 training pairs and 400 test inputs of grayscale
natural photographs, not semiconductor imagery. We treat it as a proxy: the
degradation — ×2 decimation plus signal-dependent noise — is what transfers to
inspection imagery, so we characterised the degradation empirically and
optimised for degradation robustness rather than fitting content-specific priors.

Measured: downsample kernel is a recovered 4×4 sharpening kernel (bicubic with
antialias off is within 1.22e-05 of optimal); noise is applied AFTER
downsampling, signal-dependent, no additive Gaussian floor (residual
autocorrelation ~0).

Core concept: one-step blind joint restoration, all compute at LR resolution,
×2 PixelShuffle head — no cascade, no GAN.
```

---

## Slide 4 — Proposed Solution
*(No image)*

```
Pipeline: load .npy → group by shape → batch → bf16 forward (channels_last)
→ clip [0,1] → save

Architecture: NAFSR — NAFNet-style blocks (SimpleGate, SCA channel attention,
LayerNorm) + ×2 PixelShuffle head, width 64, 32 blocks, FiLM noise-level
conditioning + heteroscedastic uncertainty head, 1,393,938 parameters.

Loss: Charbonnier (fidelity) + SSIM (structure) + FFT (frequency) — balanced,
no adversarial term (no hallucination risk for inspection use).

Augmentation: dihedral flips/rotations, on-the-fly synthetic re-degradation
with randomised order and noise levels, plus procedural structural content
(gratings, contact-hole grids, checkerboards, circuit traces) mixed into
training to close a real-SEM generalisation gap.
```

---

## Slide 5 — Innovation & Uniqueness
*(No image)*

```
(a) Empirical degradation forensics driving a matched synthetic-pair
generator — kernel and noise parameters measured from the data, not assumed
from the spec.

(b) Balanced fidelity + structure + frequency loss with an explicit no-GAN /
no-hallucination decision — justified by inspection semantics: a plausible
but wrong structure is worse than a blurry correct one when a die's fate
depends on it. Confirmed directly: the model blurs rather than invents detail
on its hardest failure cases (spectral measurement, not a guess).

(c) Throughput-engineered inference path: grouped batching by shape, bf16 +
channels_last, threaded I/O, memory-aware OOM recovery (automatic batch
halving, CPU-bicubic floor) — every optimization measured, not assumed;
batch size itself was re-swept and changed after measurement showed a smaller
batch is faster end-to-end on this hardware.
```

---

## Slide 6 — Results

**Table:**

| Method | PSNR dB | SSIM | LPIPS | n |
|---|---|---|---|---|
| Bicubic ×2 | 23.65 | 0.548 | 0.412 | 400 |
| Median 3×3 → bicubic | 25.51 | 0.613 | 0.409 | 400 |
| Non-local means → bicubic | 26.27 | 0.652 | 0.426 | 400 |
| U-Net baseline | 28.88 | 0.783 | 0.265 | 400 |
| **Our model** | **29.59** | **0.795** | **0.254** | 400 |

**Images (place side by side under the table):**
- Typical/success case:
  `C:\Users\sahit\OneDrive\Desktop\semi\results\qualitative\val_001682_typical_near_mean_psnr31.81.png`
- Honest worst-case failure:
  `C:\Users\sahit\OneDrive\Desktop\semi\results\qualitative\val_002041_weak_worst_in_split_psnr17.91.png`

```
Caption under images: "Typical case (near-mean PSNR) | Honest failure: worst-PSNR
case in the split, shown deliberately — fine broadband texture that exceeds
what the low-resolution input can recover (confirmed by spectral measurement)."

Also wins on 45 real, licensed electron-microscopy images (LPIPS significant,
PSNR/SSIM ties, no regression).
```

---

## Slide 7 — Technology & Feasibility
*(No image)*

```
PyTorch 2.11.0+cu128, CUDA 12.8. Trained on cloud (HF Jobs A100-large GPU);
inference measured on an NVIDIA RTX 4060 Laptop GPU (8 GB), the dev machine.

Model: NAFSR, 1,393,938 params, checkpoint 11.0 MB.

Inference throughput (128→256, RTX 4060, bf16, batch 4 — the measured-fastest
setting): 19.40 images/second, median 20.62s for 400 images, including
process startup and CUDA init — externally timed, not just the forward pass.

No H100 number is reported unless explicitly labelled as a projection — KLA
scores on an H100; we trained on cloud A100 and measured throughput on an
RTX 4060, and never conflate the two.
```

---

## Slide 8 — GitHub & Video Link
*(No image)*

```
Repository: https://github.com/sahithsundarw/bunker_backer
(public, verified in a logged-out window)

Demo video: [link here once recorded]
```

---

## Slide 9 — References
*(No image)*

```
Kumar, T. et al. (2024). Image Data Augmentation Approaches: A Comprehensive
Survey and Future Directions. IEEE Access, 12.
Zhai, L. et al. (2023). A Comprehensive Review of Deep Learning-Based
Real-World Image Restoration. IEEE Access, 11, 21049-21067.
Terven, J. et al. (2025). A Comprehensive Survey of Loss Functions and
Metrics in Deep Learning. Artificial Intelligence Review, 58, 195.
Monga, V. et al. (2021). Algorithm Unrolling: Interpretable, Efficient Deep
Learning for Signal and Image Processing. IEEE SPM, 38(2), 18-44.
Chen, L. et al. (2022). Simple Baselines for Image Restoration (NAFNet). ECCV.
Yoo, J. et al. (2020). Rethinking Data Augmentation for Image Super-Resolution
(CutBlur). CVPR.
Wang, Z. et al. (2004). Image Quality Assessment: SSIM. IEEE TIP.
Zhang, R. et al. (2018). The Unreasonable Effectiveness of Deep Features as a
Perceptual Metric (LPIPS). CVPR.
Shi, W. et al. (2016). Real-Time Single Image and Video SR Using an Efficient
Sub-Pixel CNN (PixelShuffle). CVPR.
```

---

## Optional extra image (appendix / backup slide)

`C:\Users\sahit\OneDrive\Desktop\semi\results\eda\proxy_ood\proxy_ood_grid.png` — shows the
procedural OOD test content used for one of the generalisation checks.
