# KLA PS01 — AI-Based Restoration of Degraded Images

SEMICON India Hackathon 2026, Track 1 · Problem Statement PS01

> ## ⚠ SCAFFOLD — NOT YET SUBMITTABLE
>
> This README is a **bootstrap stub**. The model has not been built or trained, and
> `inference.py` currently raises `NotImplementedError`. Every number below is marked
> _pending_ rather than filled with a placeholder, because a placeholder that looks like a
> result is worse than an obvious gap (V48 requires the table to match a real run).
>
> Owner: `docs-scribe`. Current status: `docs/STATE.md`.

## About the data

The problem domain is **semiconductor inspection**. The **released dataset is grayscale
natural photographs** — architecture, animals, foliage, landmarks — not semiconductor
imagery. We treat it as a **proxy**: the degradation (×2 decimation plus signal-dependent
noise) is what transfers to inspection imagery, so we characterised the degradation
empirically and optimise for degradation robustness rather than content-specific priors.

Evidence: `results/eda/content_train_gt.png`, `results/eda/content_test_inputs.png`.
Full analysis: `docs/SPEC_ADDENDUM.md` (headline finding, §7, §11).

## Result summary

| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | End-to-end (img/s) |
|---|---|---|---|---|
| Bicubic ×2 (no denoise) | **23.4247 ± 2.8319** | **0.54284 ± 0.20225** | _pending_ | _pending_ |
| Classical denoise + bicubic | _pending_ | _pending_ | _pending_ | _pending_ |
| U-Net baseline | _pending_ | _pending_ | _pending_ | _pending_ |
| **Ours (NAFSR-x2)** | _pending_ | _pending_ | _pending_ | _pending_ |

The bicubic row is measured on 200 held-out training pairs with clip-to-[0,1]
(`docs/decisions.md` D3). It is low by natural-image SR standards because the input is
genuinely noisy — expected, not a bug.

## Environment

- Windows 11 dev machine; NVIDIA GeForce RTX 4060 Laptop (8 GB), driver 610.47
- Python 3.12.10 (`py -3.12`); PyTorch **not yet installed**
- KLA scores on an H100 — local timings are labelled as local and never presented as H100 numbers

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Inference (the command KLA will run)

```bash
python inference.py --input_dir <degraded_images_dir> --output_dir <restored_output_dir>
```

Weights load automatically from `weights/best.pt`, resolved relative to the script. No edits
required. **Not yet functional** — see the scaffold notice above.

## Input / output contract

Derived from the real files; full detail in `docs/io_contract.md` (FINAL).

- **Input:** `.npy`, `float32`, 2-D `(H, W)`, grayscale, values **may lie outside [0,1]**
  (observed `[-0.28, 2.16]`). Inputs are **not** clipped — out-of-range values are
  intentional and carry information (SPEC F5).
- **Output:** `.npy`, `float32`, exactly **2× the input** in both axes, clipped to `[0,1]`,
  **no renormalisation**, filename **byte-identical** to the input, subdirectory structure
  mirrored.

No image library is used anywhere in the inference path — the data is `.npy` end to end.
That is deliberate: an image library is dead weight on a timed run and several `cv2` paths
silently convert to 8-bit or clip to [0,1].

## Training

```bash
python train.py --config configs/nafnet_x2.yaml --data_root <dataset_root>
```

Seed 42. Dataset lives outside the repo (`C:\kla-data`) and is never committed.
**Not yet functional.**

## Verification

```bash
python scripts/verify_all.py
```

Runs all 53 checks from `docs/VERIFICATION_CONTRACT.md` (V00 + V01–V52) and writes
`results/verification_report.json`. Near-total failure is the correct state at iteration 0.

## Repository map

| Path | Contents |
|---|---|
| `inference.py` | the evaluation script KLA runs |
| `train.py` | reproduces the submitted checkpoint |
| `src/` | model, blocks, dataset, degradation, losses, metrics, IO helpers |
| `scripts/` | dataset forensics, degradation fitting, verifier, evaluation, benchmarking |
| `configs/` | training configs and the committed validation split |
| `docs/` | SPEC, addendum (governs on conflict), verification contract, findings, decisions |
| `results/eda/` | dataset figures and the degradation fit |
| `weights/` | checkpoint (pending) |

## Method summary

_Pending — written once the model exists._ Degradation analysis is complete and recorded in
`docs/decisions.md` D1/D2: the downsample is a sharpening kernel (not a box), noise is applied
after decimation, and the noise is signal-dependent with **no additive Gaussian floor**
(three-parameter fit: σ=0, a=0.011253, v=0.015745).

## Assumptions

- Downsample kernel modelled as `bicubic(antialias=False)`, within 1.22e−05 residual std of
  the least-squares optimum recovered over 3.125 M equations.
- Noise applied after downsampling, per residual autocorrelation.
- An additive Gaussian term is retained in augmentation, randomised over `U(0, 0.02)`
  including zero, as a hedge for SPEC F3 even though it measures to zero.

## External resources & licences

**No external datasets or pretrained weights used.**

Phase 1 trains from scratch. Rationale in `docs/decisions.md` D9/D13: every classical ×2 SR
checkpoint assumes clean bicubic downsampling with no noise, so the pretrained prior points
the wrong way for this degradation.

## Runtime measurement

_Pending._ Will report hardware, batch size, precision, timing method (external, around the
whole process), image count, total seconds and images/second — with the device labelled.
Startup cost is ~85–95% of end-to-end wall-clock at this scale (`docs/decisions.md` D7).
