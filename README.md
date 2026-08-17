# KLA PS01 — AI-Based Restoration of Degraded Images for Semiconductor Inspection

SEMICON India Hackathon 2026 · Track 1 · Problem Statement PS01
Public repository: `https://github.com/sahithsundarw/semicon-kla-image-restoration`

Restores a degraded grayscale image — noisy and downsampled by exactly ×2 — to a clean
estimate at full resolution, in one blind pass, with the whole model running at low
resolution and a single PixelShuffle ×2 head.

---

> ## STATUS — 2026-08-17: ROUND 2 LONG-RUN CHECKPOINT SHIPS, 29.2548 dB
>
> `weights/best.pt` is present and tracked directly in the repo (Route A, `V51` exemption).
> It is the Round 2 differentiation cloud long run's checkpoint: NAFSR width=64, num_blocks=32,
> FiLM noise-level conditioning + heteroscedastic uncertainty head both enabled and trained
> (1,393,938 params), trained on an HF Jobs A100-large GPU for the full 129,700-iteration
> schedule, best-of-run selected at iteration 76,000. `inference.py --require_weights` reloads
> it with strict state-dict validation and prefers the EMA weights.
>
> On the committed 400-pair validation split, saved-output evaluation measures
> **29.2548 ± 4.6210 dB PSNR**, **0.79211 ± 0.14321 SSIM**, **0.25625 ± 0.14627 LPIPS**.
>
> **This checkpoint supersedes the prior from-scratch NAFSR w48n16 checkpoint (28.7865 dB),**
> re-scored head-to-head under one harness before promotion (paired per-image test, n=400):
> wins PSNR (+0.468 dB, t=-25.85 vs the prior checkpoint, 391/400 images better) and SSIM
> (+0.00925, t=-15.08, 378/400 better) significantly; LPIPS is a statistical tie (t=-1.14, not
> significant). **Also now beats the U-Net baseline on all three metrics** (PSNR +0.374 dB
> t=+18.25, SSIM +0.00938 t=+12.19, LPIPS -0.00900 t=-3.26, all 400/400-paired, all
> significant) — a reversal of the prior checkpoint's documented 1-win/1-loss/1-tie result.
> Full comparison: `docs/decisions.md` D61.
>
> **A real trade-off is disclosed, not hidden:** this checkpoint's generalisation to the
> real-SEM out-of-distribution set (D53) got measurably WORSE on SSIM (0.328 → 0.260) and
> LPIPS (0.569 → 0.711) versus the prior checkpoint, despite improving in-distribution and on
> the procedural proxy-OOD set. See D61 for the full numbers.
>
> - **Throughput is measured, on the dev machine, not on an H100.** `results/runtime_report.md`
>   records an externally-timed (`subprocess`-wrapped, not an internal timer) full run of the
>   same 400-image val-split input set on the **NVIDIA GeForce RTX 4060 Laptop GPU** (bf16,
>   batch 32): total wall-clock **median 23.19 s (17.3 img/s)**, n=5, spread 16.7%. This is
>   faster than the prior checkpoint's own recorded 8.3 img/s despite 3.6x more parameters — the
>   prior number's own record shows a 681% spread (high system noise), so the two are not a
>   clean apples-to-apples speed comparison; see `results/runtime_report.md` for the full,
>   honest caveat. No H100 number exists or is claimed.
> - **V22, V24**: see `docs/BLOCKERS.md` B11 and `docs/decisions.md` D42 — root-caused and
>   substantially mitigated (bf16/fp32 divergence fixed; cross-process determinism flake
>   reduced 50%→~20%, not fully eliminated).
> - **`results/restored_test_outputs/` regeneration against this checkpoint is in progress** —
>   see that folder's own README for current status; do not assume it already reflects this
>   checkpoint until its own status line says so.
>
> Live check status: `results/verification_report.json`. Ledger: `docs/STATE.md`.

---

## About the released data — read this first

**The problem domain is semiconductor inspection. The released dataset is grayscale natural
photographs.** Both statements are true and this repository holds them at once.

> The released dataset is 3200 training pairs and 400 test inputs of grayscale **natural
> photographs** — architecture, animals, foliage, landmarks — not semiconductor imagery. We
> treat it as a **proxy**: the *degradation* — ×2 decimation plus signal-dependent noise — is
> what transfers to inspection imagery, so we characterised the degradation empirically and
> optimised for degradation robustness rather than fitting content-specific priors.

Verified over 96 samples spanning both splits. Evidence: `results/eda/content_train_gt.png`
and `results/eda/content_test_inputs.png`. Full analysis: `docs/SPEC_ADDENDUM.md` (headline
finding, §7, §11) and `docs/decisions.md` D4.

We do **not** speculate about why the released data is natural imagery — deliberate proxy,
placeholder or packaging error are all consistent with what we measured. We report the
measurement, not a motive.

## Result summary

Held-out validation split of `train/`, **n = 400 pairs**, listed explicitly in
`configs/split_val.txt`. There is no `test_GT` — the released test set ships inputs only — so
no score can be computed locally against the official test set. Scores are computed on the
**reloaded on-disk `.npy` artifacts**, not on in-memory tensors, so any dtype loss is included
(SPEC §10, V30).

| Method | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | End-to-end throughput |
|---|---|---|---|---|
| Bicubic ×2 (raw NoisyLR, the floor) | 23.6524 ± 3.0236 | 0.54775 ± 0.19197 | 0.41206 ± 0.15407 | not separately measured (classical baseline, not run through `inference.py`) |
| Median 3×3 → bicubic ×2 | 25.5057 ± 3.8785 | 0.61317 ± 0.17232 | 0.40870 ± 0.15866 | not separately measured (classical baseline, not run through `inference.py`) |
| Non-local means → bicubic ×2 | 26.2722 ± 4.3037 | 0.65152 ± 0.19523 | 0.42586 ± 0.18627 | not separately measured (classical baseline, not run through `inference.py`) |
| U-Net baseline (UNetSR w32 L4, 2,970,401 params) | **28.8808 ± 4.5328** | 0.78273 ± 0.14245 | 0.26525 ± 0.14878 | not separately measured (`results/runtime_report.md` covers NAFSR only) |
| Prior shipped checkpoint (superseded 2026-08-17): NAFSR w48n16, from scratch, 388,225 params | 28.7865 ± 4.5329 | 0.78287 ± 0.14169 | 0.25324 ± 0.13193 | 8.3 img/s @N=400, RTX 4060, bf16, batch 32 (high-variance measurement, 681% spread — see `results/runtime_report.md`) |
| **Shipped — NAFSR ×2 w64 n32, EMA, FiLM noise-conditioning + uncertainty head, cloud long run (1,393,938 params)** | **29.2548 ± 4.6210** | **0.79211 ± 0.14321** | **0.25625 ± 0.14627** | **17.3 img/s** end-to-end at N=400, RTX 4060 Laptop GPU, bf16, batch 32 — dev-machine measurement, not an H100 number (`results/runtime_report.md`) |

Values are `mean ± population standard deviation`. Source: `results/metrics_summary.md`,
machine-generated by `scripts/evaluate.py`; per-image records live in
`results/baselines/*/metrics.json`, which **are** committed (`.gitignore` negates the blanket
`results/*` rule for this specific file, `!results/baselines/*/metrics.json`) — the 2000 `.npy`
prediction files in those same directories are not, since they would blow the tree-size cap.
The shipped row is the Round 2 differentiation cloud long run's checkpoint, promoted after a
paired re-score against the prior checkpoint under one harness (`docs/decisions.md` D61); the
prior row is kept for the historical comparison, not deleted.

### Against the classical baselines: wins all three metrics

Against the bicubic floor: **+5.60 dB PSNR**, +0.244 SSIM, −0.156 LPIPS. Against the strongest
classical baseline, non-local means: **+2.98 dB PSNR**, +0.141 SSIM, −0.170 LPIPS. Both
learned models clear all three classical baselines on all three metrics.

### Against the learned baseline: wins all three metrics (V28 passes)

**The shipped model now beats the U-Net baseline on all three metrics**, a reversal from the
prior checkpoint's documented 1-win/1-loss/1-tie negative result (kept below for the record).
Paired per-image difference, the same 400 images, same statistic `check_V28` uses:

| Metric | Paired mean difference (shipped − U-Net) | t | shipped better on | Verdict |
|---|---|---|---|---|
| PSNR | **+0.3740 dB** | +18.25 | 374 / 400 | **win** |
| SSIM | **+0.00938** | +12.19 | 382 / 400 | **win** |
| LPIPS | **−0.00900** | −3.26 | 239 / 400 | **win** |

V28 requires winning at least 2 of 3; winning all 3 clears it comfortably (`docs/decisions.md`
D61). The prior checkpoint's negative result (kept for the record, not deleted): PSNR −0.0943
dB (loss), SSIM +0.0001 (tie), LPIPS −0.0120 (win) — a 0.388M-parameter model matching a
2.97M-parameter one on fidelity while winning perceptual quality with 7.7x fewer parameters.
That trade-off no longer applies to the current shipped checkpoint, which now wins outright.

### Why the LPIPS column matters more than it looks

Across the classical baselines, fidelity and perceptual quality pull in *opposite* directions:
non-local means wins PSNR by 2.6 dB over bicubic yet scores the **worst** LPIPS of the three,
because it buys fidelity by over-smoothing. A restoration model that improved PSNR while
degrading LPIPS would be gaming one half of an undisclosed scoring blend at the other half's
expense. The shipped model improves both simultaneously against every classical baseline, and
now against the learned baseline too.

### Two numbers, do not confuse them

The training log reports `psnr 30.3944` at iteration 20000. That is a **100-image subset** used
for in-run checkpoint selection only. The reportable figure is the **full 400-image committed
split: 28.7865 dB**, produced by `scripts/evaluate.py` from `.npy` files reloaded from disk.
The lower number is the honest one and is the one quoted everywhere in this repository.

### Caveats stated plainly

All of these numbers come from a held-out slice of `train/`, never from the official test set,
for which no ground truth exists. Throughput is measured only for the shipped NAFSR model, only
on the RTX 4060 Laptop GPU dev machine (`results/runtime_report.md`) — not on an H100, and not
separately for the classical baselines or the U-Net. The imagery is natural
photographs used as a proxy, so these numbers measure degradation robustness, not any
inspection-domain content prior. The alternate (non-shipped) checkpoint's own closed-form-LS5
training history is preserved in `docs/STATE.md`'s teammate-session archive and
`docs/decisions.md` D48, not repeated here.

Metric implementations are pinned, because library defaults differ by non-trivial margins:

- PSNR — `skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=1.0)`
- SSIM — `skimage.metrics.structural_similarity(gt, pred, data_range=1.0,
  gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` (Wang et al. 2004 settings)
- LPIPS — `lpips.LPIPS(net='alex')`, grayscale replicated to 3 channels, rescaled to [-1,1]

The bicubic floor is low by natural-image super-resolution standards because the input is
genuinely noisy. That is expected, not a bug.

## Environment

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 (`torch.version.cuda == '12.8'`) |
| torchvision | 0.26.0+cu128 |
| GPU (closed-form LS-5 stage, superseded checkpoint) | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Scoring GPU | NVIDIA H100 (KLA's) — **no H100 number appears in this repo; any future H100 figure will be labelled a projection, not a measurement** |

**Current checkpoint's training environment differs from the table above.** `r2_nb8_psnrloss`
(the current `weights/best.pt`) was trained with `scripts/train_residual.py --device mps` on
Apple Silicon, not the RTX 4060 — see the Training section below and `docs/decisions.md`
D28/D29. Final-test inference (Phase 3/5 of this branch's integration) was likewise measured on
Apple Silicon **CPU**, not GPU; see `results/runtime_report.md` for that number labelled
correctly as a local measurement.

Clone the repository:

```bash
git clone https://github.com/sahithsundarw/semicon-kla-image-restoration.git
cd semicon-kla-image-restoration
```

Create the environment and install the pinned dependencies:

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

*(The commands above are the Windows form, which is what was executed to verify them. On
Linux/macOS the equivalents are `python3.12 -m venv .venv` and `.venv/bin/python`; nothing
else changes. Activating the venv and calling plain `python` works identically — the explicit
interpreter path is used here only so the commands are copy-pasteable with no shell state.)*

Confirm you got a CUDA build — this matters, see the note below:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

On the reference machine this prints `2.11.0+cu128 12.8 True`. On a machine with no NVIDIA
GPU the same wheels install and the last field is `False`; `inference.py` then runs on CPU
without any change.

Fresh-clone compatibility was also demonstrated in a Linux `python:3.12-slim` container with
the pinned CUDA 12.8 PyTorch packages: the clean-install and fresh-clone verifier checks V04
and V46 passed. That is a target-platform compatibility check, **not** a Linux/CUDA or H100
runtime measurement.

> **Why `requirements.txt` starts with `--extra-index-url`.** `torch==2.11.0+cu128` is
> published only on `download.pytorch.org`, never on PyPI. Installing `lpips` without that
> directive silently resolved torch **from PyPI** and replaced the CUDA build with a CPU-only
> wheel on this machine — nothing failed, nothing was logged, and `torch.cuda.is_available()`
> quietly became `False`. The `+cu128` local version in the pin is the safety catch: PyPI
> cannot host a local version, so there is no CPU candidate to fall back to and a broken
> install fails loudly instead of silently costing the entire throughput score. See
> `docs/decisions.md` D18 and `docs/BLOCKERS.md` B8.

## Inference — the command KLA runs

```bash
.venv/Scripts/python.exe inference.py --input_dir sample_inputs --output_dir results/sample_outputs
```

`sample_inputs/` holds 6 real degraded 128×128 inputs (394 KB total) so a reviewer can verify
the script without downloading the dataset. Point `--input_dir` at any directory of degraded
`.npy` files and `--output_dir` anywhere you like — those two arguments are the entire
interface (SPEC F11). No other argument is required, no file needs editing, and weights are
resolved relative to the script file, never to the working directory, so the script runs
correctly from any CWD. The verified checkpoint is committed at `weights/best.pt`; a fresh
clone needs no checkpoint download or manual placement.

Normal submission inference requires that checkpoint. If it is missing or cannot be loaded,
the two-argument command prints an explicit error and exits nonzero without writing bicubic
outputs as though they were model predictions. `--require_weights` is retained as an explicit
assertion and has the same strict behavior.

For baseline demonstrations only, `--allow_bicubic_fallback` explicitly opts into a
parameter-free bicubic result. Demo fallback output is never submission output and must not be
reported as model output. `--require_weights` overrides the demo flag when both are supplied.

Optional flags, all with defaults that make the two-argument invocation correct:
`--weights`, `--batch_size`, `--device`, `--precision {auto,bf16,fp16,fp32}`, `--compile`,
`--tta`, `--num_workers`, `--write_threads`, `--allow_bicubic_fallback`, `--require_weights`,
`--verbose`.
`--compile` and `--tta` are **off by default**: at 400 images, 30–120 s of compilation never
amortises, and an 8× self-ensemble cannot be justified against a fixed cost it does not
amortise (`docs/decisions.md` D7).

## Input / output contract

Derived from the real files and final: `docs/io_contract.md`.

**Input**
- `.npy` NumPy binary, read with `np.load(path, allow_pickle=False)`
- `float32`, 2-D `(H, W)`, grayscale, **no channel axis**
- values **may lie outside [0,1]** — observed range `[-0.28, 2.16]`, with ~3% of pixels above
  1.0 and ~0.5% below 0.0
- **inputs are never clipped.** The out-of-range values are intentional and carry information
  (SPEC F5); clipping them destroys it

**Output**
- `.npy`, `float32`, written with `np.save`
- exactly **2× the input** in both axes — `(2H, 2W)`
- **clipped to [0,1]**, and **no renormalisation**: per-image min–max renormalisation was
  measured at **−4.66 dB PSNR** and is permanently rejected (`docs/decisions.md` D3)
- filename **byte-identical** to the input filename, same extension, no `_restored` suffix
- subdirectory structure mirrored from input to output; one output per input

Demonstrate the contract on the sample data:

```bash
.venv/Scripts/python.exe -c "import numpy as np; a=np.load('sample_inputs/000000.npy'); b=np.load('results/sample_outputs/000000.npy'); print('in ', a.shape, a.dtype, round(float(a.min()),4), round(float(a.max()),4)); print('out', b.shape, b.dtype, round(float(b.min()),4), round(float(b.max()),4))"
```

which prints, for that file:

```
in  (128, 128) float32 0.001 1.5406
out (256, 256) float32 0.0 1.0
```

— input above 1.0 and untouched, output exactly doubled and clipped.

**No image library is used anywhere in the inference path.** The data is `.npy` end to end, so
`cv2`, `tifffile` and `PIL` are absent by design, not by oversight: they are dead weight on a
timed run, and several `cv2` paths silently convert to 8-bit or clip to [0,1], which would
corrupt inputs that legitimately reach 2.16. `inference.py`'s module-level imports are exactly
`argparse os sys time pathlib concurrent.futures numpy torch`, enforced statically by a
submission-blocking check.

## Training

The first CPU-feasible training run completed with:

```text
python train.py --config configs/final.yaml --data_root <dataset_root> --seed 42 \
  --closed_form_linear --out weights/best.pt
```

This fits a ridge-regularised 5x5 linear restoration kernel on the 2,800 training pairs only,
then embeds the learned residual into the configured NAFSR stem and PixelShuffle head. It is
not the projected 20,000-iteration gradient run: the checkpoint records
`training_mode=closed_form_linear_ls5` and `total_iters_run=0`. Measured on CPU, the complete
run took 24.9 s (4.43 s fitting) and produced a 3,288,933-byte checkpoint with SHA256
`d5807dabad37b251f25d066838da9e3f73c164ec37ee777505a80e23cad9e90d`.

Useful flags for checking the pipeline before committing GPU hours: `--smoke` (a handful of
steps, used by the verifier), `--overfit N` (deliberately overfit N pairs — the pipeline
sanity gate; if it cannot reach a high PSNR on 2 pairs, alignment, normalisation or the loss
is broken and nothing downstream is trustworthy), `--iters`, `--seed`, `--tag`. Every run
appends a row to `results/experiments.csv` with the git SHA, config, seed, metrics and
wall-clock. That experiment ledger is included and tracked in this repository.

The dataset lives outside the repository and is never committed. Dataset-dependent training
and verifier checks require `KLA_DATA_ROOT` to point at the dataset root; on the measured Mac,
that root is `/Users/shanmukhsai/Downloads`, which the checks also recognize directly. The
root must contain `train/GT` and `train/NoisyLR`; its final-test `NoisyLR` directory has no GT.
`configs/split_val.txt` is the committed, explicit validation file list and is never
regenerated randomly at run time, which would leak.

The gradient-training path remains available for a future accelerator run. Its projected
**73.7 min for 20k iterations** at batch 32 / 64 px patches on an RTX 4060
(`docs/decisions.md` D20) remains an extrapolation, not a completed run.

### Alternate checkpoint (not shipped): closed-form LS-5 + residual refinement

A teammate independently developed a second checkpoint on a separate line of work (Mac/MPS),
reconciled into this repo on 2026-08-16 (`docs/decisions.md` D48/D49). It freezes a closed-form
5×5 least-squares restoration fit into a NAFSR's `stem`/`head.expand`/`head.project` weights,
then trains a fresh, shallower NAFSR body as an additive residual correction on top
(`scripts/train_residual.py`):

```bash
python scripts/train_residual.py \
  --config configs/phase4_psnr_focus.yaml \
  --base_checkpoint weights/best.pt \
  --out results/residual_experiments/r2_nb8_psnrloss/model.pt \
  --num_blocks 8 \
  --iters 4000 \
  --batch_size 32 \
  --lr 2e-4 \
  --device mps \
  --seed 42
```

Self-reported (disk-verified, full 400-pair split, V30 round-trip): 28.0394 ± 4.1881 dB PSNR,
0.74804 ± 0.15275 SSIM, 0.29571 ± 0.16672 LPIPS. **Independently re-scored this session on this
session's RTX 4060, same split, same evaluate.py: reproduced exactly** (see the Result summary
table above). Not the shipped checkpoint — see D49 for the head-to-head comparison and why.

Every training run, on either line, appends a row with git SHA, config, seed, metrics and
wall-clock to `results/experiments.csv`. That file **is** in a clone — `.gitignore` negates the
blanket `results/*` rule for it specifically (`!results/experiments.csv`).

## Repository map

| Path | Contents |
|---|---|
| `inference.py` | the evaluation script KLA runs; standalone, two required arguments |
| `train.py` | reproduces the checkpoint, including the CPU-feasible closed-form training mode |
| `requirements.txt` | complete `pip freeze`, every line `==` pinned |
| `sample_inputs/` | 6 real degraded inputs so inference can be verified without the dataset |
| `src/` | `model.py` `blocks.py` `dataset.py` `degrade.py` `losses.py` `metrics.py` `io_utils.py` `utils.py` |
| `configs/` | `nafnet_x2.yaml`, `baseline_unet.yaml`, `final.yaml`, `split_val.txt` |
| `scripts/` | dataset forensics, degradation fitting, baselines, evaluation, benchmarking, `verify_all.py` |
| `docs/` | SPEC, SPEC addendum (governs on conflict), verification contract, dataset findings, I/O contract, decisions, blockers, state |
| `results/eda/` | dataset figures, degradation fit, content contact sheets |
| `results/metrics_summary.md` | machine-generated results table |
| `results/restored_test_outputs/` | mandatory model outputs, generated from the current `weights/best.pt` (28.0394 dB). **Holds a manifest and hashes, not the output bytes** — the 400-file archive (91,051,052 B, sha256 `a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98`) is prepared for a GitHub Release asset (`docs/decisions.md` D23); publication is the remaining manual step, see that folder's `README.md` |
| `results/runtime_report.md` | local Mac CPU external-process runtime for the current checkpoint (400 images, 71.72 s, 5.6 img/s) — explicitly not an H100/CUDA number; the separate 56.73 s release-generation log is labeled there |
| `weights/` | tracked `best.pt` checkpoint (SHA256 `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7`) + provenance notes |

## Method summary

1. **Degradation forensics first.** The GT→LR downsample was recovered by least squares over
   3,125,000 equations: a 4×4 **sharpening** kernel with centre weights ≈0.320 and negative
   surround lobes ≈−0.045 — provably not a box filter, which has no negative lobes.
   `bicubic(antialias=False)` sits within **1.22e−05** residual std of that optimum and is
   used as the working model.
2. **Noise is applied after decimation**, from residual autocorrelation (≈0 or slightly
   negative at every tested lag; pre-decimation noise would be strongly positive).
3. **There is no additive Gaussian floor.** The σ²+v·x² form the brief suggests overshoots the
   darkest intensity bin by 12.5×. A three-parameter fit drives σ to exactly 0 and splits the
   variance into a shot/Poisson term and a speckle term: **σ = 0, a = 0.011253, v = 0.015745**.
   The simulator implements that, not a pure speckle model.
4. **Architecture — NAFSR:** a flat NAFNet-style body at LR resolution, a PixelShuffle ×2
   head, and a global bilinear-upsample skip so the network only learns residual detail. All
   heavy compute stays at LR, where it is 4× cheaper. **The currently-shipped checkpoint
   (`weights/best.pt`) is 1,393,938 parameters (width=64, num_blocks=32), 5.584 GMAC per
   128×128 image scales accordingly with width/depth (see `results/eda/pareto_frontier.png`
   for the measured params-vs-quality frontier across six swept configs).** Fully
   convolutional with a required size multiple of 1, verified on 128→256, 256→512,
   61×97→122×194 and 1×1→2×2.
5. **Learned baseline — UNetSR:** 2,970,401 parameters, 4.478 GMAC. The shipped checkpoint is
   now at 0.47× the baseline's parameter count and beats it on all three metrics (V28).
6. **FiLM noise-level conditioning + heteroscedastic uncertainty — implemented, validated, and
   now part of the shipped checkpoint (`docs/decisions.md` D52).** A small `NoiseEstimator`
   (conv stack + global pool) embeds the input and conditions every `NAFBlock` via
   zero-initialised FiLM (scale, shift), so a freshly-constructed FiLM-enabled model is
   bit-identical to an un-conditioned one until trained — no behaviour change at init. An
   optional heteroscedastic head predicts a per-pixel log-variance
   (`return_uncertainty=True`), trained with a Gaussian NLL term that costs nothing when the
   flag is off. Both default to `film_dim=0`/`uncertainty=False` — a NAFSR built with no extra
   arguments is unchanged. **Calibration measured, not just presence-checked:** the
   uncertainty head correlates strongly with real error (per-image Pearson r=0.965,
   Spearman r=0.941, D59); the FiLM embedding's relationship to true noise level is present but
   diffuse — no single dimension or the embedding norm correlates well alone, but a 16-dim
   linear probe explains ~23% of true-noise-level variance held out (D58). An
   architecture/hyperparameter sweep with both enabled was run on cloud A100 hardware to select
   this config — see `docs/decisions.md` D52/D55/D61 and `results/eda/pareto_frontier.png`.
7. **Algorithm-unrolling hybrid — `UnrolledSR` (`src/unrolling.py`), stretch goal, NOT
   shipped.** Unrolls T proximal-gradient steps against the *measured* x2 degradation kernel
   (`RECOVERED_KERNEL_4X4`, item 1 above) as a fixed, non-trainable forward operator, with a
   small weight-tied `NAFBlock` denoiser as the learned proximal operator per step (Monga et
   al. 2021, IEEE SPM — the exact survey cited in KLA's own reference list). Stated honestly:
   its overfit sanity gate currently fails (plateaus well below the 40 dB bar that
   NAFSR/UNetSR both clear on the same fixture) and is under active root-cause debugging; it
   is not part of any shipped result and this README will be updated if/when it is.
8. **Loss** is balanced because the scoring blend is undisclosed: Charbonnier + (1−MS-SSIM) +
   an FFT-magnitude term, with LPIPS available but **off by default**. **No adversarial
   loss** — hallucinating a structure that is not there is the worst possible failure in an
   inspection context.
9. **Throughput** is measured over the whole process, not inferred from kernel timings. The
   current 400-image external run measures 71.72 s total: 70.34 s in the main pipeline and
   1.38 s in process startup/import overhead. Import hygiene still reduces fixed cost, while
   batching and the LR-resolution body address the dominant pipeline work. NAFSR profiles as
   **memory-bandwidth bound, not compute bound** (32.8% layer-norm, 17.9% conv bias-add,
   16.2% convolution), which is why `channels_last` and bf16 each move it by under 20%.

## Assumptions

- The downsample kernel is modelled as `bicubic(antialias=False)`; the true support extends
  slightly beyond 4×4 (a K=6 recovery finds max |weight| 0.01355 in the outermost ring), worth
  ~1e−05 in residual std, so the model is close but not exact.
- Noise is applied after downsampling and is signal-dependent with no additive floor. An
  additive Gaussian term is nevertheless retained in augmentation, randomised over `U(0, 0.02)`
  **including zero**, as a cheap hedge because the brief names additive Gaussian as a
  degradation and warns that test noise levels may vary.
- GT is per-image min–max normalised to exactly [0,1] (all 3200 files attain both endpoints).
  This licenses clipping the output and nothing more.
- Train and test are treated as the same domain: measured spectral peakiness and gradient
  anisotropy differ by only ×1.02–1.04.
- The hidden test set may contain genuine semiconductor imagery. We assume the **degradation**
  transfers and the **content prior** does not, and we optimise accordingly
  (`docs/decisions.md` D16).

## External resources & licences

**No external datasets or pretrained weights are used in the shipped model.** It is trained
from scratch on the provided image pairs. Two external resources are used for *evaluation
only* and never for training: LPIPS (Zhang et al., CVPR 2018) with its standard AlexNet
backbone, and a real-SEM image set used for an OOD robustness report (`docs/decisions.md`
D53). Neither contributes a gradient to the shipped checkpoint and neither is required to run
`inference.py`.

| Resource | Role | Link | Licence (verified at source) | Paper / model card |
|---|---|---|---|---|
| **LPIPS** (`lpips` 0.1.4) — linear calibration weights, shipped inside the pip package (`lpips/weights/v0.1/alex.pth`, 6,009 B) | Evaluation metric only | `https://github.com/richzhang/PerceptualSimilarity` | **BSD-2-Clause** — read from `LICENSE` at that repository (HTTP 200, fetched 2026-08-15); PyPI metadata agrees (`License :: OSI Approved :: BSD License`) | Zhang, Isola, Efros, Shechtman, Wang, *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018 |
| **AlexNet ImageNet-pretrained backbone**, pulled by LPIPS via `torchvision.models.alexnet(pretrained=True)` → `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`, **244,408,911 B measured** | Evaluation metric only — the feature extractor inside LPIPS | `https://github.com/pytorch/vision` | **BSD-3-Clause** — read from `LICENSE` at that repository (HTTP 200, fetched 2026-08-15) | Krizhevsky, Sutskever, Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS 2012; distributed via the torchvision model zoo |
| **Real-SEM OOD set** — 45 images (of 405 shipped; one per unique tile), SEM of Ni-WC metal matrix composites | Evaluation only — an out-of-distribution robustness report (`docs/decisions.md` D53), zero training/fitting | `https://zenodo.org/records/17315241` | **CC-BY 4.0** — confirmed via the Zenodo API's `metadata.license.id` field, not inferred | *Scanning Electron Microscopy (SEM) Dataset of Additively Manufactured Ni-WC Metal Matrix Composites for Semantic Segmentation*, Zenodo record 17315241 |
| External training datasets (DIV2K, Flickr2K, BSD, …) | **None used** | — | — | — |
| Pretrained super-resolution or restoration checkpoints (SwinIR, EDSR, NAFNet, …) | **None used** | — | — | — |

Rationale for training from scratch, with the alternatives costed, is in `docs/decisions.md`
D9 and D13: every public ×2 SR checkpoint is trained on clean bicubic downsampling with no
noise, so its prior points the wrong way for this degradation.

> **The condition under which the statement above stops being true.** `src/losses.py` contains
> an optional LPIPS loss term. It is **off by default** (`use_lpips=False`) and gated to the
> last 50% of training, and it is off in every shipped config. If any future iteration trains
> the shipped checkpoint with `use_lpips=True`, then ImageNet-pretrained AlexNet features
> **will** have contributed gradient to the shipped weights, the "evaluation only" framing
> above becomes false, and this section must be rewritten before submission. Stated plainly so
> the change cannot happen silently.

No confidential, unlicensed or access-restricted data is used. The provided dataset is not
redistributed in this repository — only 6 degraded input files under `sample_inputs/`, and no
ground truth of any kind.

## Runtime measurement

**Throughput is measured, on the dev machine, not on an H100.** `results/runtime_report.md`
records it: at the full 400-image test set, batch 32, precision bf16, on the **NVIDIA GeForce
RTX 4060 Laptop GPU**, total wall-clock **median 48269.4 ms (8.3 img/s)**, fitted as a fixed
startup cost of **14755 ms** plus **86.55 ms/image** marginal compute, with the fixed cost at
**30.6%** of total wall-clock at N=400. No H100 number exists or is claimed anywhere in this
repository.

The one timing on record is not that measurement: producing the 400 published test outputs took
**20.09 s (19.9 img/s)** on the RTX 4060 Laptop GPU at `device=cuda precision=bf16 batch=32`
(`results/restored_test_outputs/manifest.json`). That figure comes from `inference.py`'s own
timer, which starts at the top of `main()` — **it therefore excludes interpreter startup and
`import torch`**, which is precisely the cost that dominates a 400-image run (~85–95% of
end-to-end wall-clock, `docs/decisions.md` D7). It is a lower bound on the true cost, not a
score.

Two commitments about how the measurement above was taken, both honoured:

- timing was taken **externally around the whole process**
  (`subprocess.run([sys.executable, "inference.py", ...])`), not by an internal timer around the
  forward pass;
- every number is labelled with the device it was measured on. Training and timing happen
  on an RTX 4060 Laptop GPU; **no H100 measurement exists and none is presented as one.**

The alternate (non-shipped) checkpoint has its own runtime record: 400 images in 71.72 s
(5.6 img/s), batch 32, fp32, **local Mac CPU** external-process benchmark — not comparable to
the RTX 4060 numbers above (different device, different checkpoint). `scripts/benchmark_runtime.py`
is the shared external-process harness both lines use.

## Verification

Correctness for this project is defined by `docs/VERIFICATION_CONTRACT.md` (immutable) and
executed by `scripts/verify_all.py`, which defines 63 checks and writes
`results/verification_report.json`. Run it with
`.venv/Scripts/python.exe scripts/verify_all.py --strict`, or add `--fresh-clone` for the
clean-room checks. It is **not** listed as a fenced command here because it exits non-zero
while the project is incomplete, and no command in this README exits non-zero.

**The suite is not green, and this README does not claim it is.** Targeted checks measured on
2026-08-16 pass checkpoint presence/loading and tracked-checkout checks (V06, V35, V59), the
two-argument inference path, runtime-document checks, and validation isolation where the
dataset was supplied through `KLA_DATA_ROOT=/Users/shanmukhsai/Downloads`. Linux/CUDA
fresh-clone checks V04/V46 passed in the container noted above; they are compatibility results,
not runtime results. A historical U-Net comparison exists for the later hosted 20k model, but
no direct per-image U-Net comparison is claimed for the tracked `r2_nb8_psnrloss` checkpoint.
Publication of the final-output archive remains open. The authoritative per-check status is
`results/verification_report.json`, regenerated on every run; `docs/STATE.md` carries the
rolling ledger and `docs/BLOCKERS.md` the remaining work.
