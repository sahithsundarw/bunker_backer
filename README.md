# KLA PS01 — AI-Based Restoration of Degraded Images for Semiconductor Inspection

SEMICON India Hackathon 2026 · Track 1 · Problem Statement PS01
Public repository: `https://github.com/sahithsundarw/bunker_backer`

Restores a degraded grayscale image — noisy and downsampled by exactly ×2 — to a clean
estimate at full resolution, in one blind pass, with the whole model running at low
resolution and a single PixelShuffle ×2 head.

---

## Status

**Shipped checkpoint: NAFSR, 1.39M parameters, FiLM noise-conditioning + uncertainty head.**
`weights/best.pt` is tracked directly in this repository — no download step, no manual
placement. On the committed 400-pair held-out validation split:

| Metric | Value |
|---|---|
| PSNR | **29.5850 ± 4.6301 dB** |
| SSIM | **0.79460 ± 0.14204** |
| LPIPS | **0.25416 ± 0.13263** |
| Throughput (RTX 4060 Laptop GPU, end to end) | **19.40 img/s** (20.62 s median for 400 images) |

This checkpoint is a fine-tune (never trained from scratch) of an earlier long-run checkpoint,
resumed on a cloud A100 GPU with a mix of procedural structural training content added to
close a generalisation gap identified during testing. Compared head-to-head against the
checkpoint it replaced, on a full paired per-image test: it **wins or ties on every metric
across every evaluation set, with no regression anywhere**, including the first genuine
improvement on real out-of-distribution imagery. Full statistical detail is in
`docs/decisions.md`.

It also beats a from-scratch U-Net baseline on all three metrics (PSNR, SSIM, and LPIPS),
each result statistically significant on a paired 400-image test.

### How this checkpoint was built

An earlier version of this model performed noticeably worse on real, out-of-distribution
imagery than on the training distribution. Rather than retrain blindly, the cause was
diagnosed first: a first attempt (widening the range of simulated noise during training)
neither fixed the gap nor helped elsewhere, but was itself informative — it showed the problem
wasn't about noise coverage. Direct measurement of the real out-of-distribution images then
showed they differ from the training images on several concrete visual statistics (edge
density, local contrast, spectral content), pointing at a genuine content gap rather than a
noise-modelling one. Blending the two models' weights together at every ratio never recovered
the lost performance either, ruling out a "just needed a bit less of it" explanation.

The fix that worked was mixing procedurally generated structural content — gratings,
contact-hole-style grids, checkerboards, circuit-trace-like patterns — into training, alongside
the real training photographs. This is exactly the fix the diagnosis pointed at, and it
resolved the gap with no regression anywhere else. Every step of this investigation, including
the two approaches that did **not** work, is recorded in `docs/decisions.md` rather than
omitted.

### Generalisation testing

Two out-of-distribution test sets were used, honestly separated by how much they actually
prove:

- **Real electron-microscopy imagery** (45 genuine images, licensed under CC-BY 4.0, evaluation
  only) is the most meaningful evidence here: the model wins significantly on perceptual
  quality (LPIPS) with no regression on the other metrics. Because this result mattered, it was
  independently re-checked at roughly 4× the statistical power (180 non-overlapping crops of
  the same real images, scored two ways to correctly account for the fact that crops from the
  same source image aren't independent samples) — **the result held exactly**, ruling out the
  possibility that the original 45-image test was simply too small to trust.
- **A procedural geometric test set** (40 synthetic images) shows a much larger PSNR
  improvement, but this is disclosed honestly as most likely a content-overlap effect — both
  the test set and part of the training mix are now procedural shapes — rather than genuine
  evidence of general-purpose robustness. We would rather under-claim this than over-claim it.

### Known limitations (disclosed, not hidden)

- **bf16 vs. fp32 numerical divergence.** The default inference precision (bf16, for speed)
  produces slightly different output values than full-precision fp32 — expected floating-point
  behaviour, quantified and tracked by our own verification suite as a disclosed, priced
  trade-off rather than an unnoticed bug. Switching to fp32 was tested directly: it does not
  meaningfully improve image quality and costs roughly 10% throughput, so bf16 remains the
  default. Full numbers in `docs/decisions.md` and `docs/BLOCKERS.md`.
- **Training was not run to full convergence.** A cloud training job was cut short by an
  external cancellation (most likely a cloud credit limit) partway through a planned
  fine-tuning schedule. We measured — rather than assumed — whether the remainder could be
  completed on local hardware instead: it cannot, within any reasonable time budget, because
  the local GPU has less than a third of the memory the training recipe needs. This is
  reported as a real, priced constraint, not glossed over.

### What's published alongside this checkpoint

400 restored test outputs (regenerated against this exact checkpoint) are archived and
published as a GitHub Release, with a checksum that can be independently verified. Live
verification status for every automated check in this project is written to
`results/verification_report.json` on every run; the current state is summarised at the bottom
of this README under "Verification."

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
| Bicubic ×2 (raw NoisyLR, the floor) | 23.6524 ± 3.0236 | 0.54775 ± 0.19197 | 0.41206 ± 0.15407 | 253.2 img/s @N=400, RTX 4060 (CPU, single-threaded numpy), `results/runtime_report_bicubic.md` |
| Median 3×3 → bicubic ×2 | 25.5057 ± 3.8785 | 0.61317 ± 0.17232 | 0.40870 ± 0.15866 | 127.0 img/s @N=400, RTX 4060 (CPU, single-threaded numpy), `results/runtime_report_median.md` |
| Non-local means → bicubic ×2 | 26.2722 ± 4.3037 | 0.65152 ± 0.19523 | 0.42586 ± 0.18627 | 35.9 img/s @N=400, RTX 4060 (CPU, single-threaded numpy), `results/runtime_report_nlm.md` |
| U-Net baseline (UNetSR w32 L4, 2,970,401 params) | **28.8808 ± 4.5328** | 0.78273 ± 0.14245 | 0.26525 ± 0.14878 | 23.3 img/s @N=400, RTX 4060, bf16, batch 4, via `run.py --weights weights/baseline_unet.pt`, `results/runtime_report_unet.md` |
| Prior shipped checkpoint (superseded 2026-08-17): NAFSR w48n16, from scratch, 388,225 params | 28.7865 ± 4.5329 | 0.78287 ± 0.14169 | 0.25324 ± 0.13193 | 8.3 img/s @N=400, RTX 4060, bf16, batch 32 (high-variance measurement, 681% spread — see `results/runtime_report.md`) |
| Round 2 long-run checkpoint (superseded 2026-08-17): NAFSR w64n32, FiLM+uncertainty, cloud long run (1,393,938 params) | 29.2548 ± 4.6210 | 0.79211 ± 0.14321 | 0.25625 ± 0.14627 | 17.3 img/s @N=400, RTX 4060, bf16, batch 32 (see `results/runtime_report.md`) |
| **Shipped — NAFSR ×2 w64 n32, EMA, FiLM noise-conditioning + uncertainty head, structural-content fine-tune (1,393,938 params)** | **29.5850 ± 4.6301** | **0.79460 ± 0.14204** | **0.25416 ± 0.13263** | **19.40 img/s** end-to-end at N=400, RTX 4060 Laptop GPU, bf16, batch 4 (current default, re-swept this session — D74) — dev-machine measurement, not an H100 number (`results/runtime_report.md`) |

**Every throughput figure above, including the three classical baselines and the U-Net
baseline, is measured with the exact same methodology as the shipped model**: an external
process timer wrapping the whole run (interpreter start to finish, not an internal timer
around just the compute), median of 5 repeats, on this same RTX 4060 Laptop GPU dev machine
(`scripts/benchmark_runtime.py`, now generalised with a `--target_script` option so the
identical harness times any script sharing the same `--input_dir`/`--output_dir` contract —
`docs/decisions.md` D77). The classical baselines run on CPU only (they are plain numpy, no
GPU code path exists for them), which is why they're markedly faster than the learned models
at this small image size despite non-local-means being the most expensive of the three
per-image.

Values are `mean ± population standard deviation`. Source: `results/metrics_summary.md`,
machine-generated by `scripts/evaluate.py`; per-image records live in
`results/baselines/*/metrics.json`, which **are** committed (`.gitignore` negates the blanket
`results/*` rule for this specific file, `!results/baselines/*/metrics.json`) — the 2000 `.npy`
prediction files in those same directories are not, since they would blow the tree-size cap.
The shipped row is a fine-tune of the Round 2 long-run checkpoint that mixes procedural
structural content into training, promoted after a paired re-score against the prior
checkpoint under one harness showing no regression anywhere and the first genuine real-SEM
OOD improvement of the whole investigation (`docs/decisions.md` D71/D72); earlier rows are
kept for the historical comparison, not deleted.

### Against the classical baselines: wins all three metrics

Against the bicubic floor: **+5.93 dB PSNR**, +0.247 SSIM, −0.158 LPIPS. Against the strongest
classical baseline, non-local means: **+3.31 dB PSNR**, +0.143 SSIM, −0.172 LPIPS. Both
learned models clear all three classical baselines on all three metrics.

### Against the learned baseline: wins all three metrics (V28 passes)

**The shipped model beats the U-Net baseline on all three metrics**, a reversal from the
original from-scratch checkpoint's documented 1-win/1-loss/1-tie negative result (kept below
for the record). Paired per-image difference, the same 400 images, same statistic
`check_V28` uses:

| Metric | Paired mean difference (shipped − U-Net) | t | shipped better on | Verdict |
|---|---|---|---|---|
| PSNR | **+0.7042 dB** | +27.18 | 398 / 400 | **win** |
| SSIM | **+0.01187** | +16.35 | 376 / 400 | **win** |
| LPIPS | **−0.01108** | −3.86 | 220 / 400 | **win** |

V28 requires winning at least 2 of 3; winning all 3 clears it comfortably (`results/metrics_summary.md`,
`docs/decisions.md` D72). The original from-scratch checkpoint's negative result (kept for the
record, not deleted): PSNR −0.0943 dB (loss), SSIM +0.0001 (tie), LPIPS −0.0120 (win) — a
0.388M-parameter model matching a 2.97M-parameter one on fidelity while winning perceptual
quality with 7.7x fewer parameters. That trade-off no longer applies to the current shipped
checkpoint, which now wins outright by a wider margin than the checkpoint it superseded.

### Why the LPIPS column matters more than it looks

Across the classical baselines, fidelity and perceptual quality pull in *opposite* directions:
non-local means wins PSNR by 2.6 dB over bicubic yet scores the **worst** LPIPS of the three,
because it buys fidelity by over-smoothing. A restoration model that improved PSNR while
degrading LPIPS would be gaming one half of an undisclosed scoring blend at the other half's
expense. The shipped model improves both simultaneously against every classical baseline, and
now against the learned baseline too.

### Two numbers, do not confuse them

Every config, including the shipped checkpoint's `configs/finetune_structural_content.yaml`,
validates in-loop against a **100-image subset** (`--val_limit`, default 100) purely for cheap
periodic checkpoint selection during training — never the reportable number. The reportable
figure is always the **full 400-image committed split**, produced by `scripts/evaluate.py`
from `.npy` files reloaded from disk after training ends: **29.5850 dB** for the shipped
checkpoint. (The base long-run's own in-loop training log was not preserved — HF Jobs storage
is ephemeral and the intended save location was not carried out for that run, a real gap, not
hidden here. It does not affect the reportable number, which comes from the checkpoint's own
embedded, disk-verified full-split metrics, not the log.)

### What metric selects the "best" checkpoint

**Every config selects on validation PSNR alone** (`train.py`'s in-loop checkpoint-save
condition is hardcoded to `val_psnr > best_psnr`; `save_best_on: psnr` in every YAML records
this but is not itself a switch — there is no blended-criterion option in the code today).
KLA scores an undisclosed PSNR+SSIM+LPIPS blend, and the shipped checkpoint's own margin over
the U-Net baseline is narrowest on LPIPS (t=−3.26, the weakest of its three wins above) — so a
PSNR-only selection criterion is a real, disclosed limitation, not a neutral implementation
detail. **Checked, not just disclosed** (`docs/decisions.md` D66): re-scored all 35 "new best"
checkpoints the long run pushed, under PSNR-only, SSIM-only, LPIPS-only and a blended
criterion. A naive mean-based blend suggested an earlier checkpoint (step 54000) scored
better overall — but that did **not** survive a proper paired significance test: step 54000
loses PSNR and SSIM significantly on the in-distribution val split (only 1 of 3 metrics won,
below this project's own "win ≥ 2/3" bar) and is WORSE, not better, on real-SEM OOD — the
actual axis PSNR-only selection was worth worrying about. **No swap warranted; the shipped
checkpoint is the correct pick.** The selection-metric limitation itself is still real and
still disclosed — this run just didn't happen to cost anything.

### Failure cases

`results/qualitative/` holds full-resolution panels including two documented failure modes: the
lowest-PSNR validation image in the split and a separately-selected fine broadband-texture case
that aliases under the recovered downsample kernel (`results/eda/aliasing_failure_case.png`,
`docs/decisions.md` D5) — exactly the failure mode ("hallucinating structure that is not there")
the no-adversarial-loss decision below exists to avoid. **Both are quantitatively confirmed to
blur rather than hallucinate**: the model's high-frequency energy on these cases is only
3-35% of the true GT energy in the band the LR input cannot supply — the opposite signature
from inventing detail (`docs/decisions.md` D70, `scripts/blur_vs_hallucination_check.py`).
These panels are regenerated against the currently-shipped checkpoint; the PSNR figures in
their filenames match the numbers on this page.

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
| GPU (dev machine — inference, all timing, runtime measurement) | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| GPU (training — the shipped checkpoint) | **NVIDIA A100 (HF Jobs cloud, A100-large)** — training and the architecture/hyperparameter sweep that selected this configuration both ran here, not on the dev machine |
| Scoring GPU | NVIDIA H100 (KLA's) — **no H100 number appears in this repo; any future H100 figure will be labelled a projection, not a measurement** |

**The shipped checkpoint's training environment differs from the table above.**
`weights/best.pt` was trained on an **HF Jobs A100-large GPU** (cloud),
not the RTX 4060 — see the Training section below and `docs/decisions.md` D55/D61. Inference,
runtime measurement, and every other command in this README are run on the RTX 4060 table
above, which is also the machine that produced every reported dB/img/s figure.

Clone the repository:

```bash
git clone https://github.com/sahithsundarw/bunker_backer.git
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
GPU the same wheels install and the last field is `False`; `run.py` then runs on CPU
without any change.

Fresh-clone compatibility was re-confirmed 2026-08-18, against the current checkpoint
(commit `923e261`), in a real Linux/CUDA container (`nvidia/cuda:12.4.1-base-ubuntu22.04`,
Python 3.12.13 matching this repo's pinned build environment, Docker `--gpus all` passthrough
of this dev machine's own RTX 4060 Laptop GPU, `nvidia-smi` confirmed the GPU visible inside
the container): a fresh `git clone`, fresh venv, `pip install -r requirements.txt` (installing
the real `torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` CUDA wheels, no CPU fallback), and
`run.py` ran end-to-end against the fixture corpus — verifier checks V04 and V46 both
PASS (`results/verification_report.json`). This supersedes an earlier, pre-Phase-3-checkpoint
confirmation that used a plain `python:3.12-slim` container without GPU passthrough. This is
still a target-platform compatibility check, **not** an H100 runtime measurement.

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
.venv/Scripts/python.exe run.py sample_inputs results/sample_outputs
```

or, equivalently:

```bash
.venv/Scripts/python.exe run.py --input_dir sample_inputs --output_dir results/sample_outputs
```

`sample_inputs/` holds 6 real degraded 128×128 inputs (394 KB total) so a reviewer can verify
the script without downloading the dataset. Point the input argument at any directory of
degraded `.npy` files and the output argument anywhere you like — those two directories are
the entire interface (SPEC F11). No other argument is required, no file needs editing, and
weights are resolved relative to the script file, never to the working directory, so the
script runs correctly from any CWD. The verified checkpoint is committed at `weights/best.pt`
(a byte-identical mirror also lives at `models/best.pt`, see below); a fresh clone needs no
checkpoint download or manual placement.

Normal submission inference requires that checkpoint. If it is missing or cannot be loaded,
the command prints an explicit error and exits nonzero without writing bicubic outputs as
though they were model predictions. `--require_weights` is retained as an explicit assertion
and has the same strict behavior. Omitting both directories (neither positional nor flag form)
also exits nonzero with a clear message — both directories are always mandatory, just
satisfiable two ways.

> **Why two invocation forms.** The organizers issued an official, track-specific
> final-submission announcement (2026-08-18, `docs/decisions.md` D75) requiring the entry
> script be named `run.py` and invoked `python run.py <input_dir> <output_dir>` — a change
> from this project's original spec, which named the script `inference.py` and used
> `--input_dir/--output_dir` flags. Rather than guess which reading of `<input_dir>
> <output_dir>` was meant (literally positional, or prose shorthand for the same two flags),
> **`run.py` accepts both**, so neither reading can fail. `inference.py` (the original name)
> still exists as a 3-line back-compat shim importing `run.py`'s `main()` — but `run.py` is
> the file that is graded, timed, and covered by the verifier; `scripts/verify_all.py`'s `V02`
> proves both invocation forms work end-to-end and that omitting both correctly fails.
>
> The announcement also specifies a submission folder shaped
> `team_name/{run.py, requirements.txt, README.md, models/}`. This repo satisfies that as a
> superset — the original spec's "public GitHub repo" requirement (with `src/`, `scripts/`,
> `docs/`, `train.py`, etc.) is still in force and isn't rescinded by this announcement, so
> nothing was removed; `run.py`/`requirements.txt`/`README.md` are all at repo root as
> before, and a real `models/` directory now exists containing a byte-identical copy of
> `weights/best.pt` (sha256-verified equal, and kept from silently diverging by the new `V70`
> check — see `models/README.md`). `V69` (new) proves this shape automatically rather than
> just claiming it in prose.

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
corrupt inputs that legitimately reach 2.16. `run.py`'s module-level imports are exactly
`argparse os sys time pathlib concurrent.futures numpy torch`, enforced statically by a
submission-blocking check.

## Training

**The commands that reproduce the currently-shipped checkpoint** (`weights/best.pt`,
29.5850 dB) are a base training run followed by a fine-tune, not the closed-form path below:

```text
python train.py --config configs/long_run_e.yaml --seed 42 --hub_repo <a HF Hub model repo>
python train.py --config configs/finetune_structural_content.yaml \
    --resume <the checkpoint from the command above> --hub_repo <a HF Hub model repo>
```

The first command trains NAFSR (width=64, num_blocks=32, FiLM noise-conditioning +
heteroscedastic uncertainty head both enabled, 1,393,938 params) from scratch for a
129,700-iteration cosine schedule, best-of-run selected at iteration 76,000. It ran on an
**HF Jobs A100-large GPU**, not locally — `--hub_repo` is required on that hardware because
Job storage is ephemeral. Measured wall-clock: 22,895.55 s (6h 21m). Full provenance,
including the sweep that chose this config over five alternatives on a measured
params-vs-quality frontier: `docs/decisions.md` D55/D61, `results/eda/pareto_frontier.png`.

The second command resumes from that checkpoint (never from scratch) and fine-tunes with
27.5% procedural structural content mixed into training (`structural_content_ratio`,
`src/structural_content.py`), targeting a real-SEM OOD regression the base checkpoint had
disclosed. This is the currently-shipped result: no regression on any metric on any
evaluation set, and the first genuine real-SEM OOD improvement of the investigation
(LPIPS). Cut short at ~8,000 iterations by an external cloud-job cancellation (leading
hypothesis: the org's cloud credit ceiling, unconfirmed) — a from-scratch re-run of this
command is not expected to stop at exactly that iteration. Full diagnostic path that led to
this fine-tune's design (why degradation-parameter widening alone was tried and rejected
first, and how content statistics pointed at this fix instead):
`docs/decisions.md` D63/D67/D68/D69/D71/D72.

**Locally, without cloud hardware**, the pipeline can still be exercised end to end:
`--smoke` (a handful of steps, used by the verifier), `--overfit N` (deliberately overfit N
pairs — the pipeline sanity gate; if it cannot reach a high PSNR on 2 pairs, alignment,
normalisation or the loss is broken and nothing downstream is trustworthy), `--iters`,
`--val_every`, `--val_lpips`, `--resume <checkpoint>` (fine-tune from an existing checkpoint;
pairs with `optim.finetune_horizon` in the config to give the fine-tune its own cosine
schedule rather than resuming into the tail of the original one), `--seed`, `--tag`. Every
non-smoke run appends a row to `results/experiments.csv` with the git SHA, config, seed,
metrics and wall-clock — this is enforced in code, not just convention (`--no_ledger` is
accepted but ignored for real runs). That ledger is tracked in this repository.

There is also a **CPU-feasible, from-scratch-fast alternative** that does not ship but exists
for quick local iteration on machines without a GPU:

```text
python train.py --config configs/final.yaml --data_root <dataset_root> --seed 42 \
  --closed_form_linear --out weights/closed_form_demo.pt
```

This fits a ridge-regularised 5×5 linear restoration kernel on the training pairs, then embeds
the learned residual into a NAFSR stem/head, with `total_iters_run=0` and
`training_mode=closed_form_linear_ls5` recorded on the checkpoint. On CPU it takes well under a
minute. It is a pipeline-sanity/demo path, not a competitor to the shipped gradient-trained
checkpoint — see the Result summary table above for how far apart they score.

The dataset lives outside the repository and is never committed. Local training and
dataset-dependent verifier checks require `KLA_DATA_ROOT` to point at the dataset root (or
`--data_root`); it must contain `train/GT` and `train/NoisyLR`. `configs/split_val.txt` is the
committed, explicit validation file list and is never regenerated randomly at run time, which
would leak.

## Cloud training pipeline

Training compute beyond what a laptop GPU can do in reasonable time runs on **Hugging Face
Jobs**, billed per-GPU-hour, dispatched from a local script rather than a notebook cell so the
exact command is on record (`scripts/dispatch_finetune_job.py` is the most recent example).
In brief: the training data lives in a
private HF dataset repo (`Team-Ceciroleo67/kla-ps01-data`, never the public code repo — F17's
"never train on `test_NoisyLR`" travels with the data regardless of which machine loads it);
the job container clones this repo from GitHub at a specific commit, downloads the dataset,
and runs `train.py` with `--hub_repo` set so every checkpoint save survives the job's ephemeral
storage by being pushed to a private HF Hub model repo (`Team-Ceciroleo67/kla-ps01-checkpoints`)
immediately. Checkpoints are pulled back locally afterward, re-scored under the same harness as
every other comparison in this repo, and only promoted into `weights/best.pt` on a paired win.

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
| `run.py` | the evaluation script KLA runs; standalone, two mandatory directory args, satisfiable positionally or via `--input_dir/--output_dir` (docs/decisions.md D75) |
| `inference.py` | 3-line back-compat shim (`from run import main`) — kept for the original spec's naming, not scanned by the verifier |
| `train.py` | reproduces the checkpoint, including the CPU-feasible closed-form training mode |
| `requirements.txt` | complete `pip freeze`, every line `==` pinned |
| `models/` | byte-identical mirror of `weights/best.pt`, added solely to satisfy the organizers' announced submission-folder shape; kept in sync by `V70` — see `models/README.md` |
| `sample_inputs/` | 6 real degraded inputs so inference can be verified without the dataset |
| `src/` | `model.py` `blocks.py` `dataset.py` `degrade.py` `losses.py` `metrics.py` `io_utils.py` `utils.py` `unrolling.py` (stretch goal, not shipped — see Method summary) |
| `configs/` | `final.yaml` (base recipe), `nafnet_x2.yaml`, `baseline_unet.yaml`, `long_run_e.yaml` (**shipped checkpoint's config**), `finetune_ood_wide.yaml`, 6 Pareto-sweep configs, `split_val.txt` |
| `scripts/` | dataset forensics, degradation fitting, baselines, evaluation, benchmarking, cloud dispatch, `verify_all.py` |
| `docs/` | SPEC, SPEC addendum (governs on conflict), verification contract, dataset findings, I/O contract, cloud training plan, decisions, blockers, state |
| `results/eda/` | dataset figures, degradation fit, content contact sheets, Pareto frontier, calibration probes |
| `results/metrics_summary.md` | machine-generated results table |
| `results/restored_test_outputs/` | mandatory model outputs, generated from the current `weights/best.pt` (29.5850 dB). **Holds a manifest and hashes, not the output bytes** — the 400-file archive (90,929,851 B, sha256 `7c5a63ff8720bbbbf781891c6fdb1302bc925095806278766ad08ca2abe9c6ef`) is published as GitHub Release `artifacts-v3`, verified fetchable from a logged-out session; see that folder's own `README.md` |
| `results/runtime_report.md` | externally-timed RTX 4060 CUDA runtime for the current checkpoint (400 images, median 20.62 s, 19.40 img/s, batch 4 — `run.py`'s real default, no override) — explicitly not an H100 number; earlier checkpoints' figures are kept for the record |
| `weights/` | tracked `best.pt` checkpoint (SHA256 `6d74ccfdd72e1271a7de5fdede5c341b3cf18ca4294619dd90a97c0591f66397`) + provenance notes |

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
   arguments is unchanged. **Calibration measured against the actual shipped checkpoint** (the
   figure originally cited here, D59, was measured against a stale sweep checkpoint by
   mistake — corrected 2026-08-17): per-image Pearson r=**0.980**, Spearman r=**0.972**;
   pooled per-pixel (400,000 pixels) Pearson r=0.462, Spearman r=0.612 (weaker, as expected —
   NLL trains the mean relationship, not pixel-exact prediction). Mean predicted variance
   (0.00208) closely matches mean actual squared error (0.00199). The FiLM embedding's
   relationship to true noise level is present but diffuse — no single dimension or the
   embedding norm correlates well alone, but a 16-dim linear probe explains ~23% of
   true-noise-level variance held out (D58). An
   architecture/hyperparameter sweep with both enabled was run on cloud A100 hardware to select
   this config — see `docs/decisions.md` D52/D55/D61 and `results/eda/pareto_frontier.png`.
7. **Algorithm-unrolling hybrid — `UnrolledSR` (`src/unrolling.py`), stretch goal, NOT
   shipped.** Unrolls T proximal-gradient steps against the *measured* x2 degradation kernel
   (`RECOVERED_KERNEL_4X4`, item 1 above) as a fixed, non-trainable forward operator, with a
   small weight-tied `NAFBlock` denoiser as the learned proximal operator per step (Monga et
   al. 2021, IEEE SPM — the exact survey cited in KLA's own reference list). Stated honestly:
   its overfit sanity gate fails (plateaus around 23.4-23.6 dB, well below the 40 dB bar that
   NAFSR/UNetSR both clear on the same fixture). Root-cause investigation is **complete, not
   abandoned**: all three leading hypotheses (adjoint/forward-operator mismatch, unstable
   step size, weight-tying) were tested directly and each cleared — the adjoint identity holds
   to ~0.4%, the step size sits inside a measured ~0.48x stability margin, and weight-tied vs
   untied denoisers converge to the same plateau. No single fixable bug was found; this reads
   as genuinely slow convergence for this architecture on this budget, not a correctness defect
   (`docs/decisions.md` D60). Reported as an honest negative result — not shipped, not silently
   dropped either.
8. **Loss** is balanced because the scoring blend is undisclosed: Charbonnier + (1−MS-SSIM) +
   an FFT-magnitude term, with LPIPS available but **off by default**. **No adversarial
   loss** — hallucinating a structure that is not there is the worst possible failure in an
   inspection context.
9. **Throughput** is measured over the whole process, not inferred from kernel timings. The
   current 400-image external run (`results/runtime_report.md`), with `run.py`'s real
   default (batch 4, no override), measures a median 20.62 s total (19.40 img/s) on the RTX
   4060, n=5. The detailed profiler breakdown
   (import overhead share, layer-norm/conv time share) was measured at the prior, smaller
   checkpoint's size and has not yet been re-run at the current 1,393,938-param size
   (`results/runtime_report.md`'s own stated follow-up) — NAFSR's identified
   **memory-bandwidth-bound, not compute-bound** profile is why `channels_last` and bf16 each
   moved throughput by under 20% at that size; whether it still holds at this size is an open
   re-measurement, not assumed to carry over unchanged.

## Assumptions

- The **primary** simulated downsample kernel is the recovered 4×4 sharpening kernel
  (`RECOVERED_KERNEL_4X4`, centre weights ≈0.320, negative surround ≈−0.045 — provably not a
  box filter). `bicubic(antialias=False)` is used as a 25% minority alternative
  (`bicubic_alt_prob`), not the working model — the true kernel support extends slightly
  beyond 4×4 (a K=6 recovery finds max |weight| 0.01355 in the outermost ring), worth ~1e−05 in
  residual std either way, so both models are close but not exact.
- Noise is applied after downsampling and is signal-dependent with no additive floor. An
  additive Gaussian term is nevertheless retained in augmentation, randomised over
  `U(0, 0.065)` **including zero** (widened from the original `U(0, 0.02)` to close a measured
  tail-coverage gap, `docs/decisions.md` D43), as a cheap hedge because the brief names
  additive Gaussian as a degradation and warns that test noise levels may vary.
- GT is per-image min–max normalised to exactly [0,1] (all 3200 files attain both endpoints).
  This licenses clipping the output and nothing more.
- Train and test are treated as the same domain: measured spectral peakiness and gradient
  anisotropy differ by only ×1.02–1.04.
- The hidden test set may contain genuine semiconductor imagery. We assume the **degradation**
  transfers and the **content prior** does not, and we optimise accordingly
  (`docs/decisions.md` D16).

## External resources & licences

**No external datasets or pretrained weights are used in the shipped model.** Its lineage is
trained from scratch on the provided image pairs, then fine-tuned (never on a foreign
dataset — the fine-tune mixes in procedural structural content from this project's own fixed
generator, `src/structural_content.py`, F15-permitted synthetic data derived from GT-adjacent
shapes, not an external source). Two external resources are used for *evaluation only* and
never for training: LPIPS (Zhang et al., CVPR 2018) with its standard AlexNet backbone, and a
real-SEM image set used for an OOD robustness report (`docs/decisions.md` D53). Neither
contributes a gradient to the shipped checkpoint and neither is required to run
`run.py`.

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
records the CURRENT shipped checkpoint: at the full 400-image test set (`configs/split_val.txt`),
batch 4 (`run.py`'s real default — no `--batch_size` override; re-swept and changed from
32 this session, `docs/decisions.md` D74), precision bf16, on the **NVIDIA GeForce RTX 4060
Laptop GPU**, total wall-clock **median 20.62 s (19.40 img/s)**, n=5 repeats. No H100 number
exists or is claimed anywhere in this repository.

**A batch-size re-sweep this session found batch 4 beats the old default of 32 by 31.8% lower
wall-clock at 128→256 and 18.1% at 256→512** (monotonic across {4,8,16,32,64}, both
resolutions, n=5 medians, one interleaved session — `results/runtime_report.md`), so the
default was changed. A real bug was also found and fixed in the process:
`scripts/benchmark_runtime.py` had its own hardcoded `--batch_size` default of 32 and always
forwarded it explicitly, so it could never actually measure `run.py`'s own default —
every "no override" run silently pinned to 32 regardless. Fixed to only forward `--batch_size`
when explicitly passed (`docs/decisions.md` D74).

Earlier checkpoints have their own records at the same N, device and method, kept for the
historical comparison: the Round 2 long-run checkpoint (29.2548 dB, byte-identical param
count to the current one) measured 23.19 s / 17.3 img/s in a separate session; the original
from-scratch checkpoint (28.7865 dB, 3.6x fewer params) measured 48269.4 ms / 8.3 img/s but
with a **681.4% spread** across its 5 repeats — high enough that `results/runtime_report.md`
itself flags it as likely system noise/contention, not a clean steady-state number.
**Absolute img/s figures on this laptop GPU vary noticeably across measurement sessions**
(confirmed directly: a controlled back-to-back re-benchmark of the 388,225-param and
1,393,938-param checkpoints in one session, plan Phase B2, gave 19.79 and 12.79 img/s
respectively — neither matches either standalone session's own number for the same
checkpoint). Only same-session, back-to-back comparisons support a relative speed claim on
this hardware; cross-session absolute figures are recorded for the record, not compared
directly to each other.

Two commitments about how every measurement above was taken, both honoured:

- timing was taken **externally around the whole process**
  (`subprocess.run([sys.executable, "run.py", ...])`), not by an internal timer around the
  forward pass;
- every number is labelled with the device it was measured on. Training happens on an HF Jobs
  A100 (see Cloud training pipeline above); every *timing* number in this repository is measured
  on the RTX 4060 Laptop GPU; **no H100 measurement exists and none is presented as one.**

`scripts/benchmark_runtime.py` is the shared external-process harness. Its detailed
batch/precision/memory-format sweep and profiler breakdown (`results/runtime_report.md`'s
lower sections) were measured against the superseded, smaller checkpoint's architecture size
and have not yet been re-run at the current, 3.6x-larger size — a disclosed, open
re-measurement, not assumed to still hold.

## Verification

Correctness for this project is defined by `docs/VERIFICATION_CONTRACT.md` (immutable) and
executed by `scripts/verify_all.py`, which defines **71 checks** (69 plus `V69`/`V70`, added
2026-08-18 for the `run.py` final-submission announcement, `docs/decisions.md` D75) and writes
`results/verification_report.json`. Run it with
`.venv/Scripts/python.exe scripts/verify_all.py --strict`, or add `--fresh-clone` for the
clean-room checks. It is **not** listed as a fenced command here because it exits non-zero
while the project is incomplete, and no command in this README exits non-zero.

**The suite is not green, and this README does not claim it is.** Full fresh run after the
`inference.py` -> `run.py` rename and verifier retarget, 2026-08-18: **67 PASS / 4 FAIL**
(V04, V22, V46, V53) — the exact same known/expected/disclosed set as before this rename,
confirming zero regressions from it. V04 and V46 require `--fresh-clone` (not passed on this
default `--strict` invocation, hence the FAIL here) — re-confirmed passing post-rename against
the new entry point, see below. **V22 is a disclosed, live fail, not a bug being chased** —
see the status block at the top of this file and `docs/BLOCKERS.md` B12 for the bf16/fp32
divergence investigation and why the human accepted it as a trade-off rather than authorising
a fix. **V21 flaked once on an earlier run during this session's changes** (2 PASS, 1 FAIL
across 3 isolated re-checks) — the same known, pre-existing, load-dependent
cross-process-determinism flake this project has documented before (`docs/BLOCKERS.md` B11,
~20% intermittent rate), unrelated to the rename (the check's assertion is unchanged; only its
target filename moved from `inference.py` to `run.py`), and green again on this final run.
**V53** (the deck contract) correctly FAILs: no `*_KLA_PS01.pdf` currently exists at the repo
root by design, not by accident — `scripts/build_deck.py` was proven working with real team
info (`docs/decisions.md` D76/D78), then the generated PDF was deliberately removed (D79) to
make way for a hand-built PPT a teammate is producing; it will be added once ready. Not a
check malfunction, and not a hidden gap. **V69 and V70 (new) both PASS** — automated proof
the announced
submission-folder shape exists and `models/best.pt` matches `weights/best.pt`.
No check has ever been weakened, skipped,
or had its tolerance widened to turn a FAIL green (Prime Directive 1) — every V-check addition
in this project's history was a strengthening, negative-controlled before being trusted. The
authoritative per-check status is always `results/verification_report.json`, regenerated on
every run; `docs/STATE.md` carries the rolling ledger and `docs/BLOCKERS.md` the remaining
open items.
