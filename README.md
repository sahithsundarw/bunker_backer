# KLA PS01 — AI-Based Restoration of Degraded Images for Semiconductor Inspection

SEMICON India Hackathon 2026 · Track 1 · Problem Statement PS01
Public repository: `https://github.com/sahithsundarw/semicon-kla-image-restoration`

Restores a degraded grayscale image — noisy and downsampled by exactly ×2 — to a clean
estimate at full resolution, in one blind pass, with the whole model running at low
resolution and a single PixelShuffle ×2 head.

---

> ## STATUS — iteration 2
>
> **Trained, evaluated and published.** A 20,000-iteration run (seed 42, wall-clock **1:11:43**)
> produced the checkpoint `inference.py` loads (`weights/best.pt`, NAFSR); a U-Net baseline was
> trained under the same 20,000-iteration
> budget for the like-for-like learned comparison; both were scored on the committed 400-image
> validation split; and the 400 restored test outputs were produced by `inference.py` with
> `--require_weights`. Checkpoint and outputs are both downloadable — Release `artifacts-v1`,
> each with a published sha256 taken from the **served** bytes.
>
> **What is still open, stated plainly:**
>
> - **Throughput is measured, on the dev machine, not on an H100.** `results/runtime_report.md`
>   records an externally-timed (`subprocess`-wrapped, not an internal timer) full run of the
>   400-image test set on the **NVIDIA GeForce RTX 4060 Laptop GPU** (bf16, batch 32): total
>   wall-clock **median 48269.4 ms (8.3 img/s)**, fitted as a fixed startup cost of **14755 ms**
>   plus **86.55 ms/image** marginal compute, with the fixed cost at **30.6%** of total
>   wall-clock at N=400. No H100 number exists or is claimed.
> - **V22 is red**: bf16 and fp32 inference outputs diverge by `max 1.27e-02` against a `1e-02`
>   tolerance (the mean, `5.99e-04`, passes). This is a real numerical defect, not a tolerance
>   problem, and it is being fixed rather than waived (`docs/STATE.md`).
> - **V28 is red**: the shipped model does **not** beat the learned U-Net baseline on 2 of 3
>   metrics. See *Result summary* — it is reported, not buried.
> - **Which model ships is not decided.** NAFSR and the U-Net are a fidelity/perceptual
>   trade-off at very different parameter counts, and the decision is waiting on the throughput
>   measurement above.
> - **V04 and V46** require a `--fresh-clone` verifier run, which has not been performed since
>   the current commit.
>
> The committed `results/verification_report.json` is a **snapshot from commit `c209cd2`**
> (43 PASS / 14 FAIL) and is stale: several of those checks have since gone green and the
> verifier itself has been strengthened since. Do not read it as current. The rolling ledger
> is `docs/STATE.md`; the authority is a fresh `scripts/verify_all.py --strict` run.

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
| NAFSR ×2 w48 n16, EMA (388,225 params) | 28.7865 ± 4.5329 | 0.78287 ± 0.14169 | **0.25324 ± 0.13193** | **8.3 img/s** end-to-end at N=400, RTX 4060 Laptop GPU, bf16, batch 32 — dev-machine measurement, not an H100 number (`results/runtime_report.md`) |

Values are `mean ± population standard deviation`. Source: `results/metrics_summary.md`,
machine-generated by `scripts/evaluate.py`; per-image records live in
`results/baselines/*/metrics.json`, which **are** committed (`.gitignore` negates the blanket
`results/*` rule for this specific file, `!results/baselines/*/metrics.json`) — the 2000 `.npy`
prediction files in those same directories are not, since they would blow the tree-size cap.

### Against the classical baselines: wins all three metrics

Against the bicubic floor: **+5.13 dB PSNR**, +0.235 SSIM, −0.159 LPIPS. Against the strongest
classical baseline, non-local means: **+2.51 dB PSNR**, +0.131 SSIM, −0.173 LPIPS. Both
learned models clear all three classical baselines on all three metrics.

### Against the learned baseline: one win, one loss, one tie — and V28 is red

**The NAFSR model — the checkpoint `inference.py` currently loads — does not beat the U-Net
baseline.** Both models score the same 400
images, so the honest statistic is the **paired** per-image difference, not the gap between two
independent means:

| Metric | Paired mean difference (NAFSR − U-Net) | t | NAFSR better on | Verdict |
|---|---|---|---|---|
| PSNR | **−0.0943 dB** | −6.11 | 93 / 400 | **loss** |
| SSIM | +0.0001 | +0.29 | 172 / 400 | **tie** (\|t\| < 1.96) |
| LPIPS | **−0.0120** | −5.55 | 235 / 400 | **win** |

V28 requires winning at least 2 of 3, so **V28 is red**, correctly. The contract offers exactly
one escape hatch — a structured negative-result entry in `docs/decisions.md` *plus* shipping the
better model — and it has not been taken, because which model to ship has not been decided.

What the result actually says: a **0.388 M-parameter** model matches a **2.97 M-parameter** one
on fidelity (SSIM is a statistical tie; PSNR is a 0.0943 dB loss, about 2% of the 4.53 dB
standard deviation across the split, but consistent enough across images that the paired test
resolves it at t = −6.11) while winning perceptual quality with 7.7× fewer parameters. The
scoring blend across PSNR/SSIM/LPIPS is undisclosed (SPEC F9) and the throughput axis is
unmeasured, so neither model can be declared the submission on the evidence available today.
That decision is deferred to the runtime measurement, not resolved by rhetoric here.

### Why the LPIPS column matters more than it looks

Across the classical baselines, fidelity and perceptual quality pull in *opposite* directions:
non-local means wins PSNR by 2.6 dB over bicubic yet scores the **worst** LPIPS of the three,
because it buys fidelity by over-smoothing. A restoration model that improved PSNR while
degrading LPIPS would be gaming one half of an undisclosed scoring blend at the other half's
expense. The NAFSR model improves both simultaneously against every classical baseline, and
against the learned baseline it trades 0.09 dB of PSNR for 0.012 of LPIPS.

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
inspection-domain content prior.

Metric implementations are pinned, because library defaults differ by non-trivial margins:

- PSNR — `skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=1.0)`
- SSIM — `skimage.metrics.structural_similarity(gt, pred, data_range=1.0,
  gaussian_weights=True, sigma=1.5, use_sample_covariance=False)` (Wang et al. 2004 settings)
- LPIPS — `lpips.LPIPS(net='alex')`, grayscale replicated to 3 channels, rescaled to [-1,1],
  computed on CUDA (LPIPS can differ in the 4th decimal between CPU and CUDA)

The bicubic floor is low by natural-image super-resolution standards because the input is
genuinely noisy. That is expected, not a bug.

Qualitative evidence — 5 success cases, 2 labelled failure cases, full resolution, with the
written failure analysis — is in `results/qualitative/`.

## Environment

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 (`torch.version.cuda == '12.8'`) |
| torchvision | 0.26.0+cu128 |
| GPU (training + all timings) | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| Scoring GPU | NVIDIA H100 (KLA's) — **no H100 number appears in this repo; any future H100 figure will be labelled a projection, not a measurement** |

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
without any change. A `False` on a machine that *does* have an NVIDIA GPU means either the
driver is not usable or the environment is not the one this file pins — see the note below and
`docs/BLOCKERS.md` B8.

> **Why `requirements.txt` starts with `--extra-index-url`.** `torch==2.11.0+cu128` is
> published only on `download.pytorch.org`, never on PyPI. Installing `lpips` without that
> directive silently resolved torch **from PyPI** and replaced the CUDA build with a CPU-only
> wheel on this machine — nothing failed, nothing was logged, and `torch.cuda.is_available()`
> quietly became `False`. The `+cu128` local version in the pin is the safety catch: PyPI
> cannot host a local version, so there is no CPU candidate to fall back to and a broken
> install fails loudly instead of silently costing the entire throughput score. See
> `docs/decisions.md` D18 and `docs/BLOCKERS.md` B8.

## Get the checkpoint

`weights/best.pt` is **not** in this repository — `.gitignore` bans `weights/*.pt` and check
V51 lists `.pt` as a forbidden blob. It is published as a **GitHub Release asset**:

| artifact | Release | bytes | sha256 |
|---|---|---|---|
| `best.pt` (checkpoint, 388,225 params) | `artifacts-v1` | 3288805 | `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` |
| `restored_test_outputs.zip` (the 400 restored test outputs) | `artifacts-v1` | 91069597 | `fbdf8a652d26168cf41e01842ca28d38c53d1da1547bd8ce602b5b8e5d6ac750` |

Both digests are of the **served** bytes, re-fetched with `GITHUB_TOKEN` and `GH_TOKEN`
cleared so the fetch could not have succeeded on cached credentials: HTTP 200, byte counts as
above. The download and digest-check commands are in **`weights/README.md`** and
**`results/restored_test_outputs/README.md`**, deliberately not here: check V46 extracts and
executes every fenced shell command in this file, and a 91 MB download does not belong in a
verification run.

Place the checkpoint at `weights/best.pt`. Nothing else needs configuring — `inference.py`
resolves that path relative to its own file, never to the working directory.

**Without the checkpoint**, `inference.py` prints this on stderr and still exits 0:

```
inference.py: checkpoint not found at ...\weights\best.pt; falling back to bicubic x2 upsample
```

That fallback is deliberate — a script that runs and scores poorly is scored; one that crashes
is not — but its output is **not** a model result. Pass `--require_weights` to make a missing
checkpoint a hard error instead, and use that flag for any run whose output you intend to score.

## Inference — the command KLA runs

```bash
.venv/Scripts/python.exe inference.py --input_dir sample_inputs --output_dir results/sample_outputs --require_weights
```

`sample_inputs/` holds 6 real degraded 128×128 inputs (393,984 B total) so a reviewer can
verify the script without downloading the dataset. Point `--input_dir` at any directory of
degraded `.npy` files and `--output_dir` anywhere you like — `--input_dir` and `--output_dir`
are the entire required interface (SPEC F11), and no file needs editing. Weights are resolved
relative to the script file, never to the working directory, so the script runs correctly from
any CWD.

**`--require_weights` is included above deliberately.** Without it, a fresh clone that has not
yet downloaded `weights/best.pt` (§Get the checkpoint, above) runs to **exit 0** and silently
produces a bicubic ×2 upsample instead of a model result — the flag turns that into a loud,
non-zero-exit failure instead (adversarial review finding C1: the example command previously
shipped *without* this flag, which is exactly the invocation a reviewer following the README
literally would run). `--output_dir` must also be a directory that is neither `--input_dir`
itself nor nested inside it — `inference.py` refuses that combination rather than silently
overwriting the degraded inputs in place (finding H2/H3).

Optional flags, all with defaults that make the two-argument invocation correct:
`--weights`, `--batch_size`, `--device`, `--precision {auto,bf16,fp16,fp32}`, `--compile`,
`--tta`, `--num_workers`, `--write_threads`, `--require_weights`, `--verbose`.
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

Inspect one of the committed inputs and see the out-of-range values for yourself:

```bash
.venv/Scripts/python.exe -c "import numpy as np; a = np.load('sample_inputs/000000.npy', allow_pickle=False); print(a.shape, a.dtype, round(float(a.min()), 4), round(float(a.max()), 4))"
```

which prints, for that file:

```
(128, 128) float32 0.001 1.5406
```

— a maximum well above 1.0, passed to the model untouched. Point the same one-liner at any
file written by the inference run above and it reports `(256, 256) float32` with a range inside
`[0.0, 1.0]`. That is not an aspiration: the 400 published test outputs were re-loaded from
disk after writing and checked against this contract file by file — `float32`, `ndim == 2`,
`(256, 256)`, all finite, observed global range exactly `[0.0, 1.0]`, filename set identical to
the input set, **0 violations in 400** (`results/restored_test_outputs/manifest.json`).

**No image library is used anywhere in the inference path.** The data is `.npy` end to end, so
`cv2`, `tifffile` and `PIL` are absent by design, not by oversight: they are dead weight on a
timed run, and several `cv2` paths silently convert to 8-bit or clip to [0,1], which would
corrupt inputs that legitimately reach 2.16. `inference.py`'s module-level imports are exactly
`argparse os sys time pathlib concurrent.futures numpy torch`, enforced statically by a
submission-blocking check (V23).

## Training

The invocation that produced the published checkpoint is
`python train.py --config configs/final.yaml --seed 42 --iters 20000`. It is deliberately
**not** given as a fenced command: every fenced command in this README is executed by check
V46, and this one needs the dataset and hours of GPU time.

Useful flags for checking the pipeline before committing GPU hours: `--smoke` (a handful of
steps, used by the verifier), `--overfit N` (deliberately overfit N pairs — the pipeline sanity
gate; if it cannot reach a high PSNR on 2 pairs, alignment, normalisation or the loss is broken
and nothing downstream is trustworthy; it currently reaches **43.33 dB** on 2 pairs against a
40 dB gate, `docs/decisions.md` D26), `--iters`, `--seed`, `--tag`, `--out`.

The dataset lives outside the repository (`C:\kla-data` on the dev machine) and is never
committed. `configs/split_val.txt` is the committed, explicit validation file list — it is
never regenerated randomly at run time, which would leak.

**The two runs on record.** Both used seed 42, 20,000 iterations, batch 32 at 64 px LR patches
(→ 128 px GT patches), bf16 autocast, channels_last, AdamW with betas `[0.9, 0.9]`, a cosine
schedule with 500 warm-up iterations and EMA decay 0.999, on an RTX 4060 Laptop GPU (8 GB) with
no OOM and no batch-size reduction — the two configs differ only in the `model` block. Both
checkpoints store the **EMA** weights at the best validation PSNR, and that is what
`inference.py` loads and what every number above was measured from.

| Run id | Model | Params | Iters | Wall-clock | Checkpoint written |
|---|---|---|---|---|---|
| `20260815T062831Z-final-s42` | NAFSR w48 n16 | 388,225 | 20,000 | 1:11:43 (4303.5 s) | `weights/best.pt` — **published**, see above |
| `20260815T174833Z-baseline_unet-s42` | UNetSR w32 L4 | 2,970,401 | 20,000 | 0:15:43 (942.7 s) | `weights/baseline_unet.pt` — local only, not published |

Equal iteration budget is not equal wall-clock: NAFSR has 7.7× fewer parameters but is 4.6×
slower per iteration here, which is consistent with it being memory-bandwidth bound rather than
compute bound (`docs/decisions.md` D21).

Every run appends a row with git SHA, config, seed, metrics and wall-clock to
`results/experiments.csv`. That file **is** in a clone — `.gitignore` negates the blanket
`results/*` rule for it specifically (`!results/experiments.csv`) — so both rows can be read
directly from that file as well as restated above.

## Repository map

| Path | Contents |
|---|---|
| `inference.py` | the evaluation script KLA runs; standalone, two required arguments |
| `train.py` | the training entry point that produced `weights/best.pt` (`configs/final.yaml`, seed 42) |
| `requirements.txt` | complete `pip freeze`, every line `==` pinned |
| `sample_inputs/` | 6 real degraded inputs so inference can be verified without the dataset |
| `src/` | `model.py` `blocks.py` `dataset.py` `degrade.py` `losses.py` `metrics.py` `io_utils.py` `utils.py` |
| `configs/` | `nafnet_x2.yaml`, `baseline_unet.yaml`, `final.yaml`, `split_val.txt` |
| `scripts/` | dataset forensics, degradation fitting, baselines, evaluation, qualitative figures, benchmarking, `verify_all.py` |
| `docs/` | SPEC, SPEC addendum (governs on conflict), verification contract, dataset findings, I/O contract, decisions, blockers, state |
| `results/eda/` | dataset figures, degradation fit, content contact sheets |
| `results/qualitative/` | 5 success + 2 labelled failure figures at full resolution, plus the failure analysis |
| `results/metrics_summary.md` | machine-generated results table |
| `results/restored_test_outputs/` | mandatory model outputs. The 400 `.npy` files are **published as a Release asset** (~105 MB raw, over V51's tree caps); this folder holds `manifest.json` provenance and `sha256sums.txt` with all 400 per-file digests |
| `weights/` | `README.md` with the checkpoint URL, size and sha256. `best.pt` itself is not committed |

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
   heavy compute stays at LR, where it is 4× cheaper. **388,225 parameters, 5.584 GMAC per
   128×128 image.** Fully convolutional with a required size multiple of 1, verified on
   128→256, 256→512, 61×97→122×194 and 1×1→2×2.
5. **Learned baseline — UNetSR:** 2,970,401 parameters, 4.478 GMAC. The comparison is roughly
   FLOP-matched (0.80×) with NAFSR at 0.13× the parameters, so the proposed model gets no
   parameter advantage over the baseline it has to beat — and, as the paired comparison above
   shows, it does not beat it on fidelity.
6. **Loss** is balanced because the scoring blend is undisclosed: Charbonnier + (1−MS-SSIM) +
   an FFT-magnitude term, with LPIPS available but **off by default**. **No adversarial
   loss** — hallucinating a structure that is not there is the worst possible failure in an
   inspection context.
7. **Throughput** is treated as a startup problem, not a kernel problem: the test set is 400
   files / 25.05 MB and the forward pass is sub-millisecond per image, so fixed startup is
   ~85–95% of the scored wall-clock (`docs/decisions.md` D7). Import hygiene therefore
   outranks every micro-optimisation. Independently, NAFSR profiles as
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
from scratch on the provided image pairs. One external pretrained network is used for
*evaluation only* and never for training: LPIPS (Zhang et al., CVPR 2018) with its standard
AlexNet backbone, which the `lpips` package downloads on first use. It contributes no gradient
to the shipped checkpoint and is not required to run `inference.py`.

| Resource | Role | Link | Licence (verified at source) | Paper / model card |
|---|---|---|---|---|
| **LPIPS** (`lpips` 0.1.4) — linear calibration weights, shipped inside the pip package (`lpips/weights/v0.1/alex.pth`, 6,009 B) | Evaluation metric only | `https://github.com/richzhang/PerceptualSimilarity` | **BSD-2-Clause** — read from `LICENSE` at that repository (HTTP 200, fetched 2026-08-15); PyPI metadata agrees (`License :: OSI Approved :: BSD License`) | Zhang, Isola, Efros, Shechtman, Wang, *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*, CVPR 2018 |
| **AlexNet ImageNet-pretrained backbone**, pulled by LPIPS via `torchvision.models.alexnet(pretrained=True)` → `~/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`, **244,408,911 B measured** | Evaluation metric only — the feature extractor inside LPIPS | `https://github.com/pytorch/vision` | **BSD-3-Clause** — read from `LICENSE` at that repository (HTTP 200, fetched 2026-08-15) | Krizhevsky, Sutskever, Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, NeurIPS 2012; distributed via the torchvision model zoo |
| External training datasets (DIV2K, Flickr2K, BSD, SEM corpora, …) | **None used** | — | — | — |
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

## Verification

Correctness for this project is defined by `docs/VERIFICATION_CONTRACT.md` and executed by
`scripts/verify_all.py`, which implements **57 checks** and writes
`results/verification_report.json`. Run it with
`.venv/Scripts/python.exe scripts/verify_all.py --strict`, or add `--fresh-clone` for the
clean-room checks. It is **not** listed as a fenced command here because it exits non-zero
while the project is incomplete, and no command in this README exits non-zero.

**The suite is not green, and this README does not claim it is.** Regenerate the tally rather
than trusting any number written here or in the committed report, because it moves with every
commit. The list below reflects the state of the artifacts in this repository at iteration 2 —
it is a status, not the output of a fresh run, and **a full `--strict` run is pending**. What
is red, and why:

| Check | Why it is red |
|---|---|
| V22 | bf16 vs fp32 outputs diverge, `max 1.27e-02` against a `1e-02` bound. A real defect; being fixed, not waived |
| V28 | the NAFSR checkpoint loses PSNR and ties SSIM against the learned U-Net baseline — see *Result summary* |
| V37 V38 V39 V43 | `results/runtime_report.md` does not exist |
| V04 V46 | require a `--fresh-clone` verifier run, not performed since the current commit |

The contract grew from 53 checks to 57, and that is the more useful fact about it than any pass
count:

- **Nine of the original 53 were inert placeholders** — V25, V26, V27, V28, V29, V32, V33,
  V34, V35 each returned an unconditional FAIL that no artifact could ever turn green. They
  looked identical before and after a real defect. All nine now test their subject, each with
  an anti-vacuity guard (`docs/decisions.md` D22, D26).
- **An independent requirements audit found eleven requirements no check could turn red.** The
  four that could cost the submission outright were added as V54 (F17 on the *training* path —
  V36 only ever scanned `inference.py`), V55 (the repo is genuinely public, proven with every
  credential stripped), V56 (the outputs folder holds actual outputs, not documentation) and
  V59 (the checkpoint is genuinely obtainable). See D27; the remaining seven are tracked in
  `docs/STATE.md`.
- **A second audit found three checks that could not fail** (D31). V28's escape hatch was
  permanently unlocked by boilerplate and it counted a coin-flip SSIM tie as a win; V48 counted
  pipe characters instead of comparing numbers; V00 silently discarded the pin on the
  verification contract itself. All three were tightened, and **V28 went from green to red as a
  direct result** — which is the point of tightening them.
- **V33's pass mark used to live in the file it was grading.** Its thresholds sat in
  `src/degrade.py`, which is not hash-pinned, so they could have been widened without tripping
  the integrity check. They now live in the pinned verifier (D24).

`docs/STATE.md` carries the rolling ledger and `docs/BLOCKERS.md` the things that could not be
resolved. The committed `results/verification_report.json` is a snapshot from commit `c209cd2`,
not a current status.
