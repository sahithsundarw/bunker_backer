# KLA — AI-Based Restoration of Degraded Images for Semiconductor Inspection
## Master Engineering Spec & Build Plan (Claude Code context file)

**Event:** Hackathon 2026, organized as part of SEMICON India 2026
**Track:** Track 1 — KLA (Problem Statement PS01)
**Phase 1 deadline:** 16 August 2026 (registration + initial solution submission, same day)
**Prepared:** 15 August 2026

---

## 0. HOW TO USE THIS FILE

This file is the single source of truth for the project. It is written to be dropped into a Claude Code session as the root context document.

Reading order for the agent:
1. §1 (mission), §2 (hard facts + what is unverified), §3 (timeline pressure).
2. **§5 — run the dataset forensics protocol FIRST.** Do not write a model until §5 is complete. Several load-bearing facts (file format, dtype, scale factor confirmation, noise parameters, degradation order) can only come from the data.
3. §6–§11 (data pipeline → model → loss → training → metrics → inference).
4. §12–§15 (repo contract, README, PPT, checklist).
5. §17 — the executable task list with acceptance criteria. Work through it top to bottom.

**Notation used throughout:**
- `GT` = clean ground-truth image (512×512 or 256×256, values in [0,1])
- `NoisyLR` = degraded input (256×256 or 128×128, values may fall outside [0,1])
- `SF` = scale factor. **Confirmed ×2 in all cases.**

**Rules for the agent:**
- Never invent a fact that §2 marks as UNVERIFIED. Derive it from the dataset and record the derivation in `docs/dataset_findings.md`.
- Every design decision goes in `docs/decisions.md` with a one-line rationale. This feeds the PPT and is directly scored ("Training & compute hygiene").
- The inference script is the highest-value artifact in the repo. Treat it as production code.

---

## 1. MISSION IN ONE PARAGRAPH

Train a single model that maps a degraded grayscale semiconductor inspection image (noisy + downsampled by 2×) to a clean, full-resolution estimate of the ground truth. It must handle speckle noise, additive Gaussian noise and 2× downsampling **simultaneously and blindly** (the order in which they were applied is not disclosed and the model is not required to identify it). It must generalize to image content it has never seen (out-of-distribution semiconductor structures from different sources). It is scored on a fixed, undisclosed weighted blend of PSNR, SSIM and LPIPS against hidden ground truth, **plus** end-to-end wall-clock pipeline throughput on a common NVIDIA H100, **plus** reproducibility and engineering hygiene. Deliverables are a public GitHub repo whose inference script runs as-is with `--input_dir/--output_dir`, plus a PDF slide deck.

**The single biggest failure mode is not model quality — it is an inference script that does not run unmodified on KLA's machine. An unrunnable script cannot be benchmarked, and an unscored submission cannot win.**

---

## 2. FACT TABLE — CONFIRMED vs UNVERIFIED

### 2.1 Confirmed (from the official problem statement page, i4c.in/hackathon-2026, and the supplied problem-statement document)

| # | Fact | Consequence for design |
|---|---|---|
| F1 | Images are **grayscale, single channel**. Colour is explicitly not part of the challenge. | `in_ch = out_ch = 1`. LPIPS needs a 3-channel replication wrapper. Do not use RGB-pretrained SR models without adapting the stem. |
| F2 | Degradation is **exactly 2× downsampling**: 512×512 → 256×256, and 256×256 → 128×128. | Fixed-scale ×2 network. Use one `PixelShuffle(2)` head. No arbitrary-scale machinery needed. Huge simplification — exploit it. |
| F3 | Two noise types: **speckle noise** (multiplicative, signal-dependent, grainy) and **additive Gaussian noise** (described on the official page as causing softness/haze and loss of edge sharpness). | Degradation simulator must model both. Speckle is *signal-dependent* — its variance scales with local intensity. This matters for augmentation realism. |
| F4 | The three degradations may have been applied **in any order**; the order is not disclosed and need not be identified. | Train a blind restorer. Optionally randomize order in synthetic augmentation so the model is order-agnostic. |
| F5 | **GT values are normalized to [0,1]. NoisyLR values may extend slightly outside [0,1]** — this is intentional and caused by speckle pushing pixels past the true signal range. | **Do not clip the input.** Out-of-range values carry information. **Do clip the output to [0,1]** since GT lives there. |
| F6 | **KLA does not clip or renormalize outputs.** Images are scored exactly as saved by your pipeline. | All clipping/normalization/dtype handling must live inside your own code. A single dtype mistake silently destroys your score. |
| F7 | Test set has **in-distribution AND out-of-distribution** content (different source structures). Noise *mechanisms* are the same; sampled *levels* may vary within a similar range. | Optimize for generalization, not leaderboard overfit. Randomize noise levels in training beyond the observed range. Avoid content-memorizing tricks. |
| F8 | Images come from **diverse data origins** — different types of semiconductor structures. | Don't overfit to one texture family. Check per-source validation metrics if source is inferable from filenames/folders. |
| F9 | Scoring = **fixed internal weighted combination of PSNR, SSIM, LPIPS**. Exact weights and axis weightings **not disclosed**. No target score or latency threshold prescribed. | Balance all three. Do not go pure-PSNR (LPIPS suffers), do not go pure-GAN (PSNR/SSIM collapse). See §8. |
| F10 | **End-to-end runtime** includes disk read, preprocessing, CPU→GPU transfer, model forward, GPU→CPU transfer, post-processing and saving. Benchmarked on a common **NVIDIA H100**. | Optimize the whole script, including imports and model-load time. See §11. |
| F11 | Inference script must be a **standalone `.py` (NOT a notebook)** accepting (a) input dir, (b) output dir. Must run **without manual edits**. | Hard requirement. `argparse`. No hardcoded paths. No notebook. |
| F12 | Repo must be **public** and contain: README, evaluation/inference script, training script, trained weights (downloadable — Git LFS / Drive / HuggingFace if large), **restored test outputs folder**, `requirements.txt` (complete `pip freeze`). | Six mandatory items. The "restored test outputs" folder is easy to forget. |
| F13 | PPT submitted as **PDF**, using the official Idea Submission Template, **max 8–9 slides**, instruction slide removed, filename `TeamName_KLA_PS01.pdf` (e.g. `VisionForge_KLA_PS01.pdf`). | The 12-slide structure in the internal doc is a *content guide*; the portal's 9-slide template is the binding format. Map 12 → 9. See §14. |
| F14 | Pretrained open-source weights and public external datasets are **allowed** where licences permit competition use. Must disclose name, link, licence, and paper/model card. | Legal-safe reuse is a genuine advantage. Disclosure is mandatory and scored. |
| F15 | You may **create extra synthetic degraded pairs from the provided GT images**. | This is the main lever for OOD robustness. Do it aggressively (§5.4, §6.3). |
| F16 | No fixed parameter-count limit, but oversized models lose on throughput. | Target a small, fast model. Quality-per-FLOP is the real objective. |
| F17 | Do **not** retrain on hidden test inputs unless a later official instruction explicitly permits it. | No test-time training / self-supervised adaptation on the test set. |
| F18 | Demo video: max 5 minutes, optional-but-recommended (Slide 8 of the template). | Cheap points. Record a 2-minute screen capture of `inference.py` running. |
| F19 | Team size 2–4; UG/PG/PhD from recognised Indian institutions. Total prize pool ₹5,00,000 across both tracks. | Administrative. |

### 2.2 UNVERIFIED — must be resolved from the dataset in the first 15 minutes

| # | Open question | Why it is critical | How to resolve |
|---|---|---|---|
| U1 | **File format and dtype** of GT and NoisyLR (`.png` 8-bit? 16-bit PNG? `.tif` float32? `.npy`?) | If GT is float `.tif`/`.npy` and you save 8-bit PNG, you quantize to 256 levels and **lose several dB of PSNR instantly**. This is the #1 silent killer. | `§5.1` inspection script. Mirror the GT format exactly on output. |
| U2 | Exact **folder names and filename convention** (e.g. `GT/` + `NoisyLR/`, matched basenames? suffixes like `_gt`/`_noisy`?) | Output filenames must match what the evaluator expects. | `§5.1`. Then: **output filename = input filename, byte-identical**, same extension, unless the dataset README says otherwise. |
| U3 | **Dataset size** (number of pairs), and the split of 512-GT vs 256-GT samples. | Determines training budget, whether external data is needed, patch sampling strategy. | `§5.1` |
| U4 | **Downsampling kernel**: bicubic? bilinear? area/box average? Gaussian blur + stride-2 decimation? With or without antialias? | Matching the true kernel in your synthetic-pair generator is worth several dB. Mismatched kernel = domain gap. | `§5.2` kernel identification. |
| U5 | **Degradation order** (empirically inferable even though undisclosed). Noise applied *before* downsampling becomes spatially correlated and variance-reduced; applied *after* it stays white. | Determines whether your synthetic pairs are realistic. | `§5.3` residual autocorrelation test. |
| U6 | **Noise parameter ranges**: speckle variance, Gaussian σ. Fixed or sampled per-image? | Sets the randomization range for augmentation (train slightly wider than observed, per F7). | `§5.2` parameter fitting. |
| U7 | Is there a **dataset README / metadata file** in the Drive folder (`Data-public`)? | May answer U1–U6 outright. | Look before running forensics. |
| U8 | Whether GT/NoisyLR are perfectly **pixel-aligned** (no sub-pixel shift from the resize). | Misalignment caps achievable PSNR and would demand alignment handling. | `§5.2` cross-correlation peak check. |
| U9 | Whether a **separate test-input folder** is shipped now or released later. | Determines what goes in `results/restored_test_outputs/`. | Check Drive; if absent, use your held-out validation split's degraded inputs and label it clearly. |

**Agent instruction:** write findings for U1–U9 into `docs/dataset_findings.md` as they are resolved, with the numeric evidence. This document doubles as PPT Slide 3 content.

### 2.3 Official links

| Resource | URL |
|---|---|
| Hackathon landing page / problem statement | `https://i4c.in/hackathon-2026/` |
| Registration portal | `https://hackathon2026.i4c.in/` |
| **Dataset (Google Drive, "Data-public")** | `https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link` |
| Detailed KLA problem-statement explanation PPTX | `https://i4c.in/wp-content/uploads/2026/08/7b675083-e081-47d3-8c55-fde76a77b673.pptx` |
| Idea Submission Template (PPTX) | `https://i4c.in/wp-content/uploads/2026/07/Idea-Submission-Template_Hackathon-2026-1.pptx` |
| Webinar 1 — KLA Problem Statement Explanation (30 Jul 2026) | `https://youtu.be/RMSDaviTOIw` |
| Webinar 2 — KLA Knowledge & Q&A (7 Aug 2026) | `https://www.youtube.com/watch?v=Q__rlK1Q3uw` |
| Registration process doc | `https://i4c.in/wp-content/uploads/2026/01/How-to-register-for-IESA-Hackathon-2026.pdf` |
| Support | `support@i4c.in` / +91 98504 58254 |
| WhatsApp updates group | `https://chat.whatsapp.com/D9QI2JRBTTO5BUw57Y71wC` |

**Before submitting, re-open every one of these.** Portal instructions supersede this document if they conflict.

### 2.4 Named contacts (useful for framing the deck)

- **Akshat Singh** — ML Research Engineer, KLA. Ran the Knowledge & Q&A session. Public specialization: vision foundation models, self-supervised learning, **model compression, quantization, efficient GPU inference**.
  - *Read the signal:* the person most likely to review your submission specializes in efficient GPU inference. Your throughput engineering section is not a formality — make it the differentiator. Quantization/compilation/TensorRT discussion will land well.

---

## 3. TIMELINE

| Date | Milestone |
|---|---|
| 24 Jul 2026 | Registration opened |
| 30 Jul 2026 | Webinar 1 — KLA problem statement |
| 07 Aug 2026 | Webinar 2 — KLA key concepts & Q&A |
| 12 Aug 2026 | Webinar 5 — submission guidelines & Q&A |
| **16 Aug 2026** | **Registration closes + Phase 1 submission deadline** |
| 17–26 Aug 2026 | Round 1 evaluation |
| 27 Aug 2026 | Top 30 announced; Round 2 brief released |
| 28 Aug – 04 Sep 2026 | Semifinal Round 2 development + submission |
| 05 Sep 2026 | Semifinal evaluation |
| 06 Sep 2026 | Top 10 finalists announced |
| 07–12 Sep 2026 | Finalist mentoring |
| 17 Sep 2026 | Grand Finale, Yashobhoomi (IICC), Dwarka, New Delhi |
| 18 Sep 2026 | Winners announced |

**Implication: at the time of writing there is roughly one working day left before Phase 1 closes.** §16 contains a compressed one-day plan. Optimize for *submitted and runnable* over *maximally accurate*. Round 2 (28 Aug – 4 Sep) is where the heavy modelling investment pays off.

---

## 4. HOW THE SCORE IS ACTUALLY DECIDED

Three axes, weights undisclosed:

1. **Restoration quality** — fixed internal blend of PSNR + SSIM + LPIPS on hidden GT, across in-distribution and OOD content.
2. **End-to-end throughput** — total pipeline wall-clock on a common H100, including I/O and pre/post-processing.
3. **Training & compute hygiene** — reproducibility, clean experiment design, environment spec, code quality, efficient data pipeline, standard ML practice.

**Strategic reading:**
- Axis 3 is fully within your control and costs no GPU hours. Nail it. Most student submissions lose here.
- Axis 2 rewards small models and a well-engineered I/O path far more than exotic architectures. A 2M-parameter model that runs in 8 s can beat a 60M-parameter model that runs in 200 s.
- Axis 1 with an *undisclosed blend* argues for a balanced loss, not a metric-specific one. Assume roughly equal weighting and refuse to sacrifice any single metric badly.
- OOD is explicitly tested. Degradation randomization + external clean data + CutBlur-style augmentation are the highest-leverage moves.

---

## 5. DATASET FORENSICS PROTOCOL — DO THIS FIRST

Create `scripts/inspect_dataset.py`. It should be idempotent and dump both a printed report and figures under `results/eda/`.

### 5.1 Inventory (answers U1, U2, U3, U7)

```python
# scripts/inspect_dataset.py  — sketch
# 1. Walk the dataset root. Print the full directory tree (depth 3) and file counts by extension.
# 2. Look for README / metadata / .txt / .json / .csv at any level. Print them in full.
# 3. For 5 random GT files and 5 random NoisyLR files, report:
#      path, extension, PIL/tifffile/cv2 loader that works, numpy dtype, shape,
#      min, max, mean, std, number of unique values, whether values fall outside [0,1]
# 4. Establish the GT<->NoisyLR pairing rule. Print 10 matched basename pairs.
# 5. Histogram of (GT height, GT width) and (LR height, LR width). Confirm SF==2 for every pair.
# 6. Count pairs. Report split of 512-GT vs 256-GT.
# 7. Assert: for every pair, GT.shape == 2 * LR.shape. Log any violations loudly.
```

**Loader guidance by extension:**
- `.npy` → `np.load` (already float; check dtype)
- `.tif`/`.tiff` → `tifffile.imread` (handles float32 and uint16 correctly; **do not** use PIL for float TIFF)
- `.png` → `cv2.imread(path, cv2.IMREAD_UNCHANGED)` — `IMREAD_UNCHANGED` preserves 16-bit and prevents silent BGR conversion. Then divide by 255 or 65535 based on dtype.
- Never use plain `cv2.imread(path)` — it forces 3-channel uint8.

**Then decide the output contract and write it to `docs/io_contract.md`:**
> Output format = exactly the GT format. Output dtype = exactly the GT dtype. Output filename = exactly the input filename. Output size = 2× input size. Values clipped to [0,1] before dtype conversion.

If GT is float32 `.tif` or `.npy`, **saving PNG is a scoring catastrophe.** If GT is 8-bit PNG, saving float TIFF may make files unreadable by the evaluator. Mirror, do not improvise.

### 5.2 Degradation parameter fitting (answers U4, U6, U8)

For a sample of ~50 pairs:

```
For each candidate downsample kernel K in
    {bicubic(antialias=T/F), bilinear(antialias=T/F), area/box, gaussian(sigma in 0.5..1.5)+stride2, nearest}:
    LR_hat = K(GT)                       # noiseless prediction of the LR signal
    residual = NoisyLR - LR_hat
    record: std(residual), and corr(residual^2, LR_hat)  # speckle signature
Pick K minimizing std(residual).
```

- **Alignment check (U8):** cross-correlate `LR_hat` with `NoisyLR`; the peak must be at (0,0). A peak at a half-pixel offset means the resize convention differs (e.g. `align_corners`) — fix the kernel rather than the model.
- **Speckle vs Gaussian separation (U6):** bin pixels by `LR_hat` intensity. Plot `var(residual)` per intensity bin against bin centre.
  - Pure additive Gaussian → flat line at `σ²`.
  - Pure speckle (`y = x + n·x`, `n ~ N(0, v)`) → variance grows as `v · x²`, i.e. a parabola through the origin.
  - Both → `var(residual | x) = σ² + v·x²`. **Least-squares fit this quadratic to recover `σ²` (intercept) and `v` (curvature) directly.** Report both.
- Repeat per image to see whether `σ` and `v` are fixed globally or sampled per-image. Report the min/max range — this is your augmentation range (then widen it ~20–30% per F7).
- Save the scatter + fitted curve to `results/eda/noise_variance_vs_intensity.png`. **This figure alone is a strong PPT Slide 3.**

### 5.3 Degradation order test (answers U5)

The order is undisclosed but leaves a fingerprint:

- **Noise added AFTER downsampling** → residual is spatially **white**: normalized autocorrelation of the residual is ~1 at lag 0 and ~0 at lag 1.
- **Noise added BEFORE downsampling** → the 2× decimation averages neighbouring noise samples, so the residual is **spatially correlated** (non-trivial autocorrelation at lag 1) and its variance is **reduced by roughly the kernel's effective averaging factor** (≈4× for box, less for bicubic).

Compute the 2-D autocorrelation of the residual, print `r(0,1)`, `r(1,0)`, `r(1,1)`. Record the conclusion. Even if ambiguous, **randomizing the order in your synthetic generator makes the model robust to either case** — which is the safe play.

### 5.4 Visual audit (mandatory, not optional)

Save a grid of 12 triplets `[NoisyLR (nearest-upscaled 2× for display) | GT | absolute difference]` at full resolution to `results/eda/pairs_grid.png`. Look for:
- Structure families present (line/space arrays, contact holes, dense periodic arrays, amorphous regions, defects).
- Whether periodic patterns risk aliasing at 2× downsample — **aliased periodic structures are genuinely unrecoverable and will be your failure case.** Identify one now; you need an honest failure case for the deck (§14, Slide 6).
- Edge-brightening / SEM-like contrast behaviour.

---

## 6. DATA PIPELINE

### 6.1 Splits

- Split **by image, and if source/structure type is inferable from filenames or subfolders, split by source group** so validation measures generalization rather than memorization. Record the split as an explicit file list in `configs/split_val.txt`, committed to the repo. Never regenerate it randomly at train time — that is leakage.
- Suggested: 90% train / 10% validation. With a small dataset, 85/15.
- Hold out one *entire structure family* if more than three are present — this is your **proxy OOD set** and lets you report an honest OOD number in the deck. This is a strong differentiator; nobody else will do it.

### 6.2 Patch sampling

- Train on LR patches of 64×64 → GT patches of 128×128 (paired crop, LR crop origin `(i,j)` ⇒ GT origin `(2i,2j)`).
- Consider raising to 96→192 in later epochs; larger patches improve transformer/large-receptive-field models.
- Pre-load the whole dataset into RAM as a list of numpy arrays if it fits (likely — a few thousand 512×512 float32 images is a few GB). This eliminates the data-loader bottleneck entirely and is the single easiest "efficient data pipeline" win for Axis 3.
- `num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4`.

### 6.3 Augmentation

**Use:**
- Random horizontal/vertical flip and 90° rotations (dihedral group, 8 orientations). Apply identically to LR and GT.
- **CutBlur** (Yoo et al., CVPR 2020, "Rethinking Data Augmentation for Image Super-Resolution"): paste a random rectangle of the upscaled-LR into the HR (and vice versa). Explicitly designed for SR, improves generalization and prevents over-sharpening. Strongly recommended.
- **Synthetic pair generation from GT (F15).** For each GT, generate fresh NoisyLR on the fly with parameters randomized around the fitted range:
  ```
  order = random.choice([['down','speckle','gauss'],
                         ['speckle','gauss','down'],
                         ['speckle','down','gauss'], ...])   # randomize per F4
  v     ~ U(v_min*0.8, v_max*1.3)      # speckle variance, from §5.2
  sigma ~ U(s_min*0.8, s_max*1.3)      # gaussian sigma, from §5.2
  kernel~ choice([fitted_kernel]*4 + [bicubic, bilinear, area])   # mostly true kernel, some diversity
  ```
  Mix **real pairs and synthetic pairs** (e.g. 50/50). Real pairs anchor the true degradation; synthetic pairs provide the OOD robustness.
- **Do not clip synthetic LR to [0,1]** — matching F5 is essential, otherwise your training inputs have a different distribution from the test inputs.

**Avoid:**
- Colour jitter (grayscale; absolute intensity is physically meaningful).
- Aggressive gamma/contrast changes (may break the [0,1] GT convention).
- Random rescaling of GT (changes the effective SF).
- MixUp on whole images (blends unrelated structures; hurts fidelity metrics).

### 6.4 Speckle noise reference implementation

```python
def add_speckle(x, var):
    # MATLAB imnoise('speckle') convention: y = x + n*x, n ~ N(0, var)
    n = np.random.randn(*x.shape).astype(np.float32) * np.sqrt(var)
    return x + n * x          # NOTE: not clipped — matches F5

def add_gaussian(x, sigma):
    return x + np.random.randn(*x.shape).astype(np.float32) * sigma
```
If §5.2 shows the variance-vs-intensity curve is *linear* rather than *quadratic* in `x`, the underlying model is Poisson/shot noise rather than multiplicative speckle — adapt accordingly and document it. Note this possibility; do not assume.

### 6.5 External data (optional, licence-gated)

Permitted per F14 but **every use must be disclosed with name, link, licence and paper/model card** in the README and the deck. Candidates, in priority order:

| Dataset | Why | Licence note |
|---|---|---|
| The provided GT images, re-degraded (§6.3) | Perfect domain match, zero licence risk | N/A — provided |
| DIV2K / Flickr2K (converted to grayscale) | Large, high-quality, standard for SR | Research-use licence — **verify it permits competition use before including** |
| BSD400 / Waterloo Exploration | Classic denoising corpora | Verify terms |
| Public SEM / electron-microscopy image sets on Zenodo/Kaggle | Closest texture statistics to semiconductor imagery | Per-dataset; many are CC-BY |

**Judgement:** with a one-day budget, external data is a distraction. Skip it for Phase 1, note it as future work, revisit for Round 2. If you skip it, say so explicitly in the deck's external-resources slide ("no external datasets or pretrained weights used") — an honest empty disclosure scores better than a vague one.

---

## 7. MODEL ARCHITECTURE

### 7.1 Recommended: NAFNet-style restoration body at LR resolution + PixelShuffle ×2 head

Rationale (put this verbatim-ish in the deck):
- Because SF is fixed at ×2 (F2), almost all computation should happen at **LR resolution**, where it is 4× cheaper. Only the final upsampling layer works at HR. This is the standard efficient-SR design and directly serves Axis 2.
- **NAFNet** (Chen et al., ECCV 2022, "Simple Baselines for Image Restoration") uses no activation functions — a SimpleGate (channel-split elementwise product) and Simplified Channel Attention. It is fast, GPU-friendly, has strong PSNR/SSIM on denoising and deblurring, and its blocks are trivially fusible/compilable. Excellent quality-per-FLOP.
- A **global residual connection** — bilinear-upsample the input and add it to the network output — means the network only learns the residual detail. This converges faster and buys 0.3–0.8 dB in practice.

```
Input  (B,1,h,w) float32, values may lie outside [0,1] — DO NOT CLIP
   │
   ├─ x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)   ── skip
   │
   ├─ Conv3x3(1 → C)                                            C = 48 (or 64)
   ├─ [NAFBlock] × N                                            N = 16 (LR resolution)
   │      NAFBlock = LN → Conv1x1 → DWConv3x3 → SimpleGate → SCA → Conv1x1 → (+res, layerscale)
   │                 LN → Conv1x1 → SimpleGate → Conv1x1 → (+res, layerscale)
   ├─ Conv3x3(C → C)  + long skip from post-stem features
   ├─ Conv3x3(C → 4C) → PixelShuffle(2) → Conv3x3(C → 1)
   │
   └─ out = body_out + x_up
Output (B,1,2h,2w)
Clip to [0,1] at SAVE time only (not inside the loss).
```

Parameter budget: ~1–3 M. This is deliberately small. On an H100 with bf16 this processes 128×128→256×256 in well under a millisecond per image amortized.

### 7.2 Alternatives, with honest trade-offs

| Architecture | Quality | Speed | Verdict |
|---|---|---|---|
| **NAFNet-body + PixelShuffle (above)** | High | Very fast | **Recommended default** |
| Plain U-Net (enc-dec, 4 levels) + PixelShuffle | Medium-high | Fast | Good fallback; simple, robust. Use as the *baseline* required by the rubric. |
| RRDB / ESRGAN generator (PSNR-oriented, no GAN) | High | Medium | Solid but heavier; residual-in-residual dense blocks are memory-hungry. |
| SwinIR-light / HAT-S | Highest PSNR/SSIM | Slow | Wins Axis 1, loses Axis 2. Only if Axis 2 is lightly weighted — and you don't know the weights. Risky. |
| Restormer / Uformer | Very high | Slow at 512 | Same trade-off. Transposed-attention scales better than window attention but still costly. |
| Algorithm-unrolling (e.g. ISTA-Net-style, per the KLA-supplied Monga et al. reference) | Medium-high | Medium | **Interpretability angle.** KLA explicitly cited the algorithm-unrolling survey — a hybrid unrolled + learned-prior design would score well on "Innovation & Uniqueness" (Slide 5). Consider as a differentiator if time permits. |
| GAN / diffusion-based SR | Best LPIPS, worst PSNR | Very slow | **Avoid.** Hallucination is explicitly penalized ("without hallucinating or destroying real structure") and inventing a defect that isn't there is the worst possible failure in semiconductor inspection. Say this in the deck — it shows domain understanding. |

### 7.3 The two-input-size problem

Test inputs will be a mix of 128×128 and 256×256 (F2). A fully-convolutional network handles both natively — **but for batched inference you cannot stack different sizes in one tensor.**

Solution: **group files by resolution and batch within each group.** Simple, exact, no padding waste. Implement in `inference.py` (§11).

If a stray non-power-of-two size appears, pad reflectively to a multiple of the network's downsampling factor and crop the output back. Since the recommended body runs at a single resolution with no downsampling, the required multiple is 1 — another argument for the flat (non-U-Net) design.

### 7.4 Optional: staged approach

The problem statement permits one-step or staged restoration. A staged design (denoise at LR → then super-resolve) is more interpretable but costs two forward passes and compounds errors. **One-step joint restoration is recommended**; mention the staged alternative in the deck as a considered-and-rejected option with rationale. Evaluators like seeing rejected alternatives.

---

## 8. LOSS FUNCTION

The scoring blend is undisclosed (F9), so the loss must be balanced.

```
L_total = 1.0 · L_charbonnier
        + 0.15 · (1 − MS-SSIM)
        + 0.05 · L_fft
        + 0.02 · L_lpips        # enable only after warmup, see below
```

**Components:**

1. **Charbonnier** (`sqrt((y − ŷ)² + ε²)`, ε = 1e-3): a smooth L1. Better than pure L2 for restoration — L2 over-smooths and L1 has a non-differentiable kink. Drives PSNR.
2. **MS-SSIM loss** (`1 − MS_SSIM(y, ŷ)`): directly optimizes the structural metric. Use `pytorch-msssim`. At 128×128 patches, MS-SSIM's 5 scales need ≥161px — either use **single-scale SSIM loss** at 128 patches or raise patch size to 192+. Check this or you'll get a runtime error.
3. **FFT / frequency loss** (`L1` between the FFT magnitudes of prediction and target): penalizes missing high-frequency content, which is exactly what downsampling destroyed. Cheap and effective for sharpness without hallucination. (The problem statement explicitly permits frequency-domain methods.)
4. **LPIPS** as a *small* auxiliary term: improves the perceptual metric, but a large weight tanks PSNR. Enable only after ~50% of training. Grayscale needs `x.repeat(1,3,1,1)` and scaling to [-1,1]. Adds notable training cost — if time-constrained, drop it and rely on the FFT term; note the decision.

**Do not use adversarial loss.** Justification in §7.2.

**Clipping and the loss:** compute the loss on the *unclipped* network output so gradients flow normally. Clip only when saving. If you clip inside training, pixels that saturate get zero gradient.

**Ablation to report (rubric requires baseline comparison):** train the same architecture with (a) L1 only, (b) L1 + SSIM, (c) full loss. Report PSNR/SSIM/LPIPS for each. Even a 1-epoch version of this ablation is worth reporting honestly.

---

## 9. TRAINING RECIPE

```yaml
# configs/nafnet_x2.yaml
model:
  name: NAFSR
  width: 48
  num_blocks: 16
  scale: 2
data:
  lr_patch: 64            # -> 128 GT patch
  batch_size: 32
  synth_ratio: 0.5        # fraction of samples generated on the fly from GT
optim:
  optimizer: AdamW
  lr: 1.0e-3
  betas: [0.9, 0.9]       # NAFNet's beta2=0.9 (not 0.999) — matters
  weight_decay: 0.0
  scheduler: cosine
  min_lr: 1.0e-6
  warmup_iters: 500
  total_iters: 200000     # scale down hard for a 1-day budget: 15k-30k
train:
  amp: bf16               # H100/A100. Use fp16+GradScaler on older cards.
  grad_clip: 1.0
  ema_decay: 0.999
  seed: 42
  channels_last: true
  val_every: 1000
  save_best_on: psnr      # also log ssim, lpips
```

**Non-negotiables for Axis 3 (hygiene):**
- Seed everything: `random`, `numpy`, `torch`, `torch.cuda`, and set `torch.backends.cudnn.deterministic` where it doesn't destroy throughput (document the trade-off).
- Log every run to `results/experiments.csv`: run id, git commit hash, config path, seed, best PSNR/SSIM/LPIPS, wall-clock, checkpoint path. A CSV is sufficient — W&B/TensorBoard is nicer but a committed CSV is verifiable by an evaluator with no account.
- Save `configs/final.yaml` alongside the checkpoint, and store the config **inside** the checkpoint dict so weights can never be paired with the wrong architecture.
- Checkpoint dict: `{'model': state_dict, 'ema': ema_state_dict, 'config': cfg, 'iter': it, 'metrics': {...}, 'git': sha}`.
- **Use the EMA weights for the final submission** — almost always +0.1–0.3 dB free.

**Compressed schedule for a one-day budget:** 15k–25k iterations at batch 32 on a single modern GPU is enough to be clearly better than a bilinear/bicubic baseline. Train the small model, checkpoint every 1000 iters, and be prepared to ship whatever is best at the cutoff. Start the training run early and write the repo/README/deck *while it trains*.

---

## 10. VALIDATION & METRICS — EXACT IMPLEMENTATIONS

Metric implementations differ between libraries by non-trivial margins. **Pin your implementation and state it in the deck**, otherwise your reported numbers aren't comparable to anything.

```python
# src/metrics.py
import numpy as np, torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import lpips

def psnr(pred, gt):
    # pred, gt: float32 HxW in [0,1], pred ALREADY clipped
    return sk_psnr(gt, pred, data_range=1.0)

def ssim(pred, gt):
    # Wang et al. (2004) reference settings — state these in the deck
    return sk_ssim(gt, pred, data_range=1.0,
                   gaussian_weights=True, sigma=1.5,
                   use_sample_covariance=False)

_lp = None
def lpips_score(pred, gt, device='cuda'):
    global _lp
    if _lp is None:
        _lp = lpips.LPIPS(net='alex').to(device).eval()   # AlexNet is the standard reporting backbone
    def prep(a):
        t = torch.from_numpy(a)[None, None].to(device)     # 1,1,H,W
        t = t.repeat(1, 3, 1, 1)                           # grayscale -> 3ch
        return t * 2.0 - 1.0                               # [0,1] -> [-1,1]
    with torch.no_grad():
        return _lp(prep(pred), prep(gt)).item()
```

**Reporting requirements:**
- Report **mean and standard deviation** of each metric over the validation set, plus per-resolution breakdown (128→256 vs 256→512) and per-structure-family breakdown if available.
- Report the **proxy-OOD** split separately (§6.1). This is the honest generalization number.
- Metrics are computed on the **clipped, dtype-converted, saved-to-disk** image, not the raw tensor. Reload from disk and score that. This catches dtype/quantization bugs before KLA does.

**Mandatory baselines (rubric requires ≥1; give 3, they're nearly free):**
1. **Bicubic ×2 upsample** of the raw NoisyLR (no denoising). The floor.
2. **Median/BM3D-style denoise → bicubic ×2.** Classical pipeline.
3. **Small plain U-Net**, same training budget. The learned baseline.
4. Your final model.

A table with four rows and three metric columns, plus a runtime column, is exactly what Slide 6 needs.

---

## 11. INFERENCE PIPELINE & THROUGHPUT ENGINEERING

**This is the file KLA runs as-is. It decides whether you are scored at all.**

### 11.1 Contract

```bash
python inference.py --input_dir /path/to/degraded --output_dir /path/to/restored
```

- Must create `--output_dir` if it does not exist.
- Must process **every** image file in `--input_dir` (glob a permissive extension set: `.png .tif .tiff .npy .jpg .jpeg .bmp`, case-insensitive; also handle subdirectories — mirror the structure).
- Must write **one output per input, with the identical filename and extension**.
- Must find its own weights via a **path relative to the script file** (`Path(__file__).parent / "weights" / "best.pt"`), never a hardcoded absolute path, never a CWD-relative path. Provide `--weights` as an optional override.
- Must default to CUDA if available and **fall back to CPU without crashing**.
- Must exit non-zero with a clear message on failure, and must not crash on one bad file — log and continue.
- Optional flags (all with sensible defaults so the two-arg invocation works): `--batch_size`, `--device`, `--precision {bf16,fp16,fp32}`, `--compile`, `--num_workers`.

### 11.2 Speed checklist (Axis 2)

| Lever | Action | Typical impact |
|---|---|---|
| **Import cost** | `inference.py` imports **only** `argparse, os, pathlib, time, numpy, torch, cv2/tifffile`. **No `lpips`, no `skimage`, no `matplotlib`, no `pandas`, no `wandb`.** Heavy imports can cost seconds of the measured wall-clock. | Seconds |
| **Model load** | `torch.load(..., map_location='cuda', weights_only=True)`. Single small checkpoint. Strip optimizer state from the shipped weights. | 0.1–1 s |
| **Precision** | `torch.autocast('cuda', dtype=torch.bfloat16)`. bf16 on H100 is ~2× fp32 with no scaler and no overflow risk. | 1.5–2× |
| **Memory format** | `model.to(memory_format=torch.channels_last)`, inputs likewise. | 10–30% |
| **Batching** | Group inputs by (H,W), batch within group. Batch 16–64 on H100 at these sizes. | Large |
| **Parallel disk read** | `torch.utils.data.DataLoader` over a trivial Dataset with `num_workers=8, pin_memory=True`. Overlaps decode with compute. | Large on many files |
| **H2D transfer** | Pinned memory + `.to('cuda', non_blocking=True)`. | Moderate |
| **Parallel disk write** | `concurrent.futures.ThreadPoolExecutor(max_workers=8)` for `imwrite` — encoding releases the GIL. | Large on many files |
| **cuDNN autotune** | `torch.backends.cudnn.benchmark = True` (fixed shapes per group → autotune pays off). | 5–15% |
| **TF32** | `torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True`. | Free |
| **`torch.compile`** | Real gains (20–40%) **but** costs 30–120 s of one-time compilation, which is inside the measured window. **Default OFF; expose `--compile` and document the crossover point** (roughly: worth it above ~2000 images). Always keep the eager path working. | Situational |
| **TensorRT / ONNX** | Best raw latency, but a fragile dependency on the evaluator's machine. **Do not make it the default path.** If you ship it, auto-detect and silently fall back to PyTorch. | Risky |
| **`torch.inference_mode()`** | Not `no_grad()` — `inference_mode` also skips version counting. | Small, free |
| **Warmup** | One dummy forward per shape group before timing-critical work, to trigger cuDNN autotune outside the main loop. Note: this is still inside KLA's measured window, so keep it to one iteration. | — |

### 11.3 Skeleton

```python
#!/usr/bin/env python3
"""Standalone inference: restore degraded semiconductor images (2x SR + denoise)."""
import argparse, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from src.model import build_model            # keep this import light

EXTS = {".png", ".tif", ".tiff", ".npy", ".jpg", ".jpeg", ".bmp"}

def load_image(p: Path):
    """Return (float32 HxW array, meta) where meta lets us save identically."""
    if p.suffix.lower() == ".npy":
        a = np.load(p).astype(np.float32)
        return a, {"kind": "npy", "dtype": a.dtype}
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError(f"unreadable: {p}")
    if a.ndim == 3:
        a = a[..., 0]                        # already grayscale content
    dt = a.dtype
    if dt == np.uint8:    f = a.astype(np.float32) / 255.0
    elif dt == np.uint16: f = a.astype(np.float32) / 65535.0
    else:                 f = a.astype(np.float32)   # float tif: already normalized
    return f, {"kind": "img", "dtype": dt}

def save_image(p: Path, arr01: np.ndarray, meta: dict):
    arr01 = np.clip(arr01, 0.0, 1.0)         # GT lives in [0,1]; KLA does not clip for us
    if meta["kind"] == "npy":
        np.save(p, arr01.astype(meta["dtype"])); return
    dt = meta["dtype"]
    if dt == np.uint8:    out = (arr01 * 255.0   + 0.5).astype(np.uint8)
    elif dt == np.uint16: out = (arr01 * 65535.0 + 0.5).astype(np.uint16)
    else:                 out = arr01.astype(dt)
    cv2.imwrite(str(p), out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights",    default=str(SCRIPT_DIR / "weights" / "best.pt"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision",  default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile",    action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"No images found in {in_dir}", file=sys.stderr); sys.exit(1)

    dev = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt = torch.load(args.weights, map_location=dev, weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt.get("ema") or ckpt["model"])
    model.eval().to(dev).to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, mode="max-autotune")

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    use_amp = (dev.type == "cuda" and args.precision != "fp32")

    # --- group by shape so we can batch ---
    loaded = [(p,) + load_image(p) for p in files]
    groups = {}
    for p, arr, meta in loaded:
        groups.setdefault(arr.shape, []).append((p, arr, meta))

    pool = ThreadPoolExecutor(max_workers=8)
    n = 0
    with torch.inference_mode():
        for shape, items in groups.items():
            for i in range(0, len(items), args.batch_size):
                chunk = items[i:i + args.batch_size]
                x = np.stack([a for _, a, _ in chunk])[:, None]           # B,1,H,W
                t = torch.from_numpy(x).pin_memory().to(dev, non_blocking=True)
                t = t.to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    y = model(t)
                y = y.float().clamp_(0, 1).cpu().numpy()[:, 0]
                for (p, _, meta), o in zip(chunk, y):
                    rel = p.relative_to(in_dir)
                    dst = out_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    pool.submit(save_image, dst, o, meta)
                    n += 1
    pool.shutdown(wait=True)
    dt = time.perf_counter() - t0
    print(f"Restored {n} images in {dt:.2f}s ({n/dt:.2f} img/s) | device={dev} precision={args.precision}")

if __name__ == "__main__":
    main()
```

> Note the sketch loads all images before batching. If the test set is large, swap the eager load for a `DataLoader` with workers so decode overlaps GPU compute. Decide based on the observed dataset size; document the choice.

### 11.4 Pre-submission validation of the inference script

Run all of these and record the results:

```bash
# 1. Clean-room test — fresh venv, fresh clone, nothing cached
git clone <your-repo> /tmp/clean && cd /tmp/clean
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py --input_dir ./sample_inputs --output_dir /tmp/out

# 2. Run from a DIFFERENT working directory (catches relative-path bugs)
cd / && python /tmp/clean/inference.py --input_dir /tmp/clean/sample_inputs --output_dir /tmp/out2

# 3. Output integrity
#    - filename set identical to input filename set
#    - every output is exactly 2x the input dimensions
#    - dtype and extension match the GT convention from docs/io_contract.md
#    - no NaN/Inf; min>=0; max<=1 (after de-scaling)

# 4. Mixed-resolution test — a folder with both 128x128 and 256x256 inputs
# 5. Single-image test — a folder with exactly one image (batching edge case)
# 6. CPU fallback  — CUDA_VISIBLE_DEVICES="" python inference.py ...
# 7. Timing — report wall clock, image count, img/s, batch size, GPU model, driver,
#    CUDA version, torch version, and the exact timing method (time.perf_counter around main()).
```

---

## 12. REPOSITORY STRUCTURE

```
repository/
├── README.md                      # setup + exact commands + I/O contract + results table
├── requirements.txt               # full `pip freeze`, pinned versions
├── inference.py                   # THE evaluation script (standalone, argparse, no edits needed)
├── train.py                       # reproduces the submitted checkpoint
├── configs/
│   ├── nafnet_x2.yaml             # final config
│   ├── baseline_unet.yaml
│   └── split_val.txt              # committed validation file list (no leakage)
├── src/
│   ├── __init__.py
│   ├── model.py                   # build_model(cfg) -> nn.Module ; NAFSR + UNet baseline
│   ├── blocks.py                  # NAFBlock, SimpleGate, SCA, PixelShuffle head
│   ├── dataset.py                 # paired loader, patch sampler, synth-degradation, CutBlur
│   ├── degrade.py                 # speckle / gaussian / downsample, order randomization
│   ├── losses.py                  # Charbonnier, SSIM, FFT, LPIPS wrapper
│   ├── metrics.py                 # PSNR / SSIM / LPIPS (pinned settings)
│   └── utils.py                   # seeding, EMA, checkpoint io, logging
├── scripts/
│   ├── inspect_dataset.py         # §5 forensics
│   ├── fit_degradation.py         # §5.2 parameter fitting
│   ├── evaluate.py                # scores a restored dir against a GT dir
│   ├── make_baselines.py          # bicubic / classical / U-Net baselines
│   └── benchmark_runtime.py       # end-to-end timing harness
├── weights/
│   ├── best.pt                    # final checkpoint (model + ema + config + metrics + git sha)
│   └── README.md                  # download link + sha256 if hosted externally (Git LFS/Drive/HF)
├── results/
│   ├── eda/                       # dataset figures from §5
│   ├── metrics_summary.md         # the baseline-vs-final table
│   ├── experiments.csv            # every run: id, commit, config, seed, metrics, wall-clock
│   ├── qualitative/               # full-resolution success AND failure triplets
│   └── restored_test_outputs/     # MANDATORY (F12) — actual model outputs
├── docs/
│   ├── dataset_findings.md        # answers to U1-U9 with evidence
│   ├── io_contract.md             # format/dtype/naming rules
│   └── decisions.md               # design decisions + rationale (feeds the deck)
└── sample_inputs/                 # 4-6 small degraded images so a reviewer can run inference in 10s
```

`sample_inputs/` is not required but is the highest-ROI 10 minutes in the whole project: it lets a reviewer verify your script works without downloading the dataset.

---

## 13. README TEMPLATE

The README must let a reviewer clone and run inference **without contacting you**. Structure:

```markdown
# <Team Name> — AI-Based Restoration of Degraded Images for Semiconductor Inspection
KLA Problem Statement PS01 | SEMICON India Hackathon 2026

## Result summary
| Method | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | End-to-end (img/s) |
|---|---|---|---|---|
| Bicubic ×2 (no denoise)   | ... | ... | ... | ... |
| Classical denoise + bicubic| ... | ... | ... | ... |
| U-Net baseline            | ... | ... | ... | ... |
| **Ours (NAFSR-x2)**       | ... | ... | ... | ... |
Validation split: <N> images, held out per configs/split_val.txt. Proxy-OOD split reported separately below.

## Environment
- OS, Python 3.x, CUDA x.x, PyTorch x.x, GPU used for training and for timing
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Inference (the command KLA will run)
```bash
python inference.py --input_dir <degraded_images_dir> --output_dir <restored_output_dir>
```
Weights are loaded automatically from `weights/best.pt` (relative to the script). No edits required.
Quick check: `python inference.py --input_dir sample_inputs --output_dir /tmp/out`

## Input / output contract
- Input: grayscale, 128×128 or 256×256, <format>, values MAY lie outside [0,1].
- Output: grayscale, exactly 2× input size, <same format/dtype as GT>, clipped to [0,1].
- Output filename is byte-identical to the input filename. Subdirectory structure is mirrored.

## Training (reproduces weights/best.pt)
```bash
python train.py --config configs/nafnet_x2.yaml --data_root <dataset_root>
```
Seed 42, <N> iterations, ~<H> hours on <GPU>. Checkpoints to weights/.

## Repository map
<one line per top-level item>

## Method summary
<6-10 lines: degradation analysis, architecture, losses, augmentation, why>

## Assumptions
<explicitly list every assumption made about the data, e.g. the fitted downsample kernel>

## External resources & licences
<name | link | licence | paper/model card>   — or: "None used."

## Runtime measurement
Hardware, batch size, precision, timing method (time.perf_counter around the full main()),
number of images, total seconds, images/second.
```

---

## 14. PRESENTATION PLAN

**Binding format (F13):** official Idea Submission Template, **max 8–9 slides**, instruction slide removed, exported to **PDF**, named `TeamName_KLA_PS01.pdf`.

The 12-slide structure in the internal problem-statement doc is a *content checklist*, not the submission format. Map it onto the 9 template slides:

| Slide | Template heading | What to put on it |
|---|---|---|
| 1 | Team Details | Team name, 2–4 members with roles (e.g. modelling / data & augmentation / inference optimization / evaluation & docs), college, contacts. |
| 2 | Problem Statement Addressed | "AI-Based Restoration of Degraded Images." In your own words: why a single noisy pixel or lost detail can hide a defect and cost a die. Name the three degradations and the ×2 scale. |
| 3 | Idea Description | Dataset analysis + core concept. **Put the variance-vs-intensity figure from §5.2 here** — it proves you characterized speckle vs Gaussian empirically rather than guessing. State the fitted kernel and the degradation-order finding. Then: one-step blind joint restoration, all compute at LR, ×2 PixelShuffle head. |
| 4 | Proposed Solution | Pipeline diagram (input → load → batch-by-shape → bf16 forward → clip → save) + architecture block diagram + loss composition + augmentation list (dihedral, CutBlur, on-the-fly synthetic re-degradation with randomized order/levels). |
| 5 | Innovation & Uniqueness | Pick 3, no more: (a) empirical degradation forensics driving a matched synthetic-pair generator; (b) the balanced fidelity+structure+frequency loss and the explicit **no-GAN / no-hallucination** decision, justified by inspection semantics; (c) the throughput-engineered inference path (grouped batching, bf16, channels_last, threaded I/O, optional compile) with the measured breakdown. |
| 6 | Results | The 4-row baseline table (PSNR/SSIM/LPIPS + runtime). In-distribution vs proxy-OOD columns. Full-resolution triplet: degraded → restored → GT. **And one honest failure case** (likely an aliased dense periodic array) with a one-line explanation of why it fails. Failure honesty is explicitly rewarded. |
| 7 | Technology & Feasibility | PyTorch version, GPU used for training, training time, parameter count, model file size, per-image inference time and img/s, batch size, precision, timing method. Note H100 projection if you trained elsewhere. |
| 8 | GitHub & Video Link | Public repo URL (verify it's public in an incognito window). Optional ≤5-min demo video showing `inference.py` running end to end. |
| 9 | References | KLA-supplied surveys (below) + NAFNet + CutBlur + SSIM/LPIPS papers + every external dataset/model with licence. |

**Content that must appear somewhere across slides 3–7** (from the internal deliverables list): preprocessing, augmentation, experiments/tracking, baseline comparison, PSNR/SSIM/LPIPS, runtime + batch size + hardware, visual successes and failures, limitations, external-resource disclosure, next steps. Compress into the 9 slides; don't add slides.

**KLA-supplied references (cite these — it signals you read their material):**
- Kumar, T. et al. (2024). *Image Data Augmentation Approaches: A Comprehensive Survey and Future Directions.* IEEE Access, 12.
- Zhai, L. et al. (2023). *A Comprehensive Review of Deep Learning-Based Real-World Image Restoration.* IEEE Access, 11, 21049–21067.
- Terven, J. et al. (2025). *A Comprehensive Survey of Loss Functions and Metrics in Deep Learning.* Artificial Intelligence Review, 58, 195.
- Monga, V. et al. (2021). *Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing.* IEEE Signal Processing Magazine, 38(2), 18–44.

**Method references to add:**
- Chen, L. et al. (2022). *Simple Baselines for Image Restoration* (NAFNet). ECCV.
- Yoo, J. et al. (2020). *Rethinking Data Augmentation for Image Super-Resolution* (CutBlur). CVPR.
- Wang, Z. et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity* (SSIM). IEEE TIP.
- Zhang, R. et al. (2018). *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (LPIPS). CVPR.
- Shi, W. et al. (2016). *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN* (PixelShuffle). CVPR.

---

## 15. FINAL SUBMISSION CHECKLIST

**Portal / format**
- [ ] Team registered before the 16 Aug cutoff (registration and submission close the same day).
- [ ] Deck exported as **PDF**, ≤9 slides, instruction slide removed, named `TeamName_KLA_PS01.pdf`.
- [ ] GitHub repo is **public** — verified in a private/incognito browser window while logged out.
- [ ] Repo link appears on Slide 8 **and** in the portal form.
- [ ] Demo video (≤5 min) recorded and linked, if doing one.

**Repo contents (F12)**
- [ ] `README.md` with exact, copy-pasteable setup + inference + training commands.
- [ ] `inference.py` — standalone `.py`, `--input_dir`/`--output_dir`, runs with zero manual edits.
- [ ] `train.py` — reproduces the submitted checkpoint.
- [ ] `weights/best.pt` present and downloadable (Git LFS, or a working public Drive/HF link + sha256).
- [ ] `results/restored_test_outputs/` — actual model outputs, not placeholders.
- [ ] `requirements.txt` — complete `pip freeze`, pinned.

**Correctness**
- [ ] Clean-room test passed (fresh clone, fresh venv, different CWD).
- [ ] Output filenames byte-identical to inputs; extensions and dtype match the GT convention.
- [ ] Every output is exactly 2× the input dimensions.
- [ ] Outputs clipped to [0,1] inside the pipeline (KLA does not clip — F6).
- [ ] Inputs are **not** clipped (out-of-range values preserved — F5).
- [ ] Mixed-resolution folder (128 + 256) handled.
- [ ] Single-image folder handled.
- [ ] CPU fallback does not crash.
- [ ] No NaN/Inf in any output.

**Reporting**
- [ ] PSNR, SSIM and LPIPS reported, with the exact implementation and settings stated.
- [ ] At least one baseline compared (three recommended).
- [ ] At least one honest failure case shown at full resolution.
- [ ] End-to-end runtime, hardware, batch size, precision and timing method stated.
- [ ] Validation split defined without leakage and committed as a file list.
- [ ] Seeds, hyperparameters, checkpoints and final config tracked.
- [ ] External data/models disclosed with name, link, licence, paper/model card — or an explicit "none used".
- [ ] No confidential, unlicensed or inaccessible data.
- [ ] No retraining on hidden test inputs (F17).
- [ ] Every official link re-opened and re-verified before upload.

---

## 16. COMPRESSED ONE-DAY PLAN

Ordered by expected value per hour. **Start the training run as early as possible and write documentation while it runs.**

| Hour | Task | Gate |
|---|---|---|
| 0.0–0.5 | Download dataset. Run `inspect_dataset.py`. Resolve U1–U3, U7. Write `docs/io_contract.md`. | Format and pairing known. |
| 0.5–1.5 | Run `fit_degradation.py`. Resolve U4, U5, U6, U8. Save EDA figures. | Noise parameters and kernel known. |
| 1.5–2.5 | Build `dataset.py`, `degrade.py`, `model.py`, `losses.py`. **Overfit 2 pairs to ~45 dB** — the pipeline sanity check. | If it can't overfit 2 pairs, the pipeline is broken. Do not proceed. |
| 2.5–3.0 | Compute bicubic baseline metrics. Record in `results/metrics_summary.md`. | You now have a floor to beat. |
| 3.0–3.5 | **Launch the full training run.** Checkpoint every 1000 iters. | Training is running in the background from here on. |
| 3.5–5.0 | Write `inference.py` completely. Test against the *baseline* checkpoint. Run the entire §11.4 validation suite. | Script proven runnable before the real weights exist. |
| 5.0–6.0 | Write `README.md`, `requirements.txt` (`pip freeze`), `docs/decisions.md`. Create `sample_inputs/`. Push the repo, make it public, verify in incognito. | Repo is submittable even if training is cut short. |
| 6.0–7.0 | Build the deck skeleton with EDA figures and placeholder numbers. | Deck exists. |
| 7.0–9.0 | Monitor training. Run `evaluate.py` on the latest checkpoint. Generate qualitative triplets including the failure case. Fill in the real numbers. | Results table complete. |
| 9.0–10.0 | Swap in the best (EMA) checkpoint. Re-run the full §11.4 suite. Generate `results/restored_test_outputs/`. Benchmark runtime. | Final artifacts. |
| 10.0–11.0 | Finalize the deck, export PDF with the correct filename, record the demo video. | Ready to upload. |
| 11.0–11.5 | Walk the §15 checklist line by line. Upload with buffer time. | Submitted. |

**Cut list if time runs out, in this order:** demo video → LPIPS loss term → U-Net baseline → CutBlur → ablation study. **Never cut:** the clean-room inference test, the README, the metrics table, the failure case.

---

## 17. EXECUTABLE TASK LIST (with acceptance criteria)

Work top to bottom. Each task has a testable gate — do not advance past a failing gate.

- [ ] **T1 — Scaffold.** Create the §12 directory tree. `git init`, first commit.
      *Accept:* `tree -L 2` matches §12.
- [ ] **T2 — Dataset inventory.** Implement + run `scripts/inspect_dataset.py`.
      *Accept:* `docs/dataset_findings.md` answers U1, U2, U3, U7, U9 with numeric evidence; `docs/io_contract.md` written.
- [ ] **T3 — Degradation forensics.** Implement + run `scripts/fit_degradation.py`.
      *Accept:* fitted `σ²` and speckle `v` with ranges; identified downsample kernel with residual std for each candidate; alignment peak at (0,0); autocorrelation-based order conclusion; `results/eda/noise_variance_vs_intensity.png` saved.
- [ ] **T4 — Degradation simulator.** `src/degrade.py` implementing speckle, Gaussian, downsample, with randomized order and levels.
      *Accept:* re-degrading a GT with the fitted parameters produces a residual whose variance-vs-intensity curve statistically matches the real NoisyLR curve. Plot both on one axis and eyeball the match.
- [ ] **T5 — Data pipeline.** `src/dataset.py` with paired crops, dihedral augmentation, CutBlur, real/synthetic mixing. `configs/split_val.txt` committed.
      *Accept:* a batch visualizer script dumps 8 LR/GT pairs; every GT patch is exactly 2× its LR patch and spatially aligned.
- [ ] **T6 — Model.** `src/blocks.py` + `src/model.py` with `build_model(cfg)`. NAFSR and the U-Net baseline.
      *Accept:* forward pass on (1,1,128,128) → (1,1,256,256) and on (1,1,256,256) → (1,1,512,512). Parameter count printed. No shape errors.
- [ ] **T7 — Overfit sanity check.** Train on 2 pairs for 2000 iterations.
      *Accept:* PSNR on those 2 pairs > 40 dB. **Hard gate — if this fails, something is wrong with alignment, normalization or the loss.**
- [ ] **T8 — Metrics + baselines.** `src/metrics.py`, `scripts/evaluate.py`, `scripts/make_baselines.py`.
      *Accept:* bicubic baseline PSNR/SSIM/LPIPS computed on the validation split and written to `results/metrics_summary.md`.
- [ ] **T9 — Losses.** `src/losses.py` — Charbonnier + SSIM + FFT (+ optional LPIPS).
      *Accept:* each term returns a finite scalar on a random batch; the combined loss decreases in the overfit test.
- [ ] **T10 — Full training.** `train.py` with config loading, seeding, AMP, EMA, cosine schedule, periodic validation, checkpointing, `results/experiments.csv` logging.
      *Accept:* run launched; validation PSNR exceeds the bicubic baseline by a clear margin; checkpoint contains model + ema + config + metrics + git sha.
- [ ] **T11 — Inference script.** `inference.py` per §11.3.
      *Accept:* **all seven tests in §11.4 pass.** This is the most important gate in the project.
- [ ] **T12 — `sample_inputs/`.** 4–6 small degraded images committed.
      *Accept:* `python inference.py --input_dir sample_inputs --output_dir /tmp/out` completes in under 30 s from a clean clone.
- [ ] **T13 — Runtime benchmark.** `scripts/benchmark_runtime.py`.
      *Accept:* reports total wall-clock, img/s, batch size, precision, GPU, torch/CUDA versions, timing method. Compare eager vs `--compile` and record the crossover.
- [ ] **T14 — Qualitative results.** Full-resolution triplets for 4 successes and ≥1 honest failure, saved to `results/qualitative/`.
      *Accept:* the failure case has a written one-line technical explanation.
- [ ] **T15 — Restored test outputs.** Populate `results/restored_test_outputs/`.
      *Accept:* folder is non-empty and contains genuine model outputs; note in the README which inputs produced them.
- [ ] **T16 — Documentation.** `README.md`, `requirements.txt` (`pip freeze`), `docs/decisions.md`, `weights/README.md`.
      *Accept:* a reader who has never seen the project can clone and run inference using only the README.
- [ ] **T17 — Repo publication.** Push, set public, verify in incognito, verify weights download from a logged-out session.
      *Accept:* clone → install → infer works from a machine that has never seen the project.
- [ ] **T18 — Deck.** Fill the official template per §14, export PDF as `TeamName_KLA_PS01.pdf`.
      *Accept:* ≤9 slides, instruction slide removed, every §15 reporting item present.
- [ ] **T19 — Final checklist.** Walk §15 line by line, ticking each.
      *Accept:* all boxes ticked, submitted with time to spare.

---

## 18. PITFALLS — READ BEFORE SHIPPING

1. **Dtype/format mismatch on save.** If GT is float32 and you write 8-bit PNG, you quantize away several dB. Mirror the GT format exactly. *Verify by reloading your saved output and scoring that file, not the in-memory tensor.*
2. **Clipping the input.** F5 says out-of-range NoisyLR values are intentional. Clipping them destroys information and creates a train/test mismatch if you clip in only one place.
3. **Not clipping the output.** KLA does not clip (F6). Unclipped negative or >1 values will be scored as-is and will hurt.
4. **Hardcoded paths.** `weights/best.pt` resolved from CWD instead of `__file__` breaks the moment KLA runs the script from elsewhere. Test from `/`.
5. **Heavy imports in `inference.py`.** `import lpips` or `import matplotlib` at module level adds seconds to a *timed* run.
6. **Notebook submitted as the evaluation script.** Explicitly disallowed (F11).
7. **Private repo.** Check while logged out. This is a shockingly common failure.
8. **Weights too large for plain Git.** GitHub rejects >100 MB files. Use Git LFS or a public Drive/HF link, and *test the link from a logged-out browser*.
9. **Filename drift.** Adding `_restored` to output names breaks the evaluator's pairing. Filename in = filename out.
10. **Batching mixed resolutions.** Naively stacking 128× and 256× tensors throws. Group by shape.
11. **Validation leakage.** Selecting the checkpoint on data that appeared in training inflates your reported numbers and is explicitly checked under "compute hygiene".
12. **Chasing PSNR only.** LPIPS is in the blend. A model trained on pure L2 will be smooth and score badly perceptually.
13. **Chasing LPIPS only / using a GAN.** Hallucinated texture in a defect-inspection context is worse than blur, and PSNR/SSIM collapse. Say this in the deck.
14. **Self-ensemble (×8 flip/rotate TTA).** Adds ~0.2 dB but multiplies runtime by 8. Given that throughput is a scored axis, **default it off**; expose it as a flag and report both numbers.
15. **`torch.compile` by default.** Compilation time is inside the measured window. Off by default; documented crossover.
16. **Forgetting `results/restored_test_outputs/`.** It's an explicit mandatory repo item and the easiest one to miss.
17. **Wrong PDF filename.** `TeamName_KLA_PS01.pdf`. Follow it exactly.
18. **Assuming the internal 12-slide structure is the submission format.** The portal template (9 slides, PDF) is binding.

---

## 19. IF THERE IS TIME LEFT (Round 2 backlog, 28 Aug – 4 Sep)

Ranked by expected score impact:

1. **Wider degradation randomization + external clean grayscale data** → the biggest OOD gain, which is explicitly tested.
2. **Model scaling sweep** — width 48→64, blocks 16→32 — and measure the quality/throughput Pareto front. Present the curve; picking a point on a measured frontier is a strong engineering narrative.
3. **TensorRT / INT8 or FP8 quantization** with a PyTorch fallback. Directly addresses the reviewer's stated specialization. Report calibration method and any quality delta.
4. **Algorithm-unrolling hybrid** — a few unrolled proximal-gradient steps with a learned prior, per the Monga et al. reference KLA supplied. Strong "Innovation & Uniqueness" material with an interpretability story.
5. **Noise-level estimation head** conditioning the restoration body (FiLM-style) — helps when test noise levels drift from training.
6. **Structured knowledge distillation** from a large SwinIR/HAT teacher into the fast student. Classic way to get transformer quality at CNN speed.
7. **Per-structure-family error analysis** to find which semiconductor patterns fail and targeted augmentation for them.
8. **Uncertainty map output** — flagging low-confidence regions is genuinely valuable in inspection and is a memorable finale demo.

---

*End of spec. Keep `docs/decisions.md` current as you build — it is both the hygiene evidence and the raw material for the deck.*
