# Dataset Findings

Generated from the real files at `C:\kla-data` on 2026-08-15.
Every claim below is backed by a number produced by `scripts/inspect_dataset.py`,
`scripts/probe_quantization.py`, or the full-scan check quoted inline.
Nothing here is inferred from the problem statement.

> **Provenance.** `docs/SPEC.md` has been supplied and the U-numbers below are verified
> against SPEC §2.2. The forensics run on `scripts/inspect_dataset.py` — the name SPEC §5.1
> and §12 call for — which supersedes the throwaway `peek.py` and scans all 3200 pairs rather
> than sampling. See `docs/MISSING_INPUTS.md`; no input is outstanding.
>
> **Where SPEC and measurement disagree, `docs/SPEC_ADDENDUM.md` governs.** U4, U5 and U6 are
> answered in `docs/decisions.md` (D1, D2). Content-domain findings are in D4/D5.

---

## U1 — File format and dtype

**Container:** `.npy` (NumPy binary), loaded with `np.load(..., allow_pickle=False)`.
Not TIFF, not PNG. All 6800 files under `train/` and `test_NoisyLR/` are `.npy`;
zero files of any other extension.

| Folder | dtype | ndim | shape | global min | global max | n_unique (min/max) |
|---|---|---|---|---|---|---|
| `train/GT` | `float32` | 2 | `(256, 256)` | 0.000000 | 1.000000 | 59620 / 65500 |
| `train/NoisyLR` | `float32` | 2 | `(128, 128)` | -0.278443 | 1.850523 | 16359 / 16384 |
| `test_NoisyLR` | `float32` | 2 | `(128, 128)` | -0.224881 | 1.943198 | 16361 / 16384 |

*(dtype/ndim counted over n=200 sampled files per folder: `{'float32': 200}`, `{2: 200}` in all three.)*

**n_unique ≤ 256? NO — for all three folders.**
GT reaches 65500 distinct values in a 65536-pixel image; NoisyLR reaches 16384 distinct
values in a 16384-pixel image (i.e. *every pixel is distinct*). These are **continuous
float32 values, not 8-bit data**.

**The data is not quantised to any integer grid.** Fraction of values landing on a
`k/L` grid, measured on 5 files per folder:

| grid | GT | NoisyLR | test_NoisyLR |
|---|---|---|---|
| `/255` (8-bit) | 0.0017–0.0023 | 0.0018–0.0024 | 0.0016–0.0020 |
| `/1023` (10-bit) | 0.0018–0.0023 | 0.0019–0.0022 | 0.0016–0.0026 |
| `/4095` (12-bit) | 0.0017–0.0022 | 0.0019–0.0025 | 0.0015–0.0026 |
| `/65535` (16-bit) | 0.0019–0.0023 | 0.0015–0.0026 | 0.0018–0.0023 |

All ≈0.002, i.e. chance level. No bit-depth signature. Treat as continuous float32.

**GT is per-image min–max normalised to exactly [0, 1].** Full scan of all 3200 GT files:

```
images with min exactly 0.0 : 3200 / 3200
images with max exactly 1.0 : 3200 / 3200
min-of-min=0.00000000  max-of-min=0.00000000
min-of-max=1.00000000  max-of-max=1.00000000
```

**NoisyLR is NOT clipped and escapes [0, 1] on both ends** (n=300 sampled per folder):

| Folder | min | max | frac pixels < 0 | frac pixels > 1 |
|---|---|---|---|---|
| `train/GT` | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| `train/NoisyLR` | -0.094936 | 1.909087 | 0.002784 | 0.030298 |
| `test_NoisyLR` | -0.224881 | 2.158016 | 0.006863 | 0.032418 |

~3% of LR pixels exceed 1.0. Any pipeline that clamps its **input** to [0,1] silently
destroys that 3%.

---

## U2 — Folder names and the GT↔NoisyLR pairing rule

Folders, exactly as on disk:

```
C:\kla-data\train\GT            3200 files
C:\kla-data\train\NoisyLR       3200 files
C:\kla-data\test_NoisyLR         400 files   (renamed from the shipped `NoisyLR`)
```

**Pairing rule: identical filename, zero-padded 6-digit index, `.npy` extension.**

```
first 5 GT      : ['000000.npy', '000001.npy', '000002.npy', '000003.npy', '000004.npy']
first 5 NoisyLR : ['000000.npy', '000001.npy', '000002.npy', '000003.npy', '000004.npy']

|GT|            = 3200
|NoisyLR|       = 3200
|GT & NoisyLR|  = 3200
GT-only         = []
LR-only         = []
exact name match = True
```

So `train/GT/NNNNNN.npy` ↔ `train/NoisyLR/NNNNNN.npy` for `NNNNNN` in `000000`–`003199`.
No offsets, no suffixes, no separate manifest. Set equality is exact — no orphans either way.

### ⚠ Filename namespace collision (hazard)

`test_NoisyLR` restarts numbering at `000000.npy` and runs to `000399.npy`, so **all 400
test filenames also exist in the train folders**:

```
names shared between train/GT and test_NoisyLR : 400
train/NoisyLR/000000.npy vs test_NoisyLR/000000.npy
  shapes      : (128, 128) vs (128, 128)
  array_equal : False
  mean        : 0.218441 vs 0.659532
  => identical filenames refer to DIFFERENT images.
```

Never key a cache, index, or results dict by bare filename across the two splits. Always
qualify with the split.

---

## U3 — Pair count and resolution split

**Pair count: 3200.** Full scan of every file (not sampled):

```
GT shape histogram           : {(256, 256): 3200}
NoisyLR shape histogram      : {(128, 128): 3200}
test_NoisyLR shape histogram : {(128, 128): 400}
```

**There is no 512-vs-256 split. The resolution is uniform.**

| GT first-dim | count |
|---|---|
| 512 | **0** |
| 256 | **3200** |
| other | 0 |

The task brief anticipated a mix of 512-GT and 256-GT samples. On the actual data, GT is
`256×256` for all 3200 samples and LR is `128×128` for all 3200. The scale factor is ×2,
but the absolute sizes are half what the brief assumed.

### ×2 invariant — verified on every pair, not a sample

```
pairs checked = 3200
(GT_h, LR_h, GT_w, LR_w) histogram : {(256, 128, 256, 128): 3200}
violations = 0
RESULT: every pair satisfies GT == 2x LR in both spatial dims.
```

**Zero violations across all 3200 pairs.**

---

## U8 — Are GT and LR pixel-aligned?

**Yes.** Method: block-mean-downsample GT by 2×2 to reach LR resolution, then scan integer
shifts of ±3 px in both axes and report which shift maximises normalised cross-correlation.
A pixel-aligned pair peaks at `(0, 0)`.

```
  000000.npy   best_shift=(+0,+0) corr_best=0.9779 corr_at_(0,0)=0.9779
  000266.npy   best_shift=(+0,+0) corr_best=0.9467 corr_at_(0,0)=0.9467
  000532.npy   best_shift=(+0,+0) corr_best=0.8803 corr_at_(0,0)=0.8803
  000798.npy   best_shift=(+0,+0) corr_best=0.9620 corr_at_(0,0)=0.9620
  001064.npy   best_shift=(+0,+0) corr_best=0.9569 corr_at_(0,0)=0.9569
  001330.npy   best_shift=(+0,+0) corr_best=0.9709 corr_at_(0,0)=0.9709
  001596.npy   best_shift=(+0,+0) corr_best=0.9188 corr_at_(0,0)=0.9188
  001862.npy   best_shift=(+0,+0) corr_best=0.9161 corr_at_(0,0)=0.9161
  002128.npy   best_shift=(+0,+0) corr_best=0.9423 corr_at_(0,0)=0.9423
  002394.npy   best_shift=(+0,+0) corr_best=0.9176 corr_at_(0,0)=0.9176
  002660.npy   best_shift=(+0,+0) corr_best=0.8897 corr_at_(0,0)=0.8897
  002926.npy   best_shift=(+0,+0) corr_best=0.8177 corr_at_(0,0)=0.8177

best shift == (0,0) for 12 / 12 sampled pairs
```

Correlation at zero shift ranges 0.818–0.978; residual is noise/blur, not misregistration.
No sub-pixel offset was tested — only integer shifts. Alignment is consistent with a
**centre-aligned 2×2 average-pool decimation** (no half-pixel shift).

Supporting evidence that the degradation is zero-mean: mean of `blockmean2x(GT) − LR`
over n=20 pairs is `0.000084` (std `0.000677`, absmax `0.001683`). Folder-level means
agree to 5 decimals: GT `0.436264` vs NoisyLR `0.436257`.

---

## U9 — ANSWERED: test inputs are released

400 test input images are present and usable:

```
test_NoisyLR present at : C:\kla-data\test_NoisyLR
n_files                 : 400
shape / dtype           : (128, 128) float32
GT for test set present : False
```

There is **no** `test_GT`. Test ground truth is withheld; the test set is input-only.

**Train and test LR are drawn from the same distribution** (per-image statistics, n=400 each):

| | per-image mean | per-image std |
|---|---|---|
| `train/NoisyLR` | mean 0.4518, std 0.1706, range [0.0157, 0.9162] | mean 0.2233, std 0.0583 |
| `test_NoisyLR` | mean 0.4427, std 0.1659, range [0.0332, 0.9047] | mean 0.2203, std 0.0692 |

No meaningful covariate shift; a model fit on train LR is operating in-distribution on test LR.

---

---

## U7 — is there a dataset README or metadata file?

**No.** SPEC §5.1 step 2 requires looking for README / metadata / `.txt` / `.json` / `.csv` at
any level and printing them in full. The only non-`.npy` file shipped anywhere in the dataset
was a single loose file in `train/`:

```
C:\kla-data\train\.DS_Store   10,244 bytes
first 16 bytes: 00 00 00 01 42 75 64 31 00 00 20 00 00 00 08 00
                            ^^^^^^^^^^^ ASCII "Bud1"
```

`Bud1` is the magic number of a macOS Finder window-state database. It contains no readable
text beyond `blob` / `bpli` type markers and **documents nothing**. It answers none of U1–U9.
Moved to `C:\kla-data\_archive\train_DS_Store.bin` so it cannot be picked up by a directory
glob. After the move, zero non-`.npy` files remain under `train/` or `test_NoisyLR/`.

There is no `Data-public` README. All of U1–U9 had to be answered from the pixels.

---

## U4, U5, U6 — degradation forensics

Answered in `docs/decisions.md` (D1 kernel, D2 order and noise). Summary:

| Question | Answer |
|---|---|
| U4 downsample kernel | **Not** area/box. A sharpening kernel; `bicubic (antialias OFF)` is within 1.22e-05 residual std of the least-squares optimum, box costs 7.72e-04 |
| U5 degradation order | Noise added **after** downsampling — residual autocorrelation ≈0 or slightly negative at all lags |
| U6 noise parameters | 2-par form (SPEC §5.2): σ=0.036991, v=0.026781. But σ is an artifact — a 3-par fit gives σ=**0**, a=0.011253 (shot), v=0.015745 (speckle) |

---

## Content domain — the imagery is NOT semiconductor

Recorded here because SPEC §5.1 and §5.4 make dataset characterisation part of this
document, and because it contradicts SPEC §1/F8.

Both splits are ordinary grayscale **natural photographs** — butterflies, animals, the Eiffel
Tower, viaducts, mountains, fruit, foliage, building facades, cobblestones. Content is
consistent with DIV2K/Flickr2K converted to grayscale and cropped to 256×256. GT histogram
entropy is 4.98/6.00 bits; SEM imagery would be markedly more bimodal.

Figures: `results/eda/content_train_gt.png`, `results/eda/content_test_inputs.png`.
Full analysis and consequences: `docs/decisions.md` D4 and D5, `docs/SPEC_ADDENDUM.md` §7.

---

## Summary of corrections to prior assumptions

| Assumed (task brief and/or SPEC) | Actual (measured) |
|---|---|
| TIFF/PNG images, possibly 8-bit (SPEC U1) | `.npy` float32, continuous, no bit-depth grid |
| Both 512→256 and 256→128 regimes (SPEC F2) | Only 256→128. Zero 512-GT samples |
| Test inputs mix 128 and 256 (SPEC §7.3) | All 400 test inputs are 128×128 |
| `train/` holds a README answering U1–U9 | Loose file is `.DS_Store`, documents nothing |
| Values in [0,1] | GT yes (exactly); LR overshoots — 3.0% > 1, 0.3–0.7% < 0 |
| Downsample is plausibly area/box | Refuted — sharpening kernel, bicubic-AA-off |
| Speckle + additive Gaussian (SPEC F3) | Additive term fits to **zero**; shot + speckle instead |
| Semiconductor inspection imagery (SPEC §1, F8) | Natural photographs, DIV2K/Flickr2K-like |

---

## Proxy-OOD set — closing the U-9 generalisation gap partially, and honestly

**What this is, in one sentence:** 40 procedurally-generated (numpy primitives only,
nothing downloaded) synthetic grayscale images depicting geometric content plausible for
semiconductor inspection — line/space gratings, contact-hole-like dot grids, checkerboards,
Manhattan circuit-like traces, sharp-edged rectangles — degraded with the **existing,
already-fitted** `src/degrade.py` model, with **zero parameters refit** on this content.

**What this is NOT, stated as plainly as D4 states the content-domain finding:** this is not
semiconductor data, not SEM imagery, not a validated proxy for KLA's hidden test set. It is
synthetic procedural geometric content. Calling it anything stronger would be exactly the
overclaiming SPEC_ADDENDUM §11 forbids.

### Why this exists

D4/D16 establish that the released 3200 train pairs are natural photographs and that the
only thing that survives a shift to real semiconductor content is the **measured
degradation** (kernel + noise), not any content prior. This set tests exactly that
transferable asset: does a model generalise to *structurally different* (periodic/geometric
vs natural-photo-textured) content **under the identical measured degradation**? It says
nothing about real inspection imagery, which this project has never seen.

### Generation method (deterministic, `numpy` only)

Location: `results/eda/proxy_ood/` (GT/, NoisyLR/, manifest, membership list). Generated by
a one-off script executed against this repo's `src/degrade.py` (not committed under
`scripts/`, since script ownership is scoped to `scripts/inspect_dataset.py` and
`scripts/fit_degradation.py` — the generation logic is recorded here in full instead so it
is reproducible from this document alone).

5 categories x 8 images = **40 images**, `256x256`, seeded `[20260816, image_index]`
(`np.random.default_rng`), fully deterministic:

| category | construction | params varied |
|---|---|---|
| `line_space_grating` | oriented square wave, `phase = (x cos θ + y sin θ) mod pitch / pitch < duty`, 3x3 box-blurred | pitch `U(4,28)` px, duty `U(0.35,0.65)`, angle in {0,30,45,90}±3° |
| `contact_hole_grid` | circular dots on a (optionally jittered) regular grid, 3x3 box-blurred | pitch `U(10,32)` px, radius `0.18-0.38 x pitch`, jitter `U(0,0.6)` |
| `checkerboard` | `((y//t + x//t) % 2)`, 3x3 box-blurred | tile size `U(4,24)` px (integer) |
| `circuit_traces` | 8-18 random Manhattan (H/V) line segments width 2-5 px + 6-14 square pads 6-16 px, unioned, 3x3 box-blurred | trace/pad count and geometry |
| `sharp_edge_shapes` | 10-25 random axis-aligned rectangles at random grey levels 0.2-1.0, max-composited, 3x3 box-blurred | rectangle count, size, level |

Each raw image is then **per-image min-max normalised to exactly `[0,1]`**, matching the
measured real-GT convention (U1: all 3200 real GT files hit both endpoints exactly).
Verified on all 40 generated images: `gt_min == 0.0` and `gt_max == 1.0` for **40/40**.

### Degradation reused exactly as fitted — no refitting

Every image is degraded with `src.degrade.degrade_fitted(gt, rng)`, which is the **V33
fidelity path**, not the randomised training-augmentation path (`src.degrade.degrade`):

```
LR_clean = conv_downsample_2x(gt, kernel=RECOVERED_KERNEL_4X4)   # D1, centre weight mean 0.3204
LR_noisy = add_noise(LR_clean, rng, params=FITTED_NOISE)         # D2/D12, sigma=0, a=0.011253, v=0.015745
```

No `DegradeConfig` randomisation (`randomise_frac`, `bicubic_alt_prob`,
`gauss_pre_down_prob`) is used — this is deliberate per the task instruction: fitting or
even *stochastically varying* the degradation on new content would defeat the purpose of an
OOD-generalisation test. The kernel and noise constants are read directly from
`src/degrade.py`'s module-level constants (`RECOVERED_KERNEL_4X4`, `FITTED_NOISE`), imported,
never re-derived. Synthetic LR is **not clipped** (SPEC F5, same as training data).

### Measured properties of the resulting set (n=40)

```
gt_min == 0.0 for 40/40, gt_max == 1.0 for 40/40   (exact min-max convention match)
LR range across all 40 images : [-0.0687, 1.6807]
mean frac LR pixels < 0        : 0.0726   (real train/NoisyLR: 0.0028)
mean frac LR pixels > 1        : 0.0949   (real train/NoisyLR: 0.0303)
```

**This difference is itself a finding, not an error.** The overshoot fraction is 2-3x higher
than the real released data because these are sharp-edged, high-contrast, often near-binary
patterns (fine gratings, checkerboards) — the recovered kernel's negative side-lobes
(D1: max abs weight outside centre 2x2 = 0.0482) ring harder against a sharp step edge than
against the softer gradients of natural photographs. The finest-pitch grating in the set
(pitch 7.30 px, close to the 2x decimation's Nyquist limit of 4 px) visibly aliases into a
moiré pattern after degradation — visible directly in
`results/eda/proxy_ood/proxy_ood_grid.png`, consistent with D5's finding that fine periodic
structure is the honest hard case for this degradation, now demonstrated on genuinely
periodic content rather than the broadband-texture case D5 found in the real data.

### Membership list and disjointness (verified, not asserted)

Filenames use the prefix `proxyood_NNNNNN.npy` (`proxyood_000000.npy` … `proxyood_000039.npy`),
deliberately **not** the bare `NNNNNN.npy` pattern used by `train/GT`, `train/NoisyLR` and
`test_NoisyLR` (docs/SPEC_ADDENDUM.md §6 collision hazard), so collision is impossible by
construction. Verified computationally against the real files at `C:\kla-data`
(`results/eda/proxy_ood/membership_check.json`):

```
n_proxy_ood = 40, n_train_gt = 3200, n_train_lr = 3200, n_test = 400
intersection_with_train_gt = []   disjoint_from_train_gt = True
intersection_with_train_lr = []   disjoint_from_train_lr = True
intersection_with_test     = []   disjoint_from_test     = True
```

Membership list committed at `results/eda/proxy_ood/membership_list.txt` (one name per
line, 40 entries, `#`-commented header explaining provenance — same format convention as
`configs/split_val.txt`).

### What this proxy-OOD set can and cannot prove

**Can:** show whether a trained model's PSNR/SSIM/LPIPS degrades, and by how much, when the
input content shifts from natural-photo textures to periodic/geometric structure, holding
the degradation model fixed at its measured parameters. A large drop here would be actionable
evidence that the model over-fit natural-image priors rather than learning the degradation
inverse.

**Cannot:** validate performance on real semiconductor/SEM inspection imagery. No real
inspection imagery exists anywhere in this project (D4). It also cannot validate robustness
to a *different* degradation than the one measured here — KLA's hidden test set may carry
different noise levels (F7) or, in principle, a different acquisition pipeline entirely,
which this proxy set does not probe.

**Scope note for the next agent:** scoring a model against this set is explicitly out of
scope for this agent (owner: `dataset-forensics`) and belongs to `scripts/evaluate.py`
(owner: `loss-metrics`). This entry documents the inputs only.
