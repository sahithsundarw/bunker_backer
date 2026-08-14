# Dataset Findings

Generated from the real files at `C:\kla-data` on 2026-08-15.
Every claim below is backed by a number produced by `scripts/inspect_dataset.py`,
`scripts/probe_quantization.py`, or the full-scan check quoted inline.
Nothing here is inferred from the problem statement.

> **Provenance.** `docs/SPEC.md` has since been supplied and the U-numbers below are now
> verified against SPEC §2.2. `peek.py` was never found; `scripts/inspect_dataset.py` (the
> name SPEC §5.1 and §12 actually call for) was written instead. See `docs/MISSING_INPUTS.md`.
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
