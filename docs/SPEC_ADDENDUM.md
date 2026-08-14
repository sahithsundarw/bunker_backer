# SPEC Addendum — measurements that override SPEC

**Status: BINDING. On any conflict between this document and `docs/SPEC.md`, this document
wins.** Every claim is a measurement over the real files at `C:\kla-data`, reproducible with
the scripts named against each item.

`docs/SPEC.md` has now been located and read in full (738 lines). Section references below
are verified against the real text, not assumed.

**Precedence order for this project:**

1. `docs/VERIFICATION_CONTRACT.md` — immutable, defines "correct"
2. **`docs/SPEC_ADDENDUM.md`** (this file) — overrides SPEC on any conflict
3. `docs/SPEC.md`

The SPEC→V-check requirement extraction lives in `docs/SPEC_VCHECK_MAP.md`.

---

## 1. F2 sizes — only one of the two stated regimes exists

**SPEC F2** states: *"Degradation is exactly 2× downsampling: 512×512 → 256×256, and
256×256 → 128×128."* SPEC §7.3 builds on this: *"Test inputs will be a mix of 128×128 and
256×256."*

**Measured:** only the `256×256 → 128×128` regime is present. There are **no 512-GT pairs**,
and the released test set is **uniformly 128×128**.

Full scan of every file, no sampling (`scripts/inspect_dataset.py`):

```
GT shape histogram           : {(256, 256): 3200}
NoisyLR shape histogram      : {(128, 128): 3200}
test_NoisyLR shape histogram : {(128, 128): 400}
GT with first-dim 512 : 0
```

**Scale is still exactly ×2**, verified on all 3200 pairs with zero violations:

```
pairs checked = 3200
(GT_h, LR_h, GT_w, LR_w) histogram : {(256, 128, 256, 128): 3200}
violations = 0
```

### Binding requirements

1. **Keep the model size-agnostic.** Do not hard-code 128 or 256. The architecture must take
   any `(H, W)` and emit `(2H, 2W)`: fully convolutional, no flattened layers, no positional
   tables sized to 128. SPEC §7.1's flat NAFNet body already satisfies this — its required
   size multiple is 1 (§7.3), which is an additional argument for the non-U-Net design.
2. **Keep a 256→512 fixture.** SPEC T6's acceptance criterion already demands a forward pass
   on `(1,1,256,256) → (1,1,512,512)`. No such sample exists in the data, so this synthetic
   fixture is the *only* guard against silently baking in 128→256. Keep it in the test suite.
3. **Keep shape-grouped batching in `inference.py`** (SPEC §7.3, §11.2, §18 pitfall 10) even
   though the released test set is single-resolution. It costs nothing on uniform input and
   is the difference between working and throwing if the hidden or Round-2 set differs. Keep
   the mixed-resolution test from SPEC §11.4 step 4.
4. Do **not** build 512-specific logic against the training data — it would be dead code with
   no way to exercise it.

## 2. U1 — RESOLVED (this was an open question, not a SPEC error)

SPEC U1 asks: *"File format and dtype of GT and NoisyLR (`.png` 8-bit? 16-bit PNG? `.tif`
float32? `.npy`?)"* — `.npy` was among the listed possibilities. **Answer: `.npy`, `float32`,
continuous. Not TIFF, not PNG, not 8-bit.**

- All 6800 files are `.npy`; zero files of any other extension. Load with
  `np.load(path, allow_pickle=False)`.
- `float32`, `ndim == 2` (H, W), no channel axis — 200/200 sampled per folder.
- **Not 8-bit.** `n_unique` is 59,620–65,500 for GT in a 65,536-pixel image and
  16,359–16,384 for LR in a 16,384-pixel image — essentially every pixel distinct.
- **Not quantised at any bit depth.** Fraction of values on a `k/L` integer grid:

  | grid | GT | NoisyLR | test_NoisyLR |
  |---|---|---|---|
  | `/255` | 0.0017–0.0023 | 0.0018–0.0024 | 0.0016–0.0020 |
  | `/1023` | 0.0018–0.0023 | 0.0019–0.0022 | 0.0016–0.0026 |
  | `/4095` | 0.0017–0.0022 | 0.0019–0.0025 | 0.0015–0.0026 |
  | `/65535` | 0.0019–0.0023 | 0.0015–0.0026 | 0.0018–0.0023 |

  All ≈0.002 — chance level. No bit-depth signature exists to recover.

This closes SPEC's *"#1 silent killer"* (U1): output must be `.npy` `float32`. See
`docs/io_contract.md`.

## 3. GT is per-image min–max normalised to exactly [0,1]

SPEC F5 says GT is *"normalized to [0,1]"*. Measurement sharpens this: the normalisation is
**per image, and both endpoints are attained exactly, in every single file**. Full scan of
all 3200:

```
images with min exactly 0.0 : 3200 / 3200
images with max exactly 1.0 : 3200 / 3200
min-of-min=0.00000000  max-of-min=0.00000000
min-of-max=1.00000000  max-of-max=1.00000000
```

**This does not license renormalising predictions.** Doing so costs −4.66 dB — see
`docs/decisions.md` D3. Clip only.

## 4. NoisyLR is unclipped — confirms F5

Observed range **[-0.28, 2.16]**.

| Folder | min | max | frac < 0 | frac > 1 |
|---|---|---|---|---|
| `train/GT` | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| `train/NoisyLR` | -0.094936 | 1.909087 | 0.002784 | **0.030298** |
| `test_NoisyLR` | -0.224881 | 2.158016 | 0.006863 | **0.032418** |

**~3% of input pixels exceed 1.0.** Never clamp the input (F5, §18 pitfall 2).

## 5. Consequence — `inference.py` needs NO image library at all

The data is `.npy` end to end: `np.load` in, `np.save` out.

### The module-level allowlist is now exactly these eight

```
argparse   os   sys   time   pathlib   concurrent.futures   numpy   torch
```

**Nothing else.** In particular **no image IO library** — no `cv2`, no `tifffile`, no `PIL`.

This is a **tightening** of `CLAUDE.md` §STYLE, which previously read *"…`numpy torch` + one
image IO lib"*. The trailing allowance is removed, and `CLAUDE.md` §STYLE has been updated to
match. Tightening is permitted; loosening is not (Prime Directive 1).

The SPEC §11.3 skeleton imports `cv2` and its `load_image`/`save_image` helpers branch on
`uint8`/`uint16`/float. That skeleton was written before U1 was resolved. On this dataset the
`cv2` import is:

- **dead weight on a timed run.** SPEC §11.2 lists import cost as a *seconds-scale* lever on
  the measured wall-clock, and §18 pitfall 5 calls out heavy module-level imports by name.
- **actively hazardous.** Several `cv2` paths silently convert to 8-bit or clip to [0,1].
  Inputs here legitimately reach 2.16, so routing them through `cv2` corrupts values, which
  is precisely the failure V12 exists to catch.

Keep the permissive `EXTS` glob from §11.1 for defensive reasons, but only the `.npy` branch
ever executes on this dataset. The `uint8`/`uint16` rescaling branches of the §11.3 skeleton
should be deleted, not merely left unreachable — dead code that divides by 255 is a
liability sitting next to data that must never be divided by 255.

For the same reason, `scripts/fit_degradation.py` implements every resampling kernel from
scratch in numpy — no result depends on a third-party library's undocumented antialias or
clipping behaviour.

## 6. Consequence — test filenames collide with train filenames

`test_NoisyLR` restarts numbering at `000000.npy` through `000399.npy`. All 400 test
filenames also exist under `train/`, referring to **different images**:

```
names shared between train/GT and test_NoisyLR : 400
train/NoisyLR/000000.npy vs test_NoisyLR/000000.npy
  array_equal : False
  mean        : 0.218441 vs 0.659532
```

**Never key a cache, dict, manifest, index or results structure on the bare filename.**
Qualify by split — `("train", "000000.npy")` — or key on the full path. Because both sets are
the same shape and dtype, a collision produces silently wrong results, not an exception.

This does **not** change the output rule: output filename is still byte-identical to the
input filename (SPEC U2, §18 pitfall 9). The hazard is internal bookkeeping, not naming.

---

## 7. NEW — the imagery is not semiconductor content

**Not a SPEC error, but it contradicts the framing throughout SPEC §1, F7, F8 and §5.4, and
it changes strategy.**

SPEC §1 describes *"degraded grayscale semiconductor inspection image"*; F8 says *"different
types of semiconductor structures"*; §5.4 asks the visual audit to look for *"line/space
arrays, contact holes, dense periodic arrays"*.

**The released data contains none of that.** Both train and test are ordinary grayscale
**natural photographs**. Verified over 96 samples spread across both splits
(`scripts/content_audit.py`, figures `results/eda/content_train_gt.png` and
`content_test_inputs.png`): train GT includes a butterfly, a bear's face, the Eiffel Tower, a
stone viaduct, mountains, flowers, fruit, foliage, a car on a street, a chain-link fence;
test inputs include building facades, a statue, a bicycle wheel, oranges, a clock tower,
cobblestones and tiled floors.

The content strongly resembles **DIV2K / Flickr2K** — the standard super-resolution
benchmark — converted to grayscale and cropped to 256×256.

Supporting statistics (n=48 each): histogram entropy 4.98/6.00 bits for GT. SEM and
inspection imagery is typically dominated by a few discrete grey levels (substrate vs
feature) and would be markedly more bimodal with lower entropy.

### Consequences

1. **SPEC §6.1's "hold out one entire structure family" as a proxy-OOD split does not apply
   as written.** There are no semiconductor structure families. A proxy-OOD split must be
   built on some other axis (e.g. spectral peakiness — see below) and must be described
   honestly in the deck rather than claimed as a structure-family split.
2. **SPEC §6.5 lists DIV2K/Flickr2K as candidate external data.** If the provided data
   already *is* DIV2K/Flickr2K, adding them supplies far less novelty than §6.5 implies and
   raises a licence-disclosure question (F14) about the provided data itself. SPEC §6.5's own
   judgement — skip external data for Phase 1 — stands, now for a stronger reason.
3. **The hidden test set may still be genuine semiconductor imagery.** F7 explicitly promises
   out-of-distribution content. If the released natural images are a proxy and the hidden set
   is SEM data, the domain gap is the dominant risk and argues hard for the degradation
   randomisation in §6.3 over any content-specific tuning. Do not tune to natural-image
   statistics.
4. **Say this in the deck.** Reporting that the released data is natural imagery, with the
   figures to prove it, is exactly the "dataset analysis" Slide 3 asks for, and it is honest
   about a discrepancy most entrants will not notice.

### Train vs test content shift is MILD

Measured like-for-like on the noisy 128×128 inputs of both splits, n=400 each
(`scripts/domain_shift_check.py`):

| metric | train/NoisyLR | test_NoisyLR | ratio |
|---|---|---|---|
| spectral peakiness, median | 35.73 | 37.03 | ×1.04 |
| spectral peakiness, p90 | 68.40 | 71.17 | ×1.04 |
| gradient anisotropy, median | 1.114 | 1.136 | ×1.02 |
| strongly-periodic images | 10.0% (by construction) | **12.8%** | — |

The test set skews slightly toward periodic man-made structure, but **only slightly**. An
initial visual impression that the shift was large was not supported once measured. Treat
train and test as the same domain at similar noise levels; the per-image intensity statistics
also match closely (mean 0.4518 vs 0.4427, sd 0.2233 vs 0.2203).

## 8. NEW — measured degradation parameters

From `scripts/fit_degradation.py`, n=200 pairs. Full reasoning in `docs/decisions.md`.

- **U4 — the downsample kernel is NOT area/box.** Direct least-squares kernel recovery over
  3,125,000 equations gives centre weights ≈0.320 with negative surround lobes ≈−0.045: a
  sharpening kernel. `bicubic (antialias OFF)` sits within 1.22e-05 residual std of the
  recovered optimum; an exact box costs 7.72e-04, i.e. 63× worse. See D1.
- **U5 — noise was applied AFTER downsampling.** Residual autocorrelation is ≈0 or slightly
  negative at every tested lag. Pre-downsample noise would give strongly positive correlation
  (SPEC §5.3). See D2.
- **U6 — the noise has no additive Gaussian floor.** The σ²+v·x² form SPEC §5.2 prescribes
  returns σ=0.036991, v=0.026781, but it overshoots the darkest intensity bin by **12.5×**.
  A three-parameter fit drives σ to **exactly 0** and splits the variance into a linear
  (shot/Poisson) term a=0.011253 and a quadratic (speckle) term v=0.015745.

  **SPEC §6.4 explicitly anticipated this:** *"If §5.2 shows the variance-vs-intensity curve
  is linear rather than quadratic in x, the underlying model is Poisson/shot noise rather
  than multiplicative speckle — adapt accordingly and document it."* A substantial linear
  term is present. `src/degrade.py` must implement shot noise, not speckle alone. See D2.
- **U8 — GT and LR are pixel-aligned.** Best cross-correlation shift is `(0,0)` on 12/12
  sampled pairs (SPEC §5.2 requires the peak at (0,0)). No sub-pixel offset was tested.
- **U7 — there is no dataset README.** The only non-`.npy` file shipped was `train/.DS_Store`,
  a macOS Finder binary (magic `Bud1`) containing no documentation. Moved to
  `C:\kla-data\_archive\train_DS_Store.bin`.
- **U9 — test inputs are released now**, 400 images at `C:\kla-data\test_NoisyLR`. There is
  no `test_GT`; test ground truth is withheld, so no score can be computed locally against
  the test set.

## 9. Consequence for SPEC §11.2 — startup cost **is** the throughput score

**V23 is promoted from Tier 1 to Tier 0.** Human-issued strengthening, 2026-08-15.
`docs/VERIFICATION_CONTRACT.md` is immutable and has **not** been edited; the promotion is
recorded here and is strictly stricter, which Prime Directive 1 permits.

### The workload is tiny

| quantity | measured |
|---|---|
| test files | 400 |
| bytes per file | **65,664** (65,536 data + 128-byte `.npy` header) = 64.1 KB |
| total input volume | **25.05 MB** |
| work per image | 128×128 → 256×256, 1–3 M-param model (SPEC §7.1) |

SPEC §7.1 puts the forward pass at *"well under a millisecond per image amortized"* on an
H100 in bf16. So the entire GPU workload is on the order of **0.4 s**, against 25 MB of I/O
that any modern disk serves in well under a second.

### Fixed startup cost dominates it

Measured on this machine (Python 3.12, 5 runs each):

| stage | measured |
|---|---|
| bare interpreter start | 55–91 ms |
| interpreter + `import numpy` | 214–240 ms (numpy alone: 172.6 ms cumulative, `-X importtime`) |

`torch` is not installed yet (deliberately), so its import was **not** measured. Typical
figures are 1–3 s for `import torch` and a further 1–3 s for first CUDA context creation —
**treat those as estimates, not measurements**, and replace them with real numbers in
`results/runtime_report.md` once torch is installed (V37).

Even on the optimistic end, that puts fixed startup at roughly **3–6 s** against **~0.4 s**
of actual compute. **Startup is on the order of 85–95% of the measured wall-clock.**

### What follows

1. **V23 is Tier 0, not Tier 1.** A stray module-level `import scipy` or `import cv2` is not
   a hygiene nit here — it is a direct, large hit to a scored axis. The V23 budget of
   `-X importtime` total < 3.0 s is the single highest-leverage throughput lever available.
2. **SPEC §11.2's optimisation table is mostly irrelevant at this scale.** channels_last
   (10–30%), cuDNN autotune (5–15%) and TF32 apply to a compute budget of ~0.4 s. Shaving
   30% off 0.4 s saves 0.12 s; removing one heavy import saves an order of magnitude more.
   Keep the cheap levers (they are free) but do not spend engineering time tuning them.
3. **`torch.compile` must stay off by default** — already SPEC §11.2 and §18 pitfall 15, and
   V41. At 400 images a 30–120 s compilation is catastrophic, not marginal. SPEC's stated
   crossover of "~2000 images" is 5× the actual test-set size, so the crossover is never
   reached. Record that number in `docs/decisions.md`.
4. **Self-ensemble / TTA must stay off** (V42). 8× on 0.4 s is cheap in absolute terms, but
   it cannot be justified against a fixed cost it does not amortise.
5. **Do not build a `DataLoader` with 8 workers for 400 files.** SPEC §11.2 recommends
   `num_workers=8` and §11.3 notes the eager-load alternative. Spawning 8 worker processes
   on Windows or via spawn-start costs more than reading 25 MB. Eager-load all 400 files
   (25 MB fits trivially in RAM) and skip the DataLoader entirely. Measure before assuming
   otherwise.
6. **V37/V38 must time the whole process externally**, including interpreter start — which
   the contract already requires (`time python inference.py ...`, not an internal timer).
   That requirement is doing more work than it appears to; an internal timer around the
   forward pass would report ~0.4 s and hide 90% of the real cost.

## 10. Note on F17 wording

`docs/DATA_LOCATION.md` states the hard rule as *"Never train, fine-tune, or fit degradation
parameters on test_NoisyLR (SPEC F17)."*

SPEC F17 literally reads: *"Do not retrain on hidden test inputs unless a later official
instruction explicitly permits it"*, with §2.1 adding *"No test-time training /
self-supervised adaptation on the test set."* The extension to **fitting degradation
parameters** is stricter than F17's literal text. It is retained deliberately: fitting
degradation parameters on test inputs is a form of adaptation to the test set, and all
degradation fitting in this repo is done on `train/` pairs only, where GT is available to
fit against anyway.
