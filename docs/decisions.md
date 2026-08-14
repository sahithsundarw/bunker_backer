# Decisions

Measured decisions with their evidence. Each entry states what was decided, the numbers
behind it, and what would overturn it.

Reproduce with:

```powershell
py -3.12 scripts\fit_degradation.py C:\kla-data --n 200
py -3.12 scripts\renorm_experiment.py C:\kla-data
```

```powershell
py -3.12 scripts\visual_audit.py C:\kla-data
py -3.12 scripts\content_audit.py C:\kla-data
py -3.12 scripts\domain_shift_check.py C:\kla-data
```

Artifacts under `results\eda\`: `degradation_fit.json`, `renorm_experiment.json`,
`fit_degradation_report.txt`, `noise_variance_vs_intensity.png`, `pairs_grid.png`,
`aliasing_failure_case.png`, `content_train_gt.png`, `content_test_inputs.png`,
`domain_shift.png`.

---

## D1 — Downsample kernel: box is REFUTED; use bicubic (antialias OFF)

**Decision.** The GT→LR downsample is a **sharpening kernel**, well modelled by
`bicubic with antialias OFF`. It is **not** a 2×2 box / average-pool. The earlier
block-mean observation in `dataset_findings.md` was too crude to separate the two and is
superseded.

### Candidate ranking (n=200 pairs, all candidates)

`residual = LR_actual − kernel(GT)`; lower std is better.

Candidate set follows SPEC §5.2 exactly: bicubic (antialias T/F), bilinear (antialias T/F),
area/box, gaussian σ∈0.5..1.5 + stride2, nearest. `corr(residual², LR_hat)` is the SPEC §5.2
speckle signature.

| candidate | resid_std | std_min | std_max | mean_bias | corr(r²,LR̂) |
|---|---|---|---|---|---|
| **bicubic (antialias OFF)** | **0.085780** | 0.018071 | 0.198664 | -7.28e-06 | 0.3155 |
| box 2×2 avgpool | 0.086503 | 0.018222 | 0.205449 | -7.28e-06 | 0.3127 |
| bilinear (antialias OFF) | 0.086503 | 0.018222 | 0.205449 | -7.28e-06 | 0.3127 |
| gaussian σ=0.5 + stride2 (offset .5) | 0.086706 | 0.018281 | 0.207132 | -7.28e-06 | 0.3120 |
| gaussian σ=0.7 + stride2 (offset .5) | 0.088160 | 0.018735 | 0.218057 | -7.27e-06 | 0.3067 |
| bicubic (antialias ON) | 0.088554 | 0.018733 | 0.226515 | -8.14e-06 | 0.3073 |
| gaussian σ=0.9 + stride2 (offset .5) | 0.090607 | 0.019590 | 0.232777 | -7.16e-06 | 0.2974 |
| bilinear (antialias ON) | 0.090748 | 0.019588 | 0.235138 | -7.28e-06 | 0.2971 |
| gaussian σ=0.7 + stride2 (offset 0) | 0.094200 | 0.019163 | 0.242187 | +4.47e-06 | 0.2785 |
| gaussian σ=0.9 + stride2 (offset 0) | 0.095046 | 0.019885 | 0.245443 | +5.59e-06 | 0.2772 |
| gaussian σ=0.5 + stride2 (offset 0) | 0.095105 | 0.018979 | 0.259754 | +2.70e-06 | 0.2711 |
| gaussian σ=1.1 + stride2 (offset 0) | 0.096426 | 0.020809 | 0.250519 | +6.16e-06 | 0.2729 |
| gaussian σ=1.5 + stride2 (offset .5) | 0.097639 | 0.022779 | 0.256109 | -5.66e-06 | 0.2701 |
| gaussian σ=1.3 + stride2 (offset 0) | 0.097983 | 0.021830 | 0.254855 | +6.76e-06 | 0.2674 |
| nearest | 0.099039 | 0.020039 | 0.322678 | +2.46e-06 | 0.2576 |
| gaussian σ=1.5 + stride2 (offset 0) | 0.099557 | 0.022890 | 0.258209 | +7.53e-06 | 0.2619 |

The speckle signature is **strongly positive (0.3155)** for the best kernel: residual power
grows with intensity, confirming signal-dependent noise before any variance fitting. SPEC
§5.2 asks for exactly this number.

Two sanity checks embedded in this table:

- `bilinear (antialias OFF)` is **bit-identical** to `box`. At scale ×2 with output centres
  at `2i+0.5`, the triangle kernel puts weight `1−0.5 = 0.5` on each of the two nearest taps
  — exactly the 2×2 average. The resampler reproducing this degeneracy confirms it is correct.
- Every `offset 0` gaussian ranks below every `offset .5` gaussian, confirming the
  **centre-aligned** sampling grid (no half-pixel shift).

**The top two are 0.84% apart. That is not a verdict.** With a noise floor near 0.092, the
ranking cannot separate them, so the decision was made by direct kernel recovery instead.

### Decisive test: least-squares kernel recovery

Fit `w` in `LR[i,j] = Σ_ab w[a,b] · GT[2i+a, 2j+b]` over **3,125,000 equations** from 200
pairs. No candidate list involved. OLS stays consistent here because the noise is zero-mean
given GT (measured mean bias 8.19e-05), even though it is heteroscedastic.

Recovered 4×4 kernel (offsets from `2i`):

```
              -1        +0        +1        +2
  -1    0.014066 -0.038645 -0.045098  0.007462
  +0   -0.045368  0.327878  0.318776 -0.033904
  +1   -0.048204  0.321710  0.312729 -0.039037
  +2    0.017182 -0.039238 -0.039416  0.008900
```

| quantity | value | box would give |
|---|---|---|
| sum of all weights | 0.99979320 | 1.0 ✓ flux preserving |
| centre 2×2 mean weight | **0.32027316** | 0.25 |
| max abs weight outside centre 2×2 | **0.04820392** | 0.0 |
| ‖w − box‖∞ | **0.07787773** | 0.0 |

The signature is unmistakable: **centre weights well above 0.25, ringed by negative lobes**.
That is a sharpening filter, which a box filter is not — a box has no negative lobes at all.

Residual std on identical interior pixels:

| kernel | resid_std | cost vs optimal |
|---|---|---|
| recovered LS kernel (optimal linear) | 0.08526269 | — |
| bicubic (antialias OFF) | 0.08527491 | **+1.22e-05** |
| exact box (0.25 ×4) | 0.08603516 | +7.72e-04 |

`bicubic (antialias OFF)` is within 1.2e-05 of the optimal linear kernel — statistically
indistinguishable. The box costs 63× more.

**Independent confirmation** from the residual autocorrelation: the box leaves systematically
more structure than the recovered kernel at every tested lag (see D2 table).

**Caveat, recorded honestly.** A K=6 recovery finds max |weight| = 0.01355 in the outermost
ring, so the true support extends slightly beyond 4×4 — bicubic-AA-off is a 4-tap kernel and
cannot represent that. The effect is worth ~1e-05 in residual std, i.e. negligible, but the
recovered kernel is not *exactly* bicubic. Use `bicubic (antialias OFF)` as the working model.

**Would overturn this:** a K=8 recovery showing significant far lobes, or the SPEC stating
the generative kernel outright.

---

## D2 — Degradation order: noise added AFTER downsampling; noise is signal-dependent with no additive floor

**Decision.** The pipeline is `GT → downsample → add signal-dependent noise`. The noise has
**no additive Gaussian floor**; it is dominated by shot and speckle terms.

### Order — residual autocorrelation

If noise were injected before downsampling, the kernel would smear it and leave **strongly
positive** neighbour correlation. Measured, under the recovered kernel:

| lag | mean | sd | min | max |
|---|---|---|---|---|
| (0,1) | **-0.04130** | 0.04199 | -0.12317 | +0.18059 |
| (1,0) | **-0.05659** | 0.03339 | -0.15700 | +0.05898 |
| (1,1) | **+0.00213** | 0.02292 | -0.09142 | +0.09797 |

Same residual under an exact box kernel, for contrast:

| lag | mean | sd |
|---|---|---|
| (0,1) | -0.04483 | 0.04378 |
| (1,0) | -0.06113 | 0.03656 |
| (1,1) | +0.00177 | 0.02460 |

**Conclusion: noise was added after downsampling.** Mean |autocorr| = 0.03334. The decisive
point is the **sign**, not the magnitude: pre-downsample noise gives positive correlation, and
every measured value is ≈0 or negative. The small residual negative correlation at (0,1) and
(1,0) is leftover kernel-model error, not noise colour — it shrinks when moving from the box
to the recovered kernel, which is what kernel mismatch does and what noise colour would not.

### Noise magnitude — `var(residual | x) = σ² + v·x²`

Fitted as requested, global fit over 3,125,000 pixels:

| parameter | value |
|---|---|
| **σ** (additive Gaussian) | **0.036991** |
| **v** (speckle, multiplicative) | **0.026781** |
| speckle sd at x=1 | 0.163649 |
| residual overall std | 0.091946 |
| R² on binned variances | 0.984080 |

Per-image fits across 200 pairs:

| parameter | mean | sd | min | max | p5 | p95 |
|---|---|---|---|---|---|---|
| **σ** | 0.031594 | 0.032526 | 0.000000 | 0.182546 | 0.000000 | 0.085970 |
| **v** | 0.028180 | 0.010641 | -0.017926 | 0.073702 | 0.015350 | 0.043383 |

**Identifiability caveat:** 35/200 images (17.5%) fit σ = 0 exactly, and 3/200 (1.5%) fit
v < 0, which is unphysical. σ and v trade off against each other within a single image. The
**global fit is the reliable estimate**; the per-image ranges describe fit instability at
least as much as genuine per-image variation.

### ⚠ The 2-parameter model is wrong at low intensity

R² = 0.984 is flattering. It is dominated by high-intensity bins where the variance is large.
At the dark end the fit fails badly:

| x | var observed | var fit (2-par) | ratio |
|---|---|---|---|
| 0.0157 | 1.102e-04 | 1.375e-03 | **12.47×** |
| 0.0390 | 1.882e-04 | 1.409e-03 | 7.49× |
| 0.0568 | 2.790e-04 | 1.455e-03 | 5.21× |
| 0.0730 | 3.569e-04 | 1.511e-03 | 4.23× |
| 0.0883 | 4.802e-04 | 1.577e-03 | 3.28× |
| 0.1049 | 6.271e-04 | 1.663e-03 | 2.65× |
| 0.4144 | 6.837e-03 | 5.966e-03 | 0.87× |
| 0.9725 | 2.701e-02 | 2.670e-02 | 0.99× |

The fitted floor σ² = 1.368e-03 is **12× above** the variance actually observed in the darkest
bin. Alternative fits:

| model | mean abs rel err | R² |
|---|---|---|
| σ² + v·x² (abs-weighted) — *the requested fit* | 0.8688 | 0.984080 |
| σ² + v·x² (rel-weighted) | **0.2263** | 0.968464 |
| σ² + a·x + v·x² (abs-weighted) | 0.3356 | **0.990052** |

- rel-weighted 2-par: σ = 0.017708, v = 0.030090
- 3-par: **σ = 0.000000**, a = 0.011253, v = 0.015745

**Interpretation.** The 3-parameter fit drives the additive term to exactly zero and splits
the variance into a linear (shot/Poisson, `a·x`) and quadratic (speckle, `v·x²`) term. There
is **no additive Gaussian noise floor in this data**. The headline σ = 0.036991 is an
**upper bound artifact** of forcing a 2-parameter model, not a measurement of a real
Gaussian component.

**SPEC §6.4 anticipated exactly this case:**

> *"If §5.2 shows the variance-vs-intensity curve is linear rather than quadratic in `x`, the
> underlying model is Poisson/shot noise rather than multiplicative speckle — adapt
> accordingly and document it. Note this possibility; do not assume."*

A substantial linear term is present (a = 0.011253), so `src/degrade.py` must implement
**shot noise in addition to speckle**. The SPEC §6.4 reference `add_speckle` (`y = x + n·x`,
`n ~ N(0, var)`) reproduces only the `v·x²` term and would under-noise mid-tones while
over-noising darks. This also revises SPEC F3, which names the two mechanisms as speckle plus
*additive* Gaussian: the additive component measures to zero here.

**Practical guidance.** Report σ = 0.036991, v = 0.026781 when the 2-parameter form is
required. For anything that *simulates* the degradation — synthetic data, augmentation,
degradation-aware training — use `var = 0.011253·x + 0.015745·x²` with no additive term, or
the 2-parameter form would inject ~12× too much noise into dark regions and teach the model
to over-smooth exactly where the data is cleanest.

**Would overturn this:** a K=8 kernel recovery materially changing the residual, or SPEC
specifying the generative noise model.

---

## D3 — Renormalisation policy: CLIP, do not renormalise

**Decision.** Post-process predictions with `np.clip(pred, 0.0, 1.0)` and **nothing else**.
Do **not** apply per-image min–max renormalisation.

### The question

GT is per-image min–max normalised to exactly [0,1] (all 3200 files attain both endpoints,
full scan). That raised a genuine question: should predictions be renormalised the same way
so their statistics match the targets?

### Experiment

Model: parameter-free bicubic ×2 upsample of the LR input. Parameter-free matters — nothing
is fitted, so the comparison isolates the post-processing choice. Validation: the **last 200
training pairs** (`003000.npy`–`003199.npy`), held out.

| variant | PSNR dB (mean ± sd) | SSIM (mean ± sd) |
|---|---|---|
| V1 raw (no post-processing) | 23.1766 ± 2.9800 | 0.53906 ± 0.20599 |
| **V2 clip to [0,1]** | **23.4247 ± 2.8319** | **0.54284 ± 0.20225** |
| V3 per-image min–max renorm | 18.7611 ± 4.7949 | 0.53816 ± 0.16978 |

Pairwise, per image:

| comparison | mean ΔPSNR | wins |
|---|---|---|
| clip vs raw | **+0.2482 dB** | clip wins **199/200** |
| renorm vs clip | **−4.6636 dB** | renorm wins **9/200** |
| renorm vs clip (SSIM) | −0.004677 | renorm wins 122/200 |

Distribution of `renorm − clip` ΔPSNR: min −12.7291, p5 −9.5996, median −4.6279,
p95 −0.0243, max +0.4477.

### Why renormalisation fails

The raw prediction range explains it:

```
min : mean -0.0273  worst -0.1521   |  frac images with min < 0 : 0.765
max : mean +1.3108  worst +2.0291   |  frac images with max > 1 : 0.955
```

95.5% of predictions overshoot 1.0, reaching 2.03 in the worst case, because the input noise
itself reaches 2.16. Min–max renormalisation divides by that **outlier-driven** range,
compressing the entire image toward mid-grey and destroying contrast. It is rescaling by
noise, not by signal.

Note SSIM and PSNR disagree in sign here: renorm wins on SSIM in 122/200 images while losing
4.66 dB of PSNR on average, because SSIM is partly contrast-invariant and forgives a global
rescale that PSNR punishes. **PSNR is decisive** — the mean SSIM is still worse for renorm
(0.53816 vs 0.54284), so no metric prefers it overall.

Clipping wins 199/200 against raw and is essentially free, since it only touches pixels the
target provably cannot contain.

**Confirmed against SPEC.** F5: *"Do clip the output to [0,1] since GT lives there."* F6:
*"KLA does not clip or renormalize outputs."* §5.1: *"Values clipped to [0,1] before dtype
conversion."* SPEC nowhere mentions renormalising, and the measurement shows why. Clip-only
is both the SPEC-mandated and the empirically optimal choice.

**Would overturn this:** a contrast-invariant scoring metric dominating the undisclosed
blend (F9). Note that even the *upper* end of the renorm distribution (p95 −0.0243 dB) is
negative — renorm is not merely worse on average, it is worse nearly everywhere.

---

## D6 — Contract amendments (HUMAN-ISSUED, 2026-08-15)

**Both edits below were authorised by the human.** `docs/VERIFICATION_CONTRACT.md` is
immutable to the agent (Prime Directive 1); the agent did not originate, and may not
originate, either change. Recorded here because PD1 requires a `decisions.md` entry
explaining any modification to a pinned file.

### Amendment 1 — V39 replaced

| | text |
|---|---|
| **Was** | *Throughput floor.* ≥ 20 images/second on the dev GPU at 128→256 with default settings, or a documented justification in `docs/decisions.md` if the chosen architecture cannot reach it. Tighten this number once measured. |
| **Now** | *End-to-end wall-clock, measured and reported.* Total end-to-end wall-clock for the full 400-image test set, measured externally around the process (not an internal timer), reported in `results/runtime_report.md` with a startup-vs-compute breakdown. PASS = measured and reported. No threshold — F9 prescribes none. |

**Rationale.** The old threshold was invented by the contract, not derived from SPEC. F9
states plainly: *"No target score or latency threshold prescribed."* A fabricated 20 img/s
floor could have failed a submission that KLA would have scored fine, or — worse — driven
architecture choices to clear a number nobody asked for.

**Is this a loosening?** It removes a numeric gate, so on its face yes. Reading it as a
strengthening is defensible and is the intent: the new check demands a *startup-vs-compute
breakdown measured externally around the process*, which the old one did not, and which is
the number that actually matters here (see D7). It replaces an arbitrary threshold with a
mandatory measurement. Either way the change is human-issued, which is the only thing PD1
requires.

### Amendment 2 — V23 moved Tier 1 → Tier 0

Unambiguously a **strengthening**: V23 now blocks submission rather than being a robustness
concern. The check text is unchanged. The ID is deliberately **not** renumbered — check IDs
are stable identifiers referenced by `results/verification_report.json` and `docs/STATE.md`,
so V23 sits at the end of the Tier 0 table out of numeric sequence. That is intended.

### Consequences

- `docs/VERIFIER_SHA256` created and now pins `docs/VERIFICATION_CONTRACT.md` at
  `c1beee3d…f77624` (12,903 bytes, LF). `scripts/verify_all.py` does not exist yet, so its
  pin is a documented placeholder, not a passing check.
- **Left untouched, flagged for the human:** the SKIP whitelist still reads
  *"V39, V40 (partial) — No CUDA device available."* Under the new V39 there is no threshold,
  so it can be measured and reported on CPU and arguably never needs a skip. Removing that
  entry would be a strengthening, but only the human may make it. Not changed here.

## D7 — Startup cost is the throughput score

Measured, and the reason V23 was promoted.

| quantity | value |
|---|---|
| test files | 400 |
| bytes per file | 65,664 (65,536 data + 128-byte `.npy` header) |
| total input volume | **25.05 MB** |
| forward pass | sub-millisecond per image on H100 bf16 (SPEC §7.1) ⇒ **~0.4 s** total |
| bare interpreter start | **55–91 ms** (5 runs, Python 3.12) |
| interpreter + `import numpy` | **214–240 ms**; numpy alone 172.6 ms cumulative (`-X importtime`) |
| `import torch` + CUDA init | **not measured** — torch deliberately not installed. Typically 1–3 s each; treat as estimate, replace under V37/V39 |

Fixed startup ≈ **3–6 s** against ≈ **0.4 s** of real work: **85–95% of the scored
wall-clock is startup.**

Follows from this:

1. Import hygiene (V23) is the highest-leverage throughput lever available, by an order of
   magnitude over anything in SPEC's §11.2 table.
2. Most of the §11.2 optimisation table is near-irrelevant at this scale — 30% off 0.4 s is
   0.12 s. Keep the free levers, do not spend time tuning them.
3. `torch.compile` never reaches SPEC's stated ~2000-image crossover; the test set is 5×
   smaller. Stays off (V41).
4. **Do not build an 8-worker DataLoader for 400 files**, contrary to §11.2's recommendation.
   Spawning workers costs more than reading 25 MB. Eager-load; it fits trivially in RAM.
5. V38/V39's requirement to time externally around the process is doing real work — an
   internal timer around the forward pass would report ~0.4 s and hide 90% of the cost.

## D4 — The imagery is natural photographs, not semiconductor content

**Decision.** Treat the released data as **grayscale natural images**, most likely
DIV2K/Flickr2K-derived. Do not tune to semiconductor-specific structure priors. Report the
discrepancy openly in the deck.

SPEC §1 describes semiconductor inspection imagery; F8 promises *"different types of
semiconductor structures"*; §5.4 asks the visual audit to look for line/space arrays, contact
holes and dense periodic arrays.

**None of that is present.** Across 96 samples spread over both splits
(`results/eda/content_train_gt.png`, `content_test_inputs.png`), train GT contains a
butterfly, a bear's face, the Eiffel Tower, a stone viaduct, mountains, flowers, fruit,
foliage, a car on a street and a chain-link fence; test inputs contain building facades, a
statue, a bicycle wheel, oranges, a clock tower, cobblestones and tiled floors.

Supporting statistic: GT histogram entropy is **4.98 of 6.00 bits** (n=48). SEM and
inspection imagery is dominated by a few discrete grey levels and would be far more bimodal.

**Consequences.**

1. SPEC §6.1's proxy-OOD split *"hold out one entire structure family"* cannot be done as
   written — there are no semiconductor structure families. Build the proxy-OOD split on
   another axis and describe it honestly rather than claiming a structure-family split.
2. SPEC §6.5 proposes DIV2K/Flickr2K as external data. If the provided data already *is*
   that, adding it buys much less than §6.5 implies. §6.5's own advice — skip external data
   for Phase 1 — stands, now for a stronger reason.
3. The **hidden** test set may still be real semiconductor imagery (F7 promises OOD content).
   That argues for the degradation randomisation of §6.3 over any content-specific tuning.

**Would overturn this:** an official statement that the released set is a stand-in, or a
Round-2 release containing actual SEM data.

## D8 — Source dataset: NOT confidently identified (DIV2K is the leading hypothesis)

**Decision: do not assert a source dataset anywhere.** Time-boxed investigation
(`scripts/source_id.py`) produced strong circumstantial evidence for DIV2K but nothing
conclusive. Record it as a hypothesis, not a fact.

### Evidence FOR DIV2K

**The arithmetic fits exactly.** DIV2K ships 800 training and 100 validation images:

```
3200 train pairs = 800 x 4      400 test inputs = 100 x 4
```

Four 256×256 crops per source photograph accounts for both counts with no remainder.
Flickr2K (2650) and DF2K (3450) give no integer fit. BSD500 also fits (400×8, 100×4) but
less naturally.

**Consecutive-crop grouping is present in train.** If crops from one photo are laid out
consecutively, runs of K should be more self-similar than chance. Ratio of mean within-group
to mean between-group descriptor distance, first 480 train GT:

| K | 2 | 3 | 4 | 5 | 6 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| ratio | 0.554 | 0.647 | 0.563 | 0.777 | 0.809 | 0.826 | 0.889 |

K=2 and K=4 are both strong while K=3 is weaker — the signature of a true group size of 4
(K=2 scores well only because 2 divides 4, so every pair still falls inside one true group).
Confirmed by offsets within aligned blocks of 4, first 800 images:

```
offset 1 within block : 1.5535     across block boundary : 1.7625
offset 2 within block : 1.5750     random different-block: 1.6773
offset 3 within block : 1.5956
```

All three within-block offsets sit below the random baseline, including offset 3 (first vs
last of a block). Consistent with 4 crops per source image.

**Content type matches.** High-quality diverse photography — landmarks, wildlife, botanical
close-ups, architecture — is what DIV2K is.

### Evidence AGAINST a confident call

1. **No identifying metadata survives.** `.npy` headers carry only
   `{'descr': '<f4', 'fortran_order': False, 'shape': (256, 256)}`. No EXIF, no author, no
   original filename. Filenames are bare sequential `000000.npy`.
2. **Test shows no grouping at all** — every K from 2 to 12 gives a ratio of 0.98–1.02. If
   test were 100 photos × 4 consecutive crops, K=4 should have stood out. It does not. That
   requires an extra assumption (the test crops were shuffled), which is plausible but
   unverified.
3. **The within-block effect is modest** — 1.55–1.60 vs a 1.68 baseline is real but not
   overwhelming, and the across-boundary figure (1.7625) sitting *above* the random baseline
   is unexplained.
4. **Natural-photo corpora are visually interchangeable.** DIV2K, Flickr2K, BSD and Waterloo
   would all look like this in grayscale at 256×256. Content alone cannot separate them.
5. **No direct comparison was made.** Confirming this properly means downloading DIV2K and
   matching crops against it. Not authorised, and not done.

**Verdict: NOT IDENTIFIED.** DIV2K is the leading hypothesis and the arithmetic is
suggestive, but the evidence is circumstantial. Do not state a source dataset in the deck,
the README, or `docs/dataset_findings.md`. If it matters later, the decisive test is a direct
crop match against downloaded DIV2K — cheap, but it needs authorisation.

### ⚠ Integrity flag, independent of the identification

**If** the test inputs derive from a public corpus, then their ground truth is publicly
downloadable. Obtaining it would be equivalent to obtaining the hidden test labels.

**That is prohibited, and remains prohibited whether or not the source is ever confirmed.**
It would violate the spirit of F17 and the substance of SPEC §15's no-leakage requirements,
and it would invalidate every reported number. Score only against the held-out split of
`train/`, never against a reconstructed test GT. This is recorded so the idea is explicitly
closed off rather than left as an unexamined temptation.

## D9 — Pretrained initialisation: from scratch for Phase 1

**Decision: train from scratch.** Revisit pretrained initialisation in Round 2 as a measured
experiment, not an assumption. Nothing has been downloaded.

### Candidate ×2 SR checkpoints

| Model | Licence | Competition use | Notes |
|---|---|---|---|
| **SwinIR** (JingyunLiang/SwinIR) | **Apache-2.0** | Permitted — permissive, needs attribution + NOTICE | Classical/lightweight ×2 checkpoints exist. SwinIR-light ≈900 K params. SPEC §7.2: wins Axis 1, loses Axis 2 |
| **EDSR / RCAN via BasicSR** (XPixelGroup/BasicSR) | **Apache-2.0** (toolbox) | Permitted, **with a caveat** | BasicSR keeps a `LICENSE/` folder documenting per-model terms; reproduced weights may carry the original authors' conditions. Verify per checkpoint, not just the repo root. EDSR-baseline ×2 ≈1.37 M params — the right size |
| **NAFNet** (megvii-research/NAFNet) | Reported **MIT** | Permitted if MIT confirmed | SPEC §7.1's recommended architecture. Checkpoints are denoise/deblur, **not ×2 SR** — so this is architecture reuse, not a usable SR initialisation |

**Licences must be re-verified at source before any use.** The table reflects search results
and secondary sources, not the `LICENSE` files themselves. F14 requires disclosing name,
link, licence and paper/model card, and getting a licence wrong is a disqualification risk,
not a footnote. Read the actual `LICENSE` file in the actual commit being used.

### Adaptation cost

1. **Single-channel stem (cheap).** Every candidate is 3-channel RGB. Adapting means summing
   or averaging the input conv weights across the RGB axis (1→C instead of 3→C) and reducing
   the output conv to C→1. Standard, ~20 lines, largely lossless for the first layer.
2. **Degradation mismatch (expensive — the decisive factor).** Every classical ×2 SR
   checkpoint is trained on **clean bicubic downsampling with no noise**. Our inputs carry
   signal-dependent noise with residual std **0.092** and ~3% of pixels outside [0,1]. SR
   networks are trained to *sharpen*; applied to noisy input they amplify noise. The
   pretrained prior is not merely unhelpful here, it is pointed the wrong way, and undoing it
   costs fine-tuning that approaches from-scratch training.
3. **Colour-tuned interior.** Interior features are tuned to RGB statistics; grayscale is a
   distribution shift on top of the degradation shift.
4. **Architecture pull.** Adopting a checkpoint means adopting its architecture, which
   competes with SPEC §7.1's ~1–3 M-param throughput target and with D7's finding that
   startup dominates.

### Why from scratch wins here

- **Data is abundant and matched.** 3200 real pairs plus unlimited synthetic re-degradation
  from GT (F15). Pretraining pays when data is scarce; it is not.
- **The model is small.** 1–3 M params at 128→256 trains fast from scratch on the local
  RTX 4060 (8 GB). This is not a regime needing a warm start.
- **The transferable part is the degradation, not the content** — and no public checkpoint
  has our degradation. What pretraining offers (natural-image content priors) is the part
  least likely to survive if the hidden test set is SEM imagery.
- **Budget.** SPEC §16 is a one-day plan; integration, stem surgery and licence diligence
  cost hours that are better spent on the matched-degradation model and on `inference.py`,
  which PD4 calls the highest-value file.
- **Disclosure overhead.** F14 requires full disclosure per resource. SPEC §6.5's own
  judgement — skip external resources for Phase 1 — applies with equal force here.
- **A subtle leakage risk.** If the released data really is DIV2K, then most public SR
  checkpoints were trained on DIV2K too. A checkpoint trained on DF2K or on DIV2K including
  its validation split may have seen the very photographs behind our test inputs. Unresolved,
  and unresolvable while the source is unidentified (D8) — another reason to avoid the
  question entirely for Phase 1.

**Per SPEC §6.5, say so explicitly in the deck:** "No external datasets or pretrained weights
used." An honest empty disclosure scores better than a vague one.

**Round 2 experiment worth running:** initialise from EDSR-baseline ×2 (Apache-2.0, 1.37 M
params, right size), adapt the stem, fine-tune on our degradation, and compare against the
from-scratch model at equal wall-clock. Report the delta. That is a measurement, not the
assumption being rejected here.

## D5 — Train/test content shift is mild; the honest failure case is broadband texture

**Decision.** Treat train and test as the same domain. Do not build domain-adaptation
machinery for the released split.

Measured like-for-like on the noisy 128×128 inputs of both splits, n=400 each:

| metric | train/NoisyLR | test_NoisyLR | ratio |
|---|---|---|---|
| spectral peakiness, median | 35.73 | 37.03 | ×1.04 |
| spectral peakiness, p90 | 68.40 | 71.17 | ×1.04 |
| gradient anisotropy, median | 1.114 | 1.136 | ×1.02 |
| strongly-periodic images | 10.0% (by construction) | **12.8%** | — |

An initial visual impression that the test set was *substantially* more periodic and man-made
did not survive measurement — the shift is real but small. Recorded because the wrong version
of this claim would have justified unnecessary domain-adaptation work. Per-image intensity
statistics also match closely (mean 0.4518 vs 0.4427, sd 0.2233 vs 0.2203).

### Failure case for SPEC §14 Slide 6

SPEC §5.4 predicts the failure case will be an *aliased dense periodic array*. The measured
worst case is different and should be reported as found.

Screening 400 pairs by the fraction of GT spectral energy above the LR Nyquist limit — energy
that provably cannot survive 2× decimation:

```
distribution: mean=0.0531 sd=0.0802 min=0.0010 median=0.0276 max=0.8046
worst: 000984.npy  0.8046
       000352.npy  0.8007
       001208.npy  0.4698
```

`000984.npy` has **80.5%** of its energy above Nyquist, but it is **broadband noise-like
texture with a flat spectrum**, not a periodic grating
(`results/eda/aliasing_failure_case.png`). Its information is unrecoverable for a different
reason than SPEC anticipated: there is no sparse structure to reconstruct, so no prior helps.

Use `000984.npy` as the honest failure case, and describe it accurately as broadband texture
rather than mislabelling it periodic aliasing.
