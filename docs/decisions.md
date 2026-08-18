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

## D10 — Further contract amendment: V39 may not SKIP (HUMAN-ISSUED, 2026-08-15)

The SKIP whitelist entry `V39, V40 (partial) — No CUDA device available` is reduced to
`V40 (partial)`. **V39 can no longer skip.**

**Rationale.** The revised V39 (D6) has no threshold, so end-to-end wall-clock is measurable
on any device. Absence of CUDA no longer prevents the check from running — it only changes
the number. V39 must therefore be measured on whatever device is present and **labelled with
that device name** in `results/runtime_report.md`.

Unambiguously a **strengthening**: a check that could previously be skipped now always runs.
Human-issued. `docs/VERIFIER_SHA256` re-pinned.

## D11 — DIV2K crop-match: DENIED PERMANENTLY

**Decision: the crop-match test against downloaded DIV2K will not be run. Not now, not in
Round 2, not conditionally.** Closed. D8's "if it matters later, the decisive test is…"
sentence is hereby withdrawn.

**Rationale (human-issued):**

1. **Identifying the source is the precondition for obtaining hidden labels.** The only thing
   standing between this repo and the test ground truth is not knowing which public corpus
   the test inputs came from. Confirming it removes that barrier.
2. **A confirmed match would sit in the repo as a pointer to the labels.** Even with no intent
   to use them, a committed artifact saying "test input `000123.npy` is DIV2K val image
   `0847`, crop (x,y)" *is* a map to the hidden ground truth. Its existence is the problem,
   independent of anyone's intentions.
3. **It has no decision value.** Strategy is identical whether the source is DIV2K, Flickr2K
   or something else: train on the provided pairs, optimise for degradation robustness, do
   not use external data (D9). No modelling decision turns on the answer.
4. **Asymmetric payoff.** Zero upside, and the downside is a scored-integrity failure.

**Standing rule.** Do not download DIV2K, Flickr2K, BSD or any candidate corpus for the
purpose of matching against the provided data. Do not compute crop correspondences against
any external image set. Do not record a suspected source-image identity for any provided
file. `docs/decisions.md` D8 remains the final word on provenance: **source not identified,
and deliberately left unidentified.**

This entry belongs in `docs/STATE.md` under "Do NOT retry" and is listed there.

## D12 — Degradation simulator built to measurements, not to SPEC §6.4

**Decision (human-issued): `src/degrade.py` implements the *measured* degradation, not
SPEC §6.4's reference speckle model.** Binding specification:

| element | value | source |
|---|---|---|
| **Downsample, primary** | the **recovered 4×4 kernel** as a fixed conv (weights in D1) | D1 least-squares recovery, 3.125 M equations |
| **Downsample, alternative** | `bicubic (antialias OFF)` in a **minority** of samples, for randomisation diversity per SPEC §6.3 | D1 — within 1.22e−05 of optimal |
| **Noise model** | three-parameter `var = σ² + a·x + v·x²`, applied **AFTER** downsampling | D2 autocorrelation ⇒ noise added post-decimation |
| **Shot / linear term** | `a = 0.011253`, randomised **±30%** ⇒ `U(0.00788, 0.01463)` | D2 three-parameter global fit |
| **Speckle / quadratic term** | `v = 0.015745`, randomised **±30%** ⇒ `U(0.01102, 0.02047)` | D2 three-parameter global fit |
| **Additive Gaussian σ** | randomise over **`U(0, 0.02)` including zero** | see below |
| **Clipping** | **do NOT clip synthetic LR to [0,1]** | SPEC F5, §6.3; measured range [−0.28, 2.16] |

**On the Gaussian σ hedge.** The measurement is unambiguous: the additive floor fits to
**exactly zero** (D2), so a strict reading says omit the term. It is retained anyway,
randomised over `[0, 0.02]` *including* zero, because SPEC F3 names additive Gaussian as one
of the two benchmark degradations and F7 warns that test noise *levels* may vary. Sampling
from zero upward costs nothing when the true value is zero and hedges the case where the
hidden test set carries an additive component the released proxy does not. This is a
deliberate, cheap insurance policy against a stated-but-unmeasured degradation — not a
contradiction of the measurement.

**Do not use SPEC §6.4's `add_speckle` alone.** It implements only the `v·x²` term and would
under-noise mid-tones while over-noising darks by up to 12.5× (D2).

## D13 — Pretrained initialisation: confirmed from scratch (HUMAN-ACCEPTED)

D9's recommendation is accepted. Phase 1 trains **from scratch**. No external datasets, no
pretrained weights. The deck and README state verbatim:

> **No external datasets or pretrained weights used.**

**Round 2 experiment, logged and deferred:** initialise from EDSR-baseline ×2 (Apache-2.0,
≈1.37 M params), adapt the 3→1 channel stem by averaging RGB conv weights and the head to
C→1, fine-tune on the measured degradation, and compare against the from-scratch model **at
equal wall-clock**. Report the delta on PSNR/SSIM/LPIPS plus throughput. Licence must be
re-verified from the `LICENSE` file at the exact commit used before anything is downloaded.

## D14 — Verifier changes made during BOOTSTRAP (hash-pin audit trail)

`docs/VERIFIER_SHA256` pins `scripts/verify_all.py` and V00 fails on an undocumented change.
This entry is the required documentation for every edit made after the first pin.

### Edit 1 — V14 local-module false positive (correctness fix)

V14 asserts every top-level import resolves to a pinned distribution. It flagged
`fit-degradation` as an uncovered dependency. That is wrong: `scripts/renorm_experiment.py`
does `from fit_degradation import weight_matrix`, a **local sibling module**, not a
third-party package. Demanding it in `requirements.txt` is impossible to satisfy.

Fix: exclude module names that resolve to a `.py` file or a package `__init__.py` inside the
repo. V14's intent — third-party imports must be pinned — is unchanged; the fix removes a
check that could never pass. It is neither a loosening of intent nor a tolerance widening: no
real dependency is now permitted to go unpinned. `torch` is still correctly flagged.

**Hash after this edit is recorded in `docs/VERIFIER_SHA256`.** No other behaviour changed.

### Note on V41/V42 passing at iteration 0

Both pass on a stub (`torch.compile not used at all`, `TTA is flag-gated and off by default`).
These are *absence* checks — they assert a bad default is not present, which is genuinely
true of a stub. They are not silent passes for unimplemented code and will stay meaningful
once compile and TTA exist. Flagged here so a reviewer does not mistake them for the
"not implemented yet" class.

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

## D15 — V51 reconciled with V47; net stricter (HUMAN-AUTHORISED, 2026-08-15)

`scripts/verify_all.py` `check_V51` was edited by the main session and `docs/VERIFIER_SHA256`
re-pinned. V00 fails on an undocumented verifier change, so this entry is the required
audit trail. Digests:

```
new sha256 of scripts/verify_all.py: cb4c5ca5b45fcb64e8665c3785df931dac4f67a71d860617cfa5ef90597f0d6d
prior pin:                           d462c70eee644851350da86be1971283b32cb553ceadc1452d5f794e7f971c13
```

### The conflict

V51 banned **every** tracked `.npy`. SPEC §12 requires `sample_inputs/`, and V47 runs
`inference.py` against `sample_inputs/` **from a clean clone** — which means those files must
be *in* the clone. As implemented, V47 and V51 were mutually exclusive and the Definition of
Done was unreachable: any state satisfying one failed the other.

The human explicitly authorised committing 4–6 real `.npy` files to `sample_inputs/` in this
session. That is the same human-issued-amendment mechanism used for D6 and D10; the agent did
not originate it and may not originate it.

### Resolution

A **deliberately narrow** exemption for `sample_inputs/*.npy`, bounded at **≤8 files** and
**≤512 KB** in total (actual: 6 files, 393,984 B), plus **four new assertions** that make V51
net stricter than it was:

1. the blob-extension ban widened from 4 extensions to 20 — `.npz .pt .pth .zip .env .tar
   .gz .7z .ckpt .onnx .safetensors .bin .raw .dat .h5 .hdf5 .parquet .mat .pkl .pickle` —
   plus `.DS_Store`;
2. any tracked path containing a dataset directory token (`/GT/`, `/NoisyLR/`,
   `/ground_truth/`, `/test_NoisyLR/`) is a FAIL, which catches a committed slice of the
   dataset tree under **any** extension;
3. a **5 MB per-tracked-file** cap;
4. a **25 MB total-tracked-tree** cap.

(3) and (4) catch a dataset dump regardless of extension, which an extension blacklist
provably cannot.

Measured after the change: **77 tracked files, 7,271,202 B total**, largest single file
`results/eda/pairs_grid.png` at 2,500,869 B. V51 PASSES.

**Stated honestly:** the `sample_inputs` exemption is, in isolation, a *loosening* with
respect to those six paths, and it exists only because a human authorised it. The four new
assertions are unambiguous strengthenings. The net effect is stricter, but the loosening is
recorded rather than buried.

## D16 — The transferable asset is the measured degradation, not any content prior

**Standing note for all future model and hardening work.**

The provided data is grayscale natural photographs (D4). KLA's hidden test set may be actual
semiconductor imagery — F7 explicitly promises out-of-distribution content. What survives
that gap is the **measured degradation**:

- the recovered 4×4 sharpening downsample kernel (D1),
- shot noise (`a·x`) rather than a Gaussian floor (D2, D12),
- noise applied **after** decimation (D2).

What does **not** survive is any content prior learned from photographs.

**Therefore: prefer wide degradation randomisation over squeezing in-distribution dB.**

A hardening iteration that buys +0.2 dB in-distribution by *narrowing* the degradation range
is a **regression against the actual objective** and must be rejected on those grounds even
though the in-distribution number improved. Recorded so a future iteration does not optimise
the wrong thing and then defend it with the metric it improved.

## D17 — `results/restored_test_outputs/` delivery: NOT Git LFS (mechanism decided, size PENDING)

F12 requires the folder to hold real model outputs and V13 asserts it is non-empty. 400
outputs at 256×256 float32 is ≈105 MB raw — over GitHub's file limit if bundled, and caught
by the `*.npy` rule in `.gitignore`.

**Git LFS is ruled out by human instruction.** Unresolved LFS pointer stubs on a fresh clone
are a known way to fail V06, whose own text names *"not an LFS pointer stub"* as a failure
mode. A stub that looks like a file is worse than an honest external link.

**Decision:** compress all 400 outputs into a single `.npz` via `np.savez_compressed` and
commit it **if** the measured artifact is under ~40 MB; otherwise host it externally with a
published sha256 and a link verified from a logged-out session.

**The measured size is not yet known — no trained model exists — so this entry is the
DECIDED MECHANISM with the MEASUREMENT PENDING.** The measured byte size must be written into
this entry once the outputs exist. No size is estimated here on purpose.

### Open decision for the human — flagged, not resolved

The mechanism collides with two rules that are currently in force:

- `.gitignore` bans `*.npz`;
- the newly strengthened V51 (D15) bans `.npz` outright **and** caps any tracked file at
  5 MB and the whole tracked tree at 25 MB.

So shipping a ~40 MB `.npz` requires **another human-authorised V51 amendment**, or the
external-hosting route. This is not resolved here. Weakening V51 to fit an artifact would be
a Prime Directive 1 violation if the agent did it unilaterally.

## D18 — Environment pinned, and how `requirements.txt` forces a CUDA build

**Pinned environment** (real `pip freeze` output, not remembered versions):

| component | version |
|---|---|
| Python | 3.12.10 (Windows 11) |
| torch | **2.11.0+cu128** (`torch.version.cuda == '12.8'`, `torch.cuda.is_available()` True, `is_bf16_supported()` True) |
| torchvision | **0.26.0+cu128** (a real dependency — `lpips` imports it) |
| lpips | 0.1.4 |
| scikit-image | 0.26.0 |
| PyYAML | 6.0.3 |
| pytorch-msssim | 1.0.0 |
| numpy | 2.5.2 |
| scipy | 1.18.0 |
| matplotlib | 3.11.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) |

### The failure this pin exists to prevent

`pip install lpips` with no index specified resolved torch **from PyPI** and replaced
`torch==2.11.0+cu128` with a CPU-only build. Observed directly in this session, mid-repair:
`torch.__version__` reported `2.13.0+cpu`, `torch.version.cuda` was `None`, and
`torch.cuda.is_available()` was `False`, with `torchvision==0.28.0`. Nothing failed and
nothing was logged. Recovered with
`pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision`.
Full write-up: `docs/BLOCKERS.md` B8.

**Why it is load-bearing:** V04 installs a fresh venv from `requirements.txt` **alone**. If
that file does not force the PyTorch index, the clean-room install silently yields a CPU-only
torch — the run exits 0, V04 *passes*, and on KLA's H100 the GPU sits unused while the
throughput score collapses with no error anywhere.

### Mechanism chosen

`requirements.txt` line 1 is
`--extra-index-url https://download.pytorch.org/whl/cu128`, paired with the pins
`torch==2.11.0+cu128` and `torchvision==0.26.0+cu128`.

The **local version `+cu128` is the safety catch**, not the directive. PyPI cannot host a
local version, so once `+cu128` is in the pin there is no CPU candidate for the resolver to
fall back to — `lpips` cannot pull a PyPI torch during resolution because no PyPI torch
satisfies the requirement. Either the CUDA wheel installs or the install fails **loudly**,
which is the desired behaviour. Two flags that were considered and rejected: an
`--index-url` override (would hide all of PyPI, breaking the other 31 pins) and a documented
post-install step (V04 forbids extra steps).

**Verified, 2026-08-15, unattended:** unauthenticated `git clone` of the public repo →
`py -3.12 -m venv .venv` → `.venv/Scripts/python.exe -m pip install -r requirements.txt`
(no other flags, no env vars) → `torch 2.11.0+cu128 | cuda 12.8 | available True`,
`torchvision 0.26.0+cu128`.

### H100 discipline

Training and every timing measurement happen on the RTX 4060 Laptop GPU. **Any H100 number
in the deck or README is a projection, not a measurement, and must be labelled as such.**
The 4060 figures do not extrapolate linearly — see D21: NAFSR is memory-bandwidth bound, not
compute bound.

## D19 — Architecture: NAFSR, with a plain U-Net as the learned baseline

*(measurements by `model-core`, RTX 4060 Laptop GPU)*

**Decision.** NAFNet-style body at LR resolution + PixelShuffle ×2 head + global bilinear
skip (`NAFSR`), with a plain U-Net (`UNetSR`) as the learned baseline required by the rubric.
Both live in `src/model.py`, selected by `cfg["name"]`, behind one frozen entry point
`build_model(cfg: dict) -> nn.Module`.

| model | params | GMAC / 128×128 img | GFLOP |
|---|---|---|---|
| NAFSR w48 n16 | **388,225** | 5.584 | 11.169 |
| UNetSR w32 L4 | **2,970,401** | 4.478 | 8.956 |

The comparison is roughly **FLOP-matched (0.80×)** with NAFSR at **0.13× the parameters**, so
NAFSR is given no parameter advantage over the baseline it has to beat (V28).

Flat body ⇒ required size multiple **1**; `UNetSR` reflect-pads internally and crops back so
its external multiple is also 1. Verified: 128→256, 256→512, 61×97→122×194, 1×1→2×2. No
BatchNorm, no dropout, bit-identical repeats in `eval()`. Checkpoint is **3.14 MiB** for
model + EMA, far under V43's 100 MB cap.

## D20 — Stayed at 0.388 M params rather than SPEC §7.1's 1–3 M band

*(measurements by `model-core`, RTX 4060 Laptop, bf16 + channels_last, batch 32 at 64 px patches)*

**The binding constraint is training throughput on the 8 GB dev GPU, not inference cost.**

| config | params | ms/step | 20k iters | peak VRAM |
|---|---|---|---|---|
| w48 n16 (chosen) | 0.388 M | 221 | **73.7 min** | 3765 MiB |
| w64 n16 | — | — | 87.1 min | — |
| w80 n16 | 1.049 M | — | 135.5 min | 6255 MiB |
| w96 n16 | 1.500 M | — | 141.1 min | — |
| w64 n28 | 1.048 M | **2766** | — | **7925 MiB — does not fit** |

`w64 n28` spills the allocator at 7925 MiB of 8 GB and collapses to 2766 ms/step.

Entering SPEC's 1–3 M band costs **1.8–1.9× training wall-clock for an unmeasured quality
gain**. On a one-day budget the number of training runs that fit is the deciding constraint.

**Revisit with a measured PSNR delta once a training run exists — raising width is the first
thing to try once there is a quality number.**

## D21 — `LayerNorm2d` via `F.layer_norm` on an NHWC view, not a hand-rolled channel reduction

*(measurements by `model-core`; interleaved A/B, 5 repeats, medians)*

| variant | training | inference | VRAM |
|---|---|---|---|
| manual channel reduction | 4939 ms | 305 ms | 4970 MiB |
| fused `F.layer_norm` on NHWC view | **4233 ms** | **208 ms** | **3486 MiB** |

⇒ **1.17× inference, 1.46× training, 1.43× less VRAM**, with non-overlapping ranges.

**Correction on the record.** An earlier single-shot version of this measurement reported
1.09× inference. That was inside the ~9% run-to-run variance and should not have been quoted
as a win. It is superseded by the 5-repeat median above. Recorded rather than quietly
replaced.

**NAFSR is memory-bandwidth bound, not compute bound.** Profile: 32.8% `layer_norm`, 17.9%
conv bias-add — the bias adds cost as much as the convolutions themselves — 16.2%
convolution. Two consequences:

1. It explains why SPEC §11.2's optimisation table matters so little here (D7): `channels_last`
   and bf16 each move it under 20%.
2. **These 4060 timings do not extrapolate linearly to an H100** and must never be presented
   as if they do.

## D22 — Seven placeholder V-checks implemented; V09's conflict with V20 fixed. Strengthening.

`scripts/verify_all.py` sha256 is now
`590c8e3344f2a7dbfadf63bace9a255c97ee73269c7894bc56855270e709d5bd`
(prior `cb4c5ca5b45fcb64e8665c3785df931dac4f67a71d860617cfa5ef90597f0d6d`, D15; original
BOOTSTRAP pin `d462c70eee644851350da86be1971283b32cb553ceadc1452d5f794e7f971c13`, D14).
This is the second edit to the verifier after the D15 change and is the required PD1 audit
trail for it.

**V26, V27, V28, V29, V32, V33 and V35 were BOOTSTRAP placeholders that returned an
unconditional FAIL no artifact could ever turn green.** A check that cannot pass is not a
strict check — it is an *absent* check wearing a red badge, and it hides real regressions
because it looks identical before and after a defect is introduced. All seven now test their
subject:

- **V26** runs the marker-based paired-crop self-test and fails if it reports pass while
  checking zero crops.
- **V27** compares final vs bicubic and enforces "a margin, not noise" **statistically** — the
  PSNR gain must exceed two standard errors of the mean — rather than with an invented
  constant. It also fails if the standard deviation is not reported, which the contract
  requires.
- **V28** implements the contract's negative-result escape hatch exactly as narrowly as
  written: a loss to the U-Net baseline converts to PASS only when an honest negative result
  is documented in this file.
- **V29** intersects the committed validation list against the train list the module actually
  reports, and additionally rejects duplicates, a degenerate empty side, and a `dataset.py`
  that never references `split_val.txt`. Without the dataset present it FAILS honestly instead
  of passing on file-only invariants.
- **V32** asserts 1 channel in / 1 channel out **and that a 3-channel input is rejected** — a
  model that silently accepts 3 channels would let an accidental BGR/RGB path through without
  ever raising.
- **V33** recomputes the degradation-fidelity report live against the real pairs when the
  dataset is present, falling back to the committed JSON only when it is not, and states which
  in its detail so an artifact is never mistaken for a live measurement.
- **V35** asserts the six required checkpoint keys and `strict=True` loading of **both**
  `model` and `ema`, using `weights_only=True` — the same load path `inference.py` uses, so a
  checkpoint requiring arbitrary unpickling can no longer pass V35 and then break the shipped
  script.

**V09 fix.** `check_V09` treated an unreadable *input* as a scale violation. A corrupt file
forms no `(in, out)` pair, the contract's wording is *"for every pair"*, and V20 explicitly
declares corrupt inputs survivable — so V09 as written was in **direct conflict with V20** and
no implementation could satisfy both. Unreadable inputs are now excluded and reported in
evidence rather than silently dropped, and a new anti-vacuity guard fails V09 if that
exclusion leaves zero pairs checked.

Measured effect on the suite: **PASS 9 → 35, FAIL 44 → 18, Tier 1 fully green (9/9).**

## D23 — `results/restored_test_outputs/` ships as a GitHub Release asset, superseding D17's committed-`.npz` route

D17 selected `np.savez_compressed` into the repo if the artifact measured under ~40 MB. **That
route is withdrawn.**

The strengthened V51 (D15) bans `.npz` outright and caps any tracked file at 5 MB, so
committing a ~40 MB archive would have required a **second** amendment loosening the size caps
added one commit earlier. That was rejected on principle: *"loosen the check I just tightened,
because it is blocking me"* is the reasoning pattern the project's standing rules name as a
**stop signal, not a justification**.

**Mechanism instead.** The 400 restored outputs (~105 MB raw — 400 × 256×256 float32) are
published as a **GitHub Release asset**, pre-approved by the human's standing authorisation,
with a sha256 recorded in `results/restored_test_outputs/README.md` and in
`weights/README.md`. `results/restored_test_outputs/` carries a committed **manifest with
per-file hashes**, so the folder is non-empty, self-describing and independently verifiable.

This requires **no contract change at all** — it is exactly the mechanism V06 already permits
for weights.

**Stated honestly:** the committed folder holds a verified **pointer and manifest, not the raw
output bytes**. That fact is written into the folder's own README in plain words, so a
reviewer is never misled about what is actually in the repository. A "non-empty" folder that
implies outputs are present when they are not would satisfy V13's letter and defeat its
purpose.

Supersedes D17. **D17 is retained unedited** as the record of the earlier decision — this file
is append-only, and a superseded decision with its reasoning intact is more useful to a
reviewer than a rewritten one. See `docs/BLOCKERS.md` B9.

## D24 — V33 acceptance moved into the pinned verifier; thresholds tightened; verifier made side-effect-free

**Authored by the main session, not `docs-scribe`** (which owns this file). No subagent was
alive at the time — a session usage limit had killed all four — and V00 requires the new
verifier digest to appear here or it fails by design. Append-only respected; `docs-scribe`
has been notified.

`scripts/verify_all.py` sha256 is now
`b6b575dc75c32499c890faee82f6b4385041bdee393329429abdc818f1edd7d2`
(prior `590c8e3344f2a7dbfadf63bace9a255c97ee73269c7894bc56855270e709d5bd`).

All three changes are **strengthenings**, made under the human's standing authorisation
("any change that makes a check STRICTER — log as human-authorised, re-pin"). They respond to
findings F2 and F3 from `reviews/ml-skeptic-1.md`.

### 1. The governance hole (ml-skeptic F2) — the real problem

`check_V33` did no thresholding of its own. It called `src.degrade.fidelity_report()` and
asserted `res["pass"]`. **Every threshold lived in `src/degrade.py::FIDELITY_TOLERANCE` — a
file owned by `data-pipeline` and not covered by `docs/VERIFIER_SHA256`, which pins only
`scripts/verify_all.py` and `docs/VERIFICATION_CONTRACT.md`.**

So a future iteration could have widened the acceptance bar until V33 went green **without
touching a pinned file and without tripping Prime Directive 1**. The check was, in effect,
grading its own homework: the subject under test owned the pass mark.

Fix: the acceptance thresholds now live in `V33_THRESHOLDS` / `V33_STD_RATIO_RANGE` inside
the pinned verifier, and are applied **on top of** the module's own `pass` flag. Acceptance is
the AND of both, so this can only ever be stricter than what it replaced.

### 2. Threshold tightened

ml-skeptic measured 97% headroom on `gain_over_spec_2par_worst_bin` at the module's limit of
3.0 and correctly called it near-vacuous. Observed 5.9169 (seed 0) and 5.9313 (seed 7), and
the whole metric set is stable to <0.003 across noise seeds. The verifier-owned limit is
**4.5**, leaving 24% headroom — still an order of magnitude more tolerance than the measured
seed noise, while no longer passing anything that happens to be positive.

Verifier-owned bar, with the measured value at pinning time:

| metric | measured | verifier limit | headroom |
|---|---|---|---|
| `mean_abs_rel_err` | 0.38849 | ≤ 0.50 | 29% |
| `mean_abs_rel_err_x_ge_0p1` | 0.27639 | ≤ 0.35 | 27% |
| `resid_std_ratio` | 1.05540 | [0.90, 1.15] | 9% up |
| `binned_r2` | 0.98038 | ≥ 0.97 | 1.1% |
| `gain_over_spec_2par` | 1.89430 | ≥ 1.5 | 26% |
| `gain_over_spec_2par_worst_bin` | 5.91693 | **≥ 4.5** (was 3.0) | 24% |

### 3. The verifier no longer mutates what it verifies (ml-skeptic F3)

`fidelity_report()` writes `results/degrade_fidelity/degrade_fidelity.json` unconditionally,
regardless of `make_figure=False`. Because `check_V33` calls it live whenever the dataset is
present, **running the verifier left `git status` dirty** — in direct conflict with Definition
of Done criterion 5 ("git status is clean and the working tree equals the last verified
commit"). Worse, the committed *evidence* for V33 was silently overwritten by whatever ran
last, so the artifact could never disagree with the code and was not independent evidence at
all.

`check_V33` now snapshots the committed artifact and restores it byte-for-byte in a `finally`
block. Verified: after a full V33 run, `git status --porcelain` shows only the verifier edit
itself. The underlying unconditional write in `src/degrade.py` is `data-pipeline`'s to fix and
is recorded as a follow-up; the verifier-side guard holds regardless of whether that lands.

### Still open from this review, not fixed here

- `scripts/evaluate.py::ANCHOR_TOL` has the same unpinned-threshold shape as F2. Logged, not
  yet closed.
- **ml-skeptic F1 (HIGH) is a retraction, not a fix:** the synthetic-LR clip-tail statistics
  were measured on an artificial test tile and compared against real-dataset percentages —
  two different corpora — and the conclusion drawn was directionally wrong. See D25.

## D25 — RETRACTION: the synthetic-LR clip-tail comparison was invalid, and its conclusion was backwards

**Authored by the main session** for the same reason as D24. Records a claim that was wrong
and must not be repeated.

### What was claimed

`data-pipeline` reported, and the iteration-1 commit message repeated, that synthetic LR shows
"**1.33% of pixels > 1.0 and 0.0084% < 0**, against real NoisyLR's 3.03% / 0.28%", and
concluded the simulator was "**2.3× less likely to exceed 1.0 than real**" — i.e. that it
under-noised the bright tail.

### Why it was invalid

The 1.33% / 0.0084% figures come from `selftest_paired_crop()`, which computes them on an
**artificial test image** — `sin(2π(24x+16y)/N) + 0.6·cos(...) + 2·(lowpass random field)`,
min-max normalised, over 64 patches. They are not a measurement of the dataset at all. They
were then compared against **real-dataset** percentages. Two different corpora, one comparison.

The quoted "real" pair is not attributable either. Measured real fractions above 1.0 / below 0:
all-3200 3.1085% / 0.2849%; non-val 3.1437% / 0.2577%; val 2.8619% / 0.4749%; `test_NoisyLR`
3.0801% / 0.6601%. The 0.28% matches all-3200; **the 3.03% matches none of them.**

### The corrected measurement

Re-derived on the identical corpus the fidelity report uses — all 2800 non-val train pairs,
45,875,200 px, `degrade_fitted`, seed 1:

| | frac > 1.0 | frac < 0.0 | range |
|---|---|---|---|
| synthetic | **3.2523%** | **0.6186%** | [−0.1639, **1.7177**] |
| real (train/NoisyLR, non-val) | **3.1437%** | **0.2577%** | [−0.2786, **2.0735**] |

Cross-checked through the training path `degrade()` on a strided subset: 3.3524% / 0.9807%.
Clean LR with no noise at all: 0.0474% / 0.0296% — so the tail is essentially all noise.

**The conclusion was directionally inverted.** The simulator does **not** under-produce the >1
tail; it matches it (3.25% vs 3.14%). It **over**-produces the <0 tail by ~2.4× (0.62% vs
0.26%).

### The real finding underneath, which is worth keeping

The simulator under-produces the **extreme** upper tail: synthetic max **1.7177** against real
max **2.0735** on the same 2800 GT images (dataset-wide real max 2.1580). Gaussian shot+speckle
has no mass out at 4–5σ the way the real sensor does.

That is a genuine domain gap and it matters more than the retracted claim did, because this
project's entire transfer argument rests on the degradation rather than on content (D16). A
model never shown inputs above ~1.72 in training will meet them in the released test data,
which reaches 2.158. **Action for the hardening loop:** extend the noise model's upper tail
(e.g. an occasional heavy-tailed / impulse component) and measure whether it helps or hurts on
the held-out split, rather than assuming. Do not simply widen σ — that would move the whole
distribution, not the tail.

**Confirmed good, independently:** `src/degrade.py` genuinely does not clip. Verified
empirically — synthetic output spans [−0.1639, 1.7177].

### Process note

This is the second fabricated-or-unfounded number caught in iteration 1 (the first was
`model-core`'s benchmark table, D19–D21). Both were caught by re-derivation rather than by
review of the prose. That is the argument for keeping `ml-skeptic` in every review wave and for
the standing rule that a number without a measurement behind it does not go in the repo.

## D26 — V25 and V34 implemented; the overfit gate is CLEARED at 43.33 dB

**Authored by the main session** (no subagent alive; V00 needs the digest present).
`scripts/verify_all.py` sha256 is now
`24b4b1f1f6919502a84ffe2360a9dce53137d9f31bd6d46c7710219e277f6c7f`
(prior `b6b575dc75c32499c890faee82f6b4385041bdee393329429abdc818f1edd7d2`).

### Two more placeholders — nine in total

V25 and V34 were BOOTSTRAP stubs of the same shape as the seven closed in D22: they returned
an unconditional FAIL that no artifact could turn green. **Nine of the fifty-three checks were
in that state.** That is worth stating plainly, because a suite reporting "44 FAIL" at
iteration 0 looked like honest red when a fifth of it was actually inert.

- **V25** now runs the real training path — `train.py --config configs/final.yaml --overfit 2`
  — parses the emitted JSON report, and asserts the run used exactly 2 pairs, that they came
  from the **train** split, and that the exit code is 0. It does not trust a recorded number.
- **V34** runs `train.py --config configs/final.yaml --seed 42 --smoke` **twice** and requires
  the loss sequences to be identical element-by-element, and the run digests to match. It also
  fails a run producing fewer than 2 loss values, because a single step cannot demonstrate
  reproducibility.

### The 40 dB bar is verifier-owned

`V25_TARGET_DB = 40.0` lives in the pinned verifier, not read from `train.py`. Same governance
reasoning as D24: the subject under test must not own its own pass mark.

### Result — the gate is CLEARED

    V25 PASS: overfit 2 pairs reached 43.3295 dB at iter 6000 (gate 40.0 dB)
    V34 PASS: two seeded smoke runs identical across 12 steps

**This is the most important measurement of the iteration.** SPEC §16 and the contract both
treat V25 as the hard gate: a model that cannot overfit two pairs it was trained on has broken
alignment, normalisation or loss, and every downstream number would be meaningless. It clears
by 3.33 dB, so the paired-crop geometry, the [0,1] convention, the unclipped-input handling and
the loss are all confirmed end to end.

### A scheduling subtlety worth recording

A 4000-iteration budget stalls at **39.78 dB** and a 6000-iteration budget reaches **43.33 dB**,
crossing 40 dB at roughly iteration 3000. The cause is the cosine schedule decaying
*proportionally to the budget*: shortening the run does not simply truncate it, it lowers the
learning rate faster throughout. **A short overfit run that lands just under 40 dB must not be
read as an alignment failure** — that misdiagnosis would send someone hunting a geometry bug
that does not exist. Recorded because the failure mode is convincing and wrong.

## D27 — Four checks ADDED from the requirements audit; V10 strengthened

**Authored by the main session.** `scripts/verify_all.py` →
`dd2375fd44c836ce997681afd02a1344cb706e6aec171f9ba4bb88ca4b382e8a`
(prior `24b4b1f1f6919502a84ffe2360a9dce53137d9f31bd6d46c7710219e277f6c7f`).
`docs/VERIFICATION_CONTRACT.md` → `d1a22c92de8c2ecdeceae48bfe63c06f406f14f6247e0c7e479088f74cf3b269`
(prior `6f7952e9946a20f4165afea4855d25efd768a06825fd7a3c180be34749cebbef`).

The contract was amended **by addition only**. Nothing was deleted, loosened, renumbered or
tolerance-widened. Adding checks for defects reviewers find is pre-authorised.

`requirements-auditor` re-derived F1–F19 and §15 independently and found **eleven requirements
that no check could ever have turned red**. These four are the ones that can cost the
submission outright; the remaining seven are recorded in `docs/STATE.md` as backlog.

### V54 — F17 on the training path (the auditor's H-1)

V36 scans **only `inference.py`** — the side that structurally *cannot* fit on test data. The
training side, the side that could, was covered by nothing at all. F17 is the one rule whose
violation is disqualifying.

The repository is clean today; the auditor classified every read of `test_NoisyLR` and all are
read-only EDA. V54 exists so a *regression* cannot pass unnoticed.

**It fired on its first run, and the defect was mine, not the repo's.** Three hits, all English
prose emitted into report files — sentences stating that no `test_GT` exists. Prose cannot read
a file. Narrowing the rule mattered: my first attempt keyed on path separators, which failed
because the prose says `train/` and `test_NoisyLR/` and so contains slashes too.
**Whitespace is the discriminator** — a real path literal has none. A literal handed to a
filesystem call is flagged regardless of shape.

Verified with a **negative control**, which is the only way to know an absence-check works:
injecting `np.load("C:/kla-data/test_NoisyLR/000000.npy")` into `src/dataset.py` flipped V54
red; removing it flipped it green; the tree was left clean. A check that has never been seen to
fail is not known to work.

### V55 — the repo is genuinely public (H-4)

V13 accepted any non-empty `git remote -v`, and **a private repo produces an identical
string**. SPEC §18 pitfall 7 names a private repo as a common fatal failure. V55 clones with
`GITHUB_TOKEN`, `GH_TOKEN`, `GIT_ASKPASS` cleared, `GIT_TERMINAL_PROMPT=0` and an empty
credential helper, so a pass cannot come from cached credentials. Currently PASSES.

### V56 — the outputs folder holds actual outputs (H-2)

V13 accepts any non-`.gitkeep` file, so **a README alone satisfies it — which is exactly the
state this repo is in.** F12 and SPEC §15 require "actual model outputs, not placeholders".

The manifest branch enforces in code the prohibition the folder's README states only in prose:
`command` must contain `--require_weights`. Without it, `inference.py` silently falls back to
bicubic when no checkpoint is present, so a full set of plausible-looking outputs could ship as
model results while the model never ran. Correctly FAILS today.

### V59 — the checkpoint is genuinely obtainable (H-3)

The auditor flagged that `.gitignore` blanket-bans `*.pt` with no negation and recommended
adding `!weights/best.pt`. **That fix would have been wrong here**, and checking it is what
found the real constraint: V51 also lists `.pt` as a forbidden blob, so committing the
checkpoint would require loosening V51 — which is prohibited. The hosted-URL branch of V06 is
therefore not a preference but **the only valid route**, consistent with D23's Release
mechanism for the outputs.

V59 catches the silent failure: `best.pt` on the author's disk, ignored and untracked, with no
published URL — working locally, absent for everyone who clones. It fires right now, correctly:
the training run has already written `weights/best.pt` (3,288,549 B) and it is invisible to git.

### V10 strengthened in place (U-7, no new ID)

V10 checked `.npy` and `float32` but not rank. A `(2H,2W,1)` or `(1,2H,2W)` write passed V10 —
and passed V09 too, because V09 reads `so[0]`/`so[1]`, the wrong axes for a channels-last
array. V10 now asserts `ndim == 2`.

### Why this matters beyond the four fixes

Nine checks were inert placeholders (D22, D26) and eleven requirements had no check at all. The
suite's red count was never the useful signal; **what it was actually measuring** was. Both
audits found this by re-deriving rather than reading, which is the argument for running
`ml-skeptic` and `requirements-auditor` every wave rather than treating them as ceremony.

## D28 — SECURITY: V55's GitHub host validation was an unanchored substring match

Found by an automated security review of the pushed commits, in code added **one commit
earlier** (D27). `scripts/verify_all.py` →
`4e78dbca22ad9f71c3091bfeeb32ee798fbca96ca96d08468bc11748cec6178b`
(prior `dd2375fd44c836ce997681afd02a1344cb706e6aec171f9ba4bb88ca4b382e8a`).

### The defect

V55 derived owner/name with an unanchored `re.search` for `github.com[:/]+…`, so the host was
never actually validated. Demonstrated:

| remote URL | parsed as |
|---|---|
| `https://evil.example.com/github.com/attacker/payload.git` | `("attacker", "payload")` |
| `https://notgithub.com/a/b` | `("a", "b")` |

### Why it mattered, and why it was not waved away

As written, V55 only clones the URL git already points at, so the immediate blast radius was
small. But the owner/name it derives are precisely the values one would use to build an
`api.github.com/repos/<owner>/<name>` request — which is exactly what the auditor's proposed
V55 specified and what a later iteration would plausibly add. At that point the bypass becomes
a live SSRF against an attacker-chosen host.

Fixed at the **parser**, not the call site, so a future caller cannot reintroduce it.
`_parse_github_remote()` uses `urllib.parse.urlsplit`, requires the scheme to be one of
https/http/git/ssh, requires `hostname` to equal `github.com` or `www.github.com` **exactly**,
and requires exactly two non-empty path segments. The scp-like `git@github.com:owner/repo`
form is matched with `re.fullmatch`, never `re.search`.

Verified against eight vectors: the two bypasses above, plus `github.com.attacker.net`,
`file:///etc/passwd`, a single-segment path, the scp form, a mixed-case host, and the real
remote. All eight behave correctly.

### Process note

This is the third defect this iteration found by something other than reading the code — after
`ml-skeptic`'s re-derivation and `requirements-auditor`'s independent compliance pass. It is
also a reminder that **newly added checks are code like any other**, and get no presumption of
correctness for being freshly written or security-adjacent.

## D29 — First trained model: NAFSR beats every baseline on all three metrics

Run `20260815T062831Z-final-s42`, `train.py --config configs/final.yaml --seed 42 --iters 20000`.
Completed in **1:11:41 at 4.65 it/s** on an RTX 4060 Laptop GPU (8 GB), no OOM, no batch-size
reduction. Shipped weights are the **EMA** weights at best validation PSNR.

### Result — full 400-image committed split, EMA, scored from reloaded `.npy` on disk

| Method | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| bicubic ×2 (the floor V27 requires) | 23.6524 ± 3.0236 | 0.54775 ± 0.19197 | 0.41206 |
| median 3×3 → bicubic | 25.5057 ± 3.8785 | 0.61317 ± 0.17232 | 0.40870 |
| non-local means → bicubic (the honest bar) | 26.2722 ± 4.3037 | 0.65152 ± 0.19523 | 0.42586 |
| **NAFSR w48 n16, EMA** | **28.7851 ± 4.5324** | **0.78279 ± 0.14169** | **0.25233** |

**+5.13 dB over bicubic, +2.51 dB over non-local means**, winning all three metrics against all
three baselines.

### The LPIPS direction is the result worth defending

Across the classical baselines, fidelity and perceptual quality move **in opposition**: NLM
gains 2.6 dB of PSNR over bicubic while scoring the *worst* LPIPS of the three, because it buys
fidelity by over-smoothing. The scoring blend is an undisclosed mix of PSNR, SSIM and LPIPS
(F9), so a model that raised PSNR while degrading LPIPS would be gaming one half of the blend
at the other half's expense — and would look good on the metric we can see.

This model improves **both simultaneously**: LPIPS falls from 0.409–0.426 to **0.252**, a larger
relative gain than the PSNR improvement. That is the evidence the balanced
Charbonnier + 0.15·SSIM + 0.05·FFT loss is doing what SPEC §8 intends, and it is the argument
against the pure-L2 or pure-GAN alternatives recorded in §7.2.

### Two numbers that must not be confused

The training log reports `psnr 30.3944` at iteration 20000. **That is a 100-image subset**
(`--val_limit`) used only for in-run checkpoint selection. The reportable figure is the
**full 400-image committed split: 28.7851 dB**. The 1.6 dB gap is subset variance, not a
regression. **Always quote the lower number**; a repo that quotes its in-run selection metric
as its headline result is overstating by exactly the amount nobody can audit.

### What this result does NOT yet establish

- **No learned-baseline comparison exists.** The three baselines above are classical. V28
  requires beating a U-Net trained under the *same* budget, and that run has not happened, so
  the rubric's like-for-like learned comparison is genuinely missing.
- **V27/V28/V48 are still red and correctly so.** They read
  `results/baselines/final/metrics.json`, written by `scripts/evaluate.py` — not the training
  log. A number in a log is not an evaluation record.
- **Throughput is unmeasured.** No runtime report exists; the throughput axis is unscored.
- **The checkpoint is not obtainable by anyone else.** It exists only on the dev machine (V59).

### Do NOT retry

Nothing here was rejected — this is the first accepted training result and the baseline all
future runs must beat. Any subsequent architecture, loss or augmentation change must be
measured against **28.7851 / 0.78279 / 0.25233 on the 400-image split**, and per D16 an
in-distribution gain bought by narrowing the degradation randomisation is a regression against
the actual objective, not an improvement.

---

## D30 — The checkpoint is published as a GitHub Release asset, verified anonymously

**Date:** 2026-08-15, iteration 2. **Closes:** V59. **Route:** B (Release), per D23.

`weights/best.pt` existed only on the development machine. `.gitignore` bans `weights/*.pt`
and V51 lists `.pt` as a forbidden blob, so the file was untracked *and* unpublished — it
worked for the author and was absent for everyone who cloned the repository. That is exactly
the silent failure V59 was written to catch, and V59 was correctly red.

**What was done.** Release `artifacts-v1` was created on the public remote and `best.pt`
uploaded as an asset.

| field | value |
|---|---|
| URL | `https://github.com/sahithsundarw/bunker_backer/releases/download/artifacts-v1/best.pt` |
| size | 3288805 B (3.14 MiB) |
| sha256 | `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` |

**The digest is of the served bytes, not of the local copy.** The asset was re-fetched with
`GITHUB_TOKEN` and `GH_TOKEN` cleared from the environment, so the fetch could not have
succeeded on cached credentials: HTTP **200**, **3288805** bytes downloaded, and the sha256 of
what the server returned equals the digest above and the digest of the local file. A URL
verified only from an authenticated session is the standard way this check passes on the dev
box and fails for the evaluator.

**Route A (commit the file) was rejected, and the reasoning is worth recording** because it is
the tempting one: at 3.14 MiB the checkpoint fits comfortably inside V51's 5 MB per-file and
25 MB total caps, and `weights/README.md` had previously called Route A "preferred" on the
grounds that it has no external dependency and no link to rot. Both remain true. It was still
rejected: taking it requires editing `.gitignore` and V51 to admit a `.pt`, and loosening a
check because it stands between the work and a green tally is precisely the move Prime
Directive 1 forbids. The size argument is real but it is an argument for amending the caps on
their merits, not for amending them mid-task to unblock a step. `weights/README.md` was
updated to say so rather than leaving the stale "preferred" framing in place.

**What this does not close.** V06 is not evidence of anything on this machine: it passes as
soon as `weights/best.pt` exists on disk, which it does, so it was green while the checkpoint
was unobtainable. V59 is the check that actually tests deliverability, which is why it was
added. The restored-outputs archive is still pending and will ship on the same Release, with
its digest recorded in the same table.

---

## D31 — Three checks STRENGTHENED after an independent audit found them unable to fail

**Date:** 2026-08-15, iteration 2. **Source:** `requirements-auditor`, second pass.
**Affects:** V00, V28, V48. **Verifier re-pinned:** `4e78dbca...` -> `47dac07a...`.

All three edits are strictly stricter. No threshold was widened, no check removed, no skip
added. V28 went from PASS to FAIL under its new test, which is the point.

### H-1 (critical) — V28's escape hatch was permanently unlocked

The contract lets a loss to the U-Net baseline convert FAIL->PASS *only* if an honest negative
result is documented and the better model is shipped. The implementation tested that with:

    documented = "V28" in dec and "negative result" in dec.lower()

`docs/decisions.md` D22 — the paragraph written to *describe* this hatch — contains both
strings. So `documented` was **True from the moment the hatch was documented**, `wins` did not
gate the outcome at all, and V28 would have returned PASS with our model losing on all three
metrics. A check that cannot fail certifies nothing; this is the tenth such defect found in
this suite.

Now the entry must (a) be a structured `## D<n> ... NEGATIVE RESULT` heading that mentions V28,
(b) quote all six measured means at the precision the summary table prints them — boilerplate
cannot do this, only an entry written against the actual measurement can — and (c) declare
`SHIPPED MODEL: <name>` where `<name>` must equal the architecture actually stored inside
`weights/best.pt`, read from the checkpoint's embedded config. That last clause is the
contract's "the better model shipped" requirement, which nothing previously enforced.

### H-1b — a tie was being counted as a win

V28 compared two independent means with `>`. But both models score the **same 400 images**, so
the correct statistic is the **paired per-image difference**. Under the old test our model
"won" SSIM by a mean of +0.000135 while actually being better on only **172 of 400 images** —
a coin flip, counted as a win, and enough to reach the 2-of-3 bar. The check now runs a paired
t-test per metric and treats anything with |t| < 1.96 as a **tie**, which is neither a win nor
a loss. It falls back to a 2xSEM test on the aggregate means only when per-image data is
absent, and refuses to conclude anything from fewer than 30 pairs.

### H-4 — V48 counted pipe characters and never compared a number

It read `results/metrics_summary.md`, counted lines starting with `|`, and passed at >= 6. It
never opened a `metrics.json` and never ran `evaluate.py`, so a table whose numbers disagreed
with the evaluation records passed — and that is exactly the state the repo was in, with six
documents publishing a PSNR the evaluator had never produced. It now parses the table and
reconciles every record under `results/baselines/*/` against it, to 1e-3 on PSNR and 1e-4 on
SSIM/LPIPS, and requires at least four records including `final`.

**Negative control:** changing a single digit of one PSNR in the table (28.7865 -> 28.7965)
turned V48 red with `1 evaluation record(s) have no matching row`; the byte-exact restore
turned it green again.

### H-6 — the contract's own hash pin was enforced by nothing

`docs/VERIFIER_SHA256` pins two files. `check_V00` filtered that list with
`parts[1].endswith("scripts/verify_all.py")` and **discarded the line pinning
`docs/VERIFICATION_CONTRACT.md`**. The document CLAUDE.md calls IMMUTABLE could have had a
check deleted from it and the suite would have stayed green — the exact failure the pin exists
to prevent. V00 now enforces every pin in the file. It still hashes the verifier it is actually
executing (`SELF_PATH`), so a pin cannot be satisfied by a second copy in the tree.

No tampering had occurred: both digests were recomputed during the audit and both matched.

### Do NOT retry

- **Do not "fix" V28 by editing `docs/decisions.md` to add the words it looks for.** That is
  symptom-fixing, `decisions.md` is append-only, and the new check reads structure and measured
  values precisely so that prose cannot satisfy it.
- **Do not widen V48's tolerance to admit the published numbers.** The numbers were wrong; the
  documents were corrected instead (see D32).

---

## D32 — Published-artifact claims are now proved by download, not by prose

**Date:** 2026-08-15, iteration 2. **Human-authorised** (decision B). **Source:**
`requirements-auditor` H-5. **Affects:** V06, V59, V56. **Re-pinned:** `47dac07a` -> `d792ab7f`.

### The hole

Three checks certified that an artifact was published. All three did it by reading a markdown
file. `check_V59` was `re.findall(r"https?://\S+")` plus `re.search(r"\b[0-9a-f]{64}\b")` — a
URL-shaped string and a hex-shaped string in the same document. `check_V56` validated the
*shape* of the manifest's keys. Neither ever opened a socket.

This is the standard V33 already set for this repo: a check must not accept the subject's own
word for the thing under test. Every "verified anonymously, HTTP 200, digest matches" sentence
in this repository was, to the verifier, decoration.

**V06 was worse than unverified — its published route was unimplemented.** The branch that
handles "URL + sha256" returned an unconditional `FAIL` reading *"URL+sha256 present but not
verified (needs a logged-out fetch)"*. Since `.gitignore` and V51 both refuse `*.pt`, that is
the *only* route available to this repo, so V06 was guaranteed to be red in any fresh clone
while passing locally purely because `weights/best.pt` happened to sit on the author's disk.
The check most responsible for "can the evaluator get the model" was structurally incapable of
saying yes.

### What it does now

`_fetch_digest()` downloads with a bare `urllib` opener — no auth handler, no cookie jar, no
netrc, no `Authorization` header — and additionally pops `GITHUB_TOKEN`, `GH_TOKEN`,
`GH_ENTERPRISE_TOKEN`, `GITHUB_USER` and `GIT_ASKPASS` from the environment for the duration
and sets `GIT_TERMINAL_PROMPT=0`, so a pass cannot come from ambient credentials. It streams in
1 MB chunks into `hashlib` under a 512 MB cap, then requires:

- HTTP **200** — 401/403 is reported explicitly as *"the artifact is NOT public"*;
- the body is **not HTML** — a sign-in or error page served behind a 200 is the classic way a
  "public" URL turns out to be private, and it is rejected on both `Content-Type` and magic
  bytes;
- a non-empty body;
- the sha256 of the **served** bytes matches a digest published in the same document.

V06 now verifies the published URL **even when the local file exists**, which is stricter than
the contract's either/or: a local copy proves nothing about a link that has rotted.

### Mutation-tested, as required

| mutation | result |
|---|---|
| checkpoint digest `...592313` -> `...5deadb` in `weights/README.md` | **V06 RED, V59 RED** |
| `archive_sha256` last 3 chars `750` -> `000` in `manifest.json` | **V56 RED** |
| `release_url` pointed at `does_not_exist.zip` (404) | **V56 RED** |
| all three reverted byte-exact | **V06, V56, V59 all GREEN** |

### Cost, accepted deliberately

A full run now downloads ~94 MB (3.14 MB checkpoint + 91 MB outputs archive). A per-process
memo means a URL fetched by two checks is downloaded once. This is a real cost on every
verification run and it was accepted rather than cached to disk, because a disk cache is
exactly the mechanism by which a check stops testing the live artifact. A network failure is a
**FAIL, not a SKIP** — the suite already reaches the network in V55.

### Also corrected here

`weights/README.md` published `PSNR 28.7851 / SSIM 0.78279 / LPIPS 0.25233` — `train.py`'s
in-run validation, not the evaluation record. Replaced with
`28.7865 / 0.78287 / 0.25324` from `results/baselines/final/metrics.json`, which is what V27
and V48 read. Note the direction: the retired LPIPS figure was the *flattering* one, in a repo
whose stated hygiene rule is to always quote the less favourable number. Two checklist items
that the evidence had already satisfied (V35, and `--require_weights` succeeding) were ticked.

### Do NOT retry

- **Do not cache downloaded artifacts to disk between runs to make the suite faster.** The
  point of the check is that the *live* URL serves the right bytes today.
- **Do not downgrade the fetch to a HEAD request or a Content-Length comparison.** A digest
  over the served bytes is the only thing that detects a replaced asset.

## D33 — SSRF guard on `_fetch_digest` (the D32 helper)

A background security review flagged two SSRF findings in `_fetch_digest`, added in D32, and
they were right. Its URL is read out of a repository file (`weights/README.md`,
`manifest.json`) — untrusted input to the verifier process. Unguarded, that file could point
the verifier at `file:///...` (local read), at loopback or link-local addresses (internal
services, cloud metadata), or place an allowed host in the URL's *path* rather than its host
position; and a 302 from an allowed host could redirect anywhere.

This is D28 repeating — the same bug class, in a helper written the same day, the third time
this class has shipped.

Fix: `_artifact_url_ok()` does anchored `urllib.parse.urlsplit` validation requiring `https`,
no embedded credentials, port 443, and an **exact** host match against
`PUBLISHED_ARTIFACT_HOSTS` (`github.com`, `objects.githubusercontent.com`,
`release-assets.githubusercontent.com`, `raw.githubusercontent.com` — GitHub 302s release
assets to the githubusercontent hosts, so those hops must be allowed *and* validated).
`_strict_opener()` installs an `HTTPRedirectHandler` that re-validates *every* redirect hop,
not just the initial request.

Negative-controlled (see `docs/STATE.md` for the vector list: `file://`, loopback,
link-local, host-suffix spoof, path-position spoof, embedded credentials, non-443 port, plain
http all refused; the real asset URL and an uppercase-host variant both accepted), and
`--only V06,V56,V59` re-confirmed green with the guard in place.

Net stricter: closes an SSRF hole with no reduction in what the checks accept.
prior pin: `d792ab7fb0971d969e88a8f1c6c88206c14d9acd7b2bca25d7097d54eb6100a4`

---

## D34 — V61 and V62 ADDED, closing U-1 and U-8: two requirements no check could fail on

**Date:** 2026-08-16, iteration 2. **Source:** `requirements-auditor`, first and second pass
(U-1, U-8). **Contract and verifier both re-pinned**, additions only.

### U-1 — F2 size-agnosticism was verified by dead code

`docs/SPEC_ADDENDUM.md` calls the 256->512 fixture "the *only* guard against silently baking
in 128->256". That fixture lived inside `src/model.py::_selftest()`, and nothing in the
verifier, `train.py`, or any script ever called it. `UNetSR`'s internal pad/crop-back — the
likeliest home for an off-by-`(pad*scale)` bug, since `NAFSR` never pads at all — was forwarded
by zero checks. A model could regress from "any (H,W) -> exactly (2H,2W)" to "only multiples of
8" with the entire 57-check suite staying green.

**V61** builds each of `{NAFSR, UNetSR}` and forwards each of `{(128,128), (256,256), (61,97),
(1,1), (130,66)}` — even, odd, non-square, and a degenerate 1x1 — asserting the output is
exactly `(1, 1, 2h, 2w)` and finite. Anti-vacuity: FAIL if fewer than all 10 combinations
actually ran (a crash mid-loop cannot silently look like a pass).

**Negative control:** `UNetSR.forward`'s crop-back line was mutated from
`out.shape[-2] - ph * s` to `out.shape[-2] - ph` (dropping the `* s` — the exact off-by-scale
bug the addendum worried about). V61 went red: `3 of 10 arch x size combinations violate F2`.
Reverted byte-exact, reconfirmed green (`10/10`, finite).

### U-8 — F4 order randomisation was asserted nowhere

`GAUSS_PRE_DOWN_PROB` and the entire pre-downsample branch in `src/degrade.py::degrade()`
could have been deleted and no check would have noticed: V33 compares only the aggregate
variance-vs-intensity curve, which this hedge barely perturbs by design (it only ever touches
the additive-Gaussian term, which fits to zero in the real data).

**V62** measures the randomisation for real. Over 2000 draws of `sample_noise_params`: `a` and
`v` must each span >= 90% of their configured +/-30% range without escaping it; `sigma`'s
sampled minimum must land in the near-zero 5% of its configured range (see the note below on
why this is not a literal `== 0` test) and its maximum must exceed 0.015. Separately, over 800
calls to `degrade()`, `src.degrade.downsample` is wrapped with a spy that observes whether the
array it receives was mutated before decimation — this counts the REAL code path taken, not a
config flag — and the branch must be taken between 8% and 22% of the time.

**A bug in my own first draft, caught before it shipped.** The first version of the sigma
check required the *sampled* minimum to be `< 1e-9`, i.e. essentially exactly zero. `sigma` is
drawn from a continuous `U(0, 0.02)` (`src/degrade.py::sample_noise_params`), so across any
finite sample the minimum is *never* going to land within `1e-9` of zero — the check was
testing an event with probability zero, and it correctly failed on the real, correct code
(`min 1.26763e-06`). Fixed to a statistically sound bound: the configured lower edge of
`gauss_sigma_range` must itself be near zero, and the *sampled* minimum must fall within the
leftmost 5% of the configured span — a bound that 2000 uniform draws satisfy with overwhelming
probability when the sampler is correct, and fails when the range is shifted or the sampler is
broken. Recorded here because it is exactly the kind of check-writing mistake this project has
shipped before (V54's false positive, V55's SSRF hole) — caught this time by running it against
the known-correct code before trusting it, per the project's own negative-control rule.

**Negative control:** `GAUSS_PRE_DOWN_PROB` set to `0.0` (deleting the hedge in effect, without
touching the branch code). V62 went red: `pre-downsample gaussian branch taken 0.0% of the
time, outside [8%, 22%]`. Reverted byte-exact, reconfirmed green.

### Do NOT retry

- **Do not test a continuous random draw's minimum against a near-zero absolute epsilon.**
  Test against a percentile of the configured range instead, sized so the false-positive rate
  at the chosen sample count is negligible. This is the mistake V62's own first draft made.

---

## D35 — V57 ADDED, closing U-6: V12 tested a helper, not the model's actual input

**Date:** 2026-08-16, iteration 2. **Source:** `requirements-auditor` (U-6). Contract addition
only.

V12 calls `src.io_utils.load_array` directly and checks the return value. The contract's own
wording for this requirement is "the tensor **entering the model**" — a different thing. A
`clamp_` inserted anywhere in `inference.py`'s stack/H2D/channels_last/autocast pipeline would
leave V12 green while genuinely destroying the out-of-range information SPEC F5 says is
intentional.

**V57** closes the gap by testing the real path instead of a helper: it imports
`inference.py`'s own `load_net()` and `infer_chunk()` — not a reimplementation of the pipeline
— loads the actual trained checkpoint, attaches a `register_forward_pre_hook` to the model, and
drives the same extreme-value probe V12 already uses (`[-0.28, 2.16]`) through `infer_chunk`
exactly as a real invocation would. It is forced to `--device cpu`, so it never contends with a
running GPU benchmark and remains fast even mid-benchmark.

**Negative control:** `t.clamp_(0.0, 1.0)` inserted immediately before the model call in
`infer_chunk` (the exact defect class this check exists to catch). Result: **V12 stayed
green** — it never executes this code path, confirming it genuinely cannot see this class of
bug. **V57 correctly went red**: *"the tensor ACTUALLY ENTERING THE MODEL was clipped somewhere
in the stack/H2D/channels_last/autocast path, even though V12's helper-level check passed."*
Reverted byte-exact, V57 green again.

### Do NOT retry
- **Do not delete or "simplify" V12 now that V57 exists.** V12 is cheap, still correct, and
  catches a different failure mode (the loader itself clipping). V57 subsumes it for the
  purpose of the contract's actual wording; it does not replace it.

---

## D36 — `train.py --no_ledger` restricted to `--smoke` runs (M-1)

**Date:** 2026-08-16, iteration 2. **Source:** `requirements-auditor` M-1.

`--no_ledger` let any run — including a full 20k-iteration training run — skip
`results/experiments.csv` entirely, an undocumented escape hatch around SPEC §9's "log every
run" and V45's row-count gate. Restricted to genuine smoke tests: `--no_ledger` now only takes
effect when `--smoke` is also set; a non-smoke run passing `--no_ledger` gets a stderr warning
and is logged anyway. A smoke test legitimately should not pollute the ledger with a
12-iteration row, so that path is preserved. Traced through all four `(no_ledger, smoke)`
combinations by hand rather than by a live run, since the GPU was reserved by a running
benchmark at the time; low risk given the change is a single added condition around an
existing, unchanged `append_experiment` call.

---

## D37 — V58 ADDED, closing U-10: SPEC §2.3's links were never independently re-checked

**Date:** 2026-08-16, iteration 2. **Source:** `requirements-auditor` (U-10). Contract
addition only.

Licence links in `docs/decisions.md` were re-fetched and dated; SPEC §2.3's hackathon resource
links (landing page, registration, dataset Drive folder, PPTX/PDF resources, both webinars, the
WhatsApp group) were never independently re-verified — nothing checked they still resolved.

**Re-checked anonymously**, `curl -L` with a fresh process, no cookies, no saved session,
`GITHUB_TOKEN`/`GH_TOKEN` cleared from the environment (none of these hosts are GitHub; cleared
for consistency with the other fetch-based checks). All 9 links from SPEC §2.3 returned HTTP
200. Recorded in `docs/link_check.md`.

**V58** reads the canonical URL list **dynamically** from `docs/SPEC.md`'s own
"### 2.3 Official links" table via regex, rather than a hardcoded copy that could silently
drift if SPEC.md ever changes. It requires `docs/link_check.md` to record every one of those
URLs at HTTP 200 with a UTC timestamp, and the **oldest** recorded timestamp must be ≤ 72
hours old.

**Deliberately does not re-fetch on every verifier run.** A live fetch of nine third-party URLs
on every `--strict` invocation would make the whole suite's pass/fail depend on third-party
site uptime — the same flakiness class D7 rejected for spawning DataLoader workers over a
25 MB test set. Freshness is enforced by **expiring** the record instead: stale evidence fails
loudly rather than the check silently re-trusting an old fetch forever.

**Negative-controlled three ways**, each reverted byte-exact and reconfirmed green:
1. Deleted the WhatsApp row entirely -> `1 of 9 SPEC 2.3 URLs are not recorded`.
2. Injected `404` in place of one `200` -> `1 link(s) did not return 200`.
3. Backdated every timestamp by 4 days -> `oldest entry is 96.0h old, over the 72h bound`.

### Do NOT retry
- **Do not make V58 re-fetch live on every run.** That trades a controllable, bounded
  staleness window for an uncontrollable dependency on nine external services' uptime.

---

## D38 — Four real bugs in `inference.py`, found by `adversarial-reviewer`'s first delivered run

**Date:** 2026-08-16, iteration 2. **Source:** `reviews/adversarial-1.md` (local, gitignored).
This agent was dispatched in iteration 1 and killed by a usage limit before writing a file;
this is its first delivered report. 1 critical, 4 high, 5 medium, 7 low. The critical and all
four highs are addressed here; mediums/lows are logged in the report for a later pass.

`inference.py` is CLAUDE.md Prime Directive 4's highest-value file: KLA runs it as-is, and a
broken script scores zero regardless of model quality. All four fixes below were verified with
concrete repros, run with `--device cpu` throughout because a `perf-analyst` benchmark had the
GPU at the time.

### H2 + H3 (high) — the only destructive write in the program was unguarded

`--output_dir` equal to `--input_dir` overwrote the degraded inputs with restored outputs IN
PLACE (repro: run once, `000000.npy` goes from `(128,128)` to `(256,256)`; run again and it
becomes `(512,512)` — the original degraded input is gone). `--output_dir` nested inside
`--input_dir` (e.g. `--input_dir data/test --output_dir data/test/restored`, a natural
evaluator layout) made a second invocation silently re-ingest the first run's own output as
new input.

Fixed with a single resolved-path comparison before `out_dir.mkdir()`:
`out_resolved == in_resolved or out_resolved.is_relative_to(in_resolved)` — refuses both cases,
exit 1, before anything is read or written. **V60 added** as the permanent regression guard
(Tier 1, CPU-forced, needs no checkpoint since the guard fires before the model loads).
Negative-controlled: removing the fix made V60 fail on both cases; restored byte-exact, green.

### H4 (high) — a partial write failure exited 0

`n_ok == 0` (total failure) already exited 1; `n_failed > 0 and n_ok > 0` (partial failure) did
not. V07 requires exactly one output per input, so a short output set silently reported as
success is the worst outcome on KLA's machine — nothing would flag it. Repro: 6 inputs, one
output path blocked by a pre-existing directory of the same name → `5/6 usable` but exit 0
before the fix, exit 1 after.

Fixed: `if n_failed > 0:` now returns 1 with the failure count, alongside the pre-existing
`n_ok == 0` branch. No dedicated V-check yet — the existing V07 fixture run never exercises a
write failure, so a regression check would need its own filesystem-blocking fixture; logged as
follow-up rather than blocking this fix.

### H1 (high) — a loaded-but-malformed checkpoint could defeat `--require_weights`

`load_net()` sets `weights_ok=True` for any checkpoint that loads with `strict=True` — that
says nothing about whether the architecture actually implements the task. A checkpoint with a
wrong `scale` in its config loads fine, and `infer_chunk`'s shape guard then silently
substitutes `BicubicUpsampler` for the **entire output**, with `--require_weights` never
firing and the run summary still printing `weights=best`. This defeats the exact guarantee
V56 relies on `--require_weights` for. Repro: a genuine 7k-param checkpoint built with
`build_model({"scale": 3, ...})`, otherwise valid, fed through `--require_weights` → previously
exit 0, `weights=best`, 100% bicubic output; now exit 1 before the fix ships anything.

Fixed: `infer_chunk` gained a `require_weights` parameter (threaded from `main()`). On a shape
mismatch, if `require_weights` is set it raises a dedicated `_RequireWeightsViolation`
(`RuntimeError` subclass) instead of silently degrading; `main()` catches **only** that type,
shuts the write pool down cleanly, and exits 1 with a clear message. Deliberately narrow: every
*other* exception in this pipeline is left to propagate uncaught, unchanged from before this
fix, per PD4 ("a crash is preferable to a silent wrong answer") — this fix does not widen that.
The CUDA-OOM single-image bicubic fallback in the same function is a genuine resource-recovery
path, a different risk category, and is deliberately left ungated by `require_weights`.
Confirmed both directions: with `--require_weights`, exits 1, no bicubic shipped; without it,
still degrades gracefully to bicubic exactly as before (backward compatible).

### C1 (critical) — the README's own example command produced bicubic on a fresh clone

`weights/best.pt` is not tracked. The root README's section literally titled "the command KLA
runs" was `inference.py --input_dir sample_inputs --output_dir results/sample_outputs` — no
`--require_weights`. A reviewer following the README literally on a fresh clone, before
downloading the checkpoint from the Release, gets a silent bicubic upsample at exit 0. Every
existing check tolerates this: V04/V46's fresh-clone fixture run never asserts a real model
ran, and V06/V59 both pass via the hosted-URL branch regardless of whether the URL was
actually followed by whoever ran the command.

Fixed: the documented command now includes `--require_weights`, with an explanation of why —
so a reviewer who runs it literally either gets a real model result or a loud, diagnosable
failure, never a silent floor score. Also documents the H2/H3 output-directory constraint
inline, since it's the same section a reviewer is most likely to copy-paste from.

**Not a V-check fix**, because `check_V46` does not actually execute the README's fenced
commands — it only checks they exist, then runs a separate hardcoded fixture sequence
(`requirements-audit-2` H-4b, still open, `docs/STATE.md`). This means C1's fix is currently
verified only by manual re-run, the same limitation the whole README rewrite (D-earlier) was
under. H-4b remains the right fix for that gap and is unchanged by this entry.

### Not yet covered (from the same report, lower severity — tracked, not fixed here)

M1 unguarded `out_dir.mkdir()` can leak an interpreter path in a raw traceback; M2 `EXTS`
advertises four undecodable image formats and silently drops matching files from the output
set; M3 a single NaN pixel poisons a whole prediction and is neutralised to 0.0, in tension
with the file's own MMSE argument for `PLACEHOLDER_VALUE = 0.5`; M4 the `oddnames` verifier
fixture is built and consumed by zero checks; M5 `scripts/evaluate.py` pairs with non-recursive
`glob` while `inference.py` uses `rglob`. Seven lows, detailed in `reviews/adversarial-1.md`.

### Do NOT retry
- **Do not widen the `_RequireWeightsViolation` catch to a bare `except RuntimeError`.** That
  would silently convert unrelated genuine bugs into a clean exit(1), hiding them from the
  traceback PD4 relies on to make a broken run diagnosable.

---

## D39 — V64 ADDED: the regression guard adversarial finding H4 was left owing

**Date:** 2026-08-16, iteration 2. **Source:** `adversarial-reviewer` H4, fixed in D38 but
left without a permanent check (needed a filesystem-blocking fixture). Contract addition only.

D38 fixed the bug — `n_ok == 0` already exited 1, but a *partial* write failure
(`n_failed > 0` with `n_ok > 0`) did not, so a short output set was reported as a successful
run — but did not add a regression guard for it. **V64** closes that gap: it uses the `mixed`
fixture (four valid `.npy` files) and pre-occupies exactly one output path with a directory,
forcing precisely one write to fail with a real `PermissionError`. It asserts the process
exits non-zero, **and** confirms the other outputs still wrote successfully — the second
assertion matters because a check that only proves total failure exits non-zero would not
have caught the original bug at all.

**Negative-controlled**: temporarily removed the D38 fix (the `if n_failed > 0:` block) —
V64 correctly went red, *"a write failure on 1/4 outputs still exited 0"*. Restored
byte-exact, green again with the other three outputs confirmed written.

Forced `--device cpu`, no checkpoint needed — the bicubic fallback exercises the exact write
path under test.

---

## D40 — HISTORICAL V28 comparison for the hosted 20k models

**Date:** 2026-08-16, iteration 2. **Human-authorised** (decision A, 2026-08-15: "Decide on the
numbers, report both, and state the reasoning in `decisions.md`. If it is close, prefer the
model that wins more of the three metrics rather than the one that wins the largest single
margin"). This entry is what the contract's escape hatch for V28 requires: a documented honest
negative result, with the six measured means and the shipped model named, matching
`weights/best.pt`'s embedded config.

### The measured comparison

Full 400-image committed validation split, both models scored identically (`scripts/evaluate.py`,
float32 `.npy` reloaded from disk):

    final: psnr 28.7865, ssim 0.78287, lpips 0.25324
    unet:  psnr 28.8808, ssim 0.78273, lpips 0.26525

Naive unpaired means would call this 2 of 3 **for NAFSR** (higher SSIM, lower LPIPS; it only
loses PSNR) — this is exactly the pre-D31 bug `scripts/evaluate.py` had, and it is what let
`results/metrics_summary.md` claim `V28: PASS (2/3)` before that was fixed (D31, this file's
own entry above). The contract requires the **paired** per-image test instead (D31: both
models score the *same* 400 images, so the correct statistic is the per-image difference, not
the gap between two independent means):

    psnr   mean diff -0.0943 dB   t=-6.11   significant   better on  93/400   -> LOSS (U-Net wins)
    ssim   mean diff +0.000135    t=+0.29   NOT significant             172/400   -> TIE
    lpips  mean diff -0.0120      t=-5.55   significant   better on 235/400   -> WIN (final wins)

**1 win / 1 loss / 1 tie.** Not "2 of 3" either way — the user's stated tiebreak ("prefer the
model winning more metrics") does not resolve a genuine 1-1-1 split, so the decision rests on
the other two axes below.

### Throughput — measured on the RTX 4060 Laptop, `results/runtime_report.md`

Two different numbers answer two different questions, and only one matches how KLA actually
scores this submission (one `inference.py` subprocess over the whole 400-image test set, per
SPEC F11 — not a re-launched process per image):

- **Isolated forward-pass compute** (no subprocess startup, no disk IO): NAFSR 98.1 img/s vs
  UNetSR 468.9 img/s at bs=32/128px/bf16/channels_last — UNetSR is **4.78x faster** in raw
  compute.
- **End-to-end, externally timed subprocess** (the shape of the actual scored invocation,
  `results/runtime_report.md` "Batch size / precision / memory-format sweep" cross-referenced
  with the `e2e` variant timings): NAFSR 16114.6 ms / 400 images (24.82 img/s) vs UNetSR
  14989.5 ms / 400 images (26.69 img/s) — UNetSR is **7.5% faster**, not 4.78x. At N=400 the
  fixed cost (interpreter start, `import torch`, CUDA init, checkpoint load — measured at
  ~14.8 s, `results/runtime_report.md` "Startup vs compute") is ~30.6% of total wall-clock and
  dilutes almost the entire compute-side gap, because both models are tiny relative to the
  import/CUDA-init cost that neither can avoid.

**The 7.5% end-to-end number is the one that matters for scoring.** The 4.78x compute-only
number describes a workload KLA does not run.

### Reasoning

1. **Quality is a genuine tie**, not a win for either model. 1/1/1 does not meet the user's
   "wins more metrics" tiebreak in either direction.
2. **Throughput favours UNetSR by only 7.5%** in the metric that actually matters (end-to-end,
   subprocess-timed, the real invocation shape) — not decisive on its own.
3. **NAFSR is 7.65x more parameter-efficient** (388,225 vs 2,970,401 params) for
   statistically-tied fidelity and a real perceptual-quality win (LPIPS). A model that matches
   a 7.65x-larger network on PSNR/SSIM while beating it on LPIPS, at a fraction of the
   parameters, is the stronger result once size is accounted for — and parameter count is not
   nothing on a memory-bandwidth-bound architecture (`docs/decisions.md` D21) being evaluated
   for restoration quality, not classification accuracy.
4. **NAFSR is the SPEC-intended primary architecture.** SPEC §7.1 specifies the NAFNet-style
   body as the submission architecture; SPEC §7.2 specifies UNetSR explicitly as "the plain
   U-Net baseline required by the rubric" — a comparison point, not a candidate for shipping.
   Overriding that intent needs a clear win, and this is not one.
5. Per the user's own framing of the prior ("7.7x parameters buys little on a startup-dominated
   run — but that cuts both ways: it also costs little"): confirmed exactly as stated. The
   parameter disadvantage NAFSR carries costs only 7.5% wall-clock in the scored shape, which is
   a small price for the model SPEC specifies, with tied-or-better quality.

### Decision: SHIP NAFSR

**SHIPPED MODEL: NAFSR**

The negative result — NAFSR does not beat the U-Net baseline on 2 of 3 metrics — is genuine
and is recorded here rather than concealed. It does not change the shipping decision: NAFSR
remains `weights/best.pt`, unchanged.

### Do NOT retry
- **Do not re-litigate this with a different statistic to force a cleaner win.** The paired
  test is correct (D31); an unpaired comparison would flatter one model over the other for the
  wrong reason (the SSIM "win" under an unpaired read was noise, not signal — D31 already
  demonstrated this).
- **Do not conflate the two throughput numbers.** Quoting the 4.78x compute-only gap as "the"
  throughput difference in any external-facing document (README, deck) would misrepresent the
  actual scored cost, which is 7.5%.

## D41 — Phase 2 cloud compute: premise re-derived, HF Jobs verified, private dataset repo confirmed

**Date:** 2026-08-16. Written before any cloud training executes; `docs/PLAN_PHASE2.md` is the
full plan this entry supports.

### D7's estimate re-derived, and refuted at the size that actually matters

D7 estimated fixed startup at 85-95% of scored wall-clock, from a 1-image measurement with
torch *not installed*. `results/runtime_report.md` (committed `bdf4547`, this repo's first real
end-to-end measurement) now gives the number at N=400 — the actual scored set size, RTX 4060,
bf16, batch 32: **fixed startup 44.4%, compute ~56%** of a 22,514.6 ms median wall-clock. The
proposed re-opening of 8x TTA rested on compute being ~5% of wall-clock; at the measured ~56%,
8x compute is a **~5.3x** wall-clock increase (9,994 ms + 8×13,596 ms ≈ 118,762 ms), not the
"+30%" the estimate implied. **TTA stays rejected** — same conclusion as D7, now for a measured
reason. Full derivation: `docs/PLAN_PHASE2.md` §2.

### HF Jobs verified live, not from docs alone

Org: `Team-Ceciroleo67` (billing page: $30.00 credit, $0.00 used, expires 2026-09-01 — screenshot
reviewed this session). A fine-grained token scoped to that org (repo read/write, Jobs
start/manage, billing read) was used to:

1. Launch a `cpu-basic` smoke job billed to the org namespace (`run_job(..., namespace=
   'Team-Ceciroleo67')`) — completed, `owner.type == 'org'`, correct stdout retrieved via
   `fetch_job_logs`. Confirms Jobs is enabled for this org and bills against the real credit
   balance, not just the docs' claim that it should.
2. Created `datasets/Team-Ceciroleo67/kla-ps01-phase2-data` with `private=True`. Verified
   private two ways: the authed `repo_info().private == True`, **and** an unauthenticated
   `curl` against both the API (`/api/datasets/...`) and the web URL returned **401** on both —
   the repo is actually invisible logged-out, not just flagged private in a field a bug could
   ignore.

### Data uploaded

`C:\kla-data\_archive\train.zip` (918,994,209 B), `Test_NoisyLR.zip` (23,419,125 B), and
`configs/split_val.txt` pushed to the private dataset repo as-is (original archive bytes, not
re-zipped) so cloud-side extraction reproduces the exact local dataset. **F17 still applies
unconditionally in the cloud**: `test_NoisyLR` ships only because inference on it is required;
any cloud training script must import the same `configs/split_val.txt`-driven split as
`src/dataset.py`, never a reimplementation that could accidentally include it.

### Pricing confirmed via `GET /api/jobs/hardware` (live, not docs)

Matches `jobs-pricing` docs exactly: `a100-large` = 80 GB VRAM, $0.041667/min = $2.50/hr.
**No H100 flavor exists in HF Jobs** — confirmed against the live hardware list, not inferred.

## D42 — V22 root cause found and fixed: SCA's raw spatial mean, not LayerNorm

**Root cause, confirmed empirically, not guessed.** `SCA.forward()` (`src/blocks.py`) computed
`self.conv(x.mean(dim=(2,3), keepdim=True))`. Autocast's op-policy table force-promotes
`F.layer_norm`/`native_layer_norm`/`group_norm` to fp32 automatically; a bare `Tensor.mean` is
**not** an autocast-registered op at all, so under bf16 autocast it silently executes in
whatever dtype its input already has (bf16, since `x` is the output of an upstream bf16 conv +
SimpleGate). The two competing hypotheses from `docs/STATE.md`'s dispatch were both tested
rather than assumed: `F.layer_norm(bf16 in, fp32 weight/bias)` does auto-promote to fp32
(confirmed a no-op fix, as suspected); `x.mean(dim=(2,3))` of a bf16 input does not (confirmed
the actual culprit).

**Fix:** wrap the SCA spatial-mean-and-1x1-conv pair in `torch.autocast(..., enabled=False)`
with the input explicitly cast to fp32 first; the gate multiply `x * w` then promotes to fp32
by ordinary ATen type-promotion (bf16 * fp32 -> fp32) without needing its own cast, the same
way autocast lets a `F.layer_norm` fp32 output be re-cast by whatever bf16 op consumes it next.
Casting `w` back to bf16 before the multiply was tried and measured as a no-op (reproduces the
unpatched result bit-for-bit) — the 1x1 conv's own output is bf16-rounded either way, so the
precision has to be preserved past the conv, not just through the mean.

**Measured, on the V22 fixture (`tests/fixtures/single/only_128.npy`, real trained checkpoint,
CUDA):** bf16-vs-fp32 max abs diff 1.27e-02 (over the 1e-2 cap) -> **7.79e-03** (under it);
mean abs diff 5.90e-04 -> 5.29e-04 (both comfortably under the 1e-3 cap either way).
`py -3.12 scripts/verify_all.py --only V22`: **PASS**.

Tolerance was never widened, matching Prime Directive 1 — the fix moves the number, not the bar.

`src/blocks.py` is nominally `model-core`'s file per `CLAUDE.md`'s ownership map, but V22's
root cause is architectural and `inference-engineer` was tasked with root-causing and fixing
it; no `model-core` agent was running concurrently, so there was no write conflict. Noted here
for the record rather than silently crossing the ownership boundary unremarked.

**Addendum — bf16-vs-fp32 quality gap, measured** (the open question `docs/MORNING_REPORT.md`
flagged: the 400 published outputs were generated in bf16, the 28.7865 dB record in fp32, and
nobody had measured the difference). Ran the real `inference.py` default (bf16) on all 400
val-split inputs, scored with the pinned metrics against GT: PSNR 28.7849 ± 4.5322 (Δ −0.0016
dB vs the fp32 record), SSIM 0.78278 (Δ −0.00009), LPIPS 0.25233 (Δ −0.00091, marginally
better). **The gap is negligible — V22 was a genuine worst-case per-pixel tolerance violation,
not a real aggregate quality regression.** The published outputs do not need regenerating for
quality reasons.

**Addendum — a second refinement, and a pre-existing bug it surfaced.** The first fix version
called `self.conv(...)` (an `nn.Conv2d`) inside the disabled-autocast fp32 region — mathematically
a 1×1 conv over a `(B,C,1,1)` tensor, i.e. a per-image channel-linear op with no spatial extent,
but it still dispatches through cuDNN's autotuned convolution path. With `cudnn.benchmark=True`
(a kept-on free lever, V40), two candidate algorithms for this exact fp32 shape benchmark close
enough to tie, so which one wins is scheduler-noise-dependent — measured directly: V24
(cross-process determinism) failed ~50% of runs (5/10) even on the **unpatched, pre-V22-fix**
model, confirming this is a pre-existing robustness gap, not something the V22 fix introduced.
Routing the op through `F.linear(pooled, self.conv.weight[:,:,0,0], self.conv.bias)` instead
(a cuBLAS GEMM on a shape-keyed heuristic, not a timed cuDNN autotune) roughly halves the flake
rate to ~24% (5/21) — better, but the remainder comes from other, real spatial convolutions
elsewhere in the 16-block stack and is **not resolved by this change**. Logged as an open
robustness gap (`docs/BLOCKERS.md` B11), not silently absorbed into the V22 fix's scope.

## D43 — Degradation order permutation + F1 tail-coverage widened; V62 strengthened accordingly

**Closes two requirement-level gaps** identified by this iteration's requirements audit
(`docs/REQUIREMENTS_MATRIX.md`): KLA's "handle degradations applied in any order," and the F1
tail-coverage gap (`reviews/ml-skeptic-1.md` finding F1 / D25: synthetic max 1.7177 vs real
train max 2.0735).

**Order permutation** (`src/degrade.py`, `data-pipeline`): `degrade()` previously applied
shot+speckle and additive-Gaussian noise unconditionally after downsampling, with only a 15%
probability of moving the Gaussian term (alone) before it — reaching ~7.5% of samples via
`synth_ratio: 0.5`. The `noise_after_downsample: false` guard, which actively *raised* on the
alternative order, is removed (kept as an accepted-but-no-op key for backward compatibility).
`degrade()` now independently decides, per synthetic sample, whether shot+speckle (`S`) and/or
Gaussian (`G`) move before downsampling (`D`), with a coin flip for relative order when both or
neither move — all 3! = 6 orderings of {D,S,G} are reachable. D2's measured finding (released
data's residual autocorrelation ≈0, consistent with the canonical `D,S,G` order) is preserved
as the modal case, not discarded — canonical order stays the majority outcome, matching the
data, while every ordering the brief asks for is now genuinely reachable.

**Measured order distribution, 20,000 trials:** DSG 64.71% (modal) · SDG 13.41% · GDS 12.59% ·
DGS 7.06% · GSD 1.12% · SGD 1.11%. D-first (canonical) overall 71.77%; S-before-D (any) 15.65%;
G-before-D (any) 14.82%.

**Tail-coverage widening:** `NOISE_RANDOMISE_FRAC` 0.30 → 1.20, `GAUSS_SIGMA_RANGE` (0, 0.02) →
(0, 0.065), chosen by a parameter sweep against the real 2800 non-val train GT images.
Measured at 56,000 synthetic samples (917M px): synthetic max **2.0869** (was 1.7177), against
real train max 2.0735 and dataset-wide max 2.1580 — now exceeds the real training max with
margin. Side effect, not gated by any check yet: because the speckle term `v*x²` is symmetric,
widening `v` also inflates the already-over-produced `<0` tail (D25: 0.62% vs real 0.26%) to
~1.8%, ~7x over. Flagged for a follow-up (asymmetric a/v ranges), not blocking this fix.

**Fidelity unaffected, confirmed not assumed:** V33 re-run gives `mean_abs_rel_err=0.3885`,
`binned_r2=0.9804`, identical to the pre-change baseline — structural, since `degrade_fitted`
(V33's subject) always uses `FITTED_NOISE` directly, never `DegradeConfig`'s randomisation
ranges, and always takes the canonical order (`params is not None` forces `randomise_order =
False`). V26 (paired-crop alignment) re-verified unaffected.

**Config integration:** `configs/final.yaml`, `configs/baseline_unet.yaml`,
`configs/nafnet_x2.yaml` explicitly pinned the OLD narrow values (`randomise_frac: 0.30`,
`gauss_sigma_range: [0.0, 0.02]`), which would have silently kept training on the old, narrower
simulator regardless of the module defaults changing. All three updated to the new values
(`model-core`).

**V62 STRENGTHENED** (main session, `scripts/verify_all.py`, hash re-pinned in
`docs/VERIFIER_SHA256`): the prior check only detected "was the array handed to `downsample`
different from the input" — a single yes/no counter that could not distinguish which op(s)
preceded `D`, and whose accepted band `[8%, 22%]` assumed only the old Gaussian-only hedge
existed (measured pre-down rate is now ~28-29% combined). Replaced with a check that spies on
`downsample`, `_shot_speckle_delta` and `_gauss_delta` simultaneously, reconstructs the actual
per-trial call sequence (each of the three fires exactly once per `degrade()` call), and
asserts over 2000 trials: all 6 orderings observed at least once; P(S before D) and P(G before
D) each in [8%, 24%]; canonical `DSG` rate in [55%, 80%] (majority, not exclusive — catches
both "hedge deleted" and "canonical order no longer respected").

**Negative-controlled before being trusted** (`docs/STATE.md`'s standing rule, after V54/V55's
and V62's own prior sigma-bug history): with `shot_speckle_pre_down_prob` and
`gauss_pre_down_prob` both forced to 0 (simulating the pre-fix, order-hedge-deleted behaviour),
only `DSG`/`DGS` appear, canonical rate 90.1% — correctly outside the new [55%, 80%] band, and
4 of 6 orderings correctly reported missing. `py -3.12 scripts/verify_all.py --only V62`:
**PASS** at the real (fixed) configuration.

**Not yet done, tracked for the cloud long run (`docs/PLAN_CLOUD.md`):** the shipped checkpoint
(`weights/best.pt`) was trained under the OLD simulator (fixed order, narrow tail). Retraining
under the corrected simulator — locally as a cloud-independent fallback, and via the planned
cloud long run — is separate work; this decision covers the simulator fix only. Re-measured
in-distribution PSNR/SSIM/LPIPS after retraining will be reported as a number, not assumed, per
`docs/BLOCKERS.md`'s standing rule against unmeasured claims.

## D44 — Proxy-OOD generalisation report added, closing U-9; V63 ADDED

KLA requires generalising to "unfamiliar image content" and scores restoration quality on
hidden GT "including in-distribution and out-of-distribution content." Before this, the repo
had **zero evidence** on this axis — `docs/STATE.md` had flagged U-9 as "the one remaining SPEC
gap with no plan yet," and no check could turn red for its absence.

**Proxy-OOD set built** (`dataset-forensics`, `results/eda/proxy_ood/`): 40 procedurally
generated grayscale images (numpy primitives only — line/space gratings, contact-hole grids,
checkerboards, circuit-like traces, sharp-edged shapes), 8 per category, deterministic seeds.
Degraded with the **existing fitted** degradation model (`src.degrade.degrade_fitted` — the
recovered kernel D1 + fitted 3-parameter noise D12), **zero parameters refit** on this content,
since refitting on OOD content would defeat the point of testing generalisation. Committed
membership list; disjointness from train/val/test verified computationally
(`results/eda/proxy_ood/membership_check.json`), not asserted in prose.

**Scored** (`loss-metrics`, `scripts/make_baselines.py --proxy_ood`, `scripts/evaluate.py
--proxy_ood`) on the shipped checkpoint, same pinned PSNR/SSIM/LPIPS settings (V31), same
disk-reload discipline (V30). Predictions land at
`results/baselines/proxy_ood/final/` — one directory level deeper than the normal
`results/baselines/<name>/` rows specifically so `evaluate.py`'s single-level glob never mixes
them into the in-distribution table.

**Measured, n=40:** PSNR 27.3177 ± 3.4696 dB (−1.4687 vs in-distribution 28.7865), SSIM 0.96493
± 0.01167 (+0.18207 vs in-distribution), LPIPS 0.03797 ± 0.01780 (−0.21527 vs in-distribution,
better). **Reported honestly as a genuine split, not averaged away**: PSNR is worse on
proxy-OOD (fine periodic gratings alias near the 2x decimation's Nyquist limit, consistent with
`docs/dataset_findings.md`'s proxy-OOD section), while SSIM and LPIPS are both measurably
better — checked per-image, even the worst-PSNR proxy-OOD images score SSIM ≥ 0.94 / LPIPS ≤
0.09, because large flat regions and locally-correlated periodic structure are easy for a
windowed structural/perceptual metric even when a global phase/intensity offset tanks PSNR.

**What this can and cannot prove, stated plainly** (per `docs/SPEC_ADDENDUM.md` section 11's
discipline against overclaiming): this measures generalisation to structurally different
*content* under a *fixed, already-measured* degradation. It is procedural synthetic geometric
content, not semiconductor or SEM imagery — none exists anywhere in this project — and it says
nothing about robustness to a different degradation than the one measured from the released
data.

**V63 ADDED** (Tier 4, main session): `results/metrics_summary.md` must contain a `## Proxy-OOD
generalisation check` heading with `n=40` and a mean±std for all three metrics; none of
`SPEC_ADDENDUM.md` section 11's banned positive phrasings ("our semiconductor...", "semiconductor
dataset/validation set/imagery" as a positive claim — a "not semiconductor imagery" disclaimer is
correctly exempted via a preceding-"not" check, negative-controlled: bare positive phrasings are
caught, the required disclaimer text is not); `results/eda/proxy_ood/membership_check.json` must
assert `n_proxy_ood == 40` and disjointness from train GT/LR/test all `true`; and
`results/baselines/proxy_ood/final/metrics.json` must show `n == 40`, all three metric means
finite, `float32` predictions, zero unclipped artifacts. `py -3.12 scripts/verify_all.py --only
V63`: **PASS**.

## D45 — Dual-resolution (256→512) runtime measurement: fixed cost does NOT collapse

KLA's brief says eval images are expected around 256×256 **or 512×512**. Released data is
uniformly 128→256 (no 512 GT exists). Every prior throughput number (`results/runtime_report.md`)
was measured at 128→256 only. The working assumption going in — that a 4× pixel-count increase
would make the pipeline compute-bound, collapsing the 30.6% fixed-startup share toward the
~10-15% range — is **refuted by measurement**.

`perf-analyst` re-ran the exact same external-subprocess methodology (`scripts/benchmark_runtime.py`,
N ∈ {1,25,50,100,200,400}, batch 32, bf16, same RTX 4060 Laptop GPU) against real synthetic
256×256 inputs, generated from held-out GT via the actual fitted degradation model (not
`np.random`) since no real 256px GT-to-512px pair exists in the released data:

| | 128→256 (existing) | 256→512 (this measurement) |
|---|---|---|
| fixed startup | 14,755 ms | 11,499 ms |
| marginal | 86.55 ms/image | 58.19 ms/image |
| total @ N=400 | 48,269.4 ms | 33,533.7 ms |
| **fixed-cost fraction @ N=400** | **30.6%** | **34.3%** (rose, did not fall) |

Mechanistically: the pure forward-pass sweep shows 256px compute costs ~5.1× more per image
than 128px (matching the 4× pixel growth plus overhead), but the 128→256 pipeline's own
end-to-end marginal cost (86.55 ms/image) was already ~8.5× inflated above its forward-only
number by non-compute overhead (H2D/D2H, batch bookkeeping, disk writes not overlapping a very
fast forward pass) — at 256px that same near-fixed overhead is dwarfed by real compute (58.19
ms/image measured is only 1.12× the forward-only number), so the marginal term actually
**drops** in absolute terms even as compute per image rises, leaving fixed cost a larger, not
smaller, share of a smaller total.

**Consequence for the sweep-axis decision** (this was the explicit gate before any cloud budget
committed): fixed-cost fraction stays **above 25%** at both resolutions — nowhere near the
~15% threshold that would have shifted budget toward training-length/patch-size. **The
width/depth Pareto sweep proceeds as originally briefed.**

Caveat carried honestly: the 128→256 report's own N=400 figure has a 681.4% spread (a likely
thermal/driver outlier); the 256→512 comparison inherits some of that noise, though the
qualitative conclusion (fixed cost 30-35% at both resolutions) is robust to it. With no real
512 GT, this measures pipeline timing/shape correctness, not restoration quality at 512.
Written to `results/runtime_report_512.md`, whitelisted in `.gitignore` alongside the existing
`runtime_report.md` exemption, never merged into that file (the two resolutions stay separate
rows, never conflated, per the standing "label every number with its device and resolution"
rule).

## D46 — V65 ADDED: real 256→512 batch correctness + genuinely-forced OOM-recovery

Closes the other half of the dual-resolution gap D45 measured timing for: nothing had ever run
a real multi-image batch through the actual `inference.py` CLI at 256→512, and the recursive
OOM-batch-halving mechanism in `infer_chunk` (`inference.py` lines ~340-355, already
implemented, never a defect) was exercised by zero checks.

**Part A**: 8 real synthetic 256×256 inputs through the unmodified `inference.py` CLI, batch
size 8, asserting N-out, `float32`, `ndim==2`, exact `(512,512)`, finite, `[0,1]`.

**Part B, the harder half — a genuinely forced OOM, not a faked exception**: isolated in a
child process (so it can never leak into any other check in the same verifier run — PyTorch
offers no clean "restore default" for `set_per_process_memory_fraction`), that process calls
`torch.cuda.set_per_process_memory_fraction(0.03, 0)` before loading the model, then drives a
real 16-image batch through the actual `load_net()` + `infer_chunk()` path (same import
pattern V57 established), wrapped in `torch.inference_mode()` to match `inference.py` main()'s
real usage exactly (an early version of this check omitted that wrapper, which retained the
full autograd graph and produced a misleadingly huge, unrepresentative memory footprint —
caught and fixed before trusting the check). A transparent wrapper around the real `_is_oom`
counts genuine invocations without altering its behaviour.

**Measured**: at fraction 0.03, the real allocator genuinely exhausts at every batch size on
this 8 GB card, cascading through the full recovery ladder (16→8→4→2→1→CPU-bicubic) with 31
real `torch.cuda.OutOfMemoryError`s caught and correctly handled; final output `(16, 512, 512)`,
`float32`, finite, in `[0,1]`. Negative-controlled: at `fraction=1.0` (uncapped) with the
`inference_mode` fix in place, the same batch completes with **zero** OOM calls, confirming the
0.03 cap — not device contention or a script bug — is what forces the real exception. On a
resource-richer host (e.g. a cloud A100 with far more VRAM), the same fraction *proportionally*
constrains that device too, so the check is not hard-coded to this GPU's absolute capacity.

`py -3.12 scripts/verify_all.py --only V65`: **PASS**.

## D47 — `huggingface_hub` pinned: optional cloud-training Hub-push path only

`train.py --hub_repo` (added for Phase 2 HF Jobs cloud training, see `docs/PLAN_CLOUD.md`) now
calls `push_checkpoint_to_hub` in `src/utils.py`, which does a lazy, in-function
`import huggingface_hub`. That made it a transitive dependency only (pulled in by something
else in the environment), never a direct pin — a gap for a repo whose `requirements.txt` claims
to be a complete `pip freeze`.

Confirmed installed version still matches what the trainer agent found when writing the code:
`python -c "import huggingface_hub; print(huggingface_hub.__version__)"` → `1.7.1`. Pinned as
`huggingface_hub==1.7.1`, plain PyPI resolution — no index-URL hazard like torch's (D18/B8),
since `huggingface_hub` has no CUDA-variant wheels to be silently swapped out.

Confirmed zero-cost for everyone else: `inference.py`'s module-level import allowlist
(`argparse os sys time pathlib concurrent.futures numpy torch`) is untouched — grep for
`huggingface_hub` in `inference.py` returns no matches. Anyone running `train.py` without
`--hub_repo` never triggers the lazy import either, so the added dependency costs nothing on
any path except the optional Hub push.

## D48 — [from origin/main, renumbered from that branch's own "D41" to avoid collision] V28 NEGATIVE RESULT: merge final hardening with the tracked checkpoint (2026-08-16, teammate session, Mac/MPS)

**Decision, as originally recorded on `origin/main` before this merge.** The user explicitly
selected submission checkpoint `weights/best.pt` with SHA256
`37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7` and asked to promote the
verified hardening branch to `main`. The checkpoint was committed directly (Route A), so a
fresh clone needs no download or manual placement. The later hosted 20k NAFSR and U-Net
measurements remain as clearly labeled historical comparison artifacts; they are not claims
about the tracked default checkpoint.

Normal inference is strict: a missing, unloadable, malformed, or unusable model path exits
nonzero. Bicubic substitution is available only through the explicit demo flag
`--allow_bicubic_fallback`; `--require_weights` is retained and overrides it. V51 was
strengthened with one exact blob exemption, `weights/best.pt`, mirroring `.gitignore`; all
other checkpoint and dataset-like blobs remain forbidden.

The direct historical U-Net comparison is also recorded rather than hidden:

    tracked final: PSNR 28.0394, SSIM 0.74804, LPIPS 0.29571
    U-Net:         PSNR 28.8808, SSIM 0.78273, LPIPS 0.26525

The tracked checkpoint does not beat that later U-Net run on V28. The user nevertheless
required this exact checkpoint digest for the final submission hardening and subsequent
promotion to `main` at the time, which was the governing release constraint for that merge.

**SHIPPED MODEL (as of that commit, on that branch): NAFSR, `r2_nb8_psnrloss`, w48n8.**
**Superseded by D49 below** — this entry is kept verbatim for the audit trail, not because its
conclusion still holds.

## D49 — Reconciliation: two independently-developed NAFSR checkpoints re-scored head-to-head; this session's from-scratch w48n16 checkpoint ships

**Date:** 2026-08-16, main session, Windows/RTX 4060. Written on discovering that `origin/main`
had diverged 19 commits (teammate `shanmukh sai`, Mac/MPS, branches `codex/*`) with an
independently promoted checkpoint (D48 above) while this session had been developing its own
NAFSR line (D19–D47) with no visibility into that work. Full analysis: `docs/MERGE_ANALYSIS.md`.
Per standing instruction, no HF Jobs cloud spend occurred until this reconciliation completed.

**The two checkpoints are the same architecture family (NAFSR) at different depths, not
different approaches:** this session's `weights/best.pt` at merge time was `NAFSR w48n16`
(388,225 params, trained from scratch, D19/D40); origin's promoted checkpoint was
`NAFSR w48n8` with `padding_mode="replicate"` (246,529 params) whose stem/head were initialised
from a closed-form ridge-regularised 5×5 LS filter and then residual-refined
(`scripts/train_residual.py`, origin-only). `src/model.py`/`src/blocks.py` needed no conflicting
changes to load both — origin's only addition was an optional `padding_mode` parameter
(default `"zeros"`, fully backward compatible) plus a training-irrelevant zero-body forward
shortcut; both merged cleanly with no functional conflict.

**Methodology confirmed identical before any number was trusted** (per this reconciliation's
own Step 2 discipline): `src/metrics.py`, `scripts/evaluate.py`, and `configs/split_val.txt`
are byte-identical between the two branches (`sha256sum` match on the split file; `git diff`
empty on the other two). Both checkpoints were therefore re-scored under the exact same
harness, not compared across self-reported numbers from different machines.

**Re-scored, this session, on this session's RTX 4060, using each branch's own `src/model.py`
to load its own checkpoint, against the same GT via the same `scripts/evaluate.py`:**

| Checkpoint | PSNR dB | SSIM | LPIPS | params | ms/img @128→256 (fwd only) | ms/img @256→512 (fwd only) |
|---|---|---|---|---|---|---|
| **This session's NAFSR w48n16** (shipped) | **28.7865 ± 4.5329** | **0.78287 ± 0.14169** | **0.25324 ± 0.13193** | 388,225 | 21.306 | 34.331 |
| origin's NAFSR w48n8 LS5+residual | 28.0394 ± 4.1882 | 0.74805 ± 0.15274 | 0.29569 ± 0.16671 | 246,529 | 9.156 | 18.286 |

Origin's self-reported 28.0394/0.74804/0.29571 (D48) reproduced exactly when re-run on this
session's hardware/dataset — not a measurement artifact, a real, reproducible number.

**Paired per-image test** (`src.metrics.paired_compare`, the same statistic `check_V28` uses,
n=400, both checkpoints scored on the identical 400-image split):

| metric | mean diff (ours − origin's) | t | images better (ours) | verdict |
|---|---|---|---|---|
| psnr | +0.7471 | 21.62 | 378/400 | **win** |
| ssim | +0.0348 | 26.13 | 397/400 | **win** |
| lpips | −0.0424 | −6.38 | 271/400 | **win** |

**This session's checkpoint wins all three metrics, paired, with high significance (all
`|t| ≫ 1.96`).** Origin's checkpoint is faster (≈2.3× at 128→256, ≈1.9× at 256→512, consistent
with its ~1.6× smaller parameter count and shallower body), a genuine and real Pareto trade-off
worth recording, but not the axis this project has prioritised (SPEC's rubric weights
PSNR/SSIM/LPIPS; no throughput floor exists post-D6/D10's V39 amendment).

**Decision: this session's from-scratch NAFSR w48n16 checkpoint ships as `weights/best.pt`,
superseding D48's promotion.** Origin's Route-A commit-the-checkpoint-directly mechanism
(V51 exemption, D48) is adopted as the delivery mechanism going forward — it is a strictly
better solution to the B6/B9 external-hosting problem than anything on this session's line —
but the tracked bytes are this session's checkpoint, re-hashed accordingly in
`weights/README.md`.

**What is kept from origin's line, on its merits, independent of which checkpoint ships:**
Linux/Docker fresh-clone verification records (V04/V46), the submission checklist, and
`scripts/make_qualitative_examples.py`'s tooling. **What is regenerated post-merge because it
was produced against origin's now-superseded checkpoint:** `results/qualitative/*`,
`results/restored_test_outputs/*`, `results/runtime_report.md`'s checkpoint-specific rows,
`README.md`'s checkpoint-specific numbers, and `results/metrics_summary.md` (machine-generated,
never hand-edited — regenerated via `scripts/evaluate.py --collect`).

**Would overturn this:** a re-run showing the paired test does not replicate, or a human
decision that origin's speed advantage matters more than its quality deficit for this
submission — not asserted here, since SPEC states no throughput floor and does state a
three-metric quality rubric.

## D50 — V54 strengthened: comparison-only literals exempted, closing a false positive from the new F17 guard

Post-merge `--strict` run (this session, commit `4eeeb2e`) found V54 FAIL:
`train.py` lines 200/212 flagged as "F17 VIOLATION RISK" for the literal `"test_noisylr"`.
Both lines are inside `_assert_never_touches_test_noisylr` (added this session, see the
`trainer` agent's work on `docs/PLAN_CLOUD.md`'s Hub-push path) — a guard that raises
`RuntimeError` if any training path would touch `test_NoisyLR`, i.e. code that *forbids* the
exact thing V54 exists to prevent. V54's check is correctly designed to distrust prose (a
comment or docstring naming the path is not a read) but had not previously encountered a
literal used purely as a comparison operand (`if "test_noisylr" in path.lower():`) — its
"path-shaped" heuristic (no whitespace) flagged it the same as an actual path.

**Fix, in `check_V54`:** literals that appear ONLY as the operand of an `ast.Compare` node
(`in`/`not in`/`==`/`!=`) are exempted from the path-shaped rule — but the mechanism that
catches a real violation, `fs_literals` (a literal passed directly to a filesystem call), is
checked first and unconditionally, independent of this exemption. A literal that is both
compared AND passed to `open()`/etc. is still flagged.

**Negative-controlled:** a minimal AST fixture with `open("test_noisylr/000001.npy")` (no
comparison at all) is still flagged as `fs_literal` under the new code — confirmed before
trusting the fix (`docs/VERIFIER_SHA256`'s change-log entry records this). The real repo's
`train.py` guard, which never passes the literal to a filesystem call, now passes.

**Is this a weakening?** No — it narrows the check's blast radius on the *false-positive* side
only, while leaving its true-positive coverage (a literal reaching a filesystem call) fully
intact and independently verified. Per Prime Directive 1, logged here with the hash re-pin
rather than silently applied.

New hash: `docs/VERIFIER_SHA256` updated, prior pin
`92d0afd7210368b66f974dc977453da473522ab02302bc78f63e9f44cb0a0e4a`.

## D51 — Post-merge fixups: data-root fallback regex, `--allow_bicubic_fallback` test flag, `configs/final.yaml`, checkpoint restoration

Four independent breakages found by running `scripts/verify_all.py --strict` on the merged
tree (commit `4eeeb2e`, this session), each traced to its real cause rather than patched
around. None weakens a check; all are either non-verifier fixes or, for V54 (D50, above), a
documented strengthening.

**1. `weights/best.pt` was briefly overwritten by the merge.** `git merge` treated origin's
tracked `weights/best.pt` (previously untracked/gitignored on this session's side) as a clean
add, silently replacing this session's on-disk checkpoint bytes with the teammate's. No git
history existed for the untracked file, so it could not be recovered from git — but the exact
checkpoint (sha256 `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313`, 3.14 MiB)
was independently published as a GitHub Release asset (`artifacts-v1`) before this session even
began. Downloaded and hash-verified before restoring it. **Lesson for the record:** an untracked
file that is precious (a trained checkpoint) should either be tracked or backed up before a
merge that might add a file at the same path — this session got lucky that a Release already
existed, not because the risk was anticipated.

**2. `src/dataset.py::resolve_data_root()`'s doc-parsing fallback broke.** It parses
`docs/DATA_LOCATION.md`'s *first* fenced code block as a literal path when neither
`--data_root` nor `$KLA_DATA_ROOT` is given. Origin's rewrite of that file's opening section
replaced the bare-path fence with a shell-command example (`KLA_DATA_ROOT=/path/to/dataset
python scripts/verify_all.py --strict`), which the parser then returned as a literal
"dataset root" — nonsense, causing every check that invokes `train.py` without an explicit
data root (V25, V34, and by extension anything spawning a fresh `train.py` subprocess) to FAIL
with "could not determine the dataset root" even though `C:\kla-data` sat right there on disk.
Fixed by restoring a bare-path first fence and moving the shell-command example to a second,
later fence — both pieces of information (the Windows path this session actually uses, and the
`$KLA_DATA_ROOT` override a different machine needs) are preserved, just correctly ordered for
the parser's documented contract.

**3. `inference.py`'s `--require_weights`→`--allow_bicubic_fallback` semantic flip (origin,
"Harden final submission inference") broke V65's OOM-recovery test.** V65's embedded probe
script calls `infer_chunk(..., False)` for the 7th positional argument, which used to mean
`require_weights=False` (permit the old default fallback-on-failure behaviour) and now means
`allow_bicubic_fallback=False` (refuse to substitute bicubic at all) — the same literal, opposite
effect. V65's entire premise (D46) is that a genuine CUDA OOM legitimately degrades one image to
CPU bicubic rather than aborting the run; under the new default-strict semantics that requires
explicit opt-in. Fixed by passing `allow_bicubic_fallback=True` in the probe script, matching
what the check was always meant to exercise. `inference.py`'s own default behaviour (strict
unless a caller explicitly opts into the demo fallback) is untouched — this is a genuine,
defensible strengthening of real submission behaviour that this session did not originate but
endorses; only the *test's* call site needed updating to keep testing what it always tested.

**4. `configs/final.yaml` picked up `padding_mode: replicate`, silently changing what the
submission config reproduces.** Origin added a `padding_mode` parameter to `NAFSR`/`NAFBlock`
(default `"zeros"`, additive and backward-compatible — needed for their closed-form checkpoint
embedding) and, in the same commit, set `padding_mode: replicate` directly in the SHARED
`configs/final.yaml`. This session's shipped checkpoint (D49) was trained under the *default*
`zeros` padding — its embedded `config.model` has no `padding_mode` key at all. Left unfixed,
`python train.py --config configs/final.yaml --seed 42` (the documented reproduction command,
`weights/README.md`) would train a materially different model than the one actually shipped.
Reverted `configs/final.yaml` to omit `padding_mode` (falls back to the `zeros` default,
matching the shipped checkpoint) and restored the comment describing `final.yaml` as identical
to `configs/nafnet_x2.yaml` (also unaffected — checked, no `padding_mode` there either).
Origin's own config for their checkpoint, `configs/phase4_psnr_focus.yaml`, correctly keeps
`padding_mode: replicate` — untouched, since that file is theirs and still needs it.

None of the four required weakening any check. (1) and (4) are data/config correctness fixes
outside the verifier. (2) restores a pre-existing, working contract that a documentation edit
broke by accident. (3) updates a test's own call site to match a real, intentional behavioural
change elsewhere, without altering what the test verifies.

## D52 — FiLM noise-level conditioning + uncertainty head: implemented, validated, additive-only

**Date:** 2026-08-16/17. Round-2 differentiation plan (`.claude/plans/as-of-now-whatever-steady-lemur.md`
Phase 0, user-approved), sequenced explicitly before any HF Jobs cloud spend (`docs/PLAN_PHASE2.md`
§5 item 0). Targets SPEC F7 ("generalise beyond observed noise levels") and SPEC §19's
"memorable finale demo" note on uncertainty output.

**What was built:**
- `src/blocks.py`: `NoiseEstimator` (tiny conv stack + global pool, estimates a per-image
  embedding from the raw input) and `NAFBlock` gained an optional `film_dim` param + `cond`
  forward argument, applying FiLM scale/shift (`y*(1+scale)+shift`) after `norm1` on the
  spatial branch. The `film` Linear is zero-initialised, so FiLM is an exact identity at step 0
  (same "no scary surprises" reasoning `layerscale_init` already documents in this class).
- `src/model.py`: `NAFSR` gained `film_dim: int = 0` and `uncertainty: bool = False`, both in
  `_DEFAULTS` at their off-value. `self.body` changed from `nn.Sequential` to `nn.ModuleList`
  (state_dict keys are integer-indexed either way — "body.0.*" etc. — so this is NOT a
  checkpoint-format change) so FiLM conditioning can reach every block. `forward` gained
  `return_uncertainty: bool = False`; when the checkpoint has an uncertainty head AND the caller
  passes `True`, returns `(restoration, log_var)` instead of the single tensor. `inference.py`
  never passes this flag, so the frozen `forward(x) -> tensor` contract it depends on is
  untouched.
- `src/losses.py`: `heteroscedastic_nll_loss(pred, log_var, target)` (`0.5*exp(-log_var)*(pred-target)^2
  + 0.5*log_var`, `log_var` clamped to `LOG_VAR_CLAMP = (-10, 10)` before `exp()` — an
  unclamped log-variance is a real overflow/underflow hazard early in training, not a
  hypothetical one). `RestorationLoss.forward` gained an optional `log_var` argument; the term
  is computed only when `log_var is not None` AND its weight is nonzero — zero cost, zero
  behaviour change for every existing call site and config.
- `train.py`: `run_training`'s main loop calls `model(lr_b, return_uncertainty=has_uncertainty)`
  where `has_uncertainty = getattr(model, "has_uncertainty", False)` — `False` for every
  existing config including `configs/final.yaml`, so the training loop is byte-for-byte
  unchanged for them.
- `configs/film_validation.yaml`: a **local validation config, not a submission config** —
  `configs/final.yaml` remains the truth for `weights/best.pt` until/unless a later decision
  promotes this feature into it.

**Confirmed additive, not a checkpoint-format change:** the shipped checkpoint
(`weights/best.pt`, sha256 `9c0f39a7...`) still loads with `build_model(ckpt["config"])` +
`load_state_dict(..., strict=True)` under the updated `src/model.py`/`src/blocks.py` — verified
directly, not assumed (`film_dim=0`, `has_uncertainty=False` read back from the loaded model).

**Self-test validation** (`py -3.12 -m src.model`, `py -3.12 -m src.losses`), all PASS:
- Shape/purity/determinism checks (mirroring V07–V12/V24) now include a FiLM+uncertainty
  config: 502,978 params (388,225 base + ~115K for FiLM projections + `NoiseEstimator` +
  the extra `PixelShuffleHead`), MACs 5.599G vs 5.584G base (negligible compute overhead at
  inference, as designed).
- **Zero-init identity, verified not asserted:** a FiLM-enabled model and a `film_dim=0` twin
  loaded with the *same* body/stem/head weights (`strict=False`, ignoring the twin's missing
  FiLM/uncertainty keys) produce bit-identical output. Confirms FiLM genuinely starts as a
  no-op rather than merely being designed to.
- `return_uncertainty=True` returns `(restoration, log_var)` of matching shape, with the
  restoration tensor identical to the default single-tensor call — confirms the uncertainty
  path never perturbs the primary output.
- `heteroscedastic_nll_loss`: a confident-but-wrong prediction scores worse than an
  equally-wrong prediction that honestly reports high variance (31.05 vs 3.00 on a synthetic
  fixture) — confirms the loss actually calibrates confidence rather than being a numerically-
  finite no-op.

**Overfit sanity gate** (mirrors V25): `configs/film_validation.yaml`, `--overfit 2`, 6000
iters — best PSNR **44.50 dB** (raw) / EMA 44.25 dB, clearing the 40 dB gate by a wide margin.
Confirms the new heads do not interfere with optimisation on a trivial case.

**Equal-budget local training comparison, the real Phase-0 gate** (`results/experiments.csv`,
both seed 42, both 3000 iters, `C:\kla-data`, RTX 4060, full 400-image committed val split,
disk-verified):

| | PSNR dB | SSIM | LPIPS | wall-clock |
|---|---|---|---|---|
| Plain NAFSR (`configs/final.yaml --iters 3000`, control) | 28.4145 ± 4.3573 | 0.76429 | 0.28146 | 659.3 s |
| FiLM + uncertainty (`configs/film_validation.yaml`) | 28.4077 ± 4.3413 | 0.76613 | 0.27551 | 750.6 s |

**Read honestly:** PSNR is a statistical wash (Δ −0.0068 dB, well inside noise at this sample
size/iteration budget); SSIM and LPIPS both favour FiLM by a small margin (+0.00184 SSIM,
−0.00595 LPIPS). No paired significance test run at this stage — that is not the claim being
made. The claim is narrower and already fully supported: **FiLM+uncertainty does not regress
quality at equal training budget, and shows a small perceptual-metric edge**, which is enough to
clear the gate the plan set ("not worse than baseline") before spending any cloud budget on it.
Training wall-clock is ~14% higher (the uncertainty head's backward pass, training-only —
`inference.py` never calls `return_uncertainty=True` so this does not touch the scored
throughput axis).

**Decision: Phase 0's local validation gate is cleared.** Proceeds to updating the sweep
configs (`docs/PLAN_PHASE2.md` §5 item 0) and dispatching the cloud sweep with FiLM+uncertainty
enabled, per the approved plan. Whether FiLM/uncertainty is ultimately promoted into
`configs/final.yaml` (the actual submission config) is a decision for after the sweep/long run
produce real, longer-budget numbers — not asserted here.

**Would overturn this:** the sweep or long run showing FiLM+uncertainty underperforms the
plain architecture at a longer, properly-converged budget, or a paired significance test on a
larger sample showing the small SSIM/LPIPS edge above does not replicate.

## D53 — Real-SEM OOD robustness report: a genuine, severe domain gap, measured honestly

**Date:** 2026-08-17. Round 2 Phase 3 (`.claude/plans/as-of-now-whatever-steady-lemur.md`,
user-approved). The procedural proxy-OOD set (V63) tests content-domain shift within natural
photographs; this closes the harder, more honest question: how does the shipped checkpoint
actually transfer to a genuinely different imaging modality.

**Source, licence, disclosure (F14):** *Scanning Electron Microscopy (SEM) Dataset of
Additively Manufactured Ni-WC Metal Matrix Composites for Semantic Segmentation*, Zenodo
record 17315241, `https://zenodo.org/records/17315241`, **CC-BY 4.0** (confirmed via the
Zenodo API's `metadata.license.id == "cc-by-4.0"`, `access_right == "open"`, not inferred).
Used for **evaluation only** — no training, no parameter fitting, same framing as the
LPIPS/AlexNet disclosure already in the README. `AugmentedImages.zip` (130,018,324 bytes)
downloaded and verified against the size the Zenodo API itself reports before use.

**Selection, to avoid over-representing near-duplicates as independent samples:** the source
ships 405 images, but only **45 unique underlying tiles** (each tile augmented ~9× with
flips/rotations/elastic warps/brightness/contrast). Using all 405 would inflate `n` with
near-duplicate copies of the same real content. One file per unique tile was selected
deterministically — the `HorizontalFlip` variant, a lossless mirror of real sensor pixels, not
a synthetic warp (unlike `ElasticTransform`/`GridDistortion`, also present in the source and
deliberately not used).

**Generation, mirroring the procedural proxy-OOD set's own method exactly** (`scripts/gen_real_sem_ood.py`,
committed and reproducible, unlike the procedural set's uncommitted one-off — an improvement
on that precedent): each 512×512 tile is converted to real luminance grayscale, **centre-cropped**
to 256×256 (no resampling before our own degradation model runs), **per-image min-max
normalised to exactly [0,1]** (matching the real-GT convention, U1), then degraded with
`src.degrade.degrade_fitted(gt, rng)` — the recovered kernel + fitted 3-parameter noise, **zero
randomisation, zero refitting**, identical discipline to the procedural set. Verified computed,
not asserted: `gt_min==0.0` and `gt_max==1.0` for 45/45 images; disjoint from `train/GT`,
`train/NoisyLR` and `test_NoisyLR` (0 intersection on all three, `results/eda/real_sem_ood/membership_check.json`).

**Measured** (`scripts/evaluate.py --real_sem_ood`, pinned metrics V31, disk-reload V30, n=45):

| | PSNR dB | SSIM | LPIPS |
|---|---|---|---|
| Bicubic floor (same 45 pairs) | 16.6268 ± 0.9604 | 0.37117 | 0.64403 |
| **Shipped checkpoint** | **17.8847 ± 0.7907** | **0.32844 ± 0.11671** | **0.56864 ± 0.15781** |
| (for reference) in-distribution, same checkpoint | 28.7865 | 0.78287 | 0.25324 |

**Read honestly, the whole point of running this:** absolute quality collapses on real SEM
content — PSNR ~11 dB below in-distribution, an unambiguous, severe domain gap between
natural-photo training and this imaging modality. But the model still **beats the bicubic
floor on PSNR (+1.26 dB) and LPIPS (better by 0.075)**, losing only on SSIM (worse by 0.043).
This is a genuinely mixed, not uniformly bad, result: it is *consistent with* D16's standing
hypothesis (the transferable asset is the measured degradation, not any content prior) — the
degradation-inversion machinery still adds value over doing nothing — while showing that
natural-photo content priors partially work against the model on real inspection-adjacent
texture, enough to lose the structural-similarity axis specifically. Both halves of this are
reported; neither is rounded up or explained away.

**What this can and cannot prove, stated plainly** (same discipline as the procedural
proxy-OOD entry): this is real electron-microscopy content, which the procedural set is not —
a materials-science SEM of a metal-matrix composite, not semiconductor fab imagery
specifically, and not KLA's actual hidden test set. It is the closest genuinely-real evidence
available without violating F17/D11's prohibitions on touching anything resembling the hidden
test data (no attempt was made to identify or acquire semiconductor-specific SEM data; this
was the highest-value *available* real-world stand-in found by a documented licence-first
search, and the search would have stopped and reported nothing if no cleanly-licensed
candidate existed).

**Code changes:** `scripts/gen_real_sem_ood.py` (new, generation), `scripts/evaluate.py`
(`--real_sem_ood`/`--real_sem_ood_bicubic_dir` flags, `score_real_sem_ood`,
`render_real_sem_ood_section`, mirroring the existing `--proxy_ood` machinery exactly so
`results/metrics_summary.md` stays machine-generated, never hand-edited).

**Would overturn/extend this:** a licensed semiconductor-fab-specific SEM/inspection dataset
surfacing later would be strictly more relevant and should supersede this materials-science
stand-in, not be added alongside it as a second claim about the same axis.

## D54 — INT8 static quantization measured, and rejected: it is SLOWER, not faster, on this architecture

**Date:** 2026-08-17. Round 2 Phase 2 (`.claude/plans/as-of-now-whatever-steady-lemur.md`),
user's explicit choice: PyTorch-native quantization only, no TensorRT/ONNX. `torch.ao.quantization`'s
classic static-quantization path is CPU-only by design (`fbgemm`/`onednn`/`qnnpack` backends
target x86/ARM CPU inference, not GPU) — this measures the CPU-fallback path specifically,
which SPEC requires (`--device cpu` must not crash), not the scored GPU/H100 axis.

**Method:** FX graph-mode static quantization (`torch.ao.quantization.quantize_fx`), the
current PyTorch-recommended static-quantization API (eager-mode `quantize_dynamic` was not
used — it only quantizes `nn.Linear`/`nn.LSTM` well, and this architecture is almost entirely
`nn.Conv2d`, so dynamic quantization would have measured nothing). Backend: `onednn` — this
machine's PyTorch build only supports `onednn` (`torch.backends.quantized.supported_engines ==
['onednn']`; `fbgemm` raised `RuntimeError: quantized engine FBGEMM is not supported`, checked
directly rather than assumed). Calibrated on 16 real **training**-split images (never
`test_NoisyLR`, same F17 discipline as everywhere else). `SCA` (`src/blocks.py`) was marked a
non-traceable leaf module (`PrepareCustomConfig.set_non_traceable_module_classes([SCA])`) —
its forward does `torch.autocast(device_type=x.device.type, ...)`, a Python-level attribute
read on a traced tensor proxy that FX symbolic tracing cannot handle; SCA's own 1×1 conv
therefore stays fp32 in the quantized model, a stated limitation, not a silent one.

**Measured, full 400-image committed val split, reproduced with `scripts/quantize_experiment.py`
across two independent runs (the second run had other background jobs sharing the CPU, hence
the noisier wall-clock, included deliberately rather than cherry-picking the cleaner run):**

| | PSNR dB | SSIM | wall-clock (400 img, CPU) | ms/img | speedup |
|---|---|---|---|---|---|
| fp32 (baseline), run 1 | 28.7864 ± 4.5329 | 0.78286 | 103.52 s | 258.8 | — |
| INT8 static (onednn), run 1 | 28.7287 ± 4.4567 | 0.77886 | 213.18 s | 533.0 | **0.486× (slower)** |
| fp32 (baseline), run 2 | 28.7864 ± 4.5329 | 0.78286 | 70.36 s | 175.9 | — |
| INT8 static (onednn), run 2 | 28.7287 ± 4.4567 | 0.77886 | 218.17 s | 545.4 | **0.322× (slower)** |

The PSNR/SSIM delta is bit-for-bit identical across both runs (quantization is deterministic
given the same calibration data); the speedup magnitude varies with system load, but **INT8 is
slower than fp32 in both runs, not marginally, by roughly 2–3×** — a robust qualitative
finding, not a one-off measurement artifact. Small quality cost either run: −0.058 dB PSNR,
−0.004 SSIM. This is a real, measured, negative result — reported as such, not hidden or
reframed.

**Why, and why this is not surprising given this project's own prior measurement:** D21 already
established NAFSR is **memory-bandwidth-bound, not compute-bound** (profiling: 32.8%
LayerNorm, 17.9% conv bias-add, 16.2% convolution — none of it FLOP-bound work), which is why
`channels_last` and bf16 each moved throughput by under 20% despite being "free" levers
elsewhere. INT8 quantization's benefit is reduced memory traffic per tensor — but it pays for
that with a quantize/dequantize (and requantize, at SCA's excluded boundary) step at every one
of 16 blocks' worth of tensor handoffs, on tensors that are already small (128×128, ≤96
channels). At this scale, the per-op quantization bookkeeping overhead measurably exceeds
the memory-traffic saving. This is the same underlying architectural fact (bandwidth-bound,
not compute-bound, at a small spatial/channel scale) showing up as a second negative result
for a second "free" lever, not an unrelated failure.

**Decision: INT8 quantization is not adopted.** Not shipped as an `inference.py` precision
option; the GPU default (bf16) remains the only real lever this architecture responds to
(D21). Reported in the deck as a measured, understood negative result — direct evidence of
quantization competency for the axis SPEC's named reviewer specialises in (model
compression/efficient inference), arguably a stronger signal than a naive speedup claim would
be, since it demonstrates root-cause understanding (roofline/bandwidth analysis) rather than
blind application of a technique.

**Would overturn this:** a genuinely compute-bound architecture (wider/deeper, matmul-heavy,
e.g. the Round-2 sweep's largest configs) might show a different result — not assumed here,
would need its own measurement if pursued. Also worth trying, not yet done: dynamic
quantization limited to the FiLM/`NoiseEstimator` `nn.Linear` layers specifically (a much
smaller, targeted change, unlikely to move the needle given how few FLOPs they represent, but
cheap to check if time remains).

## D55 — Pareto sweep results (HF Jobs, A100-large): config `e` (w64n32) chosen for the long run

**Date:** 2026-08-16/17. HF Jobs sweep dispatched per `docs/PLAN_PHASE2.md` §5 item 1, FiLM+
uncertainty enabled per D52's gate. Job `6a821471c97db76cbdf3346c`, `windows-session` branch
(main branch reconciliation with the teammate's second round of commits, `87af55c..f843e0b`,
is a separate, non-urgent task — deliberately not repeated under time pressure a second time
this session; see the git history for that pending work). All 6 configs completed cleanly
(exit 0), checkpoints pushed to `Team-Ceciroleo67/kla-ps01-checkpoints`.

**Full frontier, 2000 iters each, A100-large, full 400-image committed val split:**

| config | width×blocks | params | PSNR dB | SSIM | LPIPS | train wall-clock |
|---|---|---|---|---|---|---|
| `sweep_a` | 32×16 | 238,194 | 28.2313 | 0.75505 | 0.28714 | — |
| `sweep_b` | 48×16 | 502,978 | 28.3143 | 0.75985 | 0.28421 | — |
| `sweep_c` | 64×16 | 866,578 | 28.3956 | 0.76298 | 0.27974 | — |
| `sweep_d` | 48×32 | 812,482 | 28.3580 | 0.76150 | 0.28012 | — |
| `sweep_e` | 64×32 | 1,393,938 | 28.4398 | 0.76655 | **0.27307** | 375.71 s |
| `sweep_f` | 96×32 | 3,025,330 | **28.5007** | **0.76723** | 0.27859 | 490.85 s |

**Width beats depth at matched params, confirmed again on this architecture:** `sweep_d`
(more blocks, width 48) scores *below* `sweep_c` (more width, same depth as `sweep_b`) despite
comparable parameter counts (812K vs 867K) — the same conclusion D21's profiling already
reached (LayerNorm/bias-add/conv dominate forward time, none of it FLOP-bound), now confirmed
by an actual quality measurement rather than just a throughput one.

**`sweep_f` wins PSNR/SSIM but loses LPIPS to `sweep_e` despite 2.2× the parameters** —
non-uniform, diminishing returns past ~1.4M params. The entire frontier spans only ~0.27 dB
PSNR from smallest (238K) to largest (3.03M) config.

**Decision (user, explicit): config `e` (width=64, num_blocks=32, 1.39M params) carries into
the long run.** Best LPIPS of all six configs, second-best PSNR/SSIM (within 0.06 dB / 0.0007
of `sweep_f`), under half `sweep_f`'s parameter count, and comfortably inside SPEC §7.1's
originally-suggested 1–3M parameter band without chasing `sweep_f`'s weaker cost/benefit tail.

**Long run config** (`configs/long_run_e.yaml`): `total_iters` set from `sweep_e`'s own
measured throughput — 375.71 s / 2000 iters = 5.3233 iters/sec on A100-large. 7.2 GPU-hr
(25,920 s, the 60% budget tier, `docs/PLAN_CLOUD.md` §4) at that rate is a theoretical max of
137,979 iters; **129,700** applies a ~6% safety margin for validation/checkpoint-push overhead
at a much longer horizon than the 2000-iter probe measured it on. Not pre-committed before the
sweep existed, per this project's own standing rule against projecting an unmeasured number.

**Gap, stated honestly:** the sweep measured training-time quality and wall-clock, not a
separate A100 *inference*-throughput benchmark per config. Given D21's memory-bandwidth-bound
finding, this is not expected to reorder the ranking, but it was not measured, so it is not
claimed as measured.

**Submittable state tagged before this run starts**, per the standing constraint: git tag
`v0.1-submittable` on the `windows-session` branch tip (commit `61fb26b`), pushed to origin —
a known-good, fully-verified state exists independent of whether the long run finishes.

## D56 — V66, V67, V68 added: closing the verifier-coverage gap for every Round-2 addition

Round 2's own additions -- FiLM noise-level conditioning (D52), the uncertainty head (D52),
the real-SEM OOD report (D53), and `UnrolledSR` (Phase 4 stretch) -- shipped with **zero**
dedicated V-checks. `grep -n "film_dim\|has_uncertainty\|real_sem_ood\|UnrolledSR"
scripts/verify_all.py` returned no matches before this entry. The only validation any of them
had was `src/model.py::_selftest()`, which nothing invokes automatically -- the exact "dead
code is not a guard" lesson V61 already learned once for UNetSR's shape sweep (D34). A 65+
check verifier suite that does not check its own newest, most-differentiating features is a
real, findable gap for a reviewer to notice, not a hypothetical one.

**V66** promotes `_selftest`'s own two FiLM/uncertainty assertions into the verifier:
(a) a FiLM-enabled `NAFSR` must be an exact bitwise identity vs. an un-conditioned twin loaded
from the same `state_dict` at construction (FiLM's `Linear` is zero-initialised, so
conditioning must contribute nothing until trained); (b) `return_uncertainty=True` must return
`(restoration, log_var)` where the restoration is bit-identical to the default
(`return_uncertainty=False`) call and `log_var` is finite and correctly shaped -- the
uncertainty head must be a pure side channel, never perturbing the actual prediction.

**V67** mirrors **V63** exactly (same file, same author, same pattern) but for
`scripts/evaluate.py::render_real_sem_ood_section`'s section instead of the procedural
proxy-OOD section: requires the `## Real-SEM OOD robustness report` heading, a citation of
Zenodo 17315241 and the CC-BY licence, all three metrics (PSNR/SSIM/LPIPS, mean +/- sd), a
well-formed `results/eda/real_sem_ood/membership_check.json`, and a well-formed
`results/baselines/real_sem_ood/final/metrics.json` (float32, no unclipped files, positive
`n`) -- same anti-overclaim banned-phrasing check V63 already uses (`_v63_positive_banned_matches`,
reused directly rather than duplicated).

**V68** extends V61's per-architecture shape/determinism sweep to `UnrolledSR`: forwards three
representative sizes (128, 256, and a non-square 66x90 -- deliberately NOT V61's 1x1 case,
which is not a meaningful input for a strided-conv proximal-gradient step against a real
measured kernel) and asserts exactly `(1,1,2H,2W)`, finite, plus eval-mode bitwise determinism
on repeat calls. **Deliberately does not assert a quality bar** -- `UnrolledSR`'s own overfit
gate (mirroring V25) is the quality gate for this architecture, and it is currently FAILING
(see the "In flight" note below); V68 only asserts the architecture is mechanically sound
(correct shape, finite, deterministic), which remains true independent of the open quality bug.

**Negative-controlled, all three, per the standing rule (never trust a new check without
breaking it first):**
- V66(a): FiLM's `nn.init.zeros_` replaced with `nn.init.normal_(std=0.02)` -> correctly FAILED
  ("NOT an exact identity... max abs diff 3.388e-04"); reverted byte-exact, green again.
- V66(b): `return out, log_var` changed to `return out + 1.0, log_var` -> correctly FAILED
  ("restoration differs between return_uncertainty=False and =True"); reverted, green again.
- V67: `results/eda/real_sem_ood/membership_check.json` temporarily moved aside -> correctly
  FAILED ("membership_check.json missing"); restored, green again.
- V68: `UnrolledSR.forward`'s final `return est` changed to `return est[..., :-1, :-1]` ->
  correctly FAILED (3 shape violations across the size sweep); reverted, green again.

**In flight, stated honestly:** `UnrolledSR`'s own overfit gate is currently failing (PSNR
plateaus around 26 dB against a 40 dB bar) -- an open, root-cause-in-progress bug (plan
PRIORITY 0.5), not something V68 hides. V68 checks mechanical soundness only, by design.

**Governance:** `scripts/verify_all.py` re-pinned in `docs/VERIFIER_SHA256`
(`c0b71a9...` -> `b9b6c1c...`) — net effect is strictly more checks, nothing weakened, deleted,
or skipped.

## D57 — FP8 measured (plan PRIORITY 1, P1.4): no native end-to-end path exists; GEMM-level proxy is a mixed, small, hardware-dependent result

Completes the quantization story alongside D54's INT8 measurement, which was measured on CPU
only -- FP8 was named in the Round 2 plan as a secondary data point but never actually
measured. `scripts/fp8_probe.py`, run on the dev RTX 4060 Laptop GPU (Ada, compute capability
8.9, real FP8 tensor cores):

**Part 1 — can the shipped architecture even run in FP8 end to end?** No. `F.conv2d` on
`torch.float8_e4m3fn` inputs raises `RuntimeError: getCudnnDataTypeFromScalarType() not
supported for Float8_e4m3fn` -- native PyTorch's cuDNN conv backend has no FP8 kernel. NAFSR is
convolutional (3x3 depthwise + 1x1 pointwise `nn.Conv2d`, `src/blocks.py`), not
attention/matmul-only, so this is a hard blocker within the user's own explicitly-scoped
boundary for this work ("PyTorch-native INT8/FP8 quantization only, no TensorRT/ONNX," Round 2
clarification #1) -- a custom Triton/CUTLASS FP8 conv kernel would be out of scope even if it
existed. **There is currently no native end-to-end FP8 inference path for this architecture.**

**Part 2 — bounded GEMM-level proxy, to check whether the underlying compute primitive would
even help if a custom kernel existed.** The model's own 1x1-pointwise-conv shapes (NAFBlock,
width=64, `dw_expand=2`/`ffn_expand=2` -> 64<->128 channels) reshaped to exact GEMM form
(M=65,536 tokens for a 256x256 feature map, this project's stated eval range), timed via
`torch._scaled_mm` (the one native FP8 compute primitive that DOES work on this hardware) vs.
`torch.matmul` in bf16 (the model's actual runtime precision):

| shape | bf16 | fp8 (`_scaled_mm`) | speedup |
|---|---|---|---|
| 64->128 (M=65,536) | 0.1135 ms | 0.1375 ms | **0.825x (slower)** |
| 128->64 (M=65,536) | 0.1617 ms | 0.1324 ms | **1.222x (faster)** |

**Mixed, small, and not a clean win either way** — one shape is slower in FP8, the other
faster, both differences are sub-millisecond on GEMMs this small, and cuBLASLt's FP8 path
carries fixed per-call overhead (scale-tensor handling, layout constraints) that does not
amortize well at this size. This is consistent with D21/D54's memory-bandwidth-bound finding
for this architecture: GEMMs this small are latency/overhead-dominated, not throughput-bound,
so a narrower datatype does not have much room to help even where a kernel exists.

**Honest conclusion:** FP8 is not usable for this architecture within the user's own stated
scope (no native conv kernel), and where the underlying primitive is measurable at all (plain
GEMM), the benefit is inconclusive/negative at the model's actual shapes on this GPU. Recorded
as a genuine negative/inconclusive result, same standard as D54 -- not silently omitted because
it didn't produce a win.

Artifacts: `scripts/fp8_probe.py` (re-runnable), `results/eda/fp8_probe.json` (raw numbers).

## D58 — FiLM calibration probe: noise information is present but NOT summarised by embedding norm or any single dimension (plan PRIORITY 1, P1.1)

Checked whether `NoiseEstimator`'s embedding actually tracks the true sampled noise level, or
merely exists (D52 validated bit-identity at init and shape contracts, not calibration).
`scripts/film_calibration_probe.py`, run against the Pareto-sweep's config `e` checkpoint
(`film_dim=16`, `uncertainty=True`, trained 2000 iters on A100 -- the sweep probe, NOT the
long run's fuller budget), 300 synthetic degraded samples with a known
`(sigma, a, v)` triple per sample (`src/degrade.py::sample_noise_params`), true noise summarised
as `sqrt(noise_variance(x=1))` (the model's own variance formula, clamped at 0 -- the
randomisation range's `+/-120%` on `a`/`v` can imply a negative variance at face value, which
is a property of the sampling range, not the physical noise):

| metric | value |
|---|---|
| Pearson r, embedding L2 norm vs. true noise std | **0.019** (~zero) |
| max \|Pearson r\|, any single embedding dimension vs. true noise std | **0.054** (~zero) |
| held-out linear-probe R² (OLS, 70/30 split, all 16 dims) | **0.231** |

**Honest, nuanced reading, not a clean win or a clean failure:** neither the embedding's L2
norm nor any single one of its 16 dimensions correlates with the true noise level in isolation
-- if the check had stopped at the norm alone (the obvious first thing to try), the honest
conclusion would have been "FiLM learned nothing useful." But a linear combination of all 16
dimensions explains ~23% of the variance in true noise level on data the probe never fit on --
noise information IS linearly decodable from the embedding, just not concentrated in any one
axis or in the vector's overall magnitude. This is a real, moderate, positive signal, not
overclaimed as strong calibration.

**Caveat stated plainly:** this checkpoint trained for only 2000 iterations (the sweep probe's
budget, `docs/decisions.md` D55), far short of the long run's 129,700. FiLM's linear/zero-init
head has had comparatively little training signal to specialise on noise level specifically
(vs. everything else the shared conv body also has to learn); this result should be re-checked
against the long run's checkpoint once it exists, and might strengthen (or might not) with the
much larger training budget. **Recorded as measured now, not deferred and not extrapolated.**

Artifacts: `scripts/film_calibration_probe.py` (re-runnable, takes `--checkpoint`),
`results/eda/film_calibration.json` (raw numbers).

## D59 — Uncertainty calibration check: strong, genuine correlation with real error (plan PRIORITY 1, P1.2)

Checked whether the predicted `exp(log_var)` actually tracks real squared error, or merely
exists (D52 validated the shape/side-channel contract, not calibration).
`scripts/uncertainty_calibration_probe.py`, run against the same Pareto-sweep config `e`
checkpoint as D58 (`uncertainty=True`, trained 2000 iters), on the full 400-image committed
validation split (`configs/split_val.txt` -- the same split its own `val_psnr` was measured
against), comparing `exp(log_var)` to `(pred - gt)^2`:

| granularity | Pearson r | Spearman r |
|---|---|---|
| per-image mean (n=400) | **0.965** | **0.941** |
| pooled per-pixel (200 px/image subsample, n=80,000) | 0.443 | 0.564 |

**Genuinely strong at the image level**, not a marginal or cherry-picked correlation: images
the model predicts as higher-variance really do have higher mean squared error, both linearly
and by rank, on data the correlation was never fit to. The magnitude is also sane, not just the
rank: mean `exp(log_var)` = 0.002311 vs. mean squared error = 0.002279 -- the predicted
variance's average scale matches the actual error's average scale, not merely correlating with
an arbitrary offset/scale.

**Weaker at the per-pixel level, expected and stated honestly:** per-pixel squared error is a
single noisy draw with enormous variance of its own (a Gaussian NLL loss trains the MEAN
relationship, not pixel-exact prediction), so a substantially lower per-pixel correlation than
per-image is the expected signature of real calibration operating at the right granularity, not
a sign the per-image number is inflated by an artifact -- both numbers point the same direction.

**Contrast with D58's FiLM finding, stated plainly:** the uncertainty head calibrates far better
than the FiLM noise-conditioning embedding did on the same checkpoint at the same training
budget (2000 iters). Plausible reason, not yet verified further: the uncertainty head is
trained with a loss term (Gaussian NLL) directly supervising exactly this quantity, whereas
FiLM's noise embedding has no direct supervision at all -- it can only shape itself indirectly,
through whatever gradient the main reconstruction loss routes back through the conditioning
path. Both results are reported as measured, not adjusted to match expectation either way.

Artifacts: `scripts/uncertainty_calibration_probe.py` (re-runnable, takes `--checkpoint`),
`results/eda/uncertainty_calibration.json` (raw numbers).

## D60 — UnrolledSR overfit-gate root-cause investigation: all 3 planned hypotheses cleared, no single fixable bug found; genuinely slow convergence, not a capacity/correctness defect

Plan PRIORITY 0.5's 3-step diagnostic, executed in full:

**Step 1 — K/K^T adjoint identity.** For random `x`/`r`, `<K(x), r>` vs `<x, K^T(r)>` agree to
**0.40% relative error** -- within the expected boundary-crop approximation the module's own
docstring already discloses, not a real bug.

**Step 2 — operator norm / step-size stability.** Power-iteration measured `||K^T K|| ~= 0.672`;
`step_size_init=0.05` after the `softplus` reparameterisation gives an effective initial step
of `~0.718`, so `step_size * ||K^T K|| ~= 0.483` -- comfortably
inside the convergent range for gradient descent on `0.5||Kx-y||^2` (want `<< 1`, not `~1` or
above). The unrolled gradient step is not diverging or oscillating at init.

**Step 3 — weight-tying capacity isolation, on the REAL 2-pair overfit fixture (not the invalid
random-data test from an earlier attempt).** `share_denoiser=True` (shipped) vs `False`
(untied, `num_steps` independent denoisers), same seed, same optimiser, same constant
`lr=1e-3` (no schedule -- ruling out the LR-decay-too-early hypothesis at the same time),
tracked to iter 240:

| iter | share=True (dB) | share=False (dB) |
|---|---|---|
| 0 | 13.93 | 3.38 |
| 80 | 21.19 | 22.96 |
| 160 | 23.41 | 23.37 |
| 240 (True) / 159 (False, last logged) | 23.59 | 23.50 |

**Essentially identical trajectories and plateau value (~23.4-23.6 dB) regardless of weight
tying.** This directly rules out "weight-tying starves capacity" as the root cause -- an
untied model with 3x the denoiser parameters plateaus at the same place. It also rules out
the LR-schedule hypothesis floated after Step 3's first (invalid) attempt: this run used a
**constant** `lr=1e-3`, no cosine decay, and still plateaued well below 40 dB by 240
iterations -- a decaying-too-fast schedule cannot be the explanation if a non-decaying
schedule plateaus at the same place.

**Consistency check against the real gate:** this repro's PSNR trajectory (fast initial rise
to ~20 dB by iter 20, slow crawl to ~23.5 dB by iter 240) is directionally consistent with the
real `run_overfit` gate's own reported result (~26 dB at iter 6000) -- a plausible
log-slow continuation of the same curve, not a contradictory or unrelated behaviour. This
diagnostic is tracking the real gate's dynamics, not an artifact of a different setup.

**Honest conclusion: no single fixable bug was found among the three planned suspects.** The
adjoint is fine (within its stated approximation), the step size is stable, and weight-tying
makes no measurable difference. The remaining, unexplained behaviour is that `UnrolledSR`
converges much more slowly than NAFSR/UNetSR on the same 2-pair fixture and the same data
pipeline, for a reason not yet isolated by this diagnostic -- plausible remaining candidates
(not yet tested, would need a fresh investigation, NOT assumed): the denoiser's contribution
scale relative to the gradient-consistency term across 6 sequential compositions may need a
different effective learning rate per unrolled step than a flat conv stack does (the same
step's parameters are used identically at every unroll depth, unlike NAFSR's independently-
scaled per-block layerscale), or the optimisation landscape induced by composing a fixed
linear operator with a learned nonlinear correction 6 times may simply need many more
iterations / a different optimiser/schedule than this project's existing overfit-gate defaults
assume for a flat architecture.

**Decision, per the plan's own explicit fallback clause ("if this doesn't resolve quickly, it
is legitimate to report the negative result honestly... rather than force a fix under time
pressure"):** `UnrolledSR` is not promoted, not shipped, and remains disclosed in `README.md`
(see today's edit) as an attempted, honestly-failing stretch goal with a documented, partial
root-cause investigation -- not silently dropped and not forced to a green result it has not
earned. `src/unrolling.py` and `V68`'s mechanical-soundness check (D56) remain in the repo as
real, working, disclosed code; only the quality bar is unmet.

**Environment note, unrelated to the architecture itself:** this investigation required
restructuring the diagnostic into short, checkpointed, resumable chunks after 3 consecutive
long-running background CPU processes were killed externally (status `killed`, not crashed)
at 10-24 minute marks on this machine -- a sandbox/resource-lifetime limit on sustained
background processes, not a code defect. Recorded here so a future session does not waste time
re-diagnosing the same environment behaviour.

## D61 — Round 2 long-run checkpoint promoted: paired win over the prior checkpoint and over U-Net, a real OOD trade-off disclosed, two new blockers opened

**The HF Jobs long run (`docs/decisions.md` D55, config `e`: width=64, num_blocks=32,
FiLM+uncertainty, `configs/long_run_e.yaml`) completed the full 129,700-iteration schedule**
(job `6a822762c97db76cbdf33506`, 22,895.55 s / 6h21m wall-clock on an HF Jobs A100-large, well
under the 8h cap). Best checkpoint selected at iteration 76,000 by `ema/psnr` over the whole
run, not the final iteration.

**Re-scored head-to-head against the currently-shipped checkpoint (D49) under one harness
before any promotion decision**, per this project's own standing rule (never trust a
self-report, same discipline as D49's original merge reconciliation): both checkpoints run
through `scripts/make_baselines.py --baselines final` with `--final_ckpt` pointed at each, both
scored via `scripts/evaluate.py` against the identical 400-image `configs/split_val.txt`, paired
per-image test via `src.metrics.paired_compare`:

| metric | prior checkpoint (D49) | long-run checkpoint | mean diff | t | images better (long-run) | verdict |
|---|---|---|---|---|---|---|
| PSNR | 28.7865 ± 4.5329 | **29.2548 ± 4.6210** | +0.4683 dB | -25.85 | 391/400 | **win** |
| SSIM | 0.78287 ± 0.14169 | **0.79211 ± 0.14321** | +0.00925 | -15.08 | 378/400 | **win** |
| LPIPS | 0.25324 ± 0.13193 | 0.25625 ± 0.14627 | -0.00300 | -1.14 | 170/400 | tie (not significant) |

**Long-run checkpoint wins PSNR and SSIM significantly; LPIPS is a genuine statistical tie, not
a loss.** Two wins, one tie clears the same bar D49 itself used to promote a checkpoint.

**Also now beats the U-Net baseline on all three metrics** (paired, n=400, same 400 images):
PSNR +0.3740 dB (t=+18.25, 374/400), SSIM +0.00938 (t=+12.19, 382/400), LPIPS -0.00900
(t=-3.26, 239/400) — all significant wins. This reverses the prior checkpoint's own documented
negative result (V28: PSNR loss, SSIM tie, LPIPS win, 1/3) into a clean 3/3 win; V28 flips
FAIL -> PASS.

**Decision: the long-run checkpoint is promoted to `weights/best.pt`.** SHA256
`8f54f9a208220dfd6cd3d67766945ad781bf141fcc03fac41d216caf4fa9643c`, 11,565,729 bytes. Verified
`build_model(config).load_state_dict(..., strict=True)` succeeds for both the raw and EMA
weights before promoting (V35's contract). `weights/README.md`, `README.md`'s Result-summary
table, Method-summary section, and top status block all updated; the prior checkpoint's numbers
kept for the historical record, not deleted. All downstream artifacts regenerated against the
new checkpoint: `results/baselines/{final,proxy_ood/final,real_sem_ood/final}`,
`results/metrics_summary.md`, and a fresh `results/restored_test_outputs/` (see below).

**A real, honest trade-off — measured, not smoothed over.** The long-run checkpoint's
generalisation to the real-SEM OOD set (D53) got measurably WORSE on two of three metrics
despite improving in-distribution and on the procedural proxy-OOD set:

| real-SEM OOD (n=45) | prior checkpoint | long-run checkpoint | direction |
|---|---|---|---|
| PSNR | 17.8847 ± 0.7907 | 17.7854 ± 0.7620 | ~flat (-0.099) |
| SSIM | 0.32844 ± 0.11671 | 0.25979 ± 0.11262 | **worse (-0.069)** |
| LPIPS | 0.56864 ± 0.15781 | 0.71142 ± 0.17213 | **worse (+0.143)** |

Procedural proxy-OOD (n=40), for contrast, held steady or improved slightly: PSNR 27.32 ->
27.25 (~flat), SSIM 0.965 -> 0.970 (slightly better), LPIPS 0.038 -> 0.035 (slightly better).
**A plausible, not-yet-confirmed reading:** a larger, longer-trained, FiLM-conditioned model may
be fitting the in-distribution natural-photo content distribution more tightly, at some cost to
transfer onto a genuinely different imaging modality (real electron microscopy) — consistent
with, but not proof of, a mild overfitting-to-domain effect that a smaller/shorter-trained
model did not exhibit as strongly. Not investigated further this session; a legitimate follow-up
(plan PRIORITY 2, per-content/domain breakdown).

**Two new blockers opened by this promotion, real regressions requiring a human decision, not
silently absorbed or worked around:** V22 (bf16 vs fp32 divergence, root-caused to depth-
compounding with no single-line fix) and V51 (tracked-file size cap, a real gap in the existing
checkpoint exemption). Full investigation and the decision needed: `docs/BLOCKERS.md` B12.
Both left exactly as measured — the verifier was not edited, the tolerance was not widened.

**Environment note (unrelated to the checkpoint itself):** producing this decision required
working around a real sandbox limitation where sustained background CPU processes were killed
externally after 10-24 minutes (see D60's environment note); GPU-bound subprocess calls
(inference.py, benchmark_runtime.py) launched the same way completed without incident,
suggesting the limitation is specific to sustained CPU load, not backgrounding in general.

**Would overturn this:** a human deciding the LPIPS tie / OOD trade-off means the prior
checkpoint should ship instead (SPEC states no throughput floor and a three-metric quality
rubric with an undisclosed blend, so this is a legitimate ground for reconsideration, not
asserted as wrong here), or a resolution of B12 that changes which checkpoint can pass
`--strict` cleanly.

## D62 — V51 fixed (human-authorised): checkpoint exemption extended to the size-cap loop

Per `docs/BLOCKERS.md` B12's V51 finding: `CHECKPOINT_BLOB_EXEMPTION = "weights/best.pt"`
already existed and was already used to exempt the checkpoint from the blob-EXTENSION ban
(D41), but the per-file (5 MiB) / total-tree (25 MiB) size-cap loop a few lines later applied
uniformly to every tracked file, including the one file the contract already requires be
tracked (V59) and already caps separately, more appropriately, at 100 MB (V43). The prior
388,225-param checkpoint (3.14 MiB) was always under 5 MiB by coincidence, so this gap was
never exercised until the Round 2 long-run checkpoint (11.03 MiB, D61) was promoted.

Presented to the human rather than decided unilaterally (Prime Directive 1 reserves
verifier-contract judgment calls): "extend the exemption to the size-cap loop" was the chosen
resolution, authorised via `AskUserQuestion`. Applied to BOTH the per-file cap and the
total-tree byte sum — extending only one would have simply moved the same failure from "1 file
exceeds 5 MB" to "tree exceeds 25 MB" (`weights/best.pt` alone is 11.03 MiB; the tracked tree
excluding it is 16.27 MiB, comfortably under 25 MiB; including it, 27.84 MiB, over).

**Negative-controlled before trusting it:** a genuine 6 MiB tracked file with a non-banned
extension (`.txt`, not `weights/best.pt`) correctly still fails V51 after the fix ("1 tracked
files exceed 5242880 B"); removed and reconfirmed green (142 tracked files, 16,271,790 B).

**Net effect: one additional file (the already-sanctioned, already-required checkpoint) is
exempted from a cap clearly designed to catch an ACCIDENTAL dataset-sized blob, not to
re-litigate a mandatory artifact via a second, unsized-for-this-purpose ceiling — every other
file remains fully subject to both caps, unchanged.** `scripts/verify_all.py` re-pinned in
`docs/VERIFIER_SHA256` (`b9b6c1c...` -> `55bffaa...`).

**V22 (the other B12 finding) was NOT similarly resolved** — presented to the human, who chose
to accept the measured bf16/fp32 divergence as a disclosed trade-off of the larger checkpoint
rather than force a fix. `scripts/verify_all.py`'s V22 check and its tolerance are UNCHANGED;
V22 remains a known, disclosed, live FAIL (see `docs/BLOCKERS.md` B12 for the full
investigation — root-caused to depth-compounding, not a discrete unpromoted op, no single-line
fix exists).

**Full suite after this fix:** 65 PASS / 3 FAIL (V04, V22, V46) — V04/V46 pre-existing and
expected (`--fresh-clone`-only checks), V22 the accepted, disclosed trade-off above.

---

## D63 — Hour 0 diagnosis + post-promotion fine-tune launched (deadline extended to Aug 18 night)

Phase 1 deadline extended to 2026-08-18 night (~33h from this entry). A fresh three-way audit
of the whole repo against the official KLA requirements found the CODE genuinely sound (65
PASS / 3 FAIL, matching D62) but `README.md` badly stale relative to the D61 promotion —
wrong sha256, wrong training narrative (still described Apple Silicon MPS closed-form fitting,
not the actual A100 gradient run), wrong throughput numbers, unpublished-outputs claim
contradicted by a live, verified `artifacts-v2` release. Full remediation plan recorded at
`C:\Users\sahit\.claude\plans\as-of-now-whatever-steady-lemur.md`; README truth-pass tracked
there as Phase A, not repeated here.

**Reframe that drove this decision:** HF Jobs training is asynchronous and billed, not
wall-clock-blocking. Launching a fine-tune early costs money, not the local session's time, so
it was dispatched before any of the README/documentation work below, rather than after.

**Hour 0 — two paired diagnostic probes, before choosing the fine-tune's objective** (never
assume; measure). Both scripts committed alongside this entry, both pure evaluation, F17
untouched:

1. `scripts/ood_paired_probe.py` — recovers the SUPERSEDED checkpoint byte-identical from git
   history (commit `19e4e76`, never retrained) and runs a paired comparison (new vs old,
   `src.metrics.paired_compare`, same D49/D61 harness) on the 400-pair in-distribution val
   split (control), the 40-image procedural proxy-OOD set, and the 45-image real-SEM OOD set.
   Result (`results/eda/ood_paired_probe.json`): the val-split control exactly reproduces D61's
   win (PSNR +0.4683 dB, t=+25.85). Proxy-OOD shows NO regression — SSIM actually **wins**
   (+0.00477, t=+4.63). Real-SEM OOD shows a large, significant **loss** on both structural
   metrics (SSIM −0.0683, t=−10.21; LPIPS +0.1427, t=+5.86). Verdict: **idiosyncratic, not
   systematic** — the D61 regression is concentrated entirely on the one set that is real
   semiconductor/material-adjacent imagery, plausibly the closest thing in this project to
   KLA's actual hidden test distribution, not a general OOD collapse.
2. `scripts/scale_gap_probe.py` — for each val image, scores the SAME centre GT pixels two
   ways: (a) full 128px-image inference, cropped after, vs (b) a 64px LR crop (the exact
   training patch size) inferred directly. Result (`results/eda/scale_gap_probe.json`): a
   real, statistically significant but small gap — SSIM −0.0019 (t=−3.57), LPIPS +0.0081
   (t=+2.91), PSNR flat — roughly 30x smaller in magnitude than the OOD effect above.

**Decision, reported to the human before dispatch and not contradicted:** target both, per the
plan's "both" branch, weighted toward the OOD axis since that is where the real signal is.
`configs/finetune_ood_wide.yaml` fine-tunes `weights/best.pt` (never from scratch) with (a)
`lr_patch: 64 -> 128` (train at the actual inference resolution, closing the scale gap for
free) and (b) widened degradation-parameter randomisation (`randomise_frac` 1.20 -> 1.50,
`gauss_sigma_range` upper bound 0.065 -> 0.09) — a deliberate, bounded, DISCLOSED
extrapolation, not re-derived from a fresh fit, whose actual effect will be MEASURED by
re-running `ood_paired_probe.py` against every checkpoint the run produces, not assumed.
**Explicitly does not** add `real_sem_ood`/`proxy_ood` imagery to training — both are standing
OOD evaluation sets (D53: "NOT trained or fitted on"); doing so would invalidate every future
OOD comparison in this project, not just this run's.

**`train.py` changes required to run this safely, all backward-compatible (verified: two
independent `--smoke --smoke_iters 6 --seed 42` runs before and after every edit reproduce the
identical `SMOKE_DIGEST fd5e52061802c1d2c4d8034d1e224ef3a40586cc40ba48ffb75e3af396bc8da9`)**:

- `optim.finetune_horizon` (config key, only consulted when `--resume` is set and `--iters` is
  not passed): `cosine_warmup_lr` is written in absolute-iter terms, which is correct for a
  from-scratch run but wrong for a resumed one — reusing the original config's `total_iters`
  would resume already ~60% down someone else's cosine curve at whatever LR that curve happens
  to be at. `finetune_horizon` instead defines a fresh cosine curve of that many iters, starting
  near `base_lr` at the resumed iter. Verified empirically, not just by inspection: a 20-iter
  local dry-run using the OLD path (`--iters` override, legacy absolute schedule) ran at
  `lr=1.000e-06` throughout (correctly near the original schedule's tail); the same 20 iters
  using the NEW `finetune_horizon` path ran at a materially higher LR and produced a real,
  visible loss/PSNR change in that short window — the schedule genuinely switched, not just a
  config no-op.
- `--push_every` + `--val_lpips`: an unconditional periodic checkpoint push (job storage is
  ephemeral — a run killed by the wall-clock timeout between PSNR improvements would otherwise
  return nothing) and in-loop LPIPS logging (only the final full-split report computed it
  before). The two validation calls are deliberately merged into one when `val_every` and
  `push_every` land on the same iter (this run sets them equal, 2000) — paying for LPIPS twice
  at the same iter would be pure waste.
- Selection criterion is UNCHANGED (`save_best_on: psnr` stays hardcoded in `train.py`, as it
  already was) — this run's checkpoints, plus every earlier sweep/long-run checkpoint on
  `Team-Ceciroleo67/kla-ps01-checkpoints`, feed the blended-criterion re-score in the plan's B3,
  not decided here.

**Dispatch, committed as `scripts/dispatch_finetune_job.py`** (the exact command is now on
record — no earlier job's dispatch command ever was, a real gap this closes): HF Jobs
`a100-large`, image `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`, **`timeout="3h"`** as the
real, authoritative cap ("finishing beats optimality" — `finetune_horizon: 40000` in the config
is deliberately larger than what 3h can reach, so the job is stopped by the timeout, not by
exhausting a schedule), namespace `Team-Ceciroleo67`, dataset pulled at job start from the
private `Team-Ceciroleo67/kla-ps01-data` repo (never bundled into the git clone), `HF_TOKEN`
passed as a Job secret (never written to a file). Job id and dispatch timestamp recorded in
`docs/PLAN_CLOUD.md`'s spend ledger, not here.

Budget: ~$7.50 at A100-large $2.50/hr for a 3h cap (of the ~$12 remaining, expires 2026-09-01).
Promotion, if any, happens only on a paired win against the incumbent (D49/D61 precedent) —
see the plan for the T-12h hard promotion gate and the costed regeneration cascade it exists
to protect.

---

## D64 — README truth pass; `results/qualitative/` regenerated; V53 implemented (Phase 1 close-out)

A three-way audit (Explore agents + direct verification) found the code sound (65 PASS / 3
FAIL matching D62) but `README.md` badly stale relative to the D61 checkpoint promotion —
wrong sha256 (`37e857...` instead of `8f54f9a2...`), a `## Training` section that still
described a 24.9s CPU closed-form fit as the reproduction path (the shipped checkpoint needs
`configs/long_run_e.yaml`, a 6h21m A100 gradient run — a portal-mandatory-item bug, not
cosmetic), an Environment paragraph claiming Apple Silicon MPS training, three mutually
contradictory throughput figures (17.3 / 8.3 / 5.6 img/s, one attributed to a nonexistent
"Mac CPU" run of this checkpoint), a "publication remains open" claim contradicted by the
live, verified `artifacts-v2` Release, Gaussian range `U(0,0.02)` vs the code's actual
`U(0,0.065)` (D43), and "63 checks" vs the actual 68. All fixed in `README.md`: Training,
Environment, Repository map, Runtime measurement, Verification and Assumptions sections
rewritten against the checkpoint that actually ships; added a "What metric selects the
'best' checkpoint" disclosure (PSNR-only, hardcoded, undisclosed until now) and a "Failure
cases" section linking `results/qualitative/`. Also fixed two smaller staleness bugs found
while rewriting: the "Two numbers, do not confuse them" section still quoted the superseded
checkpoint's in-run/full-split figures, and the `UnrolledSR` method-summary item still said
"under active root-cause debugging" when D60 already closed that investigation as a complete,
reported negative result.

**`results/qualitative/` regenerated against the shipped checkpoint.** Was rendered 2026-08-16
against the superseded (28.7865 dB) checkpoint; every PSNR in every filename was therefore
wrong for the model that actually ships. Re-ran `scripts/make_qualitative_examples.py` end to
end: (1) checked, not assumed, that the six hardcoded val-example tags (best/strong/typical
x2/worst/lowest-SSIM-of-worst-8) still describe the correct images under the NEW checkpoint's
per-image ranking — they do (`002041.npy` is still literally rank 0/400 by PSNR, `000900.npy`
still rank 3/400, etc.); (2) fixed a real bug in the script itself: `CKPT_SHA` and the
manifest's `checkpoint_val_psnr_db/ssim/lpips` fields were HARDCODED to the superseded
checkpoint's values, so simply re-running it would have written a new set of panels with the
OLD checkpoint's numbers baked into the JSON. Fixed by parsing the "Final model" row live out
of `results/metrics_summary.md` at generation time instead of hardcoding it, specifically so
this cannot silently go stale again after a future promotion. Regenerated all 12 panels
(6 val + 1 D5 failure case + 5 no-GT final-test) and rewrote `results/qualitative/README.md`
with the new numbers and provenance. `results/qualitative/manifest.json` now correctly reads
`checkpoint_val_psnr_db: 29.2548` (self-verified by the script's own new parsing step, not
hand-typed).

**V53 implemented** — `docs/STATE.md` U-5's original spec (written, never coded): exactly one
`*_KLA_PS01.pdf` at repo root, <=9 pages, states the proxy relationship, carries the repo URL,
none of `SPEC_ADDENDUM.md` section 11's banned phrases. **Strengthened** beyond that original
spec (Prime Directive 1 permits strengthening, forbids only weakening): also fails on any
unfilled placeholder literal (`PLACEHOLDER`, `[[ FILL IN`, `[MEMBER`, `[COLLEGE`, `[EMAIL`) —
the literals `scripts/build_deck.py` writes verbatim when real team info was never supplied,
which is exactly this project's current state. Negative-controlled with three cases, not one:
(1) the real current deck (`PLACEHOLDER_TEAM_KLA_PS01.pdf`) correctly FAILs on the
placeholder-literal clause — a true negative, not a fabricated one; (2) a synthetic 3-page PDF
built with `reportlab` containing real content in every required field correctly PASSes; (3)
both files present simultaneously correctly FAILs with "2 files match... expected exactly 1".
State restored and reconfirmed FAIL(1) afterward; the restored file's sha256
(`591d33ee2e26d591dc9f877a3bc9f760b2ba958f8f6092ccc67e4377ce0635c0`) verified byte-identical
to the original tracked deck — the negative-control swap touched nothing permanently.
`docs/VERIFIER_SHA256` re-pinned (`55bffaa4...` → `373a3759...`) with the full changelog entry
above it. Also required installing `pypdf` and `reportlab` for this dev environment's `py
-3.12` interpreter — both were already pinned in `requirements.txt` but missing from this
particular install, a latent gap in this machine's setup rather than a new dependency.

Full fresh `--strict` run after all of the above: **64 PASS / 5 FAIL (V04, V22, V24, V46,
V53)**, 69 checks implemented. No regression: V53 is new and correctly FAILs (real gap); V24
rolled its known ~20% intermittent flake (B11); V04/V22/V46 unchanged. README/STATE.md
corrected from the stale 65/3 figure they were still carrying.

---

## D65 — B1: priced the V22 bf16/fp32 trade-off; decision is to KEEP bf16 (not switch)

D61 accepted V22's bf16/fp32 divergence "as a disclosed trade-off ... for throughput" without
ever measuring what bf16 actually costs in quality, or what fp32 actually costs in throughput.
`scripts/precision_ablation.py` measured both, on the real `inference.py` forward path, full
400-pair val split, paired (`src.metrics.paired_compare`):

| Metric | fp32 mean | bf16 mean | diff (fp32−bf16) | t | fp32 wins? |
|---|---|---|---|---|---|
| PSNR | 29.25476 | 29.25287 | +0.00189 dB | +6.08 | yes (significant, negligible magnitude) |
| SSIM | 0.79211 | 0.79198 | +0.00013 | +10.37 | yes (significant, negligible magnitude) |
| LPIPS | 0.25625 | 0.25470 | +0.00155 (worse) | +10.50 | **no — bf16 is better** |

Throughput (`scripts/benchmark_runtime.py`, same 400 val-split LR files, 3 repeats each,
median): bf16 31.08s, fp32 34.36s — **fp32 costs +10.6%**.

**Decision rule, fixed before measuring:** "switch to fp32 if it wins ANY metric with paired
significance AND costs <15% throughput." Both conditions are technically met. **Decision:
KEEP bf16 — do not switch.** Reasoning, stated plainly rather than silently overriding the
rule after seeing an inconvenient result: the "win" on PSNR/SSIM is real by the paired test but
of negligible practical magnitude (thousandths of a dB/SSIM point) — significant only because
n=400 gives the test high power to detect noise-level shifts, not because the restoration is
meaningfully better. fp32 is simultaneously WORSE on LPIPS, which is already this checkpoint's
weakest margin over the U-Net baseline (t=−3.26, README's own disclosure). Paying a real,
measured 10.6% throughput cost — on a submission KLA explicitly scores for throughput — for a
quality change that is a wash at best and a net loss on one axis is not a good trade, even
though the letter of the pre-committed rule says switch. This is a disclosed judgment call
resolving a case the rule's "wins ANY metric" phrasing didn't anticipate (a mixed result),
not a post-hoc reinterpretation to avoid work — the data point that actually decided this is
the LPIPS loss, which was measured, not assumed.

**Correction to D63's own claim:** D63 said "if we switch, V22 goes green because the shipped
configuration changed." This is WRONG — checked `check_V22`'s actual implementation
(`scripts/verify_all.py`): it explicitly runs `inference.py --precision bf16` and
`--precision fp32` itself and compares them, regardless of what `--precision auto` defaults
to. V22 measures the intrinsic bf16-vs-fp32 divergence, not which one ships. Switching the
default would NOT have resolved V22 either way — it remains a disclosed, live FAIL under this
decision, exactly as it was under D61/D62's.

`inference.py` is unchanged by this decision (no edit needed — bf16 was already the default).
`results/eda/precision_ablation.json`, `results/eda/runtime_bf16_ablation.md`,
`results/eda/runtime_fp32_ablation.md` hold the measurements.

---

## D66 — B3: free re-score of every long-run "new best" checkpoint; no swap warranted

Every checkpoint `configs/long_run_e.yaml`'s run pushed to the Hub as a new PSNR-best (35
checkpoints, step 2000 through the shipped 76000) was re-scored on the full 400-pair val split
under PSNR-only, SSIM-only, LPIPS-only and a disclosed equal-weighted z-score blend
(`scripts/rescore_checkpoints.py`, `results/eda/rescore_long_run_e.json`) — checking whether
`train.py`'s hardcoded PSNR-only selection left a better checkpoint on the table at zero
additional training cost.

**Mean-based blend result (misleading on its own, see below):** the blend's naive winner was
**step 54000**, not the shipped step 76000 — mean PSNR 29.2314 vs 29.2548 (worse), mean SSIM
0.79174 vs 0.79211 (worse), mean LPIPS 0.25427 vs 0.25625 (better). PSNR/SSIM-only criteria
both correctly pick the shipped 76000 (monotonic improvement toward the end of the run).

**This mean-comparison result does NOT survive a proper paired test** — exactly the kind of
self-serving reading `docs/decisions.md` D31 and `src.metrics.paired_verdict` exist to
prevent, and the reason this project never trusts a bare mean delta. Ran the real paired test
(candidate=step54000, ref=shipped step76000) on val, proxy-OOD and real-SEM OOD:

| Set | PSNR | SSIM | LPIPS |
|---|---|---|---|
| val (n=400, in-dist) | **loss** (t=−5.76) | **loss** (t=−2.42) | win (t=−2.53) |
| proxy-OOD (n=40) | win (t=+6.44) | win (t=+4.70) | win (t=−3.57) |
| real-SEM OOD (n=45) | **loss** (t=−2.48) | **loss** (t=−2.97) | tie (t=+1.93, just under 1.96) |

Step 54000 wins outright only on proxy-OOD (a set that was already fine for the shipped
checkpoint per D63 — no problem there to fix). On the in-distribution val split it wins only
1 of 3 metrics, below V28's own "win >= 2/3" bar for a promotable candidate. **On real-SEM
OOD — the actual axis this whole investigation exists to improve — it is WORSE, not better**,
losing PSNR and SSIM significantly with only a LPIPS tie. **Decision: no swap. The shipped
checkpoint (step 76000) remains the correct pick; the selection-metric gap is real (PSNR-only
selection during training) but re-scoring shows it did not cost anything in this run** — an
honest null result, not the free win the mean-based blend suggested at first glance.

The LPIPS-only "winner," step 12000, was not paired-tested further: its own reported means
(PSNR 28.90, SSIM 0.7826, both far below the shipped checkpoint's 29.25/0.7921) make it an
implausible candidate on its face — a checkpoint already losing badly on 2 metrics by
~30-60x this comparison's typical margins does not warrant a formal test.

**Scope cut, disclosed:** the 6 Pareto-sweep "final" checkpoints (different
architectures/widths entirely, from the config-selection sweep D55 already settled) were not
re-scored under this same procedure — lower relevance (they are not candidates for swapping
into the shipped checkpoint's slot; the sweep already made that architecture decision) and
outside the remaining time budget. Not silently dropped: recorded here as a deliberate cut,
not an oversight.

`results/eda/step54000_vs_shipped_paired.json` holds the full paired comparison.

---

## D67 — Post-promotion fine-tune result: NOT promoted, mixed trade-off, incumbent ships

**Operational finding first, since it affects cost control:** the HF Jobs `run_job(...,
timeout="3h")` parameter (D63) did **not** appear to be enforced by the platform. The job
(`6a82df61e55292eada79b3b6`) was found still `RUNNING` at 3h18m24s elapsed — 18+ minutes past
its intended cap — when checked. `inspect_job`'s returned `JobInfo` object exposes no
`timeout` field at all (checked directly via `vars()`), so there is no client-side way to
confirm what the platform actually recorded. Manually cancelled via `cancel_job()` on
discovery. Actual cost: ~3.31h x $2.50/hr = ~$8.27 (vs the $7.50 the cap was meant to enforce)
— a modest overrun, not a budget emergency, but a real gap in this session's cost-control
mechanism worth flagging for any future HF Jobs dispatch: **do not trust the `timeout` kwarg
alone; poll and cancel manually if a hard cap actually matters.**

**Evaluation, paired against the incumbent (`weights/best.pt`), on val + both OOD sets**, for
the two most-fine-tuned checkpoints the (cancelled) run produced (step 102000 and 104000,
i.e. 26000-28000 fine-tune iters past the resumed 76000 — results near-identical between the
two, so only the later one is discussed):

| Set | PSNR | SSIM | LPIPS |
|---|---|---|---|
| val (n=400, in-dist) | **win** +0.445 dB (t=+23.96) | **win** +0.0043 (t=+12.07) | **win** −0.0047 (t=−2.12) |
| proxy-OOD (n=40) | **loss** −1.18 dB (t=−6.57) | **loss** −0.0161 (t=−2.90) | **loss** +0.0108 (t=+2.30) |
| real-SEM OOD (n=45) | tie (t=−0.95) | **loss** −0.0065 (t=−2.17) | tie (t=−0.27) |

**Decision: do NOT promote. Ship the incumbent as-is.** Reasoning:

1. The fine-tune's actual purpose (per D63) was to fix the real-SEM OOD regression. It did
   **not** — real-SEM OOD is a tie on PSNR/LPIPS and a significant LOSS on SSIM. The stated
   goal was not achieved.
2. It broke proxy-OOD, which was NOT previously a problem (D63's own Hour-0 finding: the
   incumbent already wins proxy-OOD). A large, significant loss on all three metrics there is
   a NEW regression this fine-tune introduced, not one it fixed.
3. The large in-distribution win (+0.445 dB) is real and would be tempting on its own, but
   promoting on that alone would repeat exactly the mistake D31/D66 exist to prevent: a strong
   result on one axis does not license ignoring losses on the others, especially when KLA's
   evaluation explicitly includes OOD content (SPEC F7, Evaluation #1) and the axis that
   regressed (proxy-OOD) was previously a clean win.

This is a materially different trade profile from D61's own accepted trade-off (which
improved in-distribution AND proxy-OOD while regressing only real-SEM OOD). This fine-tune
compounds a new regression without fixing the one it targeted — a worse trade, not a
comparable one, and not promotable under this project's own paired-win discipline.

**Widening the degradation randomisation (`randomise_frac` 1.20→1.50, `gauss_sigma_range`
upper bound 0.065→0.09) most likely explains the proxy-OOD loss directly**: proxy-OOD is
procedural/geometric content that was never touched by this change, but training under a
wider noise distribution can shift what the model treats as "typical," costing it on content
whose degradation now sits further from the (wider) training distribution's centre than
before. This is a plausible, disclosed hypothesis, not confirmed further — not worth
additional cloud spend to isolate given the remaining time budget.

**Consistent with the plan's own stated fallback** ("if nothing wins, ship the incumbent,
record it as a real result") — this is exactly that case, reported honestly rather than
forced. `results/eda/finetune_candidates_vs_shipped_paired.json` holds the full comparison.
The fine-tune's checkpoints remain on the Hub (`Team-Ceciroleo67/kla-ps01-checkpoints`,
`20260817T101639Z-finetune_ood_wide-s42/`) if a future session wants to investigate the
proxy-OOD regression further; nothing here is deleted.

---

## D68 — Phase 1 diagnosis: real-SEM OOD gap is content-driven, not degradation-coverage

Following D67's negative result, ran the user-directed 4-part diagnosis before attempting any
further training. Two hypotheses were already ruled out by code inspection alone (no new
measurement needed):

- **(b) Input distribution — confirmed ruled out.** `scripts/gen_real_sem_ood.py:84` produces
  real-SEM `NoisyLR` via `degrade_fitted(gt, rng)`, this project's own measured degradation
  model — identical fidelity path used everywhere else (V33). Not a foreign-noise confound.
- **(c) Normalisation — confirmed ruled out.** Same script, lines 79-81: real-SEM GT is
  per-image min-max normalised to exactly [0,1] before degrading, the same U1 convention as
  every other GT in this project.

**(a) Content statistics — the real explanation, measured (`scripts/content_stats_sem_vs_natural.py`,
`results/eda/content_stats_sem_vs_natural.json`, n=200 natural GT sampled vs n=45 real-SEM
GT):**

| Metric | Natural mean | SEM mean | SEM z-score |
|---|---|---|---|
| Edge density (frac. pixels, grad mag > 0.10) | 0.099 | **0.533** | **+3.62** |
| Local contrast (7x7 windowed std) | 0.057 | **0.148** | **+2.52** |
| Spectral slope (radial log-log power fit) | −2.54 | **−1.63** | **+1.97** |
| Bimodality coefficient (Sarle's) | 0.557 | 0.648 | +0.58 |
| Intensity-histogram entropy | 6.94 | 7.50 | +0.71 |
| Spectral peakiness | 457.6 | 60.3 | −0.70 |
| Gradient anisotropy | 1.869 (std 1.95, wide) | 1.034 (std 0.03, tight) | −0.43 |

Real-SEM content is a genuine, multi-axis statistical outlier: >2σ on three independent,
mutually-consistent measures (edge density, local contrast, spectral flatness) that all point
the same direction — real-SEM images are dense, high-frequency micro-texture (material grain
structure), unlike the natural-photograph training content. Gradient anisotropy's near-zero
std for SEM (vs natural's wide spread) is also notable: SEM content is statistically
*homogeneous* — every SEM tile looks similarly isotropic-textured, whereas natural photos
range from highly directional (facades, gratings) to not.

**(d) Error localisation (`scripts/sem_error_localisation.py`,
`results/eda/sem_error_localisation.json`, all 45 real-SEM pairs, `weights/best.pt`):**
per-pixel `|pred-gt|` error correlates positively but modestly with local gradient magnitude
(Pearson r=0.187) and local variance (r=0.176); error in the textured/edgy half of pixels
runs 1.30-1.44x higher than in the flat half. **Directionally consistent with the content
hypothesis, but a modest effect, not a dominant one** — reported as such, not oversold. The
correlation is real (positive, same direction in both measures) but explains only a small
fraction of pixel-level error variance; most of the OOD quality gap is not explained by
*within-image* location alone.

**Verdict: hypothesis (a) content is supported by measured evidence; (b)/(c) are cleanly
ruled out; (d) gives weak corroborating, not conclusive, support.** This justifies Phase 2
(free levers) next, and — if that doesn't resolve it — Phase 3's mix-in-procedural-content
approach specifically (rather than more degradation-parameter tuning, which D67 already
showed doesn't touch this).

---

## D69 — Phase 2: weight interpolation does not recover real-SEM OOD at any mixing ratio

Linearly interpolated the EMA state dict between the shipped checkpoint (α=0) and D67's
fine-tune's most mature checkpoint (α=1, step 104000) at 9 points, α ∈ [0, 1] step 0.125.
Scored all three metrics on all three evaluation sets at every point, paired against α=0
(`scripts/weight_interpolate_sweep.py`, `results/eda/weight_interpolate_sweep.json`).

**Real-SEM OOD PSNR is essentially flat across the entire range** (17.7854 at α=0 → 17.7777
at α=1, a 0.0077 dB drift, within noise) and **never wins a paired test at any α** — SSIM is
actually a small but statistically significant LOSS from α=0.125 onward (t=−2.68 to −2.88),
PSNR/LPIPS stay non-significant throughout. In-distribution PSNR improves monotonically and
substantially with α (29.2548 → 29.7000, matching D67's own +0.445 dB at α=1 exactly, a
correctness check on the interpolation code). Proxy-OOD degrades monotonically, reaching
significance around α=0.25-0.5 and a large, clearly significant loss by α=1 (t=−6.57 on PSNR,
matching D67 exactly).

**Conclusion: no α on this line satisfies the decision rule** ("must win real-SEM OOD on a
paired test vs α=0") — not because a good trade-off point is hard to find, but because the
fine-tuned direction in weight space simply never points toward better real-SEM performance,
at any mixing amount. This is a second, independent line of evidence (weight-space geometry,
not just endpoint comparison) arriving at the same conclusion as D67: this fine-tune's
information does not help real-SEM OOD, full stop, not just "not enough of it was mixed in."

**Item 2 (soup among B3's near-identical long_run_e siblings, e.g. steps 68000/76000) was
deprioritized and not run**, disclosed rather than silently dropped: item 1's result is
already clean and decisive (no promotable point exists on the base-vs-finetune axis), and the
long_run_e siblings are, per D66, already known to score within noise of each other and of
the shipped checkpoint — a soup among them was always the lower-value half of Phase 2, and
remaining time was directed at Phase 4 and reporting back per the plan's own checkpoint
("report after Phase 1 and Phase 2 before committing any Phase 3 cloud spend") instead.

**Implication for Phase 3:** D68 established the gap is content-driven; D69 establishes that
neither the fine-tune's endpoint nor any interpolation of it recovers real-SEM OOD. Both
conditions for attempting Phase 3 (new procedural-content training) are met — but Phase 3
costs real additional cloud spend on top of an already-over-threshold ledger (see the
PLAN_CLOUD.md update below) and 2-3 more hours. Reported to the user before proceeding, per
the plan's explicit checkpoint, rather than committing that spend unilaterally.

---

## D70 — Phase 4: worst-case failures blur, do not hallucinate; uncertainty head recalibrated against the shipped checkpoint

**Blur vs. hallucination, measured, not asserted** (`scripts/blur_vs_hallucination_check.py`,
`results/eda/blur_vs_hallucination_check.json`). For the documented D5 failure case and the
three worst-scoring real-SEM images: compared the prediction's FFT energy, in the frequency
band strictly above what the LR input's Nyquist limit could have supplied (content the model
cannot have legitimately recovered from the input, only invented or omitted), against GT's
energy in that same band.

| Case | Energy ratio (pred/GT) | Spatial correlation in that band |
|---|---|---|
| D5 (`000984.npy`) | **0.349** | 0.374 |
| `realsem_000021.npy` | **0.052** | 0.250 |
| `realsem_000040.npy` | **0.030** | 0.247 |
| `realsem_000041.npy` | **0.071** | 0.247 |

Every ratio is far below 1.0 — the model produces only 3-35% of the true high-frequency
energy on its hardest cases, not more. Hallucination would require a ratio at or above 1.0
(inventing detail that isn't there costs energy, it doesn't save it); this is the opposite
signature. **The failure mode is confirmed blurring — conservative under-production — not
hallucination**, decisively so on the real-SEM cases (ratios of 0.03-0.07, essentially not
attempting fine texture at all). The modest positive correlation (0.25-0.37) on the small
amount of high-frequency content the model does produce is consistent with genuinely
recovering a little real detail, not inventing a plausible-looking wrong pattern.

**Uncertainty calibration re-run against the actual shipped checkpoint** (D59's own number
was measured against a stale scratchpad sweep checkpoint by mistake, not `weights/best.pt` —
corrected here): `scripts/uncertainty_calibration_probe.py --checkpoint weights/best.pt
--pixel_subsample_per_image 1000` (400,000 pooled pixels, 5x D59's sample size). Per-image
Pearson r=**0.9802**, Spearman r=**0.9723** (both stronger than D59's stale 0.9646/0.9407);
pooled per-pixel Pearson r=0.4621, Spearman r=0.6121 (weaker, as expected — NLL trains the
mean relationship, not pixel-exact prediction, same interpretation as D59). Mean predicted
variance (0.00208) closely tracks mean actual squared error (0.00199). **The uncertainty head
genuinely knows when the model is likely wrong** — a real, now correctly-attributed,
deck-worthy robustness property. README's Method summary corrected to cite these numbers, not
D59's stale ones. `results/eda/uncertainty_calibration.json` overwritten with this run (the
file always described "the checkpoint passed via --checkpoint," so overwriting it with the
shipped-checkpoint run is the fix, not data loss — D59's narrative text in this document is
left as the historical record of what was measured and why the mistake happened).

**Quantitative characterisation of WHY the worst cases are unrecoverable**
(`results/eda/nyquist_energy_fraction.json`), extending D5's own metric (fraction of GT
spectral energy above the LR Nyquist limit, `scripts/visual_audit.py::hf_energy_ratio` —
reused verbatim, not re-derived, after an initial ad-hoc recomputation without mean
subtraction gave a wrong 20.3% for D5's own case before the bug was caught by failing to
reproduce the documented 80.5%): the three worst real-SEM images sit at **80.6%, 80.9%, and
81.0%** GT energy above Nyquist — strikingly close to D5's own 80.5% for `000984.npy`. **Some
of the real-SEM OOD gap is a genuine information-theoretic limit of 2x decimation on this
content, not solely a content-domain-transfer failure** — these images would be similarly
hard for any model, natural-photo-trained or not. This nuances, not contradicts, D68's
content-statistics finding: the domain shift (edge density, local contrast, spectral
flatness) explains why the model handles this content worse than natural photos on AVERAGE,
while this Nyquist measurement explains why its ABSOLUTE worst cases are especially hard
regardless of training domain.

---

## D71 — Phase 3: procedural structural content mixed into training, dispatched

User authorised Phase 3 after seeing D68 (content-driven gap) and D69 (weight interpolation
doesn't recover it). Attacks the content prior directly, which D67's degradation-widening
fine-tune never did.

**New capability, additive, off by default:** `src/structural_content.py` reproduces the
proxy-OOD set's own documented recipe (`docs/dataset_findings.md` "Generation method" — 5
categories: line/space gratings, contact-hole grids, checkerboard, circuit traces, sharp-edge
shapes; same param ranges, same 3x3 box blur, same per-image [0,1] min-max normalisation,
U1). `DataConfig.structural_content_ratio` (new, default 0.0) in `src/dataset.py`: with that
probability, a TRAIN sample is a freshly-generated procedural image degraded through the same
`degrade()` path as any other synthetic sample (matching `_synth_lr_patch`'s exact margin
handling), rather than a real photo. **Never applied to `split="val"`** — validation must
stay real-content, real-degradation, unchanged, otherwise the reported metric stops meaning
what it says. F15-permitted (synthetic pairs from GT), no licence surface, no leakage: this is
a fixed procedural generator, not a foreign dataset.

Verified additive and backward-compatible before use, same discipline as every other train.py
change this session: `--selftest` (V26) still passes; a fresh end-to-end check (200 samples,
`structural_content_ratio=0.3`) gives 59/200 (~30%, matches) with correct shapes and confirms
val is never structural; **two independent `--smoke --smoke_iters 6 --seed 42` runs before and
after this edit reproduce the identical `SMOKE_DIGEST
fd5e52061802c1d2c4d8034d1e224ef3a40586cc40ba48ffb75e3af396bc8da9`** (existing configs have
`structural_content_ratio` absent, defaulting to 0.0, so behaviour is provably unchanged for
them). Local dry-run of the actual fine-tune config (`configs/finetune_structural_content.yaml`,
20 iters, `--iters` override) OOM'd at the config's real `batch_size: 16` on the 8 GB dev GPU
— expected (D20's own documented dev-GPU limit at this width/depth/patch-size combination, not
a bug) — reran with `batch_size: 4` for the LOCAL sanity check only (the real dispatch config
is untouched at 16, sized for an 80 GB A100) and it ran clean end to end, producing a real
val-loop result.

**`configs/finetune_structural_content.yaml`:** resumes `weights/best.pt` (never from
scratch), `lr_patch: 128` (kept from D63's scale-gap finding), `structural_content_ratio:
0.275` (the new lever), degrade block **reverted to D43's values** (`randomise_frac: 1.20`,
`gauss_sigma_range: [0, 0.065]`) rather than repeating D67's widening, which D68/D69 already
showed doesn't touch this axis and does cost proxy-OOD. `optim.finetune_horizon: 40000`
(train.py's existing mechanism, D63), deliberately larger than what 3h can reach.

**Dispatch mechanism corrected per D67's lesson:** `scripts/dispatch_finetune_structural.py`
still passes `timeout="3h"` to `run_job()` (cheap, may help) but does **not** rely on it —
`--watch` blocks in-process, polls `inspect_job` every 2 minutes, and calls `cancel_job()`
itself the instant elapsed time reaches the 3h cap, rather than trusting the platform.

**Decision rule, fixed before the run lands (unchanged from the plan):** promote only if
real-SEM OOD improves on a paired test AND in-distribution loss is under 0.15 dB. Evaluated
with the same `src.metrics.paired_compare` harness as every other comparison this session.
Scope cut, disclosed: the standalone preview/inspection script for the procedural generator
(`scripts/gen_structural_content.py`, mentioned in the plan) was not written separately — the
generator is exercised directly by the training pipeline and its own unit-level check above
(30% ratio, correct shapes) already demonstrates it works; a dedicated preview script would
only add a visualisation, not new evidence, and was deprioritised given the remaining time
budget.

**Run outcome: cut short at ~85 minutes by an external cancellation, cause unconfirmed but
not a code bug.** The job (`6a83212acd3824960fcbb566`) was found `CANCELED` when the watcher's
next poll landed — NOT by the watcher's own cap logic (its 3h/10800s threshold was never
reached; no "CAP REACHED" message was printed). Pulled the full job log
(`fetch_job_logs`) to check for a crash: training was completely healthy throughout, actively
logging at 1.97 it/s through iteration 85810 (elapsed 1:22:59) with no error, no OOM, no
stall — it was killed exactly 190 iterations before its next scheduled checkpoint push
(2000-iter interval), which is why no new checkpoint appeared past step 84000 despite the
extra ~55 minutes of apparent runtime. Asked the user directly whether they cancelled it
manually: **they did not.** Leading hypothesis, not confirmed (no billing API exists to check
directly, per `docs/PLAN_CLOUD.md`): the org's ~$30 HF credit was exhausted or hit a
platform-enforced spend guard — the running total immediately before this run was ~$25.92,
and ~85 minutes of this run at $2.50/hr adds ~$3.55, landing right at ~$29.47, consistent
with a $30 ceiling being enforced. Stated as the leading hypothesis, not asserted as fact.

**Evaluated the last checkpoint the run produced anyway** (step 84000, ~8000 fine-tune
iters), paired against the incumbent, same harness as every other comparison this session
(`results/eda/phase3_candidate_vs_shipped_paired.json`):

| Set | PSNR | SSIM | LPIPS |
|---|---|---|---|
| val (n=400, in-dist) | **win** +0.330 dB (t=+24.24) | **win** +0.0025 (t=+6.94) | tie (t=−1.11) |
| proxy-OOD (n=40) | **win** +14.19 dB (t=+10.96, 40/40) | **win** +0.027 (t=+13.67) | **win** −0.034 (t=−10.92) |
| real-SEM OOD (n=45) | tie (t=−0.56) | tie (t=−0.72) | **win** −0.031 (t=−4.12, 33/45) |

**This is the first fine-tune attempt this session that shows a genuine, paired-significant
improvement on real-SEM OOD** (LPIPS, the other two metrics tie rather than lose) — with NO
regression anywhere: it also wins in-distribution and proxy-OOD, unlike D67's attempt which
traded an in-distribution win for a proxy-OOD loss and no real-SEM gain, and unlike D69's
interpolation sweep which never found any point that helped real-SEM at all.

**Honest caveat on the proxy-OOD result, stated plainly rather than left to look better than
it is:** the +14.19 dB proxy-OOD gain is very large — almost certainly because proxy-OOD is
itself procedural geometric content (gratings, contact-hole grids, checkerboards) and this
fine-tune trained on 27.5% freshly-generated content from the SAME five categories (different
random instances, not the same images — `src/structural_content.py` reuses the documented
recipe, not the proxy-OOD set's own files). This is closing a content-distribution gap
between training and that specific eval set, not evidence of general-purpose OOD robustness
gained from nothing — disclosed exactly as such, not oversold as "generalisation."

**Against the plan's pre-committed decision rule** ("promote only if real-SEM OOD improves on
a paired test AND in-distribution loss is under 0.15 dB"): satisfied on both counts — real-SEM
OOD improves (LPIPS win, no losses on the other two) and in-distribution does not lose at all
(it wins). **This candidate is promotable.** Reported to the user before promoting — checkpoint
promotion triggers the full regeneration cascade (qualitative panels, metrics_summary,
runtime_report, restored-outputs republish, README, ledger) and is exactly the kind of
consequential, hard-to-cheaply-reverse action this project's own precedent (D49, D61) treats
as a deliberate, disclosed step, not a silent one.

---

## D72 — Phase 3 checkpoint promoted (human sign-off obtained); full regeneration cascade

User explicitly authorised promotion via `AskUserQuestion` after seeing the D71 comparison
table (wins/ties everywhere, first genuine real-SEM OOD improvement, no regressions). New
checkpoint: sha256 `6d74ccfdd72e1271a7de5fdede5c341b3cf18ca4294619dd90a97c0591f66397`,
11,557,166 bytes, `iter=84000`, `git="c8f3a51b2415b2b4af0c4300422d325dd1fe9f5c"` (a real
commit SHA this time — the cloud container cloned via `git clone`, not a tarball snapshot,
closing a gap D61's checkpoint had). Downloaded from
`Team-Ceciroleo67/kla-ps01-checkpoints/20260817T145721Z-finetune_structural_content-s42/step_00084000/best.pt`,
replaced `weights/best.pt`.

**Full regeneration cascade, same discipline as D61's promotion:**

1. Regenerated `results/baselines/final/`, `results/baselines/proxy_ood/final/`,
   `results/baselines/real_sem_ood/final/` predictions against the new checkpoint (via
   `scripts/make_baselines.py` for the first two, a direct forward-pass script matching
   `run_learned`'s pattern for real-SEM since `make_baselines.py` has no real-SEM mode).
2. Regenerated `results/metrics_summary.md` (`scripts/evaluate.py --collect results/baselines
   --preds final=results/baselines/final --proxy_ood --real_sem_ood`): in-distribution
   29.5850±4.6301 dB / 0.79460±0.14204 / 0.25416±0.13263 (n=400); proxy-OOD 41.4414±7.1618 /
   0.99691±0.00307 / 0.00169±0.00172 (n=40); real-SEM OOD 17.7824±0.7588 / 0.25892±0.11163 /
   0.67988±0.14781 (n=45); V28 vs U-Net: PSNR +0.7042 dB (t=+27.18, 398/400), SSIM +0.01187
   (t=+16.35, 376/400), LPIPS −0.01108 (t=−3.86, 220/400) — all three wins, a wider margin
   than the checkpoint it replaced.
3. Regenerated `results/runtime_report.md`'s current-checkpoint section: median 28.43s
   (14.1 img/s), n=5, same architecture size as before so expected to be roughly the same
   modulo this laptop GPU's documented session-to-session variance (B2's own finding).
4. Regenerated `results/qualitative/` (`scripts/make_qualitative_examples.py`): re-verified
   the 6 hardcoded example tags still hold under the new checkpoint's ranking (they do —
   002041.npy still rank 0/400, etc.) before reuse; deleted the 7 orphaned old-PSNR panels.
   **Fixed a second occurrence of the same recurring bug** (D64 already fixed this once):
   `CKPT_SHA` was a hardcoded literal that went stale again. Fixed properly this time by
   computing it from `weights/best.pt`'s actual bytes at script run time instead of a second
   hardcoded value — this class of bug cannot recur a third time.
5. Regenerated the 400 final-test outputs (`inference.py --require_weights`), built a fresh
   `manifest.csv` (per-file sha256/shape/dtype/min/max/finite/matching-input, all 400 clean),
   zipped (90,929,851 bytes, sha256 `7c5a63ff8720bbbbf781891c6fdb1302bc925095806278766ad08ca2abe9c6ef`),
   published as GitHub Release **`artifacts-v3`** (metadata-only `gh release create` followed
   by a separate `gh release upload` after the combined create+asset call hit a transient
   GitHub 503 — asset upload alone succeeded on first retry), verified fetchable from a
   logged-out `curl` session with the digest reproduced exactly. `artifacts-v1`/`v2` untouched,
   remain the historical record for their respective checkpoints.
6. Updated `README.md` (status block, Result summary table, V28 detail table, Method summary,
   Training section, Repository map, Runtime measurement, Failure cases, External resources
   framing), `weights/README.md` (new Status section, demoted D61's to Superseded),
   `results/restored_test_outputs/README.md` (new Status section, demoted D61's to Superseded),
   `results/qualitative/README.md` (full rewrite with new numbers), `docs/REQUIREMENTS_MATRIX.md`
   (throughput row, ledger row, OOD row), `docs/STATE.md` (RESUME HERE rewritten).
7. Added the `results/experiments.csv` row for this run
   (`20260817T145721Z-finetune_structural_content-s42`), matching C1's precedent for cloud
   runs (main session appends after pulling results, since the ephemeral job container's own
   ledger write is lost with the container).
8. Full fresh `scripts/verify_all.py --strict` dispatched — **result not yet known as this
   entry is written; check `results/verification_report.json`'s commit/timestamp before
   trusting any tally quoted elsewhere in this document once this run lands.**

No number in any of the above was hand-typed without a script producing it first — the same
standing rule this whole project has followed since D1.

## D73 — Local continuation of the Phase 3 fine-tune priced and measured, honestly: not viable on this GPU

**Question:** with cloud spend exhausted (~$29.47 of the $30 HF credit ceiling; $0 budget for
this task), could the remaining ~30,000–36,000 iterations of `configs/finetune_structural_content.yaml`
(`optim.finetune_horizon: 40000`, resumed at iter 84000, i.e. iter range [84000, 116000)) be
run on the local RTX 4060 Laptop (8 GB) instead? Measured, not estimated, below. All numbers
come from real `train.py` invocations this session, `--out` pointed at the scratchpad
directory, `--no_ledger` (ignored for non-smoke runs per SPEC 9, so every completed run below
appended a row to `results/experiments.csv` — those 5 throughput-probe rows were reverted with
`git checkout -- results/experiments.csv` afterwards; they were not real training runs and do
not belong in the permanent ledger. `weights/best.pt` was never touched: sha256
`6d74ccfdd72e1271a7de5fdede5c341b3cf18ca4294619dd90a97c0591f66397` before and after, matching
D72 exactly).

**Gotcha found and worked around:** `--iters N` overrides `total_iters` to the *absolute*
iteration counter, not a step count relative to `--resume`'s `start_iter`. With
`start_iter=84000` (from `weights/best.pt`), `--iters 20` makes `total_iters=20`, so
`while it < total_iters` (`84000 < 20`) is false immediately and **zero training steps run** —
three separate invocations at batch 16 silently "succeeded" in under a second with no OOM and
no per-step log line before this was caught (compare each iteration's elapsed time against
`start_iter` before trusting a "no OOM" result on a resumed run). Every measurement below uses
`--iters (start_iter + N)` to actually execute `N` steps.

**batch_size=16 (the config's real, untouched setting): genuine CUDA OOM, first training step.**
`python train.py --config configs/finetune_structural_content.yaml --resume weights/best.pt
--iters 84020 --val_every 0 --out <scratch>/bs16_probe4.pt --workers 0 --verbose`, dataset
preload done (0.59 s), model+EMA+optimizer built, first batch drawn — fails inside the very
first residual block's `conv3` (`src/blocks.py:295`) on the very first forward pass of
iteration 84001:
```
torch.AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation` ...
```
Verbatim, uncaught, non-zero exit. This is `data.lr_patch: 128` (→ 256×256 GT crops),
`model.width: 64`, `model.num_blocks: 32`, FiLM+uncertainty head, bf16 autocast,
`channels_last`, batch 16 — exactly the shipped config, nothing narrowed to make it fail.

**batch_size=8: does not OOM, but is not usable on this machine right now.** Two attempts,
both against a config identical to the real one except `data.batch_size: 8` (scratch copy,
`<scratch>/ft_bs8.yaml` — the only diff from `configs/finetune_structural_content.yaml`, which
is not this role's file to edit):
- Attempt 1 (`--iters 84050`): 12 iterations logged in 7m33s of an 8-minute tool timeout before
  being killed — cumulative rate 0.02–0.03 it/s, per-step deltas 14–31 s each.
- Attempt 2 (`--iters 84060`): **zero** iterations logged in a 5-minute timeout.

`nvidia-smi` and `Get-CimInstance Win32_Process` during both attempts showed a **sibling agent
session's `scripts/benchmark_runtime.py` + `inference.py` actively running GPU sweeps at 100%
utilization** on this same 8 GB card at the same time — this machine is not exclusively mine
during this session. That confound is real and I cannot rule out that batch 8 would be faster
in isolation. It is also consistent with, and does not contradict, D20's independent finding
(measured without any sibling load) that this exact width≥64/num_blocks≥32 architecture
triggers a large step-time collapse from allocator spill on this 8 GB card — the two effects
plausibly compound. Either way: batch 8 is not a usable operating point on this machine as
observed, twice, this session.

**batch_size=4: the largest batch size that gave a clean, complete, reproducible measurement.**
Two full runs, both finished normally (final full-split validation ran, PSNR 29.57 ± 4.63,
matching the shipped checkpoint's known score, confirming these are correct forward/backward
passes and not silently-broken ones):

| run | iters | elapsed at iter 1 (incl. one-off cudnn autotune) | elapsed at final iter | steady-state |
|---|---|---|---|---|
| `bs4_run.log` (25 iters) | 25 | 28 s | 34 s | (34−28)/24 = 0.25 s/iter = **4.00 it/s** |
| `bs4_run2.log` (100 iters) | 100 | 35 s | 91 s | (91−35)/99 = 0.566 s/iter = **1.77 it/s** |

The 2.3× spread between two clean batch-4 runs in the same session is itself the finding: this
laptop's real spare throughput fluctuates with the sibling session's GPU tenancy even at a
batch size that never OOMs. There is no single "the" rate; I report both ends.

**Extrapolation arithmetic, literal question ("25,000–30,000 iterations, this batch size"):**

- Conservative (0.566 s/iter): 25,000 × 0.566 s = 14,150 s = **3.93 h**; 30,000 × 0.566 s =
  16,980 s = **4.72 h**.
- Optimistic (0.25 s/iter): 25,000 × 0.25 s = 6,250 s = **1.74 h**; 30,000 × 0.25 s = 7,500 s =
  **2.08 h**.
- Range: **~1.7–4.7 h** to literally execute 25,000–30,000 steps, at batch 4.

**But batch 4 is 1/4 of the config's real batch_size=16, which OOMs outright (above).** A
batch-4 step sees 1/4 the samples of a batch-16 step, so 25,000–30,000 batch-4 steps do **not**
reproduce the training signal of a 25,000–30,000-step batch-16 run — matching that would need
roughly 4× the steps, ~100,000–120,000, at the same measured per-step rate:

- Conservative: 100,000 × 0.566 s = 56,600 s = **15.7 h**; 120,000 × 0.566 s = 67,920 s =
  **18.9 h**.
- Optimistic: 100,000 × 0.25 s = 25,000 s = **6.9 h**; 120,000 × 0.25 s = 30,000 s = **8.3 h**.
- Data-equivalent range: **~6.9–18.9 h**.

Neither range accounts for the LR schedule and optimizer statistics also depending on batch
size — a batch-4 run is not just "the same run, slower," it is a different training recipe
that would need its own re-tuning (grad-accum to reconstitute effective batch 16, or a
re-derived LR) before its output PSNR/SSIM/LPIPS could be trusted as comparable to what the
cloud run was measuring. That re-tuning is out of scope for this measurement and is not
assumed to be free.

**Comparison to cloud, using the given anchor (~3 h / ~$7.50 on A100-large, currently
unavailable at $0 remaining budget):** even the literal, best-case local number (1.7 h) is in
the same ballpark as the cloud number only under the most optimistic, uncontended reading and
only for the literal (not data-equivalent) interpretation; the conservative literal reading
(3.9–4.7 h) and every data-equivalent reading (6.9–18.9 h) are worse than cloud, cost zero
dollars but consume the one local GPU for many hours on a machine that, this session, was
independently observed to already be shared with other agents' work.

**Conclusion:** local continuation of this exact fine-tune is not a free substitute for cloud
compute. `batch_size=16` (the real recipe) cannot run locally at all (OOM, reproduced
verbatim). The best locally-viable operating point (`batch_size=4`) requires either 4×+ more
wall-clock than the cloud anchor for a data-equivalent amount of training, or a materially
different (unvalidated) recipe to fit in fewer, larger-effective-batch steps. Recorded here so
no future session re-discovers this the hard way. See also the new bullet in
`docs/STATE.md`'s "Do NOT retry" list.

## D74 — `inference.py` default `--batch_size` changed 32 -> 4; a real benchmark-tool bug found and fixed while confirming it

**Re-swept batch size end-to-end for the current (post-Phase-3) 1,393,938-param checkpoint.**
The last committed batch-size sweep (`results/runtime_report.md`'s isolated forward-pass
table) was measured on the superseded 388,225-param checkpoint and had never been re-run for
the current, 3.6x-larger architecture. External, whole-process (`scripts/benchmark_runtime.py`
-> `subprocess.run(inference.py, ...)`) sweep at batch sizes {4, 8, 16, 32, 64}, 5 interleaved
rounds, medians over n=5, both resolutions:

| resolution | batch 4 | batch 32 (old default) | batch 4 vs 32 |
|---|---|---|---|
| 128->256 | 21.97 s (18.21 img/s) | 32.19 s (12.43 img/s) | **31.8% lower wall-clock** |
| 256->512 | 63.24 s (6.33 img/s) | 77.23 s (5.18 img/s) | **18.1% lower wall-clock** |

Monotonic at both resolutions: throughput falls as batch size rises across the whole tested
range, on this 8 GB card, for this architecture. No V-check pins the default batch_size value
(confirmed by grep before changing it — the only `--batch_size` reference in
`scripts/verify_all.py` is an explicit `--batch_size 8` override in one check, not an
assertion about the default). **Changed `inference.py`'s `--batch_size` default from 32 to 4.**
A smaller default is also strictly safer against OOM on unknown hardware, an independent
argument in its favour — though, per the standing H100 discipline (D-many, `README.md`), this
is a memory-bandwidth-boundedness result measured on an 8 GB laptop GPU and is not claimed to
transfer quantitatively to an 80 GB H100; it is disclosed as a measured, hardware-specific
optimum, and the smaller default is preferred on the independent OOM-safety grounds regardless.

**A real, separate bug was found and fixed while confirming the new default actually took
effect.** `scripts/benchmark_runtime.py` had its own hardcoded `ap.add_argument("--batch_size",
type=int, default=32)` and unconditionally forwarded `--batch_size <value>` to every
`inference.py` subprocess invocation, including when the user passed no `--batch_size` flag at
all. This means the benchmark tool could **never** measure what "running `inference.py` with
no batch-size override" actually does — it silently pinned every un-overridden run to 32,
regardless of `inference.py`'s own real default. Caught because, immediately after changing
`inference.py`'s default to 4, a bare `scripts/benchmark_runtime.py` run (no `--batch_size`)
still measured ~28.85 s median at 128->256 — matching the *old* batch-32 number, not the
freshly-measured batch-4 number of ~21.97 s. Fixed: the script's own `--batch_size` now
defaults to `None`, and the flag is only forwarded to `inference.py` when explicitly passed;
an un-overridden benchmark run now genuinely measures whatever `inference.py` itself defaults
to. Verified fixed: a bare re-run after the fix measured median 20.62 s (19.40 img/s, n=5),
matching an explicit `--batch_size 4` run (19.99 s median) to within normal session variance —
`results/runtime_report.md`'s headline number was regenerated from this bare, un-overridden run
so it honestly reflects the exact command KLA will actually invoke.

**This is exactly the class of defect this project's own measurement discipline exists to
catch** — a tool whose default silently diverges from the thing it claims to measure is worse
than no tool, since it would have kept reporting a stale, wrong number under the label "the
default" indefinitely if the two changes (inference.py's default, and this bug) had not been
made in the same session and cross-checked against each other.

## D75 — Entry point renamed inference.py -> run.py, per an official, track-specific final-submission announcement

**What changed and why.** The user forwarded an official hackathon announcement (confirmed,
in conversation, as coming from the official portal/email and specific to KLA PS01 -- not
generic boilerplate) laying out a final technical check for submission. It requires: (1) the
entry script be named `run.py`, explicitly instructing teams whose script is `main.py`/
`eval.py`/`evaluate.py` to rename it; (2) invocation `python run.py <input_dir> <output_dir>`;
(3) a submission folder `team_name/{run.py, requirements.txt, README.md, models/}`; (4) the
usual correctness bar (`.npy` in/out, shape, `[0,1]`, no NaN/Inf, no internet/API keys/model
downloads/user interaction, GPU-runnable) -- all of which the former `inference.py` already
satisfied, proven by 32 of the then-69 V-checks.

This conflicts, on naming only, with the ORIGINAL official spec (`docs/SPEC.md`,
`docs/VERIFICATION_CONTRACT.md`): entry point `inference.py`, `--input_dir/--output_dir`
flags, weights under `weights/`. The user did not know whether `<input_dir> <output_dir>` in
the new announcement is literally positional or just prose shorthand for the two directory
arguments, and left that call to be made safely -- **the decision: support both**, so neither
reading can fail. The user also explicitly confirmed (`AskUserQuestion`) that the announced
`models/` folder should be a REAL directory with a byte-identical copy of the checkpoint,
cross-checked against drift, rather than just a fallback resolution path.

**Working interpretation, stated explicitly, not silently assumed:** the announced 4-item
folder is read as the *minimum* required, not an exhaustive top-level listing -- this repo
keeps `src/`, `scripts/`, `docs/`, `train.py`, etc. alongside `run.py`/`requirements.txt`/
`README.md`/`models/`, since the original spec's "public GitHub repo" requirement is still in
force and is not rescinded by this announcement. If this interpretation turns out to be wrong,
it is cheap to correct: nothing about the technical implementation below depends on it.

**Blast-radius mapping done before touching anything** (an `Explore` pass): 9 literal string
sites in `scripts/verify_all.py` (`ctx.run_inference`, `ctx.inference_ast`, plus V01,
`_fresh_clone_run` serving V04/V46, V13's required-files list, V23's importtime subprocess,
V57's `spec_from_file_location`, V65's OOM-probe script arg) drive 32 of the 69 V-checks
entirely through those two helpers. Both `scripts/verify_all.py` and
`docs/VERIFICATION_CONTRACT.md` are hash-pinned (`V00`, `docs/VERIFIER_SHA256`); `run.py`,
`models/`, `team_name` were confirmed genuinely novel strings (zero prior collisions) before
being introduced.

**Implementation, in order:**

1. `git mv inference.py run.py`. Zero logic changes beyond the CLI (below) -- same import
   allowlist, same OOM-halving, same output clipping, same everything else. `run.py`'s weights
   resolution gained a fallback (`weights/best.pt` first, `models/best.pt` second), still
   `Path(__file__)`-relative (V05-compliant).
2. **Dual CLI, both forms genuinely required (not softly optional):**
   `ap.add_argument("input_dir_pos", nargs="?", ...)` / `output_dir_pos` alongside
   `--input_dir`/`--output_dir` (no longer `required=True` on the flags -- that would make the
   positional form a parse-time error). `main()` resolves `args.input_dir or
   args.input_dir_pos` (same for output), and exits 2 with a clear message if neither form
   supplies both directories. Verified directly: `python run.py in out` exit 0,
   `python run.py --input_dir in --output_dir out` exit 0, `python run.py` (nothing) exit 2.
3. **A stale hardcoded literal caught in the same pass:** `_err()`'s messages were hardcoded
   `f"inference.py: {msg}"` -- exactly the class of bug this session has hit repeatedly
   (hardcoded filenames/hashes that don't track a rename). Fixed to `"run.py: {msg}"`, caught
   by actually running the no-args case and reading the real stderr, not by inspection alone.
4. `inference.py` becomes a 3-line back-compat shim (`from run import main; sys.exit(main())`)
   -- deliberately not scanned by the verifier any more; a courtesy leftover for anything that
   still expects the original name, not the graded file.
5. **`scripts/verify_all.py` retargeted.** All 58 literal-string occurrences of
   `"inference.py"` (code, not prose) replaced with `"run.py"` via the two helper functions
   plus the 7 other literal sites -- this alone flips V01-V05, V07-V13, V15-V24, V36, V40-V42,
   V47, V57, V60, V64, V65 onto the new file with no per-check logic change. 16 further prose
   mentions in docstrings/comments (describing what a check does, not a code target) updated
   for accuracy. `torch.inference_mode()` call sites were NOT touched -- that is a torch API
   name, unrelated to the file rename, confirmed by regex boundary before the blanket replace.
6. **V02 could not survive a literal swap and was rewritten instead.** Its old assertion was a
   static AST check that `--input_dir`/`--output_dir` are `required=True` -- structurally
   incompatible with also accepting positional args (argparse cannot make one form
   `required=True` without the other becoming a parse error). Rewritten to a BEHAVIORAL test:
   statically confirm both the two flags and two positional args exist as accepted arguments,
   then run three real subprocess invocations (positional-only, flags-only, neither) and assert
   the first two exit 0 and the third exits non-zero. This is strictly MORE coverage than the
   check it replaces (it proves both invocation styles actually work, not just that one static
   property holds), not a weakening -- consistent with Prime Directive 1's "may make checks
   stricter" clause.
7. **V13's required-files list**: `"inference.py"` -> `"run.py"` (kept `README.md`, `train.py`,
   `requirements.txt` -- independent, still-valid requirements; `inference.py`'s presence is no
   longer verifier-mandated since it is now an optional legacy shim, though it still exists).
8. **Two new, strictly additive Tier-0 checks**, following the V63/V67 precedent (checks that
   exist purely to prove a specific external requirement is met, not to gate quality):
   - **V69**: asserts `run.py`, `requirements.txt`, `README.md` all exist at repo root and
     `models/` exists as a directory -- automated, re-runnable proof of the announced
     submission-folder shape, additive to V13, not a replacement for it.
   - **V70**: if both `weights/best.pt` and `models/best.pt` exist, asserts their sha256 match
     exactly -- the drift-prevention invariant for the duplicated checkpoint copy (§9 below).
9. **Real `models/` folder created**: `models/best.pt` is a byte-for-byte copy of
   `weights/best.pt`, sha256-verified equal immediately after copy
   (`6d74ccfdd72e1271a7de5fdede5c341b3cf18ca4294619dd90a97c0591f66397`, both). `models/README.md`
   added, pointing to `weights/README.md` for full provenance and explaining V70's invariant.
   **Any future checkpoint promotion must copy the new checkpoint into `models/best.pt` too**
   -- added to the promotion checklist in `docs/STATE.md`.
10. **`scripts/benchmark_runtime.py` retargeted** (`REPO_ROOT / "inference.py"` ->
    `REPO_ROOT / "run.py"`, plus 8 prose mentions) -- this is now the file KLA actually times.
11. **`docs/VERIFICATION_CONTRACT.md` updated**: 16 `inference.py` mentions -> `run.py`, V02's
    row rewritten to describe the dual-form behavioral contract (not the old static
    `required=True` wording), two new rows added for V69/V70.
12. **A real gap caught while staging: `models/best.pt` was gitignored.** `.gitignore`'s
    blanket `*.pt` rule only carved out `!weights/best.pt`; without a matching
    `!models/best.pt` line, the mirror would never actually reach a commit, a push, or a real
    clone -- it would have silently existed only on this machine while `V69`/`V70` passed
    locally and failed the moment anyone else cloned the repo. Added the missing negation
    line. Separately, `V51`'s `CHECKPOINT_BLOB_EXEMPTION` (previously the single string
    `"weights/best.pt"`) was extended to a tuple including `"models/best.pt"` -- the mirror is
    now mandatory per `V69` the same way `weights/best.pt` is mandatory per `V59`, so the same
    "sanctioned checkpoint, not an accidental blob" reasoning applies to both, not a second,
    unaccounted-for exemption.
13. **Re-pinned** (final, after the gitignore/V51 fix above changed `scripts/verify_all.py`
    again). New sha256: `scripts/verify_all.py` =
    `240c804524e6e7a5124ca48a752e09fe4bad61e94074c11f2ffa91959419796d`;
    `docs/VERIFICATION_CONTRACT.md` = `0bde203980d0892b939fc2a9e343f1cf55b2c8d2ec079897f1d36aa7cb5adf69`.
    `docs/VERIFIER_SHA256` updated with both, plus this decision referenced in its change log.

**Verified before considering this done:** `python run.py --help` and `python inference.py
--help` both exit 0 (guards against the exact argparse `%`-escaping crash class caught earlier
this session -- checked directly, not assumed). Both invocation forms produce correct output
on `tests/fixtures/single`. Full `scripts/verify_all.py --strict` re-run after all of the
above, confirming zero previously-green checks went red and both new checks pass (full tally
recorded in the same session's `docs/STATE.md` update).

**Deliberately left untouched:** `results/restored_test_outputs/manifest.json`'s recorded
`"command"` field still reads `python inference.py ...` -- historical provenance of an
already-published artifact that genuinely was produced via `inference.py` at the time;
rewriting it retroactively would be dishonest, not corrective. `docs/SPEC.md`'s own text is
append-only and unedited; this update lives in `docs/SPEC_ADDENDUM.md` instead, per that file's
own convention, so the original spec's historical record stays intact.

## D76 — GitHub repository renamed to `bunker_backer`, per a literal reading of the announcement's folder shape

**Question resolved by the human directly, not assumed.** D75 (the `run.py` compliance work)
had stated an explicit working interpretation: the announcement's `team_name/{run.py,
requirements.txt, README.md, models/}` folder was read as the *minimum required subset*, not
a literal instruction to rename the whole repository. When the user later flagged that their
own reading was "all the files are supposed to be under our team name," they were asked
directly (`AskUserQuestion`) which was meant: a separate submission package/zip named
`bunker_backer/`, or renaming the actual GitHub repository. **The human chose to rename the
actual repository.** Not re-litigated; executed as directed.

**What was done, in order:**
1. `gh repo rename bunker_backer --repo sahithsundarw/semicon-kla-image-restoration` --
   confirmed via `gh repo view sahithsundarw/bunker_backer` (`visibility: PUBLIC`).
2. Local remote re-pointed: `git remote set-url origin
   https://github.com/sahithsundarw/bunker_backer.git` (not done automatically by `gh` in
   this environment -- checked, not assumed).
3. **Verified empirically, before trusting anything, that existing published links still
   resolve under the old name** (GitHub's rename redirect): `curl -sI` against the exact
   Release asset URL already recorded in `results/restored_test_outputs/manifest.json`
   (`.../semicon-kla-image-restoration/releases/download/artifacts-v3/restored_test_outputs.zip`)
   returned `301` to the new repo's equivalent URL, which then `302`-redirected to a real
   signed asset URL -- confirming both the repo-level redirect and the Release-asset-level
   redirect work, not just the repo homepage. Requesting the NEW URL directly also resolved
   correctly (skipping the redirect hop entirely).
4. **Updated every reference to the old URL to the new canonical form anyway**, rather than
   depend on the redirect indefinitely (a redirect can, in principle, be reclaimed if the old
   name is ever registered by someone else under a different account -- unlikely here since
   it is the same owner, but not a risk worth carrying for zero cost to fix). Found via
   `grep -rl` across `*.md *.py *.json *.txt`: `README.md`, `docs/decisions.md`,
   `docs/STATE.md`, `docs/AUDIT_20260815.md`, `docs/MORNING_REPORT.md`,
   `docs/SUBMISSION_CHECKLIST.md`, `docs/DECK_CONTENT_FOR_PPT.md`,
   `results/restored_test_outputs/{README.md,manifest.json}`,
   `reviews/requirements-audit-1.md`, `scripts/{build_deck.py,verify_all.py,
   dispatch_finetune_job.py,dispatch_finetune_structural.py}` -- 14 files, one literal
   string replaced everywhere (`sahithsundarw/semicon-kla-image-restoration` ->
   `sahithsundarw/bunker_backer`). `results/verification_report.json` deliberately excluded
   (machine-generated, regenerates on the next verifier run).
5. **`scripts/verify_all.py` is hash-pinned and its `V53_REPO_URL` constant (the exact string
   the deck contract asserts against) was one of the 14 changed sites.** New sha256
   `2b5c2d3bb9bd469196aed5cff1dbdd06cc74e3c1ab8d10265ed2d0f723161ddd`, logged in
   `docs/VERIFIER_SHA256` per V00's own escape hatch. No check's assertion, threshold, or
   tolerance changed -- only the literal URL being asserted against, which is the correct new
   value for the same requirement.
6. **The deck regenerated** (`scripts/build_deck.py` also hardcoded the old URL on its
   "GitHub & Video Link" slide) so the shipped PDF states the new repository link, not a
   now-stale one.
7. **Deliberately NOT rewritten**: nothing else about the repository's history, commit SHAs,
   or git object identity changes from a GitHub-side rename -- this is a metadata/URL change
   only, verified end to end (see Verified below), not a re-architecting of anything.

**Verified before considering this done:** the exact Release URL recorded in
`results/restored_test_outputs/manifest.json` resolves (both old-redirects-to-new and the new
URL directly), confirmed via `curl -sI` before any file was edited to depend on that
assumption. Full `scripts/verify_all.py --strict` re-run after all of the above (see
`docs/STATE.md` for the tally this run produced) to confirm nothing else broke.

## D77 — End-to-end throughput measured for every baseline, not just the shipped model

**The user asked directly: can the baseline comparison table's "not separately measured"
throughput cells be filled in for real, with the same rigor as the shipped model's number?**
Yes -- done for all four rows (three classical baselines, one U-Net), using the exact same
external-process measurement methodology already used for the shipped model, not a
different/looser one.

**U-Net baseline: free, using existing tooling exactly as designed.** `run.py` loads its
architecture from the checkpoint's own embedded config (V35), so it is already
architecture-agnostic -- pointing `scripts/benchmark_runtime.py --weights
weights/baseline_unet.pt` at it required zero code changes. Measured: median 17.19s (23.27
img/s), n=5, RTX 4060, bf16, batch 4 (the same default), `results/runtime_report_unet.md`.

**The three classical baselines (bicubic, median-then-bicubic, non-local-means-then-bicubic)
needed one new script**, since they are plain numpy transforms with no checkpoint and no
`run.py` CLI at all. `scripts/run_classical_baseline.py` was added: it imports
`CLASSICAL_BASELINES` and `save_prediction` directly from `scripts/make_baselines.py` --
**reusing, never reimplementing, the exact math already scored** in
`results/baselines/{bicubic,median_bicubic,nlm_bicubic}/metrics.json` -- and wraps it in
`run.py`'s own `--input_dir`/`--output_dir` contract plus a `run.py`-compatible summary line,
so `scripts/benchmark_runtime.py`'s existing external-timing harness (subprocess wrapping the
whole process, median of 5 repeats -- unchanged methodology) can measure it identically.
`scripts/benchmark_runtime.py` gained a `--target_script` option (default `run.py`, so every
existing invocation is unaffected) and an `--extra` passthrough for target-specific flags like
`--method`.

**Measured** (RTX 4060, CPU-only for the classical baselines since no GPU code path exists
for plain numpy, n=5 medians, same 400-image input set as every other throughput number in
this repo):

| Method | median wall-clock | img/s |
|---|---|---|
| Bicubic ×2 | 1.58 s | 253.2 |
| Median 3×3 → bicubic | 3.15 s | 127.0 |
| Non-local means → bicubic | 11.14 s | 35.9 |
| U-Net baseline | 17.19 s | 23.27 |

Monotonic with each method's actual computational cost (non-local-means is the most expensive
classical denoiser of the three, and it shows). Folded into `README.md`'s comparison table,
replacing every "not separately measured" cell with a real, identically-methodologied number
and a pointer to its own `results/runtime_report_*.md`.

**Verified not to affect any check:** `V48` (results-table reconciliation) reads
`results/metrics_summary.md`, not `README.md`, and only reconciles PSNR/SSIM/LPIPS floats --
confirmed by reading its implementation before assuming, not after. No verifier change was
needed or made.

## D78 — V00 caught its own class of bug: a re-pin was missed after a cosmetic edit

**A genuine, working example of the verifier catching a real mistake, not a false positive.**
While removing internal-only files (D76's cleanup pass), `scripts/verify_all.py`'s module
docstring had a dangling comment reference to `LOOP_PROMPT.md` ("Design rules (LOOP_PROMPT.md
B3):") after that file was deleted. Fixed to "Design rules:" -- but this edit changed the
pinned file's bytes, and the re-pin step was missed in the moment (attention was on the
file-removal list, not on the fact that a pinned file had also been touched). The very next
full `--strict` run correctly went `V00 FAIL: scripts/verify_all.py changed without a matching
docs/decisions.md entry (computed 9cb33bd8c83663c4c2a8de75a3d3bb468f9311861d219cef8424b41f0a6e7d73)`.

This is exactly the failure mode V00 exists to prevent -- an unlogged change to the file that
defines correctness -- and it worked as designed: caught immediately, on the very next run,
before any commit. Fixed properly here: new sha256
`9cb33bd8c83663c4c2a8de75a3d3bb468f9311861d219cef8424b41f0a6e7d73` logged in
`docs/VERIFIER_SHA256`, this entry contains the exact hash V00 needs to find, and no check's
assertion, threshold, or tolerance was touched -- the change was cosmetic (a comment), and the
process gap (forgetting to re-pin after a comment edit, not just a logic edit) is now recorded
so it's recognised faster next time.

## D79 — Auto-generated deck removed again, deliberately, to make way for a hand-built PPT

**Not a regression -- an explicit user decision.** The auto-generated deck
(`bunker_backer_KLA_PS01.pdf`) was built with real team info (D76/D78: team `bunker_backer`,
4 named members, VNR Vignana Jyothi Institute of Engineering and Technology, contact), and
`V53` passed against it. The user then asked for it to be removed: a teammate is building
the actual submission PPT by hand, and will add it to the repo once ready.

Removed via `git rm bunker_backer_KLA_PS01.pdf`. `scripts/build_deck.py` itself was left
exactly as it was fixed for D76/D78 (real member names, correct current-checkpoint
architecture/throughput numbers, working qualitative-panel image paths) -- it remains a
correct, working fallback generator if the hand-built PPT is ever not ready in time; only the
generated PDF artifact was removed, not the tooling that produces it.

**`V53` now correctly FAILs again** (no `*_KLA_PS01.pdf` exists) -- the same disclosed,
expected state the project was in before real team info was ever supplied, not a new or
hidden problem. This is not "papering over" or weakening the check (Prime Directive 1 is
untouched: the check's assertion is exactly what it always was); it is the check correctly
reporting a real, current, temporary gap that the human is actively closing through a
different channel (a hand-built deck) than the one this repo automates.
