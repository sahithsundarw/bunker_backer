# MORNING REPORT

**Assume you read only this file.** Kept current as work proceeds, not written at the end.
Operational resume point is `docs/STATE.md` "RESUME HERE".

**Repo:** https://github.com/sahithsundarw/semicon-kla-image-restoration — public, anonymous
clone verified with credentials suppressed.

---

## CHECK TALLY

| | start of iteration 1 | now |
|---|---|---|
| PASS | 9 | **41** |
| FAIL | 44 | **12** |
| SKIP | 0 | 0 |

Tier 1 is **fully green, 9/9**. Tier 0 is **15/16** — only V06 (weights) remains.

## ✅ THE HARD GATE IS CLEARED — V25 = 43.3295 dB

`train.py --overfit 2` reaches **43.3295 dB** against the contract's 40 dB bar. This is the one
result that had to happen before any other number meant anything: a model that cannot overfit
two pairs it was trained on has broken alignment, normalisation or loss. It clears by 3.33 dB,
confirming the paired-crop geometry, the [0,1] convention, the unclipped-input handling and the
loss end to end. **The full 20k-iteration training run has since completed** — see the results
table above.

A scheduling subtlety I recorded because the failure mode is convincing and wrong: a 4000-iter
budget stalls at **39.78 dB** while 6000 reaches **43.33 dB**, because the cosine schedule decays
*proportionally to the budget* rather than truncating. A short overfit run landing just under
40 dB must not be read as an alignment failure — that misdiagnosis sends someone hunting a
geometry bug that does not exist.

## ⚠ NINE OF THE 53 CHECKS WERE INERT

V25 V26 V27 V28 V29 V32 V33 V34 V35 all returned an unconditional FAIL that **no artifact could
ever turn green**. They were BOOTSTRAP placeholders. All nine now test their real subject, with
anti-vacuity guards so none can pass by checking nothing.

This is worth your attention when reading any earlier tally from this project: the "44 FAIL" at
iteration 0 looked like honest red, but roughly a fifth of the suite was measuring nothing at
all. The checks that looked strictest were the ones certifying least.

Every remaining failure is honest and traceable to a missing artifact, not a broken fix:

| Cause | Checks |
|---|---|
| Needs the checkpoint **published** as a Release with a sha256 | V06 V59 |
| Needs `results/baselines/final/metrics.json` from `scripts/evaluate.py` — **not** the training log | V27 V28 V48 |
| Needs the UNetSR baseline trained at equal budget | V28 |
| Needs the 400 restored outputs, generated with `--require_weights` | V56 |
| Needs the runtime report (`perf-analyst`, never dispatched) | V37 V38 V39 V43 |
| Needs qualitative figures, including the honest failure case | V49 |

Nothing is blocked on you and nothing is blocked on a defect — every item is waiting on an
artifact that a specific command produces. **Work is paused here at your instruction**;
nothing is running.

---

## REVIEWER FINDINGS — `ml-skeptic` earned its place this iteration

It re-derived every headline number rather than reading the prose. **Nine of ten claims
reproduced to the published decimal** — including the bicubic 23.6524 ± 3.0236, the NLM
26.2722, and D3's anchor at 23.424736 ± 2.831883. One did not, and two structural problems
turned up that no check would have caught.

**F1 (HIGH) — a claim retracted, its conclusion was backwards.** `data-pipeline` reported that
synthetic LR has "1.33% of pixels > 1.0 against real 3.03%, so the simulator is 2.3× less
likely to exceed 1.0". Those 1.33% figures were measured on an **artificial sine-plus-noise
test tile**, then compared against real-dataset percentages — two different corpora. Re-derived
on the same 2800 pairs: synthetic **3.2523%** vs real **3.1437%**. The simulator *matches* the
bright tail; it **over**-produces the dark tail by 2.4×.
The real finding underneath is more useful and I kept it: the simulator under-produces the
**extreme** upper tail — synthetic max **1.7177** vs real max **2.0735** (dataset-wide 2.1580).
Gaussian shot+speckle has no mass at 4–5σ the way the real sensor does. A model never shown
inputs above ~1.72 will meet 2.158 in the released test data, and this project's entire transfer
argument rests on the degradation rather than on content. Queued for hardening as a measured
experiment. Logged as D25.

**F2 (MEDIUM/HIGH) — V33 was grading its own homework. This is the one I care about.**
`check_V33` asserted only `res["pass"]`, and every threshold lived in
`src/degrade.py::FIDELITY_TOLERANCE` — a file `VERIFIER_SHA256` does **not** pin. A future
iteration could have widened the bar until V33 went green **without touching a pinned file and
without tripping Prime Directive 1**. The subject under test owned its own pass mark. Fixed:
thresholds now live inside the pinned verifier and are applied *on top of* the module's flag,
so acceptance is the AND of both. Tightened while there — the worst-bin gain limit had 97%
headroom and was near-vacuous, now 3.0 → 4.5 (still 24% headroom against measured seed noise
of <0.003). Logged as D24.

**F3 (MEDIUM) — the verifier mutated what it verifies.** `fidelity_report()` writes its JSON
unconditionally, so *running the verifier* left `git status` dirty — breaking Definition-of-Done
criterion 5. Worse, V33's committed evidence was silently overwritten by whatever ran last, so
the artifact could never disagree with the code and was not independent evidence at all. The
check now snapshots and restores it byte-for-byte.

**Process note:** that is the second unfounded number caught in one iteration (the first was
`model-core`'s fabricated benchmark table). Both were caught by re-derivation, neither by
reading the claim. It is the argument for keeping `ml-skeptic` in every wave.

## DECISIONS I MADE UNDER YOUR AUTHORISATION

**1. B9 resolved: GitHub Release, not Git LFS, not a committed blob.**
You pre-authorised GitHub Releases, which resolves the deadlock I had escalated. The 400
restored test outputs (~105 MB raw) will ship as a Release asset with a published sha256, and
`results/restored_test_outputs/` will carry a manifest plus per-file hashes so the folder is
non-empty and independently verifiable. This needs **no contract change at all** — it reuses
exactly the mechanism V06 already permits for weights. Recorded honestly: the folder holds a
verified pointer and manifest, not the raw bytes, and the reason is stated in the README rather
than glossed. The alternative (a second V51 amendment to admit a ~40 MB `.npz`) would have gutted
the size caps I had just added, so I did not take it.

**2. V51 reconciled with V47 — bounded exemption, four new assertions.**
V51 banned every tracked `.npy`; V47 requires `sample_inputs/*.npy` *in a clean clone*. The two
could not both be green, so the Definition of Done was unreachable. Resolved with a narrow
exemption (≤8 files, ≤512 KB; actual 6 / 393,984 B) **plus** four strengthenings: blob-extension
ban widened 4→20 extensions, a dataset-directory-token ban, a 5 MB per-file cap and a 25 MB
total-tree cap. The last two catch a dataset dump under *any* extension, which an extension
blacklist provably cannot. Stated plainly in `BLOCKERS.md` B7: the exemption is, in isolation, a
loosening over six paths; everything else added is strictly stricter.

**3. Seven checks that could never pass are now real. This is the biggest correctness find.**
V26, V27, V28, V29, V32, V33 and V35 were BOOTSTRAP placeholders returning an unconditional FAIL
that **no artifact could ever turn green**. That is worse than a missing check: it looks identical
before and after a defect is introduced, so it silently certifies nothing. All seven now test
their subject, each with an anti-vacuity guard. Notable choices:
- **V27** enforces "a margin, not noise" *statistically* — the PSNR gain must exceed two standard
  errors of the mean — rather than with a constant I would have had to invent.
- **V28** implements the contract's negative-result escape hatch exactly as narrowly as written.
- **V32** additionally asserts a 3-channel input is **rejected**; a model that silently accepts
  3 channels would let an accidental BGR/RGB path through without ever raising.
- **V35** loads with `weights_only=True`, the same path `inference.py` uses, so a checkpoint
  needing arbitrary unpickling can no longer pass V35 and then break the shipped script.

**4. V09 was in direct conflict with V20.** It treated an unreadable *input* as a scale
violation, but a corrupt file forms no `(in, out)` pair, the contract says "for every pair", and
V20 explicitly declares corrupt inputs survivable. No implementation could satisfy both. Fixed to
match the contract wording, with unreadable inputs reported in evidence and a new guard that
fails V09 if the exclusion leaves zero pairs checked.

---

## RESULTS SO FAR

Classical baselines, 400-pair val split, scored on **reloaded float32 `.npy` from disk**:

| baseline | PSNR dB | SSIM | LPIPS |
|---|---|---|---|
| bicubic ×2 (the floor) | 23.6524 ± 3.0236 | 0.54775 ± 0.19197 | 0.41206 ± 0.15407 |
| median 3×3 → bicubic | 25.5057 ± 3.8785 | 0.61317 ± 0.17232 | **0.40870** ± 0.15866 |
| non-local means → bicubic | **26.2722** ± 4.3037 | **0.65152** ± 0.19523 | 0.42586 ± 0.18627 |
| **ours (NAFSR, EMA)** | **28.7851 ± 4.5324** | **0.78279 ± 0.14169** | **0.25233** |

**The model beats every baseline on all three metrics.** Run completed in 1:11:41 at 4.65 it/s,
20,000 iterations, seed 42, no OOM. Ledger row `20260815T062831Z-final-s42`.

| vs | ΔPSNR | ΔSSIM | ΔLPIPS (lower better) |
|---|---|---|---|
| bicubic (the floor V27 asks for) | **+5.13 dB** | +0.235 | −0.160 |
| non-local means (the honest bar) | **+2.51 dB** | +0.131 | −0.174 |

Two things worth noting about the shape of this result:

- **LPIPS improved alongside PSNR**, by a large margin (0.252 vs 0.409–0.426 for every
  baseline). That matters because the classical baselines showed PSNR/SSIM and LPIPS pulling in
  *opposite* directions — NLM won fidelity by 2.6 dB while scoring the worst LPIPS, by
  over-smoothing. A model that improved PSNR while degrading LPIPS would have been buying the
  scored blend with one hand and selling it with the other. This did not do that, which is
  evidence the balanced loss (Charbonnier + SSIM + FFT) is doing its job.
- **The in-run validation figure is higher than the headline.** Training logged
  `psnr 30.3944` at iteration 20000, but that is a **100-image subset** used for checkpoint
  selection. The headline **28.7851 is the full 400-image committed split**. The lower number
  is the honest one and is what gets reported; the gap is subset variance, not a regression.

**Two readings that change what we optimise:**
- **The honest bar is 26.27 dB, not 23.65 dB.** V27 only formally requires beating bicubic. A
  learned model that clears V27 but loses to a 35 ms classical filter is not defensible, so the
  trainer's target is set past NLM, not past bicubic.
- **PSNR/SSIM and LPIPS disagree across these baselines.** NLM wins fidelity by 2.6 dB while
  scoring the *worst* LPIPS, because it over-smooths. That is measured evidence for SPEC §8's
  balanced loss and against optimising any single metric — and it is good deck material.

D3's published anchor reproduced **exactly**: 23.424736 ± 2.831883 vs 23.4247 ± 2.8319.

**Model** (measured): NAFSR w48 n16 = **388,225 params**, 5.584 GMAC / 11.169 GFLOP per 128²,
checkpoint 3.14 MiB. UNetSR baseline = 2,970,401 params, 4.478 GMAC. Roughly FLOP-matched at
0.80× with NAFSR at 0.13× the parameters, so the baseline comparison gives NAFSR no parameter
advantage. NAFSR is **memory-bandwidth bound, not compute bound** — which is why SPEC §11.2's
optimisation table matters so little here.

**Degradation simulator**, whole-set over all 2800 non-val pairs (45,875,200 px): mean abs rel
err 0.3885, R² 0.9804, and it beats SPEC §5.2's two-parameter model by **1.894× mean / 5.917×
worst bin** — independent confirmation that SPEC's prescribed noise model is wrong here.

**Runtime:** `inference.py --help` import cost **2.35–2.48 s** against V23's 3.0 s Tier-0 budget.
The entire budget is `import torch`; inference.py's marginal cost above it is ≈0.

---

## INTEGRITY EVENTS WORTH KNOWING

**Two agents produced fabricated numbers. Both were caught, and I did not take either at its
word.** `model-core` self-disclosed inventing a benchmark table before running it, and separately
had an unchecked checkpoint size already committed. It retracted and re-measured — real values
1.17× (not the 1.09× it had quoted, which was inside run-to-run noise) and 3.14 MiB (not 4.7 MB).
`ml-skeptic` is now independently re-deriving the headline numbers rather than reviewing the
claims. The audit debt is tracked in `STATE.md`.

**`pip install lpips` silently replaces the CUDA torch with a CPU-only build.** Verified twice.
This matters far beyond the dev box: V04 installs a fresh venv from `requirements.txt` alone, so
an unpinned index yields a CPU-only torch — the run exits 0, **V04 passes**, and on KLA's H100 the
GPU sits unused while the throughput score collapses with no error anywhere. Logged as B8.

---

## THINGS AWAITING YOU

**Nothing is blocking.** B9 was the only item and your authorisation resolved it.

Two judgement calls I made that you may want to overrule later, both reversible:
1. **NAFSR is 0.388 M params, below SPEC §7.1's 1–3 M band.** Measured reason: on the 8 GB card
   the band costs 1.8–1.9× training wall-clock for an *unmeasured* quality gain, and w64/n28 does
   not fit at all (7925 MiB, 12.5× step-time collapse). On a one-day budget the number of runs
   that fit is the binding constraint. First thing to revisit once there is a quality number.
2. **`sample_inputs/` holds 6 real test inputs.** You authorised this; it is what makes V47 real.

---

## RISKS I AM TRACKING

- **V22 is expected to go red the moment a real checkpoint exists.** It reads 0.00e+00 today only
  because both precision arms take the bicubic fallback. Against an untrained NAFSR, bf16-vs-fp32
  measures **1.107e-03** against V22's 1e-3 limit. Remedies in order: re-measure with the trained
  checkpoint (random init is a worst case), keep LayerNorm/SCA in fp32, or switch to fp16 (10
  mantissa bits vs bf16's 8). **Widening the tolerance is not on the list.**
- **V23 is intermittent on a loaded machine** — 3.14 s observed under load against a 3.0 s budget,
  while bare `import torch` alone hit 5.3 s. Deferring the torch import would cut the `--help`
  measurement to ~0.25 s without changing the real run by a millisecond; that is gaming the
  metric and was deliberately not done.

---

## THE THREE THINGS TO DO NEXT — training is paused here at your instruction

1. **Publish the checkpoint as a GitHub Release with a sha256, and record it in
   `weights/README.md`.** This is the single highest-value remaining action and it needs no
   GPU. `weights/best.pt` currently exists **only on this machine** — `.gitignore` and V51 both
   refuse `*.pt`, so it is invisible to git and absent from every clone. Until it is uploaded,
   a reviewer who clones this repo gets a bicubic upsampler, not the model. That closes V06 and
   V59.
2. **Run `scripts/evaluate.py` on the trained checkpoint** to write
   `results/baselines/final/metrics.json`. V27, V28 and V48 read that file, **not** the training
   log — so despite the model comfortably beating every baseline, those checks are still red and
   will stay red until the evaluation record exists. Then generate the 400 restored outputs with
   **`--require_weights`** (without it, `inference.py` silently falls back to bicubic and would
   ship upsampler output as model results) and attach them to the same Release for V56.
3. **Train the U-Net baseline at the same 20k budget** (`configs/baseline_unet.yaml`, ~60–90
   min). This is the only remaining item that needs the GPU. V28 requires beating a *learned*
   baseline trained under an equal budget; the three baselines measured so far are classical, so
   the rubric's like-for-like comparison is genuinely missing right now.

After those: `perf-analyst` for the runtime report (V37–V39, V43), qualitative triplets
including the honest failure case `000984.npy` — described accurately as **broadband texture**,
not the periodic aliasing SPEC predicted and the measurement refuted — and a `--fresh-clone`
`--strict` run before tagging `v0.1-submittable`.
