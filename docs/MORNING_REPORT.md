# MORNING REPORT

**Assume you read only this file.** Kept current as work proceeds, not written at the end.
Operational resume point is `docs/STATE.md` "RESUME HERE".

**Repo:** https://github.com/sahithsundarw/semicon-kla-image-restoration — public, anonymous
clone verified with credentials suppressed.

---

## CHECK TALLY

| | start of iteration 1 | now |
|---|---|---|
| PASS | 9 | **35** |
| FAIL | 44 | **18** |
| SKIP | 0 | 0 |

Tier 1 is **fully green, 9/9**. Tier 0 is 12/16.

Every remaining failure is honest and traceable to a missing artifact, not a broken fix:

| Cause | Checks |
|---|---|
| Needs `decisions.md` D22 (docs-scribe, in flight) | V00 |
| Needs a `--fresh-clone` run | V04 V46 |
| Needs a trained checkpoint (trainer, in flight) | V06 V25 V27 V28 V34 V35 V43 V44 V45 V48 |
| Needs the runtime report (perf-analyst, queued) | V37 V38 V39 |
| Needs qualitative figures | V49 |
| Was blocked on you — **now unblocked**, see below | V13 |

---

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
| **ours (NAFSR)** | training in flight | — | — |

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

## THE THREE THINGS I WOULD DO NEXT

1. **Land a trained checkpoint.** It alone unblocks ten checks (V06 V25 V27 V28 V34 V35 V43 V44
   V45 V48). The V25 overfit-to-40 dB gate runs first and gates everything after it.
2. **Tag `v0.1-submittable` the moment Tier 0 is green**, so a working fallback always exists on
   the remote regardless of what happens later.
3. **Fresh-clone run** to close V04/V46 and prove B8's `requirements.txt` fix in a clean venv —
   that is the one failure mode that passes locally and fails silently on the evaluator's machine.
