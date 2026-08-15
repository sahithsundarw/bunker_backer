# STATE

---

# ⚠ RESUME HERE  (rewritten before every step — trust this over anything below)

**A prior halt in this session (safety classifier blocking Bash) resolved itself** — shell
access returned and every item that was flagged mid-halt has since been independently
confirmed. Nothing below is carried over on trust; each line was re-verified against a live
check run or a re-executed negative control just now. Record kept because the resolution
mechanism (SSRF guard, clone-hole fix, self-grading fix) is worth knowing about even though
none of it is still open:

- **SSRF guard on `_fetch_digest` (D33): PROVEN, not just written.** Re-ran the 14-case
  negative control (`<scratchpad>/test_url_guard.py`) just now: 11/11 malicious vectors
  refused (`file://`, loopback, link-local, ftp, path-position spoof, host-suffix spoof,
  subdomain-suffix spoof, embedded credentials, non-443 port, plain http), 3/3 legitimate URLs
  accepted (the real asset, an uppercase host, the githubusercontent redirect target).
  **0 incorrect.** `docs/VERIFIER_SHA256` is pinned to `160bc228...` and V00 passes.
- **V45/V48 fresh-clone hole: fixed.** `.gitignore` now tracks `results/experiments.csv` and
  `results/baselines/*/metrics.json` (NOT the 2000 `.npy` predictions per baseline, which stay
  ignored to respect V51's 25 MB cap). Confirmed: `V45 PASS "2 runs logged"`,
  `V48 PASS "table reconciles against all 5 evaluation records"`.
- **`scripts/evaluate.py`'s self-graded V28 line: fixed** (commit `9e0771d`). It was declaring
  its own PASS/FAIL from unpaired mean deltas — the exact defect D31 removed from the verifier,
  reintroduced one layer down. Now uses `paired_compare()` in `src/metrics.py`, confirmed
  byte-for-byte parity with `verify_all.py --only V28`'s own win/loss/tie counts.
- **README.md: rewritten and committed** (`4d64a82`), 12 false/contradictory claims fixed,
  4 of 5 fenced commands executed in a genuine fresh anonymous clone as part of the fix.
- **Confirmed live, this instant:** `V00 V06 V45 V48 V56 V59` all PASS. `V28` correctly
  **FAIL** — 1 win (LPIPS) / 1 loss (PSNR) / 1 tie (SSIM), escape hatch not satisfied. That is
  the check working, not a defect.

## What is actually still open (verified moments ago, not carried over)
- **`results/runtime_report.md` does not exist.** `scripts/benchmark_runtime.py` is written
  (+769 lines, a full stage pipeline: scaling/variants/sweep/divergence) but has never been
  run to completion — no output file, and the two `python.exe` processes still alive are
  near-idle (18 KB / 5 KB RSS), not mid-benchmark. **The GPU is free right now.**
  V37, V38, V39, V43 remain red on this alone.
- **V61 (U-1, size-agnosticism) and V62 (U-8, order-randomisation) were drafted but never
  applied** — `grep -c "check_V61\|check_V62" scripts/verify_all.py` is 0. Patch script:
  `<scratchpad>/patch_new_checks.py`. Still the right design (V61 forwards {NAFSR, UNetSR} x 5
  shapes asserting exactly `(1,1,2H,2W)` and finite; V62 measures degradation randomisation and
  counts the pre-downsample branch by wrapping `src.degrade.downsample`) — apply, test, pin.
- **V22** (bf16 max-abs-diff 1.27e-02 vs 1e-02) is still unfixed. Real bug, not an artifact gap.
- **`adversarial-reviewer`** still never delivered (killed in iteration 1).
- U-5 (deck), U-6, U-9, U-10, M-1 of the 7 originally-uncovered requirements remain open.

---



**Written at:** iteration 2, mid-flight. **Last verified commit:** `3e230c6`.
**Verifier SHA:** `4e78dbca22ad9f71c3091bfeeb32ee798fbca96ca96d08468bc11748cec6178b` — matches
the pin, V00 green.
**Remote:** https://github.com/sahithsundarw/semicon-kla-image-restoration (public, anonymous
clone verified).

## Tally: PASS 47 / FAIL 10 of 57

Last full `--strict` run measured **PASS 43 / FAIL 14** at `c209cd2`. Four have gone green
since, each confirmed by a targeted re-run: **V00** (see the CRLF note below), **V27**, **V48**,
**V56**. Nothing has gone red. 47 + 10 = 57.

**FAIL (10):** V04 V22 V28 V37 V38 V39 V43 V45 V46 V49

| Check | Why it is red | Owner of the fix |
|---|---|---|
| **V22** | **The only genuine engineering defect left.** See its own section below. | `inference-engineer`, needs GPU |
| V28 | needs a *learned* baseline at equal budget | U-Net run **in flight now** |
| V45 | ledger needs >= 2 data rows; has 1 | same U-Net run closes it |
| V49 | `results/qualitative/` empty | `loss-metrics`, dispatched |
| V37 V38 V39 V43 | no `results/runtime_report.md` exists | `perf-analyst`, needs exclusive GPU |
| V04 V46 | require `--fresh-clone`, not yet run | main session, after the above |

## ⚙ IN FLIGHT RIGHT NOW
- **U-Net baseline training**, PID **32828**, detached (NOT a tool-managed background job, so
  the 10-minute Bash timeout cannot kill it). Run id `20260815T174833Z-baseline_unet-s42`,
  UNetSR 2,970,401 params, 20,000 iters, seed 42, `--out weights/baseline_unet.pt`.
  Log: `<scratchpad>/unet_train.log`, stderr `<scratchpad>/unet_train.err`. Expect 60-90 min.
  **`--out` matters: without it this run would overwrite `weights/best.pt`.**
- `loss-metrics` agent building `results/qualitative/` (CPU-only, told not to touch the GPU).
- `requirements-auditor` agent doing a static re-audit into `reviews/requirements-audit-2.md`.

## What iteration 2 has banked so far

**1. The checkpoint is deliverable (V59).** Release `artifacts-v1`, asset `best.pt`,
3288805 B, sha256 `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313`.
The digest is of the **served** bytes — re-fetched with `GITHUB_TOKEN`/`GH_TOKEN` cleared,
HTTP 200. D30. Route A (committing the 3.14 MiB file) was rejected: it fits V51's caps, but
taking it means editing `.gitignore` and V51 to admit a `.pt`.

**2. The evaluation record exists (V27, V48).** `results/baselines/final/metrics.json`:

    final   n=400   psnr 28.7865 +/- 4.5329   ssim 0.78287 +/- 0.14169   lpips 0.25324 +/- 0.13193

V27 passes by **+5.1341 dB** over bicubic, clearing its two-standard-error bar rather than
merely being positive.

**3. The 400 restored outputs are published (V56).** Produced by the shipped `inference.py`
with `--require_weights` (log line `loaded weights/best.pt (ema weights)`), 400 in 20.09 s
(19.9 img/s), cuda/bf16/batch 32, 0 unreadable, 0 write errors. All 400 re-loaded from disk:
float32, ndim 2, (256,256), finite, global range exactly [0.0, 1.0], filenames identical to the
inputs — **0 violations**. Archive `restored_test_outputs.zip`, 91069597 B, sha256
`fbdf8a652d26168cf41e01842ca28d38c53d1da1547bd8ce602b5b8e5d6ac750`, also verified anonymously
at HTTP 200. `manifest.json` + `sha256sums.txt` committed.

## ⚠ THREE NUMBERS THAT WERE WRONG IN THE DOCS — use these instead

1. **Quote `28.7865 / 0.78287 / 0.25324`, not `28.7851 / 0.78279 / 0.25233`.** The old triple
   came from `train.py`'s in-run validation. The new one is `scripts/evaluate.py` scoring
   float32 `.npy` **reloaded from disk** after clipping. Fourth-decimal differences, but the
   evaluation record is the authority and it is what V27/V48 read.
2. **The 20k NAFSR run took `1:11:43`, not `1:11:41`.** `results/experiments.csv` records
   `wall_clock_s = 4303.5`. Caught by review.
3. **Still never quote `30.3944`** — that is the 100-image checkpoint-selection subset.

## V22 — the one real defect left, and how NOT to fix it

    V22  FAIL  bf16 vs fp32 diverge: mean 5.99e-04, max 1.27e-02

Tolerance is mean < 1e-3 **and** max < 1e-2. **The mean passes comfortably; only the max
fails**, at 1.27x the limit. This is the failure STATE predicted would appear the moment a real
checkpoint existed — previously it read `0.00e+00` only because both precision arms took the
bicubic fallback.

Cause is bf16's 8 mantissa bits: at an output near 1.0 the representable step is ~2^-8 = 0.0039,
and error accumulates across 16 blocks. Remedies, in preference order:

1. Keep the numerically sensitive ops (LayerNorm / SCA / SimpleGate) and/or the model tail in
   fp32 under autocast.
2. Switch the CUDA default to **fp16** — 10 mantissa bits vs bf16's 8, same tensor-core
   throughput, and our activation scale carries no overflow risk.

**Widening V22's tolerance is NOT an option** — that is the Prime Directive 1 violation, and
the contract's own wording ("guards against a silently broken AMP path") is the reason.

**Open question worth measuring while fixing it:** the shipped restored outputs were generated
in **bf16**, but the 28.7865 dB evaluation record was computed in **fp32**
(`make_baselines.py` runs the model directly, without autocast). Nobody has measured what bf16
costs in dB. Measure bf16 vs fp16 vs fp32 PSNR/SSIM/LPIPS on the val split before choosing the
fix, and if bf16 costs real quality, regenerate the published outputs.

## ⚠ NEW GOTCHA: V00 can go red from line endings alone

V00 hashes `scripts/verify_all.py`'s **raw** bytes. `.gitattributes` says `* text=auto eol=lf`
and the committed blob is LF, but an editing tool had rewritten the working copy as **CRLF**,
so the on-disk hash was `966431a4...` against a pin of `4e78dbca...` and V00 failed. **Nothing
was tampered with and the repo was never wrong** — `git diff` was empty. Fixed by rewriting the
working copies of 10 tracked text files CRLF -> LF; `git add -A` then staged **zero** content
changes, proving it was a working-copy artifact only. If V00 goes red, check line endings
before believing anything else. This is `docs/BLOCKERS.md` B3 biting in a new way: B3
anticipated it in a fresh clone, not from a local editor.

## Agents — status
COMPLETE, do not re-dispatch: `inference-engineer`, `model-core`, `data-pipeline`,
`loss-metrics` (iter 1), `docs-scribe`, `trainer`, `ml-skeptic`, `requirements-auditor` (iter 1).
**Still owed: `adversarial-reviewer`** — killed by a usage limit before it wrote its file, so
`reviews/adversarial-1.md` does not exist. It needs the GPU, so it is queued behind training.
Also queued: `perf-analyst` (exclusive GPU), `cleanroom-tester` (after the README is final).

## Remaining plan, in order
1. U-Net finishes -> score it -> **V28**, **V45**.
2. `inference-engineer` on **V22**, measurement-first (bf16 vs fp16 vs fp32 quality AND
   throughput) before changing the default.
3. `perf-analyst` -> `results/runtime_report.md` -> **V37 V38 V39 V43**. Must have the GPU to
   itself or the timings are worthless.
4. `adversarial-reviewer` and `cleanroom-tester`.
5. `--strict --fresh-clone` -> **V04 V46**. Definition of Done needs this green on **two
   consecutive** iterations, from a fresh clone in a fresh venv.
6. Tag `v0.1-submittable` once Tier 0 is green.
7. Then the LOOP_PROMPT section 3 hardening loop. Quality first, throughput second.

## Standing authorisation — unchanged
**Pre-authorised:** making a check STRICTER (log + re-pin); new V-checks for reviewer findings;
installing packages, venvs, training runs, GitHub Releases, commit and push; rejecting an
experiment that does not improve a measured number (log it in Do-NOT-retry with the number);
architecture/hyperparameter/loss choices within SPEC sections 7-9 guided by measurement.
**NEVER without the human:** weaken, delete, skip or widen the tolerance of any check; edit
`VERIFICATION_CONTRACT.md` except to add or tighten; train/fit on `test_NoisyLR`; download DIV2K
or attempt source identification; commit dataset files, weights, or anything over the V51 caps.
**If I reason toward any of these because it would unblock progress: STOP, write it to
BLOCKERS.md, work something else. That reasoning is the signal, not the justification.**

## Things a fresh session would otherwise rediscover the hard way
- **`pip install lpips` silently replaces CUDA torch with a CPU-only build.** Verified twice.
  Reinstall from the cu128 index and re-check `torch.cuda.is_available()`. Good state:
  torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, RTX 4060 Laptop, bf16.
- **Tool-managed background Bash caps at a 10-minute timeout.** A 60-90 minute training run
  launched that way gets killed. Launch it detached (PowerShell `Start-Process ... -PassThru`
  with redirected stdout/stderr) and watch the log.
- **`train.py` defaults `--out` to `weights/best.pt`.** Any baseline run without an explicit
  `--out` destroys the shipped checkpoint.
- **A new V-check is code like any other.** V54 shipped a false positive and V55 an SSRF hole,
  both the day they were written. Verify every absence-check with a **negative control**.
  Done for V56 this iteration: stripping `--require_weights` from the manifest turned it red,
  `n_files=399` turned it red, and the byte-exact restore turned it green again.
- **`sample_inputs/` is populated and committed** (6 real inputs, 393,984 B). `.gitignore`
  carries deliberate negations for it, for `results/metrics_summary.md`,
  `results/degrade_fidelity/` and `results/restored_test_outputs/`. Do not "tidy up" those
  rules — several checks read those paths from a fresh clone.
- **V23 is intermittent on a loaded box** — measured 1.60 s this iteration against a 3.0 s
  budget, but 3.14 s once under load. The entire budget is `import torch`.
- **Nothing is currently blocked on the human.** `docs/BLOCKERS.md` has no open items.

---

## V-check status — HISTORICAL SNAPSHOT, SUPERSEDED. Re-run `--strict` instead.
Out of date in four ways: V04/V13/V25/V34/V44 have since gone green, four checks were added
(suite is now **57**), V10 was strengthened, and the model has been trained. Kept only as a
progress record; the "why each was red" grouping below is the part still worth reading.

**PASS 35 / FAIL 18 / SKIP 0** at commit `530a8a0` — was PASS 9 / FAIL 44 at iteration start.

PASS (35): V01 V02 V03 V05 V07 V08 V09 V10 V11 V12 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23
           V24 V26 V29 V30 V31 V32 V33 V36 V40 V41 V42 V47 V50 V51 V52
FAIL (18): V00 V04 V06 V13 V25 V27 V28 V34 V35 V37 V38 V39 V43 V44 V45 V46 V48 V49
per tier: T0[P12/F4] T1[**P9/F0 — fully green**] T2[P7/F5] T3[P3/F4] T4[P4/F5]

Every remaining failure is honest and traceable:
| Cause | Checks |
|---|---|
| Needed a `decisions.md` entry for the new verifier digest — **since done** | V00 |
| Needed a `--fresh-clone` run — **since done, both green** | V04 V46 |
| Needed a trained checkpoint — **model now trained; the rest need the evaluation record and the Release** | V06 V25 V27 V28 V34 V35 V43 V44 V45 V48 |
| Needs `results/runtime_report.md` (`perf-analyst`, never dispatched) | V37 V38 V39 |
| Needs qualitative figures | V49 |
| Needed a delivery mechanism for the restored outputs — **resolved, D23** | V13 |

## Iteration 1 triage — what was selected and why
Tier 0 first, ordered by how many other checks depend on the subject. The dependency root was
`inference.py` + `build_model`, which together gate 20+ checks, so Tier 0/1 was taken as one
coherent unit rather than 3-6 isolated IDs. Per `BLOCKERS.md` B4 the `dataset-forensics` slot
was NOT reserved (U1-U9 are all answered) and was reallocated to Tier 0 work.

## Consecutive-failure counters
No check has been attacked twice yet. The 18 remaining failures are all at count 1, and every
one of them is waiting on an artifact that does not exist yet rather than on a failed fix.
Nothing is near the escalate-at-3 or BLOCKED-at-5 threshold.

## Measured results banked so far
Classical baselines, 400-pair val split, scored on reloaded float32 `.npy` from disk:

| baseline | PSNR dB | SSIM | LPIPS |
|---|---|---|---|
| bicubic x2 (the floor) | 23.6524 ± 3.0236 | 0.54775 ± 0.19197 | 0.41206 ± 0.15407 |
| median 3x3 → bicubic | 25.5057 ± 3.8785 | 0.61317 ± 0.17232 | **0.40870** ± 0.15866 |
| non-local means → bicubic | **26.2722** ± 4.3037 | **0.65152** ± 0.19523 | 0.42586 ± 0.18627 |

- **The honest bar is 26.27 dB, not 23.65 dB.** V27 only formally requires beating bicubic, but
  a learned model that loses to a 35 ms classical filter is not defensible.
- **PSNR/SSIM and LPIPS disagree across these baselines** — NLM wins fidelity by 2.6 dB while
  scoring the *worst* LPIPS, because it over-smooths. Direct evidence for SPEC §8's balanced
  loss and against optimising any single metric. Good deck material.
- D3's anchor reproduced exactly: bicubic on `003000-003199` gives 23.424736 ± 2.831883 vs the
  published 23.4247 ± 2.8319.

Model (measured, `model-core`): NAFSR w48 n16 = **388,225 params**, 5.584 GMAC / 11.169 GFLOP
per 128² image, checkpoint 3.14 MiB for model+ema. UNetSR w32 L4 = 2,970,401 params,
4.478 GMAC. Roughly FLOP-matched at 0.80x with NAFSR at 0.13x the parameters, so the baseline
comparison gives NAFSR no parameter advantage. NAFSR is **memory-bandwidth bound, not compute
bound** (32.8% layer_norm, 17.9% conv bias-add, 16.2% convolution), which is why SPEC §11.2's
optimisation table matters so little here — channels_last and bf16 each move it <20%.

Degradation simulator (measured, `data-pipeline`), whole-set over all 2800 non-val pairs,
45,875,200 px: mean abs rel err 0.3885, R² 0.9804, std ratio 1.0554, and it beats SPEC §5.2's
2-parameter model by 1.894x on the mean bin and 5.917x on the worst bin.

## Do NOT retry (tried and rejected, with the measurement that justified rejection)

- **Per-image min-max renormalisation of outputs.** Costs **-4.66 dB PSNR**, loses on 191/200
  held-out pairs. 95.5% of predictions overshoot 1.0, so renorm divides by an outlier-driven
  range. Clip to [0,1] and nothing else. (D3)
- **Treating the downsample kernel as a 2x2 box / average-pool.** Refuted. Least-squares
  recovery over 3.125 M equations gives centre weights 0.320 with negative surround lobes.
  Box costs +7.72e-04 residual std vs the optimum; `bicubic(antialias=False)` +1.22e-05. (D1)
- **The two-parameter noise model `sigma^2 + v*x^2` for SIMULATION.** Over-noises dark regions
  by up to 12.5x. Use sigma=0, a=0.011253, v=0.015745. The 2-par values (0.036991, 0.026781)
  remain the correct answer to SPEC §5.2 as literally posed but must never generate data. (D12)
  Now independently confirmed: the 3-par simulator beats the 2-par one by 5.9x on the worst bin.
- **Identifying the source dataset / DIV2K crop-match.** DENIED PERMANENTLY by the human. (D11)
- **Pretrained initialisation for Phase 1.** Every classical x2 SR checkpoint assumes clean
  bicubic with no noise; our inputs carry residual std 0.092 with 3% of pixels outside [0,1],
  so the prior points the wrong way. From scratch. (D9, D13)
- **`import cv2` / `tifffile` / any image library in `inference.py`.** Data is `.npy` end to
  end; several cv2 paths silently convert to 8-bit or clip to [0,1], corrupting inputs that
  legitimately reach 2.16. (SPEC_ADDENDUM §5)
- **An 8-worker DataLoader for the 400-image test set.** The whole test set is 25.05 MB and
  worker spawn costs more than the read. Eager-load. Deliberately contradicts SPEC §11.2.
  **Still not independently measured** — `perf-analyst` must confirm or refute. (D7)
- **Git LFS for `results/restored_test_outputs/`.** Ruled out by human instruction: unresolved
  LFS pointer stubs on a fresh clone are a known way to fail V06. (D17, B9)
- **`pip install lpips` without pinning the PyTorch index.** Replaces `torch==2.11.0+cu128`
  with `torch==2.13.0+cpu`; in a fresh venv this yields a CPU-only torch that PASSES V04 while
  destroying the throughput score silently. (B8)
- **Narrowing degradation randomisation to buy in-distribution dB.** Rejected on objective
  grounds: the hidden test set may be real semiconductor imagery and the measured degradation
  is the only asset that transfers. (D16)
- **Hand-rolled `(x-mean)/sqrt(var)` channel LayerNorm.** Measurably slower and 1.43x the VRAM
  than `F.layer_norm` on an NHWC view: 4939 vs 4233 ms inference, 305 vs 208 ms train step,
  4970 vs 3486 MiB, non-overlapping ranges over 5 interleaved repeats. (D21)
- **NAFSR width >=64 with num_blocks >=28 on the 8 GB dev GPU.** Does not fit: 7925 MiB of
  8 GB and a 12.5x step-time collapse from allocator spill. (D20)
- **Chasing `channels_last` / fp16 for NAFSR throughput.** Each moves it <20% because the model
  is memory-bandwidth bound, not compute bound. (D21)
- **Deferring `import torch` into `main()` to make V23 look better.** It would cut the
  `--help` measurement to ~0.25 s without changing the real run by a millisecond. Gaming the
  metric, not reducing the cost. Rejected deliberately.

## Backlog — the 7 remaining UNCOVERED requirements (requirements-auditor, iteration 1)
Four of the eleven gaps were closed as V54/V55/V56/V59 (D27). These seven remain; each is a
requirement **no check can currently turn red**, with the auditor's proposed implementation.

| Ref | Requirement | Proposed check |
|---|---|---|
| U-1 | **F2 size-agnosticism is verified only by dead code.** The 256→512 fixture the addendum calls "the *only* guard against silently baking in 128→256" lives in `src/model.py::_selftest()`, which **nothing invokes**. UNetSR's pad/crop-back is never forwarded by any check and is the likeliest home for an off-by-`ph*s` bug. | **V61** Tier 2: for arch ∈ {NAFSR, UNetSR} × (h,w) ∈ {(128,128),(256,256),(61,97),(1,1),(130,66)} assert `(1,1,2h,2w)` and finite. FAIL if <10 combinations ran. |
| U-5 | **The deck is entirely outside the contract** — F13 format, F14 deck-side disclosure, and the addendum's mandatory proxy sentence. Its absence is not even tracked. | **V53** Tier 4: exactly one `*_KLA_PS01.pdf`, ≤9 pages, text contains "natural photograph" and "proxy", carries the repo URL, and contains **none** of the addendum's banned phrases. |
| U-6 | **V12 tests a helper, not the model input.** It calls `src.io_utils.load_array` and checks the return; the contract says "the tensor **entering the model**". A `clamp_` anywhere in `inference.py`'s stack/H2D/autocast path leaves V12 green. Training path untested entirely. | **V57** Tier 0, subsumes V12: forward pre-hook records min/max of the tensor actually entering the model; require ≤−0.27 and ≥2.15 to survive. |
| U-8 | **F4 order randomisation asserted nowhere.** `GAUSS_PRE_DOWN_PROB` and the pre-downsample branch could be deleted silently; V33 compares only the variance curve, which the order hedge barely moves. | **V62** Tier 2: 2000 samples — `a`,`v` span ≥90% of their ±30% range, `sigma` attains both ~0 and >0.015, pre-down branch taken 8–22% of the time. |
| U-9 | **F7 proxy-OOD report absent.** SPEC §10 requires it and `decisions.md` D4 accepts the duty; nothing implements or checks it. | **V63** Tier 4: `metrics_summary.md` needs a proxy-OOD heading with PSNR/SSIM/LPIPS, membership from a committed list with empty train intersection. |
| U-10 | **Official links never re-verified.** Licence links were re-fetched and dated; SPEC §2.3's hackathon links were not. | **V58** Tier 4: `docs/link_check.md` with url / status / UTC timestamp per §2.3 URL, re-issued unauthenticated, ≤72 h stale. |
| M-1 | `train.py --no_ledger` lets a run skip `results/experiments.csv` entirely — an escape hatch around V45 and SPEC §9's "log every run". | Restrict to smoke paths, or write a `smoke=true` row. |

Also from that audit, lower severity: **M-3** `IMPORT_ALLOWLIST` contains `__future__`, which is
not in CLAUDE.md §STYLE's "exactly eight" and is documented nowhere — correct on the merits
(zero import cost) but an undocumented widening; add a decisions line or amend CLAUDE.md.
**M-4** V23 scans `tree.body` only, so an import nested in `if TYPE_CHECKING:`/`try:` is
invisible. **M-5** V31 is a substring scan of `src/metrics.py` and would not notice the README
or deck stating *different* settings, which §15 requires be stated. **M-6** V30 passes if
`np.load` appears anywhere in `evaluate.py` — near-vacuous, though the implementation is in
fact correct. **LOW** `weights/_probe.pt` (3.29 MB untracked scratch) will be mistaken for a
checkpoint by a reader; delete it.

## Backlog (medium/low findings, no action yet)
- `results/restored_test_outputs/` still empty. Blocked on B9 + a trained model. V13 red.
- No `--fresh-clone` run performed this iteration; V04 and V46 red as a result.
- `results/qualitative/` empty; V49 red.
- V28 needs the UNetSR baseline trained under the same budget as NAFSR to be a fair comparison.

## Next iteration plan (iteration 2)
1. Close V00 (D22), then re-run `--strict`.
2. Dispatch `perf-analyst` for the runtime report (V37 V38 V39 V43) once a checkpoint exists.
3. Run `--fresh-clone` and settle B8 end to end in a clean venv (V04, V46).
4. Generate `results/qualitative/` triplets including the honest failure case `000984.npy`
   (80.5% of GT spectral energy above the LR Nyquist limit — describe it accurately as
   **broadband texture**, NOT as periodic aliasing, which is what SPEC predicted and the
   measurement refuted).
5. Escalate B9 to the human — it blocks V13 and cannot be resolved agent-side.
