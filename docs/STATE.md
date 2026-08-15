# STATE

---

# ⚠ RESUME HERE  (rewritten before every step — trust this over anything below)

**Written at:** iteration 2, after the adversarial-review response batch. **Last verified
commit:** `<pending — commit this file>`. **Verifier SHA:** re-pinned four times this batch,
currently `001b68705ce79a93f9d3baa874dfabdaec3d51a59af79f1ab3cf4b0bd8efbbbf` — confirm against
`docs/VERIFIER_SHA256` before trusting this number, it moves often right now.

## Checks: 62 defined (was 57 at the top of this iteration)

Added since the last full tally: **V57** (U-6, real input-clip path), **V58** (U-10, SPEC 2.3
link freshness), **V60** (adversarial H2/H3, destructive output-dir guard), **V61** (U-1,
size-agnosticism forwarded), **V62** (U-8, degradation order-randomisation measured). Every one
negative-controlled at add time — see D34/D35/D37/D38 for the exact mutation and the exact
green→red→green sequence.

**No full `--strict` run has happened since these landed.** Every PASS/FAIL claim below is from
a targeted `--only` re-run at the moment it's stated, not from a suite-wide pass. **Do not trust
any tally in this file as a whole-suite number — run `--strict` fresh before believing one.**
It has been deliberately deferred because `check_V07`/`V17`/etc. default to CUDA and a
`perf-analyst` runtime benchmark has held the GPU continuously (still running at time of
writing — see "IN FLIGHT" below).

## What changed this batch, in commit order (all pushed to `origin/main`)

1. **`c209cd2`→`ba22f70`**: checkpoint + outputs published to Release `artifacts-v1`,
   evaluation record written, V27/V48/V56 closed, V00/V28/V48 strengthened (V28's escape hatch
   was permanently unlocked — see D31, the single worst thing found this iteration), V06/V56/V59
   made to fetch-verify for real (D32), SSRF-guarded (D33).
2. **`4d64a82`**: README rewritten, 12 false/contradictory claims fixed (`docs-scribe`).
3. **`9e0771d`**: `scripts/evaluate.py`'s self-graded V28 line fixed — it was declaring its own
   PASS/FAIL from unpaired mean deltas, the exact defect D31 removed from the verifier,
   reintroduced one layer down.
4. **V57, V58, V60, V61, V62 added** (D35, D37, D38, D34, D34) — closes U-1, U-6, U-8, U-10,
   and adds a permanent regression guard for the adversarial H2/H3 finding.
5. **`--no_ledger` restricted to `--smoke` runs** (D36, M-1 closed).
6. **Four real bugs fixed in `inference.py`** from `adversarial-reviewer`'s first delivered
   report (D38): H2/H3 destructive output-dir (data loss, now refused), H4 (partial write
   failure exited 0), H1 (`--require_weights` didn't cover a shape-mismatched-checkpoint
   fallback), plus a README fix for C1 (the documented "command KLA runs" produced silent
   bicubic on a fresh clone). 1 critical + 3 of 4 highs fixed; H4 has no dedicated V-check yet
   (needs a filesystem-blocking fixture); 5 mediums + 7 lows logged in
   `reviews/adversarial-1.md` (gitignored, local) for a later pass.

## The measured V28 result — use these numbers, they are the honest ones

Paired per-image test, same 400 images, both models:

    psnr   mean diff -0.0943 dB   t=-6.11   better on  93/400   -> LOSS to U-Net
    ssim   mean diff +0.000135    t=+0.29   better on 172/400   -> TIE (not a win)
    lpips  mean diff -0.0120      t=-5.55   better on 235/400   -> WIN

**1 win / 1 loss / 1 tie. V28 is correctly FAIL.** Naive unpaired-mean counting (what
`evaluate.py` did before D-earlier's fix) would have called this "2 of 3" and been wrong — the
SSIM "win" was noise, not signal.

## ⚙ IN FLIGHT RIGHT NOW
- **`perf-analyst`'s runtime benchmark**, launched via `scripts/benchmark_runtime.py all
  --sweep_divergence`, PID **30608** (parent orchestrator; spawns short-lived `inference.py`
  subprocesses as children — the parent's own near-zero CPU/RSS is expected, it's just
  `subprocess.wait()`-ing). Log: `<scratchpad>/bench_runtime.log`, stderr
  `<scratchpad>/bench_runtime.err` (empty — no crash). Stuck on the `e2e` stage (scaling series
  1/25/50/100/200/400 x 5 repeats, plus 12 variants x ~5 repeats including a slow
  `cpu_fp32_bs32` pass) for **45+ minutes** as of this writing. Not hung — confirmed via live
  child `python.exe` processes cycling — just genuinely slow. `results/runtime_report.md` does
  not exist yet. **This is what decision A (which model ships) is waiting on.**

## ⚠ V22 — the one real defect, still unfixed
Unchanged from before this batch:

    V22  FAIL  bf16 vs fp32 diverge: mean 5.99e-04, max 1.27e-02

Mean passes (limit 1e-3); only max fails, at 1.27x the 1e-2 limit. Root-cause investigation
was done (not yet applied): `LayerNorm2d` is **already** fp32 under autocast policy per
PyTorch's own promotion rules (its docstring says so), so the previously-suggested "keep
LayerNorm in fp32" remedy is likely a no-op. `SCA`'s `x.mean(dim=(2,3))` over 16384 elements is
the more plausible culprit — spatial mean is not in autocast's fp32-promote list the way
`layer_norm` is. A per-layer diagnostic script is ready but unrun:
`<scratchpad>/diag_v22.py` — hooks every leaf module, compares bf16-autocast vs fp32 output per
layer on a real input, ranks by max abs diff. **Run this on the GPU before choosing a fix**,
rather than guessing between "force SCA to fp32" and "switch default to fp16" — the
`perf-analyst` sweep already includes an `fp16_bs32` and `fp32_bs32` timing variant, so the
throughput half of that decision will exist once `e2e` finishes.

## Remaining plan, in order (unchanged from the user's explicit ordering)
1. `perf-analyst` finishes → **decide which model ships** (paired numbers above; user's stated
   prior is "prefer the model winning more of the three metrics if close" and "report both,
   state the reasoning in decisions.md") → write the D-numbered negative-result entry V28's
   escape hatch actually requires (structured heading, all six measured means quoted,
   `SHIPPED MODEL: <name>` matching `weights/best.pt`'s embedded config) → V28 resolves either
   way honestly.
2. Fix V22 — root-cause via `diag_v22.py`, not guessing.
3. `adversarial-reviewer`'s findings: H4 still needs a V-check; the 5 mediums + 7 lows in
   `reviews/adversarial-1.md` need triage.
4. Full `--strict` run — first one since this whole batch landed. Then `ml-skeptic` re-run
   (paired stats, U-Net comparison — new territory since its last pass), `cleanroom-tester`.
5. **Two consecutive clean `--strict --fresh-clone` runs** (Definition of Done #2). Note
   `check_V46` still doesn't literally execute README's fenced commands (H-4b, open) — a
   fresh-clone pass proves the hardcoded fixture path, not literally what a reviewer would
   copy-paste.
6. Tag `v0.1-submittable`.
7. **Only after that**: the deck (U-5, `docs/SPEC_ADDENDUM.md` section 11's mandatory verbatim
   proxy sentence, 9 slides max, `TeamName_KLA_PS01.pdf`). Deliberately last per explicit
   instruction — no check can catch its absence, so it doesn't gate anything else.

U-9 (proxy-OOD report, `V63`) is the one remaining SPEC gap with no plan yet — needs GPU
evaluation on an OOD image subset; slot it in whenever the GPU is free and nothing higher up
this list needs it.

## Standing authorisation — unchanged, restated for a fresh session
**Pre-authorised:** making a check STRICTER (log + re-pin); new V-checks for reviewer findings;
installing packages, venvs, training runs, GitHub Releases, commit and push; rejecting an
experiment that does not improve a measured number (log it in Do-NOT-retry with the number);
architecture/hyperparameter/loss choices within SPEC sections 7-9 guided by measurement.
**Human-authorised this session, standing:** decision A (model-selection framework — prefer
more metric wins if close, report both, reason in `decisions.md`) and decision B (V06 must
fetch-verify for real — done, D32/D33).
**NEVER without the human:** weaken, delete, skip or widen the tolerance of any check; edit
`VERIFICATION_CONTRACT.md` except to add or tighten; train/fit on `test_NoisyLR`; download DIV2K
or attempt source identification; commit dataset files, weights, or anything over the V51 caps.

## Things a fresh session would otherwise rediscover the hard way
- **`pip install lpips` silently replaces CUDA torch with a CPU-only build.** Reinstall from
  the cu128 index and re-check `torch.cuda.is_available()`. Good state: torch 2.11.0+cu128,
  torchvision 0.26.0+cu128, CUDA 12.8, RTX 4060 Laptop, bf16.
- **Tool-managed background Bash caps at a 10-minute timeout.** A long-running benchmark or
  training run launched that way gets killed silently. Launch detached (PowerShell
  `Start-Process ... -PassThru` with redirected stdout/stderr) and poll the log file.
- **`train.py` defaults `--out` to `weights/best.pt`.** Any baseline run without an explicit
  `--out` destroys the shipped checkpoint. (Now also: `--no_ledger` only skips the ledger under
  `--smoke`, D36.)
- **A new V-check is code like any other, and this project has shipped several broken ones on
  day one** (V54 false positive, V55 SSRF, and this batch's own V62-sigma bug — see D34's "do
  NOT retry" note on testing a continuous draw's minimum against an absolute epsilon). Every
  addition in this batch was negative-controlled before being trusted; keep doing that.
- **`check_V46` does not literally execute README's fenced commands** — it checks they exist,
  then runs a separate hardcoded fixture sequence (H-4b). A README fix (like C1's) is not
  proven by a green V46; verify it by hand.
- **`inference.py`'s `require_weights` guarantee had a real gap** (H1): loading successfully is
  not the same as the architecture being correct. If touching checkpoint-loading code again,
  re-run the scale-mismatch repro in D38 rather than assuming `strict=True` is sufficient
  coverage.
- **Verifying a fetch-based check (V06/V56/V58/V59) costs real time and bandwidth** — a full
  `--strict` run now downloads ~94 MB (checkpoint + outputs archive) and hits 9 external URLs
  for V58. Budget for that; it is not a bug.
- **Nothing is currently blocked on the human**, except decision A's final write-up (waiting on
  the benchmark) and U-5/U-9, both explicitly sequenced for later.

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
