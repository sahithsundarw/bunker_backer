# STATE

---

# ⚠ RESUME HERE  (rewritten before every step — trust this over anything below)

**Written at:** iteration 1. Wave A + docs-scribe COMPLETE and committed. `trainer` and three
read-only reviewers in flight. **STANDING AUTHORISATION for an autonomous overnight run is in
effect — do not stop to ask; work to LOOP COMPLETE then the §3 hardening loop.**
**Last commit:** `99f70de` (pushed). **Remote:** https://github.com/sahithsundarw/semicon-kla-image-restoration (public, anonymous clone verified).
**Verifier SHA:** `590c8e3344f2a7dbfadf63bace9a255c97ee73269c7894bc56855270e709d5bd`

## Tally: PASS 41 / FAIL 12 — Tier 0 is 15/16, only V06 left
V04, V13, V25, V34, V44 all flipped green since the last full run. **V25 — the hard gate —
CLEARED at 43.3295 dB against a 40 dB bar.** Alignment, normalisation and the loss are
confirmed end to end, so every quality number measured from here is trustworthy in a way
nothing before it was.

## ⚠ NINE of the 53 checks were inert placeholders
V25 V26 V27 V28 V29 V32 V33 V34 V35 all returned an unconditional FAIL that **no artifact
could ever turn green**. All nine are now implemented against their real subject. Worth
remembering when reading any historical tally: the "44 FAIL" at iteration 0 looked like honest
red when a fifth of the suite was measuring nothing at all.

## Training run IN FLIGHT
`py -3.12 train.py --config configs/final.yaml --seed 42 --iters 20000 --tag iter1-nafsr-20k`
Background shell `b53yt7v63`. Expected ~74 min (measured 221 ms/step at batch 32 / 64px on the
4060). On completion it writes `weights/best.pt` and appends to `results/experiments.csv`,
which unblocks V06 V27 V28 V35 V43 V45 V48.
**If it OOMs:** halve the batch size and retry, up to three times, logging each attempt. Never
stall. 8 GB card.

## Live at this write
| Agent / job | Owns | Status |
|---|---|---|
| `trainer` | `train.py`, `src/utils.py`, `results/experiments.csv`, `weights/*.pt` | RUNNING — targets V25 V34 V44 V45; its checkpoint unblocks V06 V27 V28 V35 V43 V48 |
| `adversarial-reviewer` | `reviews/adversarial-1.md` | RUNNING (read-only) |
| `requirements-auditor` | `reviews/requirements-audit-1.md` | RUNNING (read-only) |
| `ml-skeptic` | `reviews/ml-skeptic-1.md` | RUNNING (read-only) |
| bg shell `bjlhu40kn` | — | `verify_all.py --fresh-clone --only V04,V46,V47`; slow (builds a venv and installs ~2.5 GB of torch) |

COMPLETE and committed, do **not** re-dispatch: `inference-engineer`, `model-core`,
`data-pipeline`, `loss-metrics`, `docs-scribe`.

## THE NEXT CONCRETE ACTION
1. Collect the fresh-clone result for **V04** (the last Tier 0 item not waiting on weights).
2. When `trainer` lands a checkpoint: check **V25 first** (overfit 2 pairs > 40 dB). It is the
   hard gate — if it fails, alignment/normalisation/loss is broken and every downstream number
   is meaningless. Do not proceed past it.
3. **The moment Tier 0 is fully green, tag `v0.1-submittable` and push the tag.** Priority 2 of
   the standing authorisation: a working fallback must always exist on the remote.
4. Publish the outputs + checkpoint as a GitHub Release (pre-authorised), then send the real
   numbers and digests to `docs-scribe`, which is waiting to fill in the README results table,
   `weights/README.md` and the outputs manifest. It has been told not to guess them.
5. Dispatch `perf-analyst` (owns `scripts/benchmark_runtime.py`, `results/runtime_report.md`)
   for V37 V38 V39 V43, and `cleanroom-tester` once the README is final.
6. Then Tiers 1-4, then the §3 hardening loop. Model quality first, throughput second.

## Standing authorisation — what I may and may not do
**Pre-authorised:** any change making a check STRICTER (log + re-pin); new V-checks for defects
reviewers find; installing packages, venvs, training runs, GitHub Releases, commit and push;
rejecting an experiment that does not improve a measured number (add it to Do-NOT-retry with the
measurement); architecture/hyperparameter/augmentation/loss choices within SPEC §7-§9 guided by
measurement.
**NEVER without the human:** weaken, delete, skip or widen the tolerance of any check; edit
`VERIFICATION_CONTRACT.md` except to add or tighten; train/fit anything on `test_NoisyLR`;
download DIV2K or attempt source identification; commit dataset, weights, or anything over the
V51 caps. **If I find myself reasoning toward any of these because it would unblock progress:
STOP, write it to BLOCKERS.md, work something else. That reasoning is the signal, not the
justification.**

## Training guidance in force
On CUDA OOM: halve batch size and retry up to three times, logging each. Never stall. The GPU is
an RTX 4060 with 8 GB. Run the SPEC §16 gates in order. The bicubic baseline is already
established (23.6524 dB), so every later number has a floor.

## Things a fresh session would otherwise rediscover the hard way
- **`pip install lpips` silently replaces the CUDA torch with a CPU-only build.** Verified
  twice. Reinstall from the cu128 index afterwards and re-check `torch.cuda.is_available()`.
  Good state: torch 2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, RTX 4060 Laptop, bf16.
- **`scripts/verify_all.py` has been edited twice this iteration.** Any further edit needs its
  own `decisions.md` entry and a re-pin, or V00 fails by design.
- **`sample_inputs/` is populated and committed** (6 real inputs, 393,984 B). `.gitignore`
  carries explicit negations for it, for `results/metrics_summary.md` and for
  `results/degrade_fidelity/`. Do not "tidy up" those rules — three checks read those paths
  from a fresh clone.
- **B9 is blocked on the human** and blocks V13. Do not resolve it by loosening V51 again.
- **V22 is expected to go red the moment a real checkpoint exists.** It currently reads
  `mean 0.00e+00` only because there are no weights, so both precision arms take the bicubic
  fallback. Measured against an untrained NAFSR: bf16-vs-fp32 mean **1.107e-03** against
  V22's 1e-3 limit — a coin flip. Remedies in preference order: (a) re-measure with the
  TRAINED checkpoint, since random init is a worst case; (b) keep LayerNorm/SCA/SimpleGate in
  fp32; (c) switch the CUDA default to fp16 (10 mantissa bits vs bf16's 8, same tensor-core
  throughput, no overflow risk at our activation scale). **Widening V22's tolerance is NOT an
  option** — that is precisely the PD1 violation.
- **V23 is intermittent on a loaded box.** Measured `inference.py --help` at 2.35-2.48 s
  against a 3.0 s Tier-0 budget, but one run hit 3.14 s while sibling agents loaded the
  machine (bare `import torch` hit 5.3 s in the same conditions). The entire budget is
  `import torch`; inference.py's marginal cost above it is ~0. Deferring the torch import into
  `main()` would cut the `--help` measurement to ~0.25 s **without changing the real run by a
  millisecond** — that is gaming the metric, and it was deliberately not done.

---

## V-check status  (measured at `530a8a0`, full `--strict` run)
**PASS 35 / FAIL 18 / SKIP 0**  — was PASS 9 / FAIL 44 at the start of this iteration.

PASS (35): V01 V02 V03 V05 V07 V08 V09 V10 V11 V12 V14 V15 V16 V17 V18 V19 V20 V21 V22 V23
           V24 V26 V29 V30 V31 V32 V33 V36 V40 V41 V42 V47 V50 V51 V52
FAIL (18): V00 V04 V06 V13 V25 V27 V28 V34 V35 V37 V38 V39 V43 V44 V45 V46 V48 V49
per tier: T0[P12/F4] T1[**P9/F0 — fully green**] T2[P7/F5] T3[P3/F4] T4[P4/F5]

Every remaining failure is honest and traceable:
| Cause | Checks |
|---|---|
| Needs `docs/decisions.md` D22 (docs-scribe, in flight) | V00 |
| Needs a `--fresh-clone` run (not performed this pass) | V04 V46 |
| Needs a trained checkpoint (`trainer`, in flight) | V06 V25 V27 V28 V34 V35 V43 V44 V45 V48 |
| Needs `results/runtime_report.md` (`perf-analyst`, not yet dispatched) | V37 V38 V39 |
| Needs qualitative figures | V49 |
| Blocked on the human (B9) | V13 |

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
