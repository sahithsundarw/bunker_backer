# STATE

---

# ⚠ RESUME HERE  (rewritten before every step — trust this over anything below)

**Written at:** iteration 1, wave A INTEGRATED and committed, wave B (`trainer`) in flight.
**Last commit:** `530a8a0` (pushed). **Remote:** https://github.com/sahithsundarw/semicon-kla-image-restoration (public, anonymous clone verified).
**Verifier SHA:** `590c8e3344f2a7dbfadf63bace9a255c97ee73269c7894bc56855270e709d5bd`

## Live agents at this write
| Agent | Files | Status |
|---|---|---|
| `trainer` | `train.py`, `src/utils.py`, `results/experiments.csv`, `weights/*.pt` | RUNNING — wave B. Targets V25 V34 V44 V45, and produces the checkpoint that unblocks V06 V27 V28 V35 V43 V48 |
| `docs-scribe` | `README.md`, `requirements.txt`, `weights/README.md`, `docs/decisions.md` | RUNNING — has already landed requirements.txt (V14 PASS) and the disclosure (V50 PASS); still owes D15-D22 |

Wave A builders `inference-engineer`, `model-core`, `data-pipeline`, `loss-metrics` are all
**COMPLETE and committed**. Do not re-dispatch them.

## THE NEXT CONCRETE ACTION
1. Wait for `trainer` and `docs-scribe`. Then `py -3.12 scripts\verify_all.py --strict`.
2. **V00 is red until `docs/decisions.md` contains `590c8e33…09d5bd` verbatim** (entry D22,
   assigned to docs-scribe). If docs-scribe did not deliver, write D22 yourself — full content
   is in `docs/BLOCKERS.md` B7 and the commit message of `530a8a0`.
3. Run the review wave: `adversarial-reviewer`, `requirements-auditor`, `cleanroom-tester`,
   `perf-analyst`, `ml-skeptic` — all read-only, all write only to `reviews/`.
4. `perf-analyst` also owns `scripts/benchmark_runtime.py` + `results/runtime_report.md`,
   which is what V37 V38 V39 V43 need. Dispatch it once a checkpoint exists.
5. Step 7 ledger, then STOP. **Do not begin iteration 2.**

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
