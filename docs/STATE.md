# STATE

---

# ⚠ RESUME HERE — Phase 1 close-out, main session (Windows/RTX 4060), 2026-08-17

**Deadline extended to 2026-08-18 night.** Plan superseded:
`C:\Users\sahit\.claude\plans\as-of-now-whatever-steady-lemur.md` now tracks "Phase 1 close-out
— diagnose, launch, harden, ship" (the earlier "Round 2 differentiation" plan's own remaining
items — PRIORITY 0.1 origin/main reconciliation, most of PRIORITY 2/3 — are superseded or moot,
see below). Status:

- **Long run — DONE, checkpoint promoted and PUBLISHED.** HF Jobs A100,
  `configs/long_run_e.yaml` (w64n32, FiLM+uncertainty), 129,700-iter schedule, best at iter
  76,000. Won a paired head-to-head vs the prior checkpoint (PSNR/SSIM significant wins, LPIPS
  tie) and now beats U-Net on all 3 metrics. `weights/best.pt` sha256 `8f54f9a2082...`. Real,
  disclosed trade-off: real-SEM OOD SSIM/LPIPS got worse (`docs/decisions.md` D61).
  `results/restored_test_outputs/` is published (`artifacts-v2`, verified fetchable
  logged-out) — the "IN PROGRESS" status this section carried earlier is stale; confirmed live
  via `gh release view artifacts-v2` (5 downloads recorded).
- **V51 FIXED** (human-authorised, `docs/decisions.md` D62, `docs/BLOCKERS.md` B12 updated).
  **V22 stays open**, human chose to accept it as a disclosed trade-off. **Full fresh run,
  2026-08-17 (post V53 addition, 69 checks implemented): 64 PASS / 5 FAIL (V04, V22, V24,
  V46, V53).** V04/V46 `--fresh-clone`-only, independently verified passing on Linux; V22 the
  accepted trade-off; V24 a pre-existing, genuinely intermittent flake (~20%, B11 — rolled a
  fail this run, has also passed); V53 correctly FAILs because the deck is still a placeholder
  (real, open gap, not a check bug — `docs/decisions.md` D64).
- **PRIORITY 0.1 (origin/main reconciliation) is MOOT.** `main` was force-pushed to `origin`
  earlier this session (explicit, twice-confirmed human authorisation) — `origin/main` now
  equals local `main`. There is no divergence left to reconcile.
- **`UnrolledSR` overfit bug — investigation COMPLETE, honest negative result.** All 3
  hypotheses (adjoint identity, step-size stability, weight-tying) tested and cleared; no
  single fixable bug found. Not shipped, disclosed in README (`docs/decisions.md` D60).
- **A three-way audit (2026-08-17) found the code sound but `README.md` badly stale** relative
  to the D61 promotion — wrong sha256, wrong training narrative (still described Apple Silicon
  MPS closed-form fitting, not the real A100 gradient run), wrong/contradictory throughput
  numbers, an "unpublished" claim contradicted by the live Release. **README truth pass is
  IN PROGRESS as of this writing** (plan Phase A) — do not trust README's own claims about
  itself being current until this line is updated to say DONE.
- **Hour 0 diagnosis — DONE (`docs/decisions.md` D63).** Two paired probes
  (`scripts/ood_paired_probe.py`, `scripts/scale_gap_probe.py`) found the real-SEM OOD
  regression is idiosyncratic (concentrated on real-SEM only; procedural proxy-OOD actually
  wins) and a small but real train/test scale gap (128px inference vs 64px training patch).
- **Hour 0.5 — the post-promotion fine-tune ran, was evaluated, and was NOT promoted
  (`docs/decisions.md` D67). Incumbent `weights/best.pt` is UNCHANGED and remains shipped.**
  `configs/finetune_ood_wide.yaml` resumed from `weights/best.pt` on HF Jobs A100. Real
  operational finding: the `timeout="3h"` cap did NOT appear to be enforced by the platform —
  job was found running at 3h18m, manually cancelled via `cancel_job()`. Evaluated its two
  most-trained checkpoints (paired, val + both OOD sets): large in-distribution win
  (+0.445 dB PSNR) but did NOT fix real-SEM OOD (tie/loss) and BROKE proxy-OOD (significant
  loss, all 3 metrics — a set that was previously fine). Worse trade profile than the
  incumbent's own already-accepted one. Not promoted. `train.py` gained
  `optim.finetune_horizon`, `--push_every`, `--val_lpips` for this (all backward-compatible,
  verified via matching SMOKE_DIGEST before/after) — these stay in the codebase regardless,
  useful capability for any future fine-tune attempt.
- **P1.1/P1.2/P1.3/P1.4 (FiLM calibration, uncertainty calibration, Pareto plot, FP8 probe) —
  DONE.** See D57/D58/D59, `results/eda/{pareto_frontier.png,film_calibration.json,
  uncertainty_calibration.json,fp8_probe.json}`.
- **Phases A, B1, B2, B3, B4, C1, C2, C3 all DONE.** README truth pass; bf16/fp32 pricing
  (keep bf16); controlled old-vs-new throughput re-benchmark (corrects a wrong "faster despite
  bigger" claim — new checkpoint is actually ~55% slower, as expected for 3.6x params);
  free re-score of every long-run checkpoint under blended criteria (no swap warranted, D66);
  `results/experiments.csv` row for the shipped run; `results/qualitative/` regenerated; V53
  implemented. B4 folded into the Hour-0 `ood_paired_probe.py` work.
- **Still open:** deck/team info and demo video recording both need the user (plan Phase E).
  Everything else in the plan's Hours 0-4 is closed out. Remaining runway before the T-12h
  gate (2026-08-18 12:00) can go to Phase E and final verification/consolidation.

**Everything below this point (including the next "RESUME HERE" heading) is archived history
from the merge reconciliation and earlier sessions.** Kept for the audit trail; superseded by
this section.

---

# ARCHIVED — RESUME HERE (superseded above) — reconciled 2026-08-16, main session (Windows/RTX 4060)

Two independent lines of work on this repo (this session's Windows/RTX-4060 verification-driven
line, D1–D47; a teammate's `shanmukh sai` Mac/MPS `codex/*` line, D1–D48 with its own D41)
diverged from commit `9a0a4dd` and were merged today. Full reconciliation analysis:
`docs/MERGE_ANALYSIS.md`. Decision record: `docs/decisions.md` D49.

**Result: this session's from-scratch NAFSR w48n16 checkpoint ships**, having been re-scored
head-to-head against the teammate's promoted NAFSR w48n8 LS5+residual checkpoint under one
harness (identical `evaluate.py`/`metrics.py`/`split_val.txt` on both branches) and winning all
three metrics, paired, significantly (n=400: PSNR t=21.62, SSIM t=26.13, LPIPS t=-6.38 — see D49
for the full table). The teammate's checkpoint is faster (~2x at both 128→256 and 256→512,
fewer params) — a real, recorded trade-off, not a discarded finding — but SPEC's rubric is
quality-first with no throughput floor, so quality decided it.

**Adopted from the teammate's line regardless of which checkpoint ships:** Route A
(`weights/best.pt` committed directly, V51 exemption) as the checkpoint-delivery mechanism,
replacing this session's unresolved B6/B9 external-hosting question; Linux/Docker fresh-clone
verification records for V04/V46; the submission checklist;
`scripts/make_qualitative_examples.py`.

**Regenerated post-merge because they were produced against the now-superseded checkpoint:**
`results/qualitative/*`, `results/restored_test_outputs/*`, `results/runtime_report.md`'s
checkpoint rows, `README.md`'s checkpoint-specific numbers, `results/metrics_summary.md`
(machine-generated).

**Everything below this point is archived history from one branch or the other, kept for the
audit trail.** Do not treat any "RESUME HERE"/"CURRENT" heading below this one as current — this
section supersedes all of them. The immediately-following block (down to the next `---`) is the
teammate's own Mac/MPS session narrative (Phase 1–5, TTA, blend search) and is real, valuable
history — read it for that lineage's reasoning, not as a live status.

---

# ARCHIVED — teammate `shanmukh sai` session, Mac/MPS, branch `codex/residual-ls5-refinement` (2026-08-15)

## Where this branch stands
Base: `codex/train-first-model`. Verified-reproduced Phase-1 checkpoint `weights/best.pt`
(closed-form ridge-regularized 5x5 LS filter embedded into a NAFSR carrier, SHA256
`d5807dabad37b251f25d066838da9e3f73c164ec37ee777505a80e23cad9e90d`): val PSNR **26.3277 dB**,
SSIM 0.65999, LPIPS 0.39992 (400-image split, disk-verified). NLM baseline is 26.2722 dB, so the
LS-5 margin over classical is only +0.0555 dB — the reason this branch exists.

## Phase 2 — residual refinement on top of frozen LS-5: DONE, disk-verified
`scripts/train_residual.py` builds a **fresh, shallow** NAFSR (`num_blocks=4`, `width=48`),
transplants the frozen LS-5 `stem`/`head.expand`/`head.project` weights in directly (these
tensor shapes do not depend on `num_blocks`), freezes them, and trains only the new
`body`/`body_tail` (91,632 trainable params) as an additive residual on top of the closed-form
output — `NAFSR_output = LS5_output + learned_correction`. Small init
(`layerscale_init=0.02`, `body_tail_init_scale=0.02`) avoids the double-zero dead-gradient trap
that the raw LS-5 checkpoint's zeroed body would otherwise cause if gradient-resumed as-is.

Run `r1_nb4` (3000 iters, batch 32, lr 2e-4, MPS, seed 42, ~38 min wall):
```
python scripts/train_residual.py \
  --data_root /Users/shanmukhsai/Downloads \
  --out results/residual_experiments/r1_nb4/model.pt \
  --num_blocks 4 --layerscale_init 0.02 --body_tail_init_scale 0.02 \
  --iters 3000 --val_every 300 --val_limit 100 --batch_size 32 --lr 2e-4 --warmup_iters 100 \
  --device mps --seed 42 --tag phase2-r1-nb4-ls0.02-lr2e-4 --verbose
```
**Disk-verified (V30 round-trip via `make_baselines.py` + `evaluate.py`), full 400-image
split — the only authoritative number:**

| Method | PSNR dB | SSIM | LPIPS | n |
|---|---|---|---|---|
| LS-5 (Phase 1 baseline) | 26.3277 | 0.65999 | 0.39992 | 400 |
| residual_ls5 r1_nb4 | **27.7625 ± 4.0109** | **0.74462 ± 0.14524** | **0.30776 ± 0.16386** | 400 |

**+1.4348 dB over LS-5.** Clears "Good" (>27.0), approaching "Strong" (>28.0).

**Important caveat, resolved:** the in-loop model-selection validation (n=100 fixed subset)
climbed to 29.2597 dB during training — substantially higher than the true full-split number.
This is subset composition, not a bug or leakage (subset is a fixed non-leaked slice of the
committed val split, just smaller and easier-than-average). The checkpoint's embedded
`metrics` block was corrected after training to report the disk-verified 27.7625/0.74462/0.30776
as `val_psnr`/`val_ssim`/`val_lpips` (n=400), with the original in-loop number preserved under
`in_loop_selection_val_psnr_n100` for provenance. **Only ever cite the disk-verified, full-split
number.**

## Phase 3 — blend search: DONE, negative result (informative)
`scripts/blend_search.py` swept `alpha in {0.05, ..., 1.00}` for
`clip(alpha*refined + (1-alpha)*LS5, 0, 1)`. PSNR increased **monotonically** with alpha; **best
alpha = 1.00** (pure refined output), identical to the r1_nb4 result above. No blending helps —
expected, since the residual model already has the LS-5 output baked into its own forward pass
(frozen stem/head), so re-mixing raw LS-5 back in only dilutes the learned correction. No new
checkpoint was produced by this phase.

## Ledger
`results/experiments.csv` rows: `20260815T081652Z-final-closed-form-s42` (Phase 1, 26.3277),
`20260815T105016Z-residual-ls5-s42` (Phase 2 in-loop n=100 selection number, 29.2597 — kept
as-is for training-time provenance, NOT the reportable number),
`20260815T113000Z-residual-ls5-s42-diskverified` (Phase 2 authoritative, 27.7625, n=400,
explicitly noted in its `notes` field as superseding the in-loop row for reporting purposes).

## Do NOT retry (this branch)
- **Blending refined output back down with raw LS-5 output.** Strictly monotonic loss as alpha
  decreases from 1.0; every alpha < 1.0 scores worse. Measured via `scripts/blend_search.py`
  full sweep. Do not re-attempt without a materially different refined model.

## Phase 5 — TTA evaluation: DONE, negative result
Ran the real `inference.py` entrypoint (not a custom script) against `r1_nb4/model.pt` on the
400-image val-only input set, with and without `--tta` (8x dihedral self-ensemble, CPU,
`--require_weights`):

| | PSNR dB | SSIM | LPIPS | throughput |
|---|---|---|---|---|
| no TTA | 27.7625 ± 4.0109 | 0.74462 ± 0.14524 | 0.30776 ± 0.16386 | 4.1 img/s |
| `--tta` | 27.7952 ± 4.0169 | 0.74598 ± 0.14555 | 0.31113 ± 0.16618 | 0.4 img/s |

+0.0327 dB PSNR, +0.00136 SSIM, but LPIPS got **worse** (+0.00337, higher=worse), for a **9.3x**
runtime cost. Not worth it — do not enable `--tta` for the shipped recommendation.

## Phase 4 — higher-capacity, PSNR-focused fine-tune: DONE, disk-verified, NEW BEST — KEEP
Run `r2_nb8_psnrloss`: `num_blocks=8` (vs 4 in r1), `configs/phase4_psnr_focus.yaml` (new file,
does NOT touch `configs/final.yaml`) shifts loss weights toward Charbonnier
(`charbonnier=1.0, structural=0.05, fft=0.02`, `lpips` still off) since the r1_nb4 in-loop PSNR
curve was still creeping up (slowly) at iter 3000 with the default weights. 4000 iters,
`--layerscale_init 0.02 --body_tail_init_scale 0.02`, batch 32, lr 2e-4, warmup 150, MPS,
seed 42. **Training rate cratered to ~0.05 it/s (from r1's 1.31 it/s) while a concurrent
CPU-heavy `inference.py --tta` foreground job (Phase 5, above) was running** — Apple Silicon
unified memory contention between the MPS training process and the CPU inference process.
Recovered to ~0.3-0.6 it/s once the TTA job exited and stayed healthy (never dropped back below
the 0.1 it/s stop-rule floor) for the remaining ~2h20m to completion (wall_clock_s=9133.6).

**Disk-verified (V30 round-trip), full 400-image split — the authoritative number, and it
matches the training script's own in-memory final number exactly (28.0394 both times):**

| Method | PSNR dB | SSIM | LPIPS | n |
|---|---|---|---|---|
| LS-5 (Phase 1) | 26.3277 | 0.65999 | 0.39992 | 400 |
| residual_ls5 r1_nb4 (Phase 2) | 27.7625 ± 4.0109 | 0.74462 ± 0.14524 | 0.30776 ± 0.16386 | 400 |
| residual_ls5 r2_nb8_psnrloss (Phase 4) | **28.0394 ± 4.1881** | **0.74804 ± 0.15275** | **0.29571 ± 0.16672** | 400 |

**r2_nb8_psnrloss beats r1_nb4 on all three metrics** (+0.2769 dB PSNR, +0.00342 SSIM, better
LPIPS by 0.01205) — a clean win, not a metric trade-off. +1.7117 dB over LS-5. Crosses further
into "Strong" (>28.0). **This is now the recommended checkpoint**, superseding r1_nb4.
Checkpoint's embedded `metrics` corrected the same way as r1_nb4's (disk-verified numbers as
`val_psnr`/`val_ssim`/`val_lpips`, in-loop n=100 number preserved under
`in_loop_selection_val_psnr_n100` for provenance).

**Not re-verified for r2 (documented gap, not fabricated):** Phase 3 (blend search) and Phase 5
(TTA) were only run against r1_nb4, not r2_nb8_psnrloss. The blend-search conclusion (mixing in
raw LS-5 only hurts) is expected to hold by the same structural argument — r2 also has the
frozen LS-5 stem/head baked into its own forward pass — but this was **not separately
measured** for r2. Do not cite a blend or TTA number for r2 without re-running it.

## Do NOT retry (this branch), continued
- **Running a CPU-heavy job (e.g. `inference.py --tta` over 400 images) concurrently with an
  MPS training run on this machine.** Measured: training rate dropped from 1.31 it/s to
  0.05 it/s (26x slower) during the overlap. Unified-memory contention on Apple Silicon. Run
  CPU-bound and MPS-bound heavy jobs sequentially, not in parallel, on this hardware.

## Not yet done on this branch
- Optionally: re-run blend search and/or TTA specifically against r2_nb8_psnrloss to confirm
  the r1-derived conclusions transfer, if time allows.

## Submission integration — branch `codex/final-submission-28db` (2026-08-15)

Built off `codex/residual-ls5-refinement` @ `2720ccd`. Promoted `r2_nb8_psnrloss` to
`weights/best.pt` (new SHA256 `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7`,
verified via a live `inference.py --require_weights` dry-run before overwrite), regenerated
`results/restored_test_outputs/` (400/400 outputs from `/Users/shanmukhsai/Downloads/NoisyLR`,
validated from disk: shapes, dtype, finiteness, range, filename match — no PSNR/SSIM/LPIPS,
no GT exists), wrote `results/runtime_report.md` (local Mac CPU, 7.1 img/s, labelled as such,
not H100/CUDA), and updated `README.md`/`weights/README.md` to remove all "does not exist yet"
language. Old `codex/package-final-outputs` (26.3277 dB packaging) was used only as a
wording/format reference, per instruction — no old metrics or archive SHA carried forward.

**V51/V06/V59 contradiction resolved:** `weights/best.pt` was being flagged by V51's `.pt`
blob ban while V06/V59 required it tracked. Human-authorised narrow exception
(`CHECKPOINT_BLOB_EXEMPTION = "weights/best.pt"`, one exact path) added to `check_V51`;
`docs/VERIFIER_SHA256` re-pinned; full writeup in `docs/decisions.md` D30. V51 now PASSES.

**V32 `.venv-mac` false positive:** documented in D30 as local environment noise (a
differently-named, gitignored virtualenv that `check_V32`'s exact `.venv` match doesn't skip).
No code change — it does not reproduce on a fresh clone, which is what actually gets scored.

Full-suite result on this branch after the above (`--strict`, working tree, not fresh-clone):
**45 PASS / 12 FAIL**, up from 44/13 before the V51 fix. Remaining FAILs are all pre-existing
backlog (V04/V46 need `--fresh-clone`; V14 stdlib-module list gap; V25/V29/V34 need
`KLA_DATA_ROOT`; V27/V28/V38/V49 not implemented; V32 environment noise, see above) plus **V56**,
which is expected to stay red until the one remaining manual step — uploading
`/tmp/semicon_final_outputs_28db.zip` (sha256
`a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98`) as a GitHub Release asset
and filling in `manifest.json`'s `release_url` — is completed by a human.

## V04/V46 fresh-clone dry run — independently verified on real Linux (2026-08-15)

`_fresh_clone_run` uses `sys.executable` for the nested venv, so it can only ever exercise the
machine actually invoking `verify_all.py`. On this Mac dev box that machine is macOS/arm64, and
`requirements.txt` intentionally pins `torch==2.11.0+cu128` — a CUDA-only build with **no wheel
published for macOS at all** (confirmed: `pip download --platform manylinux_2_28_x86_64
--python-version 312 ... torch==2.11.0+cu128` resolves and downloads a real 820 MB wheel from
`download.pytorch.org/whl/cu128`, but the same install on macOS lists only bare, non-`+cu128`
versions). So V04/V46 FAIL on this machine by design, not because of a defect — this is exactly
the B8 "loud failure on the wrong platform" the pin exists to produce.

Independently verified on **real Linux** (`python:3.12-slim` Docker container, `git` + `numpy`
installed into the outer/orchestrating interpreter only — `numpy` is needed because `main()`
calls `build_fixtures()` at the top level, not just inside the nested clone's venv):

- `python3 scripts/verify_all.py --strict --fresh-clone --only V04` → **PASS**, "fresh clone +
  fresh venv end-to-end".
- `python3 scripts/verify_all.py --strict --fresh-clone --only V46` → **PASS**, same.
- Nested venv installed exactly `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128`, `torch.version.cuda
  == '12.8'`, matching `requirements.txt`'s own claim, with `inference.py` then running end to
  end against `tests/fixtures/single` inside that fresh venv.
- Running V04 and V46 **together** in one container (two full fresh clones + two full ~820 MB
  torch downloads back to back) produced one transient V04 FAIL with a truncated pip error
  ("line 560, in read"); re-run in isolation it PASSED cleanly. Read as resource/network
  contention from double-downloading in immediate succession inside one container, not a
  dependency or pin defect — recorded here rather than silently dismissed.

**Not committed as a code or verifier change** — the local, gitignored
`results/verification_report.json` still honestly reports V04/V46 FAIL because that is what
actually ran on this machine. This section is the durable record that the underlying
requirement (clean-environment dry run, on the Linux/CUDA platform KLA's H100 environment
actually matches) has been independently exercised and passes, even though this dev machine
cannot demonstrate that itself. `docs/BLOCKERS.md` B8 updated to match.

---

# ARCHIVED — this session's Windows/RTX-4060 line, iteration 3 status (pre-merge, 2026-08-16)

This is this session's own most recent pre-merge status (newer than the iteration-1 snapshot
further below), written with no visibility into the teammate's line above. Kept for the record
of what this session believed at merge time; superseded by the reconciled "RESUME HERE" at the
top of this file (D49).

## Since the last rewrite of this file
- **Decision A resolved (D40), committed `9a0a4dd`.** `perf-analyst`'s runtime benchmark
  (PID 30608) finished; **SHIPPED MODEL: NAFSR**, unchanged in `weights/best.pt`. Full
  reasoning and the honest V28 negative result (1 win/1 loss/1 tie, paired test) is in D40 —
  do not re-litigate with a different statistic (see Do-NOT-retry).
- `results/runtime_report.md` is now real and committed (`bdf4547`/`9a0a4dd`). End-to-end
  scored-shape throughput: NAFSR 24.82 img/s vs UNetSR 26.69 img/s (UNetSR 7.5% faster, not the
  4.78x isolated-compute gap — do not conflate the two, see D40 Do-NOT-retry).
- **Phase 2 planning done and committed (`95b73bf`, D41, `docs/PLAN_PHASE2.md`).** HF Jobs
  verified live under org `Team-Ceciroleo67` ($30 credit, expires 2026-09-01). This is Round 2
  (semifinal, deadline 2026-09-04) work — separate track from finishing Phase 1 below. Nothing
  in Phase 2 executes yet; it's blocked on deliberately sequencing after Phase 1 close-out per
  `PLAN_PHASE2.md` §8, and the tail-coverage fix (§5 item 4) is the only Phase-2-adjacent item
  that's local/no-GPU-budget and could be done anytime.
- **V22 fix dispatched to `inference-engineer`** this iteration (in flight — see below).

## Phase 1 close-out — remaining plan, in order (unchanged ordering, decision A now done)
1. ~~perf-analyst finishes, decide which model ships~~ **DONE (D40, commit 9a0a4dd).**
2. **V22 — IN FLIGHT.** Dispatched to `inference-engineer` with instruction to root-cause via a
   rebuilt per-layer diagnostic (prior one lived in scratchpad, doesn't survive a session reset)
   rather than guessing between "force LayerNorm fp32" (suspected no-op, autocast already
   promotes it) and "force SCA's spatial mean fp32" (the more plausible culprit, unverified).
   Awaiting agent report. **Since resolved — see D42.**
3. `adversarial-reviewer`'s findings: H4 now has a dedicated regression guard (**V64**, commit
   `e7e2feb`, D39). The 5 mediums + 7 lows in `reviews/adversarial-1.md` still need triage.
4. Full `--strict` run — has not happened since this whole batch landed (V57/V58/V60/V61/V62/V64
   all added since the last suite-wide pass). Then `ml-skeptic` re-run (paired stats, U-Net
   comparison — new territory since its last pass), `cleanroom-tester`.
5. **Two consecutive clean `--strict --fresh-clone` runs** (Definition of Done #2), never done
   even once yet. Note `check_V46` still doesn't literally execute README's fenced commands
   (H-4b, open) — a fresh-clone pass proves the hardcoded fixture path, not literally what a
   reviewer would copy-paste.
6. Tag `v0.1-submittable`.
7. **Only after that**: the deck (U-5, `docs/SPEC_ADDENDUM.md` section 11's mandatory verbatim
   proxy sentence, 9 slides max, `TeamName_KLA_PS01.pdf`). Deliberately last per explicit
   instruction — no check can catch its absence, so it doesn't gate anything else.

**Phase 2** (Round 2, separate track, deadline 2026-09-04): see `docs/PLAN_PHASE2.md`, not
summarised again here to avoid the two docs drifting out of sync.

U-9 (proxy-OOD report, `V63`) is the one remaining SPEC gap with no plan yet — needs GPU
evaluation on an OOD image subset; slot it in whenever the GPU is free and nothing higher up
this list needs it. **Since resolved — see D44.**

---

# ARCHIVED — unrelated prior session context (Windows/RTX-4060, 53-check framework). Ignore.

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
