# PLAN_PHASE2.md

**Written:** 2026-08-16, iteration start of Phase 2 planning.
**Supersedes:** nothing — this is a new document, append-only per repo convention once execution starts (log deviations, don't silently edit history).

## 0. Timeline and budget, stated once

Per `docs/SPEC.md` §1 (lines 116–126):

| Date | Event |
|---|---|
| 2026-08-16 (**today**) | Phase 1 submission deadline — **already met**, see §1 below |
| 2026-08-28 – 2026-09-04 | Semifinal Round 2 development + submission window |
| 2026-09-05 | Semifinal evaluation |
| 2026-09-06 | Top 10 announced |

This document is the Round 2 plan. **Real deadline: 2026-09-04.** That's 19 days from today.

**Cloud credit expires 2026-09-01** (confirmed on the Team-Ceciroleo67 billing page, screenshot reviewed this session) — **16 days from today**, 3 days short of the actual submission deadline. Any cloud-dependent work must land and be synced back to the repo by 2026-09-01; the last 3 days (09-02 to 09-04) are local-only buffer for writeup, deck, final `--strict` runs.

---

## 1. Status check — what Phase 1 already shipped (do not redo)

The master prompt this plan responds to was written without visibility into how far Phase 1 got. Checked against `docs/STATE.md`, `docs/decisions.md`, `results/experiments.csv`, and `git log` before planning anything below, so Round 2 effort isn't spent re-doing finished work.

| Master-prompt "Tier A" item | Actual status | Evidence |
|---|---|---|
| Train U-Net baseline at same 20k budget | **DONE** | `results/experiments.csv` row 2: `UNetSR`, 20,000 iters, 2026-08-15, `weights/baseline_unet.pt` |
| Publish checkpoint + restored outputs via Release, sha256, logged-out link check | **DONE** | commits `c209cd2`, `ba22f70`, `3e230c6c`; V06/V56/V59 fetch-verify for real (D32/D33) |
| `results/runtime_report.md`, startup-vs-compute, timed externally | **DONE** | committed last commit `bdf4547`; see corrected numbers in §2 below |
| `results/qualitative/` — successes + honest failure | **DONE** | commit `361b233` (V49), failure case `000984.npy` documented as broadband texture, not periodic aliasing |
| Fix README status/Training-section contradiction | **DONE** | commit `4d64a82`, "12 false/contradictory claims" fixed |

**All five Tier A items are closed.** Nothing in Tier A is open work for Round 2.

**What is genuinely still open**, per `docs/STATE.md`'s own next-iteration plan and backlog:

- **V22** — bf16 vs fp32 divergence, max 1.27e-02 vs 1e-2 limit. Root-cause script ready (`diag_v22.py`), unrun.
- **A full `--strict` run** — has not happened since the last ~15 commits landed (V57/V58/V60/V61/V62 additions, 4 inference.py bug fixes). Numbers in STATE.md are targeted `--only` re-runs, not suite-wide.
- **`adversarial-reviewer`'s 5 mediums + 7 lows** in `reviews/adversarial-1.md`, untriaged. H4 fixed (`e7e2feb`), needs its own V-check per STATE.md.
- **U-9 / V63** — proxy-OOD report. Explicitly "the one remaining SPEC gap with no plan yet." This is exactly the master prompt's Tier C ask — real, not redundant.
- **Two consecutive clean `--strict --fresh-clone` runs** (Definition of Done #2) — never done even once yet.
- **The deck** (U-5) — deliberately last, per SPEC's own one-day-plan note and STATE.md's explicit sequencing.

Round 2 work starts from here, not from a blank slate.

---

## 2. Part 0 finding: the "compute is nearly free" premise is refuted at the size that matters

The master prompt's Part 0 argument rests on D7's estimate: fixed startup 85–95% of wall-clock, compute ~5–15%. That estimate was made **at N=1 image, with torch not installed** (`docs/decisions.md` D7: "`import torch` + CUDA init — not measured — torch deliberately not installed... typically 1–3s each; treat as estimate").

`results/runtime_report.md`, generated this session's predecessor and committed at `bdf4547`, now has the real number, measured externally, at the actual scored set size (N=400, RTX 4060 Laptop GPU, bf16, batch 32):

| | D7 estimate (N=1, torch absent) | Measured (N=400, real GPU) |
|---|---|---|
| Total wall-clock | n/a | 22,514.6 ms median (n=5) |
| Fixed startup | 85–95% | **44.4%** (9,994 ms, linear fit) |
| Compute (marginal) | 5–15% | **~56%** (13,596 ms fit; forward-pass stage alone measured at 10,356.6 ms = 96.3% of the *instrumented* pipeline, excluding process/import overhead) |

**Consequence for each of Part 0's three proposed actions:**

1. **Scale the model.** Not "nearly free." Compute is already ~56% of wall-clock at N=400. NAFSR is also **memory-bandwidth-bound, not compute-bound** (`decisions.md` D21: 32.8% LayerNorm, 17.9% conv bias-add, 16.2% convolution — none of that is FLOP-bound work). A 10x parameter model does not map to "a few extra seconds"; it maps to a materially larger fraction of an already-substantial compute share, and D20 already found width≥64/blocks≥28 doesn't fit the local 8 GB card at all. **Verdict: scale the model, but justify the chosen size against measured wall-clock cost on the actual cloud GPU used for the sweep (§4), not against the refuted "compute is free" framing.**
2. **Train far longer (200k–400k iters).** Unaffected by the wall-clock finding (training cost is a one-time offline cost, not part of the scored inference run) — still worth doing if the GPU-hour budget allows. Budgeted in §5.
3. **Re-open 8x TTA.** The specific quantitative claim in the master prompt — "8x-ing compute is a ~30% wall-clock increase" — **is arithmetically wrong under the now-measured numbers.** At N=400: 8x the compute term is `9,994ms + 8 × 13,596ms ≈ 118,762ms`, a **~5.3x increase in total wall-clock**, not +30%. The 30% figure only holds if compute is ~5% of the total, which the measurement refutes. **Verdict: do not re-open 8x self-ensemble TTA on this basis.** D7's original rejection reasoning was built on a wrong number, but the corrected number reinstates the same conclusion, not the opposite one — TTA remains rejected, now for a *measured* reason instead of an estimated one. Log this as a decision entry (see §7).

**One caveat stated for the record, not acted on:** all of the above is measured on an RTX 4060 Laptop GPU. No H100 number exists anywhere in this repo (D7: "not measured"), and none will be produced by this plan either — HF Jobs does not offer H100 as a hardware flavor (§3 table). Any claim about H100 wall-clock fractions would be a projection, which CLAUDE.md's STYLE section and this plan's own §6 constraints forbid presenting as measurement. The directional argument ("H100 is faster, so fixed cost matters more") is plausible but unquantified — noted, not used to justify any budget item below.

---

## 3. Part 1: cloud compute, verified

**Credit balance:** confirmed via billing page screenshot this session — `Team-Ceciroleo67`, **$30.00 credits**, automatic recharge off, current period usage $0.00, **ends 2026-09-01**. This is a real, currently-unspent balance, not a projection.

**Mechanism:** confirmed via official docs (`huggingface.co/docs/hub/jobs-pricing`, `jobs-overview`, fetched this session):

- **HF Jobs** run arbitrary Docker/UV commands on Hub-managed hardware, billed **per minute**, requires a positive credit balance. Bill-to-org via `hf jobs run --namespace Team-Ceciroleo67 ...`.
- **Multi-hour jobs are supported.** Default timeout is 30 minutes but is explicitly configurable (`--timeout 3h`, `--timeout 1d`, etc.) — this is not an inference-only, short-lived mechanism. No documented hard ceiling was found; treat "set an explicit timeout ≤ remaining budget" as the operative safety rail, not a platform limit.
- **Storage is ephemeral, not persistent.** Every GPU hardware flavor's docs table labels its disk "Ephemeral Storage" — wiped when the job ends. **A checkpoint not pushed to a Hub repo during the run is lost**, confirmed independently by community docs on `push_to_hub` + `hub_strategy="every_save"`. This is the load-bearing operational fact for §5: the training script must push a checkpoint to a private Hub model repo on every save, not rely on job-local disk.

**GPU pricing** (official table, `jobs-pricing`, fetched 2026-08-16):

| Hardware | GPU mem | Hourly | $30 buys |
|---|---|---|---|
| 1x L4 | 24 GB | $0.80 | 37.5 hr |
| A10G large | 24 GB | $1.50 | 20.0 hr |
| L40S | 48 GB | $1.80 | 16.7 hr |
| A100 large | 80 GB | $2.50 | 12.0 hr |
| RTX PRO 6000 | 96 GB | $2.75 | 10.9 hr |
| H200 | 141 GB | $5.00 | 6.0 hr |

**No H100 flavor is offered by HF Jobs at all** — moot point for the "KLA scores on H100" framing in the master prompt; that fact is about KLA's *evaluation* infra, unrelated to what hardware is available to *train* on here.

**Choice for the sweep + long run: A100-large ($2.50/hr).** Given the model is memory-bandwidth-bound (D21), A100's ~2 TB/s HBM2e bandwidth is the relevant spec, not raw FLOPs; 80 GB headroom removes the D20-measured 8 GB ceiling that blocked width≥64/blocks≥28 locally. L40S is cheaper per hour but GDDR6 (~864 GB/s) — worse fit for a bandwidth-bound workload per unit dollar. This choice is a judgment call stated plainly, not a measurement; it will be checked against the sweep's own measured throughput in §4 and revised if the sweep contradicts it.

**Data route:**
- 919 MB dataset and the 400 `test_NoisyLR` inputs go to a **private** HF dataset repo under `Team-Ceciroleo67`. Privacy will be verified (repo settings show "Private", checked while logged out or via a second unauthenticated session) **before** any upload, and that verification recorded in `docs/decisions.md`.
- **F17 travels with the data unconditionally:** never train, fine-tune, or fit anything on `test_NoisyLR`, cloud or local. This is enforced the same way locally-trained code already enforces it (STATE.md standing prohibition) — the cloud training script must import the same `configs/split_val.txt`-driven split logic already in `src/dataset.py`, not a reimplementation.
- Checkpoints sync back via `hub_strategy="every_save"` to a private HF **model** repo; pulled down and copied into `weights/` locally, with an experiments.csv row appended per SPEC/CLAUDE.md §6 (git SHA, config, seed — `run_id` prefixed `cloud-` to distinguish from local rows).

**Blocked on:** an actual HF access token to (a) verify the org's Jobs feature is enabled (billing page alone doesn't distinguish Jobs-eligible from inference-only credit), (b) create the private dataset repo, (c) launch the sweep. Requested from the user; nothing in §4/§5 executes until it's provided.

---

## 4. Budget allocation

Total: **$30**, hard stop **2026-09-01**.

| Tier | % | $ | GPU-hr @ $2.50/hr (A100) | What |
|---|---|---|---|---|
| Pareto sweep | 20% | $6.00 | 2.4 hr | 4–6 configs, width/depth grid, short runs (~1,500–3,000 iters each — enough for a val-metric + measured throughput point, not convergence), plot the frontier |
| Long run | 60% | $18.00 | 7.2 hr | One run at the sweep-chosen config, checkpointed every N steps via `hub_strategy="every_save"`, resumable if interrupted |
| Reserve | 20% | $6.00 | 2.4 hr | Not pre-allocated to a specific item — tail-coverage fix re-run, larger-patch experiment, or recovery if a run dies mid-way |

**Iteration count for the long run is deliberately not pre-committed.** The only real iters/hour number that exists today is local: 20,000 iters at 388,225 params took 4,303.5 s (71m43s) on the RTX 4060 (`experiments.csv` row 1) — that's the *old* model at the *old* GPU, not informative about a scaled-up model on an A100. The sweep (§4 row 1) exists specifically to produce a real iters/hour figure on the actual training hardware before the long run's exact budget is spent — committing to "60k–150k iters" now would be exactly the kind of projection-as-measurement this plan's own §6 forbids. The sweep's measured throughput, divided into the remaining 7.2 GPU-hours, sets the number.

**Kept local (per master prompt, confirmed still correct):** U-Net baseline (already done, §1), evaluation (`scripts/evaluate.py`), qualitative figures (already done, §1), runtime measurement (already done, §1, and must stay on the RTX 4060 — it's the throughput-scoring proxy device already used consistently in `runtime_report.md`; an A100 number would not answer the question runtime_report.md exists to answer).

---

## 5. Tier B — quality (cloud-budget items, from §4)

0. **FiLM noise-conditioning + uncertainty head (blocking, local, zero cloud cost) — MUST land
   and pass local verification before item 1 dispatches anything to HF Jobs.** Per the Round-2
   differentiation plan (`.claude/plans/as-of-now-whatever-steady-lemur.md`, user-approved
   2026-08-16): a small `NoiseEstimator` head produces FiLM scale/shift parameters conditioning
   the NAFBlock stack (targets F7, the OOD/noise-generalisation axis SPEC explicitly tests),
   plus an optional second output channel predicting per-pixel uncertainty via a heteroscedastic
   NLL term. Additive to `_DEFAULTS`, default-disabled, so every existing checkpoint (including
   `weights/best.pt`) still loads unchanged under V35's `strict=True`. Gate before spending any
   cloud budget: overfit-2-pairs >~40 dB, V07–V12/V24/V61 green against the new `build_model`
   config, and a short local training run (RTX 4060) showing val PSNR/SSIM/LPIPS is not worse
   than the current shipped baseline. **The sweep and long run below train this architecture,
   not the plain NAFSR the six `configs/sweep_*.yaml` files currently specify** — those configs
   get a FiLM toggle added (or a parallel `sweep_*_film.yaml` set) once 0 is verified.
1. **Pareto sweep.** 4–6 (width, num_blocks) configs, chosen to bracket the current 48×16 point and the D20-known ceiling (width≥64 + blocks≥28 didn't fit 8 GB locally — now testable on 80 GB). Measure: val PSNR/SSIM/LPIPS at a short, fixed iter budget, plus measured ms/iter on A100. Plot the frontier, pick the operating point, put the plot in the deck.
2. **One long run** at the chosen config, iters set by measured throughput (§4). Checkpoint every save via `hub_strategy="every_save"`.
3. **Larger training patches (96 or 128)** for the long run, budget permitting — gated on the MS-SSIM ≥161px constraint already logged in STATE.md's "Do NOT retry" section (single-scale SSIM is already the fallback in use, so this is compatible, not blocked).
4. **Tail-coverage fix (F1).** Synthetic max 1.7177 vs real train 2.0735 vs test 2.158 — measured gap already in `dataset_findings.md`. Widen noise randomisation until synthetic ≥ real train max, re-measure validation metrics, add the V-check asserting the coverage relationship. **This is a local, no-GPU-budget task** (touches `src/degrade.py`, `data-pipeline` owner) — do first, before the sweep, since it changes what the sweep and long run should even train on.
5. **LPIPS loss term, gated late-training, adopt only if cost <0.1 dB PSNR.** If adopted: rewrite the README's external-resources section in the same commit (AlexNet features now contribute gradient, not just evaluation) — this is a hard requirement from the master prompt and matches CLAUDE.md's "no silent scope change" spirit.

## 6. Tier C — differentiation (local, no cloud budget needed)

This is **U-9/V63**, already flagged in STATE.md as the one open SPEC gap with no plan — the master prompt's Tier C ask is genuinely new work, not a duplicate.

- Synthesize a semiconductor-proxy validation set (line/space arrays, contact-hole grids, dense periodic patterns, edge-heavy geometry).
- Degrade with the already-measured degradation model (recovered kernel, D1; 3-parameter shot-noise model, D12) — reuse `src/degrade.py` as-is, do not refit it.
- Report PSNR/SSIM/LPIPS on this set as a separate column in every results table, alongside in-distribution numbers.
- Add `V63` per the auditor's proposed spec in STATE.md: `metrics_summary.md` needs a proxy-OOD heading, membership from a committed list with empty train intersection.
- **No GPU-hour cost** — this is local synthesis + local eval on the existing checkpoint(s). Owner: `dataset-forensics` for the proxy-set synthesis, `loss-metrics` for wiring it into `evaluate.py`.

---

## 7. Constraints (restated, unchanged from the master prompt, cross-checked against this repo's actual mechanisms)

- Every V-check green at the end; standing authorisation (STATE.md, restated 2026-08-15) covers stricter checks, new V-checks, package/venv/training/Release/commit — **not** cloud spend or third-party dataset upload, which is why §3 was gated on explicit confirmation before acting.
- Tag `v0.1-submittable` before any long run starts (STATE.md already has this in its own remaining-plan; folding it in here rather than duplicating).
- No number enters a doc/README/deck unless a repo script produced it and can re-run it — this plan follows that rule itself: every number in §2 and §3 above is either read from a committed file (`runtime_report.md`, `experiments.csv`) or fetched from official HF docs this session, none guessed.
- Every timing number labelled with its device. The A100 sweep/long-run numbers, once they exist, get their own row — they do not get merged into or compared directly against the RTX 4060 `runtime_report.md` numbers, which remain the authoritative throughput-scoring proxy.
- MORNING_REPORT.md updated continuously including spend to date, once execution starts.

## 8. Immediate next step — updated 2026-08-16, post-merge

Steps 1–3 below are **done** (`docs/PLAN_CLOUD.md` has the full execution log): token received,
Jobs verified live under `Team-Ceciroleo67` (real smoke job completed), private dataset repo
(`kla-ps01-data`) and checkpoint repo (`kla-ps01-checkpoints`) created and privacy-verified
logged-out, tail-coverage fix (§5 item 4) landed pre-merge (`dd61ef1`, D43). A teammate's
independent line of work was also discovered and reconciled this
session (`docs/MERGE_ANALYSIS.md`, D49) — this session's checkpoint ships, verifier at 61/65
(4 known/expected FAILs).

**Current blocking step, per the user's explicit instruction: §5 item 0 (FiLM + uncertainty)
must land and pass local verification before any HF Job is dispatched.** Nothing in §4's budget
executes until that gate is green. Once it is:
1. Add the FiLM toggle to `configs/sweep_a_w32n16.yaml` … `sweep_f_w96n32.yaml` (or a parallel
   `_film` set).
2. Tag `v0.1-submittable` (§7, still outstanding).
3. Dispatch the Pareto sweep (§5 item 1).
