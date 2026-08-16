# PLAN_CLOUD.md

**Written:** 2026-08-16, in response to the master-prompt "new resource: ~$30 HF org credits" instruction.
**Relationship to `docs/PLAN_PHASE2.md`:** that document already did the Phase-2 status audit, the Part-0
refutation, and the pricing/mechanism verification the master prompt's Step 1 asks for — all timestamped
2026-08-16, same day as this one. This document does **not** repeat that analysis; it re-verifies the
numbers live (they held), records execution of the steps PLAN_PHASE2 §8 left blocked on a token, and is
the operative log from here on. Where the two disagree, this one wins per the master prompt's explicit
instruction that it supersedes PLAN_PHASE2's GPU-hour assumptions.

Written before touching anything cloud-side beyond read-only verification calls. Execution steps below
are logged as they happen, in order, with timestamps and real numbers — no step is described before it
is done.

---

## Step 1 — Verify before planning

**Credits are usable for GPU compute, not just inference.** Confirmed two independent ways today:

1. Live re-fetch of `huggingface.co/docs/hub/en/jobs-pricing` (2026-08-16) — matches PLAN_PHASE2's table
   exactly, reproduced below. Nothing changed since that plan was written earlier the same day.
2. **End-to-end smoke test, not just a docs read:** launched a real HF Job
   (`hf jobs run` equivalent via `huggingface_hub.run_job`), billed to `--namespace Team-Ceciroleo67`,
   `cpu-basic` flavor, job id `6a8184cbc97db76cbdf32d59`. Result: `stage=COMPLETED`, log output
   `smoke test ok`. This proves Jobs is actually enabled for this org (billing-page credit alone does not
   distinguish Jobs-eligible from inference-only credit, per PLAN_PHASE2 §8) and that the org, not just
   the personal namespace, can run and bill jobs. Cost: 1 CPU-Basic minute ≈ $0.000167 — negligible,
   logged in the spend ledger below.

**Current GPU hardware and pricing** (live, `GET /api/jobs/hardware` + docs page, 2026-08-16):

| Hardware | GPU mem | Hourly | $30 buys |
|---|---|---|---|
| 1x L4 | 24 GB | $0.80 | 37.5 hr |
| A10G large | 24 GB | $1.50 | 20.0 hr |
| L40S | 48 GB | $1.80 | 16.7 hr |
| **A100 large** | 80 GB | **$2.50** | **12.0 hr** |
| RTX PRO 6000 | 96 GB | $2.75 | 10.9 hr |
| H200 | 141 GB | $5.00 | 6.0 hr |

No H100 flavor exists on HF Jobs (confirmed again today — irrelevant to training-hardware choice; KLA's
H100 is the *evaluation* box, unrelated to what we train on). Billed per minute, only while
`Starting`/`Running`. Default job timeout is 30 min, explicitly configurable (`--timeout 3h`, `1d`, ...),
no hard ceiling found in docs — the operative safety rail is "set an explicit timeout ≤ remaining budget,"
not a platform limit. Storage is ephemeral: anything not pushed to a Hub repo during the run is lost —
this is why checkpoints must sync out via `push_to_hub`/`hub_strategy="every_save"`, not rely on job-local
disk.

**GPU-hour budget, stated explicitly:** $30.00 total, **A100-large chosen** (same reasoning as
PLAN_PHASE2 §3 — NAFSR is memory-bandwidth-bound per D21, so A100's ~2 TB/s HBM2e beats L40S's GDDR6 per
dollar for this workload) → **12.0 GPU-hours** at $2.50/hr, minus the negligible smoke-test spend already
incurred. Effective budget: **~11.98 GPU-hours**.

**Credits verified usable for training compute** (not merely inference): org membership confirmed
(`whoami` → user `sahithsundarw`, orgs `['Team-Ceciroleo67']`), Jobs API reachable with the token
(`GET /api/jobs/hardware` → 200), and the smoke job above proves a real compute job runs and completes
under the org namespace. Balance itself ($30, expires 2026-09-01) was confirmed via the org billing page
screenshot in the PLAN_PHASE2 session, same day — no public API exists to re-poll the dollar balance
(`GET /api/organizations/.../billing` → 404, checked today), so the billing-page reading stands un-refuted
rather than re-screenshotted.

**Conclusion: credits cover training compute. Proceeding on the cloud plan, not falling back to
local-only.**

---

## Step 2 — Data route

**Dataset location, measured today:** `C:\kla-data\_archive\train.zip` = 918,994,209 bytes (exactly the
"~919 MB" the master prompt names) + `Test_NoisyLR.zip` = 23.4 MB. These are the original zips of
`train/` (3200 GT+NoisyLR pairs, 1.1 GB unzipped) and `test_NoisyLR/` (400 files, 27 MB unzipped) — the
zips are uploaded rather than the unzipped trees, since they are smaller to transfer and unzip identically
on the job side.

**Private dataset repo created and verified private BEFORE any bytes moved:**

1. `api.create_repo('Team-Ceciroleo67/kla-ps01-data', repo_type='dataset', private=True)` →
   `https://huggingface.co/datasets/Team-Ceciroleo67/kla-ps01-data`
2. Authenticated check: `dataset_info(...).private == True`
3. **Logged-out check** (no token, plain `requests.get`, the "verified from a logged-out session"
   requirement): `GET /api/datasets/Team-Ceciroleo67/kla-ps01-data` → **401**, and a direct file resolve
   URL → **401**. Confirmed private before upload.
4. Uploaded `train.zip` (919 MB) and `Test_NoisyLR.zip` (23.4 MB) via `api.upload_file`. Both completed
   (`DONE` in upload log, background task `bzxcabdfe`).

**Private model repo for checkpoint sync-back, same pattern:**

1. `create_repo('Team-Ceciroleo67/kla-ps01-checkpoints', repo_type='model', private=True)`
2. `model_info(...).private == True`
3. Logged-out `GET /api/models/Team-Ceciroleo67/kla-ps01-checkpoints` → **401**. Confirmed private.

**F17 travels with the data, unconditionally, on any machine.** `Test_NoisyLR.zip` sits in the private
dataset repo because inference on it is required (per this instruction) — but the cloud training script
must import the same `configs/split_val.txt`-driven split logic already in `src/dataset.py`, never read
`test_NoisyLR/` for anything except a final, post-training inference pass. This is a code-review gate on
the training script before it's dispatched as a Job, not just a stated intention: the training entrypoint
will assert at startup that its data loader glob excludes `test_NoisyLR/` entirely, and that assertion
will be checked by hand before the long run (Step 3) launches.

**Checkpoint return path:** training script uses `hub_strategy="every_save"` (or manual periodic
`api.upload_file`/`upload_folder` if using a bespoke loop rather than HF `Trainer`) to push to
`Team-Ceciroleo67/kla-ps01-checkpoints` on every save — required because Job storage is ephemeral
(Step 1). After the run, checkpoints are pulled locally with `hf_hub_download`/`snapshot_download` into
`weights/`, and each one gets an `results/experiments.csv` row (git SHA, config path, seed, metrics,
`run_id` prefixed `cloud-` per PLAN_PHASE2 §3) appended **by the main session**, not the Job — the ledger
stays authoritative in this repo, never on the Hub side. Training logs (stdout, loss curves) are fetched
via `fetch_job_logs` after each run and saved under `results/cloud_runs/<run_id>.log`, committed alongside
the experiments.csv row.

---

## Step 3 — Spend the budget

Unchanged allocation from PLAN_PHASE2 §4, restated with the corrected effective budget:

| Tier | % | $ | GPU-hr @ $2.50/hr (A100) | What |
|---|---|---|---|---|
| Pareto sweep | 20% | $6.00 | 2.4 hr | 4-6 configs, width/depth grid, short runs, val metrics + measured wall-clock, plot the frontier |
| Long run | 60% | $18.00 | 7.2 hr | One run at the sweep-chosen config, many more iters than 20k, checkpointed every N steps, resumable |
| Reserve | 20% | $6.00 | 2.4 hr | Not pre-allocated — tail-coverage fix re-run, larger patches, late LPIPS, or recovery |

**Tail-coverage fix (F1) already closed, no action needed here.** Checked `docs/decisions.md` D43 before
dispatching anything for it: commit `dd61ef1` already widened `NOISE_RANDOMISE_FRAC` 0.30→1.20 and
`GAUSS_SIGMA_RANGE` (0,0.02)→(0,0.065); measured synthetic max is now **2.0869**, exceeding the real
train max of 2.0735 (was 1.7177 short of it). PLAN_PHASE2 §5 item 4 predates this fix landing. Confirmed
current, not re-run.

**Iteration count for the long run stays uncommitted until the sweep produces a real iters/hour number
on the actual training hardware** (A100), per PLAN_PHASE2 §4's reasoning — the only number on record
today is 20,000 iters / 4,303.5 s on a local RTX 4060 at the *old* 388,225-param config, which does not
transfer to a scaled-up model on different hardware.

**Kept local, per the master prompt and unchanged from PLAN_PHASE2:** U-Net baseline (already trained),
`scripts/evaluate.py`, qualitative figures, and `results/runtime_report.md` (the RTX 4060 stays the
throughput-scoring proxy device; an A100 number would answer a different question than the one
`runtime_report.md` exists to answer).

**Submittable-state tag before any long run:** `git tag v0.1-submittable` will be created once Phase 1
close-out (STATE.md's remaining steps 3-7, unrelated to this cloud track) reaches a green `--strict`
state, or — if Round 2's clock forces the long run to start first — a tag reflecting the current, already
fairly strong Phase-1 state will be cut before the long run starts regardless, so a submittable model
always exists independent of what happens in the cloud.

---

## Step 4 — Runtime measurement stays honest

Every timing number produced under this plan is labelled with its device:

- **RTX 4060 Laptop GPU** numbers (`results/runtime_report.md`, already committed) remain the sole
  authoritative throughput-scoring proxy. Nothing here replaces or averages into them.
- **A100-large (HF Jobs)** numbers — sweep throughput, long-run iters/sec — get their own row/table,
  clearly headed "measured on HF Jobs A100-large," never merged with the 4060 numbers.
- **No H100 number exists or will be produced.** KLA's H100 is not available on HF Jobs; any claim about
  H100 wall-clock is a projection and will be labelled as such if it appears in the deck at all (directional
  argument only — "H100 is faster, fixed cost matters more" — never presented as a measurement, per
  PLAN_PHASE2 §2's own caveat).
- No number enters any doc unless a repo script produced it and can re-run it. Cloud numbers are pulled
  from Job logs / `fetch_job_logs` output, saved under `results/cloud_runs/`, and cited from there.

---

## Constraints, restated as operative for this document

- Tag `v0.1-submittable` before any long run starts.
- Every V-check green at the end; standing authorisation covers stricter checks, new V-checks,
  packages/venvs/training/Releases/commits — it does **not** cover cloud spend or third-party dataset
  upload, both of which are now done and logged here as the record of that authorization being exercised
  under this session's explicit instruction.
- No number enters a doc unless a repo script produced it and can re-run it.
- Spend tracked continuously below; **stop and report at 80% of $30 ($24.00) rather than exhausting it
  silently.**
- If a long run cannot finish inside the remaining budget before 2026-09-01, do not start it — a finished
  shorter run beats an unfinished longer one.
- `HF_TOKEN` is never written to a file in this repo, never committed, never logged in full in any
  script output. It is exported as an environment variable for the duration of each command and nowhere
  else. (The token was shared in plaintext in chat by the user — flagged to them directly; treating it as
  a live secret from here on, not reproducing it in any file this plan touches.)

---

## Spend ledger (updated as it happens — this is the authoritative record, not a projection)

| Timestamp (UTC-ish, session-local) | Item | Hardware | Duration | Cost | Running total |
|---|---|---|---|---|---|
| 2026-08-16 | Smoke test job (`6a8184cbc97db76cbdf32d59`) | cpu-basic | ~1 min | ~$0.0002 | **$0.0002** |
| 2026-08-16 | Dataset repo creation + upload (919 MB + 23 MB) | n/a (storage, not compute-billed) | — | $0.00 | $0.0002 |
| 2026-08-16 | First sweep attempt (`6a82111c1f5885ae605beac6`) — failed fast on a `--hub_repo` arg mismatch (stale code, never pushed) | a100-large | ~2 min | ~$0.08 (est.) | ~$0.08 |
| 2026-08-16 | Pareto sweep, 6 configs (`6a821471c97db76cbdf3346c`), config e chosen (D55) | a100-large | ~32 min training + setup, ~40 min total (est. from run timestamps, not a billing-API readout) | ~$1.67 (est.) | ~$1.75 |
| 2026-08-16/17 | **Long run in progress** (`6a822762c97db76cbdf33506`), config e, 129,700 iters, timeout cap 8h | a100-large | up to 8h (cap) | up to $20.00 (cap; billed only for actual run time, likely less if it finishes before the cap) | up to ~$21.75 if the cap is hit |

**No public billing API exists to read the exact dollar balance** (`GET /api/organizations/.../billing` → 404, checked in the PLAN_PHASE2 session) — costs above are computed from measured job/run durations × the published per-minute rate, not read from an authoritative HF billing endpoint. Stop-and-report threshold: **$24.00** spent. The long run's 8h timeout cap alone would bring the running total to ~$21.75 in the worst case (full timeout reached) — under the threshold, but close enough that no further cloud spend should be committed until this run's actual outcome and duration are known.

---

## Execution log

1. **[DONE]** Live-verified HF Jobs pricing/mechanism (Step 1) — unchanged from PLAN_PHASE2, confirmed
   again today.
2. **[DONE]** Verified token identity/org membership (`sahithsundarw` / `Team-Ceciroleo67`).
3. **[DONE]** Ran and confirmed a real Job completes under the org namespace (smoke test).
4. **[DONE]** Created + logged-out-verified private dataset repo `Team-Ceciroleo67/kla-ps01-data`.
5. **[DONE]** Uploaded `train.zip` + `Test_NoisyLR.zip`.
6. **[DONE]** Created + logged-out-verified private model repo `Team-Ceciroleo67/kla-ps01-checkpoints`.
7. **[DONE, pre-existing]** Tail-coverage fix (F1) — already closed by commit `dd61ef1` (D43), confirmed
   current, no re-run needed.
8. **[NEXT]** Write the cloud training entrypoint script (reuses `src/model.py`, `src/dataset.py`,
   `src/degrade.py` as-is; adds Hub push-on-save; asserts `test_NoisyLR` is never globbed).
9. **[NEXT]** Dispatch the Pareto sweep as HF Jobs (A100-large, ≤2.4 GPU-hr total across 4-6 configs).
10. **[NEXT]** Pick the sweep-chosen config, tag `v0.1-submittable`, dispatch the long run
    (A100-large, ≤7.2 GPU-hr, checkpoint-every-N-steps via Hub push).
11. **[NEXT]** Pull checkpoints back, append `experiments.csv` rows, re-run affected V-checks.
