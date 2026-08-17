# BLOCKERS

Things that could not be resolved, and why. Per `CLAUDE.md` Prime Directive 1, a V-check that
seems wrong is logged here — never edited.

**Status at bootstrap: no check was left unimplemented.** All 53 (V00 + V01-V52) have a
function in `scripts/verify_all.py`. The entries below are structural gaps in the
*instructions*, not in the verifier.

---

## B1 — LOOP_PROMPT §5 defines TEN agents, not nine

The BOOTSTRAP instruction asked for "the nine subagent definitions". `LOOP_PROMPT.md` §5
actually specifies **ten**:

`dataset-forensics`, `inference-engineer`, `adversarial-reviewer`, `ml-skeptic`,
`cleanroom-tester`, `requirements-auditor`, `perf-analyst`, `model-core`, `trainer`,
`docs-scribe`.

**Resolution:** all ten written. Not a blocker; recorded so the count discrepancy is not
mistaken for a missing file later.

## B2 — `CLAUDE.md`'s ownership map references two agents §5 never defines

The FILE OWNERSHIP MAP in `CLAUDE.md` assigns files to `data-pipeline` and `loss-metrics`:

| Owner | Files | Defined in §5? |
|---|---|---|
| `data-pipeline` | `src/dataset.py`, `src/degrade.py`, `configs/split_val.txt` | **NO** |
| `loss-metrics` | `src/losses.py`, `src/metrics.py`, `scripts/evaluate.py`, `scripts/make_baselines.py` | **NO** |
| `throughput-optimizer` | `scripts/benchmark_runtime.py`, `results/runtime_report.md` | as `perf-analyst` |

Without definitions for the first two, no agent may write the degradation simulator or the
metrics — and the ITERATION loop assigns work strictly by the ownership map, so those files
would be unassignable.

**Resolution:** `data-pipeline` and `loss-metrics` were written, clearly marked at the top as
additions beyond §5. `throughput-optimizer` was **not** created — `perf-analyst` covers that
role and owns those files; treat the two names as the same agent.

**Human decision needed:** confirm these two additions, or delete them and instead extend §5.

## B3 — `docs/VERIFIER_SHA256` pins a file that BOOTSTRAP itself creates

`CLAUDE.md` PD1 says `scripts/verify_all.py` is hash-pinned in `docs/VERIFIER_SHA256`. That
file was created earlier (to pin the contract) at a point when `verify_all.py` did not exist,
so its verifier line was a documented placeholder.

**Resolution:** now that `verify_all.py` exists, its sha256 is pinned and V00 checks it. V00
passes when the hash matches, or when a changed hash appears in `docs/decisions.md` — which
is the mechanism that permits authorised strengthening.

**Known limitation:** the pin is over working-tree bytes. This repo is on Windows with git
converting LF→CRLF on checkout, so a hash computed in a fresh clone may differ from the pin
without any tampering. `docs/VERIFIER_SHA256` documents the LF-normalised comparison. If V00
fails in a clean-room run for this reason, that is a real finding to fix by adding
`.gitattributes` with `* text eol=lf`, not by loosening V00.

## B4 — LOOP_PROMPT §2 Step 3 reserves a slot for work that is already done

Step 3 says: *"Reserve one parallel slot for `dataset-forensics` while any of U1-U9 in SPEC
§2.2 remain unanswered."*

**All nine are answered** with numeric evidence (`docs/SPEC_VCHECK_MAP.md`, V52 readiness
table). The reservation is therefore unnecessary from iteration 1 and that slot should go to
Tier 0 work instead. Recorded so a future iteration does not burn a parallel slot re-deriving
settled facts.

## B5 — Several Tier 2/3/4 checks can only report `not implemented yet` for now

Not a defect. Per LOOP_PROMPT B3, a check whose subject code does not exist must FAIL with
`detail="not implemented yet"` rather than SKIP. Checks in this state at iteration 0:
V25-V29, V33-V35 (need training/model/eval), V37/V39/V43 (need a runtime report), V45
(needs an experiment ledger), V47-V49 (need samples, metrics and qualitative output).

They will turn green as their subjects are built. None is blocked in the sense of being
impossible.

## B6 — `results/restored_test_outputs/` cannot be committed as-is

SPEC F12 requires this folder to contain **real model outputs**, and V13 asserts it is
non-empty. The outputs will be 400 `.npy` files at 65,664 B ≈ **105 MB total**, which exceeds
GitHub's limit and is caught by the `*.npy` rule in `.gitignore`.

**Needs a human decision before submission:** Git LFS, or external hosting with a sha256 in
`weights/README.md`, or committing a reduced-precision/subset artifact (which would not
satisfy "actual model outputs"). Currently a `.gitkeep` placeholder only, so V13 fails
honestly rather than appearing to pass.

**UPDATE, iteration 1:** Git LFS is now ruled out by human instruction — unresolved LFS
pointer stubs on a fresh clone are a known way to fail V06, and V06's own text names
"not an LFS pointer stub" as a failure mode. The chosen mechanism is a single
`np.savez_compressed` archive if it measures under ~40 MB, else external hosting with a
published sha256 verified from a logged-out session. See `docs/decisions.md` D17. **This
is not yet resolved** — see B9 below for the constraint it collides with.

## B7 — V47 and V51 were mutually exclusive as implemented (RESOLVED, human-authorised)

V51 banned **every** tracked `.npy`. V47 requires
`python inference.py --input_dir sample_inputs --output_dir /tmp/o` to complete
**from a clean clone**, which requires `.npy` files to be *in* that clone. SPEC §12 lists
`sample_inputs/` as a repo item for the same reason. The two checks could therefore never
both be green, and the Definition of Done was unreachable.

This was not a judgement call the agent made on its own: the human explicitly instructed
"copy 4-6 real .npy files from C:\kla-data\test_NoisyLR, ~400 KB total, commit directly.
Unblocks V47" — the same human-issued-amendment mechanism as D6 and D10.

**Resolution:** a narrow, bounded exemption for `sample_inputs/*.npy` (≤8 files, ≤512 KB;
actual 6 files / 393,984 B) plus four **new** assertions that make V51 net stricter — the
blob-extension ban widened from 4 to 20 extensions, a dataset-directory-token ban, a 5 MB
per-file cap and a 25 MB total-tree cap. The last two catch a dataset dump under *any*
extension, which the previous extension blacklist provably could not. Recorded in
`docs/decisions.md` D15 with the new verifier digest; `docs/VERIFIER_SHA256` re-pinned.

**Flagged honestly for the human:** the exemption is, in isolation, a loosening with
respect to those six paths. The four new assertions are unambiguous strengthenings. If the
human prefers the alternative reading — leave V51 red and treat the conflict as permanent —
revert commit and this entry stands as the record of why.

## B8 — `pip install lpips` silently downgrades torch to a CPU-only build

Measured, iteration 1. `torch==2.11.0+cu128` was installed from
`https://download.pytorch.org/whl/cu128` and verified (`torch.cuda.is_available()` True,
RTX 4060 Laptop GPU). A subsequent
`pip install scikit-image lpips pyyaml pytorch-msssim` resolved `lpips`'s torch/torchvision
dependency **from PyPI**, replacing the CUDA build with `torch==2.13.0+cpu` /
`torchvision==0.28.0+cpu`. After that, `torch.version.cuda` was `None` and
`torch.cuda.is_available()` was `False`.

**Why this matters beyond the dev box:** V04 installs into a fresh venv with only
`pip install -r requirements.txt`. If `requirements.txt` does not force the PyTorch index,
a clean-room install produces a **CPU-only torch** — the run still "passes" V04 while
KLA's H100 sits idle, and the throughput score collapses with no error message anywhere.
This is exactly the class of silent failure V04 exists to catch, and it would not have been
caught by reading the file.

**Resolution:** `requirements.txt` pins the `+cu128` local version with an explicit
`--extra-index-url` directive (see the file's header comment for the full mechanism).

**Verified end to end, 2026-08-15** — but not by V04 on this particular dev machine, which is
macOS/arm64 and has no `+cu128` wheel to install at all (that platform gap is itself the correct
"loud failure" B8 describes, just triggered by the wrong platform rather than a silent
downgrade). Independently verified on real Linux (`python:3.12-slim` Docker container): a fresh
clone + fresh venv + bare `pip install -r requirements.txt` (no extra flags) installs exactly
`torch==2.11.0+cu128` / `torchvision==0.26.0+cu128` with `torch.version.cuda == '12.8'`, and
`python3 scripts/verify_all.py --strict --fresh-clone --only V04` / `--only V46` both PASS in
that environment. Full evidence and commands logged in `docs/STATE.md` under "V04/V46
fresh-clone dry run". The silent-CPU-downgrade failure mode B8 exists to catch has not
reappeared since the index pin was added.

## B9 — RESOLVED 2026-08-15 by standing authorisation: GitHub Release + sha256

**Resolution.** The human's standing authorisation pre-approves GitHub Releases. The 400
restored test outputs ship as a **Release asset** with a published sha256, and
`results/restored_test_outputs/` carries a committed manifest with per-file hashes so the
folder is non-empty and independently verifiable.

This needs **no contract change**: it reuses exactly the mechanism V06 already permits for
weights. Option 2 below — a second human-authorised V51 amendment to admit a ~40 MB `.npz` —
was explicitly **rejected**, because it would have gutted the 5 MB / 25 MB size caps added
one commit earlier, and "loosen the check I just tightened, because it blocks me" is the
exact reasoning pattern the authorisation names as a stop signal.

Recorded honestly: the committed folder holds a **verified pointer and manifest, not the raw
output bytes**. That is stated in the folder's README rather than glossed over, so a reviewer
is never misled about what is in the repository.

Original analysis retained below.

---

## B9 (original) — the D17 `.npz` route collides with the V51 strengthening from B7

`docs/decisions.md` D17 selects a single `np.savez_compressed` archive as the delivery
mechanism for `results/restored_test_outputs/`. The V51 rewrite in B7 bans `.npz` outright
and caps any tracked file at 5 MB. A ~40 MB `.npz` therefore cannot be committed without
**another** human-authorised V51 amendment.

The agent will not resolve this itself in either direction: loosening V51 a second time to
admit a 40 MB blob is precisely the pattern Prime Directive 1 forbids, and it would gut the
size caps that were just added. **Human decision required, two options:**

1. External hosting with a published sha256 and a logged-out-verified link — needs no
   contract change at all, and is the route the current V51 already permits.
2. A second human-authorised V51 amendment carving out exactly
   `results/restored_test_outputs/*.npz` with its own byte cap.

Option 1 is the recommendation: it requires no further weakening of a check that exists to
stop dataset blobs entering the repo. Blocked pending the human, and blocking V13.

## B10 — No third split: degradation fitting, checkpoint selection and the headline numbers
all share `split_val.txt`

Found by the three-agent audit preceding this iteration's requirements pass
(`docs/REQUIREMENTS_MATRIX.md`, Validation And Reporting #1). Three separate uses of the same
400-image validation split, each individually reasonable, that compound into an optimistic bias
nobody has measured:

1. **`scripts/fit_degradation.py:206-212`** draws a random 200-of-3200 sample of *all* train GT
   filenames, with no split filter, to fit the recovered downsample kernel (D1) and the noise
   parameters `a=0.011253`, `v=0.015745` (D12). On average ~25 of those 200 are val-split
   images. Every synthetic training pair is generated from a degradation model that was
   partly fit on data the model is later validated against.
2. **Checkpoint selection** (`train.py:548-556`, `configs/final.yaml:51 save_best_on: psnr`)
   picks the best EMA iterate using **only the first 100 of the 400 val images**
   (`--val_limit 100`).
3. **The headline numbers** (`results/metrics_summary.md`, 28.7865 dB / 0.78287 / 0.25324) are
   then reported on the **full 400-image split** — the same split, 100 of which drove (2).

There is no third split (train / val-for-selection / held-out-for-reporting) anywhere in the
pipeline. None of the three uses is a bug in isolation — (1) is a small, plausible-sounding
convenience; (2) is standard checkpoint selection; (3) is the correct thing to report *if*
selection hadn't touched the same pool. Together, the reported PSNR is optimistically biased
by an unmeasured amount, and the degradation simulator was calibrated with partial knowledge of
its own test set.

**Not resolved this iteration.** Too structural to fix inside a 3-day Round 1 window — it would
mean re-splitting, re-fitting the degradation model, and retraining, none of which can be
verified in time without risking the submission deadline. Recorded here per Prime Directive 3
(never fabricate; if something can't be verified clean, say so) rather than silently shipped.
Stated as an explicit caveat in the README and on the deck's limitations slide. **No V-check
currently catches this** and none is proposed for this window — a real fix (three-way split)
is Round 2 work, not a check to bolt onto the existing split structure. Human should decide
whether this is worth a stated limitation only, or worth delaying submission to fix; the
recommendation is: state it, ship on time, fix it properly in Round 2 if selected.

## B11 — V24 (cross-process determinism) is genuinely flaky, pre-existing, not from the V22 fix

**2026-08-17 update: the same flake class now confirmed also hitting V21.** A full `--strict`
run reported `V21 FAIL: repeat runs are not byte-identical` for the first time. `check_V21`
(`scripts/verify_all.py`) shares V24's exact mechanism — it launches `inference.py` as
separate subprocesses and compares outputs, the same cross-process-boundary path where
`cudnn.benchmark=True` algorithm selection can tie-break differently. Re-ran `--only V21`
three times immediately after: PASS all three, confirming this is the same known intermittent
class (B11), not a new regression from any edit this session made. Broadening this entry's
scope to cover both V21 and V24 rather than opening a second blocker for the identical cause.

**2026-08-18 update: a related but distinct flake, V65, observed once during the Phase 3
checkpoint promotion's post-promotion `--strict` run.** `V65 FAIL: 256->512 batch exited 124:
timeout after 300s`. Different mechanism from V21/V24 (this is a genuine subprocess timeout
under system load, not a numerical cross-process divergence), but the same category of
issue: a real, environment/load-sensitive intermittent failure, not a regression from the
checkpoint promotion. Confirmed directly: re-ran `--only V65` in isolation immediately after
(no other processes competing for the GPU) and it passed cleanly — "real 256->512 batch of 8:
N-out, float32, (512,512), finite, [0,1] all confirmed; OOM-recovery...". Architecture is
byte-identical between the promoted checkpoint and its predecessor (same param count), so
there is no mechanism by which this check's behaviour should depend on which weights are
loaded — this was system-load contention during a session that had run many other GPU jobs
that same day, not a checkpoint-specific effect.

**A third occurrence, same night: `V25 FAIL: train.py --overfit 2 produced no parsable
report (rc=124)`** on a SECOND immediate full-suite re-run (done specifically to get a clean
confirmation after the V65 flake above). Re-ran `--only V25` alone immediately after: PASS
cleanly ("overfit 2 pairs reached 43.7827 dB at iter 6000, gate 40.0 dB") — same pattern
exactly. **Generalising the finding rather than chasing each instance individually**: this
machine, after a very long single-session day of repeated GPU-heavy work (training runs,
benchmarks, dozens of evaluation passes), exhibits intermittent subprocess timeouts (`rc=124`)
on WHICHEVER GPU-heavy check happens to run at a moment of contention — V65 the first time,
V25 the second, neither a repeat, both clean in isolation. This is environmental load, not a
per-check or per-checkpoint defect, and not investigated further to root cause tonight given
the time budget — every individual occurrence has been directly confirmed harmless by
isolated re-test, which is the standard this project already applies to V21/V24's flakes.
Recorded here, not hidden, and not used to justify touching any check's tolerance.

Found by `inference-engineer` while fixing V22 (`docs/decisions.md` D42), confirmed
independently by the main session: `py -3.12 scripts/verify_all.py --only V24` fails roughly
half the time (measured: PASS, FAIL, PASS, FAIL over 4 consecutive runs). **This is present on
the unpatched, pre-V22-fix model too** — it is not a regression introduced by anything this
iteration changed.

**Root cause:** `inference.py`'s `tune_backends()` sets `torch.backends.cudnn.benchmark = True`
unconditionally (a Tier-0 "free optimization," V40). For at least one convolution shape in the
16-block NAFSR stack, two cuDNN candidate algorithms benchmark close enough to tie, so which one
wins — and therefore the exact floating-point result — depends on process-to-process scheduling
noise, not on the seed. `V22`'s fix incidentally exercises this same class of shape (a 1×1 conv
over a `(B,C,1,1)` tensor) and was itself refined once already (routing through `F.linear`
instead of `nn.Conv2d`, roughly halving that one op's contribution from ~50% to ~24% flake
rate) — but the remainder comes from other, real spatial convolutions elsewhere in the network
and is **not resolved**.

**Why not fixed yet:** the two candidate remedies both trade against a Tier-0 requirement:
- `torch.backends.cudnn.deterministic = True` forces deterministic algorithm selection but is
  documented to cost real throughput on some shapes, and interacts with `benchmark = True` in
  ways that need measuring, not assuming, before shipping — exactly the kind of unmeasured
  tradeoff this project's own standing rules forbid presenting as free.
- Disabling `cudnn.benchmark` entirely would fix determinism everywhere but costs throughput on
  every convolution in the network, not just the flaky one, and directly regresses V40.

**Partially resolved, main session, this session.** Measured the throughput cost of the two
obvious remedies before applying either (per this project's own standing rule against
unmeasured tradeoffs): `torch.backends.cudnn.deterministic = True` costs 355.0 → 355.2 ms/batch
(0.06%); additionally disabling TF32 everywhere (`cudnn.allow_tf32` / `matmul.allow_tf32 =
False`, previously on for the bf16 default path) costs 355.08 → 355.78 ms (0.2%). Both
noise-level, not a real tradeoff — applied both in `inference.py::tune_backends()`, plus
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (set before any CUDA context exists, per PyTorch's own
documented requirement for deterministic cuBLAS) and `torch.use_deterministic_algorithms(True,
warn_only=True)`.

**Result: substantially improved, not fully eliminated.** Flake rate measured across repeated
`--only V24` runs: **50%** (unpatched baseline) → **~24%** (F.linear routing alone, D42
addendum) → **~20%** (full stack: deterministic cudnn + TF32 off + CUBLAS workspace config +
deterministic algorithms). An isolated manual repro (same settings, same checkpoint, outside
the verifier's subprocess harness) gave 6/6 identical checksums, so the residual source is
specific to something in the full `inference.py` CLI path (`ctx.run_inference`'s actual
subprocess invocation) that a minimal in-process repro does not reproduce — not yet isolated.
One plausible contributor, not yet confirmed: concurrent GPU use by other processes on this
dev machine (several were observed via `tasklist`/`Get-CimInstance` during this session) could
perturb cuDNN's algorithm-selection cache or memory layout even under `deterministic=True`,
in a way that would not reproduce on a single-tenant clean CI/eval box.

**Still not fully resolved.** All changes made are strict improvements (measured, negligible
cost, real flake-rate reduction) and are committed. Full elimination needs further isolation
work this session did not have time to complete. **This still blocks Definition of Done #2**
(two consecutive clean `--strict --fresh-clone` runs) in the strict sense that V24 could land
FAIL on either run by chance (~1-in-5). Operational recommendation: if a `--strict` run reports
V24 FAIL and nothing else is red, re-running the suite is legitimate (V24 is independently
seeded per invocation, not gamed) rather than treating one flaky-check failure as a full
iteration failure — but this should be re-investigated properly, not relied upon indefinitely.

## B12 — V22 and V51: two real regressions from promoting the Round 2 long-run checkpoint, each requiring a human policy decision, not a code fix (docs/decisions.md D61)

Promoting the long-run checkpoint (width=64, num_blocks=32, FiLM+uncertainty, 1,393,938
params — 3.6x the prior 388,225-param checkpoint) turned two previously-green checks red.
Both were investigated with the same empirical rigor as D42, root-cause found in each case,
and in neither case is there a single-line code fix available — both are genuine scale-vs-
policy trade-offs, which Prime Directive 1 reserves for a human ("if a check seems wrong, log
it here and stop — do not edit it").

### V22 (bf16 vs fp32 divergence): root cause is depth-compounding, not an unpromoted op

Measured: mean 1.85e-03 (cap 1e-3), max 2.65e-02 (cap 1e-2) on the V22 fixture.

Investigated in order:
1. **Is FiLM the cause?** No. Measured the SAME width=64/num_blocks=32 architecture with
   `film_dim=0` (FiLM disabled) at random init: mean 2.30e-03, max 1.24e-02 — already over
   the cap with FiLM entirely absent. FiLM only adds a small increment (2.30e-03→2.39e-03
   mean, 1.24e-02→1.32e-02 max).
2. **Is D42's SCA fix still in effect?** Yes, verified present and unmodified in
   `src/blocks.py`.
3. **Does keeping the residual accumulator in fp32 across the block loop help?** No —
   monkeypatch-tested directly (confirmed the patch executed and the accumulator really did
   stay fp32 through every block via a dtype trace), and the final divergence was
   bit-for-bit IDENTICAL to unpatched (mean 3.24e-03, max 2.73e-02 either way). ATen's
   ordinary type promotion (`fp32 + bf16*fp32 -> fp32`, confirmed in isolation) does keep the
   accumulator's container fp32, but this does not help: the information lost is inside each
   block's own bf16-computed conv/gate/SCA branch output, which is already rounded before it
   re-enters the accumulator — no amount of accumulator-dtype bookkeeping recovers that.
4. **Is the divergence a discrete jump at one op (an unpromoted-op bug, D42's class), or
   smooth compounding (a genuine scale effect)?** Traced divergence at every 4th block:
   grows from mean 3.06e-02 (block 3) to a peak of ~2.03e-01 (block 23), then DECREASES
   toward the end of the stack (4.81e-02 at block 31), then drops sharply at `body_tail`
   (3.30e-02) and again at the final output (3.24e-03) — the deterministic bilinear-upsample
   skip connection (computed identically in both precisions) dilutes most of the internal
   noise. This is the signature of ordinary compounding random-walk-like rounding error over
   32 sequential lossy transformations, not a discrete unpromoted op — there is no single
   line to fix, matching D42's SCA case.

**Why the old checkpoint passed:** D42 measured the prior 16-block checkpoint at max 7.79e-3,
already close to the 1e-2 cap. Doubling the block count pushes the same natural bf16 rounding
process over the line — a margin-vs-scale issue, not a bug this session introduced or can
patch away without either (a) running substantially more of the forward pass in fp32 (real
throughput cost, undermining much of bf16's benefit at exactly the resolution/batch sizes
this project has spent effort optimizing), or (b) a human decision on the tolerance/architecture
trade-off. **Not fixed. Verifier tolerance NOT touched.**

**2026-08-17 update — the trade-off is now priced, not just disclosed (`docs/decisions.md`
D65).** Full 400-pair paired comparison, real `inference.py` forward path: fp32 wins PSNR
(+0.00189 dB, t=+6.08) and SSIM (+0.00013, t=+10.37) with statistical significance but
negligible practical magnitude, while LOSING LPIPS (bf16 is actually better there, t=+10.50)
and costing a measured **+10.6% throughput** (`scripts/benchmark_runtime.py`, median of 3
repeats each precision). Confirmed directly against `check_V22`'s own source: switching the
`--precision auto` default would NOT make V22 pass either way — V22 explicitly runs both
`--precision bf16` and `--precision fp32` itself and compares them, independent of the
script's default. **Decision: keep bf16 as the default, do not switch.** The quality "win" is
noise-level, not a real improvement, and it comes with a real throughput cost on exactly the
metric (LPIPS) this checkpoint is already weakest on. V22 remains a disclosed, live FAIL —
this update prices the trade-off decided in D61/D62, it does not change the decision.

### V51 (tracked-file size cap): a real gap in the existing checkpoint exemption

`weights/best.pt` is now 11,565,729 bytes (11.03 MiB), exceeding `MAX_TRACKED_FILE_BYTES`
(5 MiB) in `scripts/verify_all.py`'s per-file size-cap loop (the section commented "Size caps
catch a dataset dump regardless of extension").

`CHECKPOINT_BLOB_EXEMPTION = "weights/best.pt"` already exists in the same function and is
used one code block earlier to exempt `best.pt` from the blob-EXTENSION ban (D41) — but the
per-file SIZE-cap loop that follows does not check this exemption at all; it applies the 5 MiB
cap to every tracked file uniformly, including the one file the contract already treats as a
sanctioned, mandatory, deliberately-tracked large blob (V59 requires it be tracked; V43
separately caps it at 100 MB specifically). The prior 388,225-param checkpoint (3.14 MiB) was
simply always under 5 MiB by coincidence, so this gap was never exercised before.

This reads like an implementation gap (the exemption exists but was only wired into one of the
two places it needed to be), not a deliberately-designed 5 MiB ceiling on the checkpoint
specifically — but per Prime Directive 1, that judgment is not this session's to act on
unilaterally. **Not fixed. `scripts/verify_all.py` NOT edited.**

### Decision needed (human) — RESOLVED for V51, still open for V22

**V51: RESOLVED 2026-08-17, human-authorised (`docs/decisions.md` D62).** Chosen resolution:
extend `CHECKPOINT_BLOB_EXEMPTION`'s use to the size-cap loop (both the per-file and
total-tree caps), consistent with its existing purpose and with V43's separate, more
appropriate 100 MB cap already governing this exact file. Applied, negative-controlled (a
genuine 6 MiB tracked `.txt` file still correctly fails after the fix), and re-pinned in
`docs/VERIFIER_SHA256`. **V51 now PASSES** in `results/verification_report.json`. This
paragraph is left here, not deleted, as the investigation record the fix was built on — the
narrative above (the gap itself, why it happened) still stands; only the "not fixed" status
line is stale and is superseded by this note.

**V22: still open, per the human's own choice.** Presented alongside V51 above; the human
chose "accept as a disclosed trade-off, leave the check red" rather than force a fix or
reconsider the checkpoint. `scripts/verify_all.py`'s V22 check and tolerance remain untouched.
This is not an oversight — it is the recorded decision. See the top of `README.md` and
`docs/decisions.md` D61/D62 for where this is disclosed to a reviewer.
