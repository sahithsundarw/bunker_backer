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
