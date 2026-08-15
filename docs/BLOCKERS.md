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

**Resolution in flight:** torch/torchvision reinstalled from the cu128 index;
`requirements.txt` must pin the index explicitly and `docs/ENVIRONMENT.md` must state the
ordering hazard. Assigned to `docs-scribe` (owner of `requirements.txt`). Not yet verified
end to end in a fresh venv — that is V04's job and V04 is still red.

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
