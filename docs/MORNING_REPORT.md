# MORNING REPORT

Written for a human returning to this project cold, and for a fresh agent session with no
memory. Updated continuously **before** each step, not after, so that if the session is cut
off mid-operation this file still describes what was in flight.

**Read `docs/STATE.md` "RESUME HERE" first.** This file is the narrative; that one is the
operational resume point.

---

## Where the project stands

**Repo is live and public:** https://github.com/sahithsundarw/semicon-kla-image-restoration

Verified, not assumed:
- `gh repo view` reports `"visibility": "PUBLIC"`, `"isPrivate": false`.
- An **unauthenticated** clone succeeds — run with `GIT_TERMINAL_PROMPT=0`,
  `GIT_ASKPASS=/bin/false`, `credential.helper=` and global/system git config nulled, so it
  is not passing on cached credentials. Exit 0, 71 files.
- Push size **6.4 MiB**. The instruction was to stop and escalate above ~10 MB; well under.

**Nothing is trained yet.** There is no checkpoint, no metric number, and no restored test
output. Iteration 1 built the machinery, not the model. Any results table you see should be
marked not-yet-measured; if you find one with numbers in it, distrust it and re-derive.

## What was done, in order

1. **Pre-push audit.** Proved no dataset or weight blob is tracked *or ever was*: scanned
   `git ls-files` for 13 forbidden extensions, scanned every path for `kla-data` / `GT` /
   `NoisyLR` tokens, and walked all reachable history with
   `git log --all --diff-filter=A --name-only` plus a largest-blob-in-history scan. Clean.
   Largest tracked object is `results/eda/pairs_grid.png` (2,500,869 B), a required EDA
   figure, not data. `C:\kla-data` has never entered the repo.
2. **Created the public remote** and verified V06/V13's network preconditions as above.
3. **Populated `sample_inputs/`** with 6 real degraded inputs (128x128 float32, 393,984 B
   total). Five of the six carry values outside [0,1] — min -0.1397, max 1.5914 — so the
   folder genuinely exercises the "never clip the input" path rather than a sanitised sample.
4. **Measured the baseline:** PASS 9 / FAIL 44 / SKIP 0 at commit `a980b2f`.
5. **Resolved a contract conflict** (see below), and **dispatched five builders in parallel.**

## The three things worth a human's attention

### 1. V47 and V51 were mutually exclusive — resolved, but flagged honestly
V51 banned every tracked `.npy`. V47 requires inference to run against `sample_inputs/`
**from a clean clone**, which requires those `.npy` files to be in the clone, and SPEC §12
lists the folder as a repo item. Both checks could not be green at once, so the Definition of
Done was literally unreachable.

Resolved the way D6 and D10 were: the human explicitly authorised committing the files, so a
**narrow bounded exemption** was added (`sample_inputs/*.npy`, ≤8 files, ≤512 KB) together
with **four new assertions that make V51 net stricter** — blob-extension ban widened from 4
to 20 extensions, a dataset-directory-token ban, a 5 MB per-file cap and a 25 MB total-tree
cap. The last two catch a dataset dump under *any* extension, which the old blacklist
provably could not.

**Stated plainly:** the exemption is, in isolation, a loosening with respect to six paths.
Everything else added is a strengthening. Full record in `docs/BLOCKERS.md` B7 and
`docs/decisions.md` D15. If you prefer the stricter reading, revert that commit — B7 stands
as the record of the reasoning either way.

### 2. `pip install lpips` silently destroys the CUDA install — and would silently destroy the throughput score
Measured, twice. Installing `lpips` resolves its torch dependency from PyPI and **replaces
`torch==2.11.0+cu128` with `torch==2.13.0+cpu`**. After that `torch.cuda.is_available()` is
`False`.

The reason this is more than a dev-box annoyance: **V04 installs a fresh venv from
`requirements.txt` alone.** If that file does not force the PyTorch index, the clean-room
install produces a CPU-only torch, the run still exits 0, V04 still *passes* — and on KLA's
H100 the GPU sits unused while the throughput score collapses with no error anywhere. This is
the exact class of silent failure V04 exists to catch, and no amount of reading the file
would reveal it. Logged as `docs/BLOCKERS.md` B8; `requirements.txt` fix assigned to
docs-scribe. **Not yet verified end to end in a fresh venv — V04 is still red.**

### 3. B9 needs a human decision and blocks V13
`docs/decisions.md` D17 chose a single `np.savez_compressed` archive as the delivery
mechanism for `results/restored_test_outputs/` (~105 MB raw as 400 `.npy`). Git LFS is
already ruled out by human instruction, because unresolved LFS pointer stubs on a fresh clone
are a known way to fail V06.

But the V51 strengthening in item 1 bans `.npz` and caps any tracked file at 5 MB. So a
~40 MB archive cannot be committed without **a second** human-authorised V51 amendment —
which would gut the size caps just added, and is precisely the pattern Prime Directive 1
forbids an agent from doing on its own initiative.

**Two options, recommendation first:**
1. **External hosting with a published sha256**, link verified from a logged-out session.
   Requires no contract change at all; the current V51 already permits it.
2. A second human-authorised V51 amendment carving out
   `results/restored_test_outputs/*.npz` with its own byte cap.

Blocked pending your call.

## Known-red and honest about it
- **V13** — `results/restored_test_outputs/` is empty. Blocked on B9 and on a trained model.
- **V06** — no `weights/best.pt` exists; `weights/*.pt` is gitignored.
- **V04 / V46 / V47** — need a `--fresh-clone` run, not performed in this pass.
- **V00** — red until `docs/decisions.md` carries the new verifier digest (D15, docs-scribe).
- Everything Tier 2 that needs a trained model: V25 V27 V28 V34 V35 V45 V48 V49.

## Standing strategic note (D16)
The provided imagery is grayscale **natural photographs**, not semiconductor imagery — a
proxy. KLA's hidden test set may be genuine inspection imagery. What transfers across that
gap is the **measured degradation** (recovered 4x4 sharpening downsample kernel, shot noise
rather than Gaussian, noise applied *after* downsampling), not any content prior learned from
photographs. So prefer wide degradation randomisation over squeezing in-distribution dB. A
future iteration that buys +0.2 dB in-distribution by narrowing the degradation range is a
**regression** against the real objective and must be rejected on those grounds, even though
the in-distribution number improved.
