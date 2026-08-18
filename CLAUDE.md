# CLAUDE.md — Operating Rules

Project: **KLA PS01 — AI-Based Restoration of Degraded Images for Semiconductor Inspection**
(SEMICON India Hackathon 2026, Track 1)

> ⚠ **The problem is semiconductor inspection; the released data is natural photographs.**
> All 3600 provided images are ordinary grayscale photos (architecture, animals, foliage,
> landmarks), not SEM or inspection imagery. Treat the released set as a **proxy**: optimise
> for degradation robustness, not content priors. Never describe the provided imagery as
> semiconductor in code comments, docs, README or the deck.
> See `docs/SPEC_ADDENDUM.md` (headline finding, §7, §11) and `docs/decisions.md` D4/D5.

## Canonical documents

| File | Role | Mutable? |
|---|---|---|
| `docs/SPEC.md` | Full technical spec: requirements, dataset forensics protocol, architecture, losses, training, inference, pitfalls. **Read it before doing anything.** | Append-only (add findings; never delete requirements) |
| `docs/VERIFICATION_CONTRACT.md` | The locked list of V-checks that define "done". | **IMMUTABLE.** See Prime Directive 1. |
| `docs/STATE.md` | Rolling ledger: what is done, what is in flight, what is blocked. Survives context resets. | Yes — you maintain it |
| `docs/dataset_findings.md` | Empirical answers to open questions U1–U9 in SPEC §2.2, with numeric evidence | Yes |
| `docs/io_contract.md` | The exact file format / dtype / naming rule derived from the real dataset | Yes, but changes require re-running all V-checks |
| `docs/decisions.md` | Every design decision + one-line rationale | Append-only |
| `docs/BLOCKERS.md` | Things you could not resolve and why | Yes |

---

## PRIME DIRECTIVES

**1. The verifier is the boss. You may never weaken it.**
`docs/VERIFICATION_CONTRACT.md` and `scripts/verify_all.py` define correctness. You may make checks *stricter*. You may **never** delete a check, loosen a threshold, add a skip, widen a tolerance, wrap a check in `try/except pass`, mark a test `xfail`, or reduce the test corpus in order to turn a red check green. If a check seems wrong, log it in `docs/BLOCKERS.md` and stop — do not edit it.
`scripts/verify_all.py` is hash-pinned in `docs/VERIFIER_SHA256`. Any iteration that changes the file without a corresponding entry in `docs/decisions.md` explaining the *strengthening* is a failed iteration and must be reverted.

**2. Fix causes, not symptoms.** If a V-check fails, find the root cause. Do not special-case the input that triggered it.

**3. Never fabricate a fact about the dataset.** Anything marked UNVERIFIED in SPEC §2.2 must be derived from the actual data and recorded in `docs/dataset_findings.md` with the evidence (numbers, not prose). If the dataset is unavailable, say so in `docs/BLOCKERS.md` and build against synthetic stand-in data that is clearly labelled as such.

**4. `run.py` is the highest-value file in the repo.** KLA runs it as-is on an H100 to produce both your quality score and your throughput score. A repo with a mediocre model and a flawless inference script scores; a repo with a great model and a broken script scores zero. Prioritize accordingly. (Renamed from `inference.py` 2026-08-18 per an official, track-specific final-submission announcement — see `docs/decisions.md` D75. `inference.py` still exists as a 3-line back-compat shim, but `run.py` is the file that is graded, timed, and covered by the verifier.)

**5. No hardcoded paths, ever.** Resolve weights relative to `Path(__file__).resolve().parent`. Never relative to CWD. Never absolute. Test from `/`.

**6. Determinism and provenance.** Every training run writes a row to `results/experiments.csv` including git SHA, config path, seed, and metrics. Every checkpoint embeds its own config.

**7. Small commits, always green.** Commit after every V-check transition red→green, with the check ID in the message: `fix(V07): resolve weights via __file__ not cwd`. Never leave the tree in a state where previously-green checks are red.

**8. If you are unsure whether something satisfies a requirement, assume it does not**, and write a V-check that proves it either way.

---

## FILE OWNERSHIP MAP (for parallel subagents)

Parallel agents must never write to the same file. When fanning out, assign by this map. Any agent asked to touch a file outside its column must instead report the needed change back to the main session.

| Owner | May write |
|---|---|
| `dataset-forensics` | `scripts/inspect_dataset.py`, `scripts/fit_degradation.py`, `docs/dataset_findings.md`, `docs/io_contract.md`, `results/eda/**` |
| `model-core` | `src/model.py`, `src/blocks.py`, `configs/*.yaml` |
| `data-pipeline` | `src/dataset.py`, `src/degrade.py`, `configs/split_val.txt` |
| `loss-metrics` | `src/losses.py`, `src/metrics.py`, `scripts/evaluate.py`, `scripts/make_baselines.py` |
| `inference-engineer` | `run.py`, `src/io_utils.py` |
| `throughput-optimizer` | `scripts/benchmark_runtime.py`, `results/runtime_report.md` — and may propose (not apply) diffs to `run.py` |
| `trainer` | `train.py`, `src/utils.py`, `results/experiments.csv` |
| `docs-scribe` | `README.md`, `docs/decisions.md`, `requirements.txt`, `weights/README.md` |
| Read-only reviewers | Write **only** to `reviews/<agent>-<iteration>.md`. Never touch source. |
| Main session only | `scripts/verify_all.py`, `docs/STATE.md`, `docs/BLOCKERS.md`, `.claude/**`, git operations |

**Only the main session commits.** Subagents leave the working tree dirty; main session reviews, runs verification, then commits.

---

## STATE LEDGER PROTOCOL

`docs/STATE.md` is the memory that survives context compaction. Update it at the **end of every iteration**, before anything else. Format:

```markdown
# STATE
Iteration: 27
Last verified commit: a1b2c3d
Verifier SHA: <sha256 of scripts/verify_all.py>

## V-check status  (from results/verification_report.json)
PASS: V01 V02 V03 V05 V06 ...
FAIL: V11 (3rd consecutive failure — escalating), V19
SKIP: V04 (dataset unavailable, see BLOCKERS)

## In flight
- V11: bf16 output has 3 pixels differing from fp32 by >1e-2 — investigating clamp order

## Consecutive-failure counters
V11: 3    V19: 1

## Do NOT retry (tried and rejected)
- torch.compile as default: adds 71s startup, net loss below ~2000 images (measured, see results/runtime_report.md)
- MS-SSIM at 128px patches: requires >=161px, use single-scale SSIM

## Next iteration plan
1. ...
```

The **"Do NOT retry"** section is the single most important part of this file. It is what stops a long loop from re-attempting the same dead end forever. Append to it every time you reject an approach, with the measurement that justified rejection.

---

## DEFINITION OF DONE

The project is done when, on **two consecutive iterations**, all of the following hold:

1. `python scripts/verify_all.py --strict` exits 0 with zero FAIL and zero SKIP (except SKIPs explicitly whitelisted in `docs/BLOCKERS.md` with a human-readable reason).
2. Verification was run against a **fresh `git clone` into a fresh virtualenv**, not the working tree.
3. All read-only reviewer agents return no findings of severity `high` or `critical`.
4. `docs/STATE.md`, `docs/decisions.md` and `README.md` are current with the code.
5. `git status` is clean and the working tree equals the last verified commit.

When done, print a final report and **stop looping**. Do not invent new work.

---

## STYLE

- Python 3.10+, type hints on public functions, docstrings on modules.
- `run.py` module-level imports are **exactly** this allowlist, nothing else:
  `argparse os sys time pathlib concurrent.futures numpy torch`
  **No image IO library.** The dataset is `.npy` end to end (`np.load` / `np.save`), so
  `cv2`, `tifffile` and `PIL` are dead weight on a timed run and actively hazardous —
  several `cv2` paths silently convert to 8-bit or clip to [0,1], which corrupts inputs
  that legitimately reach 2.16. See `docs/SPEC_ADDENDUM.md` §5.
  Also **no `skimage`, `lpips`, `matplotlib`, `pandas`, `wandb`, `yaml`, `scipy`** at module
  level — import cost is inside KLA's measured window (SPEC §11.2).
  *This allowlist was tightened by the human on 2026-08-15 (removal of "+ one image IO lib").
  V23 checks against it and is promoted to Tier 0 — see `docs/SPEC_ADDENDUM.md` §10.*
- No `print` debugging left in shipped scripts; use a `--verbose` flag.
- Prefer standard library and already-listed dependencies. Every new dependency must be justified in `docs/decisions.md` and pinned in `requirements.txt`.
