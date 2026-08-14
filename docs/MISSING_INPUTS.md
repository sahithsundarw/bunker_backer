# Input Provenance

Record of which setup inputs were expected, what happened to each, and what was built.

## Nothing is missing

**All expected inputs are accounted for. There are no outstanding missing files and no
blocked work resulting from a missing input.** This document is retained as a provenance
record, not as an open-items list. Genuine blockers belong in `docs/BLOCKERS.md`.

| Expected input | Destination | Status |
|---|---|---|
| `CLAUDE.md` | `CLAUDE.md` | **present** |
| `LOOP_PROMPT.md` | `LOOP_PROMPT.md` | **present** |
| `VERIFICATION_CONTRACT.md` | `docs/VERIFICATION_CONTRACT.md` | **present** |
| `KLA_IMAGE_RESTORATION_MASTER_SPEC.md` | `docs/SPEC.md` | **present** (arrived late — see below) |
| `peek.py` | — | **CLOSED — superseded, not missing** |

## `peek.py` — CLOSED

`peek.py` was a throwaway inspection script. It does not exist and will not exist. It has
been **superseded** by `scripts/inspect_dataset.py`, which is the name SPEC §5.1 and §12
actually call for, and which is strictly better than what it replaced:

- scans **all 3200 pairs** for shape and scale-factor verification rather than sampling;
- adds the **bit-depth grid test** at 8/10/12/16 bits, which is what settled U1;
- adds the **alignment shift scan** (±3 px cross-correlation), which is what settled U8.

This item requires no further action and must not be re-opened or searched for again.

## `docs/SPEC.md` arrived after the first pass

At initial setup the SPEC did not exist on this machine; a recursive scan of `Downloads`,
`OneDrive`, `Documents`, `Desktop` and `AppData\Local\Temp` returned no match, and no
candidate archive contained it. It was subsequently supplied at
`C:\Users\sahit\Downloads\KLA_IMAGE_RESTORATION_MASTER_SPEC.md` (66,092 bytes, mtime
2026-08-15 02:02:33 — later than the searches). Copied to `docs/SPEC.md`, 738 lines, parsed
in full.

Everything written before it arrived has since been reconciled against the real SPEC text.
`docs/SPEC_ADDENDUM.md`, `docs/io_contract.md` (now FINAL), `docs/dataset_findings.md` and
`docs/decisions.md` all cite verified section numbers.

## Scripts built

| Script | Purpose | SPEC anchor |
|---|---|---|
| `scripts/inspect_dataset.py` | dataset inventory — U1, U2, U3, U7, U9 | §5.1, §12 (named) |
| `scripts/fit_degradation.py` | kernel, noise model, degradation order — U4, U5, U6, U8 | §5.2, §5.3, §12 (named) |
| `scripts/probe_quantization.py` | bit-depth grid, clipping, train/test distribution | §5.1 |
| `scripts/visual_audit.py` | 12-triplet grid + aliasing screen | §5.4 |
| `scripts/renorm_experiment.py` | renormalisation policy (D3) | §5.1 → `io_contract.md` |
| `scripts/content_audit.py` | content-domain characterisation (D4) | §5.4 |
| `scripts/domain_shift_check.py` | train-vs-test content shift (D5) | §6.1 |

Both scripts SPEC names explicitly — `inspect_dataset.py` and `fit_degradation.py` — exist
and have been run. That requirement is satisfied.

## Not yet built (deferred by instruction, not blocked)

`src/`, `train.py`, `inference.py`, `configs/`, `weights/`, `scripts/verify_all.py`,
`docs/STATE.md`, `docs/BLOCKERS.md`. Model building has not been authorised. These are
pending work, not missing inputs.
