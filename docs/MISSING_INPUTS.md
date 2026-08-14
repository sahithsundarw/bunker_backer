# Input Provenance

Record of which setup inputs were found, when, and what was substituted for what was not.

## Status

| Expected file | Destination | Status |
|---|---|---|
| `CLAUDE.md` | `CLAUDE.md` | found |
| `LOOP_PROMPT.md` | `LOOP_PROMPT.md` | found |
| `VERIFICATION_CONTRACT.md` | `docs/VERIFICATION_CONTRACT.md` | found |
| `KLA_IMAGE_RESTORATION_MASTER_SPEC.md` | `docs/SPEC.md` | **found late** — see below |
| `peek.py` | `scripts/peek.py` | **never found** |

## The SPEC arrived after the first pass

At first setup, `KLA_IMAGE_RESTORATION_MASTER_SPEC.md` did not exist on this machine. A
recursive scan of `C:\Users\sahit\Downloads`, `OneDrive`, `Documents`, `Desktop` and
`AppData\Local\Temp` returned no match, and no candidate archive contained it. The only
`*spec*.md` files present were unrelated (`kpi_system_implementation_spec.md`,
`research-agent-spec*.md`).

It was subsequently supplied at `C:\Users\sahit\Downloads\KLA_IMAGE_RESTORATION_MASTER_SPEC.md`
(66,092 bytes, mtime **2026-08-15 02:02:33** — later than the searches, which is consistent
with it being added afterwards rather than missed). Copied to `docs/SPEC.md`, 738 lines, read
in full.

**Everything written before it arrived has since been reconciled against the real SPEC text.**
`docs/SPEC_ADDENDUM.md`, `docs/io_contract.md` (now FINAL, was PROVISIONAL),
`docs/dataset_findings.md` and `docs/decisions.md` all cite verified section numbers rather
than assumed ones.

## `peek.py` was never found and was not reconstructed

Searched the same locations plus every candidate zip; no match. It was **not** invented.

SPEC §5.1 and §12 in fact call for `scripts/inspect_dataset.py`, not `peek.py`, so the
substitute carries the name the SPEC expects:

| Script | Purpose |
|---|---|
| `scripts/inspect_dataset.py` | SPEC §5.1 inventory — U1, U2, U3, U7, U9 |
| `scripts/probe_quantization.py` | bit-depth, clipping, train/test distribution |
| `scripts/fit_degradation.py` | SPEC §5.2 / §5.3 — kernel, noise model, order |
| `scripts/renorm_experiment.py` | renormalisation policy (D3) |
| `scripts/visual_audit.py` | SPEC §5.4 visual audit and aliasing screen |
| `scripts/content_audit.py` | content-domain characterisation (D4) |
| `scripts/domain_shift_check.py` | train-vs-test content shift (D5) |

All are original code. None reconstructs `peek.py`.

## Still outstanding

Nothing blocking. The SPEC deliverables not yet built (`src/`, `train.py`, `inference.py`,
`configs/`, `weights/`) are deliberately deferred — model building has not been authorised.
