# Missing Inputs

Two of the five files named in the setup brief could **not** be found on this machine.
They were not fabricated or reconstructed.

## Not found

| Expected file | Intended destination | Status |
|---|---|---|
| `KLA_IMAGE_RESTORATION_MASTER_SPEC.md` | `docs/SPEC.md` | **MISSING** |
| `peek.py` | `scripts/peek.py` | **MISSING** |

### Where we looked

- `C:\Users\sahit\Downloads` — full recursive scan
- `C:\Users\sahit\OneDrive\Desktop` — full recursive scan
- `C:\Users\sahit\Desktop` — does not exist (Desktop is redirected into OneDrive)
- Contents of every candidate archive: `files.zip`, `files (1).zip`,
  `botsplanet-intelligence-proact.zip`, and the `drive-download-*` folders
- `C:\Users\sahit\Downloads\hackathon_requirements\` — contains only PDFs, PNG screenshots,
  and two `.txt` notes; no `.md` and no `.py`

The recursive scan for `*.md` matched only these, none of them the SPEC:

```
C:\Users\sahit\Downloads\files (2)\CLAUDE.md
C:\Users\sahit\Downloads\files (2)\LOOP_PROMPT.md
C:\Users\sahit\Downloads\files (2)\VERIFICATION_CONTRACT.md
C:\Users\sahit\OneDrive\Desktop\60x40\CLAUDE.md              (unrelated project)
C:\Users\sahit\OneDrive\Desktop\projects\os\CLAUDE.md        (unrelated project)
```

## Found and placed

| Source | Destination |
|---|---|
| `C:\Users\sahit\Downloads\files (2)\CLAUDE.md` | `CLAUDE.md` |
| `C:\Users\sahit\Downloads\files (2)\LOOP_PROMPT.md` | `LOOP_PROMPT.md` |
| `C:\Users\sahit\Downloads\files (2)\VERIFICATION_CONTRACT.md` | `docs/VERIFICATION_CONTRACT.md` |

## Consequences

**`docs/SPEC.md` is absent**, therefore:

- The U1–U9 question list in "SPEC §2.2" could not be read. `docs/dataset_findings.md`
  answers U1, U2, U3, U8, U9 using the labels from the task brief, but the wording of the
  actual questions is unverified.
- "SPEC F17" is quoted in `docs/DATA_LOCATION.md` from the brief, not from the SPEC itself.
- The **output/submission format is unknown**. `docs/io_contract.md` is therefore marked
  PROVISIONAL and derives the output contract by symmetry with GT. Its `[UNCONFIRMED]`
  items — container format, directory layout, per-image renormalisation, and the scoring
  metric — must be resolved before any submission is built.

**`scripts/peek.py` is absent**, therefore:

- `scripts/inspect_dataset.py` was written as a substitute, taking the same CLI shape
  (`python scripts/inspect_dataset.py C:\kla-data`). It is original code, not a
  reconstruction of `peek.py`.
- `scripts/probe_quantization.py` is a second, follow-up probe covering bit-depth,
  clipping, and train-vs-test distribution.

## To resolve

Recover `KLA_IMAGE_RESTORATION_MASTER_SPEC.md` from wherever `files (2)` originally came
from — the three files that *were* found share a timestamp of 2026-08-15 01:18:30, so the
SPEC was likely in the same delivery and simply not downloaded.
