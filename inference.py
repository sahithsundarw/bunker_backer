#!/usr/bin/env python3
"""Back-compat alias -- the graded, timed entry point is now `run.py` (docs/decisions.md D75).

Renamed per the organizers' final-submission announcement, which requires the entry script
be named `run.py`. This file is kept only so anything still invoking `inference.py` by the
original spec's name continues to work; the verifier (`scripts/verify_all.py`) targets
`run.py`, not this file. Do not add logic here -- it must stay a trivial alias so `run.py`
remains the single source of truth.

    python inference.py --input_dir <degraded> --output_dir <restored>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
