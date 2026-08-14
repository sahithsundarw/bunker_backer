#!/usr/bin/env python3
"""Standalone inference: restore degraded images (2x SR + joint denoise).

THE FILE KLA RUNS AS-IS. A crash here means the submission is unscored (CLAUDE.md PD4).

    python inference.py --input_dir <degraded> --output_dir <restored>

Module-level imports are limited to EXACTLY the allowlist in CLAUDE.md STYLE:
    argparse os sys time pathlib concurrent.futures numpy torch
NO image IO library. The dataset is .npy end to end. V23 (Tier 0) asserts this statically
and caps `python -X importtime` at 3.0 s. Startup is ~85-95% of the scored wall-clock
(docs/decisions.md D7), so every import costs real score.

Owner: inference-engineer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Permissive glob per SPEC 11.1; only the .npy branch executes on this dataset.
EXTS = {".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def build_argparser() -> argparse.ArgumentParser:
    """Exactly two required args (V02). Everything else optional with working defaults."""
    ap = argparse.ArgumentParser(description="Restore degraded images (2x SR + denoise).")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default=str(SCRIPT_DIR / "weights" / "best.pt"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    _ = build_argparser().parse_args(argv)
    raise NotImplementedError("inference.main: not implemented yet")


if __name__ == "__main__":
    sys.exit(main())
