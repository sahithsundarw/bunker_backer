"""Shared dataset-layout resolution for standalone diagnostic scripts."""

from __future__ import annotations

import os
from pathlib import Path


def default_dataset_root() -> Path:
    """Portable diagnostic default: environment, measured Mac root, then historical Windows."""
    if os.environ.get("KLA_DATA_ROOT"):
        return Path(os.environ["KLA_DATA_ROOT"]).expanduser()
    measured = Path("/Users/shanmukhsai/Downloads")
    if (measured / "train" / "GT").is_dir():
        return measured
    return Path(r"C:\kla-data")


def resolve_test_input_dir(root: str | Path) -> Path:
    """Return the released test-input directory for current or historical layouts."""
    base = Path(root).expanduser().resolve()
    candidates = (base / "NoisyLR", base / "test_NoisyLR")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    shown = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"released test inputs not found; checked: {shown}")
