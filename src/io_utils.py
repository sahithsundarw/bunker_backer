"""Shared array IO helpers for the inference path.

The dataset is .npy float32 end to end: np.load in, np.save out. NO image library is
imported here or in inference.py -- see docs/SPEC_ADDENDUM.md section 5 and CLAUDE.md STYLE.
V23 (Tier 0) asserts the module-level import allowlist.

Contract (docs/io_contract.md, FINAL):
  in  : .npy float32 (H,W), values unbounded, observed [-0.28, 2.16]. NEVER clipped on input.
  out : .npy float32 (2H,2W), clipped to [0,1], NO renormalisation, filename byte-identical.

Owner: inference-engineer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_array(path: Path) -> np.ndarray:
    """Load a 2-D float32 array. Does NOT clip -- out-of-range values are intentional."""
    raise NotImplementedError("load_array: not implemented yet")


def save_array(path: Path, arr: np.ndarray) -> None:
    """Clip to [0,1], cast to float32, np.save. No renormalisation (docs/decisions.md D3)."""
    raise NotImplementedError("save_array: not implemented yet")
