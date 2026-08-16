"""Shared array IO helpers for the inference path.

The dataset is .npy float32 end to end: np.load in, np.save out. NO image library is
imported here or in inference.py -- see docs/SPEC_ADDENDUM.md section 5 and CLAUDE.md STYLE.
V23 (Tier 0) asserts the module-level import allowlist, and this module is imported by
inference.py, so its own imports sit inside KLA's measured window too: numpy and pathlib
only, nothing else, ever.

Contract (docs/io_contract.md, FINAL):
  in  : .npy float32 (H,W), values unbounded, observed [-0.28, 2.16]. NEVER clipped on input.
  out : .npy float32 (2H,2W), clipped to [0,1], NO renormalisation, filename byte-identical.

Owner: inference-engineer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_array", "save_array", "list_inputs", "mirror_path"]


def load_array(path: Path) -> np.ndarray:
    """Load a 2-D float32 array. Does NOT clip -- out-of-range values are intentional.

    ~3% of real input pixels exceed 1.0 and reach 2.16, and ~0.5% fall below 0.0 (SPEC F5,
    docs/SPEC_ADDENDUM.md section 4). Clipping or rescaling here would destroy information
    the model is trained to use, so this function does neither: the only transformations are
    a dtype cast to float32 and the removal of a singleton channel axis.

    Raises on anything that is not a plain 2-D (or trivially squeezable 3-D) numeric array,
    so that inference.py can log the file and continue instead of aborting the run (V20).
    """
    a = np.load(path, allow_pickle=False)
    if not isinstance(a, np.ndarray):
        raise ValueError(f"not a plain ndarray: {type(a).__name__}")
    if a.ndim == 3:
        # Tolerate (H,W,1) and (1,H,W); anything else is genuinely ambiguous.
        if a.shape[-1] == 1:
            a = a[..., 0]
        elif a.shape[0] == 1:
            a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {a.shape}")
    if a.dtype == np.dtype(object) or a.dtype.kind not in "fiub":
        raise ValueError(f"non-numeric dtype {a.dtype}")
    out = np.asarray(a, dtype=np.float32)
    return np.ascontiguousarray(out)


def save_array(path: Path, arr: np.ndarray) -> None:
    """Atomically clip to [0,1], cast to float32, and save without renormalising.

    Per-image min-max renormalisation was measured at -4.66 dB PSNR, losing 191/200 held-out
    pairs, so the ONLY range handling is the clip that SPEC F5/section 5.1 mandates.

    The array is written through an open file handle rather than by path, because
    ``np.save(path, ...)`` appends ``.npy`` unless the name already ends in exactly that
    lowercase suffix -- an input named ``FOO.NPY`` would become ``FOO.NPY.npy`` and break the
    byte-identical filename rule (V08). Writing to a handle never renames anything.

    Non-finite values are neutralised here as a last line of defence for V11 (a NaN survives
    ``np.clip`` untouched); callers should still log when they see them.
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D array to save, got shape {a.shape}")
    a = a.astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    a = np.ascontiguousarray(a, dtype=np.float32)
    # Import locally: inference.py deliberately keeps its measured import path minimal.
    import os
    import tempfile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, a, allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def list_inputs(root: Path, exts: set[str]) -> list[Path]:
    """Every file under ``root`` whose suffix is in ``exts``, case-insensitively.

    Recurses into subdirectories (SPEC 11.1 requires mirrored structure) and returns a
    deterministically sorted list -- output must not depend on filesystem walk order (V24).
    """
    root = Path(root)
    found = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def mirror_path(in_root: Path, out_root: Path, src: Path) -> Path:
    """Destination for ``src``: same relative path, byte-identical filename (V08, V18)."""
    return Path(out_root) / Path(src).relative_to(in_root)
