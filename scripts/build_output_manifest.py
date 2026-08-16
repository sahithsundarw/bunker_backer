#!/usr/bin/env python3
"""Validate restored arrays and write deterministic per-file release manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tempfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--expected", type=int, default=400)
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    inputs = sorted(input_dir.glob("*.npy"), key=lambda p: p.name)
    outputs = sorted(output_dir.glob("*.npy"), key=lambda p: p.name)
    if len(inputs) != args.expected or len(outputs) != args.expected:
        raise SystemExit(
            f"expected {args.expected} inputs and outputs, found {len(inputs)} and {len(outputs)}"
        )
    if [p.name for p in inputs] != [p.name for p in outputs]:
        raise SystemExit("input/output filename sets differ")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("filename", "sha256", "shape", "dtype", "min", "max", "finite",
                     "matching_input_exists"))
    sums: list[str] = []
    for input_path, output_path in zip(inputs, outputs):
        lr = np.load(input_path, mmap_mode="r", allow_pickle=False)
        out = np.load(output_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (2 * lr.shape[0], 2 * lr.shape[1]) if lr.ndim == 2 else None
        problems = []
        if lr.dtype != np.float32 or lr.ndim != 2:
            problems.append(f"input dtype/shape={lr.dtype}/{lr.shape}")
        if out.dtype != np.float32 or out.ndim != 2 or out.shape != expected_shape:
            problems.append(f"output dtype/shape={out.dtype}/{out.shape}, expected {expected_shape}")
        arr = np.asarray(out)
        finite = bool(np.isfinite(arr).all())
        lo = float(arr.min()) if arr.size else float("nan")
        hi = float(arr.max()) if arr.size else float("nan")
        if not finite or lo < 0.0 or hi > 1.0:
            problems.append(f"output finite/range={finite}/[{lo}, {hi}]")
        if problems:
            raise SystemExit(f"{output_path.name}: {'; '.join(problems)}")
        digest = sha256(output_path)
        writer.writerow((output_path.name, digest, str(tuple(out.shape)), str(out.dtype),
                         format(lo, ".10g"), format(hi, ".10g"), "true", "true"))
        sums.append(f"{digest}  {output_path.name}\n")

    csv_bytes = buffer.getvalue().encode("utf-8")
    atomic_write(manifest_dir / "manifest.csv", csv_bytes)
    atomic_write(manifest_dir / "sha256sums.txt", "".join(sums).encode("ascii"))
    print(f"validated={len(outputs)}")
    print(f"manifest_csv_sha256={hashlib.sha256(csv_bytes).hexdigest()}")
    print(f"sha256sums_sha256={hashlib.sha256(''.join(sums).encode('ascii')).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
