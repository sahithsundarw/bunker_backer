#!/usr/bin/env python3
"""External-process runner for a classical baseline, timed the same way as run.py.

Mirrors run.py's I/O contract (.npy in, .npy out, same filenames, clipped [0,1]) and prints a
run.py-compatible summary line, so scripts/benchmark_runtime.py's existing external-timing
harness (subprocess.run wrapping the whole process, not an internal timer) can measure a
classical baseline's genuine end-to-end throughput with the exact same methodology used for
the shipped model -- a real, comparable number instead of "not separately measured"
(docs/decisions.md D77).

Reuses scripts/make_baselines.py's CLASSICAL_BASELINES registry and save_prediction directly
-- this script adds only CLI/timing, it never reimplements or approximates the transform
already scored in results/baselines/*/metrics.json.

    python scripts/run_classical_baseline.py --method bicubic \
        --input_dir C:\\tmp\\val400_lr --output_dir C:\\tmp\\out
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_baselines import CLASSICAL_BASELINES, save_prediction  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--method", required=True, choices=sorted(CLASSICAL_BASELINES))
    # Accepted for CLI compatibility with scripts/benchmark_runtime.py's harness, which
    # always passes --precision/--require_weights/--verbose and optionally --device/
    # --batch_size/--weights -- none of those concepts apply to an unbatched, CPU-only,
    # weight-free classical transform, so they are parsed and ignored, not errors.
    ap.add_argument("--precision", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--require_weights", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bl = CLASSICAL_BASELINES[args.method]

    files = sorted(in_dir.glob("*.npy"))
    t_start = time.perf_counter()
    written = 0
    for p in files:
        lr = np.load(p, allow_pickle=False)  # never clip the INPUT (SPEC F5)
        pred = bl.fn(lr)
        save_prediction(out_dir, p.name, pred)
        written += 1
    elapsed = time.perf_counter() - t_start

    rate = written / elapsed if elapsed > 0 else 0.0
    sys.stdout.write(
        f"restored {written}/{len(files)} in {elapsed:.2f}s ({rate:.1f} img/s) "
        f"device=cpu precision=fp64 batch=1\n"
    )
    return 0 if written == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
