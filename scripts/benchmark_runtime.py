#!/usr/bin/env python3
"""End-to-end runtime harness (SPEC 11.4 step 7; V37, V38, V39).

Times the WHOLE PROCESS externally -- interpreter start, imports, model load, IO and
compute -- not an internal timer around the forward pass. Startup is ~85-95% of the
scored wall-clock (docs/decisions.md D7), so an internal timer would hide most of it.

V39 has NO threshold and may NOT be skipped: measure on whatever device is present and
LABEL the report with that device name (docs/decisions.md D10).

Writes results/runtime_report.md with a startup-vs-compute breakdown.

Owner: perf-analyst (proposes), main session (applies).
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out", default="results/runtime_report.md")
    ap.add_argument("--repeats", type=int, default=3)
    _ = ap.parse_args(argv)
    raise NotImplementedError("benchmark_runtime.main: not implemented yet")


if __name__ == "__main__":
    sys.exit(main())
