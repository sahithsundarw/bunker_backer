#!/usr/bin/env python3
"""Score a restored directory against a GT directory (SPEC 10).

V30: reloads the SAVED files from disk and scores those, never in-memory tensors.

Owner: loss-metrics.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", default="results/metrics_summary.md")
    _ = ap.parse_args(argv)
    raise NotImplementedError("evaluate.main: not implemented yet")


if __name__ == "__main__":
    sys.exit(main())
