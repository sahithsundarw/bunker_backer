#!/usr/bin/env python3
"""Training entry point. Reproduces weights/best.pt (SPEC 9).

    python train.py --config configs/nafnet_x2.yaml --data_root <dataset_root>

Owner: trainer.
"""

from __future__ import annotations

import argparse
import sys


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the restoration model.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="few steps only; used by the verifier")
    ap.add_argument("--resume", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    _ = build_argparser().parse_args(argv)
    raise NotImplementedError("train.main: not implemented yet")


if __name__ == "__main__":
    sys.exit(main())
