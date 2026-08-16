#!/usr/bin/env python3
"""Expand checkpoint metadata from a canonical config without changing model tensors."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SOURCE_FILES = (
    "train.py",
    "src/blocks.py",
    "src/dataset.py",
    "src/degrade.py",
    "src/losses.py",
    "src/metrics.py",
    "src/model.py",
    "src/utils.py",
    "configs/final.yaml",
    "configs/split_val.txt",
)


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def source_manifest(commit: str) -> list[dict[str, str]]:
    rows = []
    for path in TRAINING_SOURCE_FILES:
        content = git_bytes(commit, path)
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{commit}:{path}"], cwd=ROOT, text=True
        ).strip()
        rows.append({"path": path, "git_blob": blob,
                     "sha256": hashlib.sha256(content).hexdigest()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--source_commit", required=True)
    ap.add_argument("--original_sha256", required=True)
    ap.add_argument("--original_url", required=True)
    args = ap.parse_args()

    path = Path(args.checkpoint)
    original = path.read_bytes()
    observed = hashlib.sha256(original).hexdigest()
    if observed != args.original_sha256:
        raise SystemExit(f"original checkpoint SHA mismatch: {observed}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    with Path(args.config).open("r", encoding="utf-8") as fh:
        canonical = yaml.safe_load(fh)
    original_git = str(payload.get("git", "unknown"))
    payload["config"] = canonical
    payload["provenance"] = {
        "schema_version": 1,
        "original_release_url": args.original_url,
        "original_checkpoint_sha256": args.original_sha256,
        "original_checkpoint_git": original_git,
        "canonical_training_source_commit": args.source_commit,
        "canonical_training_source_files": source_manifest(args.source_commit),
        "metadata_normalization": (
            "Expanded implicit model/loss defaults into configs/final.yaml and added this "
            "source manifest; model and EMA tensors are byte-identical to the release asset."
        ),
        "original_worktree_marker_preserved": original_git.endswith("-dirty"),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
