"""Dispatch the Hour-0.5 OOD/scale-gap fine-tune (configs/finetune_ood_wide.yaml) as an HF Jobs
A100 run. See docs/decisions.md D63 for why this run exists and configs/finetune_ood_wide.yaml
for the exact objective (both are read verbatim; nothing here re-derives the rationale).

Not a general-purpose cloud launcher: this is a one-shot, disclosed script for THIS run,
committed so the exact command is on record (docs/PLAN_CLOUD.md's own gap -- no earlier job's
dispatch command was ever committed; this one is).

Hard requirements enforced here, from the plan:
  - image has CUDA + a recent torch preinstalled, additional deps pip-installed at job start
  - --timeout 3h is the REAL cap (finishing beats optimality); finetune_horizon in the config
    is deliberately larger so the job is stopped by this timeout, not by exhausting a schedule
  - --resume weights/best.pt (fine-tune, never from scratch)
  - --push_every / --val_every both set so a partial run still returns usable checkpoints
  - --hub_repo required (job storage is ephemeral)

The dataset is pulled from the private HF dataset repo (docs/PLAN_CLOUD.md Step 1) inside the
job, never bundled into the git clone. The job's own HF_TOKEN (passed as a Job "secret", never
written to a file, never the launching machine's token reused as a file) is what a NEW cloud
process needs for both that download and the periodic checkpoint push.

Usage:
    python scripts/dispatch_finetune_job.py                  # dispatch for real
    python scripts/dispatch_finetune_job.py --dry_run         # print the command, do not launch
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GITHUB_REPO = "https://github.com/sahithsundarw/semicon-kla-image-restoration.git"
DATA_REPO = "Team-Ceciroleo67/kla-ps01-data"
HUB_REPO = "Team-Ceciroleo67/kla-ps01-checkpoints"
NAMESPACE = "Team-Ceciroleo67"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "a100-large"
TIMEOUT = "3h"          # the real, authoritative wall-clock cap
CONFIG = "configs/finetune_ood_wide.yaml"
VAL_EVERY = 2000
PUSH_EVERY = 2000       # est. ~10-15 min at the throughput this config's patch/batch implies;
                        # errs toward MORE frequent, not less, if the estimate is off


def _build_command(git_ref: str) -> list[str]:
    """The exact bash -c script the job container runs, as a single string, so it appears
    verbatim in this file's diff and in the job's own recorded command (inspect_job)."""
    script = f"""
set -euo pipefail
echo "[dispatch] installing extra deps not in the base image..."
pip install -q --no-input numpy scikit-image lpips pyyaml huggingface_hub pytorch-msssim tqdm

echo "[dispatch] cloning {GITHUB_REPO} at {git_ref}..."
git clone --quiet {GITHUB_REPO} /app
cd /app
git checkout --quiet {git_ref}

echo "[dispatch] downloading dataset from {DATA_REPO}..."
python -c "
from huggingface_hub import hf_hub_download
import zipfile, os
os.makedirs('/data', exist_ok=True)
for fname in ('train.zip', 'Test_NoisyLR.zip'):
    p = hf_hub_download(repo_id='{DATA_REPO}', repo_type='dataset', filename=fname)
    print('extracting', p)
    with zipfile.ZipFile(p) as z:
        z.extractall('/data')
"
export KLA_DATA_ROOT=/data
echo "[dispatch] KLA_DATA_ROOT=$KLA_DATA_ROOT contents:"
ls -la /data
ls -la /data/train

echo "[dispatch] starting fine-tune..."
python train.py \\
    --config {CONFIG} \\
    --resume weights/best.pt \\
    --hub_repo {HUB_REPO} \\
    --push_every {PUSH_EVERY} \\
    --val_every {VAL_EVERY} \\
    --val_limit 100 \\
    --val_lpips \\
    --seed 42 \\
    --verbose

echo "[dispatch] train.py exited 0 -- job script complete"
""".strip()
    return ["bash", "-c", script]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--git_ref", default=None,
                    help="commit SHA or branch to clone (default: current HEAD, which MUST "
                         "already be pushed to origin -- the job clones from GitHub, not this "
                         "machine)")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the exact command and run_job() kwargs, do not dispatch")
    args = ap.parse_args(argv)

    import subprocess
    git_ref = args.git_ref
    if git_ref is None:
        git_ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        if dirty:
            print("ERROR: working tree is dirty. The job clones from GitHub, so uncommitted "
                  "changes are invisible to it. Commit and push first, or pass --git_ref "
                  "explicitly.", file=sys.stderr)
            return 2

    command = _build_command(git_ref)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in this shell's environment. Required to (a) verify "
              "the target ref is actually on origin and (b) pass as the job's own secret so "
              "it can pull the private dataset and push checkpoints.", file=sys.stderr)
        return 2

    print(f"[dispatch] git_ref={git_ref}")
    print(f"[dispatch] image={IMAGE} flavor={FLAVOR} timeout={TIMEOUT} namespace={NAMESPACE}")
    print("[dispatch] command:")
    print(command[-1])

    if args.dry_run:
        print("\n[dispatch] --dry_run: not launching.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    job = api.run_job(
        image=IMAGE,
        command=command,
        flavor=FLAVOR,
        timeout=TIMEOUT,
        namespace=NAMESPACE,
        secrets={"HF_TOKEN": token},
        labels={"run": "finetune_ood_wide", "config": Path(CONFIG).stem},
    )
    print(f"[dispatch] launched job id={job.id}")
    print(f"[dispatch] inspect with: python -c \"from huggingface_hub import inspect_job; "
          f"print(inspect_job('{job.id}', namespace='{NAMESPACE}'))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
