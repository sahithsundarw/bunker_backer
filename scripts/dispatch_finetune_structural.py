"""Dispatch the plan Phase 3 fine-tune (configs/finetune_structural_content.yaml) as an HF
Jobs A100 run. See docs/decisions.md D68/D69 for why this run exists (the real-SEM OOD gap
is content-driven; weight interpolation of the D67 fine-tune does not recover it at any
mixing ratio) and configs/finetune_structural_content.yaml for the exact objective.

Near-identical to scripts/dispatch_finetune_job.py (D63's dispatch), with one hard-learned
correction from D67: **`run_job(timeout=...)` was found NOT to be reliably enforced by the
platform** -- the D67 job was still `RUNNING` at 3h18m, 18+ minutes past its intended cap, and
had to be cancelled manually on discovery. This script still passes `timeout` (cheap, may
help), but does NOT rely on it: `--watch` blocks in-process, polling `inspect_job` every 2
minutes, and calls `cancel_job()` itself the moment the job exceeds its cap, rather than
trusting the platform to stop it.

Usage:
    python scripts/dispatch_finetune_structural.py                 # dispatch, do not watch
    python scripts/dispatch_finetune_structural.py --watch          # dispatch, then poll+cancel
    python scripts/dispatch_finetune_structural.py --dry_run        # print the command only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GITHUB_REPO = "https://github.com/sahithsundarw/bunker_backer.git"
DATA_REPO = "Team-Ceciroleo67/kla-ps01-data"
HUB_REPO = "Team-Ceciroleo67/kla-ps01-checkpoints"
NAMESPACE = "Team-Ceciroleo67"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "a100-large"
TIMEOUT_STR = "3h"          # passed to the platform, but NOT trusted alone (D67)
CAP_SECONDS = 3 * 3600      # the REAL cap, enforced by --watch's own poll+cancel
CONFIG = "configs/finetune_structural_content.yaml"
VAL_EVERY = 2000
PUSH_EVERY = 2000


def _build_command(git_ref: str) -> list[str]:
    script = f"""
set -euo pipefail
echo "[dispatch] installing extra deps not in the base image..."
apt-get update -qq && apt-get install -y -qq git > /dev/null
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
    ap.add_argument("--git_ref", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--watch", action="store_true",
                    help="block after dispatch, polling every 2 min, and manually cancel at "
                         "the 3h cap -- do not rely on the platform timeout (D67)")
    args = ap.parse_args(argv)

    import subprocess
    git_ref = args.git_ref
    if git_ref is None:
        git_ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        if dirty:
            print("ERROR: working tree is dirty. Commit and push first, or pass --git_ref.",
                  file=sys.stderr)
            return 2

    command = _build_command(git_ref)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set.", file=sys.stderr)
        return 2

    print(f"[dispatch] git_ref={git_ref}")
    print(f"[dispatch] image={IMAGE} flavor={FLAVOR} timeout={TIMEOUT_STR} namespace={NAMESPACE}")
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
        timeout=TIMEOUT_STR,
        namespace=NAMESPACE,
        secrets={"HF_TOKEN": token},
        labels={"run": "finetune_structural", "config": Path(CONFIG).stem},
    )
    print(f"[dispatch] launched job id={job.id}")
    print(f"JOB_ID={job.id}")  # machine-parseable line for the caller

    if not args.watch:
        return 0

    print(f"[dispatch] watching (manual poll+cancel at {CAP_SECONDS}s -- D67's timeout kwarg "
          f"was not reliably enforced by the platform)")
    t_start = time.time()
    while True:
        time.sleep(120)
        elapsed = time.time() - t_start
        st = api.inspect_job(job_id=job.id, namespace=NAMESPACE).status
        print(f"[dispatch] t={elapsed:.0f}s stage={st.stage}")
        if st.stage != "RUNNING":
            print(f"JOB_TERMINAL={st.stage}")
            return 0
        if elapsed >= CAP_SECONDS:
            print(f"[dispatch] CAP REACHED ({elapsed:.0f}s >= {CAP_SECONDS}s) -- cancelling "
                  f"manually, not trusting the platform timeout")
            api.cancel_job(job_id=job.id, namespace=NAMESPACE)
            print("JOB_TERMINAL=CANCELLED_BY_WATCHER")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
