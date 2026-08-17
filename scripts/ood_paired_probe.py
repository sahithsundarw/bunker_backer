"""H0.1 -- is D61's real-SEM OOD regression systematic, or idiosyncratic to one set?

D61 promoted the current checkpoint (weights/best.pt, 29.2548 dB in-distribution) over the
superseded D49 checkpoint despite an UNPAIRED real-SEM OOD regression: SSIM 0.328 -> 0.260,
LPIPS 0.569 -> 0.711 (docs/decisions.md D61). That comparison was never run PAIRED, and KLA
explicitly scores OOD content -- so before launching any cloud run this script settles, with a
paired test (the same src.metrics.paired_compare / PAIRED_T_CRIT=1.96 harness as D49/D61), on
three sets:

  - the 400-pair in-distribution val split (control -- should reproduce D61's win)
  - the 40-image procedural proxy-OOD set (results/eda/proxy_ood/)
  - the 45-image real-SEM OOD set (results/eda/real_sem_ood/)

The superseded checkpoint is recovered from git history (commit 19e4e76, the merge commit
that landed it -- NOT re-trained, NOT re-fit; a byte-identical checkpoint recovered from the
tracked history). Both checkpoints run through the same in-process forward pass (mirrors
scripts/make_baselines.py's run_learned), never through inference.py (no need to time this).

No training, no fitting anywhere in this script (F17 untouched -- this is pure evaluation).

Writes results/eda/ood_paired_probe.json with a verdict: "systematic" if BOTH OOD sets show a
significant paired loss on the new checkpoint for SSIM or LPIPS, "idiosyncratic" if only one
does, "no_regression" if neither does.

Owner: main session (read-only on model/weights, writes only to results/eda/ and this script).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import resolve_data_root, train_val_names  # noqa: E402
from src.metrics import paired_compare, score_pair  # noqa: E402
from src.model import build_model  # noqa: E402

OLD_CKPT_COMMIT = "19e4e76"  # the merge commit that landed the superseded D49 checkpoint
NEW_CKPT_PATH = ROOT / "weights" / "best.pt"
SCRATCH_OLD_CKPT = ROOT / "results" / "eda" / "_scratch_old_best.pt"

PROXY_OOD_DIR = ROOT / "results" / "eda" / "proxy_ood"
REAL_SEM_OOD_DIR = ROOT / "results" / "eda" / "real_sem_ood"


def _recover_old_checkpoint() -> Path:
    """Extract weights/best.pt as it stood at OLD_CKPT_COMMIT, byte-identical, no retraining."""
    if SCRATCH_OLD_CKPT.exists():
        return SCRATCH_OLD_CKPT
    SCRATCH_OLD_CKPT.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "show", f"{OLD_CKPT_COMMIT}:weights/best.pt"],
        cwd=ROOT, capture_output=True, check=True,
    )
    SCRATCH_OLD_CKPT.write_bytes(proc.stdout)
    return SCRATCH_OLD_CKPT


def _load_model(ckpt_path: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", cfg) if isinstance(cfg, dict) else {}
    model = build_model(model_cfg)
    state = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    meta = {"sha256_len": ckpt_path.stat().st_size,
            "val_metrics": ckpt.get("metrics") if isinstance(ckpt, dict) else None,
            "config": model_cfg}
    return model, meta


def _predict(model: torch.nn.Module, lr: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(lr, dtype=np.float32))[None, None].to(device)
    with torch.no_grad():
        y = model(x)
    return np.clip(y[0, 0].float().cpu().numpy(), 0.0, 1.0)


def _score_set(model: torch.nn.Module, device: str, lr_dir: Path, gt_dir: Path,
               names: list[str], with_lpips: bool) -> list[dict[str, Any]]:
    out = []
    for name in names:
        lr = np.load(lr_dir / name, allow_pickle=False)
        gt = np.load(gt_dir / name, allow_pickle=False)
        pred = _predict(model, lr, device)
        scores = score_pair(pred, gt, with_lpips=with_lpips, device=device)
        out.append({"file": name, **scores})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val_n", type=int, default=400,
                     help="cap on in-distribution val images (0 = all)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    t_start = time.time()

    old_ckpt_path = _recover_old_checkpoint()
    old_model, old_meta = _load_model(old_ckpt_path, device)
    new_model, new_meta = _load_model(NEW_CKPT_PATH, device)
    if args.verbose:
        print(f"old ckpt: {old_ckpt_path} ({old_ckpt_path.stat().st_size} B)")
        print(f"new ckpt: {NEW_CKPT_PATH} ({NEW_CKPT_PATH.stat().st_size} B)")

    sets: dict[str, dict[str, Any]] = {}

    # -- in-distribution val split (control) --------------------------------------------
    data_root = resolve_data_root(args.data_root)
    _, names = train_val_names(data_root)
    if args.val_n and args.val_n < len(names):
        names = names[: args.val_n]
    lr_dir = data_root / "train" / "NoisyLR"
    gt_dir = data_root / "train" / "GT"
    if args.verbose:
        print(f"[val] scoring {len(names)} pairs, old checkpoint...")
    old_scores = _score_set(old_model, device, lr_dir, gt_dir, names, with_lpips=True)
    if args.verbose:
        print(f"[val] scoring {len(names)} pairs, new checkpoint...")
    new_scores = _score_set(new_model, device, lr_dir, gt_dir, names, with_lpips=True)
    sets["val_in_distribution"] = {"n": len(names),
                                    "paired": paired_compare(new_scores, old_scores)}

    # -- procedural proxy-OOD -----------------------------------------------------------
    if PROXY_OOD_DIR.is_dir():
        proxy_names = sorted(p.name for p in (PROXY_OOD_DIR / "NoisyLR").glob("*.npy"))
        if args.verbose:
            print(f"[proxy_ood] scoring {len(proxy_names)} pairs, both checkpoints...")
        old_p = _score_set(old_model, device, PROXY_OOD_DIR / "NoisyLR", PROXY_OOD_DIR / "GT",
                            proxy_names, with_lpips=True)
        new_p = _score_set(new_model, device, PROXY_OOD_DIR / "NoisyLR", PROXY_OOD_DIR / "GT",
                            proxy_names, with_lpips=True)
        sets["proxy_ood"] = {"n": len(proxy_names), "paired": paired_compare(new_p, old_p)}
    else:
        sets["proxy_ood"] = {"n": 0, "paired": {}, "note": "directory not found locally"}

    # -- real-SEM OOD --------------------------------------------------------------------
    if REAL_SEM_OOD_DIR.is_dir():
        real_names = sorted(p.name for p in (REAL_SEM_OOD_DIR / "NoisyLR").glob("*.npy"))
        if args.verbose:
            print(f"[real_sem_ood] scoring {len(real_names)} pairs, both checkpoints...")
        old_r = _score_set(old_model, device, REAL_SEM_OOD_DIR / "NoisyLR",
                            REAL_SEM_OOD_DIR / "GT", real_names, with_lpips=True)
        new_r = _score_set(new_model, device, REAL_SEM_OOD_DIR / "NoisyLR",
                            REAL_SEM_OOD_DIR / "GT", real_names, with_lpips=True)
        sets["real_sem_ood"] = {"n": len(real_names), "paired": paired_compare(new_r, old_r)}
    else:
        sets["real_sem_ood"] = {"n": 0, "paired": {}, "note": "directory not found locally"}

    # -- verdict --------------------------------------------------------------------------
    def _regressed(set_name: str) -> bool:
        paired = sets.get(set_name, {}).get("paired", {})
        return any(paired.get(m, {}).get("loss") for m in ("ssim", "lpips"))

    proxy_regressed = _regressed("proxy_ood")
    real_regressed = _regressed("real_sem_ood")
    if proxy_regressed and real_regressed:
        verdict = "systematic"
    elif proxy_regressed or real_regressed:
        verdict = "idiosyncratic"
    else:
        verdict = "no_regression"

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "old_checkpoint": {"source": f"git show {OLD_CKPT_COMMIT}:weights/best.pt",
                            "path": str(old_ckpt_path), "meta": old_meta},
        "new_checkpoint": {"path": str(NEW_CKPT_PATH), "meta": new_meta},
        "sets": sets,
        "verdict": verdict,
        "verdict_rule": ("systematic = paired SSIM-or-LPIPS loss (new vs old, |t|>=1.96) on "
                          "BOTH OOD sets; idiosyncratic = on exactly one; no_regression = "
                          "on neither"),
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    out_path = ROOT / "results" / "eda" / "ood_paired_probe.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"VERDICT: {verdict}")
    for set_name, s in sets.items():
        for metric, v in s.get("paired", {}).items():
            print(f"  [{set_name}] {metric}: n={v['n']} mean_diff={v['mean_diff']:+.5f} "
                  f"t={v['t']:+.2f} win={v['win']} loss={v['loss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
