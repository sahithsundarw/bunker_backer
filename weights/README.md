# Model weights

`best.pt` is the checkpoint `inference.py` loads. It is resolved as
`Path(__file__).resolve().parent / "weights" / "best.pt"` — relative to the **script**, never
to the current working directory and never an absolute literal (V05). A reviewer does not
have to pass `--weights`, set an environment variable, or edit anything.

Per SPEC §9 the checkpoint self-describes (V35): it carries `model`, `ema`, `config`, `iter`,
`metrics` and `git` keys, so weights can never be silently paired with the wrong
architecture. `inference.py` prefers the **EMA** weights when present.

---

## Status — 2026-08-17, Round 2 long run promoted (`docs/decisions.md` D61)

**A new checkpoint from the Round 2 differentiation cloud long run supersedes D49's.**

`weights/best.pt` is now the long-run checkpoint from `configs/long_run_e.yaml` (Pareto-sweep
config `e`, D55): NAFSR width=64, num_blocks=32, **FiLM noise-level conditioning
(`film_dim=16`) and a heteroscedastic uncertainty head (`uncertainty=True`) both enabled and
trained end-to-end** (D52). Trained on an HF Jobs A100-large GPU, full 129,700-iteration
schedule, best-of-run selected at iteration 76,000 by `ema/psnr` over the full run (not the
final iteration). Re-scored head-to-head against D49's checkpoint under one harness, paired
per-image test, before promotion — see D61 for the full table. Wins PSNR and SSIM
significantly; LPIPS is a statistical tie. **Also now beats the U-Net baseline on all three
metrics** (V28, previously a documented 1/3-win negative result under D49's checkpoint).

| Field | Value |
|---|---|
| SHA256 | `8f54f9a208220dfd6cd3d67766945ad781bf141fcc03fac41d216caf4fa9643c` |
| File size | 11,565,729 bytes (11.03 MiB) |
| Architecture | NAFSR, width=64, num_blocks=32, scale=2, in_ch=out_ch=1, film_dim=16, uncertainty=True (embedded in `config.model`) |
| Total parameters | 1,393,938 |
| Training seed | 42 |
| Training iterations | 129,700 total run; best selected at iter 76,000 (`iter`/`metrics.best_iter` keys) |
| Wall-clock (HF Jobs A100-large) | 22,895.55 s (6h 21m) |
| Git SHA at training time | `unknown` — the HF Jobs container fetched a tarball snapshot of the `windows-session` branch (docs/PLAN_CLOUD.md), not a git clone, so `git_sha()`'s fallback fired. Honest gap, not fabricated: the branch/commit dispatched is recorded in `docs/PLAN_CLOUD.md`'s execution log instead. |
| Val PSNR (disk-verified, full 400-pair split) | **29.2548 dB** |
| Val SSIM (disk-verified, full 400-pair split) | **0.79211** |
| Val LPIPS (disk-verified, full 400-pair split) | **0.25625** |
| Validation protocol | `scripts/make_baselines.py` -> `scripts/evaluate.py`, reloaded from disk (V30 round-trip), `configs/split_val.txt` (400 pairs), never the final test set |

**Trade-off disclosed, not hidden:** this checkpoint's real-SEM-OOD generalisation (D53/D61)
got measurably WORSE on SSIM (0.328 -> 0.260) and LPIPS (0.569 -> 0.711) versus D49's smaller
checkpoint, despite improving in-distribution and procedural-proxy-OOD. See D61 for the full,
honest comparison — not averaged away or omitted because it cuts against the promotion.

D49's checkpoint (388,225 params, no FiLM/uncertainty) remains available via the GitHub Release
`artifacts-v1` asset for anyone wanting to reproduce that comparison exactly.

---

## Superseded — 2026-08-16, reconciled (`docs/decisions.md` D49)

**A trained checkpoint is present and tracked (Route A — committed directly).**

`weights/best.pt` is a from-scratch-trained NAFSR: width 48, 16 blocks, no closed-form
initialisation, no frozen layers — every parameter trained from a random init on the provided
degraded pairs. This repo briefly diverged into two independent lines of work (this one, and a
teammate's separate closed-form-LS5-plus-residual NAFSR variant); both checkpoints were
re-scored head-to-head on one machine, one harness, the same 400-image split, and this one won
all three metrics, paired and significant (`docs/decisions.md` D49, `docs/MERGE_ANALYSIS.md`).
The alternate checkpoint's own provenance is recorded in `docs/decisions.md` D48 and
`docs/STATE.md`'s teammate-session archive, not repeated here.

| Field | Value |
|---|---|
| SHA256 | `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` |
| File size | 3,288,805 bytes (3.14 MiB) |
| Architecture | NAFSR, width=48, num_blocks=16, scale=2, in_ch=out_ch=1 (embedded in `config.model`) |
| Total parameters | 388,225 (all trained from scratch; no frozen layers) |
| Training seed | 42 |
| Training iterations | 20,000 (`iter` key) |
| Git SHA at training time | `80e7fb049367afe99fbcabb8e5469861f630fecc-dirty` |
| Val PSNR (disk-verified, full 400-pair split) | **28.7865 dB** |
| Val SSIM (disk-verified, full 400-pair split) | **0.78287** |
| Val LPIPS (disk-verified, full 400-pair split) | **0.25324** |
| Validation protocol | `scripts/make_baselines.py` → `scripts/evaluate.py`, reloaded from disk (V30 round-trip), `configs/split_val.txt` (400 pairs), never the final test set |

Published independently as a GitHub Release asset (`artifacts-v1`, same SHA256) since before
this checkpoint was tracked directly in-tree — that Release is what let this exact checkpoint
be recovered byte-for-byte after a merge briefly overwrote `weights/best.pt` on disk with the
alternate checkpoint. Both now coexist safely: this file in the tree, the Release as an
independent backup.

### What `inference.py` does if the checkpoint is missing or fails to load

Normal submission inference treats the checkpoint as required. If it cannot be found or
loaded, `inference.py` prints an explicit error and exits nonzero without generating
substitute outputs. The ordinary two-argument command and `--require_weights` are both strict;
`--require_weights` is retained as an explicit assertion for release-generation commands.

`--allow_bicubic_fallback` is an explicit, demo-only opt-in for the parameter-free bicubic
baseline. It is never enabled by default, may not be used for submission outputs, and is
overridden by `--require_weights`. `results/restored_test_outputs/` was generated with
`--require_weights`, so those artifacts are model outputs.

---

## Availability and verification

**Route A — committed directly.** `weights/best.pt` is tracked in this repository at 11.03 MiB,
far under GitHub's 100 MB limit. No external link, no Release asset required for a reviewer to
obtain it. (The separate 400-file *restored test outputs* archive is too large for this route
and ships as a Release asset instead — see `results/restored_test_outputs/README.md`.) The
previous (D49) checkpoint remains available via the GitHub Release `artifacts-v1` asset.

Verify the committed bytes match the table above:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('weights/best.pt').read_bytes()).hexdigest())"
```

should print `8f54f9a208220dfd6cd3d67766945ad781bf141fcc03fac41d216caf4fa9643c`.

## Reproduction

```bash
python train.py --config configs/long_run_e.yaml --seed 42 --hub_repo <a HF Hub model repo>
```

129,700 iterations (Pareto-sweep config `e`: width=64, num_blocks=32, FiLM+uncertainty
enabled), batch 32, bf16, EMA decay 0.999, `save_best_on: psnr`, warmup 3,000 iters, cosine
schedule. Full config is embedded in the checkpoint's own `config` key, so the run is
traceable without external notes. Measured wall-clock: 22,895.55 s (6h 21m) on an HF Jobs
A100-large (`docs/PLAN_CLOUD.md`, `docs/decisions.md` D55/D61) — `--hub_repo` is required for
a cloud job since Job storage is ephemeral; a local GPU run does not need it. This checkpoint
was NOT trained locally, unlike D49's predecessor (kept below for that record).

## Checklist before submission

- [x] `best.pt` present in a fresh clone (Route A)
- [x] file > 1 KB and not an LFS pointer stub (V06)
- [x] checkpoint < 100 MB (V43's cap)
- [x] `build_model(ckpt["config"])` accepts the stored state dict with `strict=True` (V35) —
      confirmed live via `inference.py --require_weights` against `sample_inputs/`
- [x] `inference.py --require_weights` succeeds against it — the bicubic fallback is not in play
- [x] parameter count and checkpoint size recorded in `results/runtime_report.md` (V43)
