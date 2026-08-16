# Model weights

`best.pt` is the checkpoint `inference.py` loads. It is resolved as
`Path(__file__).resolve().parent / "weights" / "best.pt"` — relative to the **script**, never
to the current working directory and never an absolute literal (V05). A reviewer does not
have to pass `--weights`, set an environment variable, or edit anything.

Per SPEC §9 the checkpoint self-describes (V35): it carries `model`, `ema`, `config`, `iter`,
`metrics` and `git` keys, so weights can never be silently paired with the wrong
architecture. `inference.py` prefers the **EMA** weights when present.

---

## Status — 2026-08-16, reconciled (`docs/decisions.md` D49)

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

**Route A — committed directly.** `weights/best.pt` is tracked in this repository at 3.14 MiB,
far under GitHub's 100 MB limit. No external link, no Release asset required for a reviewer to
obtain it — though a Release copy also exists (`artifacts-v1`), which is how it was recovered
after the merge described above. (The separate 400-file *restored test outputs* archive is too
large for this route and ships as a Release asset instead — see
`results/restored_test_outputs/README.md`.)

Verify the committed bytes match the table above:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('weights/best.pt').read_bytes()).hexdigest())"
```

should print `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313`.

## Reproduction

```bash
python train.py --config configs/final.yaml --seed 42
```

20,000 iterations, batch 32, 64px patches, bf16, EMA decay 0.999, `save_best_on: psnr`. Full
config is embedded in the checkpoint's own `config` key, so the run is traceable without
external notes. Measured wall-clock: 4,303.5 s (1:11:43) on an RTX 4060 Laptop GPU
(`results/experiments.csv`, run `20260815T062831Z-final-s42`).

## Checklist before submission

- [x] `best.pt` present in a fresh clone (Route A)
- [x] file > 1 KB and not an LFS pointer stub (V06)
- [x] checkpoint < 100 MB (V43's cap)
- [x] `build_model(ckpt["config"])` accepts the stored state dict with `strict=True` (V35) —
      confirmed live via `inference.py --require_weights` against `sample_inputs/`
- [x] `inference.py --require_weights` succeeds against it — the bicubic fallback is not in play
- [x] parameter count and checkpoint size recorded in `results/runtime_report.md` (V43)
