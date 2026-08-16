# Model weights

`best.pt` is the checkpoint `inference.py` loads. It is resolved as
`Path(__file__).resolve().parent / "weights" / "best.pt"` — relative to the **script**, never
to the current working directory and never an absolute literal (V05). A reviewer does not
have to pass `--weights`, set an environment variable, or edit anything.

Per SPEC §9 the checkpoint self-describes (V35): it carries `model`, `ema`, `config`, `iter`,
`metrics` and `git` keys, so weights can never be silently paired with the wrong
architecture. `inference.py` prefers the **EMA** weights when present.

---

## Status — 2026-08-15, `codex/final-submission-28db`

**A trained checkpoint is present and tracked (Route A — committed directly).**

`weights/best.pt` is the `r2_nb8_psnrloss` residual-refinement checkpoint: a closed-form 5×5
least-squares restoration fit (`stem`, `head.expand`, `head.project`) with its `body` frozen,
plus a fresh 8-block NAFSR residual correction trained on top, described in
`docs/decisions.md` D28/D29. It supersedes the earlier closed-form-only checkpoint
(SHA256 `d5807dab…`, val PSNR 26.3277 dB), which improved on every one of the three scored
metrics.

| Field | Value |
|---|---|
| SHA256 | `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7` |
| File size | 2,068,091 bytes (1.97 MiB) |
| Architecture | NAFSR, width=48, num_blocks=8, scale=2, in_ch=out_ch=1 (embedded in `config.model`) |
| Total parameters | 246,529 (84,049 frozen LS-5 stem/head + 162,480 trainable residual body) |
| Training seed | 42 |
| Training iterations | 4,000 (`iter` key) |
| Git SHA at training time | `73696a694d3b2be13fe17d2dc4e891d2165da020-dirty` |
| Val PSNR (disk-verified, full 400-pair split) | **28.0394 dB** |
| Val SSIM (disk-verified, full 400-pair split) | **0.74804** |
| Val LPIPS (disk-verified, full 400-pair split) | **0.29571** |
| Validation protocol | `scripts/make_baselines.py` → `scripts/evaluate.py`, reloaded from disk (V30 round-trip), `configs/split_val.txt` (400 pairs), never the final test set |

The checkpoint's own `metrics` block also retains the in-loop n=100 selection number
(`in_loop_selection_val_psnr_n100`) under a name that makes clear it is **not** the reported
result — the reported 28.0394 dB figure is the full-400-split, disk-verified number, per
`docs/decisions.md` D28/D29.

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

**Route A — committed directly.** `weights/best.pt` is tracked in this repository at 1.97 MiB,
far under GitHub's 100 MB limit. No external link, no Release asset, no link-rot risk for the
checkpoint itself. (The separate 400-file *restored test outputs* archive is too large for this
route and ships as a Release asset instead — see `results/restored_test_outputs/README.md`.)

Verify the committed bytes match the table above:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('weights/best.pt').read_bytes()).hexdigest())"
```

should print `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7`.

## Reproduction

The checkpoint was produced in two stages, both on branch `codex/residual-ls5-refinement`:

1. **Closed-form LS-5 fit** (frozen stem/head), reproduced with:
   ```bash
   python train.py --config configs/final.yaml --seed 42 --closed_form_linear --out weights/best.pt
   ```
2. **Residual refinement on top of the frozen fit** (`scripts/train_residual.py`, Phase 2/4 of
   `docs/decisions.md` D28/D29), using `configs/phase4_psnr_focus.yaml` and `--num_blocks 8`.
   Full command and config are recorded in `docs/decisions.md` D29 and embedded in the
   checkpoint's own `config`/`git` keys, so the run is traceable without external notes.

## Checklist before submission

- [x] `best.pt` present in a fresh clone (Route A)
- [x] file > 1 KB and not an LFS pointer stub (V06)
- [x] checkpoint < 100 MB (V43's cap)
- [x] `build_model(ckpt["config"])` accepts the stored state dict with `strict=True` (V35) —
      confirmed live via `inference.py --require_weights` against `sample_inputs/`
- [x] `inference.py --require_weights` succeeds against it — the bicubic fallback is not in play
- [x] parameter count and checkpoint size recorded in `results/runtime_report.md` (V43) — done
      for this checkpoint in Phase 5 of this branch's integration
