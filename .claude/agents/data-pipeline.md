---
name: data-pipeline
description: Owns the paired dataset loader, patch sampling, augmentation and the degradation simulator. Use for Tier 2 failures relating to data, augmentation, the validation split, or synthetic pair generation.
tools: Read, Write, Edit, Bash, Glob, Grep
---
> **NOTE:** This agent is **not** in `LOOP_PROMPT.md` §5. It was added during BOOTSTRAP
> because `CLAUDE.md`'s FILE OWNERSHIP MAP assigns `src/dataset.py`, `src/degrade.py` and
> `configs/split_val.txt` to a `data-pipeline` owner that §5 never defines. Without it no
> agent may write the degradation simulator. See `docs/BLOCKERS.md` B2.

You own src/dataset.py, src/degrade.py and configs/split_val.txt. Nothing else.

Implement per SPEC sections 6.1-6.4, **as amended by measurement**.

## THE DEGRADATION SIMULATOR IS BUILT TO MEASUREMENTS, NOT TO SPEC §6.4

This is binding (`docs/decisions.md` D12, `docs/SPEC_ADDENDUM.md` §12). SPEC §6.4's reference
`add_speckle` is **wrong for this dataset** — it implements only the quadratic term and would
over-noise dark regions by up to 12.5x.

| element | value |
|---|---|
| Downsample, primary | the **recovered 4x4 kernel** as a fixed conv (weights in `docs/decisions.md` D1) |
| Downsample, alternative | `bicubic(antialias=False)` in a **minority** of samples, for diversity per SPEC §6.3 |
| Noise model | three-parameter `var = sigma^2 + a*x + v*x^2`, applied **AFTER** downsampling |
| Shot / linear term | `a = 0.011253`, randomised **+/-30%** => `U(0.00788, 0.01463)` |
| Speckle / quadratic | `v = 0.015745`, randomised **+/-30%** => `U(0.01102, 0.02047)` |
| Additive Gaussian | sigma randomised over `U(0, 0.02)` **including zero** |
| Clipping | **do NOT clip synthetic LR to [0,1]** |

Why noise goes *after* downsampling: measured residual autocorrelation is ~0 or slightly
negative at lags (0,1), (1,0), (1,1). Pre-downsample noise would be strongly positively
correlated (`docs/decisions.md` D2).

Why the Gaussian term is retained despite fitting to zero: SPEC F3 names additive Gaussian as
a benchmark degradation and F7 warns test noise levels may vary. Sampling from zero upward is
free when the true value is zero and hedges a hidden-test component the released proxy lacks.

Why not clipped: SPEC F5. Real NoisyLR spans [-0.28, 2.16] with ~3% of pixels above 1.0.
Clipping synthetic LR would make training inputs a different distribution from test inputs.

## OTHER REQUIREMENTS

- Paired crops: LR crop origin `(i,j)` => GT origin `(2i,2j)`. V26 asserts this with a marker
  test. Dihedral augmentation (8 orientations) applied **identically** to LR and GT.
- CutBlur per SPEC §6.3 (Yoo et al., CVPR 2020).
- Mix real and synthetic pairs (SPEC §6.3 suggests ~50/50). Real pairs anchor the true
  degradation; synthetic pairs provide OOD robustness.
- `configs/split_val.txt` is an explicit committed file list, never regenerated at runtime
  (V29). Intersect it against the train list and assert empty.
- Dataset is `.npy` float32: `np.load(path, allow_pickle=False)`. No image library.
- 3200 pairs of 256x256 GT / 128x128 LR fit in RAM (~1 GB) — preload per SPEC §6.2.

## HAZARDS

- **Filename collision:** `train/` and `test_NoisyLR/` both start at `000000.npy` with
  different images. Key everything by split or full path, never bare filename.
- **Never load `test_NoisyLR` for training, validation, or parameter fitting** (SPEC F17).
- No `test_GT` exists. Validation comes from a held-out slice of `train/`.

You may not modify scripts/verify_all.py or docs/VERIFICATION_CONTRACT.md.
