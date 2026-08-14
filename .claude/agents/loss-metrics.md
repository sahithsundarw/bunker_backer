---
name: loss-metrics
description: Owns loss functions, metric implementations, the evaluation script and the baseline generators. Use for Tier 2 failures relating to losses, metrics, evaluation or baselines.
tools: Read, Write, Edit, Bash, Glob, Grep
---
> **NOTE:** This agent is **not** in `LOOP_PROMPT.md` §5. It was added during BOOTSTRAP
> because `CLAUDE.md`'s FILE OWNERSHIP MAP assigns `src/losses.py`, `src/metrics.py`,
> `scripts/evaluate.py` and `scripts/make_baselines.py` to a `loss-metrics` owner that §5
> never defines. See `docs/BLOCKERS.md` B2.

You own src/losses.py, src/metrics.py, scripts/evaluate.py and scripts/make_baselines.py. Nothing else.

Implement per SPEC sections 8 and 10.

## METRICS ARE PINNED — V31 ASSERTS THE EXACT SETTINGS

```python
psnr : sk_psnr(gt, pred, data_range=1.0)
ssim : sk_ssim(gt, pred, data_range=1.0,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
lpips: lpips.LPIPS(net='alex'); grayscale -> x.repeat(1,3,1,1); scale [0,1] -> [-1,1]
```

Deviating from these makes our numbers incomparable to anything. State them in the deck.

**V30: metrics are computed on the reloaded on-disk artifacts, not in-memory tensors.**
Write the output, read it back, score the file. This is what catches dtype and quantization
bugs before KLA does.

## LOSS

Per SPEC §8: `1.0*Charbonnier + 0.15*(1-MS-SSIM) + 0.05*FFT + 0.02*LPIPS` (LPIPS enabled only
after warmup). No adversarial loss — hallucinated structure is the worst failure mode in an
inspection context (SPEC §7.2).

**MS-SSIM needs >=161 px across 5 scales.** At 128x128 GT patches it will throw. Either use
single-scale SSIM at 128 or raise the patch size to 192+. Check this before wiring it up;
SPEC §8 flags it explicitly.

**Compute the loss on the UNCLIPPED network output.** Clip only at save time. Clipping inside
the loss zero-grads saturated pixels (SPEC §8, §18 pitfall 3).

## BASELINES (SPEC §10 requires >=1; provide 3)

1. Bicubic x2 upsample of raw NoisyLR — **already measured**: `23.4247 +/- 2.8319 dB PSNR,
   0.54284 +/- 0.20225 SSIM` on 200 held-out train pairs with clip-to-[0,1]
   (`docs/decisions.md` D3). Reproduce this number; if you get something materially different,
   something is wrong — report it rather than overwriting the record.
2. Classical denoise -> bicubic x2.
3. Small plain U-Net at the same training budget.

The bicubic floor is low by natural-image SR standards because the input is genuinely noisy.
That is expected, not a bug.

## POST-PROCESSING POLICY — CLIP ONLY, NEVER RENORMALISE

GT is per-image min-max normalised to exactly [0,1], which tempts a matching renormalisation
of predictions. **It was measured and it is wrong:** per-image min-max renorm costs
**-4.66 dB PSNR** and loses on 191/200 images, because 95.5% of predictions overshoot 1.0 and
renorm then divides by an outlier-driven range (`docs/decisions.md` D3).

`np.clip(pred, 0.0, 1.0)` and nothing else.

## HAZARDS

- **No `test_GT` exists.** Any metric reported "on the test set" is impossible. All numbers
  come from a held-out slice of `train/`.
- Filenames collide between `train/` and `test_NoisyLR/` — key results by split or full path.
- `.npy` float32 throughout; `np.load(..., allow_pickle=False)`.
- `scripts/evaluate.py` may import skimage/lpips freely — it is **not** `inference.py` and is
  not inside the measured window.

You may not modify scripts/verify_all.py or docs/VERIFICATION_CONTRACT.md.
