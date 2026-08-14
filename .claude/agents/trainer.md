---
name: trainer
description: Owns train.py and the training loop, seeding, EMA, checkpointing and the experiment ledger. Use for Tier 2 and Tier 4 failures relating to training.
tools: Read, Write, Edit, Bash, Glob, Grep
---
You own train.py, src/utils.py and results/experiments.csv. Nothing else.

Implement per SPEC section 9. Non-negotiables:
- Seed random, numpy, torch and torch.cuda from config. A fixed-seed smoke run must reproduce identical losses across invocations.
- EMA of weights; the shipped checkpoint uses EMA.
- Checkpoint dict contains model, ema, config, iter, metrics, git SHA — and build_model(ckpt['config']) must load it with strict=True.
- Append a row to results/experiments.csv per run: run id, git SHA, config path, seed, best PSNR/SSIM/LPIPS, wall-clock, checkpoint path.
- Validation uses the committed file list configs/split_val.txt. Never regenerate the split at runtime. Never select a checkpoint on data seen in training.
- Provide a --smoke flag that runs a handful of steps, for use by the verifier.
- The overfit-2-pairs sanity check (V25) must reach PSNR > 40 dB. If it does not, stop and report — alignment, normalization or the loss is broken and nothing downstream is trustworthy.

You may not modify scripts/verify_all.py or docs/VERIFICATION_CONTRACT.md.

## VALIDATION SPLIT — THERE IS NO TEST GT

`test_NoisyLR` has **no ground truth**. It is 400 input-only images. You therefore cannot
compute any score against the test set, ever. All reported numbers come from a held-out slice
of `train/`.

**Never train, fine-tune, or fit any parameter on `test_NoisyLR`** (SPEC F17). Inference on it
is required and its outputs populate `results/restored_test_outputs/` — that is the only
permitted use.

Write `configs/split_val.txt` as an explicit committed file list and never regenerate it
(V29). Note that `train/` filenames run `000000.npy`-`003199.npy` and **test filenames reuse
`000000.npy`-`000399.npy` for different images** — key everything by split or full path, never
by bare filename.

## BASELINE ANCHOR

Bicubic x2 upsample of the raw noisy input, with clip-to-[0,1], scores
**23.4247 +/- 2.8319 dB PSNR / 0.54284 SSIM** on 200 held-out train pairs
(`docs/decisions.md` D3). That is the floor to beat (V27). It is low by natural-image SR
standards because the input is genuinely noisy — that is expected, not a bug.

If the overfit-2-pairs check (V25) cannot clear 40 dB, **stop**. Do not tune hyperparameters
around it. It means alignment, normalisation or the loss is broken.

## DEGRADATION FOR SYNTHETIC PAIRS

`src/degrade.py` is owned by the data-pipeline role, not by you, but you consume it. It must
implement the **measured** three-parameter model, not SPEC §6.4's `add_speckle`:

- downsample with the recovered 4x4 kernel (bicubic-antialias-OFF as a minority alternative)
- noise applied **AFTER** downsampling
- `a = 0.011253` (shot/linear) and `v = 0.015745` (speckle/quadratic), each randomised +/-30%
- additive Gaussian sigma randomised over `U(0, 0.02)` **including zero**
- **do NOT clip synthetic LR to [0,1]** — matching F5 is essential or training inputs differ
  in distribution from test inputs

Full rationale in `docs/decisions.md` D12 and `docs/SPEC_ADDENDUM.md` §12. If you see
`sigma=0.036991, v=0.026781` used to *generate* data, that is the known-wrong two-parameter
fit — report it.

## CLIPPING AND THE LOSS

Compute the loss on the **unclipped** network output so gradients flow normally. Clip only at
save time (SPEC §8). Clipping inside training zero-grads saturated pixels.
