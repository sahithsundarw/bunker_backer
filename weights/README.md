# Model weights

`best.pt` is the tracked checkpoint loaded by `inference.py`. The path is resolved relative
to the script, not the current working directory. Runs intended to produce scored or submitted
outputs must use `--require_weights` so a missing checkpoint cannot silently fall back to
bicubic upsampling.

## Promoted Checkpoint

| Field | Measured value |
|---|---|
| Architecture | `AdaptiveLinearSR` |
| Local filter | 7x7 |
| Gating | 16 local-intensity bins x 8 local-texture bins |
| Scale / channels | x2, grayscale 1 -> 1 |
| Parameters | 25,600 |
| File size | 106,231 bytes |
| SHA256 | `a2c53f8667c3f63efb23cb1411bac8eb2f394ac818e6cd7a9f5352736c3990cf` |
| Producing code SHA | `1fabda2c0a59d63bda59f31c68962a7ab042d305` |
| Best config | `configs/backtrack_best.yaml` |
| Fit data | 2,800 training names; committed validation names excluded |

The checkpoint carries the required `model`, `ema`, `config`, `iter`, `metrics`, and `git`
keys. `build_model(ckpt["config"])` loads its state dict with `strict=True`.

## Validation

All metrics below are measured from 400 saved `.npy` predictions reloaded from disk on
`configs/split_val.txt`:

| Inference | PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|
| Normal | 27.437587 | 0.730039 | **0.322458** |
| 8-way TTA | **27.437886** | **0.730049** | 0.323138 |

TTA is selected for the PSNR objective, but its +0.000299 dB gain is negligible and LPIPS is
slightly worse. The 29.0 dB target was not reached. The best score is +1.110209 dB over the
LS-5 parent and +1.165686 dB over NLM bicubic.

The final input-only dataset has no ground truth and was not used for fitting or metric
computation.

## Reproduction

```bash
python scripts/backtrack_search.py \
  --data_root /Users/shanmukhsai/Downloads \
  --out_root runs/backtrack_search \
  --target 29.0 \
  --max_trials 11 \
  --device cpu
```

The complete experiment tree, commands, metrics, runtimes, and backtrack decisions are in
`results/backtrack_experiments.csv`.
