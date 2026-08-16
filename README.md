# KLA Image Restoration

AI restoration for paired noisy low-resolution grayscale arrays. The submitted NAFSR model
jointly denoises and upsamples each `NoisyLR` input by exactly 2x, writing a float32 `.npy`
array at the GT resolution.

## Submission Status

- `weights/best.pt` is tracked in Git and available in a fresh clone. No manual checkpoint
  download or placement is required.
- Tracked checkpoint SHA256:
  `cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054`.
- The public 400-output archive is available at the
  [artifacts-v1 release](https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/tag/artifacts-v1).
- Normal inference fails loudly if the checkpoint is missing or invalid. Bicubic is available
  only through the explicit `--allow_bicubic_fallback` demo flag; `--require_weights` is
  preserved and takes precedence.

## Results

All quality metrics below are from the committed 400-pair validation split, scored from
reloaded float32 output files. Higher PSNR/SSIM is better; lower LPIPS is better.

| Method | PSNR dB | SSIM | LPIPS | Parameters |
|---|---:|---:|---:|---:|
| Bicubic x2 | 23.6524 | 0.54775 | 0.41206 | 0 |
| Median 3x3 + bicubic | 25.5057 | 0.61317 | 0.40870 | 0 |
| Non-local means + bicubic | 26.2722 | 0.65152 | 0.42586 | 0 |
| U-Net baseline | 28.8808 | 0.78273 | 0.26525 | 2,970,401 |
| **Submitted NAFSR** | **28.7864** | **0.78286** | **0.25323** | **388,225** |

The recovered 20k NAFSR checkpoint fixes the prior checkpoint-selection regression that
scored 28.0394 / 0.74804 / 0.29571. Against U-Net, the selected model gives a statistically
significant LPIPS improvement, an SSIM tie, and a 0.0944 dB PSNR tradeoff while using 7.65x
fewer parameters. The honest paired comparison and selection rationale are recorded in
`docs/decisions.md` D41; full dispersions and per-image statistics are in
`results/metrics_summary.md` and `results/baselines/*/metrics.json`.

The final test set contains no GT. No final-test PSNR, SSIM, or LPIPS is computed or claimed.

## Setup

The target environment is Python 3.12 with CUDA 12.8 PyTorch packages pinned in
`requirements.txt`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; the remaining commands are
unchanged. NVIDIA GPU execution is selected automatically when CUDA is available. Use
`--device cpu` only when explicitly testing the CPU path.

## Inference

The required submission command has exactly two required arguments:

```bash
python inference.py --input_dir sample_inputs --output_dir /tmp/kla-restored
```

The same strict model path can be stated explicitly:

```bash
python inference.py --input_dir sample_inputs --output_dir /tmp/kla-restored-required --require_weights
```

For the released data, point `--input_dir` at its `NoisyLR` or historical `test_NoisyLR`
folder. Inputs must be numeric 2-D `.npy` arrays. Raster formats such as PNG/JPEG/TIFF are
rejected before processing rather than silently skipped. A successful run atomically writes
exactly one clipped float32 `.npy` output per input, mirrors nested paths, removes stale `.npy`
contract outputs, and preserves unrelated files.

## Dataset

Dataset-dependent training, evaluation, diagnostics, and verifier checks require either:

- `KLA_DATA_ROOT` set to the extracted dataset root; or
- the measured local root `/Users/shanmukhsai/Downloads`.

The root must contain:

```text
<data-root>/
  train/GT/            # 3200 float32 arrays, 256x256
  train/NoisyLR/       # 3200 float32 arrays, 128x128
  NoisyLR/             # 400 final-test inputs; historical name test_NoisyLR/
```

`configs/split_val.txt` is the fixed 400-pair validation split. Training never reads final-test
inputs. The final-test directory has no GT and is used only for inference.

## Training

`configs/final.yaml` is the one canonical configuration embedded exactly in the tracked
checkpoint: NAFSR width 48, 16 blocks, zero padding, LayerScale 1.0, AdamW at `1e-3`, beta
`(0.9,0.9)`, 500-step warmup, cosine decay, 20,000 iterations, bf16, seed 42, EMA 0.999, and
the explicit Charbonnier/structural/FFT loss settings.

```bash
python train.py --config configs/final.yaml --data_root /path/to/dataset --seed 42
```

New checkpoints include model, optimizer, schedule, global step, best metric, EMA values and
update count, epoch/batch position, and Python/NumPy/PyTorch CPU/CUDA RNG states. Resume uses
that complete state and rejects legacy inference-only checkpoints:

```bash
python train.py --config configs/final.yaml --data_root /path/to/dataset --resume /path/to/resume.pt
```

The tracked release checkpoint predates resumable-state support and is intentionally treated
as inference-only. Its provenance block preserves the original release digest and dirty marker,
pins the canonical source commit, and records Git blob plus SHA256 identities for every relevant
training source file. The original model and EMA tensors are unchanged.

## Evaluation

Generate and score the submitted model on the fixed validation split:

```bash
python scripts/make_baselines.py --data_root /path/to/dataset --baselines final --device cuda
python scripts/evaluate.py --data_root /path/to/dataset --preds final=results/baselines/final --device cuda
```

`scripts/make_baselines.py` returns nonzero if any requested baseline fails or produces no
outputs. External split files such as `--split /tmp/split.txt` are supported. CUDA transform
timing synchronizes before and after every measured forward; CPU timing is labeled separately.

Qualitative figures can be regenerated from tracked artifacts without an ignored prediction
directory:

```bash
python scripts/make_qualitative_examples.py --data_root /path/to/dataset
```

The script accepts `--checkpoint`, `--val_pred_dir`, `--final_test_lr_dir`, and
`--final_test_pred_dir` overrides. When prediction paths are omitted, selected examples are
generated through strict production inference.

## Published Outputs

The release asset `restored_test_outputs.zip` contains all 400 required final-test outputs:

- URL: https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip
- SHA256: `fbdf8a652d26168cf41e01842ca28d38c53d1da1547bd8ce602b5b8e5d6ac750`
- Size: 91,069,597 bytes

`results/restored_test_outputs/manifest.json`, `manifest.csv`, and `sha256sums.txt` provide
archive-level and per-file verification. The CSV is generated with canonical LF endings so its
recorded SHA is identical in the current checkout and a fresh cross-platform clone.

## Runtime

**Headline: local Mac CPU external-process benchmark, 400 images in 106.43 s (3.8 img/s),
batch size 32, fp32.** It includes process startup, imports, model load, IO, compute, transfers,
post-processing, atomic writes, and output reconciliation.

The historical **release-output generation** run used an NVIDIA RTX 4060 Laptop GPU and
reported 20.09 s from the internal CUDA/bf16 pipeline timer. Linux/CUDA fresh-clone
compatibility passed, but no final Linux/CUDA or H100 runtime was measured. These measurements
are not presented as the same benchmark. See `results/runtime_report.md`.

## Verification

The reproducible source of truth is the verifier command, not a local ignored report:

```bash
KLA_DATA_ROOT=/path/to/dataset python scripts/verify_all.py --strict --fresh-clone
python -m unittest discover -s tests -v
```

Dataset-independent checks and sample inference run from a fresh clone without local files.
Checks that inspect training pairs require `KLA_DATA_ROOT` or the measured Mac root above.

## Repository Layout

```text
repository/
  README.md
  requirements.txt
  train.py
  inference.py
  configs/
  src/
  scripts/
  tests/
  weights/best.pt
  results/
```

`results/experiments.csv` is tracked and records the training experiment ledger. Qualitative
success and failure cases are tracked under `results/qualitative/`.

## External Resources

No external dataset or pretrained restoration weights are used by the submitted model. LPIPS
evaluation uses the package's ImageNet-pretrained AlexNet feature network; it is evaluation
only and contributes no training gradient. Python dependencies retain their upstream open-source
licenses. The released KLA dataset remains outside the repository and is never redistributed.
