# Model checkpoint

`weights/best.pt` is tracked in Git and loaded by default relative to `inference.py`, not the
current working directory. A fresh clone needs no manual download, environment variable, or
checkpoint placement.

| Field | Value |
|---|---|
| Tracked SHA256 | `cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054` |
| Size | 3,291,621 bytes |
| Architecture | NAFSR, width 48, 16 blocks, 2x, 388,225 parameters |
| Selected weights | EMA at iteration 20,000 |
| Validation, CPU fp32, 400 pairs | PSNR 28.7864 / SSIM 0.78286 / LPIPS 0.25323 |
| Canonical config | `configs/final.yaml`, exactly equal to checkpoint `config` |

The checkpoint contains `model`, `ema`, `config`, `iter`, `metrics`, `git`, and `provenance`.
Both state dictionaries load strictly into the embedded architecture. Normal submission
inference requires this checkpoint; `--allow_bicubic_fallback` is an explicit demo-only mode,
and `--require_weights` is retained.

## Provenance normalization

The model was recovered from the public `artifacts-v1` release checkpoint:

- Original URL: https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/best.pt
- Original SHA256: `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313`
- Original size: 3,288,805 bytes
- Original training marker: `80e7fb049367afe99fbcabb8e5469861f630fecc-dirty`

The tracked file preserves that dirty marker rather than claiming the original tree was clean.
Its `provenance` block pins canonical training source commit
`80e7fb049367afe99fbcabb8e5469861f630fecc` and records the Git blob ID plus SHA256 for every
relevant training source/config file. `configs/final.yaml` expands defaults that were implicit
in the original metadata, including zero padding and all loss settings.

Model and EMA tensors are byte-identical between the original release asset and the tracked
checkpoint. Only config/provenance metadata was normalized. Reproduce the metadata operation
with `scripts/normalize_checkpoint_metadata.py` after obtaining an original byte-identical
release asset; the script refuses an unexpected original digest and writes atomically.

The original URL remains documented as independent provenance and disaster recovery, not as a
required installation step.

## Canonical training

```bash
python train.py --config configs/final.yaml --data_root /path/to/dataset --seed 42
```

Canonical optimizer/training values are AdamW, LR `1e-3`, betas `(0.9,0.9)`, no weight decay,
500 warmup iterations, cosine decay to `1e-6`, 20,000 total iterations, bf16, EMA `0.999`, and
seed 42. Every loss default is explicit in the YAML.

New training checkpoints carry complete resume state. This historical release checkpoint does
not carry optimizer/RNG state and is therefore inference-only; `train.py --resume` rejects it
with a clear error instead of pretending continuation is equivalent.

## Verify

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('weights/best.pt'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
python inference.py --input_dir sample_inputs --output_dir /tmp/kla-weight-check --require_weights
```
