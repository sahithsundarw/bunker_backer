# `results/restored_test_outputs/` — restored outputs for the released test set

SPEC F12 makes this folder a mandatory repository item and V13 asserts it is non-empty.

---

## ⚠ Read this first: what is actually in this folder

**This folder contains a manifest and per-file sha256 hashes. It does not contain the restored
`.npy` files themselves.** The output archive is prepared for publication as a **GitHub
Release asset**, but that publication step is still pending.

That is said plainly, up front, because a "non-empty" folder that leaves a reviewer believing
the outputs are committed would satisfy V13's letter and defeat its purpose.

**Why.** 400 outputs at 256×256 float32 is ≈91 MB raw (archived). Committing them — even
compressed into a single `.npz` — would require loosening V51's per-file and total-tree caps,
which were added specifically to stop a dataset-sized blob entering the tree. Weakening a check
because it is in the way is a stop signal, not a justification. Git LFS was ruled out separately:
an unresolved LFS pointer stub on a fresh clone is a known way to fail V06, whose own text names
that failure mode.

A Release asset plus a published sha256 needs **no contract change at all** — it is exactly the
mechanism V06 already permits for the model checkpoint. Full reasoning: `docs/decisions.md` D23
(superseding D17) and `docs/BLOCKERS.md` B9.

---

## Status — 2026-08-15, `codex/final-submission-28db`

**The outputs exist and have been generated from the 28.0394 dB checkpoint.** `manifest.csv`
in this directory is real, measured data — 400 rows, each computed from the actual restored
`.npy` file reloaded from disk. The only remaining step is uploading the archive as a Release
asset (see "Manual step" below); everything else in this document is a measured fact, not a
placeholder.

Normal `inference.py` submission runs fail if the checkpoint is missing or cannot be loaded.
The parameter-free bicubic baseline is available only through the explicit demo flag
`--allow_bicubic_fallback` and must never be published in this folder. The run that produced
these outputs used `--require_weights`, so a missing or unloadable checkpoint would have
failed loudly.

## Provenance — what produced these outputs

| field | value |
|---|---|
| Inputs | the **400 released test inputs**, `/Users/shanmukhsai/Downloads/NoisyLR/000000.npy` … `000399.npy` |
| Input properties | `.npy`, `float32`, 2-D `(128, 128)`, grayscale |
| Outputs | 400 files, `.npy`, `float32`, `(256, 256)` — exactly 2× — clipped to `[0, 1]` |
| Filenames | **byte-identical** to the inputs; no suffix, no extension change (verified: `filenames_match_inputs = true` for all 400 rows) |
| Ground truth | **none exists.** The released test set ships inputs only, so no score can be computed against it locally. Every metric below is measured on the held-out validation split of `train/` |
| Checkpoint used | `weights/best.pt`, sha256 `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7` |
| Checkpoint validation metrics (disk-verified, full 400-pair val split — **not** a final-test score) | PSNR 28.0394 dB / SSIM 0.74804 / LPIPS 0.29571 |
| Command used | `python inference.py --input_dir /Users/shanmukhsai/Downloads/NoisyLR --output_dir /tmp/semicon_final_outputs_28db --weights weights/best.pt --require_weights --batch_size 32 --device cpu --precision fp32 --verbose` |
| Runtime headline | **Local Mac CPU external-process benchmark: 400 images in 71.72 s (5.6 img/s), batch size 32, fp32.** Process creation through exit; not a Linux/CUDA or H100 benchmark. |
| Release-output generation log | **Local Mac CPU, internal `main()` timer:** `loaded weights/best.pt (ema weights)` / `restored 400/400 in 56.73s (7.1 img/s) \| device=cpu precision=fp32 batch=32 shapes=[(128, 128)] weights=best unreadable=0 write_errors=0`. This records the producing run and is not the headline benchmark. |
| Git SHA of the producing run | `a2694c5e9e99914e1604eee1f83110f0a38113db` (also embedded in the checkpoint under `git`) |

**No PSNR/SSIM/LPIPS is computed on these 400 outputs.** The final test set has no ground
truth, so no such score is possible; the 28.0394 dB / 0.74804 / 0.29571 figures above describe
the checkpoint's validation-split performance, not this folder's outputs.

**Filename collision hazard.** All 400 test filenames also exist under `train/`, referring to
different images. Never key a cache, manifest or results structure on a bare filename — qualify
it by split or use the full path. Both sets share shape and dtype, so a collision produces
silently wrong results rather than an exception.

## Download and verify

Releases page: `https://github.com/sahithsundarw/semicon-kla-image-restoration/releases`

| field | value |
|---|---|
| Archive name | `semicon_final_outputs_28db.zip` |
| Archive sha256 | `a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98` |
| Archive size (bytes) | 91,051,052 |
| File count inside | 400 |
| Release asset URL | *pending — remaining manual step, see below* |

Once the asset URL is filled in (`manifest.json.release_url`), a reviewer verifies the download
like this:

```
curl -L -o semicon_final_outputs_28db.zip <ASSET_URL>
sha256sum semicon_final_outputs_28db.zip   # must equal a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98
```

On Windows without `sha256sum`, either of these prints the same digest:

```
py -3.12 -c "import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" semicon_final_outputs_28db.zip
powershell -Command "(Get-FileHash semicon_final_outputs_28db.zip -Algorithm SHA256).Hash.ToLower()"
```

## Manifest

`manifest.csv` in this directory lets a reviewer verify the extracted archive file by file,
without trusting the archive digest alone. Header and first row:

```
filename,sha256,shape,dtype,min,max,finite,matching_input_exists
000000.npy,<sha256>,"(256, 256)",float32,<min>,<max>,true,true
```

Every row was measured on the file **reloaded from disk**, not the in-memory tensor: `float32`,
shape `(256, 256)`, all-finite, `min >= 0.0`, `max <= 1.0`, and a matching input file confirmed
present. All 400 rows satisfy every condition — verified with a NumPy scan of the on-disk
outputs immediately after generation.

`manifest.json` records archive-level provenance (checkpoint SHA, command, archive SHA/size,
output contract). Its `release_url` is empty until the manual upload step below is completed.

## Manual step remaining

Upload `/tmp/semicon_final_outputs_28db.zip` as a GitHub Release/submission asset, verify the
downloaded bytes reproduce sha256 `a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98`,
then set `manifest.json.release_url` to the public asset URL. The archive itself is
intentionally not committed to Git (see "what is actually in this folder" above).

## Checklist before submission

- [x] 400 outputs produced with `--require_weights` (no bicubic fallback in play)
- [ ] archive uploaded as a Release asset; URL fetched successfully from a **logged-out**
      session
- [ ] asset sha256 recorded in `manifest.json.release_url` field once uploaded, matching the
      served bytes (the sha256 itself is already recorded above and in `manifest.json`)
- [x] `manifest.csv` committed here with 400 rows and per-file hashes
- [x] every row satisfies the `docs/io_contract.md` contract, verified from disk
- [x] this README updated so the "outputs do not exist yet" status no longer stands
