# `results/restored_test_outputs/` — restored outputs for the released test set

SPEC F12 makes this folder a mandatory repository item and V13 asserts it is non-empty.

---

## ⚠ Read this first: what is actually in this folder

**This folder contains a manifest and per-file sha256 hashes. It does not contain the restored
`.npy` files themselves.** The output archive is published as a **GitHub Release asset**
(`artifacts-v1`), verified fetchable from a logged-out session before this document was
written.

**Why not committed directly.** 400 outputs at 256×256 float32 is ≈91 MB archived. Committing
them — even compressed into a single `.npz` — would require loosening V51's per-file and
total-tree caps, which were added specifically to stop a dataset-sized blob entering the tree.
Weakening a check because it is in the way is a stop signal, not a justification. Git LFS was
ruled out separately: an unresolved LFS pointer stub on a fresh clone is a known way to fail
V06, whose own text names that failure mode.

A Release asset plus a published sha256 needs **no contract change at all** — it is exactly the
mechanism V06 already permits for the model checkpoint. Full reasoning: `docs/decisions.md` D23
(superseding D17) and `docs/BLOCKERS.md` B9.

---

## Status — 2026-08-16, reconciled (`docs/decisions.md` D49)

**The outputs exist, are published, and have been generated from the shipped 28.7865 dB
checkpoint.** This repo briefly diverged into two lines of work with two different checkpoints;
these outputs were regenerated from the checkpoint that won the head-to-head re-score
(`docs/decisions.md` D49), using the real `inference.py --require_weights` production
entrypoint, not a re-implemented forward pass.

Normal `inference.py` submission runs fail if the checkpoint is missing or cannot be loaded.
The parameter-free bicubic baseline is available only through the explicit demo flag
`--allow_bicubic_fallback` and must never be published in this folder. The run that produced
these outputs used `--require_weights`, so a missing or unloadable checkpoint would have
failed loudly.

## Provenance — what produced these outputs

| field | value |
|---|---|
| Inputs | the **400 released test inputs**, `C:\kla-data\test_NoisyLR\000000.npy` … `000399.npy` |
| Input properties | `.npy`, `float32`, 2-D `(128, 128)`, grayscale |
| Outputs | 400 files, `.npy`, `float32`, `(256, 256)` — exactly 2× — clipped to `[0, 1]` |
| Filenames | **byte-identical** to the inputs; no suffix, no extension change (verified: `matching_input_exists = true` for all 400 rows) |
| Ground truth | **none exists.** The released test set ships inputs only, so no score can be computed against it locally. Every metric below is measured on the held-out validation split of `train/` |
| Checkpoint used | `weights/best.pt`, sha256 `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` |
| Checkpoint validation metrics (disk-verified, full 400-pair val split — **not** a final-test score) | PSNR 28.7865 dB / SSIM 0.78287 / LPIPS 0.25324 |
| Command used | `python inference.py --input_dir C:\kla-data\test_NoisyLR --output_dir <out> --require_weights --verbose` |
| Runtime headline | See `results/runtime_report.md` — **NVIDIA GeForce RTX 4060 Laptop GPU**, not Mac CPU, not H100. |
| Producing-run log | `loaded weights/best.pt (ema weights)` / `restored 400/400 in 25.08s (15.9 img/s) \| device=cuda precision=bf16 batch=32 shapes=[(128, 128)] weights=best unreadable=0 write_errors=0` |
| Git SHA of the producing run | `4eeeb2e1e145fd25c3d61300a4de08e9932dcc82` (merge-reconciliation commit; checkpoint's own `git` key records its training-time SHA `80e7fb0…`) |

**No PSNR/SSIM/LPIPS is computed on these 400 outputs.** The final test set has no ground
truth, so no such score is possible; the 28.7865 dB / 0.78287 / 0.25324 figures above describe
the checkpoint's validation-split performance, not this folder's outputs.

**Filename collision hazard.** All 400 test filenames also exist under `train/`, referring to
different images. Never key a cache, manifest or results structure on a bare filename — qualify
it by split or use the full path. Both sets share shape and dtype, so a collision produces
silently wrong results rather than an exception.

## Download and verify

Releases page: `https://github.com/sahithsundarw/semicon-kla-image-restoration/releases`

| field | value |
|---|---|
| Archive name | `restored_test_outputs.zip` |
| Archive sha256 | `b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1` |
| Archive size (bytes) | 90,990,452 |
| File count inside | 400 |
| Release asset URL | `https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip` |

Verified fetchable from a **logged-out** session (plain `urllib.request`, no auth header) with
the sha256 above reproduced exactly before this README was written.

```
curl -L -o restored_test_outputs.zip https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip
sha256sum restored_test_outputs.zip   # must equal b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1
```

On Windows without `sha256sum`, either of these prints the same digest:

```
py -3.12 -c "import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" restored_test_outputs.zip
powershell -Command "(Get-FileHash restored_test_outputs.zip -Algorithm SHA256).Hash.ToLower()"
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
output contract), including the live `release_url`.

## Checklist before submission

- [x] 400 outputs produced with `--require_weights` (no bicubic fallback in play)
- [x] archive uploaded as a Release asset; URL fetched successfully from a **logged-out**
      session
- [x] asset sha256 recorded in `manifest.json.release_url`, matching the served bytes
- [x] `manifest.csv` committed here with 400 rows and per-file hashes
- [x] every row satisfies the `docs/io_contract.md` contract, verified from disk
- [x] this README updated so the "outputs do not exist yet" status no longer stands
