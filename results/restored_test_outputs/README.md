# `results/restored_test_outputs/` — restored outputs for the released test set

SPEC F12 makes this folder a mandatory repository item and V13 asserts it is non-empty.

---

## ⚠ Read this first: what is actually in this folder

**This folder contains a manifest and per-file sha256 hashes. It does not contain the restored
`.npy` files themselves.** The output archive is published as a **GitHub Release asset**
(currently `artifacts-v3`), verified fetchable from a logged-out session before this document
was last updated.

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

## Status — 2026-08-17, Round 2 Phase 3 structural-content checkpoint (`docs/decisions.md` D71/D72)

**The outputs exist, are published, and have been regenerated from the promoted 29.5850 dB
Phase 3 checkpoint**, which superseded the prior long-run checkpoint (29.2548 dB) after a
paired comparison showing wins/ties on every metric on every evaluation set, including the
first genuine real-SEM-OOD improvement of the whole investigation (`docs/decisions.md`
D71/D72), using the real `inference.py --require_weights` production entrypoint, not a
re-implemented forward pass. Published as a **new** GitHub Release (`artifacts-v3`) rather
than overwriting `artifacts-v2`, which remains available unchanged as the historical prior
checkpoint's record (as does `artifacts-v1` before it).

Normal `inference.py` submission runs fail if the checkpoint is missing or cannot be loaded.
The parameter-free bicubic baseline is available only through the explicit demo flag
`--allow_bicubic_fallback` and must never be published in this folder. The run that produced
these outputs used `--require_weights`, so a missing or unloadable checkpoint would have
failed loudly.

## Provenance — what produced these outputs (current)

| field | value |
|---|---|
| Inputs | the **400 released test inputs**, `C:\kla-data\test_NoisyLR\000000.npy` … `000399.npy` |
| Input properties | `.npy`, `float32`, 2-D `(128, 128)`, grayscale |
| Outputs | 400 files, `.npy`, `float32`, `(256, 256)` — exactly 2× — clipped to `[0, 1]` |
| Filenames | **byte-identical** to the inputs; no suffix, no extension change (verified: `matching_input_exists = true` for all 400 rows) |
| Ground truth | **none exists.** The released test set ships inputs only, so no score can be computed against it locally. Every metric below is measured on the held-out validation split of `train/` |
| Checkpoint used | `weights/best.pt`, sha256 `6d74ccfdd72e1271a7de5fdede5c341b3cf18ca4294619dd90a97c0591f66397` |
| Checkpoint validation metrics (disk-verified, full 400-pair val split — **not** a final-test score) | PSNR 29.5850 dB / SSIM 0.79460 / LPIPS 0.25416 |
| Command used | `python inference.py --input_dir C:\kla-data\test_NoisyLR --output_dir <out> --require_weights --verbose` |
| Runtime headline | See `results/runtime_report.md` — **NVIDIA GeForce RTX 4060 Laptop GPU**, not Mac CPU, not H100. |
| Git SHA of the producing run | `eb0849e7ff54cf54cb9bdf4465a9f1133fa4b4ea` (this checkpoint's own `git` key genuinely records the training commit, `c8f3a51b...` — the HF Jobs container cloned via git this time, not a tarball snapshot; see `weights/README.md`) |

**No PSNR/SSIM/LPIPS is computed on these 400 outputs.** The final test set has no ground
truth, so no such score is possible; the figures above describe the checkpoint's
validation-split performance, not this folder's outputs.

**Filename collision hazard.** All 400 test filenames also exist under `train/`, referring to
different images. Never key a cache, manifest or results structure on a bare filename — qualify
it by split or use the full path. Both sets share shape and dtype, so a collision produces
silently wrong results rather than an exception.

## Download and verify (current, `artifacts-v3`)

| field | value |
|---|---|
| Archive name | `restored_test_outputs.zip` |
| Archive sha256 | `7c5a63ff8720bbbbf781891c6fdb1302bc925095806278766ad08ca2abe9c6ef` |
| Archive size (bytes) | 90,929,851 |
| File count inside | 400 |
| Release asset URL | `https://github.com/sahithsundarw/bunker_backer/releases/download/artifacts-v3/restored_test_outputs.zip` |

Verified fetchable from a **logged-out** session with the sha256 above reproduced exactly
(via `curl`) before this README was updated.

```bash
curl -L -o restored_test_outputs.zip https://github.com/sahithsundarw/bunker_backer/releases/download/artifacts-v3/restored_test_outputs.zip
sha256sum restored_test_outputs.zip   # must equal 7c5a63ff8720bbbbf781891c6fdb1302bc925095806278766ad08ca2abe9c6ef
```

---

## Superseded — 2026-08-17, Round 2 long-run checkpoint (`docs/decisions.md` D61)

The tables below this line describe the D61 long-run checkpoint's (29.2548 dB) outputs, still
published at `artifacts-v2` for anyone wanting to reproduce that exact comparison. They are
NOT what `weights/best.pt` currently produces.

| field | value |
|---|---|
| Checkpoint used | `weights/best.pt`, sha256 `8f54f9a208220dfd6cd3d67766945ad781bf141fcc03fac41d216caf4fa9643c` |
| Checkpoint validation metrics | PSNR 29.2548 dB / SSIM 0.79211 / LPIPS 0.25625 |
| Archive | `artifacts-v2/restored_test_outputs.zip`, sha256 `6355b2bf802d0d7817d6c42d10893dff96e99285f2b03b4888c2a6310a8e7364`, 90,963,266 bytes |
| Git SHA | `2e530586c55f9baf5ad92154d319534226adaf73` (checkpoint's own `git` key recorded `unknown` — tarball snapshot, not a git clone) |

---

## Superseded — 2026-08-16, reconciled (`docs/decisions.md` D49)

The tables below this line describe the PRIOR (28.7865 dB) checkpoint's outputs, still
published at `artifacts-v1` for anyone wanting to reproduce that exact comparison. They are
NOT what `weights/best.pt` currently produces, and NOT what `manifest.csv`/`manifest.json` in
this directory currently describe (those now reflect the current checkpoint above).

## Provenance — what produced these outputs (superseded, D49)

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

## Download and verify (superseded, `artifacts-v1`)

Releases page: `https://github.com/sahithsundarw/bunker_backer/releases`

| field | value |
|---|---|
| Archive name | `restored_test_outputs.zip` |
| Archive sha256 | `b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1` |
| Archive size (bytes) | 90,990,452 |
| File count inside | 400 |
| Release asset URL | `https://github.com/sahithsundarw/bunker_backer/releases/download/artifacts-v1/restored_test_outputs.zip` |

Verified fetchable from a **logged-out** session (plain `urllib.request`, no auth header) with
the sha256 above reproduced exactly before this README was written.

```
curl -L -o restored_test_outputs.zip https://github.com/sahithsundarw/bunker_backer/releases/download/artifacts-v1/restored_test_outputs.zip
sha256sum restored_test_outputs.zip   # must equal b1d3a581c93d6a609ccc5146d7af82c1188f04dabfaa0e2672361912674954a1
```

On Windows without `sha256sum`, either of these prints the same digest:

```
py -3.12 -c "import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" restored_test_outputs.zip
powershell -Command "(Get-FileHash restored_test_outputs.zip -Algorithm SHA256).Hash.ToLower()"
```

## Manifest (current — describes the checkpoint in the "current" section above, not the superseded one)

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
