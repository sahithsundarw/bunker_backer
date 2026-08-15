# `results/restored_test_outputs/` — restored outputs for the released test set

SPEC F12 makes this folder a mandatory repository item and V13 asserts it is non-empty.

---

## ⚠ Read this first: what is actually in this folder

**This folder contains a manifest and per-file sha256 hashes. It does not contain the restored
`.npy` files themselves.** The outputs are published as a **GitHub Release asset** and are
downloaded on demand.

That is said plainly, up front, because a "non-empty" folder that leaves a reviewer believing
the outputs are committed would satisfy V13's letter and defeat its purpose.

**Why.** 400 outputs at 256×256 float32 is ≈105 MB raw. Committing them — even compressed into
a single `.npz` — would require loosening V51's 5 MB per-file and 25 MB total-tree caps, which
were added one commit earlier specifically to stop a dataset-sized blob entering the tree.
Weakening a check because it is in the way is a stop signal, not a justification. Git LFS was
ruled out separately: an unresolved LFS pointer stub on a fresh clone is a known way to fail
V06, whose own text names that failure mode.

A Release asset plus a published sha256 needs **no contract change at all** — it is exactly
the mechanism V06 already permits for the model checkpoint. Full reasoning:
`docs/decisions.md` D23 (superseding D17) and `docs/BLOCKERS.md` B9.

---

## Status — 2026-08-15, iteration 2

**The outputs exist and are published.** All 400 were produced on 2026-08-15 by the shipped
`inference.py` with **`--require_weights`**, so the bicubic fallback provably was not in play:
the run log line reads `loaded weights/best.pt (ema weights)` and the flag turns a missing or
unloadable checkpoint into a hard failure rather than a silent upsample. Machine-readable
provenance is in `manifest.json`; per-file digests are in `sha256sums.txt`.

`inference.py` falls back to a parameter-free bicubic ×2 upsample when the checkpoint is
missing. **Bicubic fallback output must never be published in this folder** — that prohibition
is now enforced in code, not just stated here: V56 fails if `manifest.json`'s recorded
`command` does not contain `--require_weights`.

## Provenance — what produced these outputs

| field | value |
|---|---|
| Inputs | the **400 released test inputs**, `test_NoisyLR/000000.npy` … `000399.npy` |
| Input properties | `.npy`, `float32`, 2-D `(128, 128)`, grayscale, values **not** clipped on input (observed range `[-0.28, 2.16]`) |
| Outputs | 400 files, `.npy`, `float32`, `(256, 256)` — exactly 2× — clipped to `[0, 1]`, no renormalisation |
| Filenames | **byte-identical** to the inputs; no suffix, no extension change |
| Ground truth | **none exists.** The released test set ships inputs only, so no score can be computed against it locally. Every metric in this repository is measured on a held-out split of `train/` |
| Checkpoint used | `weights/best.pt`, **EMA** weights, sha256 `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` (also in `weights/README.md`) |
| Command used | `py -3.12 inference.py --input_dir C:/kla-data/test_NoisyLR --output_dir <out> --require_weights --verbose` |
| Git SHA of the producing run | `c209cd213f6b4df0f5a6676a1671d5a8828a057c`, working tree clean |
| Runtime | 400 images in **20.09 s** (19.9 img/s), `device=cuda precision=bf16 batch=32`, RTX 4060 Laptop; 0 unreadable inputs, 0 write errors |
| Verified on reload | all 400 re-loaded from disk: `float32`, `ndim==2`, `(256, 256)`, all finite, observed global range exactly `[0.000000, 1.000000]`, filename set identical to the input set. **0 violations.** |

**Filename collision hazard.** All 400 test filenames also exist under `train/`, referring to
different images. Never key a cache, manifest or results structure on a bare filename — qualify
it by split or use the full path. Both sets share shape and dtype, so a collision produces
silently wrong results rather than an exception.

## Download and verify

Releases page (live, returns HTTP 200): `https://github.com/sahithsundarw/semicon-kla-image-restoration/releases`

| field | value |
|---|---|
| Release tag | `artifacts-v1` |
| Asset name | `restored_test_outputs.zip` |
| Asset URL | https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip |
| sha256 of the asset | `fbdf8a652d26168cf41e01842ca28d38c53d1da1547bd8ce602b5b8e5d6ac750` |
| Asset size (bytes) | 91069597 |
| File count inside | 400, flat at the archive root, no directory prefix |

**The digest is of the served bytes, not of the local copy.** The asset was re-fetched with
`GITHUB_TOKEN` and `GH_TOKEN` cleared from the environment, so the fetch could not have
succeeded on cached credentials: HTTP **200**, **91069597** bytes downloaded, and the sha256 of
what the server returned equals the digest above. Reproduce it:

```
curl -L -o restored_test_outputs.zip https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip
sha256sum restored_test_outputs.zip          # must equal the sha256 above
```

These commands are recorded here rather than in the root `README.md`, where every fenced
command is extracted and executed by V46 — a 91 MB download does not belong in a verification
run.

On Windows without `sha256sum`, either of these prints the same digest — both were run
against a committed file in this repository to confirm the syntax, and they agree:

```
py -3.12 -c "import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" restored_test_outputs.zip
powershell -Command "(Get-FileHash restored_test_outputs.zip -Algorithm SHA256).Hash.ToLower()"
```

## Manifest

Two committed files let a reviewer verify the extracted archive file by file, without
trusting the archive digest alone:

- **`manifest.json`** — machine-readable provenance. This is the file V56 reads. It carries
  `release_url`, `archive_sha256`, `n_files`, `producing_git_sha`, `checkpoint_sha256` and the
  exact `command`, plus the runtime and the on-reload validation result.
- **`sha256sums.txt`** — 400 rows, `<sha256>  <filename>`, sorted by filename. After
  extracting the archive, `sha256sum -c sha256sums.txt` checks every output individually.

`manifest.csv` was the format originally sketched here; `manifest.json` shipped instead because
V56 requires a machine-checkable JSON and explicitly rejects a `.csv`.

Every row is asserted at generation time to satisfy the output contract in
`docs/io_contract.md`: `float32`, `ndim == 2`, exactly 2× the corresponding input, no NaN, no
Inf, `min >= 0.0`, `max <= 1.0` — measured on the file **reloaded from disk**, not on the
in-memory tensor, so any dtype or quantisation loss is caught here rather than by KLA.

## Checklist before submission

- [x] 400 outputs produced with `--require_weights` (no bicubic fallback in play)
- [x] archive uploaded as a Release asset; URL fetched successfully from a **logged-out**
      session — HTTP 200, 91069597 bytes, digest matched
- [x] asset sha256 recorded above and in `weights/README.md`, matching the served bytes
- [x] per-file hashes committed here — `sha256sums.txt`, 400 rows (as `manifest.json` +
      `sha256sums.txt`, not `manifest.csv`; V56 requires JSON)
- [x] every row satisfies the `docs/io_contract.md` contract, verified from disk — 400/400,
      0 violations
- [x] this README updated so the "outputs do not exist yet" status no longer stands
