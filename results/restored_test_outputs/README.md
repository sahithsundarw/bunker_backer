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

## Status — 2026-08-15, iteration 1

**The outputs do not exist yet.** No training run has completed, so there is no checkpoint and
therefore nothing to restore. The manifest, the archive, its sha256 and the Release URL are
all genuinely unknown — every field below is an explicit placeholder, not a value that was
measured and then omitted, and **nothing here is fabricated**.

`inference.py` currently falls back to a parameter-free bicubic ×2 upsample when the
checkpoint is missing. **Bicubic fallback output must never be published in this folder.** Any
run that produces the artifact described here must be invoked with `--require_weights`, so a
missing or unloadable checkpoint fails loudly instead of silently shipping an upsampler's
output as a model result.

## Provenance — what produced these outputs

| field | value |
|---|---|
| Inputs | the **400 released test inputs**, `test_NoisyLR/000000.npy` … `000399.npy` |
| Input properties | `.npy`, `float32`, 2-D `(128, 128)`, grayscale, values **not** clipped on input (observed range `[-0.28, 2.16]`) |
| Outputs | 400 files, `.npy`, `float32`, `(256, 256)` — exactly 2× — clipped to `[0, 1]`, no renormalisation |
| Filenames | **byte-identical** to the inputs; no suffix, no extension change |
| Ground truth | **none exists.** The released test set ships inputs only, so no score can be computed against it locally. Every metric in this repository is measured on a held-out split of `train/` |
| Checkpoint used | *pending — `weights/best.pt`, sha256 to be recorded in `weights/README.md`* |
| Command used | *pending — `python inference.py --input_dir <test_NoisyLR> --output_dir results/restored_test_outputs --require_weights`* |
| Git SHA of the producing run | *pending — also embedded in the checkpoint under `git`* |

**Filename collision hazard.** All 400 test filenames also exist under `train/`, referring to
different images. Never key a cache, manifest or results structure on a bare filename — qualify
it by split or use the full path. Both sets share shape and dtype, so a collision produces
silently wrong results rather than an exception.

## Download and verify

Releases page (live, returns HTTP 200): `https://github.com/sahithsundarw/semicon-kla-image-restoration/releases`

| field | value |
|---|---|
| Release tag | *pending* |
| Asset name | *pending* |
| Asset URL | *pending — do not fabricate; must return HTTP 200 from a logged-out session* |
| sha256 of the asset | *pending — must be the digest of the served bytes, not of a local copy* |
| Asset size (bytes) | *pending* |
| File count inside | *pending — must be 400* |

Once the fields above are filled in, a reviewer verifies the download like this. The URL and
digest below are placeholders, so this block **cannot be executed yet** — it is recorded here
rather than in the root `README.md`, where every fenced command is extracted and executed by
V46 and must actually run.

```
curl -L -o restored_test_outputs.zip <ASSET_URL>
sha256sum restored_test_outputs.zip          # must equal the sha256 above
```

On Windows without `sha256sum`, either of these prints the same digest — both were run
against a committed file in this repository to confirm the syntax, and they agree:

```
py -3.12 -c "import hashlib,pathlib,sys;print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())" restored_test_outputs.zip
powershell -Command "(Get-FileHash restored_test_outputs.zip -Algorithm SHA256).Hash.ToLower()"
```

## Manifest

`manifest.csv` (to be generated alongside the outputs) lets a reviewer verify the extracted
archive file by file, without trusting the archive digest alone:

```
filename,sha256,shape,dtype,min,max
000000.npy,<sha256>,"(256, 256)",float32,0.0,1.0
...
```

Every row is asserted at generation time to satisfy the output contract in
`docs/io_contract.md`: `float32`, `ndim == 2`, exactly 2× the corresponding input, no NaN, no
Inf, `min >= 0.0`, `max <= 1.0` — measured on the file **reloaded from disk**, not on the
in-memory tensor, so any dtype or quantisation loss is caught here rather than by KLA.

## Checklist before submission

- [ ] 400 outputs produced with `--require_weights` (no bicubic fallback in play)
- [ ] archive uploaded as a Release asset; URL fetched successfully from a **logged-out**
      session
- [ ] asset sha256 recorded above and in `weights/README.md`, matching the served bytes
- [ ] `manifest.csv` committed here with 400 rows and per-file hashes
- [ ] every row satisfies the `docs/io_contract.md` contract, verified from disk
- [ ] this README updated so the "outputs do not exist yet" status no longer stands
