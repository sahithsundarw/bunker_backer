# I/O Contract — **PROVISIONAL**

**Status: PROVISIONAL.** Everything about the *input* side is measured from the real files
and is firm. The *output* side is derived by symmetry with GT, **not** from any submission
spec — `KLA_IMAGE_RESTORATION_MASTER_SPEC.md` was not found on this machine, so the
authoritative output/scoring requirements have not been read. See `docs/MISSING_INPUTS.md`.

Anything marked **[UNCONFIRMED]** must be re-checked against SPEC once it is recovered.

---

## Input contract (measured — firm)

| Property | Value | Evidence |
|---|---|---|
| Container | `.npy`, NumPy binary | 6800/6800 files, no other extension |
| Load call | `np.load(path, allow_pickle=False)` | arrays are plain, no object dtype |
| dtype | `float32` | 200/200 sampled per folder |
| Shape | `(128, 128)`, 2-D, single channel | full scan: `{(128,128): 3200}` and `{(128,128): 400}` |
| Layout | H, W — no channel axis | `ndim == 2` for 200/200 sampled |
| Value range | **unbounded float**, empirically `[-0.28, 2.16]` | 3.0% of pixels > 1.0, 0.3–0.7% < 0.0 |
| Scaling | already in ~[0,1] units; **do not rescale, do not clip** | see below |

> **Do not clamp the input.** ~3% of NoisyLR pixels legitimately exceed 1.0 and ~0.5% fall
> below 0.0. Clipping the input to [0,1] discards real signal. Normalise-in / denormalise-out
> is a no-op here — the data is already in the right units.

## Target contract (measured — firm)

| Property | Value | Evidence |
|---|---|---|
| Container | `.npy` | 3200/3200 |
| dtype | `float32` | 200/200 sampled |
| Shape | `(256, 256)` = exactly 2× the input in both axes | full scan, 0 violations across 3200 pairs |
| Value range | **exactly `[0.0, 1.0]`, closed** | full scan of all 3200: min==0.0 for 3200, max==1.0 for 3200 |
| Normalisation | **per-image min–max to [0,1]** | every single GT file attains both endpoints |
| Alignment | pixel-aligned with input; centre-aligned 2×2 average-pool decimation | best NCC shift `(0,0)` on 12/12 sampled pairs |

---

## Output contract (derived from GT — **PROVISIONAL**)

Produce, for each file in `C:\kla-data\test_NoisyLR`, one restored array that matches the
GT contract exactly:

| Property | Required value | Confidence |
|---|---|---|
| Format | `.npy`, `np.save(path, arr)` | **[UNCONFIRMED]** — mirrors GT; SPEC may demand PNG/TIFF instead |
| dtype | `float32` | firm (matches GT) |
| Shape | `(256, 256)` — 2× the `(128,128)` input | firm |
| Value range | clipped to `[0.0, 1.0]` | firm — GT never leaves this range |
| Scaling | `[0,1]` float, **not** 0–255 | firm |
| Filename | **same basename as the input**: `test_NoisyLR/000123.npy` → `000123.npy` | **[UNCONFIRMED]** — mirrors the train pairing rule |
| Destination | `results\restored_test_outputs\` | from the task brief |
| Count | 400 files, `000000.npy` … `000399.npy` | firm |

### Reference writer

```python
import os
import numpy as np

OUT_DIR = r"results\restored_test_outputs"

def save_restored(basename: str, arr: np.ndarray) -> None:
    """basename: the input filename, e.g. '000123.npy'."""
    assert arr.shape == (256, 256), arr.shape
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, basename), arr)
```

`np.save` appends `.npy` if absent; passing `'000123.npy'` yields `000123.npy`, not
`000123.npy.npy`.

### The per-image normalisation question — **[UNCONFIRMED], and it matters**

Every GT image attains exactly 0.0 and exactly 1.0, which means GT was **min–max normalised
per image** before release. The consequence:

- If the metric is computed against these normalised GT arrays, then output that spans the
  full [0,1] range scores better, and a final per-image min–max renormalisation of the
  prediction is *arguably* the matched choice.
- If the metric instead compares against un-normalised originals, renormalising would be wrong.

**Do not renormalise output per-image until SPEC §scoring is read.** The safe default,
encoded above, is a plain `clip(0,1)` with no renormalisation — it cannot be worse than
mildly conservative, whereas a wrong renormalisation is a systematic error on all 400 files.

### Open items to resolve against SPEC

1. Output container — `.npy` vs an image format.
2. Whether outputs go in a flat directory, a zip, or a named subfolder.
3. Whether per-image min–max renormalisation is expected.
4. The metric (PSNR / SSIM / LPIPS) and whether it is computed on [0,1] floats or 8-bit.
5. Whether `float32` or `float64` is accepted.
