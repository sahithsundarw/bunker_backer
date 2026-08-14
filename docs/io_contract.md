# I/O Contract — **FINAL**

**Status: FINAL.** Derived from the real files at `C:\kla-data` and reconciled against
`docs/SPEC.md` §5.1, §11.1, F5 and F6. Nothing below is provisional.

SPEC §5.1 states the rule this document instantiates:

> Output format = exactly the GT format. Output dtype = exactly the GT dtype. Output
> filename = exactly the input filename. Output size = 2× input size. Values clipped to
> [0,1] before dtype conversion.

GT is `.npy` `float32`. Therefore output is `.npy` `float32`. SPEC §5.1 also warns:
*"If GT is float32 `.tif` or `.npy`, saving PNG is a scoring catastrophe."*

---

## Input contract (measured)

| Property | Value | Evidence |
|---|---|---|
| Container | `.npy` NumPy binary | 6800/6800 files; zero other extensions |
| Load call | `np.load(path, allow_pickle=False)` | plain arrays, no object dtype |
| dtype | `float32` | 200/200 sampled per folder |
| Shape | `(128, 128)`, 2-D, single channel | full scan: `{(128,128): 3200}` and `{(128,128): 400}` |
| Layout | `H, W` — no channel axis | `ndim == 2`, 200/200 sampled |
| Value range | **unbounded**; observed `[-0.28, 2.16]` | 3.0% of pixels > 1.0, 0.3–0.7% < 0.0 |
| Scaling | already in [0,1]-style units | do not rescale |

> ### Do NOT clip the input — SPEC F5
> ~3% of NoisyLR pixels legitimately exceed 1.0 and ~0.5% fall below 0.0. F5 states this is
> intentional and carries information. Clipping the input destroys it and creates a
> train/test mismatch if applied in only one place (SPEC §18 pitfall 2).

## Target contract (measured)

| Property | Value | Evidence |
|---|---|---|
| Container | `.npy` | 3200/3200 |
| dtype | `float32` | 200/200 sampled |
| Shape | `(256, 256)` = exactly 2× input | full scan, 0 violations across 3200 pairs |
| Value range | **exactly `[0.0, 1.0]`, closed** | full scan: min==0.0 in 3200/3200, max==1.0 in 3200/3200 |
| Alignment | pixel-aligned, centre-aligned decimation | best NCC shift `(0,0)` on 12/12 sampled pairs |

---

## Output contract — BINDING

For every file in the input directory, emit exactly one output:

| Property | Required value |
|---|---|
| Format | **`.npy`**, written with `np.save` |
| dtype | **`float32`** |
| Shape | **exactly 2× the input** in both spatial axes — `(2H, 2W)` |
| Value range | **clipped to `[0.0, 1.0]`** |
| Scaling | `[0,1]` float. **Not** 0–255, **not** 0–65535 |
| Renormalisation | **NONE.** Clip only — see `docs/decisions.md` D3 |
| Filename | **byte-identical to the input filename**, same extension |
| Subdirectories | mirrored from input to output |
| Count | one output per input; 400 for the released test set |
| Destination | `results/restored_test_outputs/` (SPEC F12 — mandatory repo item) |

### Why clip and nothing else

SPEC F6: *"KLA does not clip or renormalize outputs. Images are scored exactly as saved by
your pipeline."* All range handling must therefore live in our code.

Clipping is correct because GT provably never leaves [0,1] (full scan of 3200 files).
Renormalising is **wrong** and was measured to cost **−4.66 dB PSNR** over 200 validation
pairs, losing on 191/200 images (D3). Do not renormalise.

### Reference writer

```python
import os
import numpy as np

def save_restored(out_dir: str, basename: str, arr: np.ndarray) -> None:
    """basename: the input filename verbatim, e.g. '000123.npy'."""
    assert arr.ndim == 2, arr.shape
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, basename), arr)
```

`np.save` appends `.npy` only when absent, so passing `'000123.npy'` yields `000123.npy`,
never `000123.npy.npy`. Verify this in the output-integrity test (SPEC §11.4 step 3).

### `inference.py` needs no image library

The data is `.npy` end to end. **Remove `cv2` and `tifffile` from `inference.py`.** The SPEC
§11.3 skeleton imports `cv2` because it was written before the format was known; on this
dataset that import is dead weight on a *timed* run (SPEC §11.2 lists import cost as a
seconds-scale lever) and is actively hazardous, since several `cv2` paths silently convert to
8-bit or clip to [0,1] — fatal for inputs that legitimately reach 2.16.

Keep the `EXTS` glob permissive per SPEC §11.1, but the only branch that executes on this
dataset is the `.npy` one.

### Filename collision hazard

`test_NoisyLR` numbering restarts at `000000.npy`, so all 400 test filenames also exist under
`train/` while referring to different images (`array_equal == False`, means 0.218 vs 0.660).
Never key a cache, dict, manifest or results structure on the bare filename — qualify by
split or use the full path. Nothing will crash if you get this wrong; the shapes and dtypes
match.

---

## Verification checklist (SPEC §11.4 step 3)

- [ ] output filename set is identical to the input filename set
- [ ] every output is exactly 2× its input in both dimensions
- [ ] every output is `.npy`, `float32`, `ndim == 2`
- [ ] no NaN, no Inf
- [ ] `min >= 0.0` and `max <= 1.0` for every output
- [ ] outputs reloaded **from disk** and scored — not the in-memory tensor (SPEC §18 pitfall 1)
- [ ] no filename drift: no `_restored` suffix, no extension change
