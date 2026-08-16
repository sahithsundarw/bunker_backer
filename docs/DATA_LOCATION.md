# Data Location

## Dataset root

`src/dataset.py::resolve_data_root()` parses the **first fenced code block below** as its
last-resort fallback path when neither `--data_root` nor `$KLA_DATA_ROOT` is passed — so that
fence must be a bare path, nothing else. Do not put a shell command or anything other than the
literal path in it, or every check that invokes `train.py`/`evaluate.py` without an explicit
`--data_root` breaks (found the hard way post-merge, `docs/decisions.md` D51).

```
C:\kla-data
```

Dataset-dependent training and verifier checks on a different machine instead set
`KLA_DATA_ROOT` explicitly — e.g. the measured Mac dataset lived at
`/Users/shanmukhsai/Downloads` for a teammate's session:

```bash
KLA_DATA_ROOT=/path/to/dataset python scripts/verify_all.py --strict
```

The root must contain `train/GT` and `train/NoisyLR`. The released final-test inputs are under
`NoisyLR` on the measured Mac and `test_NoisyLR` on the historical Windows extraction.

Deliberately **outside** OneDrive and outside the repo. It is never committed, never synced,
never copied into `C:\Users\sahit\OneDrive\Desktop\semi`.

## Folders and verified counts

| Path | Count | Shape | dtype | Role |
|---|---|---|---|---|
| `C:\kla-data\train\GT` | **3200** | `(256, 256)` | `float32` | training targets |
| `C:\kla-data\train\NoisyLR` | **3200** | `(128, 128)` | `float32` | training inputs |
| `C:\kla-data\test_NoisyLR` | **400** | `(128, 128)` | `float32` | **official test inputs** |

All files are `.npy`. Counts verified by directory scan on 2026-08-15; every file in all
three folders has extension `.npy` and zero non-`.npy` files remain.

Pairing: `train/GT/NNNNNN.npy` ↔ `train/NoisyLR/NNNNNN.npy`, identical basename,
`000000`–`003199`. See `docs/dataset_findings.md`.

## Also present

| Path | Contents |
|---|---|
| `C:\kla-data\_archive\train.zip` | 918,994,209 B — original train archive |
| `C:\kla-data\_archive\Test_NoisyLR.zip` | 23,419,125 B — original test archive |
| `C:\kla-data\_archive\train_DS_Store.bin` | 10,244 B — the `.DS_Store` that sat loose in `train\`, moved here |

`_archive` exists so the zips are not re-extracted by accident. Do not extract into
`C:\kla-data\` — it would recreate `__MACOSX` junk and a second folder named `NoisyLR`.

## Renames applied

The shipped test folder was named `NoisyLR`, colliding with `train\NoisyLR`. It was renamed
to `test_NoisyLR`. Two folders named `NoisyLR` is a correctness hazard — a glob or a
mis-joined path would silently train on test inputs.

## ⚠ Filename namespace collision

`test_NoisyLR` numbers restart at `000000.npy`, so **all 400 test filenames also exist in
`train\`** — and they are different images (`train/NoisyLR/000000.npy` mean 0.218 vs
`test_NoisyLR/000000.npy` mean 0.660, `array_equal == False`).

Never key a cache, manifest, index, or results dict by bare filename. Always qualify by split.

---

## HARD RULE

> **Never train, fine-tune, or fit degradation parameters on `test_NoisyLR` (SPEC F17).**
> **Inference on it is required; its outputs populate `results\restored_test_outputs\`.**

This forbids, on `test_NoisyLR`:

- including it in any training or validation split;
- fitting noise, blur, or degradation parameters to it;
- computing normalisation statistics (mean/std/min/max) from it;
- test-time training, self-supervised adaptation, or per-image fine-tuning on it;
- any hyperparameter or checkpoint selection driven by it.

Permitted, and required: a single forward pass per image, writing 400 outputs to
`results\restored_test_outputs\`.

Note: there is **no** `test_GT`. Test ground truth is withheld, so no score can be computed
locally against the test set. Hold out a slice of `train\` for validation instead.
