# Submission checklist

> **⚠ Superseded, 2026-08-16.** This snapshot describes the teammate-line checkpoint
> (`r2_nb8_psnrloss`, 28.0394 dB) as it stood before this repo's two lines of work were
> reconciled. The checkpoint actually shipped is a different one that won a head-to-head
> re-score on all three metrics — see `docs/decisions.md` D49 and `weights/README.md`. Kept
> here verbatim as the audit trail for that earlier state, not as a live status.

Snapshot taken 2026-08-15 on branch `codex/final-submission-28db`, commit at time of writing
`1d3c23d` (this checklist and `results/qualitative/` land in the commit(s) immediately after).
Regenerate the live numbers with `python scripts/verify_all.py --strict`;
`results/verification_report.json` is the authoritative machine-readable source, this file is
the human-readable narrative on top of it.

## Status at a glance

| Item | Status |
|---|---|
| Checkpoint present and SHA verified | ✅ Done |
| Validation metrics recorded | ✅ Done — see caveat below |
| Final outputs generated | ✅ Done |
| Manifest present | ✅ Done |
| Archive SHA recorded | ✅ Done |
| `release_url` | ⏳ **Pending — human action required, see Task 1 below** |
| No final-test metrics claimed | ✅ Verified — none found |
| Fresh-clone Linux/CUDA V04/V46 | ✅ Independently verified on real Linux (not reproducible on this Mac dev box, by design) |
| Remaining verifier failures explained | ✅ All 11 traced to a known, documented cause |

---

## 1. Checkpoint present and SHA verified

- `weights/best.pt` is tracked in the repository (V06 PASS, V59 PASS).
- SHA256: `37e8571047218a0344c43bcd2246dc559184a75fe301995fea24463dfd341fa7`.
- Verified live: `python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('weights/best.pt').read_bytes()).hexdigest())"` reproduces the pinned digest (checked during this session).
- Checkpoint self-describes and loads strictly (`V35 PASS`: `build_model(ckpt['config'])` with `strict=True`).
- Full provenance: `weights/README.md`.

## 2. Validation metrics recorded

- **28.0394 ± 4.1881 dB PSNR, 0.74804 ± 0.15275 SSIM, 0.29571 ± 0.16672 LPIPS**, n=400,
  `configs/split_val.txt`, scored on the reloaded on-disk `.npy` predictions (V30 round-trip),
  pinned metric settings (V31 PASS).
- **Independently re-verified in this session**: re-ran `scripts/evaluate.py` against
  `results/residual_experiments/r2_nb8_psnrloss/preds/final/` and reproduced
  **28.0394 ± 4.1881 / 0.74804 ± 0.15275** exactly (PSNR/SSIM to 4 decimal places; LPIPS not
  re-run in that pass — no-LPIPS mode was used for speed, PSNR/SSIM alone already reproduce
  the number the checkpoint reports).
- Also confirmed on disk: `weights/README.md`, `results/restored_test_outputs/README.md`,
  and `README.md` all cite the same three numbers consistently.
- ⚠ **Known caveat, not fixed by this pass**: `results/metrics_summary.md` (the
  machine-generated file `scripts/evaluate.py` normally writes) is **stale** — its "Final
  model" row still shows **26.3277 dB**, the earlier LS-5-only checkpoint's score, not the
  current 28.0394 dB checkpoint. Confirmed by diffing the two prediction directories:
  `results/baselines/final/` (what `metrics_summary.md` was scored against) and
  `results/residual_experiments/r2_nb8_psnrloss/preds/final/` (what the reported 28.0394 dB
  number actually comes from) contain **different, non-identical `.npy` files** for the same
  filenames. This file is owned by `loss-metrics` per `CLAUDE.md`'s file ownership map, so it
  was not silently overwritten in this session — flagging it here instead so it gets picked up
  deliberately. Regenerating it is a one-line command:
  ```
  python scripts/evaluate.py --data_root <data_root> \
    --preds final=results/residual_experiments/r2_nb8_psnrloss/preds/final \
    --device cpu
  ```
  (swap in bicubic/median/nlm rows too if those prediction directories still exist locally).

## 3. Final outputs generated

- 400/400 `.npy` files produced from the released `test_NoisyLR` inputs using
  `inference.py --require_weights` (guarantees no silent bicubic fallback).
- Verified from disk: shape `(256, 256)`, `float32`, all finite, values in `[0, 1]`, filenames
  byte-identical to inputs. Recorded per-file in `results/restored_test_outputs/manifest.csv`.
- **No ground truth exists for this set** — no PSNR/SSIM/LPIPS is computed or claimed for it,
  anywhere in the repo (see item 6 below).
- Local runtime for this run: 400 images, 56.73 s, 7.1 img/s, `results/runtime_report.md`.

## 4. Manifest present

- `results/restored_test_outputs/manifest.json` — present, machine-generated, parses as valid
  JSON, `n_files: 400`. `results/restored_test_outputs/manifest.csv` — 400 rows, one per
  output file, each with sha256/shape/dtype/min/max/finite/`matching_input_exists`.
- V13 (mandatory-items check): **PASS**. V56 (manifest machine-checkability): **FAIL**, but
  only on the one still-empty field — see item 5.

## 5. Archive SHA recorded

- Archive: `semicon_final_outputs_28db.zip`, 400 files, 91,051,052 bytes.
- SHA256: `a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98`.
- Recorded in `results/restored_test_outputs/manifest.json` (`archive_sha256`) and
  `results/restored_test_outputs/README.md`.

## 6. `release_url` — pending / what exactly needs to be filled

**Checked in this session**: `results/restored_test_outputs/manifest.json` still has
`"release_url": ""`. This is the **one remaining manual step** before the repo is fully
submission-complete, and it cannot be done from inside this session (no ability to upload a
GitHub Release asset). Reporting the exact requirement rather than fabricating a URL:

1. Upload `/tmp/semicon_final_outputs_28db.zip` (91,051,052 bytes) as a **GitHub Release
   asset** on `https://github.com/sahithsundarw/semicon-kla-image-restoration` (Releases →
   Draft a new release, or attach to an existing one).
2. Verify the uploaded asset reproduces `sha256
   a33a9a5a129bb006eccb5cf3367abad3456c63d96c1e7bb26e76800e7e375f98` — download it from a
   **logged-out** browser session or `curl -L` and hash it, so the check matches what an
   anonymous reviewer will actually fetch.
3. Set `results/restored_test_outputs/manifest.json`'s `"release_url"` field to the **exact
   public asset download URL** GitHub gives the uploaded file (the direct asset link, of the
   form `https://github.com/<owner>/<repo>/releases/download/<tag>/semicon_final_outputs_28db.zip`
   — copy it from the Release page after upload, do not hand-construct it).
4. Also update `results/restored_test_outputs/README.md`'s "Release asset URL" row (currently
   `*pending — remaining manual step, see below*`) and its two checklist checkboxes under
   "Manual step remaining" / "Checklist before submission".
5. Re-run `python scripts/verify_all.py --strict --only V56` — it should flip to PASS once the
   four `manifest.json` fields it checks (`release_url`, `archive_sha256`, `n_files`,
   `producing_git_sha`) are all present and `archive_sha256` is a 64-hex digest.

Until step 3 is done, `manifest.json.status` should remain
`"archive_ready_release_upload_pending"` — do not change it to a "done" state prematurely.

## 7. No final-test metrics claimed

Checked in this session by reading every metrics-adjacent claim in the repo:

- `manifest.json`: `"metrics": null`, with an explicit `metrics_note` stating no PSNR/SSIM/
  LPIPS is computed for the final-test set.
- `results/restored_test_outputs/README.md`: explicitly states "No PSNR/SSIM/LPIPS is computed
  on these 400 outputs" and that the 28.0394/0.74804/0.29571 figures describe the checkpoint's
  *validation*-split performance, not this folder's outputs.
- `README.md`: "There is no `test_GT` ... no score can be computed locally against the
  official test set."
- `results/qualitative/README.md` (this session's addition): explicitly labels every
  final-test panel "NO GROUND TRUTH" in the image itself (red title text), and the D5
  out-of-split illustrative measurement is explicitly called out as not part of the reported
  400-pair mean.
- No occurrence found of a final-test PSNR/SSIM/LPIPS number anywhere in `docs/`, `README.md`,
  `weights/README.md`, or `results/`.

**No violation found.**

## 8. Fresh-clone Linux/CUDA V04/V46

- On this Mac dev box, V04/V46 **FAIL by design**: `requirements.txt` pins
  `torch==2.11.0+cu128` (CUDA-only, no macOS wheel exists at all), so the nested fresh-venv
  install cannot succeed here — this is the intended "loud failure on the wrong platform"
  behaviour, not a defect.
- **Independently verified PASS on real Linux** (`python:3.12-slim` Docker container,
  documented in `docs/STATE.md`): both `--fresh-clone --only V04` and
  `--fresh-clone --only V46` passed cleanly in isolation, installing the exact pinned
  `torch 2.11.0+cu128` / `torchvision 0.26.0+cu128` and running `inference.py` end to end
  inside the fresh clone's own venv. A double-clone-in-one-container run hit one transient
  network-contention FAIL on V04, isolated to double-downloading ~1.6 GB of torch back to
  back — re-ran in isolation and passed. Not a dependency or pin defect.
- The local `results/verification_report.json` continues to honestly report V04/V46 FAIL
  because that reflects what actually ran on this machine — the Linux verification is recorded
  separately in `docs/STATE.md`, not substituted into the local report.

## 9. Remaining verifier failures explained

The pre-hardening full `--strict` snapshot was **46 PASS / 11 FAIL** (up from the previously
recorded 45/12 — see the V49 note below). The 2026-08-16 final-hardening pass subsequently
resolved V38 with a working external harness and a measured 400-image run; the table preserves
the other snapshot failures that still require external state or additional model work.

| Check | Cause | Disposition |
|---|---|---|
| V04 | Requires `--fresh-clone`; CUDA-only pin has no macOS wheel | Independently PASSED on Linux — item 8 |
| V14 | `requirements.txt` import-coverage scan flags stdlib/internal names (`--builtin--`, `-abcoll`, `-pytest`, …) as "not covered" | Pre-existing backlog gap in the check's stdlib allowlist, not a missing dependency |
| V25 | Overfit-gate smoke run needs `KLA_DATA_ROOT` set to a real dataset root | Historically the dataset lived off-machine; **now available locally** at `/Users/shanmukhsai/Downloads` — re-run with `KLA_DATA_ROOT=/Users/shanmukhsai/Downloads` to get a live result (checked in this session, see below) |
| V27 | `unet_baseline` comparison row not implemented/generated | Backlog item — the U-Net baseline was never trained; `docs/STATE.md` records this as open work, not a defect in the shipped model |
| V28 | Same as V27 — needs the `unet_baseline` row | Same |
| V29 | Same `KLA_DATA_ROOT` requirement as V25 | Same — checked live in this session |
| V32 | Scans every `*.py` under the repo including `.venv-mac/` (only literal `.venv` is excluded, not `.venv-mac`) and finds a plain `cv2.imread(` inside a **third-party package installed in the local venv**, not in this repo's own source | Documented environment noise, `docs/decisions.md` D30 — does not reproduce on a fresh clone, which is what actually gets scored |
| V34 | Same `KLA_DATA_ROOT` requirement as V25/V29 | Same — checked live in this session |
| V46 | Same fresh-clone requirement as V04 | Independently PASSED on Linux — item 8 |
| V56 | `manifest.json.release_url` still empty | The one pending human step — item 6 |

**V49 update (this session):** `results/qualitative/` now has 12 files including a
`failurecase_D5_*.png`, so V49 ("≥4 successes + ≥1 labelled failure") now **PASSES**
(`14 qualitative artifacts` reported — the count includes `results/eda/`'s existing pair-grid
figures alongside the new panels). This was previously `not_impl` in the documented 45/12
baseline; it is now a genuine 46th PASS, not a re-count of the same failures.

**V25/V29/V34 live check (this session):** re-ran with `KLA_DATA_ROOT` pointed at the local
copy of the dataset (`/Users/shanmukhsai/Downloads`, which has `train/GT`, `train/NoisyLR`,
and `NoisyLR` present — see item 3). This exercises real training/overfit code and can take
several minutes; results reported separately once the run completed (see task notes / rerun
`KLA_DATA_ROOT=/Users/shanmukhsai/Downloads python scripts/verify_all.py --strict --only
V25,V29,V34` to reproduce). If they pass with the env var set, that only reduces the *local*
count of unexplained failures — it does not change what KLA's own environment will see, since
KLA supplies its own dataset root and env var, not this one.

---

## Not claimed anywhere in this document

- No final-test PSNR/SSIM/LPIPS (see item 7).
- No H100/CUDA runtime number (see `results/runtime_report.md` — explicitly local Mac CPU).
- No fabricated `release_url` (see item 6 — left as an explicit pending action).
