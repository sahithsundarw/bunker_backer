# REQUIREMENTS_MATRIX.md

Traceability of every requirement in the authoritative KLA PS01 restatement against the actual
code, re-derived independently rather than read off `docs/SPEC_VCHECK_MAP.md`. Columns:
`requirement | satisfied | evidence | V-check / UNCOVERED`. Any disagreement with
`SPEC_VCHECK_MAP.md`, found after this matrix was built, is recorded in the note at the bottom.

Status key: **Y** satisfied · **P** partial · **N** not satisfied / uncovered.

**Updated post-landing** (commits `dd61ef1`, `9e130a4`, `b43484b`): order permutation, tail
coverage, dual-resolution timing, V22, and proxy-OOD wiring have all landed and been
re-verified. Rows below reflect the current, committed state — `[[ IN FLIGHT ]]` markers removed
where resolved; genuinely still-open items (deck team info, selection-metric disclosure, the
no-third-split structural gap) remain marked.

## Main Task

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | Take degraded noisy, low-resolution images as input | Y | `inference.py:315-339` loads `.npy`, no clip, stacks by shape | V02, V12, V57 |
| 2 | Handle speckle noise | Y | `src/degrade.py` speckle term `v*x^2`, fitted v=0.015745 (D12) | V33 |
| 3 | Handle additive Gaussian noise | Y | `src/degrade.py` sigma term, D2/D12 | V33 |
| 4 | Handle downsampling | Y | recovered 4x4 kernel, D1; V09 asserts exact x2 | V09, V33 |
| 5 | Handle degradations even when applied in any order | **Y** | `src/degrade.py:degrade()` now permutes {D,S,G}, all 3!=6 orderings reachable (measured 20,000 trials: DSG 64.71% modal, others 1.1-13.4%), canonical order preserved as modal per D2's measurement. Old fixed-order guard removed | V62 (strengthened, D43) |
| 6 | Restore to expected GT resolution | Y | exact x2 enforced, bicubic-fallback substituted on any shape mismatch | V09, V61, V65 |
| 7 | Generalize to familiar and unfamiliar content | **P** | Familiar (in-distribution): Y, 400-image val split. Unfamiliar (OOD): proxy-OOD set (40 procedural geometric images, `results/eda/proxy_ood/`) scored: PSNR 27.32dB (-1.47 vs in-dist), SSIM 0.965 (+0.182), LPIPS 0.038 (-0.215) -- real, honestly-mixed evidence, not prose. Still **P** not **Y**: this is procedural geometric content, not real semiconductor/SEM imagery, which does not exist anywhere in this project | V63 (implemented, D44) |
| 8 | Run efficiently as a complete NVIDIA GPU inference pipeline | Y | `results/runtime_report.md`: 8.3 img/s end-to-end at N=400, 128->256, RTX 4060, externally timed | V37, V38, V39, V43 |

## Dataset Rules

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | Paired GT/NoisyLR training data | Y | `docs/DATA_LOCATION.md`, 3200 pairs, block-of-4 alignment verified | V26 |
| 2 | Hidden test provides degraded inputs only, GT withheld | Y | `docs/DATA_LOCATION.md`: "no test_GT"; `inference.py` never reads GT | V54 |
| 3 | GT normalized to [0,1] | Y | measured, `docs/SPEC_ADDENDUM.md` section 3 | V29 (split), metrics protocol |
| 4 | NoisyLR may go outside [0,1]; code must handle intentionally | Y | `src/io_utils.py:25-50` no clip on load; measured range [-0.28, 2.16] | V12, V57 |
| 5 | Official dataset dimensions; eval ~256x256 or 512x512 | **Y** | Released data is 128->256 only (no 512 GT exists, `docs/SPEC_ADDENDUM.md:57-64`), so 256->512 is exercised with synthetic real-degraded inputs: shape-forwarded (V61), real timing measured (`results/runtime_report_512.md`, D45: fixed-cost share 34.3% at N=400, does not collapse), real batch + OOM-recovery exercised (V65, D46) | V61, V65 |
| 6 | Test data includes in-distribution and OOD content | **P** | Same as Main Task #7 | V63 |
| 7 | Noise mechanisms same, sampled levels may vary | Y | `src/degrade.py` a/v randomised +/-120% per sample (widened from +/-30% to close F1 tail-coverage gap, D43): synthetic max 2.0869 vs real train max 2.0735 (was 1.7177) | V62 (strengthened) |
| 8 | KLA scores outputs exactly as saved -- clip/normalize inside solution | Y | `inference.py:366` + `src/io_utils.py:73-76`, double clip to [0,1] float32; no renorm (measured -4.66dB rejected, D3) | V11 |
| 9 | Grayscale/single-channel only | Y | `src/model.py` in_ch=out_ch=1; V32 asserts 3-channel input rejected | V32 |

## Model Development

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | CNN/transformer/unrolling/custom architecture allowed | Y | NAFSR, NAFNet-style CNN blocks, justified in README Method summary | -- |
| 2 | Pretrained weights/external datasets allowed if licensed, disclosed | Y | README External resources table: LPIPS (BSD-2), AlexNet backbone (BSD-3), both eval-only; explicit "None used" for training data/pretrained SR weights | V50 |
| 3 | External disclosure: name, link, license, card | Y | same table, per-row | V50 |
| 4 | Synthetic degraded pairs from GT allowed | Y | `src/dataset.py` synth_ratio 0.5, on-the-fly via `src/degrade.py` | -- |
| 5 | Justify preprocessing, augmentation, architecture, losses | Y | README Method summary, `docs/decisions.md` D1/D2/D9/D12/D13/D21 | -- |
| 6 | Frequency-domain methods allowed, not mandatory | Y | FFT loss term used (justified, not mandatory) | -- |
| 7 | No fixed param limit; large models may lose throughput marks | Y (policy, not a check) | NAFSR 388,225 params, deliberately below SPEC 7.1's 1-3M band, reasoned in `configs/nafnet_x2.yaml` header (training wall-clock cost on 4060, D20) | -- |

## Inference Requirements

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | Standalone Python inference script | Y | `inference.py`, single file | V01 |
| 2 | Accepts input-dir / output-dir args | Y | `--input_dir` / `--output_dir`, exactly 2 required | V02 |
| 3 | Loads every degraded image, restores, saves every output | Y | N-in produces N-out | V07 |
| 4 | Preserve official filename and format | Y | byte-identical filenames | V08 |
| 5 | Support NVIDIA GPU execution | Y | CUDA path default, `--device` override | V03, V19 (CPU fallback) |
| 6 | Batch processing preferred when GPU memory permits | Y | grouped-by-shape batching, `--batch_size`; recursive OOM halving + CPU-bicubic single-image floor, now exercised at 256->512 with a genuinely-forced OOM (D46: 31 real OutOfMemoryErrors caught and recovered) | V17, V65 |
| 7 | All weights/config/deps included, no manual edits | Y | `weights/README.md` download instructions, `requirements.txt` pinned, config self-describes in checkpoint | V06, V14, V35, V59 |
| 8 | Evaluators must not edit source/notebook/paths | Y | `--input_dir`/`--output_dir` only; weights resolved via `__file__` | V05 |
| 9 | Runtime includes disk read, pre/post-proc, H2D/D2H, save | Y | `results/runtime_report.md` external `subprocess.run` timing, stage breakdown | V38, V39 |
| 10 | Evaluation script used as-is; must run without edits | Y | fresh-clone tested (though **never as a full `--strict --fresh-clone` suite run**, see Deferred) | V04, V46 (partial -- H-4b open) |

## Validation And Reporting

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | Clean validation split, no leakage | **P** | Block-of-4 alignment guarded (`src/dataset.py:144-241`), empty train/val intersection asserted every construction. Residual leaks: `scripts/fit_degradation.py:206-212` fit on an unfiltered 200-of-3200 sample including ~25 val images; checkpoint selection + headline numbers both on the same 400-split, no third split | V29 (split-integrity, not fit-independence) |
| 2 | Report PSNR, SSIM, LPIPS | Y | `results/metrics_summary.md`, pinned settings (V31) | V27, V28, V31 |
| 3 | Report extra metric used for model selection | **N** [[ IN FLIGHT ]] | `configs/final.yaml:51 save_best_on: psnr` -- PSNR alone, undisclosed as such in README/deck. `trainer`+`docs-scribe` work dispatched (blended criterion + disclosure) | UNCOVERED -> pending V66 |
| 4 | Compare at least one baseline with final method | Y | 4 baselines + paired t-test | V27, V28 |
| 5 | Full-res examples incl. success and failure | Y | `results/qualitative/`, 5 successes (percentile-selected) + 2 deterministically-selected failures with band-limited oracle ceilings | V49 (gate is weak: filename substring only) |
| 6 | Report runtime, batch size, hardware, versions, timing method | **Y** | Complete at both resolutions: 128->256 (`results/runtime_report.md`) and 256->512 (`results/runtime_report_512.md`, D45), each labelled with device and input size | V37, V38, V39, V43 |
| 7 | Track experiments, seeds, hyperparams, checkpoints, final config | Y | `results/experiments.csv`, checkpoint self-describes config (V35) | V44, V45 |

## Phase 1 Deliverables

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | Solution PPT/PDF | **P** | Deck built via `scripts/build_deck.py`, data-driven (reads real committed numbers, no hand-typed figures), 9 pages, self-checked (proxy sentence verbatim, no banned phrases). All numeric slides now populated with real data (metrics table, both-resolution runtime, proxy-OOD). Still needs: real team name/members/college (placeholder, user to fill), V53 not yet implemented | pending V53 |
| 2 | Accessible GitHub repo link | Y | public, confirmed `private: false` via API today | V13, V55 |
| 3 | Standalone inference script | Y | `inference.py` | V01 |
| 4 | Training code reproducing submitted checkpoint | Y | `train.py --config configs/final.yaml --seed 42 --iters 20000`, documented in README | -- |
| 5 | Final model weights/config + download instructions | Y | `weights/README.md`, Release `artifacts-v1`, sha256 published and fetch-verified (V06/V56/V59) | V06, V35, V59 |
| 6 | README with exact setup, commands, I/O contract, assumptions | Y | verified present; 3 stale claims fixed this session (throughput-exists claims, committed-artifact claims) | V46 (partial -- doesn't literally exec fenced commands) |
| 7 | requirements.txt pinned | Y | complete `pip freeze`, `+cu128` index directive, D18 rationale documented | V14 |
| 8 | Results/output samples: metric summary, images, failure analysis | Y | `results/metrics_summary.md`, `results/qualitative/` | V27, V28, V48, V49 |

## GitHub Repository

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | README.md | Y | present, truthed-up this session | V46 |
| 2 | requirements.txt | Y | present | V14 |
| 3 | train.py | Y | present | -- |
| 4 | inference.py | Y | present | V01 |
| 5 | configs/ | Y | `baseline_unet.yaml`, `final.yaml`, `nafnet_x2.yaml`, `split_val.txt` | -- |
| 6 | src/ | Y | 8 modules | -- |
| 7 | weights/ | Y | gitignored `*.pt` + `weights/README.md` with Release download instructions (deliberate, per B9) | V06, V59 |
| 8 | results/ | Y | populated per above; one empty stub (`results/baselines/wavelet_bicubic/`) to delete | -- |
| 9 | solution_presentation.pptx (repo layout suggestion) | **N** | Deck being built as PDF per binding format (F13); repo layout's `.pptx` suggestion is superseded by the portal's binding PDF requirement -- not a gap, a format choice already resolved by SPEC section 14 | -- |

## Evaluation

| # | Requirement | Sat. | Evidence | Check |
|---|---|---|---|---|
| 1 | Restoration quality: fixed PSNR+SSIM+LPIPS blend, hidden GT, in-dist + OOD | **P** | Metrics pinned and reported for in-distribution; OOD now has a real, honestly-mixed proxy measurement (procedural content, not real semiconductor imagery) | V27, V28, V31, V63 |
| 2 | End-to-end throughput, common H100, incl. I/O and pre/post-proc | **Y** | Real externally-timed numbers exist at both 128->256 and 256->512 on RTX 4060 (never H100 -- none fabricated). No H100 number exists or is claimed anywhere | V37, V38, V39, V43 |
| 3 | Training & compute hygiene: reproducibility, clean experiments, env spec, code quality, efficient pipeline, ML practice | Y | experiments.csv ledger, seeded, pinned deps, `docs/decisions.md` append-only log, verifier contract | V44, V45, V14, and the whole Tier 4 |
| 4 | Exact metric weights undisclosed by KLA | -- | N/A, informational | -- |
| 5 | No target score/latency threshold prescribed | -- | N/A, informational | -- |

---

## Summary of remaining P/N rows requiring action

1. **OOD generalization reporting is genuine but partial** (Main Task #7, Dataset Rules #6,
   Evaluation #1) -- V63 implemented and PASS. Ceiling on "Y" is structural, not a to-do: no
   real semiconductor/SEM imagery exists anywhere in this project, so this can only ever be a
   procedural proxy, honestly labelled as such.
2. **Selection-metric disclosure** (Validation #3) -- still open. `configs/final.yaml` selects
   on PSNR alone while KLA scores a PSNR+SSIM+LPIPS blend; the shipped model loses PSNR to its
   own U-Net baseline. Blended criterion + README/deck disclosure + V66 not yet started.
3. **Deck** (Phase 1 Deliverables #1) -- built and data-driven, all real numbers populated;
   needs team info (user-provided placeholder pending) and V53 not yet implemented.
4. **Validation-split independence** (Validation #1) -- structural, deferred; recorded in
   `docs/BLOCKERS.md` B10 and the deck's limitations line rather than silently fixed.
5. **`results/baselines/wavelet_bicubic/`** -- empty stub, deleted.
6. **B11 (new): V24 cross-process determinism is genuinely flaky (~24-50%)** under
   `cudnn.benchmark=True`, pre-existing (confirmed present before this iteration's changes too).
   Blocks Definition of Done #2 (two consecutive clean `--strict` runs) until resolved or
   explicitly accepted. Not a requirements-matrix row (V24 is a hygiene/robustness check, not a
   KLA-stated requirement) but recorded here since it gates the verification end-state.

Resolved this iteration (were P/N, now Y): order permutation (#5 Main Task), 256/512 dual-res
timing + batch/OOM exercise (Dataset Rules #5, Inference #6, Validation #6, Evaluation #2),
tail coverage (Dataset Rules #7).

## Reconciliation note

Not yet diffed against `docs/SPEC_VCHECK_MAP.md`. Remaining open items above (selection-metric
disclosure, deck team info, V53/V66) are still moving; the diff will be done once those land.
