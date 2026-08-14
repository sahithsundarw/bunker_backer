# SPEC → V-check Requirement Map

Extraction of every requirement in `docs/SPEC.md` (738 lines, parsed in full) that bears on a
check in `docs/VERIFICATION_CONTRACT.md` (V01–V52).

Purpose: `scripts/verify_all.py` has to implement 52 checks. This map is where each check's
pass criterion comes from, so no check is implemented against a guess.

## Counts

| quantity | n |
|---|---|
| V-checks in the contract | **52** (V01–V52, five tiers) |
| V-checks with at least one direct SPEC anchor | **48** |
| V-checks that are contract-only strengthenings beyond SPEC | **4** (V17, V21, V24, V39) |
| Distinct SPEC requirement → V-check edges extracted | **131** |
| Distinct SPEC clauses cited as anchors | **74** |

The 74 anchors break down as: 17 of the 19 `F` facts, all 9 `U` questions, 28 numbered
sections/subsections, 13 of the 18 §18 pitfalls, and 7 `T` task acceptance criteria.

`F13`, `F18`, `F19` (deck filename, demo video, team eligibility) are submission-portal
requirements with no V-check — they are covered by SPEC §15 rather than the verifier.

---

## Tier 0 — submission-blocking (14 checks)

| V | SPEC anchors | Requirement extracted |
|---|---|---|
| V01 | F11; §11.1; §18.6 | Standalone `.py`, not a notebook; `ast.parse` must succeed |
| V02 | F11; §11.1 | Exactly two required args, `--input_dir` / `--output_dir`; exit 0 |
| V03 | §11.1; §11.4.2; §18.4; PD5 | Runs from arbitrary CWD; weights not CWD-relative |
| V04 | §11.4.1; F12 | Fresh clone + fresh venv + `pip install -r requirements.txt`, no manual edits |
| V05 | §11.1; §18.4; PD5 | Weight path derived from `Path(__file__)`, never absolute, never CWD |
| V06 | F12; §18.8 | Weights obtainable: in-repo (LFS resolved) or URL 200 from logged-out + sha256 |
| V07 | §11.1 | One output per input; N in → N out |
| V08 | **U2**; §11.1; §18.9 | Output filename byte-identical to input, including subdir path |
| V09 | **F2**; §5.1.7; §11.4.3 | `out.shape == 2 × in.shape`, zero exceptions |
| V10 | **U1**; §5.1; §18.1 | Container + dtype mirror GT exactly; float32 GT ⇒ float32 out |
| V11 | **F5**; **F6**; §5.1; §18.3 | Output clipped to [0,1]; no NaN/Inf; KLA does not clip for us |
| V12 | **F5**; §18.2; §6.3 | Input **not** clipped — out-of-range values reach the model |
| V13 | F12; §12; §15; §18.7 | README, inference.py, train.py, requirements.txt, weights, non-empty `restored_test_outputs/`; repo public |
| V14 | F12; §13 | `requirements.txt` complete `pip freeze`, every line `==` pinned |

## Tier 1 — robustness (10 checks)

| V | SPEC anchors | Requirement extracted |
|---|---|---|
| V15 | §7.3; §11.4.4; §18.10 | Mixed-resolution folder in one invocation; group by shape, do not stack naively |
| V16 | §11.4.5 | Single-image folder — batching edge case |
| V17 | *(contract-only)* | ≥200 images without OOM at default batch. SPEC §11.2 implies batching but sets no floor |
| V18 | §11.1 | Nested subdirs found; output mirrors structure |
| V19 | §11.1; §11.4.6 | CPU fallback completes, does not crash |
| V20 | §11.1 | One bad file logged and skipped; run continues |
| V21 | *(contract-only)* | Idempotence across two runs. Not stated in SPEC |
| V22 | §11.2 | bf16 vs fp32 agree within tolerance — guards a broken AMP path |
| V23 | **§11.2**; **§18.5**; CLAUDE.md §STYLE | Module-level imports on allowlist; `-X importtime` < 3.0 s. **Promoted to Tier 0** — see `SPEC_ADDENDUM.md` §9 |
| V24 | *(contract-only; §9 adjacent)* | Inference determinism in- and cross-process. SPEC §9 seeds training only |

## Tier 2 — ML correctness (12 checks)

| V | SPEC anchors | Requirement extracted |
|---|---|---|
| V25 | §16 (hr 1.5–2.5); T7 | Overfit 2 pairs > 40 dB. Hard gate — SPEC calls it the pipeline sanity check |
| V26 | §6.2; T5 | Paired crop: LR origin `(i,j)` ⇒ GT origin `(2i,2j)` |
| V27 | §10 baseline 1; T8 | Beat bicubic ×2 on PSNR **and** SSIM, lower LPIPS, mean ± std |
| V28 | §10 baseline 3; §7.2 | Beat the U-Net baseline on ≥2 of 3 metrics, or document the negative result |
| V29 | §6.1; §18.11 | `configs/split_val.txt` committed; no overlap with train; not regenerated at runtime |
| V30 | §10; §18.1 | Score the reloaded **on-disk** artifacts, not in-memory tensors |
| V31 | §10 | PSNR/SSIM/LPIPS settings pinned exactly (`data_range=1.0`; SSIM gaussian σ=1.5, `use_sample_covariance=False`; LPIPS AlexNet, gray→3ch, [-1,1]) |
| V32 | **F1**; §5.1 loader guidance | Single channel in/out; no BGR/RGB conversion. *Moot here — `.npy` path, no `cv2`* |
| V33 | §5.2; T4 | Synthetic re-degradation reproduces the real variance-vs-intensity curve |
| V34 | §9 | `train.py --seed 42` reproducible across two invocations |
| V35 | §9 | Checkpoint carries `model`, `ema`, `config`, `iter`, `metrics`, `git`; `strict=True` load |
| V36 | **F17** | No optimizer, `.backward()`, or param update in `inference.py` |

## Tier 3 — throughput (7 checks)

| V | SPEC anchors | Requirement extracted |
|---|---|---|
| V37 | **F10**; §11.4.7; §14 slide 7 | Report wall-clock incl. process start, count, img/s, batch, precision, GPU, driver, CUDA, torch, timing method |
| V38 | **F10** | Measured window spans read → preprocess → H2D → forward → D2H → postprocess → save; timed externally |
| V39 | *(contract-only; §4 axis 2)* | ≥20 img/s floor. SPEC F9 explicitly prescribes **no** latency threshold, so this number is contract-invented |
| V40 | §11.2 table | channels_last, `inference_mode`, TF32, cuDNN benchmark, AMP on by default; threaded writes; shape-grouped batching |
| V41 | §11.2; §18.15 | `torch.compile` OFF by default, opt-in, crossover documented |
| V42 | §18.14 | Self-ensemble / TTA OFF by default, flag-gated, both timings reported |
| V43 | F16; §18.8 | Param count and checkpoint size recorded; < 100 MB or LFS/hosted |

## Tier 4 — hygiene (9 checks)

| V | SPEC anchors | Requirement extracted |
|---|---|---|
| V44 | §9 | Seed `random`, `numpy`, `torch`, `torch.cuda` from config |
| V45 | §9 | `results/experiments.csv` ≥2 rows with git SHA, config, seed, metrics, wall-clock |
| V46 | §13 | Every fenced shell command in README executes |
| V47 | §12; T12 | `sample_inputs/` runs end-to-end in < 60 s from a clean clone |
| V48 | §10; §14 slide 6 | `results/metrics_summary.md` ≥3 baselines + final, matching a fresh evaluate run |
| V49 | §5.4; §14 slide 6; T14 | ≥4 success triplets, ≥1 labelled failure with written explanation |
| V50 | **F14**; §6.5; §14 slide 9 | External resources: name/link/licence/paper — or explicit "None used." |
| V51 | §12; §15 | No secrets, no `__pycache__`, no dataset blobs; `.gitignore` present |
| V52 | **§2.2 U1–U9**; §12 | `STATE.md`, `dataset_findings.md`, `io_contract.md`, `decisions.md` non-stub; every U1–U9 answered or in `BLOCKERS.md` |

---

## Where the addendum overrides the anchor

Four V-checks take their pass criterion from `docs/SPEC_ADDENDUM.md` rather than from SPEC,
because measurement contradicted the SPEC text. The addendum governs.

| V | SPEC says | Measured — addendum governs |
|---|---|---|
| V09, V15 | F2/§7.3: both 512→256 and 256→128; test inputs mix 128 and 256 | Only 256→128 exists; all 400 test inputs are 128×128. Keep shape-grouped batching and a 256→512 fixture anyway |
| V10 | U1 open: PNG 8-bit? 16-bit? TIFF float32? `.npy`? | `.npy` `float32`, continuous, no bit-depth signature |
| V23 | §11.2 allowlist + "one image IO lib" | **No image IO library at all.** Allowlist is exactly the eight in `CLAUDE.md` §STYLE |
| V32 | §5.1: `cv2.imread` must use `IMREAD_UNCHANGED` | Moot — no `cv2` in the IO path. Check becomes "assert no image library is imported" |

## V52 readiness

V52 requires every U1–U9 answered with evidence or listed in `BLOCKERS.md`. Current state:

| U | Status | Where |
|---|---|---|
| U1 format/dtype | answered | `dataset_findings.md` |
| U2 folder/filename rule | answered | `dataset_findings.md` |
| U3 pair count + size split | answered | `dataset_findings.md` |
| U4 downsample kernel | answered | `decisions.md` D1 |
| U5 degradation order | answered | `decisions.md` D2 |
| U6 noise parameters | answered | `decisions.md` D2 |
| U7 dataset README | answered (none exists) | `dataset_findings.md` |
| U8 pixel alignment | answered | `dataset_findings.md` |
| U9 test folder shipped | answered (yes, 400) | `dataset_findings.md` |

**All nine resolved with numeric evidence. No U-item needs a `BLOCKERS.md` entry.**
V52 still cannot pass: `docs/STATE.md` does not exist yet.
