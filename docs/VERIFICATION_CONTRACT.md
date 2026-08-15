# VERIFICATION CONTRACT — IMMUTABLE

> This file defines "correct" for this project. **It may only be made stricter, and only by the human.**
> The agent may not delete, loosen, skip, or tolerance-widen any check. See CLAUDE.md Prime Directive 1.
>
> Implementation lives in `scripts/verify_all.py`, which must:
> - run every check below,
> - write `results/verification_report.json` of the form
>   `{"iteration": N, "commit": "sha", "checks": [{"id":"V01","status":"PASS|FAIL|SKIP","detail":"...","evidence":{...}}], "summary": {...}}`,
> - print a one-line-per-check table to stdout,
> - exit `0` only if zero FAIL and (under `--strict`) zero un-whitelisted SKIP.
>
> Each check is independent and must not assume another passed.

---

## TIER 0 — SUBMISSION-BLOCKING (a failure here means the submission cannot be scored)

| ID | Check | Pass criterion |
|---|---|---|
| **V01** | `inference.py` exists at repo root, is a `.py` (not a notebook), and is executable as `python inference.py`. | File exists; `ast.parse` succeeds; no `.ipynb` is presented as the evaluation script. |
| **V02** | Argument contract. | `python inference.py --input_dir X --output_dir Y` runs to completion with **exit code 0** and **no other required arguments**. Verified by `argparse` introspection: exactly two `required=True` args, named `--input_dir` and `--output_dir`. |
| **V03** | Runs from an arbitrary CWD. | Same invocation from `cd /` and from `cd /tmp` succeeds, using absolute paths to the script. Catches CWD-relative weight loading. |
| **V04** | Runs from a **fresh clone into a fresh venv** with only `pip install -r requirements.txt`. | Full end-to-end pass. No manual edits, no extra installs, no env vars required. |
| **V05** | Weights are auto-loaded. | The script locates its checkpoint with **zero** user action. Static check: no absolute path literal and no `os.getcwd()`/bare-relative path in the weight resolution; must derive from `Path(__file__)`. |
| **V06** | Weights are obtainable. | Either the checkpoint file is present in the clone (Git LFS resolved — assert file size > 1 KB and it is not an LFS pointer stub), **or** `weights/README.md` contains a URL that returns HTTP 200 from a logged-out session, plus a `sha256` that matches after download. |
| **V07** | Output completeness. | For an input dir of N images, exactly N files are written to the output dir. |
| **V08** | Filename fidelity. | The set of output filenames (including extension, including relative subdirectory path) is **byte-identical** to the set of input filenames. No suffixes, no case changes, no extension changes. |
| **V09** | Scale factor. | For every pair, `out.shape == (2*in.shape[0], 2*in.shape[1])`. Zero exceptions. |
| **V10** | Format & dtype fidelity. | Every output file's container format and numpy dtype match the rule recorded in `docs/io_contract.md` (which itself must have been derived from the real GT files). If GT is float32, outputs must be float32 — an 8-bit write is a FAIL. |
| **V11** | Value range. | Every output satisfies `min >= 0.0` and `max <= 1.0` after de-scaling to the [0,1] convention. No NaN, no Inf. (KLA does not clip — SPEC F6.) |
| **V12** | Input is NOT clipped. | Instrumented check: feed an input containing values `< 0` and `> 1`; assert the tensor entering the model still contains them. (SPEC F5 — out-of-range values are intentional and informative.) |
| **V13** | Public repo & required contents. | `README.md`, `inference.py`, `train.py`, `requirements.txt`, weights, `results/restored_test_outputs/` (non-empty) all present. `git remote -v` resolves and an unauthenticated `git clone` of the remote succeeds. |
| **V14** | `requirements.txt` is complete and pinned. | Every top-level import across the repo resolves to a listed distribution; every line has an `==` pin; the fresh-venv install (V04) succeeds. |
| **V23** | No module-level heavy imports in `inference.py`. | Static AST scan of module-level imports against the allowlist in CLAUDE.md §STYLE. `python -X importtime inference.py --help` total import time < 3.0 s. |

> **V23 was moved from Tier 1 to Tier 0** on 2026-08-15 by human authorisation. Rationale: the
> test set is 400 files totalling 25.05 MB and the forward pass is sub-millisecond per image,
> so fixed startup cost is ~85–95% of the scored wall-clock. A stray module-level import is a
> submission-blocking throughput failure here, not a hygiene nit. See `docs/decisions.md` D6
> and `docs/SPEC_ADDENDUM.md` §9. The ID is deliberately **not** renumbered — check IDs are
> stable identifiers referenced by `results/verification_report.json` and `docs/STATE.md`.

## TIER 1 — ROBUSTNESS (silent scoring killers)

| ID | Check | Pass criterion |
|---|---|---|
| **V15** | Mixed-resolution folder. | A folder containing both 128×128 and 256×256 inputs processes correctly in one invocation. Catches naive batch-stacking. |
| **V16** | Single-image folder. | A folder with exactly one image succeeds. Catches batching edge cases. |
| **V17** | Large-batch folder. | ≥ 200 images (may be duplicated) succeed without OOM at the default batch size. |
| **V18** | Nested subdirectories. | Inputs in subdirs are found and outputs mirror the structure. |
| **V19** | CPU fallback. | With `CUDA_VISIBLE_DEVICES=""` the script completes successfully (slower is fine, crashing is not). |
| **V20** | Corrupt-file resilience. | A truncated/unreadable file in the input dir does not abort the run; it is logged and the remaining N-1 images are still produced. |
| **V21** | Idempotence. | Running twice into the same output dir produces byte-identical outputs (no accumulation, no random state leaking into inference). |
| **V22** | Precision equivalence. | Outputs under `--precision bf16` and `--precision fp32` agree to within a documented tolerance (default: mean abs diff < 1e-3, max abs diff < 1e-2). Guards against a silently broken AMP path. |
| **V24** | Determinism of inference. | Two runs in the same process and two runs in separate processes give identical outputs. No dropout/noise active in eval mode. |

> **V23 has moved to Tier 0.** See the Tier 0 table above. Not deleted, not loosened.

## TIER 2 — CORRECTNESS OF THE ML (does the thing actually work)

| ID | Check | Pass criterion |
|---|---|---|
| **V25** | Pipeline sanity: overfit. | Training the model on 2 fixed pairs for a bounded number of steps reaches **PSNR > 40 dB** on those pairs. A failure here means alignment, normalization or loss is broken — nothing downstream is trustworthy. |
| **V26** | Paired-crop alignment. | For random crops, the GT crop is exactly the 2× region of the LR crop. Asserted by cropping a synthetic image with a known marker and checking marker position. |
| **V27** | Beats the trivial baseline. | Final model PSNR **and** SSIM on the held-out validation split exceed bicubic-×2-upsample of the raw input, and LPIPS is lower. By a margin, not noise: report mean ± std over the split. |
| **V28** | Beats a learned baseline. | Final model outperforms the U-Net baseline trained under the same budget on at least 2 of the 3 metrics; if not, this is recorded as an honest negative result in `docs/decisions.md` (which converts FAIL→PASS only with the negative result documented and the better model shipped). |
| **V29** | No validation leakage. | `configs/split_val.txt` is a committed explicit file list; no filename in it appears in the training file list; the split is not regenerated randomly at runtime. Asserted by intersecting the two lists. |
| **V30** | Metrics are scored on disk artifacts. | `scripts/evaluate.py` reloads the **saved output files** and scores those, not in-memory tensors. Catches dtype/quantization loss. Static + runtime check. |
| **V31** | Metric implementations pinned. | PSNR/SSIM/LPIPS settings match SPEC §10 exactly (`data_range=1.0`; SSIM `gaussian_weights=True, sigma=1.5, use_sample_covariance=False`; LPIPS AlexNet, grayscale→3ch, [-1,1]). Asserted by inspecting the call sites. |
| **V32** | Grayscale handling. | Model is single-channel in and out; no accidental BGR/RGB conversion anywhere in the IO path (`cv2.imread` must use `IMREAD_UNCHANGED`). |
| **V33** | Degradation simulator fidelity. | Synthetic re-degradation of GT with fitted parameters produces a residual whose variance-vs-intensity curve matches the real NoisyLR curve within a documented tolerance. Evidence figure saved. |
| **V34** | Reproducibility of training. | `train.py --config configs/final.yaml --seed 42` for a short smoke run produces identical loss values across two invocations. |
| **V35** | Checkpoint self-describes. | `best.pt` contains `model`, `ema`, `config`, `iter`, `metrics`, `git` keys, and `build_model(ckpt['config'])` accepts the stored state dict with `strict=True`. |
| **V36** | No test-time training. | Static scan: `inference.py` contains no optimizer, no `.backward()`, no `requires_grad_(True)`, no parameter update. (SPEC F17.) |

## TIER 3 — THROUGHPUT (scored axis)

| ID | Check | Pass criterion |
|---|---|---|
| **V37** | End-to-end timing is measured and reported. | `scripts/benchmark_runtime.py` reports total wall-clock **including process start**, image count, img/s, batch size, precision, GPU name, driver, CUDA and torch versions, and the timing method. Written to `results/runtime_report.md`. |
| **V38** | Timing covers the full pipeline. | The measured window includes disk read, preprocess, H2D, forward, D2H, postprocess and save — asserted by timing the subprocess externally (`time python inference.py ...`), not by an internal timer around the forward pass only. |
| **V39** | End-to-end wall-clock, measured and reported. | Total end-to-end wall-clock for the full 400-image test set, measured externally around the process (not an internal timer), reported in `results/runtime_report.md` with a startup-vs-compute breakdown. PASS = measured and reported. No threshold — F9 prescribes none. |
| **V40** | Optimizations are on by default where free. | `channels_last`, `inference_mode`, TF32, cuDNN benchmark, AMP enabled by default; threaded writes present; shape-grouped batching present. Asserted by static scan. |
| **V41** | `torch.compile` is OFF by default. | Compilation cost sits inside KLA's measured window. Must be opt-in via `--compile`, with the measured crossover documented. |
| **V42** | Self-ensemble / TTA is OFF by default. | If implemented, it must be flag-gated, with both timing numbers reported. |
| **V43** | Model size sanity. | Parameter count and checkpoint size recorded in `results/runtime_report.md`; checkpoint < 100 MB or Git LFS / external hosting verified by V06. |

## TIER 4 — HYGIENE (scored axis: "training & compute hygiene")

| ID | Check | Pass criterion |
|---|---|---|
| **V44** | Seeds set everywhere. | `random`, `numpy`, `torch`, `torch.cuda` seeded from config in `train.py`. |
| **V45** | Experiment ledger. | `results/experiments.csv` has ≥ 2 rows with git SHA, config, seed, metrics, wall-clock. |
| **V46** | README runnability. | Every fenced shell command in `README.md` is extracted and executed (or dry-run validated); all succeed. **This is the check that proves a reviewer can clone and run.** |
| **V47** | Sample inputs work. | `python inference.py --input_dir sample_inputs --output_dir /tmp/o` completes in < 60 s from a clean clone. |
| **V48** | Results table exists and matches. | `results/metrics_summary.md` contains ≥ 3 baselines + final, and the numbers match a fresh run of `scripts/evaluate.py` within tolerance. |
| **V49** | Qualitative evidence. | `results/qualitative/` contains ≥ 4 success triplets and ≥ 1 labelled failure case at full resolution, plus a written explanation of the failure. |
| **V50** | External resources disclosed. | `README.md` has an external-resources section listing name/link/licence/paper for every external dataset or pretrained weight — or an explicit "None used." |
| **V51** | No secrets, no junk. | No API keys, no `.env`, no `__pycache__`, no `.ipynb_checkpoints`, no dataset blobs committed. `.gitignore` present. |
| **V52** | Docs current. | `docs/STATE.md`, `docs/dataset_findings.md`, `docs/io_contract.md`, `docs/decisions.md` all exist and are non-stub. Every UNVERIFIED item U1–U9 from SPEC §2.2 is either answered with evidence or listed in `docs/BLOCKERS.md`. |

---

## ADDED CHECKS — iteration 1, from `requirements-auditor` findings

> These are **additions**, which this contract permits. Nothing existing was deleted,
> loosened or renumbered. Each closes a requirement that **no existing check could ever have
> turned red** — the auditor found eleven such gaps and these are the four that can cost the
> submission outright. Rationale and evidence in `docs/decisions.md` D27.

| ID | Tier | Check | Pass criterion |
|---|---|---|---|
| **V54** | 2 | **F17 on the TRAINING path.** V36 scans only `inference.py` — the side that *cannot* fit on test data. The training side was covered by nothing. | AST scan of `train.py`, `src/dataset.py`, `src/degrade.py`, `src/losses.py`, `src/metrics.py`, `src/utils.py`, `scripts/evaluate.py`, `scripts/make_baselines.py`. FAIL on any executable string literal naming `test_NoisyLR`/`test_GT` that is **path-shaped** (contains no whitespace) or is passed to a filesystem call. Docstrings and prose are exempt — they cannot read a file. Verified with a negative control: injecting `np.load("…/test_NoisyLR/000000.npy")` flips it red. |
| **V55** | 0 | **The repo is genuinely PUBLIC.** V13 accepted any non-empty `git remote -v`, which a private repo produces identically. SPEC §18 pitfall 7 calls this a common fatal failure. | `git clone --depth 1` of `origin` succeeds with `GITHUB_TOKEN`, `GH_TOKEN`, `GIT_ASKPASS` cleared, `GIT_TERMINAL_PROMPT=0` and `credential.helper=` empty. A pass cannot come from cached credentials. |
| **V56** | 0 | **`results/restored_test_outputs/` holds ACTUAL model outputs.** V13 accepts any non-`.gitkeep` file, so a README alone satisfies it. F12 and SPEC §15 say "actual model outputs, not placeholders". | Either ≥400 committed `.npy` (sample 16: `float32`, `ndim==2`, even dims, finite, within [0,1]), **or** a `manifest.json` carrying `release_url`, a 64-hex `archive_sha256`, `n_files == 400`, `producing_git_sha`, `checkpoint_sha256` and `command` — where `command` **must contain `--require_weights`**, so outputs produced by the no-checkpoint bicubic fallback can never be shipped as model results. |
| **V59** | 0 | **The checkpoint is genuinely obtainable.** `.gitignore` blanket-bans `*.pt` and V51 lists `.pt` as a forbidden blob, so `weights/best.pt` can never be committed in this repo — the hosted-URL branch of V06 is the only valid route. | PASS if `weights/best.pt` is tracked, **or** `weights/README.md` carries a URL and a 64-hex sha256. FAIL specifically when `best.pt` exists on disk but is neither tracked nor published — the silent case where it works for the author and is missing for everyone who clones. |
| **V60** | 1 | **`--output_dir` equal to or nested inside `--input_dir` must be refused, not silently run.** The former overwrites the degraded inputs with restored outputs in place; the latter makes a second invocation re-ingest the previous run's own output (adversarial review H2, H3). | Copy the `single` fixture into a scratch directory and invoke `inference.py --device cpu` with `--output_dir` equal to that directory, then again with `--output_dir` a subdirectory of it. Both must exit non-zero and must not have written or mutated anything. |
| **V64** | 1 | **A partial write failure must exit non-zero, not report success.** V07 requires exactly one output per input; a short output set silently reported as success is the worst outcome on KLA's machine, since nothing would flag it (adversarial review H4). | Using the `mixed` fixture (≥ 2 valid inputs), pre-occupy exactly one output path with a directory so its write fails, then invoke `inference.py --device cpu`. Must exit non-zero, and the *other* input(s) must still have written successfully — proving this is a genuine partial-failure test, not merely a total-failure one. |
| **V57** | 0 | **The tensor entering the model is unclipped, on the real path.** V12 tests `src.io_utils.load_array` directly; the contract's own wording is "the tensor entering the model", and a `clamp_` anywhere in `inference.py`'s stack/H2D/channels_last/autocast pipeline leaves V12 green. | Import `inference.py`'s own `load_net()` and `infer_chunk()` (not a reimplementation), load the real trained checkpoint, attach a forward pre-hook to the model, and drive an extreme-value probe (`min≈-0.28`, `max≈2.16`) through `infer_chunk` exactly as a real run would. FAIL if the hook observes the model receiving anything narrower than `[-0.27, 2.15]`. Forced to `--device cpu` so it never contends with a GPU benchmark. |
| **V58** | 4 | **SPEC §2.3's official links are independently re-verified, not just cited once at bootstrap.** The licence links in `docs/decisions.md` were re-fetched and dated; SPEC §2.3's hackathon resource links never were. | The URL list is read dynamically from `docs/SPEC.md`'s "### 2.3 Official links" table (not a copy that could drift). `docs/link_check.md` must record every one at HTTP 200 with a UTC timestamp; the oldest recorded timestamp must be ≤ 72 hours old. Does not re-fetch on every verifier run — freshness is enforced by the file expiring, not by making the suite depend on third-party uptime. |

> **V10 was also strengthened in place** (no new ID): it now asserts `ndim == 2` on every
> output, not just `.npy` + `float32`. A `(2H,2W,1)` or `(1,2H,2W)` write previously passed
> V10, and passed V09 too, because V09 reads `so[0]`/`so[1]` — the wrong axes for a
> channels-last array.

| **V61** | 2 | **F2 size-agnosticism, forwarded on every architecture, not read from dead code.** SPEC_ADDENDUM's 256→512 fixture — "the only guard against silently baking in 128→256" — lived in `src/model.py::_selftest()`, which nothing ever invoked, and `UNetSR`'s pad/crop-back was forwarded by zero checks. | For every architecture in `{NAFSR, UNetSR}` and every `(h, w)` in `{(128,128), (256,256), (61,97), (1,1), (130,66)}`, `build_model` then a forward pass must yield exactly `(1, 1, 2h, 2w)`, all finite. FAIL if fewer than all 10 combinations actually ran. |
| **V62** | 2 | **F4 degradation-order randomisation is measured, not assumed.** V33 compares only the aggregate variance curve, which the pre-downsample order hedge barely moves — `GAUSS_PRE_DOWN_PROB` and its entire code branch could be deleted with every existing check staying green. | Over 2000 draws of `sample_noise_params`: `a` and `v` must each span ≥ 90% of their configured ±30% range without escaping it; `sigma`'s sampled minimum must fall in the near-zero 5% of its configured range and its maximum must exceed 0.015. Separately, over 800 calls to `degrade()` with a spy wrapped around `src.degrade.downsample`, the pre-downsample branch must be observed taken between 8% and 22% of calls. |

## WHITELIST FOR SKIPS

A check may report SKIP **only** if listed here with a reason. The agent may append to this list only for genuine environmental impossibility (e.g. no GPU present), never for difficulty.

| Check | Permitted skip reason |
|---|---|
| V40 (partial) | No CUDA device available in the dev environment. Must still pass static-scan portions. |
| V06 (remote branch) | No git remote configured yet — permitted only before first push. |

> **V39 may no longer SKIP** (human-authorised strengthening, 2026-08-15). The revised V39 has
> no threshold, so end-to-end wall-clock is measurable on any device. It must be measured on
> whatever device is present and **labelled with that device name** in
> `results/runtime_report.md`. Absence of CUDA is not an excuse to skip it. See
> `docs/decisions.md` D10.

_Anything else that cannot be run is a FAIL, not a SKIP._
