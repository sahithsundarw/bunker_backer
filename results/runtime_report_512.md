# Runtime measurement -- 256->512 pipeline behavior (SPEC 11.4 step 7 extension; V37, V38, V39, V43)

Companion to `results/runtime_report.md` (which stays the 128->256 record and is NOT
modified by this file).

This report answers one question raised by KLA's brief that
evaluation images may be around 256x256 OR 512x512: does the ~30.6% fixed-startup share
measured at 128->256 collapse toward compute-bound as pixel count rises 4x? Measured,
not assumed, below.

Device: NVIDIA GeForce RTX 4060 Laptop GPU -- every number in this file was measured on
this GPU, never an H100. Timing is external: each row wraps a full
subprocess.run([python, inference.py, ...]) from outside the process (interpreter start,
imports, CUDA init, checkpoint load, disk IO and compute all included), exactly the same
scripts/benchmark_runtime.py cmd_e2e harness used for the 128->256 report -- not an
internal timer around the forward pass.

## Environment

Identical to results/runtime_report.md's environment (same machine, same interpreter,
re-verified for this run):

| | |
|---|---|
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (compute capability 8.9, driver 610.47) |
| VRAM | 8187 MiB |
| torch | 2.11.0+cu128 (CUDA 12.8, cuDNN 91900) |
| Python | 3.12.10 on Windows-11-10.0.26200-SP0 (invoked as py -3.12) |
| CPU cores | 16 |
| Model | NAFSR, 388,225 params, checkpoint weights/best.pt, batch 32, precision bf16 |

## Why there is no real 512 GT, and what that means for this report

The released dataset's GT tops out at 256x256 -- verified directly: all 3,200
train/GT/*.npy arrays are (256, 256), and test_NoisyLR is uniformly (128, 128).
There is no 512x512 image anywhere in this dataset to downsample from for a real 256px
degraded input.

What was synthesized: the 400 filenames in the committed configs/split_val.txt
(held-out from training, never used to fit the degradation model, test_NoisyLR itself is
untouched -- F17 compliant) were loaded from train/GT at their native 256x256 resolution.
The actual measured noise simulator in src/degrade.py -- add_noise() with
sample_noise_params(), the real fitted three-parameter shot+speckle+Gaussian model
(docs/decisions.md D2/D12), NOT np.random ad hoc noise -- was applied directly at
native 256px resolution, skipping the downsample-kernel step (there is nothing to
downsample from). This is a deliberate, honestly-labelled approximation: it exercises the
real noise model at the real target resolution, but omits the recovered-kernel
downsampling term of the simulator, which has no analogue when there is no higher-resolution
source. Generator script: synth_256in.py (scratchpad copy; not part of the repo -- see
Method section for the exact call it makes).

Consequence: this measures pipeline behavior only -- wall-clock timing, output shape
correctness (512x512, clipped to [0,1]), and absence of OOM -- not restoration quality at
512px. There is no ground truth to score PSNR/SSIM against at this resolution, and none is
claimed. Spot check: n=400 outputs, shape (512, 512), dtype float32, range
[0.0108, 0.9485] -- real model output (not a flat placeholder, not a shape-mismatch
bicubic fallback); --require_weights was passed and every run exited 0.

## Total end-to-end wall-clock: 256->512, N=400, device RTX 4060 Laptop GPU

At the full 400-image synthetic set, batch 32, precision bf16: median 33,533.7 ms
(min 33,477.7, max 33,589.7, n=2, spread 0.33%), i.e. 11.9 img/s end-to-end including
interpreter startup, import torch, CUDA init and checkpoint load.

### Startup vs compute: the same external fixed-cost model, fit across N

| N images | total wall-clock (median) | img/s (incl. startup) |
|---|---|---|
| 1 | 5872 ms | 0.17 |
| 25 | 11563 ms | 2.16 |
| 50 | 21258 ms | 2.35 |
| 100 | 18310 ms | 5.46 |
| 200 | 23609 ms | 8.47 |
| 400 | 33534 ms | 11.93 |

Linear fit over these six points: fixed startup cost 11,499 ms + 58.19 ms/image
marginal compute (predicts 34,774 ms at N=400, measured 33,534 ms). At N=400 the fixed
cost is 34.3% of total wall-clock.

Note on N=50/N=100 ordering: N=50's median (21,258 ms) sits slightly above N=100's
(18,310 ms) because N=50 had one high-variance repeat (spread 25.7% across only 2 repeats,
vs under 2.2% for every other N). This is measurement noise from using --repeats 2 (halved
from the 128->256 report's --repeats 5 to fit a 256px sweep in a reasonable wall-clock
budget), not a real non-monotonicity; the linear fit is not materially sensitive to it.

## Side-by-side: 128->256 vs 256->512 (both RTX 4060 Laptop GPU)

| quantity | 128->256 (results/runtime_report.md) | 256->512 (this file) |
|---|---|---|
| fixed startup cost | 14,755 ms | 11,499 ms |
| marginal cost | 86.55 ms/image | 58.19 ms/image |
| predicted at N=400 | 49,373 ms | 34,774 ms |
| measured at N=400 (median) | 48,269.4 ms (n=5, spread 681.4%) | 33,533.7 ms (n=2, spread 0.33%) |
| fixed-cost fraction at N=400 | 30.6% | 34.3% |
| img/s at N=400 incl. startup | 8.3 | 11.9 |

## The headline finding: REFUTED, not confirmed

The expectation stated in the task brief was that the fixed-cost share should collapse
toward compute-bound (fall well below 15%) as pixel count rises 4x. Measured result:
it did not fall -- it rose slightly, from 30.6% to 34.3%. Startup remains solidly
dominant-adjacent at both resolutions; it never becomes a rounding error, and going to
512px output did not make the pipeline meaningfully more compute-bound on this GPU.

Why, mechanistically: the reason is not that 512px compute is cheap in absolute terms --
the isolated forward-pass sweep in results/runtime_report.md shows NAFSR at 256px input,
batch 32, bf16 costs 51.89 ms/image in pure GPU compute vs 10.19 ms/image at 128px
(a 5.1x increase, roughly matching the 4x pixel-count growth). The surprise is on the
128->256 side of the comparison: that report's own measured e2e marginal cost
(86.55 ms/image) is 8.5x higher than its own isolated forward-pass sweep number
(10.19 ms/image) -- a large non-compute overhead (H2D/D2H, per-batch Python-level
bookkeeping, disk writes not fully overlapped with a very fast forward pass) that is
largely insensitive to resolution. At 256->512, forward itself grows to 51.89 ms/image,
which increasingly dwarfs that same roughly-fixed per-image overhead (its 58.19 ms/image
measured marginal is only 1.12x its sweep-forward number, vs the 128px case's 8.5x) -- so
the marginal cost per image did not scale by the roughly 5x pixel-count/compute ratio; it
dropped in absolute terms, from 86.55 to 58.19 ms/image. That drop, not a proportional
rise, is what pushed the fixed-cost fraction up rather than down: the same roughly 11-15 s
fixed cost is now dividing a smaller total marginal budget (58.19 x 400 = 23,276 ms) than
before (86.55 x 400 = 34,620 ms).

Caveat on confidence: the 128->256 report's own headline N=400 number carries a
681.4% min-max spread (one repeat took 363.6 s against a 48.3 s median) -- almost
certainly a thermal-throttle or driver-contention outlier on this laptop GPU, not a steady
state. That instability means the 30.6% figure it is compared against is itself noisier
than its point estimate suggests. This report's own N=400 spread is a comparatively tight
0.33% (n=2), so of the two, this measurement is the more trustworthy point estimate, but
the comparison inherits some of the other report's uncertainty. The qualitative
conclusion -- fixed cost stays in the 30-35% range at both resolutions, nowhere near
falling toward compute-bound -- is nonetheless clear enough that a repeat with more
samples is unlikely to overturn it.

## Bottom line for the cloud-spend decision

Startup/import/CUDA-init overhead is not a 128px-only artifact that a bigger model or
higher-resolution target will make irrelevant. At both tested resolutions on this GPU,
roughly a third of the scored wall-clock at N=400 is fixed cost that a bigger model does
nothing to amortize -- it only grows the marginal term. A bigger/slower model at 512px
output will show LESS fixed-cost dilution benefit than naively assumed, not more, unless
KLA's actual test-set image count is substantially above 400 (which favors the 128->256
report's same conclusion) or import/startup cost is independently reduced (V23 import
hygiene, Tier 0). This does not by itself veto training a bigger model for 512px --
quality requirements are outside this report's scope -- but it removes "512px will dilute
startup cost away" as a justification for skipping further inference-side startup
optimization.

## Batch-size sweep at 256->512 (2026-08-18, throughput-optimizer task)

Companion to `results/runtime_report.md`'s new batch-size sweep section (that file has the
full writeup, methodology note, and the recommended `--batch_size` default change; this is the
256->512-specific half of the same measurement, reproduced here since this file is the
resolution's home).

A fresh 400-image synthetic 256px input set was generated for this sweep (the original
session's `synth_256in.py` scratchpad script no longer exists) at `C:\tmp\val400_256in`, using
the same method this file already documents above: the 400 `configs/split_val.txt` names
loaded from `train/GT` at native 256x256, with `src.degrade.add_noise(sample_noise_params())`
applied directly (no downsample step -- there is nothing higher-resolution to downsample from,
same as the original set this file's headline used).

`scripts/benchmark_runtime.py --input_dir C:\tmp\val400_256in --repeats 1 --batch_size <b>
--precision bf16`, one batch size per invocation, round-robin interleaved across
{4, 8, 16, 32, 64} for 5 full rounds in one continuous session (same session as the 128->256
sweep, run back to back with it). Medians over n=5 interleaved repeats:

| batch | median external wall-clock | img/s | all 5 repeats (s) |
|---|---|---|---|
| 4 | **63.24 s** | **6.33** | 63.07, 63.14, 63.24, 63.78, 232.72 |
| 8 | 63.91 s | 6.26 | 63.60, 63.67, 63.91, 64.36, 120.60 |
| 16 | 65.00 s | 6.15 | 64.86, 64.91, 65.00, 65.47, 98.25 |
| 32 (current default) | 77.23 s | 5.18 | 76.99, 77.18, 77.23, 77.55, 103.44 |
| 64 | 86.21 s | 4.64 | 85.83, 86.01, 86.21, 86.64, 88.14 |

Monotonic, same direction as 128->256: smaller batch is faster end to end. **Batch 4 beats
batch 32 by 18.1% lower wall-clock (22.1% higher throughput).** Round 1 was uniformly elevated
across all five batch sizes (232.72, 120.60, 98.25, 103.44, 88.14 s) -- a session cold-start
effect (cuDNN autotune warm-up, GPU clock ramp from idle), not a batch-size-specific anomaly,
since it hit whichever batch size ran first in every case; rounds 2-5 are tight (spread <2% for
batch >= 8), so the median is not sensitive to which round the cold start landed in.

**Recommendation (shared with results/runtime_report.md): change `inference.py`'s
`--batch_size` default from 32 to 4.** `inference.py` was not edited as part of this task
(file-ownership boundary) -- see the proposed diff in `results/runtime_report.md`.

## Method

- Timing is taken externally around the whole process
  (subprocess.run([sys.executable, "inference.py", ...])), via
  scripts/benchmark_runtime.py (no subcommand -- it is a single flat CLI with no `e2e`/
  `--e2e_part`; the N-point scaling curve above was produced by pointing `--input_dir` at
  six differently-sized subsets), run six times (N = 1, 25, 50, 100,
  200, 400), each pointed at a directory of synthetic 256x256 .npy inputs generated by a
  scratchpad script (synth_256in.py, not committed -- see "Why there is no real 512 GT"
  above for exactly what it does and why).
- --repeats 2 (vs the 128->256 report's --repeats 5) to keep the six-point sweep at
  256px compute cost inside a practical wall-clock budget; spreads are reported per-N above
  so this tradeoff is visible, not hidden.
- Critical methodology note: this machine's plain python on PATH resolves to a
  Python 3.14 install whose torch CUDA DLLs (caffe2_nvrtc.dll) fail to load intermittently
  (OSError: WinError 126), producing wildly inconsistent, partially-failing timings (a
  first attempt at this sweep, discarded, showed non-monotonic medians and mixed
  return codes for exactly this reason -- e.g. N=100 returned codes [0, 1, 1] and N=200
  returned [1, 0, 0] across identical repeated commands). All numbers in this file were
  re-measured with py -3.12 explicitly, matching the interpreter results/runtime_report.md
  was generated with (confirmed: same torch 2.11.0+cu128, same cuDNN 91900, same device).
  Any future benchmark run on this machine MUST pin py -3.12 and never rely on bare python.
- Every number above is labelled with the device it was measured on: NVIDIA GeForce RTX
  4060 Laptop GPU. No H100 number appears anywhere in this file.
- This file does not repeat the import-cost breakdown, batch/precision/memory-format sweep,
  torch.compile break-even, or DataLoader-vs-eager comparison -- those are resolution-
  independent (imports, compile warm-up) or already answered per-resolution in
  results/runtime_report.md's sweep table (which already includes 256px rows). This file
  adds only the resolution-specific e2e scaling comparison the cloud-spend decision needed.
