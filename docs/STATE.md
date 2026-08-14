# STATE
Iteration: 0
Last verified commit: (pending — bootstrap commit)
Verifier SHA: see `docs/VERIFIER_SHA256`

Bootstrap complete. No model built. No training run.

## V-check status  (from results/verification_report.json)
PASS: (see report — expected near-zero at iteration 0)
FAIL: all checks whose subject code does not exist yet, reported as `not implemented yet`
SKIP: V06 (no git remote configured yet — whitelisted, pre-push only)

Near-total failure at iteration 0 is the **correct** starting state (LOOP_PROMPT B6).

## In flight
- Nothing. Awaiting the first ITERATION pass.

## Consecutive-failure counters
(all zero — no fix has been attempted yet)

## Do NOT retry (tried and rejected, with the measurement that justified rejection)

- **Per-image min-max renormalisation of outputs.** Costs **-4.66 dB PSNR**, loses on
  191/200 held-out pairs. 95.5% of predictions overshoot 1.0, so renorm divides by an
  outlier-driven range. Clip to [0,1] and nothing else. (`docs/decisions.md` D3)
- **Treating the downsample kernel as a 2x2 box / average-pool.** Refuted. Least-squares
  recovery over 3.125 M equations gives centre weights 0.320 with negative surround lobes.
  Exact box costs +7.72e-04 residual std vs the optimum; `bicubic(antialias=False)` costs
  +1.22e-05. (`docs/decisions.md` D1)
- **The two-parameter noise model `sigma^2 + v*x^2` for SIMULATION.** It over-noises dark
  regions by up to 12.5x. Use the three-parameter fit: sigma=0, a=0.011253, v=0.015745.
  The 2-par values (sigma=0.036991, v=0.026781) remain the correct answer to SPEC §5.2 as
  literally posed, but must never be used to generate data. (`docs/decisions.md` D12)
- **Identifying the source dataset / DIV2K crop-match.** DENIED PERMANENTLY by the human,
  not pending. Identifying the source is the precondition for obtaining hidden labels, a
  confirmed match would sit in the repo as a pointer to them, and strategy is unchanged
  either way. Do not revisit in Round 2. (`docs/decisions.md` D11)
- **Pretrained initialisation for Phase 1.** Rejected: every classical x2 SR checkpoint
  assumes clean bicubic with no noise, while our inputs carry residual std 0.092 with 3% of
  pixels outside [0,1] — the prior points the wrong way. From scratch.
  (`docs/decisions.md` D9, D13)
- **`import cv2` / `tifffile` / any image library in `inference.py`.** The data is `.npy`
  end to end. The import is dead weight on a timed run and several cv2 paths silently
  convert to 8-bit or clip to [0,1], corrupting inputs that legitimately reach 2.16.
  Allowlist is exactly the eight in `CLAUDE.md` §STYLE. (`docs/SPEC_ADDENDUM.md` §5)
- **An 8-worker DataLoader for the 400-image test set.** Predicted net loss: the whole test
  set is 25.05 MB and worker spawn costs more than the read. Eager-load. Contradicts SPEC
  §11.2's recommendation deliberately. **Not yet measured** — `perf-analyst` must confirm or
  refute before this becomes final. (`docs/decisions.md` D7)

## Backlog (medium/low findings, no action yet)
- `results/restored_test_outputs/` will hold 400 `.npy` files (~105 MB) and is required by
  F12. Over GitHub's file limit and currently caught by the `*.npy` ignore. Needs Git LFS or
  hosted storage + sha256 before submission.
- `sample_inputs/` is empty; V47 needs 4-6 small `.npy` files committed.
- No git remote configured; V13 fails and V06 skips until one exists.

## Next iteration plan
1. Tier 0 first. The dependency root is `inference.py` + `src/io_utils.py` — V01-V12, V23
   all trace back to them. `inference-engineer` owns both.
2. `model-core` must land `build_model(cfg)` before inference can load anything; run it in
   the same wave (different files, no ownership collision).
3. `docs-scribe` for `README.md` + `requirements.txt` (V14, V46, V50).
4. Do NOT reserve a `dataset-forensics` slot — U1-U9 are all answered
   (`docs/SPEC_VCHECK_MAP.md`). That slot is free for other work, contrary to the standing
   instruction in LOOP_PROMPT §2 Step 3.
