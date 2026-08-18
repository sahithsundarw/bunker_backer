# Demo video — 3-minute overview script

Target: **~3 minutes**, screen capture, no editing beyond trimming dead air. Gives the basic
idea only — what the project does, how good it is, and how it's verified. Not a deep
technical walkthrough (see git history for the older 5-minute shot-list version if a longer
cut is ever wanted). Every command below is copy-pasteable from `README.md` — nothing here is
staged or faked.

Read the narration roughly as written; it's short on purpose. At a natural pace this is under
3 minutes even with command output visible on screen.

---

## Shot 1 — What this is (25s)

**Screen:** `README.md` top of file.
**Say:** "KLA PS01 — restoring degraded semiconductor-inspection images. Input is a noisy,
downsampled grayscale image; output is a clean estimate at double the resolution, in one
blind pass — the model doesn't know the noise level or degradation order in advance. The
released dataset is natural photographs, not semiconductor imagery — we say that plainly
instead of hiding it, and optimized for degradation robustness rather than content."

## Shot 2 — Run it (35s)

**Screen:** terminal, repo already cloned.
**Run:**
```bash
.venv/Scripts/python.exe run.py sample_inputs results/sample_outputs
```
**Say:** "Two arguments, input directory and output directory — that's the entire interface.
No checkpoint download, no editing, weights are resolved relative to the script itself so
this runs from anywhere." *(Let the log line print — `restored 6/6 in ... img/s ...
device=cuda precision=bf16` — that line is the evidence.)*

## Shot 3 — The numbers (35s)

**Screen:** README's result table or `results/metrics_summary.md`.
**Say:** "29.59 dB PSNR, 0.795 SSIM, 0.254 LPIPS on a held-out 400-image split — beats every
classical baseline and a from-scratch U-Net baseline on all three metrics, paired and
statistically significant. 19.4 images per second end to end on this RTX 4060, including
process startup and CUDA init, not just the forward pass — measured on this laptop, never
claimed as an H100 number."

## Shot 4 — Show, don't just tell (30s)

**Screen:** open two qualitative panels side by side — a strong result and the documented
worst case.
**Say:** "This is the best-scoring restoration in the split. And this is the worst — shown
deliberately, not hidden — fine texture that provably exceeds what the low-resolution input
can recover, confirmed by direct spectral measurement. The model blurs here, it doesn't
invent detail that isn't there, which matters most in an inspection context."

## Shot 5 — Generalization, honestly (30s)

**Screen:** the OOD section of README.
**Say:** "We also tested on 45 genuine electron-microscopy images — a real, licensed dataset,
not our own synthetic content. The model wins there too on perceptual quality, with no
regression on any metric. Where a result looked too good — a 40-image procedural test set —
we say so directly: that's likely just content overlap with training, not proof of general
robustness. We'd rather under-claim than over-claim."

## Shot 6 — How this is kept honest (25s)

**Screen:** `results/verification_report.json` or the terminal after a verifier run.
**Say:** "71 automated checks validate this end to end — I/O correctness, no data leakage,
reproducibility, throughput measurement method, even that the model can't accidentally train
on the hidden test set. Every failure is either explained with its root cause or fixed with
the fix documented — nothing is ever silently skipped to turn a check green."

## Shot 7 — Close (10s)

**Screen:** repo's GitHub page.
**Say:** "Everything shown here is reproducible from this repository with no manual edits.
Thank you."

---

## Notes for whoever records this

- If GPU isn't available on the recording machine, Shot 2 falls back to CPU automatically —
  say so on camera rather than editing around it; CPU fallback is documented, expected
  behavior, not a failure.
- Don't run the full verifier live on camera (it takes several minutes and forks subprocess
  benchmarks) — show a completed run's summary table instead, and say so.
- Don't narrate any number that isn't already written in `README.md` / `results/*.md` — if
  asked something live that isn't scripted here, read it off the file rather than estimating.
- Keep terminal font large enough to read on a 1080p export.
