# Demo video — shot list and narration script

Target: ≤5 minutes (portal recommendation), screen capture, no editing required beyond
trimming dead air. Every command below is copy-pasteable from `README.md` — nothing here is
staged or faked. Record in a terminal + a file explorer/image viewer for the panels.

Read the narration lines aloud (or leave as on-screen captions) roughly as written; they are
short on purpose. Total speaking time at a natural pace is under 5 minutes even with command
output visible on screen between lines.

---

## Shot 1 — Title (10s)

**Screen:** `README.md` top of file, or a blank terminal with the repo name.
**Say:** "KLA PS01 — AI-based restoration of degraded images for semiconductor inspection.
This is a two-argument inference script, a checkpoint trained on a measured degradation
model, and a verifier that catches its own regressions. Full clone-to-inference walkthrough,
no edits."

## Shot 2 — Fresh clone (20s)

**Screen:** terminal, empty directory.
**Run:**
```bash
git clone https://github.com/sahithsundarw/semicon-kla-image-restoration.git
cd semicon-kla-image-restoration
```
**Say:** "Public repo, clean clone."

## Shot 3 — Environment (30s)

**Run:**
```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```
**Say:** "Complete pinned `pip freeze`. `requirements.txt` uses the PyTorch CUDA index
explicitly — installing `lpips` without that pin silently downgrades torch to a CPU-only
wheel, so this line matters more than it looks." *(Can cut the actual install wait with a
jump-cut — say "this takes about two minutes" over a sped-up clip.)*

## Shot 4 — Confirm CUDA (10s)

**Run:**
```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
**Say:** "Confirms the CUDA build actually installed — `True` at the end, not `False`."

## Shot 5 — Inference on the sample inputs (30s)

**Run:**
```bash
.venv/Scripts/python.exe inference.py --input_dir sample_inputs --output_dir results/sample_outputs
```
**Say:** "Two required arguments, input directory and output directory. That's the entire
interface. Weights are resolved relative to the script file, not the working directory, so
this runs correctly from anywhere. No checkpoint download needed — `weights/best.pt` is
tracked directly in the repo."
**Screen note:** let the log line print (`restored 6/6 in ... img/s ... device=cuda
precision=bf16`) — that line IS the evidence, don't talk over it.

## Shot 6 — Prove the I/O contract (20s)

**Run:**
```bash
.venv/Scripts/python.exe -c "import numpy as np; a=np.load('sample_inputs/000000.npy'); b=np.load('results/sample_outputs/000000.npy'); print('in ', a.shape, a.dtype, round(float(a.min()),4), round(float(a.max()),4)); print('out', b.shape, b.dtype, round(float(b.min()),4), round(float(b.max()),4))"
```
**Say:** "Input can legitimately exceed 1.0 — that's intentional, never clipped on the way in.
Output is exactly double the resolution, clipped to zero-one, same filename, same extension."

## Shot 7 — The numbers (30s)

**Screen:** `README.md`'s Result summary table, or `results/metrics_summary.md` directly.
**Say:** "29.25 dB PSNR, 0.792 SSIM, 0.256 LPIPS on the held-out 400-pair split — beats every
classical baseline and the from-scratch U-Net baseline on all three metrics, paired,
significant. 17.3 images per second end to end on this RTX 4060, including interpreter
startup and CUDA init, not just the forward pass."

## Shot 8 — A qualitative panel (20s)

**Screen:** open `results/qualitative/val_002041_weak_worst_in_split_psnr17.63.png` in an
image viewer, then `results/qualitative/failurecase_D5_*.png`.
**Say:** "This is the worst-scoring image in the entire validation split, shown deliberately,
not hidden. And this is a documented failure case — fine broadband texture that provably
exceeds what the low-resolution input can recover, identified by direct spectral measurement,
not guessed at."

## Shot 9 — The degradation forensics (20s)

**Screen:** `results/eda/noise_variance_vs_intensity.png` or the recovered-kernel figure.
**Say:** "The downsample kernel and noise model weren't assumed from the brief — they were
recovered by least squares over three million equations, and it turned out to be a
sharpening kernel with negative side lobes, not a box filter, and the noise has no additive
floor, contrary to the brief's own suggested model."

## Shot 10 — The verifier (25s)

**Run:**
```bash
.venv/Scripts/python.exe scripts/verify_all.py --strict
```
*(Let it run for a few seconds on screen, then cut to the final summary table — this can take
a long time end to end; do not run it live in full.)*
**Say:** "68 automated checks. Every one that's ever failed is either explained in
`docs/BLOCKERS.md` with the root cause, or fixed with the fix documented in `docs/decisions.md`
— nothing is silently skipped or weakened to turn a red check green."

## Shot 11 — Close (15s)

**Screen:** back to `README.md` top, or the repo's GitHub page.
**Say:** "Everything shown here — training command, inference command, every number — is
reproducible from this README with no manual edits. Thank you."

---

## Notes for whoever records this

- If GPU/CUDA isn't available on the recording machine, Shot 4 will print `False` and Shot 5
  will run on CPU — say so explicitly on camera rather than editing around it; the README
  documents CPU fallback as expected behavior, not a failure.
- Do not run the full `scripts/verify_all.py --strict` live end-to-end on camera — several
  checks intentionally spawn `inference.py` as a fresh subprocess (to measure real CUDA-init
  cost), and a few run multi-minute training smoke tests. Cut to a pre-recorded completed run
  for the final table, and say so.
- Keep terminal font large enough to read on a 1080p export.
- Do not narrate any number that isn't already written in `README.md`/`results/*.md` — if
  something is asked live that isn't scripted here, read it off the relevant file rather than
  estimating from memory.
