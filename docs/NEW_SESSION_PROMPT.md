# New-session prompt

Paste the block below into a fresh Claude Code session started in
`C:\Users\sahit\OneDrive\Desktop\semi`. It is written to be self-contained: it assumes the
model knows nothing and points it at the on-disk ledger rather than restating it.

---

```
Fresh session, no memory of prior work. Everything is on disk.

Project: KLA PS01 — AI-Based Restoration of Degraded Images (SEMICON India Hackathon 2026).
Repo: C:\Users\sahit\OneDrive\Desktop\semi   Python: `py -3.12`   GPU: RTX 4060 Laptop, 8 GB.
Remote: https://github.com/sahithsundarw/semicon-kla-image-restoration (public).
Dataset lives OUTSIDE the repo at C:\kla-data and must never be committed.

READ IN THIS ORDER BEFORE DOING ANYTHING:
  1. docs\STATE.md            — start at "RESUME HERE". This is the resume point; trust it
                                over anything else. It says what is done, what is next, and
                                what NOT to retry.
  2. docs\MORNING_REPORT.md   — narrative status: results, what is red and why.
  3. CLAUDE.md                — operating rules, prime directives, file-ownership map.
  4. docs\SPEC_ADDENDUM.md    — BINDING. It overrides docs\SPEC.md wherever they conflict.
                                Four of SPEC's dataset assumptions are measurably wrong.
  5. docs\VERIFICATION_CONTRACT.md — the checks that define "done" (57 of them).
  6. docs\decisions.md D1-D29 and docs\BLOCKERS.md B1-B9.
Precedence: VERIFICATION_CONTRACT > SPEC_ADDENDUM > SPEC.

STEP 0 — DO THIS FIRST, BEFORE ANY OTHER WORK.
A previous session completed a large amount of work but lost shell access partway through, so
the tree is fully consistent but UNPUSHED. Nothing is half-written. Commit and push it:

    git status --short          # sanity: no *.pt, no *.npy outside sample_inputs/
    git add -A
    git commit -F "C:\Users\sahit\.claude\jobs\350d944b\tmp\commit_msg.txt"
    git push origin main

If that message file is gone, write your own — the substance is in docs\decisions.md D28+D29.
Then run `py -3.12 scripts\verify_all.py --strict` to get a real tally. Do not trust any
pass/fail number written in the docs; they predate the last four checks and the trained model.

WHERE THINGS STAND
A 20k-iteration NAFSR run has COMPLETED and beats every baseline on all three metrics, on the
400-image committed validation split, scored from reloaded .npy on disk:

    bicubic x2               23.6524 dB / 0.54775 / LPIPS 0.41206
    median 3x3 -> bicubic    25.5057 dB / 0.61317 / LPIPS 0.40870
    non-local means -> bic   26.2722 dB / 0.65152 / LPIPS 0.42586
    NAFSR w48 n16 (EMA)      28.7851 dB / 0.78279 / LPIPS 0.25233   <- ours

Nothing is running. No agent, no training job, no background shell.

THEN, IN THIS ORDER:
  1. Publish weights\best.pt as a GitHub Release asset with a sha256, recorded in
     weights\README.md. Highest value, no GPU needed. Closes V06 and V59. Until this happens,
     anyone cloning the repo gets a bicubic upsampler instead of the model.
  2. Run scripts\evaluate.py on the trained checkpoint so results\baselines\final\metrics.json
     exists. V27, V28 and V48 read THAT FILE, not the training log — a number in a log is not
     an evaluation record, which is why those checks are still red despite a good model.
  3. Generate the 400 restored test outputs with --require_weights (mandatory — see gotchas)
     and attach them to the same Release. Closes V56.
  4. Tag v0.1-submittable and push the tag once Tier 0 is green, so a working fallback always
     exists on the remote.
  5. Train the U-Net baseline at the SAME 20k budget (configs\baseline_unet.yaml, ~60-90 min).
     V28 needs a LEARNED baseline at equal budget; the three above are classical, so the
     rubric's like-for-like comparison is genuinely missing right now.
  6. Dispatch perf-analyst for the runtime report (V37-V39, V43). Re-run adversarial-reviewer,
     which was killed before it ever wrote its file and is the one review still owed. Run
     cleanroom-tester once the README is final.
  7. Then the §3 hardening loop in LOOP_PROMPT.md. Model quality first, throughput second.

HARD RULES — never, without asking me:
  - Weaken, delete, skip, or widen the tolerance of any check. Making a check STRICTER is
    pre-authorised; log it and re-pin docs\VERIFIER_SHA256.
  - Edit docs\VERIFICATION_CONTRACT.md except to ADD or TIGHTEN.
  - Train, fine-tune, or fit anything on C:\kla-data\test_NoisyLR. There is no test ground
    truth; score only against configs\split_val.txt.
  - Download DIV2K or attempt to identify the source dataset. Permanently denied.
  - Commit any dataset file, *.pt, or anything over the V51 caps (5 MB/file, 25 MB total).
  If you find yourself reasoning toward any of these because it would unblock progress: STOP,
  write it to docs\BLOCKERS.md, and work something else. That reasoning is the signal, not the
  justification.

GOTCHAS THAT WILL COST YOU HOURS IF YOU REDISCOVER THEM:
  - The released images are grayscale NATURAL PHOTOGRAPHS, not semiconductor imagery. The
    problem domain is inspection; the data is a PROXY. Never describe the provided data as
    semiconductor in code, docs, commits or the deck. What transfers is the measured
    degradation, not any content prior — so prefer wide degradation randomisation over
    squeezing in-distribution dB.
  - inference.py silently falls back to bicubic when no checkpoint is present. Always pass
    --require_weights when producing artifacts, or upsampler output ships as model results.
  - `pip install lpips` silently replaces the CUDA torch with a CPU-only build. Reinstall from
    the cu128 index and re-check torch.cuda.is_available().
  - Quote 28.7851 dB (full 400-image split), never the 30.3944 in the training log — that is a
    100-image checkpoint-selection subset.
  - A new V-check is code like any other. Two shipped broken the same day they were written
    (a false positive, and an SSRF hole). Verify every new absence-check with a NEGATIVE
    CONTROL: inject the defect, confirm red, remove it, confirm green.
  - Nine of the original 53 checks were inert placeholders that no artifact could ever turn
    green, and eleven requirements had no check at all. Do not assume a green suite means
    covered; read what a check actually asserts.

Update docs\STATE.md and docs\MORNING_REPORT.md as you go, commit and push after every
meaningful step, and tell me what you measured rather than what you expect.
```
