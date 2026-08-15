# Backtracking search summary

- Best experiment: `F001-tta`
- Best validation PSNR: **27.437886 dB**
- Best validation SSIM: **0.730049**
- Best validation LPIPS: **0.323138**
- Target: 29.0 dB
- Target reached: **no**
- Completed adaptive trials: 11
- Gain over LS-5: **+1.110209 dB**
- Gain over NLM bicubic: **+1.165686 dB**
- Validation protocol: 400 committed validation names; predictions saved and reloaded from disk.
- Data isolation: filters fitted on 2,800 training names only; final input-only data was not opened.

The best non-TTA model scored 27.437587 dB PSNR, 0.730039 SSIM, and 0.322458 LPIPS.
TTA added only +0.000299 dB and slightly worsened LPIPS to 0.323138. Finer 32-bin intensity
gating regressed by 0.014494 dB and was backtracked. The last accepted architecture gain was
only +0.007118 dB, so this closed-form branch is in clear diminishing-return territory.

## Budget decision

The configured NAFSR batch-32 step took 39 seconds on the available CPU. With CUDA and
MPS both unavailable, 20,000 iterations project to about 216 hours before periodic
validation. Neural length, loss, and larger-architecture branches therefore exhausted the
local compute budget before a full trial. The CPU-feasible adaptive search was completed
instead; exact parentage, commands, metrics, runtimes, and backtracks are in
`results/backtrack_experiments.csv`.
