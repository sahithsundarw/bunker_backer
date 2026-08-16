# Qualitative examples

These panels were regenerated from tracked `weights/best.pt` with checkpoint SHA256
`cc67c22f7cfc9926af7425bfa1af448237162d320970be6848129e0a3309d054`.

Validation panels show NoisyLR, restored output, GT, and absolute error. Their displayed
PSNR/SSIM values are single-file measurements using the same pinned settings as
`scripts/evaluate.py`. The D5 panel is an explicitly labeled out-of-split failure case and is
not included in the 400-pair validation mean.

Final-test panels show NoisyLR and restoration only. They state that no GT exists and do not
claim PSNR, SSIM, or LPIPS.

Regenerate from the canonical dataset layout with no untracked prediction dependency:

```bash
python scripts/make_qualitative_examples.py \
  --data_root /path/to/dataset \
  --checkpoint weights/best.pt
```

The script generates the selected validation and final-test predictions through
`inference.py --require_weights` when prediction directories are omitted. Existing prediction
directories may be supplied with `--val_pred_dir` and `--final_test_pred_dir`; current and
historical test-input layouts are accepted through `--final_test_lr_dir` or automatic
discovery. Checkpoint digest and embedded metrics are read dynamically, and stale generated
panel files are removed before the new set is written.

Machine-readable filenames, per-panel metrics, checkpoint identity, and the no-final-test-GT
note are in `manifest.json`.
