# Model weights

`best.pt` is the checkpoint `inference.py` loads. It is resolved as
`Path(__file__).resolve().parent / "weights" / "best.pt"` — relative to the **script**, never
to the current working directory and never an absolute literal (V05). A reviewer does not
have to pass `--weights`, set an environment variable, or edit anything.

Per SPEC §9 the checkpoint self-describes (V35): it carries `model`, `ema`, `config`, `iter`,
`metrics` and `git` keys, so weights can never be silently paired with the wrong
architecture. `inference.py` prefers the **EMA** weights when present.

---

## Status — 2026-08-15, iteration 2

**The checkpoint exists and is published as a GitHub Release asset (Route B, below).**

`weights/best.pt` is deliberately **not** in this repository: `.gitignore` bans `weights/*.pt`
and verification check V51 lists `.pt` as a forbidden blob, so committing it is not an option
here. It is instead served from a Release, with the digest of the served bytes recorded below.
Every value in this file was measured; none is a placeholder.

**If you cloned this repository, you do not have the model yet.** Download it before running
anything you intend to score — see *Download*, below. Without it `inference.py` falls back to a
bicubic upsample (details in the next section), which is not a model result.

### What `inference.py` does in the meantime

If the checkpoint cannot be found or cannot be loaded, `inference.py` prints

```
inference.py: checkpoint not found at <path>; falling back to bicubic x2 upsample
```

on stderr and completes with **exit code 0**, producing a parameter-free bicubic ×2 upsample
of each input. That is a deliberate degradation-not-crash policy: a script that runs and
scores badly is scored, and a script that crashes is not (CLAUDE.md PD4). It is **not** a
model result and must never be reported as one. Pass `--require_weights` to turn the fallback
into a hard failure — use that flag in any run whose output you intend to score.

---

## Download

**Route B — GitHub Release asset — is the route in force.** Route A (committing the file) was
available on size grounds (3.14 MiB, far under GitHub's 100 MB limit and under V43's cap) but
is closed here: `.gitignore` bans `weights/*.pt` and V51 lists `.pt` as a forbidden blob, and
weakening either to admit the file is not a change this project permits. GitHub Releases are
pre-approved by standing human authorisation (`docs/decisions.md` D23) and need no contract
change — V06 already permits exactly this mechanism.

Git LFS is **ruled out** — see `docs/decisions.md` D17 and D23. An unresolved LFS pointer stub
on a fresh clone is a known way to fail V06, and V06's own text names that failure mode.

### The checkpoint

| field | value |
|---|---|
| URL | https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/best.pt |
| sha256 | `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` |
| file size (bytes) | 3288805 |
| parameter count | 388,225 (NAFSR w48 n16) |
| architecture / config | `configs/final.yaml`; also embedded in the checkpoint under `config` |
| training seed | 42 |
| run id | `20260815T062831Z-final-s42` |
| git SHA of the training run | `80e7fb049367afe99fbcabb8e5469861f630fecc` (tree was dirty at launch; recorded as `-dirty` in `results/experiments.csv`) |
| weights shipped | EMA, at the best validation PSNR |
| validation | PSNR 28.7865 / SSIM 0.78287 / LPIPS 0.25324 over the full 400-image committed split, from `results/baselines/final/metrics.json` |

**Verified anonymously**, not from an authenticated tab: fetched with `GITHUB_TOKEN` and
`GH_TOKEN` cleared, the URL returned **HTTP 200**, **3288805 bytes**, and the sha256 of the
**served** bytes equals the digest above and the digest of the local file. Re-verify with:

```
curl -sSL -o best.pt https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/best.pt
py -3.12 -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('best.pt').read_bytes()).hexdigest())"
```

Place the file at `weights/best.pt`. `inference.py` resolves that path relative to its own
file (`Path(__file__).resolve().parent`), so nothing else needs configuring — no flag, no
environment variable, and it does not matter what directory you run from.

### Both artifacts on one Release

The same Release carries the restored test outputs archive
(`results/restored_test_outputs/README.md`, `docs/decisions.md` D23), so both digests are
verifiable from one place:

| artifact | Release asset | sha256 |
|---|---|---|
| `best.pt` (checkpoint) | [`artifacts-v1/best.pt`](https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/best.pt) | `9c0f39a72542a313aa74c00d6d0b40205b8504b8fcf3d5acfe92ba1149592313` |
| restored test outputs archive (400 files) | [`artifacts-v1/restored_test_outputs.zip`](https://github.com/sahithsundarw/semicon-kla-image-restoration/releases/download/artifacts-v1/restored_test_outputs.zip) | `fbdf8a652d26168cf41e01842ca28d38c53d1da1547bd8ce602b5b8e5d6ac750` |

Both digests were taken from the **served** bytes after an anonymous re-fetch, not from the
local copies. Provenance for the archive is in `results/restored_test_outputs/manifest.json`.

Releases page (live, HTTP 200):
`https://github.com/sahithsundarw/semicon-kla-image-restoration/releases`

> These commands live here rather than in the root `README.md` because V46 executes every
> fenced shell command in `README.md`, and a 3 MB download does not belong in a verification
> run. The same digest one-liner, pointed at `scripts/verify_all.py`, is what produced the
> hash pinned in `docs/VERIFIER_SHA256`.

## Checklist before submission

- [x] `best.pt` present in a **fresh clone** (Route A) **or** the table above complete and the
      URL fetched successfully from a logged-out session (Route B) — **Route B, verified
      HTTP 200 anonymously, served digest matches**
- [x] file > 1 KB and not an LFS pointer stub (V06) — 3288805 B, a real torch archive
- [x] checkpoint < 100 MB (V43) — 3.14 MiB
- [x] `build_model(ckpt["config"])` accepts the stored state dict with `strict=True` (V35)
      — V35 green
- [x] `inference.py --require_weights` succeeds against it — i.e. the bicubic fallback is
      *not* silently in play — 400/400 test outputs produced this way, see
      `results/restored_test_outputs/manifest.json`
- [ ] parameter count and checkpoint size recorded in `results/runtime_report.md` (V43)
