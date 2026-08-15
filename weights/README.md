# Model weights

`best.pt` is the checkpoint `inference.py` loads. It is resolved as
`Path(__file__).resolve().parent / "weights" / "best.pt"` — relative to the **script**, never
to the current working directory and never an absolute literal (V05). A reviewer does not
have to pass `--weights`, set an environment variable, or edit anything.

Per SPEC §9 the checkpoint self-describes (V35): it carries `model`, `ema`, `config`, `iter`,
`metrics` and `git` keys, so weights can never be silently paired with the wrong
architecture. `inference.py` prefers the **EMA** weights when present.

---

## Status — 2026-08-15, iteration 1

**No checkpoint exists yet. Training has not been run.**

`weights/best.pt` is absent from this repository, and `weights/*.pt` is currently in
`.gitignore`. **V06 is therefore FAILING, correctly.** It will stay red until one of the two
routes below is complete. Nothing in this file is a placeholder standing in for a value that
was measured and then omitted — the values are genuinely not yet known.

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

## Download — to be completed when the checkpoint exists

Exactly one of these two routes must be satisfied before submission:

**Route A — commit the file.** The checkpoint is small enough for plain git: D19 measures a
constructed NAFSR w48 n16 checkpoint (model + EMA, 388,225 parameters) at **3.14 MiB**, far
under GitHub's 100 MB limit and under V43's cap. This is the preferred route because it has
no external dependency and no link to rot. It requires removing the `*.pt` line from
`.gitignore` for this one path, which is a `.gitignore` change the main session owns.

**Route B — GitHub Release asset.** Publish the file as a Release asset, then fill in the
table below with a URL that returns **HTTP 200 from a logged-out session** (verify in a
private browser window, not just in your own tab) and the sha256 of the exact bytes served.
GitHub Releases are pre-approved by standing human authorisation (`docs/decisions.md` D23) and
need no contract change — V06 already permits exactly this mechanism.

Git LFS is **ruled out** — see `docs/decisions.md` D17 and D23. An unresolved LFS pointer stub
on a fresh clone is a known way to fail V06, and V06's own text names that failure mode.

The same Release carries the restored test outputs (`results/restored_test_outputs/README.md`,
`docs/decisions.md` D23). Record that archive's digest here too, so both artifacts are
verifiable from one place:

| artifact | Release asset | sha256 |
|---|---|---|
| `best.pt` (checkpoint) | *pending* | *pending* |
| restored test outputs archive (400 files) | *pending* | *pending* |

Releases page (live, HTTP 200):
`https://github.com/sahithsundarw/semicon-kla-image-restoration/releases`

| field | value |
|---|---|
| URL | *pending — no checkpoint yet; do not fabricate* |
| sha256 | *pending — must be the sha256 of the served bytes, not of a local copy* |
| file size (bytes) | *pending* |
| parameter count | *pending — 388,225 for NAFSR w48 n16 if that config ships unchanged (D19)* |
| architecture / config | *pending — mirrors `configs/final.yaml`, also embedded in the checkpoint* |
| training seed | *pending — 42 unless the shipped run says otherwise* |
| git SHA of the training run | *pending — embedded in the checkpoint under `git`* |

Compute the digest with:

```
py -3.12 -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('weights/best.pt').read_bytes()).hexdigest())"
```

> Recorded here rather than in the root `README.md` because there is no file for it to hash
> yet, and V46 executes every fenced shell command in `README.md`. The same one-liner,
> pointed at `scripts/verify_all.py`, is what produced the digest pinned in
> `docs/VERIFIER_SHA256`.

## Checklist before submission

- [ ] `best.pt` present in a **fresh clone** (Route A) **or** the table above complete and the
      URL fetched successfully from a logged-out session (Route B)
- [ ] file > 1 KB and not an LFS pointer stub (V06)
- [ ] checkpoint < 100 MB (V43)
- [ ] `build_model(ckpt["config"])` accepts the stored state dict with `strict=True` (V35)
- [ ] `inference.py --require_weights` succeeds against it — i.e. the bicubic fallback is
      *not* silently in play
- [ ] parameter count and checkpoint size recorded in `results/runtime_report.md` (V43)
