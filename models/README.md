# models/

This directory exists solely to satisfy the organizers' final-submission announcement, which
specifies a submission folder shaped `team_name/{run.py, requirements.txt, README.md, models/}`
(docs/decisions.md D75). This repo's own convention for the checkpoint directory is
`weights/` (see `weights/README.md` for the full provenance record, sha256, training lineage,
and reproduction commands) — `models/best.pt` is a byte-identical mirror of `weights/best.pt`,
not a second, independent checkpoint.

`run.py` resolves its default weights at `weights/best.pt` first, falling back to
`models/best.pt` if that path doesn't exist (e.g. if a submission package strips everything
except this folder's contents). Either copy works.

**Keeping the two in sync is enforced, not just documented:** `scripts/verify_all.py`'s `V70`
check asserts `weights/best.pt` and `models/best.pt` have matching sha256 whenever both exist,
and fails loudly if they ever diverge. **Any future checkpoint promotion must copy the new
`weights/best.pt` into `models/best.pt` as one of its steps** — see the promotion checklist in
`docs/STATE.md`.
