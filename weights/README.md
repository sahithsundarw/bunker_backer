# Model weights

`best.pt` is the final checkpoint: EMA weights, config, iteration, metrics and git SHA
(V35). It is loaded automatically by `inference.py` via `Path(__file__).parent` — no
user action required (V05).

## Status

**Not yet produced.** Training has not been run. V06 will FAIL until either the file is
present in the clone (Git LFS resolved, >1 KB, not a pointer stub) or this file carries a
download URL returning HTTP 200 from a logged-out session plus a matching `sha256`.

## Download

<!-- TODO(docs-scribe): URL + sha256 once the checkpoint exists. -->

| field | value |
|---|---|
| URL | _pending_ |
| sha256 | _pending_ |
| size | _pending_ |

Checkpoint must stay under 100 MB or be hosted via Git LFS / an external link (V43,
SPEC 18 pitfall 8). Test any link from a logged-out browser before submission.
