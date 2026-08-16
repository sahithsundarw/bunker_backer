# Link check — SPEC §2.3 official resources

Closes U-10 (`requirements-auditor`): the licence links in `docs/decisions.md` were re-fetched
and dated, but SPEC §2.3's hackathon resource links were never independently re-verified.

Every URL below is re-issued **unauthenticated** — no cookies, no saved session, a fresh
process, `GITHUB_TOKEN`/`GH_TOKEN` cleared from the environment even though none of these hosts
are GitHub — so a pass cannot be an artifact of a signed-in browser tab. `-L` follows redirects
and the reported status is the code for the **final** URL in the chain.

| URL | Status | Checked (UTC) |
|---|---|---|
| `https://i4c.in/hackathon-2026/` | 200 | 2026-08-15T19:48:49Z |
| `https://hackathon2026.i4c.in/` | 200 | 2026-08-15T19:48:49Z |
| `https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link` | 200 | 2026-08-15T19:48:49Z |
| `https://i4c.in/wp-content/uploads/2026/08/7b675083-e081-47d3-8c55-fde76a77b673.pptx` | 200 | 2026-08-15T19:48:49Z |
| `https://i4c.in/wp-content/uploads/2026/07/Idea-Submission-Template_Hackathon-2026-1.pptx` | 200 | 2026-08-15T19:48:49Z |
| `https://youtu.be/RMSDaviTOIw` | 200 | 2026-08-15T19:48:49Z |
| `https://www.youtube.com/watch?v=Q__rlK1Q3uw` | 200 | 2026-08-15T19:48:49Z |
| `https://i4c.in/wp-content/uploads/2026/01/How-to-register-for-IESA-Hackathon-2026.pdf` | 200 | 2026-08-15T19:48:49Z |
| `https://chat.whatsapp.com/D9QI2JRBTTO5BUw57Y71wC` | 200 | 2026-08-15T19:48:49Z |

All 9 links from SPEC §2.3 are live. Reproduce with:

```
curl -sS -o /dev/null -w "%{http_code}\n" -L --max-time 20 -A "Mozilla/5.0" "<url>"
```

**Staleness.** This file is only trustworthy for **72 hours** past the timestamp above (V58's
bound) — re-run the check and update this table if it is older than that when read.
