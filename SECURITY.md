# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 8.x.x | Yes |
| < 8.0 | No |

## Reporting a Vulnerability

Do not open a public issue for a security vulnerability.

Report privately to: abinazebinoy@gmail.com

Include:
- A description of the vulnerability
- Steps to reproduce it
- Its potential impact
- A suggested fix, if you have one

Expected initial response time: 48 hours. Please allow a reasonable period to investigate and release a fix before any public disclosure.

## Security Measures in Place

- **Access control** — API-key based role access control (admin / analyst), enforced through a single shared FastAPI dependency (`backend/core/auth.py`) applied uniformly across every router, rather than a per-file reimplementation that could drift or be forgotten on a new endpoint
- **Key storage** — API keys are stored as salted hashes only; the raw key is shown exactly once, at creation
- **Admin gate fails closed** — admin-only endpoints return `503` unless `ADMIN_KEY_HASH` is explicitly configured, rather than falling back to a weaker check by default
- **SSRF protection** — webhook registration resolves the target hostname and rejects private, loopback, link-local, reserved, and multicast IP ranges before accepting the URL
- **Input validation** — file type is checked against actual file content (magic bytes), not just the extension or the client-supplied MIME type, and the two are cross-checked against each other
- **Rate limiting** — per-IP sliding-window limits on every endpoint, both a per-route override and a global default
- **Output escaping** — user-supplied strings that reach the frontend (filename, EXIF field values, tampering-flag text) are HTML-escaped before being rendered, to prevent stored XSS
- **No persistent storage of uploaded images** — uploaded files are processed in memory for the duration of a single request and are not written to disk
- **Dependency scanning** — `pip-audit` runs on every CI build against all pinned dependency files

## What Is Logged

VeriFile-X keeps an append-only audit log (`backend/core/audit_log.py`) recording the uploaded filename, a content hash, and the resulting classification for each analysis, for operational accountability. This is a disclosed, deliberate feature, not incidental data collection — the raw image itself is not persisted, and the log does not record request source IPs in plaintext (IPs are hashed before being written, where logged at all).

If your deployment has additional privacy requirements (for example, not recording filenames), that is a configuration decision to make before deploying, not something this document guarantees on your behalf.

## Known, Tracked Limitations

- `data/api_keys.jsonl` grows by one line per successful authentication with no rotation yet implemented; tracked in [PHASE_ROADMAP.md](PHASE_ROADMAP.md)
- Webhook SSRF validation currently runs at registration time; re-validating at delivery time (to defeat DNS rebinding between registration and delivery) is a tracked follow-up
- Several third-party dependencies (notably `transformers` and `torch`) have open, upstream-unfixed CVEs at the time of writing — see the repository's Dependabot alerts for current status; these are tracked, not ignored, and will be resolved as upstream stable fixes ship

## License and Responsible Disclosure

This project is licensed under the terms in [LICENSE](LICENSE). Reporting a vulnerability does not grant any additional rights beyond that license, and does not obligate the maintainer to a bug bounty or similar compensation.
