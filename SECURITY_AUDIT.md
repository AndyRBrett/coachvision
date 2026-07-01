# Security audit — coachvision

Scope: full repository (Python pipeline, GitHub Actions workflows, the `docs/`
PWA). Reviewed for injection, secret handling, SSRF, XSS, and unsafe use of
`subprocess`/`eval`/deserialization. No `eval`/`exec`/`pickle`/`yaml.load`/
`os.system`/`shell=True` usage exists anywhere in the codebase; every
`subprocess.run` call uses an argv list, never a shell string.

Findings 1-3 below have been **fixed**.

## Findings

### 1. Fixed — unrestricted URL fetch (SSRF / local-file read) in `decode_video.py`

`resolve_source()` → `_download()` passed `clip_url` straight to
`urllib.request.urlopen()` with no scheme or host allowlist. `urlopen` honors
any registered handler, including `file://`, so a `workflow_dispatch` caller
could point `clip_url` at `file:///etc/passwd` (or any path on the runner) or
at an internal-only address; ffmpeg's failure stderr (truncated to 500 bytes)
would then leak into the Actions log — a limited file-read/SSRF primitive,
worse on a **self-hosted runner** where it could reach internal services or a
cloud metadata endpoint (e.g. `169.254.169.254`).

Fix (`decode_video.py`): `_validate_download_url()` now rejects any scheme
other than `http`/`https` and resolves the hostname, rejecting
private/loopback/link-local/reserved/multicast addresses. A
`_SafeRedirectHandler` re-validates every redirect hop against the same
rules, so an initially-valid URL can't redirect its way past the check.

### 2. Fixed — incomplete ffmpeg `drawtext` escaping for tag text

`ffmpeg_trim_cmd()` escaped only `\`, `:`, and `'` before interpolating tag
text into the `drawtext` filter string, leaving other filtergraph-special
characters (`,`, `;`, `[`, `]`, `%`) unescaped. `tag_segment()` passes through
*unknown* event types verbatim, so a hand-crafted `tracking.json`/events
sidecar fed to the standalone `highlights.py` CLI could break out of the
filter argument. (Not reachable via the automated `process-footage` workflow,
where event types are always the fixed `hand_strike`/`leg_strike` strings or
vocabulary-restricted Cosmos tags.)

Fix (`highlights.py`): replaced character-by-character escaping with a
whitelist — `_sanitize_drawtext_label()` keeps only
`[A-Za-z0-9 ,_-]` and drops everything else, which covers the known tag
vocabulary exactly and removes the whole escaping-correctness question for
anything else that ends up there.

### 3. Fixed — `clip_path` accepted unrestricted filesystem paths

`resolve_source()` only checked `os.path.isfile()` — there was no confinement
to the repo root, so a workflow_dispatch caller could point ffmpeg at
arbitrary files on the runner via `../../etc/passwd` or an absolute path.

Fix (`decode_video.py`): `_require_within_cwd()` resolves the real path and
rejects anything outside the current working directory (the repo root, as
checked out by the workflow) before it's used.

### 4. Informational — GitHub PAT stored in `localStorage` (by design)

`docs/app.js` stores the user's GitHub token in plaintext `localStorage`
(`CFG_KEY`). This is a deliberate tradeoff for a backend-less PWA and is
disclosed in the README/UI ("Stored only in this browser"). The rest of
`app.js` is careful to avoid DOM XSS — all user/repo-sourced text (titles,
coaching feedback Markdown, summaries) is rendered via `textContent`/a
purpose-built safe Markdown renderer, never `innerHTML`, and the one
`innerHTML` template (`card()`'s stats line) only ever interpolates numbers
already computed by the trusted pipeline. No XSS path was found. Recommended
hardening (optional, not a bug): document scoping the PAT to the minimum
fine-grained permissions (already suggested in the README) and consider a
short-lived-token flow if this app ever grows beyond a single-user tool.

## What's already solid

- GitHub Actions `workflow_dispatch` inputs are passed through `env:` and
  referenced as shell variables (`"${NAME}"`, `"${CLIP_URL}"`, …) rather than
  interpolated directly as `${{ inputs.x }}` inside `run:` script bodies —
  this avoids the classic GitHub Actions script-injection vulnerability
  class.
- `permissions:` blocks are scoped to `contents: write` (or narrower) rather
  than defaulting to broad tokens.
- All `subprocess` invocations build argv lists; none pass through a shell.
- No secrets are committed in the repo; `ANTHROPIC_API_KEY` /
  `COACHVISION_COSMOS_API_KEY` are read only from environment variables and
  never logged or written to disk.
- `docs/sw.js` never caches GitHub API responses, avoiding stale/leaked data
  across accounts on a shared device.
