# Security audit — coachvision

Scope: full repository (Python pipeline, GitHub Actions workflows, the `docs/`
PWA). Reviewed for injection, secret handling, SSRF, XSS, and unsafe use of
`subprocess`/`eval`/deserialization. No `eval`/`exec`/`pickle`/`yaml.load`/
`os.system`/`shell=True` usage exists anywhere in the codebase; every
`subprocess.run` call uses an argv list, never a shell string.

## Findings

### 1. Medium — unrestricted URL fetch (SSRF / local-file read) in `decode_video.py`

`resolve_source()` → `_download()` (`decode_video.py:84-100`) passes
`clip_url` straight to `urllib.request.urlopen()` with no scheme or host
allowlist. `urlopen` honors any registered handler, including `file://`, so a
`workflow_dispatch` caller can point `clip_url` at `file:///etc/passwd` (or
any path on the runner) or at an internal-only address. ffmpeg will fail to
decode the result, but its stderr (truncated to the last 500 bytes,
`decode_video.py:67`) is echoed into the Actions log, giving a limited file/
SSRF read primitive. On GitHub-hosted runners the blast radius is small
(ephemeral, no interesting local secrets beyond what the job already has in
env); on a **self-hosted runner** this could reach internal services or cloud
metadata endpoints (e.g. `169.254.169.254`).

Fix: restrict `clip_url` to `http`/`https`, and if self-hosted runners are
ever used, block link-local/private-IP destinations (and disable redirects to
them).

### 2. Low — incomplete ffmpeg `drawtext` escaping for tag text

`ffmpeg_trim_cmd()` (`highlights.py:136-160`) escapes only `\`, `:`, and `'`
before interpolating tag text into the `drawtext` filter string. Other
characters meaningful to ffmpeg's filtergraph syntax (`,`, `;`, `[`, `]`,
`%`) are not escaped. `tag_segment()` passes through *unknown* event types
verbatim (`highlights.py:112-133`), so a hand-crafted `tracking.json`/events
sidecar with a malicious `type` string could break out of the filter
argument. In the automated `process-footage` workflow this isn't reachable
(event types there are the fixed `hand_strike`/`leg_strike` strings from
`fight_analysis.py`, and Cosmos-derived tags are vocabulary-restricted in
`merge_tags`), but the standalone `highlights.py` CLI accepts arbitrary
tracking JSON directly. Escape the full set of ffmpeg-special characters, or
validate tags against the domain vocabulary before use.

### 3. Low — `clip_path` accepts unrestricted filesystem paths

`process-footage.yml`'s `clip_path` input and `decode_video.resolve_source()`
only check `os.path.isfile()` — there's no confinement to `drop/` or the repo
root. Anyone able to trigger the workflow (already requires Actions-write on
the repo) can point ffmpeg at arbitrary files on the runner. Low impact since
it requires the same privilege as editing the workflow directly, but
consider confining `clip_path` to a subtree (e.g. reject paths outside
`drop/` after `os.path.realpath` normalization) for defense in depth.

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
