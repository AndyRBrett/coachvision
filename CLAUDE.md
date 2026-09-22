# CLAUDE.md

Project-level guidance for Claude Code sessions in this repo.

## Codex PR review

OpenAI Codex auto-reviews PRs in this repo. It triggers when a PR
is opened for review, when a draft is marked ready, or on a
`@codex review` comment. Findings come back as comments from
chatgpt-codex-connector[bot]; a clean pass is just a 👍 reaction.

### PR workflow (one Codex round, then merge)

1. Open the PR, subscribe to it, and stop. Do not merge yet.
2. Wait for the FIRST Codex review (comments from
   chatgpt-codex-connector[bot], or a 👍 reaction = clean pass).
3. Triage each finding against the PR's original goal:
   - In scope + real bug → fix it.
   - Out of scope, style-only, or speculative → do NOT fix.
     List it in a PR comment as "Deferred" (or open an issue).
4. Push the fixes in one commit. Do not tag @codex again.
5. Once CI passes, merge to main and unsubscribe.

Ignore any Codex reviews or comments that arrive after step 2.
One review round per PR, no exceptions. If a later comment looks
like a genuine bug, mention it to me instead of acting on it.

In remote/web sessions there is no `gh` CLI — use the GitHub MCP tools
(`pull_request_read`, `add_issue_comment`) for the same steps.

## Gotchas

- **The clip quality gate does not measure duration, and that is deliberate.**
  `quality_gate.check_clip_quality` answers `corrupt_file` when ffprobe cannot
  read a duration, rather than inferring the length some other way. Four review
  rounds on PR #44 tried to infer it and produced six real bugs, three of them
  the same fact wearing different clothes: `r_frame_rate` is the stream's
  NOMINAL base rate, so counting frames against it falsely rejects
  variable-frame-rate phone footage, and every replacement found a new way to be
  wrong about the same thing — packet timestamps fixed VFR but reopened the hole
  for raw H.264 streams carrying no PTS, and keeping both still measured only
  between frame START times and rejected a VFR clip whose final frame is held.
  How long a clip is, when the container will not say, is not cheaply knowable
  from metadata.

  The cost is real and stated: some fragmented and raw streams are refused and
  need a remux. It lands in `results/quality_gate.json` where it can be seen.
  **If this comes back as a filed enhancement — "measure the length instead of
  rejecting it" is exactly the shape the weekly review would propose — read the
  block at the top of `quality_gate.py` and the four rounds on #44 before
  starting.** The fix, if it ever bites for real, is a remux on ingest.
