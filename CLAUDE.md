# CLAUDE.md

Project-level guidance for Claude Code sessions in this repo.

## Codex PR review

OpenAI Codex auto-reviews PRs in this repo. It triggers when a PR
is opened for review, when a draft is marked ready, or on a
`@codex review` comment. Findings come back as comments from
chatgpt-codex-connector[bot]; a clean pass is just a 👍 reaction.

### PR workflow (one Codex round, then merge)

1. Open the PR, then immediately subscribe to its activity so Codex
   review comments come back to this session. Do this automatically;
   don't ask me first. Do not merge yet.
2. Wait for the FIRST Codex review (comments from
   chatgpt-codex-connector[bot], or a 👍 reaction = clean pass).
   A clean pass sends NO event: the 👍 is a reaction on the PR body,
   and reactions never reach the subscription, so waiting on events
   alone misses it until the next check-in. So right after opening,
   schedule check-ins (`send_later`) at 2, 4 and 6 minutes, then
   hourly as a backstop. On every check-in AND every event that does
   arrive (CI finishing, a comment), read the PR's `reactions` with
   MCP `issue_read` on the PR number — the comments calls don't
   return them. Only Codex's 👍 counts, and `issue_read` gives counts,
   not authors: Codex marks a review in progress with 👀 and swaps it
   for the 👍 when it passes, so read a pass as 👀 gone and 👍 present.
   A 👍 while 👀 is still there may be someone else's — keep waiting.
   Cancel the remaining check-ins once the review lands.
3. Evaluate each finding yourself and act without asking me:
   - Valid and in scope → implement the fix.
   - Out of scope, style-only, or speculative → don't fix. List it
     in a PR comment under "Deferred" with a one-line reason.
4. Push all fixes in one commit. Do not tag @codex again.
5. Once CI passes, merge to main and unsubscribe from the PR.
   Then give me a short summary: what you fixed, what you deferred.

Ignore any Codex reviews or comments that arrive after step 2.
One review round per PR. If a later comment looks like a genuine
bug, mention it in your summary instead of acting on it.

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
