# volleyball

A lightweight volleyball computer-vision pipeline with health monitoring for the
**Project Overseer** (a weekly automated reviewer that reads
`overseer-status.json` to tell whether the pipeline is *healthy-but-idle* or
*broken*).

## Components

| File | Purpose |
| --- | --- |
| `write_status.py` | Publishes `overseer-status.json` on every run (heartbeat + ingest signals + idle nudge). |
| `ingest_watch.py` | Watches a drop folder, auto-detects new footage, and enqueues unseen clips. |
| `highlights.py` | Segments tracking data into rallies and emits tagged highlight clips. |
| `cosmos_tagger.py` | Optional clip tag enrichment via NVIDIA **Cosmos Reason**. |

## Idle-footage nudge + drop-folder auto-detect (issue #8)

The pipeline can "work but go unused" — a capable CV system that nothing is
feeding. `ingest_watch.py` closes that gap:

```bash
# Watch a folder (local dir or synced cloud bucket) for new footage
VOLLEYBALL_DROP_DIR=drop python ingest_watch.py
```

- Recursively scans `VOLLEYBALL_DROP_DIR` (default `drop/`) for video files.
- Diffs against a small seen-state manifest (`ingest_state.json`) so each clip is
  enqueued **once**, and only after its size settles across two scans (so a clip
  still being copied isn't processed half-written).
- Writes pending clips to `ingest_queue.json`.

`write_status.py` then surfaces an **idle nudge** so prolonged idleness is
visible instead of silently passing as healthy:

```json
{
  "days_since_last_footage": null,
  "idle_threshold_days": 14,
  "pending_footage": 0,
  "needs_footage": true,
  "nudge": "No footage has ever been ingested -- drop clips in the watched folder to start."
}
```

The nudge fires when footage has never been ingested, when the last ingest is
older than `VOLLEYBALL_IDLE_THRESHOLD_DAYS` (default 14), or when clips are
queued but unprocessed (a stalled pipeline). The weekly workflow runs the scan
before writing status.

## Auto highlight clips with coaching tags (issue #9)

Turns ball+player tracking data into rewatchable, tagged clips — the project's
stated purpose (coaching feedback).

```bash
# Build a manifest (no rendering needed)
python highlights.py examples/sample_tracking.json --output-dir highlights

# Preview the ffmpeg commands, or render the clips (requires ffmpeg)
python highlights.py examples/sample_tracking.json --dry-run
python highlights.py examples/sample_tracking.json --render
```

Three stages:

1. **Rally segmentation** — splits continuous play into rallies from ball-motion
   gaps (ball missing/still longer than `max_gap_s`).
2. **Coaching tags** — attaches `serve`/`reception`/`set`/`attack`/`block`/`dig`/
   … tags whose event timestamps fall inside each rally.
3. **Manifest** — writes `highlights/manifest.json` (the dashboard artifact),
   one entry per rally with an `ffmpeg` trim+overlay command.

ffmpeg rendering is **optional and guarded**: the command is always recorded in
the manifest but only executed with `--render` when ffmpeg is installed, so the
core logic stays pure and testable. See `examples/sample_tracking.json` for the
expected tracking-input schema.

## Optional: NVIDIA Cosmos Reason tagging

In June 2026 NVIDIA open-sourced the **Cosmos 3** family, including **Cosmos
Reason** — a reasoning vision-language model for multimodal video understanding,
served as an NVIDIA NIM microservice. `cosmos_tagger.py` can use it to enrich the
heuristic event-window tags with model-derived coaching tags:

```bash
export VOLLEYBALL_COSMOS_NIM_URL="https://integrate.api.nvidia.com/v1/chat/completions"
export VOLLEYBALL_COSMOS_API_KEY="nvapi-..."           # for hosted NIM
export VOLLEYBALL_COSMOS_MODEL="nvidia/cosmos-reason-3" # optional override
python highlights.py examples/sample_tracking.json --cosmos
```

It is strictly optional and best-effort: with no endpoint configured (or on any
request failure) the pipeline falls back to heuristic tags and never crashes.
Inference runs behind the NIM HTTP endpoint, so no GPU or NVIDIA SDK is required
on the runner — the module is stdlib-only (`urllib`).

## Tests

```bash
python -m unittest discover -s tests -v
```

CI runs the suite on every push/PR (`.github/workflows/tests.yml`); the weekly
`overseer-status` workflow scans for footage and publishes status.
