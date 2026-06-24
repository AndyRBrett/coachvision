# coachvision

A lightweight computer-vision coaching pipeline with health monitoring for the
**Project Overseer** (a weekly automated reviewer that reads
`overseer-status.json` to tell whether the pipeline is *healthy-but-idle* or
*broken*).

It ships with two interchangeable **sport domains** — **volleyball** and
**martial arts** — so you can point it at whatever footage you're actually
recording (see [Switching sports](#switching-sports-volleyball--martial-arts)).

## Components

| File | Purpose |
| --- | --- |
| `domains.py` | Sport domains (volleyball / martial arts): detector choice, tag vocabulary, report wording, and segmentation defaults. |
| `decode_video.py` | ffmpeg bridge: real video (`MP4`/`MOV`/URL) → the gzipped P5/PGM frames the detector reads. CPU-only. |
| `process_footage.py` | One-shot: decode → pipeline → publish `reports/<clip>/` + `reports/index.json`. Driven by the `process-footage` workflow. |
| `detect.py` | CV front-end: turns raw clip frames into subject-track tracking data (ball or fighter). |
| `pipeline.py` | Runs the full pipeline (detect → highlights → coaching) and the self-test. |
| `coaching.py` | Per-clip coaching report: segment length, subject speed, action-zone heatmap (worded per domain — rally/ball/contact for volleyball, exchange/fighter/strike for martial arts). |
| `highlights.py` | Segments tracking data into plays and emits tagged highlight clips. |
| `cosmos_tagger.py` | Optional clip tag enrichment via NVIDIA **Cosmos Reason**. |
| `ingest_watch.py` | Watches a drop folder, auto-detects new footage, and enqueues unseen clips. |
| `write_status.py` | Publishes `overseer-status.json` (heartbeat + ingest signals + idle nudge + self-test verification). |

## Switching sports (volleyball ⇄ martial arts)

The pipeline is mechanically sport-agnostic: it tracks one **subject point** per
frame, segments play from gaps in that point's motion, tags each segment from
timed events, and bins events onto a **surface** grid. A `Domain` (`domains.py`)
bundles the only things that actually differ between sports:

| | volleyball | martial arts |
| --- | --- | --- |
| detector | brightest blob (the ball) | **motion energy** (the moving fighter) |
| play segment | rally | exchange |
| subject / surface | ball / court | fighter / mat |
| action tags | serve, set, attack, block, dig… | jab, cross, hook, kick, knee, takedown, clinch… |

Why a different detector? Martial arts has no high-contrast ball to track, so the
subject is recovered with **motion-energy temporal segmentation** — thresholding
the frame-to-frame pixel difference and taking the centroid of what changed. A
fighter standing still produces no motion and reads as the gap between exchanges,
the direct analogue of a volleyball going out of play. (Standard dependency-free
approach; see e.g. [energy-guided temporal segmentation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7506802/).)

Select the domain with the `--domain` flag or the `COACHVISION_DOMAIN` env var
(default `volleyball`, preserving prior behaviour and the overseer's status
contract):

```bash
# Process martial-arts footage end-to-end
python pipeline.py clip.pgm.gz --events clip.events.json --domain martial_arts

# Or set it once for the whole session (detect.py / highlights.py / coaching.py
# all read it)
export COACHVISION_DOMAIN=martial_arts
python pipeline.py --self-test          # runs the martial-arts reference clip
```

For the automated weekly workflow, set the repository variable
`COACHVISION_DOMAIN` (Settings → Secrets and variables → Actions → Variables) to
`martial_arts` to switch detection, the self-test, and the published status
without touching code. Each domain bundles its own reference clip
(`fixtures/`), so the self-test proves *its* detector end-to-end.

The pipeline's machine-readable keys (`segment_count`, `frames_processed`, …) stay
stable across domains so the Project Overseer keeps working; only the words in
the human-facing coaching summary change (exchanges/strikes/mat vs.
rallies/contacts/court).

## Run it on GitHub — no Mac, no GPU

The CV here is pure Python (no `torch`/`opencv`/CUDA), so the whole thing runs on
GitHub's free Linux runners. The only piece that touches real video is decoding,
and that's plain **ffmpeg** (CPU, preinstalled on `ubuntu-latest`) — no GPU
anywhere. Two new pieces make this a one-button job:

| File | Purpose |
| --- | --- |
| `decode_video.py` | ffmpeg → gzipped P5/PGM frames (`MP4`/`MOV`/… → what `detect.py` reads). Samples to a low fps/width since the detector is O(pixels). |
| `process_footage.py` | Decode → run the pipeline for a domain → publish `reports/<clip>/` and upsert `reports/index.json`. |

**Trigger it** from the **Actions tab → `process-footage` → "Run workflow"** (the
GitHub mobile app and the REST API can fire the same `workflow_dispatch`):

- `domain` — `martial_arts` (default) or `volleyball`.
- `clip_path` — a video already committed under `drop/` (e.g. `drop/spar.mp4`), **or**
- `clip_url` — a link the runner downloads (handy for a phone upload / cloud link).
- `fps` — sample rate (default `10`).

It commits browsable outputs back to the repo:

```
reports/
  index.json                     # catalog of every processed clip (PWA-friendly)
  <clip>/coaching/report.json    # machine-readable coaching report
  <clip>/coaching/summary.txt    # the coach-facing summary
  <clip>/highlights/manifest.json
```

`reports/index.json` is a flat list of clips with relative paths to each
artifact — exactly what a **static phone PWA** (served from GitHub Pages or the
raw repo) can fetch to render a session gallery, with no server to run. The same
`workflow_dispatch` is the PWA's "analyze this clip" button.

> **Detection quality, honestly.** v1 finds *where the action is*: motion-energy
> segments a fixed-camera recording into exchanges and reports their length and
> fighter speed. It does **not** yet classify individual strikes (jab vs. kick) —
> segments come back `untagged` unless you supply an events sidecar or enable the
> optional [Cosmos Reason](#optional-nvidia-cosmos-reason-tagging) VLM tagging
> (which also runs off-box behind an HTTP endpoint, so still no local GPU). Works
> best with a steady camera; a panning phone adds background motion.

## End-to-end pipeline + self-test

The pipeline runs in three stages, each a small stdlib-only module:

```
clip frames ──detect.py──▶ tracking ──highlights.py──▶ tagged rally manifest
                              └────────coaching.py────▶ coaching report (length/speed/heatmap)
```

`detect.py` is the CV front-end that was missing — the reason the pipeline had
*processed 0 frames ever*. It reads a clip stored as a gzipped Netpbm (P5/PGM)
frame sequence and recovers the ball position per frame as the centroid of the
brightest blob, emitting the same tracking schema the rest of the pipeline
consumes. No ffmpeg/opencv/GPU required, so it runs anywhere CI does.

A short (~10s) **reference clip** is bundled as a fixture so the whole pipeline
can be proven end-to-end:

```bash
python pipeline.py --self-test          # detect → highlights → coaching on the reference clip
```

The self-test fails (non-zero exit) if detection finds no ball, no rallies
segment, or the coaching report comes back empty — so a broken pipeline **fails
the build** (`.github/workflows/tests.yml`) instead of silently sitting at zero
frames. It also writes `results/selftest.json`, which `write_status.py` surfaces
as `pipeline_selftest` in the overseer status, distinguishing *healthy-but-idle*
(pipeline verified, just no new footage) from *broken*.

The fixtures live in `fixtures/` and are regenerated with
`python fixtures/make_reference_clip.py` (volleyball) and
`python fixtures/make_martialarts_clip.py` (martial arts).

## Auto coaching reports per clip

`coaching.py` turns a clip's tracking data into a coach-facing summary — the
project's core user-facing value:

```bash
python pipeline.py fixtures/reference_clip.pgm.gz --events fixtures/reference_clip.events.json \
    --meters-per-pixel 0.1125         # full run: manifest + coaching report + metrics
python coaching.py tracking.json      # coaching report only, from existing tracking JSON
```

Per rally and per clip it reports:

- **rally length** — duration in seconds and the number of tracked contacts.
- **ball speed** — average and peak from frame-to-frame motion (px/s, plus m/s
  when a `--meters-per-pixel` court calibration is given).
- **contact-zone heatmap** — a coarse court grid counting where contacts
  happened, rendered as ASCII in the human-readable summary:

```
  rally_001: 1.7s, 3 contacts [serve, set, attack], peak 56.57 px/s (6.36 m/s)
  ...
  contact-zone heatmap (court top-left origin):
    |  @   |
    | = =@ |
    |@=    |
```

Output is written to `coaching/report.json` (machine-readable) and
`coaching/summary.txt` (the coach-facing artifact).

## Idle-footage nudge + drop-folder auto-detect (issue #8)

The pipeline can "work but go unused" — a capable CV system that nothing is
feeding. `ingest_watch.py` closes that gap:

```bash
# Watch a folder (local dir or synced cloud bucket) for new footage
COACHVISION_DROP_DIR=drop python ingest_watch.py
```

- Recursively scans `COACHVISION_DROP_DIR` (default `drop/`) for video files.
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
older than `COACHVISION_IDLE_THRESHOLD_DAYS` (default 14), or when clips are
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
export COACHVISION_COSMOS_NIM_URL="https://integrate.api.nvidia.com/v1/chat/completions"
export COACHVISION_COSMOS_API_KEY="nvapi-..."           # for hosted NIM
export COACHVISION_COSMOS_MODEL="nvidia/cosmos-reason-3" # optional override
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
