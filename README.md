# volleyball

A lightweight volleyball computer-vision pipeline with health monitoring for the
**Project Overseer** (a weekly automated reviewer that reads
`overseer-status.json` to tell whether the pipeline is *healthy-but-idle* or
*broken*).

## Components

| File | Purpose |
| --- | --- |
| `detect.py` | CV front-end: turns raw clip frames into ball-track tracking data. |
| `pipeline.py` | Runs the full pipeline (detect → highlights → coaching) and the self-test. |
| `coaching.py` | Per-clip coaching report: rally length, ball speed, contact-zone heatmap. |
| `highlights.py` | Segments tracking data into rallies and emits tagged highlight clips. |
| `cosmos_tagger.py` | Optional clip tag enrichment via NVIDIA **Cosmos Reason**. |
| `ingest_watch.py` | Watches a drop folder, auto-detects new footage, and enqueues unseen clips. |
| `write_status.py` | Publishes `overseer-status.json` (heartbeat + ingest signals + idle nudge + self-test verification). |

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
`python fixtures/make_reference_clip.py`.

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

## Optional: real-footage detection via supervision (Ultralytics YOLO + ByteTrack)

The default `detect.py` finds the subject as the *brightest blob* — enough to
prove the pipeline on the synthetic reference clip, but not robust on real gym
footage (jerseys, lights, and reflections are all brighter than a ball).
`detect_supervision.py` is the upgrade path: a real object detector
([Ultralytics YOLO](https://github.com/ultralytics/ultralytics)) whose boxes are
normalised by [`roboflow/supervision`](https://github.com/roboflow/supervision)
into `sv.Detections`, given a stable identity by supervision's **ByteTrack**, and
emitted in the **exact same tracking schema** the rest of the pipeline consumes.

```bash
pip install supervision ultralytics            # heavy deps, opt-in only
VOLLEYBALL_DETECTOR=supervision python pipeline.py drop/match.mp4 --events match.events.json
# or explicitly:
python pipeline.py drop/match.mp4 --detector supervision
```

Like the Cosmos tagger, it is **strictly optional and guarded**: the heavy stack
(`supervision`/`ultralytics`/`torch`/`opencv`) is imported lazily and only when
the backend is selected. The stdlib geometric detector stays the default, the
self-test always runs on it, and CI remains dependency-free. Selecting the
backend without it installed fails fast with a clear install hint.

### Sport-agnostic (volleyball → basketball, golf, karate, …)

Nothing in the backend is volleyball-specific — it tracks a single configurable
**target class**, so the same pipeline extends to other sports by changing the
weights and the class name:

| Sport | `VOLLEYBALL_YOLO_MODEL` | `VOLLEYBALL_TARGET_CLASS` |
| --- | --- | --- |
| volleyball / basketball / golf | tuned weights, or stock COCO | `sports ball` (or `volleyball`/`basketball`/`golf ball` for tuned weights) |
| karate / gymnastics | tuned weights, or stock COCO | `person` (track the athlete) |

```bash
export VOLLEYBALL_DETECTOR=supervision
export VOLLEYBALL_YOLO_MODEL=yolov8n.pt          # or sport-tuned weights
export VOLLEYBALL_TARGET_CLASS="person"          # e.g. for karate
export VOLLEYBALL_YOLO_CONF=0.25                 # min detection confidence
python pipeline.py drop/kata.mp4 --detector supervision
```

The tracking schema still calls the per-frame position `ball` for backward
compatibility — read it generically as "the tracked subject's position".
(`VOLLEYBALL_BALL_CLASS` is still accepted as a legacy alias for
`VOLLEYBALL_TARGET_CLASS`.)

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
