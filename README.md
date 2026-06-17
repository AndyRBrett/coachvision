# volleyball

Video recognition and analysis for volleyball footage: detect the ball, track
players, compute metrics, and turn the results into coaching feedback.

This is an **offline batch tool**, not a real-time system — you record a clip,
the pipeline processes it once (slow on an old laptop, but you walk away), and
the UI plays back the *already-processed* result. That framing is deliberate: it
keeps the project runnable on modest hardware (e.g. a 2017 Intel MacBook, CPU
only).

## Architecture (target)

```
1. CAPTURE     phone records the court -> upload a video
2. PIPELINE    YOLOv8 ball + player detection, ByteTrack tracking,
               court homography -> structured JSON per clip
3. METRICS     rally length, court zones/heatmaps, ball speed, contact points
4. COACHING    feed structured stats to Claude (claude-opus-4-8) -> analysis
5. WEB UI      upload, annotated playback, coaching panel, session history
```

Build order follows risk: **validate ball detection first** (`poc/`), then the
metrics layer, then the coaching call, then the UI.

## Status

- [x] **PoC: ball + player detection** (`poc/`) — pipeline runs end-to-end,
      player detection + tracking validated, structured JSON + annotated video
      out. Ball-accuracy on real volleyball footage is the open question (see
      below).
- [ ] Court calibration (homography from 4 corners)
- [ ] Metrics layer
- [ ] Claude coaching call (`claude-opus-4-8`)
- [ ] FastAPI backend + web UI

## Quick start

See [`poc/README.md`](poc/README.md). Short version:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # Intel Mac: CPU build
pip install -r requirements.txt
python3 poc/detect_ball.py -i your_clip.mp4
```

## The one honest caveat

Claude Code can build the *system* — pipeline glue, metrics, backend, UI — and
that part is reliable. What it **cannot** guarantee is that the ball detector
performs well on *your* specific camera angle and lighting; that depends on the
underlying model's quality on your footage and is a tuning problem, not a coding
problem. The PoC starts with the stock YOLOv8 COCO `sports ball` class (robust,
no broken downloads) and is structured so better volleyball-trained weights drop
in via a single `--model` flag.

## Notes on models

Coaching feedback (step 4) uses Anthropic's current top Opus model,
**`claude-opus-4-8`** (1M-token context at standard pricing — enough to feed a
whole match's structured events in one call). Detection/tracking uses YOLOv8 via
`ultralytics`, fully local.
