# TODO / parking lot

Deferred ideas to revisit. Active build order lives in the README "Status"
section; this file is for things we've consciously postponed.

## Cut the Roboflow API dependency (run ball detection locally)

**Status:** deferred (chose to keep the hosted API while building metrics/coaching).

Today, ball detection calls the Roboflow-hosted model **once per processed
frame**. That's fine for the handful of clips we process while building features,
but for routine use it means: monthly free-tier quota, internet required,
network-bound speed (~3 fps), and frames leaving the machine.

**Goal:** run the ball model locally (a volleyball-trained YOLOv8 `.pt`), exactly
like the local `yolov8n.pt` player model — fully offline, no API calls, faster,
private.

**Work:**
- Add a `--ball-weights path/to/model.pt` option to `poc/pipeline.py` that runs
  the ball model via local `ultralytics` instead of Roboflow (use the model's own
  `ball`/`volleyball` class index). Keep `--rf-model` as the fallback.
- Obtain a local `.pt`, whichever is easier:
  - download a YOLOv8 weights file if a Roboflow project / HF repo offers one, or
  - export the public `volleyball_detection` dataset from Roboflow and train
    YOLOv8-nano in a free Colab GPU (~30 min, one-time) → `best.pt`.
- Once local, the whole pipeline runs offline; the webapp no longer needs
  `ROBOFLOW_API_KEY`.

## Housekeeping

- **Regenerate the Roboflow API key** that was shared during setup (it was pasted
  into a chat). Roboflow → Settings → API Keys → regenerate.
- Player ID stability looked good on test footage; if ID swaps show up on busier
  clips, revisit ByteTrack params or a stronger player model.
