# Web UI shell

Upload a clip → the pipeline processes it in the background → play back the
annotated video with the ball's **full trajectory drawn as a path**.

FastAPI backend + plain HTML/JS frontend (no Node/npm build step).

## Run it

From the repo root, with the venv active:

```bash
pip install -r requirements.txt -r webapp/requirements.txt

# Optional: enable ball detection. Without a key it runs players-only.
export ROBOFLOW_API_KEY=your_roboflow_key

uvicorn webapp.app:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

1. Pick a clip, choose how many frames to skip (higher = faster / fewer API calls), upload.
2. It appears under **Jobs** as `processing`; when it flips to `done`, click it.
3. The result plays the annotated video with the ball path overlaid — toggle the
   line with **Show ball path**. A red ring marks the ball's current position as
   the video plays.

## How it works

- `POST /api/upload` saves the clip to `webapp/jobs/<id>/` and launches
  `poc/pipeline.py` as a background subprocess.
- The job's `status.json` tracks `queued → processing → done|error`; the page polls it.
- `GET /api/jobs/<id>/video` and `/events` serve the annotated MP4 and the
  per-frame JSON. The frontend draws the ball-path overlay from that JSON,
  scaling the stored pixel coordinates to the displayed video size.

## Notes

- Processing is offline and slow (the ball detector calls the hosted model once
  per processed frame). A few minutes per clip is normal; that's why uploads are
  background jobs.
- `webapp/jobs/` holds uploads + outputs and is gitignored.
- Config via env: `ROBOFLOW_API_KEY`, `RF_MODEL` (default `volleyball_detection/2`),
  `PIPELINE_STRIDE` (default `5`).

## Next

Metrics (rally segmentation, ball speed, heatmaps) and the Claude coaching
readout will surface here as additional panels on the result page, computed from
the same events JSON.
