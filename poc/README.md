# PoC: ball + player detection

Step 1 of the pipeline. Goal: prove we can find the ball and track players on a
clip and emit structured data — *before* building any system around it.

It uses the stock YOLOv8 COCO model (`yolov8n.pt`), which is pip-installable
with no broken weight links and runs on CPU. COCO gives us `person` (players)
and `sports ball` (the volleyball, approximately).

## Run it (macOS, 2017 Intel MacBook)

```bash
cd volleyball
python3 -m venv .venv && source .venv/bin/activate

# CPU build of torch — important on an Intel Mac (skips the huge CUDA build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Annotated video + JSON next to the input
python3 poc/detect_ball.py -i path/to/your_clip.mp4
```

First run downloads `yolov8n.pt` (~6 MB) automatically.

### Useful flags on a slow machine

```bash
# Quick smoke test: first 60 processed frames, every 2nd frame, smaller inference
python3 poc/detect_ball.py -i clip.mp4 --max-frames 60 --stride 2 --imgsz 480

# JSON only (skip writing the video — much faster)
python3 poc/detect_ball.py -i clip.mp4 --no-video
```

| Flag | What it does |
|------|--------------|
| `--stride N` | Process every Nth frame (N=2 → ~half the work) |
| `--imgsz 416` | Smaller inference size = faster, less accurate |
| `--max-frames N` | Stop after N processed frames (smoke testing) |
| `--no-video` | Skip annotated video, write JSON only |
| `--model PATH` | Swap in better (volleyball-trained) weights |
| `--conf 0.2` | Lower threshold = more (and more false) detections |

## What to look at

1. **The annotated video** — are players boxed? Does the ball marker track the
   real ball, or jump around / disappear on fast spikes and serves?
2. **The `_events.json`** — the structured per-frame data (ball center, player
   boxes + track IDs, timestamps) that the metrics layer and the Claude coaching
   call will consume later.
3. **The printed ball-detection rate** — the headline number for the go/no-go
   decision.

## Expected result, and the honest caveat

Player detection and tracking are solid out of the box. **Ball detection is the
real unknown.** The stock COCO `sports ball` class was not trained on volleyball
footage, so on your camera angle and lighting it will miss fast or occluded
balls. A low ball-detection rate here is the known limitation, not a bug.

If ball detection is too weak on your footage, the fix is **better weights, not
more code** — swap `--model` for volleyball-trained weights (e.g. VolleyVision's)
or train a small custom detector. The rest of this script is unchanged.

## Output JSON shape

```jsonc
{
  "source": "clip.mp4", "model": "yolov8n.pt", "fps": 30.0,
  "frame_size": [1920, 1080], "stride": 1,
  "frames_processed": 300, "frames_with_ball": 142,
  "events": [
    {
      "frame": 0, "time_s": 0.0,
      "ball": { "center": [960.0, 540.0], "bbox": [950,530,970,550], "conf": 0.61 },
      "players": [
        { "track_id": 1, "bbox": [100,200,180,420], "conf": 0.88 }
      ]
    }
  ]
}
```
