#!/usr/bin/env python3
"""Volleyball analysis — web UI shell (FastAPI + plain HTML/JS).

Upload a clip, the existing pipeline (poc/pipeline.py) processes it in the
background (players via local YOLOv8, ball via Roboflow), and the results page
plays the annotated video with the full ball trajectory drawn as a path.

Run:
    pip install -r requirements.txt -r webapp/requirements.txt
    export ROBOFLOW_API_KEY=your_key      # optional; without it, players-only
    uvicorn webapp.app:app --reload
    # open http://127.0.0.1:8000

Processing is offline/slow (the Roboflow call is per-frame over the network),
so uploads run as background jobs you poll for status.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE = REPO_ROOT / "poc" / "pipeline.py"
JOBS_DIR = Path(__file__).resolve().parent / "jobs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
JOBS_DIR.mkdir(exist_ok=True)

RF_MODEL = os.environ.get("RF_MODEL", "volleyball_detection/2")
DEFAULT_STRIDE = int(os.environ.get("PIPELINE_STRIDE", "5"))

app = FastAPI(title="Volleyball Analysis")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "status.json"


def _read_status(job_id: str) -> dict | None:
    p = _status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _write_status(job_id: str, **fields) -> None:
    p = _status_path(job_id)
    current = _read_status(job_id) or {}
    current.update(fields)
    p.write_text(json.dumps(current, indent=2))


def _process_job(job_id: str, clip: Path, stride: int) -> None:
    """Run the pipeline as a subprocess; record status as it goes."""
    job_dir = clip.parent
    annotated = job_dir / "annotated.mp4"
    events = job_dir / "events.json"
    api_key = os.environ.get("ROBOFLOW_API_KEY")

    cmd = [sys.executable, str(PIPELINE), "-i", str(clip),
           "-o", str(annotated), "--json", str(events), "--stride", str(stride)]
    if api_key:
        cmd += ["--rf-model", RF_MODEL]
    else:
        cmd += ["--no-ball"]

    _write_status(job_id, status="processing", message="Running pipeline…",
                  ball_enabled=bool(api_key), started=_now())
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                              text=True, env={**os.environ})
    except Exception as e:  # noqa: BLE001
        _write_status(job_id, status="error", message=f"Failed to launch: {e}")
        return

    if proc.returncode != 0 or not events.exists():
        tail = (proc.stderr or proc.stdout or "")[-800:]
        _write_status(job_id, status="error",
                      message=f"Pipeline exited {proc.returncode}.\n{tail}")
        return

    # Summarize for the UI.
    try:
        data = json.loads(events.read_text())
        frames = data.get("frames_processed", 0)
        with_ball = data.get("frames_with_ball", 0)
        pct = round(100.0 * with_ball / frames, 1) if frames else 0.0
    except (json.JSONDecodeError, OSError):
        frames, with_ball, pct = 0, 0, 0.0

    _write_status(job_id, status="done", message="Complete.", finished=_now(),
                  frames=frames, frames_with_ball=with_ball, ball_pct=pct,
                  has_video=annotated.exists())


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), stride: int = Form(DEFAULT_STRIDE)):
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    clip = job_dir / f"clip{suffix}"
    with clip.open("wb") as f:
        f.write(await file.read())

    stride = max(1, min(int(stride), 30))
    _write_status(job_id, id=job_id, filename=file.filename, status="queued",
                  stride=stride, created=_now())

    threading.Thread(target=_process_job, args=(job_id, clip, stride),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs")
async def list_jobs():
    jobs = []
    for d in sorted(JOBS_DIR.iterdir(), reverse=True) if JOBS_DIR.exists() else []:
        if d.is_dir():
            st = _read_status(d.name)
            if st:
                jobs.append(st)
    return jobs


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    st = _read_status(job_id)
    if st is None:
        raise HTTPException(404, "Unknown job.")
    return st


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    path = JOBS_DIR / job_id / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "No video for this job.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    path = JOBS_DIR / job_id / "events.json"
    if not path.exists():
        raise HTTPException(404, "No events for this job.")
    return FileResponse(path, media_type="application/json")


# Static frontend (mounted last so /api/* wins).
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
