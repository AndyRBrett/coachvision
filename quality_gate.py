#!/usr/bin/env python3
"""Pre-processing clip quality gate, run before the CV pipeline (Overseer #43).

Auto-ingest connectors (drop-folder watch, mobile quick-upload) are more likely
to receive inconsistent-quality footage than a manual drop-folder use: a
too-short recording, a lens-cap-on dark clip, a thumbnail-resolution video, or a
half-written/corrupt file. Feeding those straight into detection/pose tracking
burns real compute (ffmpeg decode + per-frame CV) to produce degraded or empty
output that looks like a silent pipeline failure rather than a bad input.

This module runs a cheap check first: ffprobe metadata (resolution, duration,
frame rate) plus a tiny low-res/low-fps frame sample (mean luminance, as an
exposure/lighting proxy) -- a fraction of the cost of a real decode. It uses
ffmpeg/ffprobe rather than OpenCV so it stays on the same dependency (already
required by decode_video.py) instead of adding a new one to the base pipeline.
"""
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone

import decode_video
import detect

DEFAULT_STATUS_PATH = "results/quality_gate.json"

MIN_DURATION_S = 1.0
MIN_WIDTH = 64
MIN_HEIGHT = 64
# Frame sample used for the exposure check: tiny + low fps + first few seconds
# only, so it costs a fraction of a real decode.
SAMPLE_FPS = 1.0
SAMPLE_WIDTH = 32
SAMPLE_DURATION_S = 3.0
# Mean luminance (0-255, PGM bytes are already grayscale) below this reads as a
# lens-cap-on / dark-room clip rather than real footage.
MIN_MEAN_LUMINANCE = 16.0

REASON_TOO_SHORT = "too_short"
REASON_TOO_DARK = "too_dark"
REASON_RESOLUTION_TOO_LOW = "resolution_too_low"
REASON_CORRUPT_FILE = "corrupt_file"


def _run_ffprobe(cmd):
    """Run ffprobe and return its parsed JSON stdout. Isolated so tests can patch it."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise ValueError(f"ffprobe failed: {proc.stderr.decode('utf-8', 'replace')[-300:]}")
    try:
        return json.loads(proc.stdout)
    except ValueError:
        raise ValueError("ffprobe produced unreadable output")


def probe_metadata(src):
    """Return ``{width, height, duration, fps}`` for ``src`` via ffprobe.

    Raises ValueError if ffprobe fails or the file has no readable video stream
    -- both read as a corrupt/unusable file rather than a resolution/duration
    problem.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", src,
    ]
    data = _run_ffprobe(cmd)
    streams = data.get("streams") or []
    if not streams or "width" not in streams[0] or "height" not in streams[0]:
        raise ValueError(f"no readable video stream in {src!r}")
    stream = streams[0]

    duration = stream.get("duration") or data.get("format", {}).get("duration")
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = None

    fps = None
    rate = stream.get("r_frame_rate")
    if rate and "/" in rate:
        num, _, den = rate.partition("/")
        try:
            den_f = float(den)
            fps = float(num) / den_f if den_f else None
        except ValueError:
            fps = None

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration": duration,
        "fps": fps,
    }


def _sample_decode_cmd(src):
    """ffmpeg argv for a tiny grayscale frame sample, used for the exposure check."""
    vf = f"fps={SAMPLE_FPS},scale={SAMPLE_WIDTH}:-2,format=gray"
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-t", str(SAMPLE_DURATION_S), "-i", src,
        "-vf", vf,
        "-f", "image2pipe", "-vcodec", "pgm", "pipe:1",
    ]


def sample_mean_luminance(src):
    """Return the mean pixel value across a tiny grayscale frame sample of ``src``.

    Reuses decode_video's ffmpeg pipe (grayscale P5 PGM, so bytes are already
    luminance) at a fraction of the real decode's resolution/frame rate/length.
    Returns None if the sample produced no frames.
    """
    pgm_bytes = decode_video._run_ffmpeg(_sample_decode_cmd(src))
    if not pgm_bytes:
        return None
    with tempfile.NamedTemporaryFile(suffix=".pgm") as tmp:
        tmp.write(pgm_bytes)
        tmp.flush()
        _w, _h, frames = detect.load_pgm_frames(tmp.name)
    if not frames:
        return None
    total = sum(sum(frame) for frame in frames)
    pixel_count = sum(len(frame) for frame in frames)
    return total / pixel_count if pixel_count else None


def check_clip_quality(src):
    """Run the pre-processing quality gate on ``src``.

    Returns ``{"ok": True, "metadata": {...}}`` or ``{"ok": False, "reason":
    ..., "detail": ..., "metadata": {...}}`` where ``reason`` is one of
    too_short / too_dark / resolution_too_low / corrupt_file. Checks run
    cheapest-first so a corrupt file never reaches the frame-sampling step.
    """
    try:
        metadata = probe_metadata(src)
    except ValueError as exc:
        return {"ok": False, "reason": REASON_CORRUPT_FILE, "detail": str(exc)}

    if metadata["fps"] is None or metadata["fps"] <= 0:
        return {
            "ok": False, "reason": REASON_CORRUPT_FILE,
            "detail": f"unreadable frame rate ({metadata['fps']!r})",
            "metadata": metadata,
        }

    if metadata["width"] < MIN_WIDTH or metadata["height"] < MIN_HEIGHT:
        return {
            "ok": False, "reason": REASON_RESOLUTION_TOO_LOW,
            "detail": f"{metadata['width']}x{metadata['height']} is below the "
                      f"{MIN_WIDTH}x{MIN_HEIGHT} minimum",
            "metadata": metadata,
        }

    # An UNKNOWN duration is not a short duration, and this gate must not
    # discard footage on the strength of missing metadata. Some containers
    # simply do not carry one -- a fragmented MP4 from a phone, a stream copy --
    # and mobile quick-upload (#34) is exactly the ingest path this gate was
    # asked for. A genuinely truncated or half-written file does not reach this
    # line: ffprobe fails outright on it and probe_metadata already answers
    # corrupt_file.
    #
    # This is the same call made for luminance below, and the opposite of the
    # one made for frame rate above: an unreadable fps means the decode itself
    # cannot be planned, while an unreadable duration costs only this check.
    # Three unknowns, and each is answered on what its absence actually implies.
    if metadata["duration"] is not None and metadata["duration"] < MIN_DURATION_S:
        return {
            "ok": False, "reason": REASON_TOO_SHORT,
            "detail": f"duration {metadata['duration']}s is below the {MIN_DURATION_S}s minimum",
            "metadata": metadata,
        }

    try:
        luminance = sample_mean_luminance(src)
    except RuntimeError as exc:  # ffmpeg missing/failed on a file ffprobe accepted
        return {"ok": False, "reason": REASON_CORRUPT_FILE, "detail": str(exc), "metadata": metadata}

    if luminance is not None and luminance < MIN_MEAN_LUMINANCE:
        return {
            "ok": False, "reason": REASON_TOO_DARK,
            "detail": f"mean luminance {luminance:.1f} is below the {MIN_MEAN_LUMINANCE} minimum",
            "metadata": metadata,
        }

    return {"ok": True, "metadata": metadata}


def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_rejection(reason, detail, metadata=None, status_path=None, now_iso=None):
    """Persist one more rejected clip to ``results/quality_gate.json``.

    Keeps a running ``rejected_total`` (so a rejection is never silently lost
    between runs, only overwritten by the next one) plus the most recent
    rejection, in the same latest-snapshot style as ``results/selftest.json``.
    This is what write_status.py reads to surface rejections in
    overseer-status.json instead of a clip failing the gate leaving no trace.
    """
    status_path = status_path or DEFAULT_STATUS_PATH
    now_iso = now_iso or _utc_now_iso()
    try:
        with open(status_path) as fh:
            prior = json.load(fh)
    except (FileNotFoundError, ValueError):
        prior = {}

    last_rejection = {"rejected_at": now_iso, "reason": reason, "detail": detail}
    if metadata is not None:
        last_rejection["metadata"] = metadata

    record = {
        "updated_at": now_iso,
        "rejected_total": prior.get("rejected_total", 0) + 1,
        "last_rejection": last_rejection,
    }
    os.makedirs(os.path.dirname(status_path) or ".", exist_ok=True)
    with open(status_path, "w") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return record
