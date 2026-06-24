#!/usr/bin/env python3
"""Decode a real video into the frame sequence the detector consumes.

The CV front-end (detect.py) reads a clip as a gzipped sequence of Netpbm P5/PGM
frames, not an MP4 -- so that the detector itself stays pure-Python and GPU-free.
This module is the bridge: it uses **ffmpeg** (CPU-only, preinstalled on GitHub's
Ubuntu runners) to turn an ordinary recording -- the phone/camera footage you
actually capture -- into that frame sequence.

ffmpeg is asked to emit grayscale P5 PGM frames straight to stdout
(``-f image2pipe -vcodec pgm``); concatenated, that stream is exactly what
detect.load_pgm_frames already parses. We downsample (fps + width) because the
detector cost is O(pixels) in pure Python, and a coaching pass doesn't need full
resolution to find where the action is.

No GPU, no torch, no opencv -- just ffmpeg's software decoder and the standard
library. The actual ffmpeg invocation is isolated in ``_run_ffmpeg`` so the rest
is unit-testable without ffmpeg installed.
"""
import gzip
import os
import shutil
import subprocess
import tempfile
import urllib.request

import detect

DEFAULT_FPS = 10.0
DEFAULT_WIDTH = 160   # downsample width in px; height keeps aspect (scale=W:-2)


def ffmpeg_decode_cmd(src, fps=DEFAULT_FPS, width=DEFAULT_WIDTH):
    """Build the ffmpeg argv that streams ``src`` as concatenated P5 PGM frames.

    Returned as a list (never shell-joined) so it is safe for subprocess. The
    filter chain samples to ``fps``, scales to ``width`` (even height, aspect
    preserved), and forces grayscale so each frame is a single-byte raster.
    """
    vf = f"fps={fps},scale={int(width)}:-2,format=gray"
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", src,
        "-vf", vf,
        "-f", "image2pipe", "-vcodec", "pgm", "pipe:1",
    ]


def _run_ffmpeg(cmd):
    """Run ffmpeg and return its stdout bytes. Isolated so tests can patch it."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install it (GitHub's ubuntu runners have it "
            "preinstalled) to decode real video."
        )
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    return proc.stdout


def _download(url, dest):
    """Fetch ``url`` to ``dest`` (so a phone/PWA can pass a link, not a commit)."""
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    return dest


def decode_to_pgm_gz(src, out_path, fps=DEFAULT_FPS, width=DEFAULT_WIDTH):
    """Decode video ``src`` to a gzipped P5/PGM frame sequence at ``out_path``.

    Returns a summary dict (``out_path``, ``frame_count``, ``width``,
    ``height``, ``fps``). Raises ValueError if ffmpeg produced no frames (an
    unreadable/empty video) so a broken input fails loudly instead of yielding
    an empty clip the pipeline would silently report as zero segments.
    """
    pgm_bytes = _run_ffmpeg(ffmpeg_decode_cmd(src, fps=fps, width=width))
    if not pgm_bytes:
        raise ValueError(f"ffmpeg produced no frames for {src!r}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with gzip.open(out_path, "wb") as fh:
        fh.write(pgm_bytes)

    # Reuse the detector's own parser so we report exactly what it will read.
    w, h, frames = detect.load_pgm_frames(out_path)
    if not frames:
        raise ValueError(f"decoded clip {out_path!r} contains no frames")
    return {
        "out_path": out_path,
        "frame_count": len(frames),
        "width": w,
        "height": h,
        "fps": fps,
    }


def resolve_source(clip_path=None, clip_url=None, work_dir=None):
    """Return a local video path from either a repo path or a URL to download.

    Exactly one of ``clip_path`` / ``clip_url`` should be given. Downloads land
    in ``work_dir`` (a temp dir when omitted). Raises ValueError if neither is
    provided or a given local path is missing.
    """
    if clip_url:
        work_dir = work_dir or tempfile.mkdtemp(prefix="coachvision_")
        dest = os.path.join(work_dir, "input_video")
        return _download(clip_url, dest)
    if clip_path:
        if not os.path.isfile(clip_path):
            raise ValueError(f"clip not found: {clip_path}")
        return clip_path
    raise ValueError("provide a clip path or a clip URL")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Decode a video into a gzipped P5/PGM frame clip.")
    parser.add_argument("src", nargs="?", help="Path to a local video file")
    parser.add_argument("--url", help="Download the video from this URL instead")
    parser.add_argument("--out", required=True, help="Output .pgm.gz path")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    src = resolve_source(clip_path=args.src, clip_url=args.url)
    summary = decode_to_pgm_gz(src, args.out, fps=args.fps, width=args.width)
    print(f"Decoded {src} -> {args.out}: "
          f"{summary['frame_count']} frames @ {summary['width']}x{summary['height']}")


if __name__ == "__main__":
    main()
