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
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request

import detect

DEFAULT_FPS = 10.0
DEFAULT_WIDTH = 160   # downsample width in px; height keeps aspect (scale=W:-2)

# Present a normal browser User-Agent: many hosts (Google Storage/Drive, CDNs)
# answer urllib's default agent with 403 Forbidden.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


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


def direct_download_url(url):
    """Best-effort rewrite of a share link to a direct-download URL.

    Google Drive ``.../file/d/<id>/view`` viewer links don't serve bytes; rewrite
    them to the ``uc?export=download`` form. Only helps files shared "Anyone with
    the link" -- a private file still returns 401/403. Other URLs pass through.
    """
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url


def _download(url, dest):
    """Fetch ``url`` to ``dest`` with a browser User-Agent, following redirects.

    Raises a clear RuntimeError on HTTP errors so the workflow log says *why*
    (e.g. a private/Drive-view link) instead of a bare urllib traceback.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
            shutil.copyfileobj(resp, fh)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"could not download {url!r}: HTTP {exc.code} {exc.reason}. "
            "Use a public, direct-download link (not a Google Drive 'view' page "
            "or a private file), or commit the clip under drop/ and pass a path."
        ) from exc
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
        return _download(direct_download_url(clip_url), dest)
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
