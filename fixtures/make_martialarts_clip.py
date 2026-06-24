#!/usr/bin/env python3
"""Generate the bundled martial-arts reference clip used by the pipeline self-test.

The volleyball reference clip proves the brightest-blob detector; this one proves
the *motion-energy* detector that martial arts uses (there is no ball). Like its
volleyball sibling it is a short sequence of grayscale Netpbm (P5/PGM) frames,
gzipped, decodable with only the standard library -- no ffmpeg/opencv/GPU.

Key difference: the subject (a bright "fighter" disc) is visible in *every*
frame. What signals play is whether it is *moving*. During an exchange the
fighter sweeps across the mat with a striking bob; between exchanges it freezes
in place. ``detect.detect_motion`` thresholds the frame-to-frame difference, so a
frozen fighter produces no motion and reads as a gap -- exactly the signal
highlights.segment_plays uses to split play into exchanges.

Outputs (committed as fixtures, regenerate with ``python make_martialarts_clip.py``):
  martialarts_clip.pgm.gz       -- gzipped P5/PGM frame sequence (the "clip").
  martialarts_clip.events.json  -- ground-truth martial-arts events bundled with it.
"""
import gzip
import json
import math
import os

WIDTH = 80
HEIGHT = 45
FPS = 10
FIGHTER_RADIUS = 2
FIGHTER_VALUE = 255

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP_PATH = os.path.join(HERE, "martialarts_clip.pgm.gz")
EVENTS_PATH = os.path.join(HERE, "martialarts_clip.events.json")

# Each exchange is a run of frames where the fighter moves; between them the
# fighter freezes (no motion) for longer than the martial-arts gap threshold
# (1.0s -> 10 frames at 10 fps), ending the exchange.
# (start_frame, end_frame_inclusive, x0, x1, y_base) -- a horizontal sweep with a
# vertical striking bob superimposed.
_EXCHANGES = [
    (0, 14, 12, 64, 22),    # exchange A: left -> right
    (29, 45, 66, 14, 26),   # exchange B: right -> left (after a 1.4s frozen gap)
    (60, 76, 16, 60, 20),   # exchange C: left -> right (after a 1.4s frozen gap)
]
TOTAL_FRAMES = _EXCHANGES[-1][1] + 1
_BOB_AMPLITUDE = 6   # px of vertical strike bob across an exchange


def _moving_center(frame_idx):
    """Fighter (cx, cy) for a frame inside an exchange, or None when between."""
    for start, end, x0, x1, y_base in _EXCHANGES:
        if start <= frame_idx <= end:
            span = max(1, end - start)
            p = (frame_idx - start) / span  # 0..1 across the exchange
            cx = x0 + (x1 - x0) * p
            # Three striking bobs across the exchange so the fighter is always
            # displaced frame-to-frame (keeps motion energy non-zero throughout).
            cy = y_base + _BOB_AMPLITUDE * math.sin(p * math.pi * 3)
            return cx, cy
    return None


def _exchange_end_center(frame_idx):
    """Frozen rest position: the end position of the most recent exchange."""
    rest = None
    for start, end, x0, x1, y_base in _EXCHANGES:
        if end < frame_idx:
            rest = _moving_center(end)
    return rest


def fighter_center(frame_idx):
    """Fighter center for any frame: moving inside an exchange, else frozen.

    The fighter is always on the mat (visible), so a gap is conveyed by the
    fighter standing still rather than disappearing -- which is what the
    motion-energy detector keys on.
    """
    moving = _moving_center(frame_idx)
    if moving is not None:
        return moving
    return _exchange_end_center(frame_idx)


def _render_frame(frame_idx):
    """Render one frame as a flat bytes buffer (row-major, 1 byte/pixel)."""
    buf = bytearray(WIDTH * HEIGHT)  # all dark
    center = fighter_center(frame_idx)
    if center is None:
        return bytes(buf)
    cx, cy = center
    icx, icy = int(round(cx)), int(round(cy))
    r2 = FIGHTER_RADIUS * FIGHTER_RADIUS
    for dy in range(-FIGHTER_RADIUS, FIGHTER_RADIUS + 1):
        py = icy + dy
        if not (0 <= py < HEIGHT):
            continue
        for dx in range(-FIGHTER_RADIUS, FIGHTER_RADIUS + 1):
            px = icx + dx
            if not (0 <= px < WIDTH):
                continue
            if dx * dx + dy * dy <= r2:
                buf[py * WIDTH + px] = FIGHTER_VALUE
    return bytes(buf)


# Ground-truth martial-arts events, each timed inside an exchange so the tagger
# and strike-zone heatmap have real strikes to bin. ``pos`` is taken from the
# fighter path at that instant so binning is deterministic. Times are seconds.
def _event(t, etype):
    center = fighter_center(int(round(t * FPS)))
    pos = [round(center[0], 2), round(center[1], 2)] if center else None
    return {"t": t, "type": etype, "pos": pos}


_EVENTS = [
    _event(0.3, "jab"),
    _event(0.7, "cross"),
    _event(1.1, "hook"),
    _event(3.2, "kick"),
    _event(3.8, "takedown"),
    _event(4.2, "knee"),
    _event(6.3, "elbow"),
    _event(6.9, "clinch"),
    _event(7.3, "jab"),
]


def build_clip_bytes():
    """Build the full gzipped P5/PGM frame-sequence container as bytes."""
    out = bytearray()
    header = b"P5\n%d %d\n255\n" % (WIDTH, HEIGHT)
    for i in range(TOTAL_FRAMES):
        out += header
        out += _render_frame(i)
    return gzip.compress(bytes(out), mtime=0)


def write_fixtures():
    with open(CLIP_PATH, "wb") as fh:
        fh.write(build_clip_bytes())
    events_doc = {"fps": FPS, "source": "fixtures/martialarts_clip.pgm.gz", "events": _EVENTS}
    with open(EVENTS_PATH, "w") as fh:
        json.dump(events_doc, fh, indent=2)
        fh.write("\n")
    return CLIP_PATH, EVENTS_PATH


def main():
    clip, events = write_fixtures()
    size = os.path.getsize(clip)
    print(f"Wrote {clip} ({TOTAL_FRAMES} frames, {size} bytes) and {events}")


if __name__ == "__main__":
    main()
