#!/usr/bin/env python3
"""Turn the computed fight stats + a few annotated stills into real coaching
feedback, using Claude (Anthropic) vision.

The numbers the CV pipeline produces -- strike attempts, hand vs leg split,
fighter presence, exchange count -- are honest but terse. To deliver feedback a
coach would actually *say*, we hand those stats AND a few annotated still frames
(boxes tracking the fighters, HAND/LEG markers on detected strikes) to Claude and
ask for specific, actionable notes.

This step is optional and cost-gated: it runs only when ``ANTHROPIC_API_KEY`` is
set (a GitHub Actions secret). With no key the pipeline skips it cleanly and the
rest of the report is unaffected -- ``generate_feedback`` just returns ``None``.

The stat-summary and message-building are pure functions (unit-tested without the
network); the single API call is isolated in :func:`generate_feedback` and the
``anthropic`` SDK is imported lazily so the dependency is only needed when a key
is actually present.
"""
import base64
import os

# Default to Claude Opus 4.8 -- the most capable Opus-tier model. Adaptive
# thinking + streaming follow the SDK's defaults for non-trivial vision work.
MODEL = "claude-opus-4-8"
MAX_TOKENS = 3000
MAX_STILLS = 4

SYSTEM_PROMPT = (
    "You are a {role} giving feedback on a short sparring clip. You are given "
    "automatically-computed stats and a few annotated still frames: coloured "
    "boxes track the fighters and HAND/LEG markers flag detected strike attempts. "
    "The stats come from a lightweight CPU pose model -- it can miss or miscount "
    "strikes and occasionally tracks the wrong person -- so treat the numbers as "
    "rough signal, lean on what you can actually see in the frames, and never "
    "invent precision the data doesn't support. Give the fighter specific, "
    "actionable feedback. Answer in Markdown with exactly these sections, each a "
    "few sentences or tight bullets:\n"
    "**What I see** -- the picture the stats + frames paint.\n"
    "**Strengths** -- what looks good.\n"
    "**To work on** -- the highest-leverage fixes.\n"
    "**One drill** -- a single concrete drill for the next session."
)


def summarize_stats(tracking, segment_count=None, events=None):
    """Build a compact, human-readable stats block from the analysis output.

    Pure: takes the ``tracking`` dict (and optional ``segment_count``) the
    pipeline already computed and returns the text we put in front of the model.
    """
    events = events if events is not None else tracking.get("events", [])
    hand = sum(1 for e in events if e.get("type") == "hand_strike")
    leg = sum(1 for e in events if e.get("type") == "leg_strike")
    fps = tracking.get("fps") or 1
    frames = tracking.get("frame_count", 0)
    detected = tracking.get("detected_frames", 0)
    dur = frames / fps if fps else 0.0
    present = (100 * detected / frames) if frames else 0.0

    lines = [
        f"Clip length: {dur:.1f}s ({frames} frames @ {fps:g} fps)",
        f"Fighters in frame: {present:.0f}% of frames",
        f"Strike attempts detected: {hand + leg} (hand {hand}, leg {leg})",
    ]
    if dur > 0:
        lines.append(f"Strike rate: {(hand + leg) / dur * 60:.1f} per minute")
    if hand + leg:
        lines.append(f"Hand/leg mix: {round(100 * hand / (hand + leg))}% hand / "
                     f"{round(100 * leg / (hand + leg))}% leg")
    if segment_count is not None:
        lines.append(f"Exchanges (motion segments): {segment_count}")
    return "\n".join(lines)


def _image_block(path):
    """A base64 image content block (jpeg) for the Messages API."""
    with open(path, "rb") as fh:
        data = base64.standard_b64encode(fh.read()).decode("ascii")
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def build_messages(stats_text, still_paths):
    """Assemble the user turn: stats text, then up to MAX_STILLS annotated frames.

    Pure (apart from reading the image files). Returns the ``messages`` list for
    ``messages.create`` / ``messages.stream``.
    """
    content = [{"type": "text",
                "text": "Computed stats for this clip:\n\n" + stats_text}]
    for path in still_paths[:MAX_STILLS]:
        content.append(_image_block(path))
    content.append({"type": "text",
                    "text": "Give your coaching feedback now, using the sections above."})
    return [{"role": "user", "content": content}]


def generate_feedback(tracking, still_paths, role="martial arts coach",
                      segment_count=None, api_key=None, model=MODEL):
    """Call Claude for coaching feedback; return Markdown, or None if disabled.

    Cost-gated: with no ``ANTHROPIC_API_KEY`` (env or arg) this returns ``None``
    so the pipeline skips the step cleanly. Otherwise it streams one Opus 4.8
    vision request (adaptive thinking) and returns the assistant's Markdown text.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic  # lazy: only needed when a key is present

    stats_text = summarize_stats(tracking, segment_count=segment_count)
    messages = build_messages(stats_text, still_paths)
    client = anthropic.Anthropic(api_key=api_key)
    # Stream to avoid request timeouts on a thinking + vision call; collect the
    # final message via the SDK helper.
    with client.messages.stream(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT.format(role=role),
        thinking={"type": "adaptive"},
        messages=messages,
    ) as stream:
        final = stream.get_final_message()
    return "".join(b.text for b in final.content if getattr(b, "type", None) == "text").strip()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Stats summary / feedback smoke test.")
    parser.add_argument("tracking_json", help="Path to a tracking.json (or report) file")
    parser.add_argument("--still", action="append", default=[], help="Annotated still image(s)")
    parser.add_argument("--segments", type=int, default=None)
    args = parser.parse_args()
    with open(args.tracking_json) as fh:
        tracking = json.load(fh)
    print(summarize_stats(tracking, segment_count=args.segments))
    print("---")
    fb = generate_feedback(tracking, args.still, segment_count=args.segments)
    print(fb if fb is not None else "(no ANTHROPIC_API_KEY -> feedback skipped)")
