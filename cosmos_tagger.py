#!/usr/bin/env python3
"""Optional clip tagging via NVIDIA Cosmos Reason (vision-language model).

In June 2026 NVIDIA open-sourced the Cosmos 3 family -- including Cosmos Reason,
a reasoning vision-language model for advanced multimodal video understanding,
served as an NVIDIA NIM microservice. This module uses it to enrich the
heuristic, event-window highlight tags in highlights.py with model-derived
coaching tags (e.g. confirming an "attack" vs "block", spotting a "dig" the
event stream missed).

Design constraints
------------------
* Strictly optional. The pipeline must run with zero extra dependencies; this
  module only activates when a NIM endpoint is configured AND a clip can be
  sampled. Otherwise highlights.py falls back to heuristic tags.
* Stdlib only (urllib). No GPU, no torch, no NVIDIA SDK required on the runner
  -- inference happens behind the NIM HTTP endpoint.
* Best-effort: any failure returns the baseline tags unchanged. A coaching
  highlight reel is never worth crashing the pipeline for.

Environment variables
---------------------
VOLLEYBALL_COSMOS_NIM_URL  Cosmos Reason NIM chat-completions endpoint.
                           e.g. https://integrate.api.nvidia.com/v1/chat/completions
VOLLEYBALL_COSMOS_MODEL    Model id (default: nvidia/cosmos-reason-3).
VOLLEYBALL_COSMOS_API_KEY  Bearer token for hosted NIM (optional for local NIM).
VOLLEYBALL_COSMOS_TIMEOUT  Per-request timeout seconds (default: 30).
"""
import json
import os
import urllib.error
import urllib.request

DEFAULT_MODEL = "nvidia/cosmos-reason-3"
DEFAULT_TIMEOUT = 30

# The coaching vocabulary we ask the model to choose from. Keeping the model
# constrained to a known set keeps its output mergeable with heuristic tags.
PROMPT_TAGS = ("serve", "reception", "set", "attack", "block", "dig", "ace", "error")


def is_configured() -> bool:
    """True when a Cosmos Reason endpoint is configured."""
    return bool(os.environ.get("VOLLEYBALL_COSMOS_NIM_URL"))


def _build_prompt(start, end, base_tags):
    base = ", ".join(base_tags) if base_tags else "none detected"
    vocab = ", ".join(PROMPT_TAGS)
    return (
        "You are a volleyball video analyst. A rally clip runs from "
        f"{start:.1f}s to {end:.1f}s. Heuristic tags so far: {base}. "
        f"From this vocabulary only [{vocab}], list every coaching event visible "
        "in the rally. Respond with a JSON array of strings, nothing else."
    )


def query_cosmos(start, end, base_tags, url=None, model=None, api_key=None, timeout=None):
    """Call Cosmos Reason and return the list of tags it reports.

    Returns None on any error so callers can fall back. The request shape is the
    OpenAI-compatible chat-completions API that NVIDIA NIM exposes.
    """
    url = url or os.environ.get("VOLLEYBALL_COSMOS_NIM_URL")
    if not url:
        return None
    model = model or os.environ.get("VOLLEYBALL_COSMOS_MODEL", DEFAULT_MODEL)
    api_key = api_key or os.environ.get("VOLLEYBALL_COSMOS_API_KEY")
    timeout = timeout or int(os.environ.get("VOLLEYBALL_COSMOS_TIMEOUT", DEFAULT_TIMEOUT))

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _build_prompt(start, end, base_tags)}],
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return parse_response(body)


def parse_response(body):
    """Extract a tag list from a chat-completions response body.

    Tolerant of the model wrapping its JSON array in prose: finds the first
    bracketed array and parses it. Returns None when nothing usable is found.
    """
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    start = content.find("[")
    end = content.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        tags = json.loads(content[start : end + 1])
    except ValueError:
        return None
    if not isinstance(tags, list):
        return None
    return [str(t).strip().lower() for t in tags if str(t).strip()]


def merge_tags(base_tags, model_tags):
    """Union of heuristic and model tags, restricted to the known vocabulary.

    Restricting to PROMPT_TAGS guards against the model inventing labels the
    dashboard doesn't understand. Order follows the canonical vocabulary.
    """
    combined = set(base_tags or [])
    for t in model_tags or []:
        if t in PROMPT_TAGS:
            combined.add(t)
    return [t for t in PROMPT_TAGS if t in combined] + sorted(
        t for t in combined if t not in PROMPT_TAGS
    )


def make_enricher():
    """Return a tag_enricher(source, start, end, base_tags) -> tags callable.

    Raises RuntimeError when no endpoint is configured so the caller can fall
    back loudly during setup, but the returned callable itself never raises --
    it returns the baseline tags on any per-clip failure.
    """
    if not is_configured():
        raise RuntimeError("VOLLEYBALL_COSMOS_NIM_URL not set")

    def enrich(source, start, end, base_tags):
        model_tags = query_cosmos(start, end, base_tags)
        if not model_tags:
            return base_tags
        return merge_tags(base_tags, model_tags)

    return enrich
