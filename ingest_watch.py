#!/usr/bin/env python3
"""Drop-folder auto-detect for the volleyball pipeline (Overseer issue #8).

The pipeline was fully idle -- a capable CV system that nothing was feeding.
This module closes the gap between "pipeline works" and "pipeline gets used" by
watching an ingestion folder for new footage and enqueueing unseen clips so the
next run processes them automatically.

It is intentionally dependency-free (stdlib only) and side-effect light: one
scan reads the drop folder, diffs it against a small seen-state manifest, and
writes a queue of new clips. The overseer status step then surfaces the queue
depth and an idle nudge so prolonged idleness is visible instead of silent.

Environment variables
---------------------
VOLLEYBALL_DROP_DIR       Folder to watch for new footage (default: drop).
VOLLEYBALL_INGEST_STATE   Seen-state manifest path (default: ingest_state.json).
VOLLEYBALL_INGEST_QUEUE   Pending-clip queue path (default: ingest_queue.json).
"""
import json
import os
from datetime import datetime, timezone

DEFAULT_DROP_DIR = "drop"
DEFAULT_STATE_PATH = "ingest_state.json"
DEFAULT_QUEUE_PATH = "ingest_queue.json"

# Extensions we treat as ingestible footage. Lower-cased compare.
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".mpg", ".mpeg")


def _utc_now_iso() -> str:
    """Current UTC time as ISO-8601 with a trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_video(name: str) -> bool:
    """True when a filename looks like ingestible footage."""
    return name.lower().endswith(VIDEO_EXTENSIONS)


def load_json(path: str, default):
    """Read a JSON file, returning ``default`` when it does not exist."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default


def scan_drop_folder(drop_dir: str):
    """Return a sorted list of footage records found in ``drop_dir``.

    Each record is keyed by relative path and carries size/mtime so a clip that
    is still being copied (size changing between scans) is not enqueued until it
    settles. Missing folders yield an empty list rather than an error -- an
    unconfigured drop folder is "idle", not "broken".
    """
    records = {}
    if not os.path.isdir(drop_dir):
        return records
    for root, _dirs, files in os.walk(drop_dir):
        for name in files:
            if not is_video(name):
                continue
            full = os.path.join(root, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = os.path.relpath(full, drop_dir)
            records[rel] = {
                "path": rel,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
            }
    return records


def diff_new_footage(found: dict, seen: dict):
    """Return clips in ``found`` that are new or have changed since ``seen``.

    A clip is "new" if its path is unseen, or if its size changed (a re-upload /
    still-copying file). Comparing size guards against enqueuing a partial file
    captured mid-copy: it will only be considered stable once its size matches a
    prior scan.
    """
    new = []
    for rel, rec in found.items():
        prev = seen.get(rel)
        if prev is None or prev.get("size") != rec.get("size"):
            new.append(rec)
    return sorted(new, key=lambda r: r["path"])


def run_scan(
    drop_dir=None,
    state_path=None,
    queue_path=None,
    now_iso=None,
):
    """Scan the drop folder, enqueue stable new clips, and persist state.

    Returns a summary dict: number found, number newly stable-and-enqueued, the
    queue depth, and the most recent footage timestamp. Two-phase stability is
    used: a clip first seen is recorded but only enqueued once a later scan sees
    the same size, so files copied slowly aren't processed half-written.
    """
    drop_dir = drop_dir or os.environ.get("VOLLEYBALL_DROP_DIR", DEFAULT_DROP_DIR)
    state_path = state_path or os.environ.get("VOLLEYBALL_INGEST_STATE", DEFAULT_STATE_PATH)
    queue_path = queue_path or os.environ.get("VOLLEYBALL_INGEST_QUEUE", DEFAULT_QUEUE_PATH)
    now_iso = now_iso or _utc_now_iso()

    seen = load_json(state_path, {}).get("clips", {})
    found = scan_drop_folder(drop_dir)

    queue = load_json(queue_path, {}).get("pending", [])
    queued_paths = {item["path"] for item in queue}

    newly_enqueued = []
    next_seen = {}
    for rel, rec in found.items():
        prev = seen.get(rel)
        record = dict(rec)
        if prev is None:
            # First sighting: record it but wait for a stable second scan.
            record["first_seen"] = now_iso
            record["enqueued"] = False
        else:
            record["first_seen"] = prev.get("first_seen", now_iso)
            stable = prev.get("size") == rec.get("size")
            already = prev.get("enqueued") or rel in queued_paths
            if stable and not already:
                record["enqueued"] = True
                record["enqueued_at"] = now_iso
                newly_enqueued.append(record)
                queue.append({"path": rel, "enqueued_at": now_iso, "size": rec["size"]})
                queued_paths.add(rel)
            else:
                record["enqueued"] = prev.get("enqueued", False)
                if record["enqueued"]:
                    record["enqueued_at"] = prev.get("enqueued_at", now_iso)
        next_seen[rel] = record

    last_footage_at = None
    if found:
        # Most recent clip mtime expressed as a timestamp.
        newest = max(found.values(), key=lambda r: r["mtime"])
        last_footage_at = datetime.fromtimestamp(
            newest["mtime"], tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    state = {"updated_at": now_iso, "clips": next_seen}
    with open(state_path, "w") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)
        fh.write("\n")

    queue_doc = {"updated_at": now_iso, "pending": queue}
    with open(queue_path, "w") as fh:
        json.dump(queue_doc, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return {
        "drop_dir": drop_dir,
        "found": len(found),
        "newly_enqueued": len(newly_enqueued),
        "pending": len(queue),
        "last_footage_at": last_footage_at,
    }


def main() -> None:
    summary = run_scan()
    print(f"ingest scan: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
