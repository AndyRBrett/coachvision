#!/usr/bin/env python3
"""Dispatch process-footage workflow runs for items in the ingest queue.

Reads ingest_queue.json, triggers the process-footage.yml workflow via the
GitHub API for each pending clip, then removes successfully dispatched items
so they are not re-triggered on the next run.

Runs inside the overseer-status workflow after ingest_watch.py, so footage
dropped into the watched folder is automatically queued and kicked off in the
same weekly (or on-demand) run -- closing the gap between "detect" and
"process" that the original ingest_watch.py left open.

Environment
-----------
GITHUB_TOKEN        Actions token (needs actions: write); set automatically.
GITHUB_REPOSITORY   owner/repo string; set automatically in Actions.
GITHUB_REF_NAME     Branch to run the workflow from (default: main).
COACHVISION_DOMAIN  Domain to pass to the process-footage workflow.
COACHVISION_INGEST_QUEUE  Path to the queue file (default: ingest_queue.json).
COACHVISION_DROP_DIR      Drop-folder prefix (default: drop).
COACHVISION_WORKFLOW      Workflow filename (default: process-footage.yml).
"""
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_QUEUE_PATH = "ingest_queue.json"
DEFAULT_DROP_DIR = "drop"
DEFAULT_WORKFLOW = "process-footage.yml"
DEFAULT_FPS = "5"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def trigger_workflow(owner, repo, workflow, ref, inputs, token):
    """POST a workflow_dispatch event; returns True on HTTP 204."""
    url = (f"https://api.github.com/repos/{owner}/{repo}"
           f"/actions/workflows/{workflow}/dispatches")
    payload = json.dumps({"ref": ref, "inputs": inputs}).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status == 204
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()
        print(f"  dispatch failed {exc.code}: {body}")
        return False


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set — skipping queue dispatch")
        return

    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repository:
        print("GITHUB_REPOSITORY not set — skipping queue dispatch")
        return

    owner, repo = repository.split("/", 1)
    ref = os.environ.get("GITHUB_REF_NAME", "main")
    domain = os.environ.get("COACHVISION_DOMAIN", "martial_arts")
    workflow = os.environ.get("COACHVISION_WORKFLOW", DEFAULT_WORKFLOW)
    queue_path = os.environ.get("COACHVISION_INGEST_QUEUE", DEFAULT_QUEUE_PATH)
    drop_dir = os.environ.get("COACHVISION_DROP_DIR", DEFAULT_DROP_DIR)

    try:
        with open(queue_path) as fh:
            queue_doc = json.load(fh)
    except (FileNotFoundError, ValueError):
        print("No ingest queue — nothing to dispatch")
        return

    pending = queue_doc.get("pending", [])
    if not pending:
        print("Ingest queue empty — nothing to dispatch")
        return

    print(f"Dispatching {len(pending)} queued clip(s) as {domain} …")
    dispatched_paths = set()
    for item in pending:
        rel = item.get("path", "")
        if not rel:
            continue
        clip_path = f"{drop_dir}/{rel}"
        print(f"  → {clip_path}")
        ok = trigger_workflow(
            owner, repo, workflow, ref,
            inputs={"clip_path": clip_path, "domain": domain, "fps": DEFAULT_FPS},
            token=token,
        )
        if ok:
            dispatched_paths.add(rel)
            print(f"    dispatched")

    if dispatched_paths:
        queue_doc["pending"] = [i for i in pending if i.get("path") not in dispatched_paths]
        queue_doc["updated_at"] = _utc_now_iso()
        with open(queue_path, "w") as fh:
            json.dump(queue_doc, fh, indent=2, sort_keys=True)
            fh.write("\n")

    remaining = len(queue_doc.get("pending", []))
    print(f"Dispatched {len(dispatched_paths)}/{len(pending)}; "
          f"{remaining} remaining in queue.")


if __name__ == "__main__":
    main()
