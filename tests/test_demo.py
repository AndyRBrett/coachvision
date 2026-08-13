"""Tests for the bundled demo session (issue #21).

The demo exists so the dashboard isn't empty before real footage arrives, and so
the report-WRITING path is exercised continuously rather than first attempted on
the day real footage lands. The load-bearing property is the separation: a demo
must never be counted as ingested footage.
"""

import json
import os

import pytest

import demo


@pytest.fixture
def published(tmp_path):
    reports = str(tmp_path / "reports")
    entry = demo.build_demo(reports, domain="martial_arts")
    return reports, entry


def test_demo_publishes_a_full_report(published):
    reports, entry = published
    report = json.load(open(os.path.join(reports, "demo", "coaching", "report.json")))
    assert report["segment_count"] >= 1
    assert report["total_play_s"] > 0
    assert any(s["tags"] for s in report["segments"]), "no coaching tags in the demo report"


def test_demo_writes_every_artifact_a_real_session_does(published):
    reports, _ = published
    for rel in ("coaching/report.json", "coaching/summary.txt", "results/metrics.json"):
        assert os.path.exists(os.path.join(reports, "demo", rel)), f"missing {rel}"


def test_demo_entry_is_flagged_so_it_cannot_pass_as_real_footage(published):
    # THE ONE THAT MATTERS. Without demo:true the entry is indistinguishable from
    # ingested footage, and "has this ever processed anything real?" silently
    # starts answering yes.
    _, entry = published
    assert entry["demo"] is True


def test_demo_lands_in_the_shared_catalog(published):
    reports, entry = published
    index = json.load(open(os.path.join(reports, "index.json")))
    ids = [c["id"] for c in index["clips"]]
    assert entry["id"] in ids


def test_republishing_does_not_duplicate_the_catalog_entry(published):
    reports, _ = published
    demo.build_demo(reports, domain="martial_arts")
    index = json.load(open(os.path.join(reports, "index.json")))
    assert sum(1 for c in index["clips"] if c["id"] == demo.DEMO_ID) == 1


def test_demo_does_not_disturb_existing_real_sessions(tmp_path):
    reports = str(tmp_path / "reports")
    os.makedirs(reports, exist_ok=True)
    real = {"id": "clip-real", "title": "Sparring", "processed_at": "2026-08-01T00:00:00Z"}
    with open(os.path.join(reports, "index.json"), "w") as fh:
        json.dump({"clips": [real]}, fh)

    demo.build_demo(reports, domain="martial_arts")
    index = json.load(open(os.path.join(reports, "index.json")))
    kept = next(c for c in index["clips"] if c["id"] == "clip-real")
    assert kept["title"] == "Sparring"
    assert not kept.get("demo")


def test_check_passes_on_a_freshly_published_demo(published):
    reports, _ = published
    assert demo.check_demo(reports) == []


def test_check_detects_a_stale_published_report(published):
    # The continuous-coverage case: the pipeline changed, the committed demo
    # didn't. Silence here would let the sample drift away from reality.
    reports, _ = published
    path = os.path.join(reports, "demo", "coaching", "report.json")
    report = json.load(open(path))
    report["segment_count"] = report["segment_count"] + 99
    json.dump(report, open(path, "w"))
    problems = demo.check_demo(reports)
    assert any("stale" in p for p in problems)


def test_check_flags_a_demo_entry_that_lost_its_flag(published):
    reports, _ = published
    index_path = os.path.join(reports, "index.json")
    index = json.load(open(index_path))
    for c in index["clips"]:
        if c["id"] == demo.DEMO_ID:
            c["demo"] = False
    json.dump(index, open(index_path, "w"))
    assert any("real footage" in p for p in demo.check_demo(reports))


def test_check_reports_a_missing_demo_instead_of_crashing(tmp_path):
    reports = str(tmp_path / "reports")
    os.makedirs(reports, exist_ok=True)
    json.dump({"clips": []}, open(os.path.join(reports, "index.json"), "w"))
    problems = demo.check_demo(reports)
    assert problems and "no 'demo' entry" in problems[0]


def test_committed_demo_matches_the_current_pipeline():
    # Guards the artifact actually in the repo, the way the workflow will run it.
    os.environ.setdefault("COACHVISION_DOMAIN", "martial_arts")
    assert demo.check_demo("reports") == []
