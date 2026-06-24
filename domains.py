#!/usr/bin/env python3
"""Sport domains: make the CV pipeline switchable across volleyball / martial arts.

The pipeline is mechanically domain-neutral. It tracks a single *subject* point
per frame, segments play from gaps in that point's motion, tags each segment
from timed events, and bins events onto a *surface* grid. What actually differs
between sports is:

  * how the subject is found in the pixels (a bright ball vs. a moving fighter),
  * the action vocabulary used as coaching tags,
  * the words used in the human-facing report, and
  * sensible segmentation thresholds.

A :class:`Domain` bundles exactly those differences so every other module can
stay generic and just ask the active domain. Select it with the
``COACHVISION_DOMAIN`` environment variable or a ``--domain`` CLI flag. The default
is volleyball, which preserves prior behaviour and the overseer's existing
machine-readable status contract (``segment_count``, ``frames_processed``, ...).

Why these two detectors?
------------------------
Volleyball has an obvious high-contrast subject -- the ball -- so a brightest
-blob centroid is an honest, dependency-free detector. Martial arts has no ball;
the subject is the fighter, and the signal that "play" is happening is *motion*.
The standard dependency-free approach is motion-energy temporal segmentation:
threshold the frame-to-frame pixel difference and take the centroid of what
changed, so the tracked point follows the action and quiet moments read as gaps
between exchanges. See domains' ``motion_energy`` detector in detect.py.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    """Everything that distinguishes one sport from another in the pipeline."""

    key: str                 # canonical id, e.g. "volleyball" / "martial_arts"
    label: str               # human label for reports, e.g. "Martial arts"

    # CV front-end: which detector recovers the subject point from pixels.
    detector: str            # "brightest_blob" | "motion_energy"
    detect_threshold: int    # per-pixel brightness (blob) or |delta| (motion)
    detect_min_pixels: int   # fewer qualifying pixels than this -> no subject

    # Coaching vocabulary: the action tags this sport recognises, in display
    # order. The first match in this tuple sorts first in a segment's tag list.
    tags: tuple

    # Segmentation defaults (seconds). Martial-arts exchanges are shorter and
    # the lulls between them briefer than a volleyball rally's dead time.
    max_gap_s: float         # subject missing/still longer than this ends a play
    min_segment_s: float     # discard blips shorter than this
    pad_s: float             # context padding added before/after each clip

    # Report vocabulary. The pipeline's internal JSON keys stay stable for the
    # overseer; only the words humans read change with the domain.
    segment_noun: str        # "rally" / "exchange"  (also the clip id prefix)
    segment_plural: str      # "rallies" / "exchanges"
    subject_noun: str        # "ball" / "fighter"
    surface_noun: str        # "court" / "mat"
    action_noun: str        # "contact" / "strike"

    # Role string handed to the optional Cosmos Reason vision-language tagger.
    analyst_role: str        # "volleyball video analyst" / ...

    @property
    def action_plural(self) -> str:
        return self.action_noun + "s"


VOLLEYBALL = Domain(
    key="volleyball",
    label="Volleyball",
    detector="brightest_blob",
    detect_threshold=200,
    detect_min_pixels=3,
    tags=("serve", "reception", "set", "attack", "block", "dig", "ace", "error"),
    max_gap_s=2.0,
    min_segment_s=1.0,
    pad_s=1.0,
    segment_noun="rally",
    segment_plural="rallies",
    subject_noun="ball",
    surface_noun="court",
    action_noun="contact",
    analyst_role="volleyball video analyst",
)

MARTIAL_ARTS = Domain(
    key="martial_arts",
    label="Martial arts",
    detector="motion_energy",
    detect_threshold=60,      # |frame delta| at/above this counts as motion
    detect_min_pixels=3,
    tags=(
        "jab", "cross", "hook", "uppercut", "kick", "knee", "elbow",
        "takedown", "clinch", "sweep", "submission", "block", "dodge",
    ),
    # Exchanges fire and reset quickly, so a shorter gap closes a segment and a
    # shorter minimum keeps brief flurries instead of discarding them.
    max_gap_s=1.0,
    min_segment_s=0.5,
    pad_s=0.8,
    segment_noun="exchange",
    segment_plural="exchanges",
    subject_noun="fighter",
    surface_noun="mat",
    action_noun="strike",
    analyst_role="martial arts video analyst",
)

# Registry keyed by canonical id, plus friendly aliases people actually type.
REGISTRY = {d.key: d for d in (VOLLEYBALL, MARTIAL_ARTS)}
_ALIASES = {
    "vb": "volleyball",
    "volley": "volleyball",
    "mma": "martial_arts",
    "ma": "martial_arts",
    "martialarts": "martial_arts",
    "martial-arts": "martial_arts",
    "fight": "martial_arts",
    "fighting": "martial_arts",
}

DEFAULT_DOMAIN = "volleyball"
ENV_VAR = "COACHVISION_DOMAIN"


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def get_domain(name=None) -> Domain:
    """Resolve a Domain from an explicit name, the env var, or the default.

    Accepts a :class:`Domain` (returned as-is), a canonical id, or a friendly
    alias (``mma``, ``martial-arts``, ...). Raises ``ValueError`` on an unknown
    name so a typo fails loudly instead of silently running the wrong sport.
    """
    if isinstance(name, Domain):
        return name
    if name is None:
        name = os.environ.get(ENV_VAR, DEFAULT_DOMAIN)
    key = _normalize(str(name))
    key = _ALIASES.get(key, key)
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"unknown domain {name!r}; known domains: {known}")
    return REGISTRY[key]
