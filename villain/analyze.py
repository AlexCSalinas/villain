"""One place where a profile becomes a finished read.

``build_profile`` produces numbers; the archetype, the leaks and the rating are
separate steps that every consumer needs and that each used to run on its own.
Doing it here means the terminal report, the JSON export and anything written
later cannot disagree about what a player is.
"""

from __future__ import annotations

from .archetypes import ARCHETYPE_BY_NAME, match
from .exploits import find_leaks
from .playbook import combinations_for
from .profile import Profile
from .skill import rate

#: Internal counters. They exist so aggression frequencies can be derived from
#: raw action mixes; as standalone frequencies they mean nothing, so they are
#: kept out of anything user-facing.
INTERNAL_PREFIXES = ("act:", "seat:", "saw:")


def enrich(profile: Profile) -> Profile:
    """Attach archetype, leaks and rating. Idempotent."""
    if profile.archetype == "unknown":
        profile.archetype, profile.archetype_confidence, profile.archetype_mix = match(profile)
    if not profile.tags:
        profile.tags = find_leaks(profile)
    if profile.skill is None:
        profile.skill = rate(profile)
    return profile


def enrich_all(profiles: list[Profile]) -> list[Profile]:
    return [enrich(p) for p in profiles]


def is_public(stat: str) -> bool:
    return not stat.startswith(INTERNAL_PREFIXES)


def as_dict(profile: Profile) -> dict:
    """Machine-readable profile, for feeding a solver or a spreadsheet."""
    enrich(profile)
    return {
        "player_id": profile.player_id,
        "name": profile.name,
        "regime": profile.regime,
        "hands": profile.hands,
        "sample_quality": profile.sample_quality,
        "first_seen": profile.first_seen,
        "last_seen": profile.last_seen,
        "archetype": profile.archetype,
        "archetype_confidence": profile.archetype_confidence,
        "archetype_mix": profile.archetype_mix,
        "skill": {
            "score": profile.skill.score,
            "tier": profile.skill.tier,
            "confidence": profile.skill.confidence,
            "exploitability_bb100": profile.skill.exploitability,
            "components": [
                {"name": c.name, "score": round(c.score, 1), "weight": c.weight,
                 "note": c.note}
                for c in profile.skill.components
            ],
            "observed_bb100": profile.winrate_bb100,
            "adjusted_bb100": profile.skill.adjusted_bb100,
        },
        "stats": {
            stat: {"value": round(est.value, 4), "opportunities": est.opps,
                   "raw": None if est.raw is None else round(est.raw, 4)}
            for stat, est in sorted(profile.stats.items()) if is_public(stat)
        },
        "leaks": [
            {"id": l.id, "headline": l.headline, "severity_bb100": round(l.severity, 3),
             "confidence": round(l.confidence, 3), "tier": l.tier,
             "value": round(l.value, 4), "breakeven": round(l.threshold, 4),
             "sample": l.opps, "advice": l.advice, "stat": l.stat,
             "direction": l.direction,
             # the plain-language layer
             "behaviour": l.behaviour, "why": l.why, "do": l.do, "dont": l.dont,
             "size": l.size, "priority": l.priority, "pressure": l.pressure,
             "in_words": l.in_words}
            for l in profile.tags
        ],
        # Leaks that compound. Two tendencies pointing the same way call for a
        # more aggressive adjustment than either would on its own.
        "combinations": [
            {"headline": c.headline, "body": c.body, "leaks": sorted(c.leaks)}
            for c in combinations_for(l.id for l in profile.tags)
        ],
        "plan": (ARCHETYPE_BY_NAME[profile.archetype].plan
                 if profile.archetype in ARCHETYPE_BY_NAME else ""),
    }
