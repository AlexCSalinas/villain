"""A lineup briefing: who is at this table, and what to do about them.

Deliberately not an expected win rate. Per-session results in this pool have a
standard deviation around 183 bb/100, so telling two lineups apart on expected
value would need a gap near 229 bb/100 before the difference was detectable at
all -- any number smaller than that is noise wearing a decimal point. What a
briefing can honestly say is who is here, which of them you have a confirmed
read on, and which single leak is worth the most against each.

Ordered by what to do first, which is not the same as who is worst: a big leak
on somebody who has position on you all night is worth more than a bigger one
on the player you rarely contest a pot with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .analyze import enrich
from .exploits import find_leaks, find_watchlist
from .skill import weaknesses


@dataclass
class Read:
    """One opponent at the table, and the one thing to do about them."""

    player_id: int
    name: str
    hands: int
    archetype: str
    archetype_confidence: float
    skill: float
    headline: str | None = None
    plan: str | None = None
    severity: float = 0.0
    tier: str = ""
    status: str = "none"          # confirmed | watch | rated | none
    sample: str = ""

    @property
    def sort_key(self) -> tuple:
        rank = {"confirmed": 0, "watch": 1, "rated": 2, "none": 3}[self.status]
        return (rank, -self.severity, -self.hands)


@dataclass
class Briefing:
    reads: list[Read] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        out: list[str] = []
        if not self.reads:
            out.append("Nobody at this table is in the database yet.")
        for r in self.reads:
            conf = f"{r.archetype} {r.archetype_confidence:.0%}"
            out.append(f"{r.name}  ({r.hands} hands, {conf}, rated {r.skill:.0f})")
            if r.headline:
                mark = {"confirmed": "->", "watch": "?", "rated": "~"}.get(r.status, " ")
                note = f" [{r.tier}, {r.severity:.1f} bb/100]" if r.status == "confirmed" else \
                       (f" [{r.sample}]" if r.sample else "")
                out.append(f"   {mark} {r.headline}{note}")
                if r.plan:
                    out.append(f"      {r.plan}")
            else:
                out.append("   -  nothing clears the noise floor yet; play them straight")
        if self.missing:
            out.append("")
            out.append("Not in the database: " + ", ".join(self.missing))
        out.append("")
        out.append("No expected win rate, on purpose: per-session results here vary by "
                   "about 183 bb/100, so\nany lineup-level number smaller than ~229 "
                   "bb/100 would be noise. This says who is here and\nwhat to do, "
                   "not what you are going to make.")
        return "\n".join(out)


def brief(store, names: list[str]) -> Briefing:
    """Build the briefing for a list of names as they appear at the table."""
    briefing = Briefing()
    for needle in names:
        matches = store.find_player(needle)
        if not matches:
            briefing.missing.append(needle)
            continue
        # find_player returns id and display_name only, so pick the account
        # with the most hands by building each and comparing.
        profile = None
        for candidate in matches:
            built = store.profile(int(candidate["id"]))
            if built is not None and (profile is None or built.hands > profile.hands):
                profile, row = built, candidate
        if profile is None:
            briefing.missing.append(needle)
            continue
        enrich(profile)
        read = Read(
            player_id=int(row["id"]),
            name=profile.name or row["display_name"],
            hands=profile.hands,
            archetype=profile.archetype,
            archetype_confidence=profile.archetype_confidence,
            skill=profile.skill.score,
        )
        leaks = find_leaks(profile, dedupe=True)
        if leaks:
            top = leaks[0]
            read.headline, read.plan = top.headline, top.advice
            read.severity, read.tier, read.status = top.severity, top.tier, "confirmed"
        else:
            watch = find_watchlist(profile)
            if watch:
                top = watch[0]
                read.headline, read.plan = top.headline, top.advice
                read.status = "watch"
                read.sample = (f"{top.confidence:.0%} sure over {top.opps:.0f} spots"
                               " -- not confirmed")
            else:
                weak = weaknesses(profile.skill)
                if weak:
                    read.headline = f"Weakest part of their game: {weak[0].name}"
                    read.status = "rated"
                    read.sample = (f"scores {weak[0].score:.0f}/100 -- from the rating,"
                                   " not a measured frequency")
        briefing.reads.append(read)
    briefing.reads.sort(key=lambda r: r.sort_key)
    return briefing
