"""Timing-tell profile from measured shares and relative outcomes.

Online clocks are noisy -- disconnects, phones, multi-tabling -- so these reads
are for breaking ties, never for the whole decision. Each cell answers:

* Of this player's timed checks/calls/raises on this street, what share were
  snap or tank?
* After that clock+action, did they win the pot / go to showdown / fold to the
  next bet *differently* than after the same action at normal pace?

River is omitted on purpose; by then the hand is usually defined and the cards
tell you more than the clock. Folklore captions ("Giving up") are not used.
"""

from __future__ import annotations

from dataclasses import dataclass

from .profile import Profile

#: Below this many pace hits a cell stays empty rather than guessing.
MIN_PACE = 5
#: Outcome deltas need a thicker sample on both sides before we claim a tell.
MIN_OUTCOME = 8
MIN_SD = 6
#: Absolute gaps vs normal-pace baseline that count as a signal.
DELTA_WON = 0.12
DELTA_WTSD = 0.12
DELTA_FOLD = 0.15
DELTA_SD = 0.12

STREETS = ("flop", "turn")
PACES = ("snap", "tank")
#: Stored as ``aggro`` in the books; shown as "raise" in the grid.
ACTIONS = ("check", "call", "aggro")
ACTION_LABEL = {"check": "check", "call": "call", "aggro": "raise"}


@dataclass(frozen=True)
class TimingCell:
    pace: str
    street: str
    action: str          # check | call | aggro
    n: int
    total: int           # timed instances of this action on this street
    share: float | None  # n / total
    won: float | None
    won_base: float | None
    wtsd: float | None
    wtsd_base: float | None
    fold_next: float | None
    fold_next_base: float | None
    fold_next_n: int
    sd_strength: float | None
    sd_base: float | None
    sd_n: int
    label: str
    read: str

    @property
    def action_label(self) -> str:
        return ACTION_LABEL[self.action]


def timing_tells(profile: Profile) -> list[TimingCell]:
    """Every pace \u00d7 street \u00d7 action cell, including empty ones."""
    return [_cell(profile, pace, street, action)
            for pace in PACES
            for street in STREETS
            for action in ACTIONS]


def _rate(profile: Profile, stat: str) -> tuple[float | None, int]:
    est = profile.stats.get(stat)
    if est is None or not est.opps or est.raw is None:
        return None, 0
    return float(est.raw), int(round(est.opps))


def _mean(profile: Profile, stat: str) -> tuple[float | None, int]:
    value = profile.means.get(stat)
    n = int(profile.means.get(f"{stat}#n", 0) or 0)
    if value is None or n <= 0:
        return None, 0
    return float(value), n


def _cell(profile: Profile, pace: str, street: str, action: str) -> TimingCell:
    pace_est = profile.stats.get(f"pace:{pace}:{street}:{action}")
    timed_est = profile.stats.get(f"timed:{street}:{action}")
    if pace_est is None or not pace_est.opps or pace_est.raw is None:
        n, total = 0, int(timed_est.opps) if timed_est else 0
    else:
        n = int(round(pace_est.raw * pace_est.opps))
        total = int(round(timed_est.opps)) if timed_est and timed_est.opps else int(round(pace_est.opps))
    share = (n / total) if total else None

    won, won_n = _rate(profile, f"after:{pace}:{street}:{action}:won")
    won_base, won_base_n = _rate(profile, f"after:normal:{street}:{action}:won")
    wtsd, wtsd_n = _rate(profile, f"after:{pace}:{street}:{action}:wtsd")
    wtsd_base, wtsd_base_n = _rate(profile, f"after:normal:{street}:{action}:wtsd")
    fold_next, fold_n = _rate(profile, f"after:{pace}:{street}:{action}:fold_next")
    fold_base, fold_base_n = _rate(profile, f"after:normal:{street}:{action}:fold_next")
    sd, sd_n = _mean(profile, f"after:{pace}:{street}:{action}:sd_strength")
    sd_base, sd_base_n = _mean(profile, f"after:normal:{street}:{action}:sd_strength")

    label, read = _interpret(
        pace, street, action, n, total, share,
        won, won_n, won_base, won_base_n,
        wtsd, wtsd_n, wtsd_base, wtsd_base_n,
        fold_next, fold_n, fold_base, fold_base_n,
        sd, sd_n, sd_base, sd_base_n,
    )
    return TimingCell(
        pace=pace, street=street, action=action, n=n, total=total, share=share,
        won=won, won_base=won_base, wtsd=wtsd, wtsd_base=wtsd_base,
        fold_next=fold_next, fold_next_base=fold_base, fold_next_n=fold_n,
        sd_strength=sd, sd_base=sd_base, sd_n=sd_n,
        label=label, read=read,
    )


def _pct(value: float | None) -> str:
    return "\u2014" if value is None else f"{100 * value:.0f}%"


def _delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return value - base


def _interpret(
    pace: str, street: str, action: str, n: int, total: int, share: float | None,
    won, won_n, won_base, won_base_n,
    wtsd, wtsd_n, wtsd_base, wtsd_base_n,
    fold_next, fold_n, fold_base, fold_base_n,
    sd, sd_n, sd_base, sd_base_n,
) -> tuple[str, str]:
    act = ACTION_LABEL[action]
    if n < MIN_PACE:
        return ("Not enough data",
                f"Need a few more {pace} {act}s on the {street} before this "
                f"cell means anything.")

    share_bit = (f"{100 * share:.0f}% of {street} {act}s ({n}/{total})"
                 if share is not None and total else f"{n} timed")
    lines = [share_bit]

    d_won = _delta(won, won_base)
    d_wtsd = _delta(wtsd, wtsd_base)
    d_fold = _delta(fold_next, fold_base)
    d_sd = _delta(sd, sd_base)

    outcome_ready = (
        won_n >= MIN_OUTCOME and won_base_n >= MIN_OUTCOME
    )
    fold_ready = fold_n >= MIN_OUTCOME and fold_base_n >= MIN_OUTCOME
    sd_ready = sd_n >= MIN_SD and sd_base_n >= MIN_SD

    signals: list[str] = []
    if outcome_ready and d_won is not None and abs(d_won) >= DELTA_WON:
        direction = "wins less" if d_won < 0 else "wins more"
        signals.append("weaker" if d_won < 0 else "stronger")
        lines.append(
            f"Won pot {_pct(won)} after vs {_pct(won_base)} normal "
            f"(\u0394{100 * d_won:+.0f}pp) \u2014 {direction}."
        )
    elif won_n or won_base_n:
        lines.append(
            f"Won pot {_pct(won)} after ({won_n}) vs {_pct(won_base)} normal "
            f"({won_base_n})."
        )

    if outcome_ready and d_wtsd is not None and abs(d_wtsd) >= DELTA_WTSD:
        signals.append("showdown")
        lines.append(
            f"WTSD {_pct(wtsd)} after vs {_pct(wtsd_base)} normal "
            f"(\u0394{100 * d_wtsd:+.0f}pp)."
        )

    if fold_ready and d_fold is not None and abs(d_fold) >= DELTA_FOLD:
        signals.append("folds next" if d_fold > 0 else "continues")
        lines.append(
            f"Folded next bet {_pct(fold_next)} after vs {_pct(fold_base)} "
            f"normal (\u0394{100 * d_fold:+.0f}pp)."
        )
    elif fold_n or fold_base_n:
        lines.append(
            f"Folded next bet {_pct(fold_next)} ({fold_n}) vs "
            f"{_pct(fold_base)} normal ({fold_base_n})."
        )

    if sd_ready and d_sd is not None and abs(d_sd) >= DELTA_SD:
        signals.append("shown weak" if d_sd < 0 else "shown strong")
        lines.append(
            f"Shown strength {100 * sd:.0f}th pctile vs "
            f"{100 * sd_base:.0f}th normal (\u0394{100 * d_sd:+.0f})."
        )

    if not signals:
        # High share of snap checks with no outcome edge is the OOP range-check case.
        if pace == "snap" and action == "check" and share is not None and share >= 0.35:
            return ("Mostly routine",
                    f"{share_bit}. Fast checks this often with no outcome gap "
                    f"vs normal pace usually means an automatic range check, "
                    f"not a give-up. " + " ".join(lines[1:]))
        return ("No clear tell",
                f"{share_bit}. Outcomes after this clock look like normal-pace "
                f"{act}s so far. " + " ".join(lines[1:]))

    # Label from the strongest signal, still backed by the numbers in read.
    if "weaker" in signals or "folds next" in signals or "shown weak" in signals:
        label = "Weaker than usual"
    elif "stronger" in signals or "continues" in signals or "shown strong" in signals:
        label = "Stronger than usual"
    else:
        label = "Mixed signal"
    return label, " ".join(lines)
