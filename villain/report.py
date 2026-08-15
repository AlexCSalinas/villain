"""Rendering profiles as something you can read between hands.

Plain text, fixed width, no dependencies -- the output has to survive being
read in a terminal next to a poker table. The ordering is deliberate: what to
do comes first, who they are second, and the numbers that justify it last.
Nobody mid-session needs a table of statistics before being told to stop
bluffing.
"""

from __future__ import annotations

from .analyze import enrich
from .archetypes import ARCHETYPE_BY_NAME, deviations
from .glossary import versus_behaviour
from .model import STREET_LABELS
from .playbook import combinations_for
from .profile import Profile

WIDTH = 78
RULE = "-" * WIDTH


def _bar(value: float, lo: float = 0.0, hi: float = 1.0, width: int = 18) -> str:
    filled = int(round(width * max(0.0, min(1.0, (value - lo) / (hi - lo)))))
    return "#" * filled + "." * (width - filled)


def profile_card(profile: Profile, verbose: bool = False) -> str:
    """The full read on one player in one regime."""
    enrich(profile)
    leaks = profile.tags
    arch = ARCHETYPE_BY_NAME.get(profile.archetype)

    out: list[str] = []
    out.append(RULE)
    out.append(f"{profile.name}  --  {profile.hands} hands ({profile.sample_quality})")
    if profile.contributions and len(profile.contributions) > 1:
        # Pooled from several table sizes, so say which and on whose terms.
        out.append(f"  {profile.table_mix}")
        out.append(f"  measured against {profile.regime_label} norms")
    else:
        out.append(f"  {profile.regime_label}")

    # 1. The plan.
    if arch:
        out.append(f"READ: {profile.archetype}  "
                   f"(confidence {profile.archetype_confidence:.0%})")
        out.append(f"  {arch.summary}")
        out.append("")
        for line in _wrap(arch.plan, WIDTH - 4):
            out.append(f"  {line}")
    if profile.archetype_mix and profile.archetype_mix[1][1] > 0.15:
        alt = ", ".join(f"{n} {s:.0%}" for n, s in profile.archetype_mix[1:3])
        out.append(f"  (also plausibly: {alt})")
    out.append("")

    # 2. What to attack, in order of what it is worth.
    out.append("EXPLOITS" + (f"  ({len(leaks)} found)" if leaks else ""))
    if not leaks:
        out.append("  Nothing clears the evidence bar yet. Play them straight and")
        out.append("  collect more hands.")
    for leak in leaks:
        out.append(f"  {leak.headline}   ~{leak.severity:.2f} bb/100 "
                   f"[{leak.size}, {leak.tier}]")
        for line in _wrap(leak.in_words, WIDTH - 6):
            out.append(f"      {line}")
        out.append("")
        for label, text in (("what", leak.behaviour), ("why", leak.why),
                            ("do", leak.do), ("don't", leak.dont)):
            if not text:
                continue
            wrapped = _wrap(text, WIDTH - 14)
            out.append(f"      {label:6s}  {wrapped[0]}")
            for line in wrapped[1:]:
                out.append(f"              {line}")
        out.append(f"      {'size':6s}  " + _wrap(leak.priority, WIDTH - 14)[0])
        for line in _wrap(leak.priority, WIDTH - 14)[1:]:
            out.append(f"              {line}")
        out.append("")

    from .exploits import find_watchlist
    from .skill import weaknesses

    watch = find_watchlist(profile)
    if watch:
        out.append("  not confirmed yet")
        for leak in watch:
            if leak.confirms_in is None:
                confirm = "may never confirm at this rate"
            else:
                confirm = f"confirms in ~{leak.confirms_in} more"
            out.append(f"    {leak.headline}  ({leak.confidence:.0%} sure, "
                       f"n={leak.opps:.0f}, {confirm})")
            for line in _wrap(leak.in_words, WIDTH - 6):
                out.append(f"      {line}")
        out.append("")

    weak = weaknesses(profile.skill)
    if weak and not leaks:
        out.append("  weakest parts of their game")
        out.append("  (no frequency clears the evidence bar, but this is where")
        out.append("   the rating says their game is thinnest)")
        for c in weak:
            note = f" -- {c.note}" if c.note else ""
            out.append(f"    {c.name} {c.score:.0f}/100{note}")
        out.append("")

    combos = combinations_for(l.id for l in leaks)
    if combos:
        out.append("  these compound")
        for combo in combos:
            out.append(f"    {combo.headline}")
            for line in _wrap(combo.body, WIDTH - 6):
                out.append(f"      {line}")
            out.append("")
    out.append("")

    out.extend(_against_you(profile, verbose))

    # 3. The rating and how it was reached.
    skill = profile.skill
    out.append(f"SKILL: {skill.label}   confidence {skill.confidence:.0%}")
    out.append(f"  {skill.blurb}")
    for c in sorted(skill.components, key=lambda c: c.score):
        note = f"  {c.note}" if c.note else ""
        out.append(f"    {c.name:26s} {_bar(c.score / 100)} {c.score:5.1f}{note}")
    if skill.winrate_bb100 is not None:
        line = f"  observed {skill.winrate_bb100:+.1f} bb/100"
        if skill.adjusted_bb100 is not None:
            line += f", shrunk and all-in adjusted {skill.adjusted_bb100:+.1f} bb/100"
        out.append(line)
    out.append("")

    # 4. The numbers, for anyone who wants to check the work.
    out.append("key numbers  (shrunk estimate, raw sample in brackets)")
    for stat, label in _HEADLINE_STATS:
        est = profile.stats.get(stat)
        if est is None:
            continue
        raw = f"{100 * est.raw:.0f}% of {est.opps:.0f}" if est.raw is not None else "no data"
        out.append(f"    {label:24s} {100 * est.value:5.1f}%   [{raw}]")

    if verbose:
        out.append("")
        out.append("deviations from the field  (in population spreads)")
        for feature, z in sorted(deviations(profile).items(), key=lambda kv: -abs(kv[1]))[:12]:
            out.append(f"    {feature:26s} {z:+.2f}")
        timing = profile.means.get("think:fold"), profile.means.get("think:aggro")
        if any(t is not None for t in timing):
            out.append("")
            out.append("timing")
            for key in ("think:fold", "think:call", "think:check", "think:aggro"):
                value = profile.means.get(key)
                if value:
                    out.append(f"    {key:26s} {value / 1000:5.1f}s "
                               f"over {profile.means.get(key + '#n', 0):.0f} actions")
    out.append(RULE)
    return "\n".join(out)


def _against_you(profile: Profile, verbose: bool) -> list[str]:
    """Where they play you differently from everybody else.

    Absent unless there is something to say. Most players against most
    opponents have no adjustment at all, and a section that appears every time
    to announce it costs a line of a card that has to fit on one screen. In
    verbose mode it says so, because there the reader asked for everything.
    """
    found = getattr(profile, "adjustments", None)
    if not found:
        return ["  no adjustment against you clears the bar yet", ""] if verbose else []

    out = [f"AGAINST YOU  ({len(found)} found)",
           "  Against their own game, not against the field.",
           ""]
    for a in found:
        # Capped at 99%: this comes out of a normal approximation to a Beta
        # posterior, which will happily print certainty it cannot have.
        sure = min(a.confidence, 0.99)
        otherwise = f"({a.baseline:.0%} otherwise)"
        out.append(f"    {versus_behaviour(a.stat):28s}{a.versus:5.0%}   "
                   f"{otherwise:16s} {a.opps:.0f} seen, {sure:.0%} sure")
    out.append("")
    return out


_HEADLINE_STATS = [
    ("vpip", "VPIP"), ("pfr", "PFR"), ("three_bet", "3-bet"),
    ("fold_to_three_bet", "fold to 3-bet"), ("cbet:flop", "c-bet flop"),
    ("fold_vs_bet:flop", "fold vs flop bet"), ("fold_vs_bet:turn", "fold vs turn bet"),
    ("fold_vs_bet:river", "fold vs river bet"), ("check_raise:flop", "check-raise flop"),
    ("wtsd", "went to showdown"), ("wsd", "won at showdown"),
]


def roster(profiles: list[Profile]) -> str:
    """One line per player: the table you scan before sitting down."""
    rows = [f"{'player':16s} {'table':10s} {'hands':>6s}  {'read':11s} {'skill':>10s}"
            f"  {'exploit':>8s}  top leak"]
    rows.append(RULE)
    for p in profiles:
        enrich(p)
        top = p.tags[0].headline if p.tags else "-"
        rows.append(
            f"{p.name[:16]:16s} {p.regime_label[:10]:10s} {p.hands:6d}  "
            f"{p.archetype:11s} {p.skill.score:5.0f}/100  "
            f"{p.skill.exploitability:6.1f}bb/100  {top[:28]}")
    return "\n".join(rows)


def hero_card(name: str, visibility: float, hands: int, ranges: dict, fold_report,
             missed_report, sizing, timing, narrowing: list) -> str:
    """Everything only hero's own hand history can show -- a counted preflop
    range, postflop folds and checks graded against what that line usually
    represents in this database, whether hero's own bet sizing or timing is
    a tell, and whether hero's continuing range actually narrows street by
    street. See :mod:`villain.hero`."""
    from .hero import POSITION_ORDER

    out: list[str] = [RULE, f"{name}  --  hero, cards known {visibility:.1%} of {hands} hands", ""]

    out.append("PREFLOP RANGE  (counted directly from every hand you were dealt, not modelled)")
    positions = sorted(ranges.values(), key=lambda p: POSITION_ORDER.get(p.position, 99))
    for pos in positions:
        if not pos.hands:
            continue
        raised, called, checked, folded = (100 * n / pos.hands for n in
            (pos.raised, pos.called, pos.checked, pos.folded))
        out.append(f"    {pos.position:5s} {pos.hands:4d} hands   "
                   f"raised {raised:4.0f}%  called {called:4.0f}%  "
                   f"checked {checked:4.0f}%  folded {folded:4.0f}%")
    out.append("")

    # Every number from here down is a percentile -- stated once here rather
    # than left for the reader to infer from a bare number.
    out.append("(Numbers below: percentiles -- the share of hands on that board yours beats.)")
    out.append("")

    out.append("FOLD GRADES  (postflop folds, graded against what a bet like that one usually is)")
    if not fold_report.graded:
        out.append("  Not enough postflop folds with a clean line to grade yet.")
    else:
        out.append(f"    {fold_report.graded} folds graded, {len(fold_report.mistakes)} "
                   f"({fold_report.mistake_rate:.1%}) had more edge than the bet typically shows")
        by_street = fold_report.by_street()
        if by_street:
            parts = [f"{STREET_LABELS.get(s, s)} {m / n:.1%}"
                     for s, (m, n) in sorted(by_street.items())]
            out.append(f"    by street:   {'   '.join(parts)}")
        by_texture = fold_report.by_texture()
        if by_texture:
            parts = [f"{t} {m / n:.1%}" for t, (m, n) in sorted(by_texture.items())]
            out.append(f"    by texture:  {'   '.join(parts)}")
        worst = fold_report.worst()
        if worst:
            out.append("    worst folds:")
            for i, g in enumerate(worst, 1):
                cards = "".join(g.hole_cards)
                out.append(f"      {i}. {STREET_LABELS.get(g.street, g.street):6s} {cards}  "
                          f"[hand {g.hand_id}]")
                for line in _wrap(g.in_words, WIDTH - 10):
                    out.append(f"         {line}")
    out.append("")

    out.append("MISSED VALUE  (postflop checks, graded against what a check like that usually is)")
    if not missed_report.graded:
        out.append("  Not enough postflop checks with a clean line to grade yet.")
    else:
        out.append(f"    {missed_report.graded} checks graded, {len(missed_report.missed)} "
                   f"({missed_report.missed_rate:.1%}) had more edge than the check typically shows")
        by_street = missed_report.by_street()
        if by_street:
            parts = [f"{STREET_LABELS.get(s, s)} {m / n:.1%}"
                     for s, (m, n) in sorted(by_street.items())]
            out.append(f"    by street:   {'   '.join(parts)}")
        by_texture = missed_report.by_texture()
        if by_texture:
            parts = [f"{t} {m / n:.1%}" for t, (m, n) in sorted(by_texture.items())]
            out.append(f"    by texture:  {'   '.join(parts)}")
        worst = missed_report.worst()
        if worst:
            out.append("    worst checks:")
            for i, g in enumerate(worst, 1):
                cards = "".join(g.hole_cards)
                out.append(f"      {i}. {STREET_LABELS.get(g.street, g.street):6s} {cards}  "
                          f"[hand {g.hand_id}]")
                for line in _wrap(g.in_words, WIDTH - 10):
                    out.append(f"         {line}")
    out.append("")

    out.append("SIZING TELL  (does your bet size change with the hand behind it?)")
    any_row = False
    for street in sorted(sizing.by_street):
        strong, weak = sizing.by_street[street]
        if not strong.hands and not weak.hands:
            continue
        any_row = True
        for line in _wrap(sizing.describe(street), WIDTH - 4):
            out.append(f"    {line}")
    if not any_row:
        out.append("  Not enough postflop bets or raises with a clean line to compare yet.")
    out.append("")

    out.append("TIMING TELL  (does your think time change with the hand behind it?)")
    any_row = False
    for street in sorted(timing.by_street):
        strong, weak = timing.by_street[street]
        if not strong.hands and not weak.hands:
            continue
        any_row = True
        for line in _wrap(timing.describe(street), WIDTH - 4):
            out.append(f"    {line}")
    if not any_row:
        out.append("  Not enough postflop bets or raises with a clean line to compare yet.")
    out.append("")

    out.append("RANGE NARROWING  (average hand strength among hands still live, by street)")
    if not narrowing:
        out.append("  Not enough hands reaching each street yet.")
    else:
        parts = [f"{STREET_LABELS.get(s.street, s.street)} {s.avg_strength:.0%} ({s.hands})"
                 for s in sorted(narrowing, key=lambda s: s.street)]
        out.append(f"    {'   '.join(parts)}")
        strengths = [s.avg_strength for s in sorted(narrowing, key=lambda s: s.street)]
        if len(strengths) >= 2 and all(b >= a for a, b in zip(strengths, strengths[1:])):
            out.append("    Narrows street by street, as a continuing range should.")
        elif len(strengths) >= 2:
            out.append("    Does not narrow monotonically -- worth a look at which street gives it back.")
    out.append(RULE)
    return "\n".join(out)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines
