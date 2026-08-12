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
    out.append(RULE)

    # 1. The plan.
    if arch:
        out.append(f"READ: {profile.archetype.upper()}  "
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
        for label, text in (("WHAT", leak.behaviour), ("WHY", leak.why),
                            ("DO", leak.do), ("DON'T", leak.dont)):
            if not text:
                continue
            wrapped = _wrap(text, WIDTH - 14)
            out.append(f"      {label:6s}  {wrapped[0]}")
            for line in wrapped[1:]:
                out.append(f"              {line}")
        out.append(f"      {'SIZE':6s}  " + _wrap(leak.priority, WIDTH - 14)[0])
        for line in _wrap(leak.priority, WIDTH - 14)[1:]:
            out.append(f"              {line}")
        out.append("")

    combos = combinations_for(l.id for l in leaks)
    if combos:
        out.append("  THESE COMPOUND")
        for combo in combos:
            out.append(f"    {combo.headline}")
            for line in _wrap(combo.body, WIDTH - 6):
                out.append(f"      {line}")
            out.append("")
    out.append("")

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
    out.append("KEY NUMBERS  (shrunk estimate, raw sample in brackets)")
    for stat, label in _HEADLINE_STATS:
        est = profile.stats.get(stat)
        if est is None:
            continue
        raw = f"{100 * est.raw:.0f}% of {est.opps:.0f}" if est.raw is not None else "no data"
        out.append(f"    {label:24s} {100 * est.value:5.1f}%   [{raw}]")

    if verbose:
        out.append("")
        out.append("DEVIATIONS FROM THE FIELD  (in population spreads)")
        for feature, z in sorted(deviations(profile).items(), key=lambda kv: -abs(kv[1]))[:12]:
            out.append(f"    {feature:26s} {z:+.2f}")
        timing = profile.means.get("think:fold"), profile.means.get("think:aggro")
        if any(t is not None for t in timing):
            out.append("")
            out.append("TIMING")
            for key in ("think:fold", "think:call", "think:check", "think:aggro"):
                value = profile.means.get(key)
                if value:
                    out.append(f"    {key:26s} {value / 1000:5.1f}s "
                               f"over {profile.means.get(key + '#n', 0):.0f} actions")
    out.append(RULE)
    return "\n".join(out)


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
            f"{p.skill.exploitability:6.1f}bb  {top[:28]}")
    return "\n".join(rows)


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
