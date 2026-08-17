"""Local web UI.

Runs on 127.0.0.1 using the standard library's HTTP server, so the project
picks up no new dependencies and the page is served inline from this one
module. Everything it shows comes from the same functions the CLI uses -- the
UI is a view over :mod:`villain.analyze`, never a second opinion about what a
player is.

There are two tabs because there are two questions. *Session* answers "who am
I playing right now" from a file you just dropped in, and touches no database
at all -- a session you never save leaves no trace. *Database* answers "who is
this, and what do I know about them" across every session ever imported.

Saving a session is a separate, optional step, and it is the only moment the
tool asks questions. Identity is where a profiler quietly destroys its own
data: merge two people and both profiles become fiction, split one person and
you throw away half of what you know. So the save flow surfaces every identity
decision it is unsure about and takes an answer, rather than guessing and
being confidently wrong for the next thousand hands.

The visual grammar is deliberately narrow. Every read in this tool is a
frequency with an uncertainty and a reference point, so almost every chart is
the same mark: an interval band, a dot for the estimate, and a tick for the
threshold that makes it matter. One neutral ramp throughout, stepped by
confidence; the single warm tick marks breakeven, the one number that decides
whether a deviation is worth money.
"""

from __future__ import annotations

import gzip
import json
import secrets
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .analyze import as_dict, enrich
from .archetypes import ARCHETYPE_BY_NAME, deviations
from .db import DEFAULT_PATH, Store, split_key
from .dynamics import adjustments
from .exploits import RULES, find_watchlist
from .features import record_hands
from .evidence import find as find_evidence
from .glossary import payload as glossary_payload, stat_help
from .model import hand_from_dict, hand_to_dict
from .identity import askable_questions, auto_answers, session_questions, suggest_links
from .skill import weaknesses
from .narrate import Unavailable, enabled as narrator_enabled, narrate
from .parsers import UnknownFormat, parse_file
from .priors import population_mean
from .profile import build_profiles, build_unified, primary_regime
from .stats import VS_HERO
from .timing import timing_tells
from .replay import replay

# Stats worth a row in the profile view, in reading order.
DISPLAY_STATS = [
    ("vpip", "VPIP", "hands played"),
    ("pfr", "PFR", "hands raised preflop"),
    ("three_bet", "3-bet", "raises facing a raise"),
    ("fold_to_three_bet", "fold to 3-bet", "after opening"),
    ("four_bet", "4-bet", "facing a 3-bet"),
    ("five_bet", "5-bet", "facing a 4-bet"),
    ("squeeze", "squeeze", "after a raise and a caller"),
    ("cold_call", "cold call", "calls a raise, no money in"),
    ("rfi", "open (RFI)", "first in, folded to them"),
    ("bb_defend", "BB defence", "big blind vs a raise"),
    ("cbet:flop", "c-bet flop", "as the preflop raiser"),
    ("cbet:turn", "c-bet turn", "after betting the flop"),
    ("fold_vs_bet:flop", "fold vs flop bet", "facing a bet"),
    ("fold_vs_bet:turn", "fold vs turn bet", "facing a bet"),
    ("fold_vs_bet:river", "fold vs river bet", "facing a bet"),
    ("check_raise:flop", "check-raise flop", "after checking"),
    ("wtsd", "went to showdown", "after seeing the flop"),
    ("wsd", "won at showdown", "of showdowns reached"),
    ("aggression:flop", "flop aggression", "bets+raises of all actions"),
    ("aggression:turn", "turn aggression", "bets+raises of all actions"),
    ("tank_fold", "tank-fold", "folds after a long pause"),
    ("tank_fold:flop", "tank-fold flop", "flop folds after a long pause"),
    ("tank_fold:turn", "tank-fold turn", "turn folds after a long pause"),
    ("tank_fold:river", "tank-fold river", "river folds after a long pause"),
    ("snap_call", "snap-call", "calls made instantly"),
    ("snap_call:flop", "snap-call flop", "flop calls made instantly"),
    ("snap_call:turn", "snap-call turn", "turn calls made instantly"),
    ("snap_call:river", "snap-call river", "river calls made instantly"),
]

# Stats whose exploit threshold is worth drawing as a second reference tick.
_THRESHOLD_RULES = {rule.stat: rule for rule in RULES}


def _references(stat: str, regime: str, profile) -> dict:
    """Population frequency and, where one exists, the breakeven threshold."""
    out = {"population": round(population_mean(stat, regime), 4)}
    rule = _THRESHOLD_RULES.get(stat)
    if rule is not None:
        try:
            out["breakeven"] = round(rule.threshold(profile), 4)
            out["breakeven_label"] = ("bluff breaks even"
                                      if stat.startswith(("fold_vs_bet", "fold_to_cbet"))
                                      else "exploit threshold")
        except Exception:
            pass
    return out


def profile_payload(profile, player_id: int | None = None) -> dict:
    """``as_dict`` plus the reference points the charts need to be readable."""
    enrich(profile)
    payload = as_dict(profile)
    # Carried so the UI can link a read back to the hands behind it. Absent for
    # an unsaved session, whose hands are not in the database to look up.
    payload["player_id"] = player_id
    payload["rows"] = []
    for stat, label, denominator in DISPLAY_STATS:
        est = profile.stats.get(stat)
        if est is None or est.opps <= 0:
            continue
        payload["rows"].append({
            "stat": stat, "label": label, "denominator": denominator,
            "value": round(est.value, 4), "lo": round(est.lo, 4), "hi": round(est.hi, 4),
            "raw": None if est.raw is None else round(est.raw, 4),
            # Opportunity counts are fractional inside the model (pooling
            # across table sizes), but a sample size rendered as
            # 92.86041666666667 is noise on screen.
            "opps": round(est.opps, 1), "weight": round(est.weight, 3),
            **_references(stat, profile.regime, profile),
        })
    arch = ARCHETYPE_BY_NAME.get(profile.archetype)
    payload["plan"] = arch.plan if arch else ""
    payload["summary"] = arch.summary if arch else ""
    payload["regime_label"] = profile.regime_label
    payload["deviations"] = [
        {"feature": f, "z": round(z, 2)}
        for f, z in sorted(deviations(profile).items(), key=lambda kv: -abs(kv[1]))[:10]
    ]
    payload["timing"] = {
        key.split(":", 1)[1]: {"seconds": round(profile.means[key] / 1000, 2),
                               "n": int(profile.means.get(f"{key}#n", 0) or 0)}
        for key in ("think:fold", "think:call", "think:check", "think:aggro",
                    "think:pf", "think:flop", "think:turn", "think:river")
        if profile.means.get(key)
    }
    payload["timing_tells"] = [
        {"pace": c.pace, "street": c.street, "action": c.action,
         "action_label": c.action_label, "n": c.n, "total": c.total,
         "share": None if c.share is None else round(c.share, 3),
         "won": None if c.won is None else round(c.won, 3),
         "won_base": None if c.won_base is None else round(c.won_base, 3),
         "wtsd": None if c.wtsd is None else round(c.wtsd, 3),
         "wtsd_base": None if c.wtsd_base is None else round(c.wtsd_base, 3),
         "fold_next": None if c.fold_next is None else round(c.fold_next, 3),
         "fold_next_base": None if c.fold_next_base is None else round(c.fold_next_base, 3),
         "fold_next_n": c.fold_next_n,
         "sd_strength": None if c.sd_strength is None else round(c.sd_strength, 3),
         "sd_base": None if c.sd_base is None else round(c.sd_base, 3),
         "sd_n": c.sd_n,
         "label": c.label, "read": c.read}
        for c in timing_tells(profile)
    ]
    return payload


#: A book this small is a rounding error, not a profile -- somebody who sat
#: down for one hand at a different table size should not get their own row.
MIN_ROSTER_HANDS = 5


def roster_payload(store: Store) -> list[dict]:
    """One row per player. Table sizes are pooled, not listed separately."""
    rows = []
    for player in store.players():
        profile = store.profile(int(player["id"]))
        if profile is not None:
            enrich(profile)
            top = profile.tags[0] if profile.tags else None
            # Fall back through what is known: a priced leak, then an
            # unconfirmed one, then the weakest rated part of their game.
            # "None clears the bar" is true and useless -- it leaves the
            # weakest player on the table looking like the safest.
            headline, status, note = None, None, ""
            if top is not None:
                headline, status = top.headline, "confirmed"
                note = f"{top.severity:.2f} bb/100, {top.tier} read"
            else:
                watch = find_watchlist(profile)
                if watch:
                    headline, status = watch[0].headline, "watch"
                    note = (f"{watch[0].confidence:.0%} sure over "
                            f"{watch[0].opps:.0f} spots -- not confirmed")
                else:
                    weak = weaknesses(profile.skill)
                    if weak:
                        headline, status = weak[0].name, "rated"
                        note = (f"scores {weak[0].score:.0f}/100 here"
                                + (f" ({weak[0].note})" if weak[0].note else "")
                                + " -- from the rating, not a measured frequency")
            rows.append({
                "player_id": int(player["id"]),
                "name": profile.name or player["display_name"],
                "aliases": player["aliases"],
                "regime": profile.regime,
                "regime_label": profile.regime_label,
                "table_mix": profile.table_mix,
                "hands": profile.hands,
                "sample_quality": profile.sample_quality,
                "archetype": profile.archetype,
                "confidence": profile.archetype_confidence,
                "skill": profile.skill.score,
                "skill_tier": profile.skill.tier,
                "skill_confidence": profile.skill.confidence,
                "exploitability": profile.skill.exploitability,
                "top_leak": headline,
                "top_leak_status": status,
                "top_leak_note": note,
                "top_leak_severity": round(top.severity, 2) if top else 0.0,
                "leak_count": len(profile.tags),
                "last_seen": profile.last_seen,
            })
    rows.sort(key=lambda r: (-r["hands"],))
    return rows


# ---------------------------------------------------------------------------
# uploaded sessions, held in memory until saved
# ---------------------------------------------------------------------------
# A session is deliberately not written anywhere. You can drop a file in, read
# the table, and close the tab without the database gaining a single hand.

#: Hostnames the UI may be reached on. Anything else is a rebinding attempt.
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", ""})

#: Cap on a request body. The upload route holds what it reads in memory.
MAX_BODY_BYTES = 64 * 1024 * 1024

SESSIONS: dict[str, dict] = {}
SESSION_TTL = 6 * 3600
MAX_SESSIONS = 12


def _reap_sessions() -> None:
    now = time.time()
    stale = [k for k, v in SESSIONS.items() if now - v["created"] > SESSION_TTL]
    for key in stale:
        SESSIONS.pop(key, None)
    while len(SESSIONS) > MAX_SESSIONS:
        oldest = min(SESSIONS, key=lambda k: SESSIONS[k]["created"])
        SESSIONS.pop(oldest, None)


def parse_upload(filename: str, content: str):
    """Parse an uploaded file by writing it somewhere a parser can sniff it.

    The parser registry works off file paths so it can identify a format from
    the extension and the first few bytes; a temporary file keeps that contract
    intact rather than adding a second, divergent code path for uploads.
    """
    suffix = Path(filename).suffix or ".json"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as fh:
        fh.write(content)
        temp = Path(fh.name)
    try:
        return parse_file(temp)
    finally:
        temp.unlink(missing_ok=True)


def database_merges(store: Store, hands: list) -> dict:
    """Accounts this session shares that the database already calls one player.

    So a merge made anywhere shows up everywhere. Answering at upload and
    merging later from the suggestions panel are the same decision, and a
    session that ignored the second would contradict the database it was about
    to be saved into.
    """
    alias = {(r["site"], r["account"]): (int(r["player_id"]), r["name"])
             for r in store.conn.execute(
                 "SELECT site, account, player_id, name FROM aliases")}
    names = {int(r["id"]): r["display_name"] for r in store.players()}

    seen: dict[tuple[str, str], int] = {}
    counts: dict[tuple[str, str], int] = {}
    for hand in hands:
        for seat in hand.seats:
            key = (hand.site, seat.player_id)
            counts[key] = counts.get(key, 0) + 1
            hit = alias.get(key) or alias.get((hand.site, split_key(seat.player_id, seat.name)))
            if hit:
                seen[key] = hit[0]

    grouped: dict[int, list] = {}
    for key, player_id in seen.items():
        grouped.setdefault(player_id, []).append(key)

    merges = {}
    for player_id, keys in grouped.items():
        if len(keys) < 2:
            continue
        target = max(keys, key=lambda k: counts.get(k, 0))
        for key in keys:
            merges[key] = {"account": target[1],
                           "name": names.get(player_id, target[1])}
    return merges


def merged_hands(session: dict, extra: dict | None = None) -> list:
    """The session's hands with confirmed same-person accounts pooled.

    Applied to a copy. The stored hands must keep the account ids the site
    actually wrote, because identity is a decision layered on top of them and
    decisions get revised; the hands themselves are evidence and do not.
    """
    merges = dict(session.get("merges") or {})
    merges.update(extra or {})
    if not merges:
        return session["hands"]
    hands = [hand_from_dict(hand_to_dict(h)) for h in session["hands"]]
    for hand in hands:
        for seat in hand.seats:
            target = merges.get((hand.site, seat.player_id))
            if target:
                seat.player_id, seat.name = target["account"], target["name"]
    return hands


def session_identity_labels(session: dict) -> dict[str, dict]:
    """For each pooled display name, session aliases and a database name if linked.

    After auto-merge the roster title is often already the database name; the
    muted line still needs the other side so you can see what is merging with
    what.
    """
    answers = session.get("answers") or {}
    by_keep: dict[str, dict] = {}
    for question in session.get("questions") or []:
        answer = answers.get(question.id)
        if not answer or not answer.get("same"):
            continue
        keep = answer.get("name") or question.default_name
        if not keep:
            continue
        entry = by_keep.setdefault(keep, {"db_name": None, "session_names": []})
        sides = [s for s in (question.left, question.right) if s]
        db_side = next((s for s in sides
                        if "database" in (s.get("where") or "")), None)
        if db_side and db_side.get("name"):
            entry["db_name"] = db_side["name"]
        for side in sides:
            name = side.get("name")
            where = side.get("where") or ""
            if name and "session" in where and name not in entry["session_names"]:
                entry["session_names"].append(name)
    return by_keep


def session_payload(token: str, store: Store | None = None) -> dict:
    """Profiles for an uploaded session. Reads the store, never writes to it."""
    session = SESSIONS[token]
    extra = database_merges(store, session["hands"]) if store is not None else None
    books = record_hands(merged_hands(session, extra))

    def _unified(by_regime):
        # Same shrink as database profiles when this pool has fitted priors.
        priors = None
        if store is not None and by_regime:
            fitted = store.fitted_priors(primary_regime(by_regime))
            priors = fitted or None
        profile = build_unified(by_regime, priors=priors)
        if profile is not None:
            # Store.profile attaches these for saved players; an uploaded
            # session has no store to do it, and a preview that silently drops
            # a section the same hands produce once saved is a preview of
            # something else.
            profile.adjustments = adjustments(by_regime, priors=priors)
        return profile

    profiles = [p for p in (_unified(by_regime) for by_regime in books.values())
                if p is not None]
    profiles.sort(key=lambda p: -p.hands)

    labels = session_identity_labels(session)
    # Also surface database display names from already-linked aliases when the
    # session did not need a question (same account id, same name).
    if store is not None:
        alias_names = {
            (r["site"], r["account"]): r["name"]
            for r in store.conn.execute(
                "SELECT site, account, name FROM aliases")
        }
        player_names = {int(r["id"]): r["display_name"] for r in store.players()}
        alias_player = {
            (r["site"], r["account"]): int(r["player_id"])
            for r in store.conn.execute(
                "SELECT site, account, player_id FROM aliases")
        }
        for hand in session["hands"]:
            for seat in hand.seats:
                key = (hand.site, seat.player_id)
                pid = alias_player.get(key) or alias_player.get(
                    (hand.site, split_key(seat.player_id, seat.name)))
                if pid is None:
                    continue
                db_name = player_names.get(pid) or alias_names.get(key)
                if not db_name:
                    continue
                # Profile name after merge is the keep/db name.
                keep = db_name
                entry = labels.setdefault(
                    keep, {"db_name": None, "session_names": []})
                entry["db_name"] = db_name
                if seat.name and seat.name not in entry["session_names"]:
                    entry["session_names"].append(seat.name)

    rows = []
    profile_payloads = []
    for profile in profiles:
        enrich(profile)
        top = profile.tags[0] if profile.tags else None
        link = labels.get(profile.name) or {}
        db_name = link.get("db_name")
        session_names = [n for n in (link.get("session_names") or [])
                         if n and n != profile.name]
        if db_name and db_name != profile.name and profile.name not in session_names:
            session_names = [profile.name] + session_names
        row = {
            "player_id": None, "name": profile.name,
            "db_name": db_name if db_name else None,
            "session_names": session_names,
            "regime": profile.regime, "regime_label": profile.regime_label,
            "table_mix": profile.table_mix,
            "hands": profile.hands, "sample_quality": profile.sample_quality,
            "archetype": profile.archetype, "confidence": profile.archetype_confidence,
            "skill": profile.skill.score, "skill_tier": profile.skill.tier,
            "skill_confidence": profile.skill.confidence,
            "exploitability": profile.skill.exploitability,
            "top_leak": top.headline if top else None,
            "leak_count": len(profile.tags),
        }
        rows.append(row)
        pp = profile_payload(profile)
        pp["db_name"] = row["db_name"]
        pp["session_names"] = row["session_names"]
        profile_payloads.append(pp)
    return {
        "token": token,
        "files": session["files"],
        "hands": len(session["hands"]),
        "players": rows,
        "profiles": profile_payloads,
        "saved": session.get("saved", False),
        "questions": [question_payload(q) for q in askable_questions(
            session.get("questions") or [])],
        "answered": bool(session.get("answers")),
        "auto_merged": len(auto_answers(session.get("questions") or [])),
        "merges": [{"from": k[1], "to": v["name"]}
                   for k, v in (session.get("merges") or {}).items()],
    }


def question_payload(question) -> dict:
    return {
        "id": question.id, "kind": question.kind, "prompt": question.prompt,
        "detail": question.detail, "default": question.default,
        "confidence": question.confidence, "left": question.left, "right": question.right,
        "names": question.names, "default_name": question.default_name,
        "auto": question.auto,
    }


def apply_answers(session: dict, answers: dict) -> None:
    """Record identity decisions on a session and pool the merged accounts.

    Asked at upload rather than at save, so the session you are reading has
    already combined them. One player split across two names halves both
    samples exactly when sample size is the scarce thing. New answers merge
    onto any auto-applied ones rather than replacing them.

    Pairs that sat together more than a reconnect glitch are never pooled
    here — ``commit_session`` would refuse the link, and showing a merged
    profile the save step cannot keep is worse than leaving them apart.
    """
    from .db import SPURIOUS_OVERLAP
    from .identity import _incoming_co_occurrence

    merged_answers = dict(session.get("answers") or {})
    merged_answers.update(answers or {})
    session["answers"] = merged_answers
    blocked = _incoming_co_occurrence(session.get("hands") or [])
    merges: dict[tuple[str, str], dict] = {}
    for question in session.get("questions", []):
        answer = merged_answers.get(question.id) or {}
        if not answer.get("same"):
            continue
        keep_name = answer.get("name") or question.default_name
        sides = [side for side in (question.left, question.right) if side.get("account")]
        if not sides:
            continue
        keys = [(side["site"], side["account"]) for side in sides]
        overlap = 0
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                overlap = max(overlap, blocked.get(frozenset((a, b)), 0))
        if overlap > SPURIOUS_OVERLAP:
            continue
        # Everything folds onto the busiest account present in this session.
        target = max(sides, key=lambda side: side.get("hands", 0))
        for side in sides:
            merges[(side["site"], side["account"])] = {
                "account": target["account"], "name": keep_name}
    session["merges"] = merges


def commit_session(store: Store, token: str, answers: dict) -> dict:
    """Save an uploaded session, applying the identity answers given.

    Order matters. Hands are stored first so that every account exists as a
    player, then declined renames are re-keyed, then accepted aliases are
    merged. Doing the merges first would mean linking players that do not exist
    yet.
    """
    session = SESSIONS[token]
    questions = {q.id: q for q in session.get("questions", [])}
    answers = answers or session.get("answers") or {}

    def said_same(qid: str, question) -> bool:
        answer = answers.get(qid)
        if isinstance(answer, dict):
            return bool(answer.get("same"))
        if isinstance(answer, bool):
            return answer
        return question.default

    def chosen_name(qid: str, question) -> str:
        answer = answers.get(qid)
        if isinstance(answer, dict) and answer.get("name"):
            return answer["name"]
        return question.default_name

    name_splits = set()
    for qid, question in questions.items():
        if question.kind != "rename" or said_same(qid, question):
            continue
        right = question.right
        name_splits.add((right["site"], right["account"], right["name"]))

    report = store.add_hands(session["hands"], name_splits=name_splits)
    # Refit the population from the pool itself. This used to be a button, but
    # it is not a preference: measuring a home game against a generic online
    # population makes every deviation wrong by the gap between the two, and
    # the fit already refuses (8+ players, 5+ opportunities per stat) when the
    # data cannot support it. Announced rather than silent, because it moves
    # the reference point every read is measured from.
    priors_fitted = None
    fitted = store.fit_priors()
    if fitted:
        players = store.conn.execute(
            "SELECT COUNT(DISTINCT player_id) c FROM books").fetchone()["c"]
        priors_fitted = {"regimes": fitted, "players": players}
        # No rebuild here. Books are counts; the fitted prior is applied when a
        # profile is *read*, so refitting takes effect immediately. Rebuilding
        # recomputed every player from every stored hand for nothing, which on
        # a 12,000-hand database is most of a minute with the window blocked.

    # A confirmed rename means the player is now known by the new name; the old
    # one stays reachable as an alias.
    for qid, question in questions.items():
        if question.kind != "rename" or not said_same(qid, question):
            continue
        player_id = _player_id_of(store, question.right)
        if player_id is not None:
            store.conn.execute("UPDATE players SET display_name = ? WHERE id = ?",
                               (chosen_name(qid, question), player_id))
    store.conn.commit()

    merged, blocked = 0, []
    for qid, question in questions.items():
        if question.kind != "alias" or not said_same(qid, question):
            continue
        try:
            keep = _player_id_of(store, question.left)
            absorb = _player_id_of(store, question.right)
        except LookupError:
            continue
        if keep is None or absorb is None or keep == absorb:
            continue
        try:
            store.link(keep, absorb)
            store.conn.execute("UPDATE players SET display_name = ? WHERE id = ?",
                               (chosen_name(qid, question), keep))
            store.conn.commit()
            merged += 1
        except ValueError as exc:
            blocked.append(str(exc))

    session["saved"] = True
    return {
        "hands_new": report.hands_new, "duplicates": report.duplicates,
        "priors_fitted": priors_fitted,
        # Surfaced so a batch that silently stored unreadable hands says so.
        "unusable": report.unusable,
        "players_new": report.players_new, "merged": merged, "blocked": blocked,
    }


def _player_id_of(store: Store, side: dict) -> int | None:
    """Resolve one side of an alias question to an internal player id."""
    if side.get("player_id"):
        return int(side["player_id"])
    row = store.conn.execute(
        "SELECT player_id FROM aliases WHERE site = ? AND account = ?",
        (side.get("site"), side.get("account"))).fetchone()
    return int(row["player_id"]) if row else None


def leaderboard_payload(store: Store) -> dict:
    """Every known player, ranked.

    Two orderings matter and they are not the same question. Skill answers
    "who is dangerous"; attackable bb/100 answers "who is worth sitting with".
    A competent player with one exploitable habit can be worth more to you than
    a weak player you have barely seen, so both are shown and the table sorts
    on either.
    """
    ranked = roster_payload(store)
    return {"players": sorted(ranked, key=lambda r: -r["skill"])}


# ---------------------------------------------------------------------------
# hero: what only your own hand history can show
# ---------------------------------------------------------------------------
# Grading every fold means fitting the population hand-strength model first
# (villain.reads.fit) and walking hero's several thousand hands through the
# 7-card evaluator -- tens of seconds on a database this size, and unchanged
# from one request to the next unless new hands were imported. An in-memory
# cache alone only pays that once *per running server*, and this UI gets
# stopped and restarted often -- so the finished payload (JSON-safe: no
# sklearn object in it) is also persisted next to the database, keyed by hand
# count rather than time, the same as the in-memory layer. The model itself
# is cheap to refit inside one process and expensive to pickle safely across
# versions, so only the memory layer holds it.
_HERO_MODEL_CACHE: dict[str, tuple[int, object]] = {}
_HERO_PAYLOAD_CACHE: dict[tuple[str, int | None], tuple[int, dict | None]] = {}
#: The server handles requests on their own thread, so two Hero tab loads
#: landing close together used to each start their own fit -- the actual
#: incident this guards against: two ~40s fits running at once pegged every
#: core for minutes and starved every other tab's requests, not just Hero's.
#: One lock serialises fitting; the cache re-check after acquiring it means
#: the second request pays nothing once the first finishes.
_HERO_LOCK = threading.Lock()


def _hand_count(store: Store) -> int:
    return store.conn.execute("SELECT COUNT(*) c FROM hands").fetchone()["c"]


#: find_hero() itself is cheap (no model fit, just a scan of every seat), but
#: the roster loads on every visit to the Database tab, so it is cached the
#: same way -- by hand count -- rather than re-scanning every hand each time.
_HERO_ID_CACHE: dict[str, tuple[int, int | None]] = {}


def _cached_hero_id(store: Store) -> int | None:
    from .hero import find_hero

    key = str(store.path)
    hand_count = _hand_count(store)
    cached = _HERO_ID_CACHE.get(key)
    if cached and cached[0] == hand_count:
        return cached[1]
    hero_id = find_hero(store)
    _HERO_ID_CACHE[key] = (hand_count, hero_id)
    return hero_id


def _hero_model(store: Store):
    from .hero import fit_population_model

    key = str(store.path)
    hand_count = _hand_count(store)
    cached = _HERO_MODEL_CACHE.get(key)
    if cached and cached[0] == hand_count:
        return cached[1]
    model = fit_population_model(store)
    _HERO_MODEL_CACHE[key] = (hand_count, model)
    return model


#: Bump whenever _build_hero_payload's returned shape changes, so an old
#: cache file from a previous version of this module is a miss rather than a
#: served-stale response with fields the current frontend does not expect.
_HERO_CACHE_VERSION = 5


def _hero_disk_cache_path(store: Store) -> Path:
    return store.path.with_name(store.path.name + ".hero-cache.json")


def _hero_disk_cache_load(store: Store, hero_id: int | None,
                          hand_count: int) -> tuple[bool, dict | None]:
    """(hit, payload). ``hit`` is separate from ``payload`` because a cached
    "no hero found" answer is a legitimate ``None`` that should not trigger
    a recompute -- only a genuine cache miss should."""
    path = _hero_disk_cache_path(store)
    if not path.exists():
        return False, None
    try:
        saved = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, None
    entry = saved.get(str(hero_id))
    if (entry and entry.get("hand_count") == hand_count
            and entry.get("version") == _HERO_CACHE_VERSION):
        return True, entry.get("payload")
    return False, None


def _hero_disk_cache_save(store: Store, hero_id: int | None, hand_count: int,
                          payload: dict | None) -> None:
    path = _hero_disk_cache_path(store)
    try:
        saved = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        saved = {}
    saved[str(hero_id)] = {
        "hand_count": hand_count, "version": _HERO_CACHE_VERSION, "payload": payload,
    }
    try:
        path.write_text(json.dumps(saved))
    except OSError:
        pass    # a stale/missing cache costs time next request, not correctness


def hero_payload(store: Store, hero_id: int | None = None) -> dict | None:
    key = (str(store.path), hero_id)
    hand_count = _hand_count(store)

    cached = _HERO_PAYLOAD_CACHE.get(key)
    if cached and cached[0] == hand_count:
        return cached[1]

    with _HERO_LOCK:
        # Re-check both caches: another thread may have finished this exact
        # computation while this one was waiting for the lock.
        cached = _HERO_PAYLOAD_CACHE.get(key)
        if cached and cached[0] == hand_count:
            return cached[1]
        hit, payload = _hero_disk_cache_load(store, hero_id, hand_count)
        if not hit:
            payload = _build_hero_payload(store, hero_id)
            _hero_disk_cache_save(store, hero_id, hand_count, payload)
        _HERO_PAYLOAD_CACHE[key] = (hand_count, payload)
        return payload


def _hero_self(store: Store, hero_id: int) -> dict | None:
    """Hero read through the ordinary villain machinery. Hero is just another
    player in the database, so their own StatBook yields a Profile with priced
    leaks and headline numbers exactly like a villain's -- the difference is
    only that the UI frames them for self-coaching. Never priced against a
    villain range; these are deviations from the population you were measured
    against."""
    try:
        prof = store.profile(hero_id)
    except Exception:
        return None
    if prof is None:
        return None
    return profile_payload(prof, hero_id)


def _build_hero_payload(store: Store, hero_id: int | None) -> dict | None:
    from .hero import (NotEnoughData, combined_grid, fold_grades, hero_visibility,
                       missed_value, preflop_range, range_narrowing, sizing_tell,
                       timing_tell)
    from .model import STREET_LABELS

    if hero_id is None:
        hero_id = _cached_hero_id(store)
    if hero_id is None:
        return None
    row = next((r for r in store.players() if int(r["id"]) == hero_id), None)
    if row is None:
        return None

    hero_hands = store.player_hands(hero_id)
    ranges = preflop_range(hero_hands, hero_id)
    seen, total = hero_visibility(hero_hands, hero_id)
    sizing = sizing_tell(hero_hands, hero_id)
    timing = timing_tell(hero_hands, hero_id)
    narrowing = range_narrowing(hero_hands, hero_id)

    try:
        model = _hero_model(store)
        report = fold_grades(hero_hands, hero_id, model)
        missed_report = missed_value(hero_hands, hero_id, model)
        grade_error = None
    except NotEnoughData as exc:
        report = missed_report = None
        grade_error = str(exc)

    def _fold_json(g):
        return {"hand_id": g.hand_id, "street": STREET_LABELS.get(g.street, g.street),
               "hole_cards": list(g.hole_cards), "board": g.board, "texture": g.texture,
               "summary": g.summary, "in_words": g.in_words}

    def _bucketed_json(report, mistakes_attr, rate_attr):
        return {
            "graded": report.graded, "flagged": len(getattr(report, mistakes_attr)),
            "rate": getattr(report, rate_attr),
            "by_street": {STREET_LABELS.get(s, s): {"flagged": m, "graded": n}
                         for s, (m, n) in sorted(report.by_street().items())},
            "by_texture": {t: {"flagged": m, "graded": n}
                          for t, (m, n) in sorted(report.by_texture().items())},
            "worst": [_fold_json(g) for g in report.worst()],
        }

    def _tell_json(tell, avg_attr):
        tell_streets = {s for s, _ in tell.tells()}
        return {
            STREET_LABELS.get(street, street): {
                "strong": {"hands": strong.hands, "avg": getattr(strong, avg_attr)},
                "weak": {"hands": weak.hands, "avg": getattr(weak, avg_attr)},
                "in_words": tell.describe(street, lead=False),
                "is_tell": street in tell_streets,
            }
            for street, (strong, weak) in sorted(tell.by_street.items())
            if strong.hands or weak.hands
        }

    return {
        "hero_id": hero_id, "name": row["display_name"],
        "visibility": round(seen / total, 4) if total else 0.0, "hands": row["hands"] or 0,
        "ranges": [
            {"position": p.position, "hands": p.hands, "raised": p.raised,
             "called": p.called, "checked": p.checked, "folded": p.folded}
            for p in ranges.values()
        ],
        "grid": {cls: {"played": played, "dealt": dealt}
                for cls, (played, dealt) in combined_grid(ranges).items()},
        "fold_grades": None if report is None else _bucketed_json(
            report, "mistakes", "mistake_rate"),
        "missed_value": None if missed_report is None else _bucketed_json(
            missed_report, "missed", "missed_rate"),
        "grade_error": grade_error,
        "sizing": _tell_json(sizing, "avg_size"),
        "timing": _tell_json(timing, "avg_think_s"),
        "narrowing": [
            {"street": STREET_LABELS.get(s.street, s.street), "hands": s.hands,
             "avg_strength": round(s.avg_strength, 4)}
            for s in sorted(narrowing, key=lambda s: s.street)
        ],
        "self": _hero_self(store, hero_id),
    }


class Handler(BaseHTTPRequestHandler):
    db_path = DEFAULT_PATH

    def log_message(self, *args):
        pass                      # the terminal belongs to the user, not the server

    def _send(self, code: int, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path)
        path = route.path
        try:
            if path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/sessions":
                with Store(self.db_path) as store:
                    out = []
                    for sess in store.sessions():
                        out.append({k: v for k, v in sess.items() if k != "hand_ids"})
                    return self._send(200, out)
            if path.startswith("/api/session-detail"):
                sid = int(parse_qs(urlparse(self.path).query).get("id", ["0"])[0])
                with Store(self.db_path) as store:
                    match = next((x for x in store.sessions() if x["id"] == sid), None)
                    if match is None:
                        return self._send(404, {"error": "no such session"})
                    return self._send(200, {
                        "id": match["id"], "started_at": match["started_at"],
                        "ended_at": match["ended_at"], "hands": match["hands"],
                        "players": store.session_detail(match)})
            if path == "/api/roster":
                with Store(self.db_path) as store:
                    n_players = store.conn.execute(
                        "SELECT COUNT(*) c FROM players").fetchone()["c"]
                    n_fitted = store.conn.execute(
                        "SELECT COUNT(*) c FROM fitted_priors").fetchone()["c"]
                    return self._send(200, {
                        "players": roster_payload(store),
                        "db": str(self.db_path),
                        "hands": store.conn.execute(
                            "SELECT COUNT(*) c FROM hands").fetchone()["c"],
                        "hero_id": _cached_hero_id(store),
                        "fit_priors": {
                            "suggested": n_players >= 8 and n_fitted == 0,
                            "players": n_players,
                            "has_fitted": n_fitted > 0,
                        },
                    })
            if path.startswith("/api/player/"):
                player_id = int(path.rsplit("/", 1)[1])
                with Store(self.db_path) as store:
                    row = store.conn.execute(
                        "SELECT display_name FROM players WHERE id = ?",
                        (player_id,)).fetchone()
                    if row is None:
                        return self._send(404, {"error": "no such player"})
                    unified = store.profile(player_id)
                    profiles = [profile_payload(unified, player_id)] if unified else []
                    # The per-table breakdown stays available for anyone who
                    # wants to check that the pooling is not hiding something.
                    by_table = [profile_payload(p)
                                for p in store.profiles(player_id,
                                                        min_hands=MIN_ROSTER_HANDS)]
                    aliases = [dict(r) for r in store.conn.execute(
                        "SELECT site, account, name, hands FROM aliases WHERE player_id = ?",
                        (player_id,))]
                    return self._send(200, {
                        "player_id": player_id,
                        "display_name": row["display_name"],
                        "aliases": aliases,
                        "profiles": profiles,
                        "by_table": by_table if len(by_table) > 1 else [],
                        "notes": [dict(n) for n in store.notes(player_id)],
                        "hero_id": _cached_hero_id(store),
                    })
            if path.startswith("/api/session/"):
                token = path.rsplit("/", 1)[1]
                if token not in SESSIONS:
                    return self._send(404, {"error": "session expired -- upload again"})
                with Store(self.db_path) as store:
                    return self._send(200, session_payload(token, store))
            if path == "/api/leaderboard":
                with Store(self.db_path) as store:
                    return self._send(200, leaderboard_payload(store))
            if path == "/api/hero":
                with Store(self.db_path) as store:
                    payload = hero_payload(store)
                    if payload is None:
                        return self._send(404, {
                            "error": "Could not identify hero automatically -- "
                                     "no player has cards known on enough of their "
                                     "own hands."})
                    return self._send(200, payload)
            if path == "/api/evidence":
                query = parse_qs(route.query)
                player_id = int(query.get("player", ["0"])[0])
                stat = query.get("stat", [""])[0]
                if not stat:
                    return self._send(400, {"error": "stat required"})
                with Store(self.db_path) as store:
                    hands = store.player_hands(player_id)
                # Count over *every* matching hand, then truncate. Truncating
                # first made "count" a synonym for the cap and, worse, computed
                # "hits" inside that window -- so a player who limped 4 times in
                # 6,210 hands showed 0 of 60, because none of the four fell in
                # the slice. The instances are what you came to see, so they go
                # first and the rest fill the remainder.
                found = find_evidence(hands, str(player_id), stat)
                hits = [e for e in found if e.hit]
                misses = [e for e in found if not e.hit]
                recent = lambda xs: sorted(
                    xs, key=lambda e: e.started_at or 0, reverse=True)
                shown = recent(hits)[:60] + recent(misses)[:max(0, 60 - len(hits))]
                # One line saying what the count actually means, from the
                # glossary's own high/low readings -- a number without a
                # reading is the thing this whole tool exists to avoid.
                reading, rate, pop = "", None, None
                against = "the field"
                with Store(self.db_path) as store:
                    prof = store.profile(player_id)
                if prof is None:
                    pass
                elif stat.startswith(VS_HERO):
                    # An against-you slice is not among the shrunk stats and
                    # has no population -- there is no field frequency for
                    # "folds to that guy". What it is read against is the
                    # player's own baseline, so that is what the verdict
                    # compares it with, and it says which.
                    parent = stat[len(VS_HERO):]
                    match = next((a for a in prof.adjustments
                                  if a.stat == parent), None)
                    against = "everyone else"
                    if match is not None:
                        rate, pop = match.versus, match.baseline
                        entry = stat_help(parent) or {}
                        reading = entry.get("high" if rate >= pop else "low", "")
                elif prof.stats.get(stat) is not None:
                    rate = prof.stats[stat].value
                    pop = prof.population(stat)
                    entry = stat_help(stat) or {}
                    reading = entry.get("high" if rate >= pop else "low", "")
                return self._send(200, {
                    "stat": stat, "count": len(found), "hits": len(hits),
                    "rate": None if rate is None else round(rate, 4),
                    "population": None if pop is None else round(pop, 4),
                    "compared_to": against,
                    "reading": reading,
                    "shown_hits": sum(1 for e in shown if e.hit),
                    "hands": [vars(e) for e in shown],
                })
            if path.startswith("/api/hand/"):
                hand_id = path.rsplit("/", 1)[1]
                focus = parse_qs(route.query).get("focus", [None])[0]
                with Store(self.db_path) as store:
                    row = store.conn.execute(
                        "SELECT payload FROM hands WHERE hand_id = ?", (hand_id,)).fetchone()
                    if row is None:
                        return self._send(404, {"error": "no such hand"})
                    data = json.loads(gzip.decompress(row["payload"]))
                    hand = hand_from_dict(data)
                    accounts = {
                        (r["site"], r["account"]): int(r["player_id"])
                        for r in store.conn.execute(
                            "SELECT site, account, player_id FROM aliases")}
                for seat in hand.seats:
                    pid = (accounts.get((hand.site, split_key(seat.player_id, seat.name)))
                           or accounts.get((hand.site, seat.player_id)))
                    if pid is not None:
                        seat.player_id = str(pid)
                return self._send(200, replay(hand, focus=focus))
            if path == "/api/meta":
                return self._send(200, {"narrator": narrator_enabled()})
            if path == "/api/glossary":
                return self._send(200, glossary_payload())
            if path == "/api/suggestions":
                with Store(self.db_path) as store:
                    return self._send(200, [{
                        "keep": s.keep, "absorb": s.absorb,
                        "keep_name": s.keep_name, "absorb_name": s.absorb_name,
                        "matched_a": s.matched_a, "matched_b": s.matched_b,
                        "confidence": s.confidence, "reason": s.reason,
                    } for s in suggest_links(store)])
            return self._send(404, {"error": "not found"})
        except Exception as exc:                      # keep the server alive
            return self._send(500, {"error": str(exc)})

    def _same_origin(self) -> bool:
        """Accept only requests the local UI itself could have made.

        Checks Origin when the browser sends one, falls back to Referer, and
        validates Host either way so a DNS-rebinding name cannot point at this
        port and read the database. A request with neither header is allowed:
        that is curl and the CLI, which are not a browser and carry no
        ambient cookies or cross-site risk.
        """
        host = (self.headers.get("Host") or "").split(":")[0]
        if host and host not in LOCAL_HOSTS:
            return False
        stated = self.headers.get("Origin") or self.headers.get("Referer")
        if not stated:
            return True
        try:
            parsed = urlparse(stated)
        except ValueError:
            return False
        return parsed.hostname in LOCAL_HOSTS

    def do_POST(self):
        if not self._same_origin():
            # Every POST here is a write, and two of them are irreversible: a
            # merge cannot be undone and a reset empties the database. Without
            # this check any page open in the same browser could fire one at
            # localhost -- a text/plain body is a CORS-simple request, so it is
            # sent without a preflight and the typed "delete everything"
            # confirmation, which lives in the page, never enters into it.
            return self._send(403, {"error": "cross-origin request refused"})
        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            return self._send(400, {"error": "bad content-length"})
        if length < 0 or length > MAX_BODY_BYTES:
            return self._send(413, {"error": "body too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
        if not isinstance(body, dict):
            return self._send(400, {"error": "body must be an object"})
        route = urlparse(self.path).path
        try:
            if route == "/api/upload":
                return self._upload(body)
            if route == "/api/narrate":
                try:
                    result = narrate(body.get("profile") or {})
                except Unavailable as exc:
                    return self._send(503, {"error": str(exc)})
                return self._send(200, {"text": result.text, "model": result.model})
            if route == "/api/reset":
                if body.get("confirm") != "delete everything":
                    return self._send(400, {"error": "reset not confirmed"})
                with Store(self.db_path) as store:
                    return self._send(200, store.reset())
            if route == "/api/fit-priors":
                with Store(self.db_path) as store:
                    fitted = store.fit_priors(min_players=int(body.get("min_players", 8)))
                    return self._send(200, {"fitted": fitted})
            if route.startswith("/api/session/") and route.endswith("/identity"):
                token = route.split("/")[3]
                if token not in SESSIONS:
                    return self._send(404, {"error": "session expired -- upload again"})
                apply_answers(SESSIONS[token], body.get("answers") or {})
                with Store(self.db_path) as store:
                    return self._send(200, session_payload(token, store))
            if route.startswith("/api/session/") and route.endswith("/plan"):
                token = route.split("/")[3]
                if token not in SESSIONS:
                    return self._send(404, {"error": "session expired -- upload again"})
                return self._send(
                    200, [question_payload(q) for q in SESSIONS[token].get("questions", [])])
            if route.startswith("/api/session/") and route.endswith("/commit"):
                token = route.split("/")[3]
                if token not in SESSIONS:
                    return self._send(404, {"error": "session expired -- upload again"})
                with Store(self.db_path) as store:
                    return self._send(200, commit_session(
                        store, token, body.get("answers") or {}))
            with Store(self.db_path) as store:
                if route == "/api/link":
                    store.link(int(body["keep"]), int(body["absorb"]))
                    return self._send(200, {"ok": True})
                if route == "/api/unlink":
                    new_id = store.unlink(int(body["player_id"]),
                                          str(body["site"]), str(body["account"]))
                    return self._send(200, {"ok": True, "player_id": new_id})
                if route == "/api/note":
                    store.add_note(int(body["player_id"]), str(body["body"]))
                    return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except ValueError as exc:
            return self._send(409, {"error": str(exc)})
        except Exception as exc:
            return self._send(500, {"error": str(exc)})

    def _upload(self, body: dict):
        """Parse uploaded files into a session held in memory."""
        files = body.get("files") or []
        if not files:
            return self._send(400, {"error": "no files"})
        hands, names, rejected = [], [], []
        for item in files:
            name = str(item.get("name", "upload"))
            try:
                parsed = parse_upload(name, item.get("content", ""))
            except (UnknownFormat, ValueError, KeyError) as exc:
                rejected.append({"name": name, "reason": str(exc) or "unrecognised format"})
                continue
            if not parsed:
                rejected.append({"name": name, "reason": "no hands in file"})
                continue
            hands.extend(parsed)
            names.append({"name": name, "hands": len(parsed)})
        if not hands:
            return self._send(400, {"error": "nothing could be parsed", "rejected": rejected})

        # One hand id can appear in two exports of the same game.
        unique, seen = [], set()
        for hand in sorted(hands, key=lambda h: h.started_at):
            if hand.hand_id in seen:
                continue
            seen.add(hand.hand_id)
            unique.append(hand)

        _reap_sessions()
        token = secrets.token_urlsafe(9)
        SESSIONS[token] = {"hands": unique, "files": names, "created": time.time()}
        # Identity is settled up front so the session being read is already
        # pooled. Reading the database to ask a better question is not the
        # same as writing to it -- nothing is stored until you save.
        with Store(self.db_path) as store:
            questions = session_questions(store, unique)
            SESSIONS[token]["questions"] = questions
            # Clear matches to existing players (and same-account renames) are
            # applied immediately — keep the database display name, only ask
            # about leftover net-new / ambiguous pairs.
            auto = auto_answers(questions)
            if auto:
                apply_answers(SESSIONS[token], auto)
            payload = session_payload(token, store)
        payload["rejected"] = rejected
        return self._send(200, payload)


def serve(db: Path = DEFAULT_PATH, port: int = 8766, open_browser: bool = True):
    Handler.db_path = Path(db)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"villain UI on {url}  (database: {db})")
    print("ctrl-c to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="villain-ui", description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    serve(db=args.db, port=args.port, open_browser=not args.no_browser)
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Villain</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f6f6f5; --panel: #ffffff; --ink: #111111; --muted: #6b6b68;
    --line: #e3e2df; --edge: #979590; --accent: #111111; --accent-soft: #f1f0ee;
    --warn: #b4532a; --danger: #b4532a; --red: #b4532a;
    --hero: #2f6fe0; --hero-soft: #eaf1fd;   /* you; the villain world is warm, you are cool */
    /* Neutral ordinal ramp, validated light->dark on the panel surface:
       light end clears 2:1, monotone lightness, visible step gaps. Shade
       carries confidence, never identity. */
    --mark-1: #8f8d89;   /* tentative */
    --mark-2: #575552;   /* likely */
    --mark-3: #111111;   /* strong */
    --band: #e8e7e4;     /* credible interval wash */
    --grid: #e6e5e2; --axis: #8a8a86; --tick: #6f6e69;
    --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #0d0d0d; --panel: #17181a; --ink: #f2f1ee; --muted: #98968f;
      --line: #2a2b2d; --edge: #606166; --accent: #f2f1ee; --accent-soft: #202123;
      --warn: #e5645a; --danger: #e5645a; --red: #e5645a;
    --hero: #6aa6ff; --hero-soft: #16233a;
      --hero: #6aa6ff; --hero-soft: #16233a;
      --mark-1: #787774; --mark-2: #adaba6; --mark-3: #f0efec;
      --band: #26272a;
      --grid: #232427; --axis: #75746f; --tick: #9b9a94;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0d0d0d; --panel: #17181a; --ink: #f2f1ee; --muted: #98968f;
    --line: #2a2b2d; --edge: #606166; --accent: #f2f1ee; --accent-soft: #202123;
    --warn: #e5645a; --danger: #e5645a; --red: #e5645a;
    --mark-1: #787774; --mark-2: #adaba6; --mark-3: #f0efec;
    --band: #26272a;
    --grid: #232427; --axis: #75746f; --tick: #9b9a94;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 24px 20px 90px; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
  h1 {
    font-size: 38px; margin: 0; letter-spacing: -0.05em; font-weight: 800;
    line-height: 0.95;
  }
  h1 a { color: inherit; text-decoration: none; display: inline-flex; align-items: center; }
  h1 .dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--red);
    display: inline-block; margin-left: 8px; margin-bottom: 18px;
  }
  .iconbtn {
    border: 1px solid var(--edge); background: transparent; color: var(--muted);
    border-radius: 999px; width: 34px; height: 34px; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center; padding: 0;
  }
  .iconbtn:hover { color: var(--ink); border-color: var(--accent); }
  .sub { color: var(--muted); font-size: 12.5px; }
  nav { display: flex; gap: 4px; margin: 18px 0 0; flex-wrap: wrap;
        border-bottom: 1px solid var(--line); }
  nav button {
    border: 0; border-bottom: 2px solid transparent; background: none; color: var(--muted);
    font: inherit; font-size: 14px; padding: 8px 14px; cursor: pointer; border-radius: 0;
  }
  nav button:hover { color: var(--ink); }
  nav button.on { color: var(--ink); border-bottom-color: var(--red); font-weight: 600; }
  /* Hero is you. The tool is otherwise about the villain across the table, so
     anything that reads your own play -- the Hero tab, and the against-you
     panel that exists only because the export knows which seat is yours --
     carries a cool blue identity. Done by remapping the accent tokens inside a
     scope, so every accented element (bars, links, markers, the active tab)
     turns blue while the monochrome-plus-confidence system stays put
     everywhere else. */
  .hero-scope {
    --red: var(--hero); --warn: var(--hero); --danger: var(--hero);
    --accent-soft: var(--hero-soft); --mark-3: var(--hero);
  }
  .hero-scope .linkbtn { color: var(--hero); }
  .hero-scope .linkbtn:hover { text-decoration-color: var(--hero); }
  nav button[data-tab="hero"].on { border-bottom-color: var(--hero); }
  nav button[data-tab="hero"]::before {
    content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: var(--hero); margin-right: 7px; vertical-align: 1px;
  }
  /* A small blue wordmark reading "this is about you" -- reused next to the
     against-you panel and on your own profile. */
  .hero-badge {
    display: inline-flex; align-items: center; gap: 5px; vertical-align: 2px;
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.14em;
    font-weight: 700; color: var(--hero); margin-left: 6px;
  }
  .hero-badge::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--hero);
  }
  .panel {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 12px; padding: 18px; margin: 16px 0;
  }
  /* Dashboard: tiles sized by content, flowing into as many columns as fit.
     align-items:start stops a tall tile stretching its neighbours. */
  .dash {
    display: grid; gap: 14px; align-items: start; margin: 16px 0;
    /* 420px min gives two columns at the 1060px wrap and one on a laptop in a
       split view. A narrower minimum fitted three, which left a dead column
       whenever there were only two tiles to place. */
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  }
  /* No height:100% -- a short tile stretching to match a tall neighbour is
     empty space pretending to be content. */
  .dash > .panel { margin: 0; }
  /* The tiles flow into balanced columns rather than being assigned to a fixed
     left and right. Which panel is tallest changes with the data -- an empty
     "What to do" on a thin sample is short, a long leak list is not -- so any
     fixed split leaves a column of air on some player. */
  /* Two real columns, packed by height in JS. CSS column-count fills the first
     column before the second, so a tall panel and a short one landed together
     and left the other column 740px empty. */
  .dash-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
               align-items: start; margin: 0; }
  @media (max-width: 780px) { .dash-cols { grid-template-columns: 1fr; } }
  .dash-cols { align-items: stretch; }
  .dash-cols > .col { display: flex; flex-direction: column; gap: 14px; min-width: 0; }
  .dash-cols > .col > .panel:last-child { flex: 1 1 auto; }
  .dash-cols .panel { margin: 0; }
  /* The interval chart is drawn in a 0-300 viewBox, so it can scale to the
     cell instead of forcing one: at a fixed 300px every stat label in the
     column wrapped to three lines. */
  .detail-body td svg { width: 100%; height: auto; display: block; }
  .detail-body th, .detail-body td { white-space: nowrap; }
  .detail-body td.label { white-space: normal; }
  /* Anything marked wide spans the row -- the read, the balanced column block,
     the full-width panels. Scoping this to .panel meant the column container
     was placed in a single grid track and then split again inside it. */
  .dash > .wide { grid-column: 1 / -1; }
  @media (max-width: 700px) { .dash { grid-template-columns: 1fr; } }
  .panel h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); margin: 0 0 14px; font-weight: 600;
  }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
       color: var(--muted); font-weight: 600; cursor: pointer; white-space: nowrap; }
  th.sorted::after { content: " \25B4"; color: var(--accent); }
  th.sorted.desc::after { content: " \25BE"; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr.clickable { cursor: pointer; }
  tbody tr.clickable:hover { background: var(--accent-soft); }
  .name { font-weight: 600; }
  /* The archetype pill keeps the accent whatever the confidence; a solid edge
     means the read has cleared 50%, a dashed one means it has not. Colour on
     its own was carrying that distinction, and dropping the colour to fix
     that lost the accent instead. */
  .tag.arch { border-color: var(--red); color: var(--ink); }
  .tag.arch:not(.on) { border-style: dashed; }
  .tag {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; border: 1px solid var(--line); color: var(--muted);
    white-space: nowrap;
  }
  .tag.on { border-color: var(--red); color: var(--ink); }
  /* Same accent the archetype pill and sort caret already use for "this one
     is notable" -- not a new colour for a new meaning. */
  .hero-row-marker { background: var(--hero-soft); }
  .hero-tag { margin-left: 6px; border-color: var(--hero); color: var(--hero); font-weight: 600; }
  .scroller { overflow-x: auto; }
  .muted { color: var(--muted); }
  .small { font-size: 12.5px; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  /* Wraps, and the text side is allowed to shrink. Without min-width:0 a flex
     child refuses to go narrower than its own content, so a long description
     shoves whatever sits beside it straight out of the panel -- which is how
     the reset button ended up overflowing. */
  .spread {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 12px; flex-wrap: wrap;
  }
  .spread > * { min-width: 0; }
  .spread > :first-child { flex: 1 1 auto; }
  .leak-head > * { min-width: 0; }
  /* Buttons come in three shapes and no others: a pill for actions, a plain
     text link for inline affordances, and the icon square in the header.
     Anything needing a fourth is probably not a button. */
  button.act {
    /* Padding in em, so it tracks whatever font-size the variant sets and a
       short label is never left rattling around inside a wide pill. flex:none
       stops a flex parent from squeezing the button below its own text. */
    display: inline-flex; align-items: center; justify-content: center;
    text-align: center;
    font: inherit; font-size: 14px; line-height: 1.25;
    padding: 0.46em 0.9em; border-radius: 8px;
    border: 1px solid var(--edge); background: transparent;
    color: var(--ink); cursor: pointer; flex: none;
    /* A label longer than its container wraps rather than running past the
       edge. nowrap plus flex:none is how a button escapes its panel. */
    max-width: 100%; white-space: normal; text-align: left;
    transition: border-color .12s, background .12s, color .12s, opacity .12s;
  }
  button.act:hover:not(:disabled) { border-color: var(--accent); }
  button.act:active:not(:disabled) { transform: translateY(0.5px); }
  button.act.primary {
    background: var(--accent); color: var(--panel); border-color: var(--accent);
    font-weight: 600;
  }
  button.act.primary:hover:not(:disabled) { opacity: .88; }
  button.act.small { font-size: 11px; }
  button.act:disabled { opacity: .45; cursor: default; }
  button.act.danger { border-color: var(--danger); color: var(--danger); }
  button.act.danger:hover:not(:disabled) {
    background: var(--danger); color: var(--panel); border-color: var(--danger);
  }
  /* Inline affordance: reads as part of the sentence it sits in, because a
     pill dropped mid-paragraph breaks the line it is trying to annotate. */
  button.linkbtn {
    font: inherit; font-size: inherit; padding: 0; border: 0; background: none;
    color: var(--ink); cursor: pointer; text-decoration: underline;
    text-underline-offset: 2px; text-decoration-color: var(--axis);
  }
  button.linkbtn:hover { text-decoration-color: var(--accent); }
  .hero { font-size: 38px; line-height: 1.05; letter-spacing: -0.03em; }
  .hero-name { min-width: 0; }
  .hero-sub { font-size: 15px; color: var(--muted); margin-top: 2px; }
  .hero-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .skill-badge {
    width: 58px; height: 58px; border-radius: 50%; flex: none;
    border: 2px solid var(--red); color: var(--red);
    display: inline-flex; flex-direction: column;
    align-items: center; justify-content: center;
    cursor: help; line-height: 1;
  }
  .skill-badge .score { font-size: 20px; font-weight: 800; letter-spacing: -0.03em; }
  .skill-badge .of { font-size: 11px; font-weight: 600; margin-top: 2px; letter-spacing: .04em;
                     text-transform: uppercase; opacity: .85; }
  .read-copy .summary { color: var(--muted); margin: 0 0 10px; }
  .read-copy .plan { margin: 0; max-width: 72ch; }
  .read-copy .summary { max-width: 72ch; }
  .read-meta { font-size: 12.5px; color: var(--muted); margin-top: 10px; line-height: 1.7; }
  .skill-side { margin: 0; min-width: 0; max-width: none; text-align: left; }
  .skill-side .metric { grid-template-columns: 1fr 90px 28px; gap: 8px; margin: 4px 0; }
  .skill-side .metric .small { font-size: 12.5px; }
  .drop {
    border: 1.5px dashed var(--edge); border-radius: 12px; padding: 34px 20px;
    text-align: center; color: var(--muted); cursor: pointer; transition: border-color .12s;
  }
  .drop.compact { padding: 14px; margin-bottom: 12px; }
  .bulk {
    display: flex; gap: 10px; align-items: flex-start; cursor: pointer;
    border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px;
    margin: 0 0 14px;
  }
  .bulk input { margin-top: 3px; }
  .comp { padding: 8px 0; border-top: 1px solid var(--line); }
  .comp:first-child { border-top: 0; }
  .comp-head { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
  .comp-name { font-size: 14px; }
  .comp-score { font-variant-numeric: tabular-nums; color: var(--muted); }
  .comp.weak .comp-score { color: var(--red); font-weight: 600; }
  .comp-bar { margin: 5px 0 0; }
  .comp-bar svg { display: block; width: 100%; height: auto; }
  .comp-note { margin-top: 3px; }
  .comp-why { margin-top: 4px; color: var(--muted); }
  .comp.weak .comp-why { color: var(--ink); }
  .comp-ev { margin-top: 5px; }
  .comp-ev:empty { display: none; }
  .cards-row { display: inline-flex; gap: 3px; vertical-align: middle; }
  .card {
    display: inline-flex; align-items: center; gap: 1px;
    border: 1px solid var(--line); border-radius: 4px;
    background: var(--panel); padding: 2px 5px; line-height: 1;
    font-size: 14px; font-variant-numeric: tabular-nums; min-width: 26px;
    justify-content: center;
  }
  .card.red { color: var(--red); }
  .card.black { color: var(--ink); }
  .card .r { font-weight: 600; }
  .small-cards .card { font-size: 11px; padding: 1px 4px; min-width: 22px; }
  .sess-delta {
    display: grid; grid-template-columns: minmax(0,240px) 56px 62px minmax(0,1fr);
    gap: 10px;
    align-items: baseline; padding: 5px 0; border-top: 1px solid var(--line);
  }
  .sess-delta:first-child { border-top: 0; }
  .sess-who { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .sess-regime-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0 8px; }
  .sess-regime-tab {
    border: 1px solid var(--edge); background: transparent; color: var(--muted);
    border-radius: 999px; padding: 3px 11px; font: inherit; font-size: 12.5px;
    cursor: pointer;
  }
  .sess-regime-tab:hover { color: var(--ink); }
  .sess-regime-tab.on { border-color: var(--red); color: var(--ink); font-weight: 600; }
  td.worth { font-variant-numeric: tabular-nums; }
  td.worth.big { color: var(--red); font-weight: 600; }
  .detail-body td.label { min-width: 156px; }
  .sheet { box-shadow: 0 24px 60px rgba(0,0,0,.45); }
  .cards-row.hole { margin-left: 16px; padding-left: 16px;
                    border-left: 1px solid var(--edge); }
  tbody tr.on { background: var(--accent-soft); }
  .sess-layout {
    display: grid; grid-template-columns: 260px minmax(0, 1fr);
    gap: 14px; align-items: start; margin: 16px 0;
  }
  .sess-layout.collapsed { grid-template-columns: 46px minmax(0, 1fr); }
  .sess-layout.collapsed .sess-list h2,
  .sess-layout.collapsed #sess-rows { display: none; }
  .sess-list {
    position: sticky; top: 12px;
    /* Content-side cap. calc(100vh - 40px) never engaged at 20 sittings, so
       the list defined the grid row and left a 650px hole beside it. */
    max-height: min(calc(100vh - 132px), 560px);
    overflow: auto; margin: 0;
  }
  .sess-main { margin: 0; }
  #sess-rows { display: flex; flex-direction: column; gap: 2px; margin-top: 8px; }
  .sess-item {
    display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
    text-align: left;
    font: inherit; background: none; color: var(--ink); cursor: pointer;
    border: 0; border-left: 2px solid transparent; border-radius: 8px;
    padding: 7px 8px;
  }
  .sess-item:hover { background: var(--accent-soft); }
  .sess-item.on { background: var(--accent-soft); border-left-color: var(--red); }
  .sess-when { font-size: 14px; }
  @media (max-width: 780px) {
    .sess-layout, .sess-layout.collapsed { grid-template-columns: 1fr; }
    .sess-list { position: static; max-height: 240px; }
  }
  .sess-delta .up { color: var(--red); }
  .sess-delta .down { color: var(--muted); }
  .linkish { cursor: pointer; text-decoration: underline;
             text-underline-offset: 3px; text-decoration-color: var(--axis); }
  select { font: inherit; font-size: 14px; padding: 5px 8px; border-radius: 8px;
           background: var(--panel); color: var(--ink); border: 1px solid var(--edge); }
  .q.group .members { margin: 8px 0; }
  .q.group .member {
    display: flex; gap: 10px; align-items: baseline; padding: 4px 0;
    border-top: 1px solid var(--line);
  }
  .q.group .member:first-child { border-top: 0; }
  /* A merged suggestion is finished business: it dims and leaves. */
  .leak.merged { opacity: 0; transition: opacity .35s ease; }
  @media (prefers-reduced-motion: reduce) { .leak.merged { transition: none; } }
  .veil.busy { cursor: progress; }
  .busy-sheet { display: flex; gap: 14px; align-items: center; max-width: 420px; }
  .spinner {
    width: 20px; height: 20px; flex: none; border-radius: 50%;
    border: 2px solid var(--line); border-top-color: var(--red);
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @media (prefers-reduced-motion: reduce) { .spinner { animation-duration: 2.4s; } }
  /* button.act wraps by design (long leak labels); a toolbar button must not. */
  .act.nowrap { white-space: nowrap; }
  .drop:hover, .drop.over { border-color: var(--red); color: var(--ink); }
  .leak { padding: 14px 0; border-bottom: 1px solid var(--line); }
  .leak:last-child { border-bottom: 0; }
  .leak-head { display: flex; justify-content: space-between; gap: 14px; align-items: baseline; }
  .leak-head .headline b {
    font-size: 15px; font-weight: 600; letter-spacing: -0.01em;
  }
  .leak-advice {
    color: var(--ink); font-size: 14px; max-width: 68ch; margin-top: 6px;
    line-height: 1.45;
  }
  .leak .numbers { margin-top: 2px; }
  .how { margin-top: 8px; }
  .how-link { margin-top: 10px; }
  .how-body { color: var(--ink); font-size: 14px; line-height: 1.45; }
  .howblock { margin: 10px 0; max-width: 66ch; }
  .howlabel { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
              color: var(--muted); margin-bottom: 3px; font-weight: 600; }
  /* hero: preflop range chart. Magnitude (how often a hand is played) is
     shade of the same neutral ink ramp the skill bars use elsewhere --
     shade already carries confidence/frequency in this app, never identity,
     so this is the existing convention, not a new one. */
  .range-grid {
    display: grid; grid-template-columns: repeat(13, 1fr); gap: 2px;
    margin: 12px 0; max-width: 480px;
  }
  .range-cell {
    aspect-ratio: 1; border-radius: 3px; border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; font-variant-numeric: tabular-nums;
    color: var(--ink);
  }
  .range-cell.dark-text { color: var(--panel); }
  .range-legend {
    display: flex; align-items: center; gap: 8px; font-size: 12px;
    color: var(--muted); margin: 6px 0 18px;
  }
  .range-legend .ramp {
    width: 120px; height: 10px; border-radius: 4px; border: 1px solid var(--line);
    background: linear-gradient(to right, var(--panel), var(--ink));
  }
  .hero-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .fold-row {
    display: grid; grid-template-columns: 56px 70px minmax(0,1fr) auto;
    gap: 12px; align-items: baseline; padding: 7px 0; border-top: 1px solid var(--line);
    font-size: 13.5px; cursor: pointer;
  }
  .fold-row:first-child { border-top: 0; }
  .fold-row:hover { background: var(--accent-soft); }
  svg { display: block; overflow: visible; }
  .tip {
    position: fixed; pointer-events: none; z-index: 40; max-width: 270px;
    background: var(--panel); color: var(--ink); border: 1px solid var(--line);
    border-radius: 8px; padding: 8px 10px; font-size: 12.5px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.16); opacity: 0; transition: opacity .1s;
  }
  .tip.on { opacity: 1; }
  .empty { color: var(--muted); padding: 8px 0; }
  .err { color: var(--warn); }
  /* per-player tabs inside a result */
  .ptabs {
    display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 16px;
    padding-bottom: 2px;
  }
  .ptab {
    border: 1px solid var(--edge); background: transparent; color: var(--muted);
    border-radius: 999px; padding: 6px 13px; font: inherit; font-size: 14px;
    cursor: pointer; display: flex; align-items: center; gap: 7px;
  }
  .ptab:hover { color: var(--ink); }
  .ptab.on { border-color: var(--red); color: var(--ink); font-weight: 600; }
  .ptab .meta { font-size: 11px; color: var(--muted); font-weight: 400; }
  /* the hover-for-meaning affordance */
  .info {
    display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%; border: 1px solid var(--line);
    color: var(--muted); font-size: 11px; font-weight: 700; cursor: help;
    vertical-align: 1px; margin-left: 5px; font-style: normal; flex: none;
  }
  .info:hover { color: var(--ink); border-color: var(--accent); }
  :where(button, [href], input, select, summary, [tabindex]):focus-visible,
  tr.clickable:focus-visible, .ev:focus-visible, th:focus-visible {
    outline: 2px solid var(--red); outline-offset: 2px; border-radius: 4px;
  }
  .tip .hl { color: var(--ink); font-weight: 600; }
  .tip .dir { margin-top: 6px; }
  .tip .dir b { display: inline-block; min-width: 34px; }
  details > summary {
    cursor: pointer; color: var(--ink); font-size: 12.5px; list-style: none;
    padding: 4px 0; font-weight: 600; letter-spacing: 0.02em;
  }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before {
    content: "\25B8 "; color: var(--red); font-weight: 700;
  }
  details[open] > summary::before { content: "\25BE "; color: var(--red); }
  details > summary:hover { color: var(--red); }
  .headline { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .plan { max-width: 62ch; }
  .meta { text-align: right; line-height: 1.7; }
  .metric { display: grid; grid-template-columns: 170px 1fr 34px; gap: 10px;
            align-items: center; margin: 4px 0; }
  .footnote { font-size: 12.5px; display: flex; gap: 8px; flex-wrap: wrap;
              align-items: baseline; margin: 22px 2px; }
  .danger-link { color: var(--danger); text-decoration-color: var(--danger); }
  /* Marks a claim the evidence does not fully support. Deliberately quiet:
     it is a caveat, not an alarm. */
  .flag {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px; margin-left: 6px; border-radius: 50%;
    border: 1px solid var(--red); color: var(--red);
    font-size: 11px; font-weight: 700; cursor: help; vertical-align: 1px;
  }
  .flag:hover { background: var(--red); color: var(--panel); }
  .leak.watch { opacity: .82; }
  .leak.weakspots .metric { grid-template-columns: 1fr 150px 30px; }
  /* Two bars to a row, the same length, so the shift is the thing you see
     rather than something to be worked out from two percentages. */
  .adjust .metric { grid-template-columns: 88px 1fr 40px; margin: 3px 0; }
  /* The reads sit two-up (more on a very wide screen) so a full-width panel is
     not sparse; separation is the grid gap, not a per-read rule. */
  .adjust-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                 gap: 6px 40px; }
  .adjust-grid .leak { border-bottom: 0; }
  /* Position breakdown: a fixed name column, a bar that fills, and a value
     column wide enough that "21% played" never wraps -- the old .metric row
     truncated it. */
  .hero-pos { display: grid; row-gap: 6px; }
  .hero-pos .pos-row {
    display: grid; grid-template-columns: 96px 1fr auto; align-items: center; gap: 12px;
  }
  .hero-pos .pos-name { font-weight: 600; }
  .hero-pos .pos-bar { display: flex; min-width: 0; }
  .hero-pos .pos-bar svg { width: 100%; }
  .hero-pos .pos-val { white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
  .timing-street { margin-top: 14px; }
  .timing-street .street-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
    color: var(--muted); font-weight: 600; margin-bottom: 6px;
  }
  .timing-grid {
    display: grid; grid-template-columns: 64px 1fr 1fr 1fr; gap: 1px;
    background: var(--line); border: 1px solid var(--line);
  }
  .timing-grid > * { background: var(--panel); padding: 9px 10px; min-width: 0; }
  .timing-grid .corner { background: var(--accent-soft); }
  .timing-grid .colhead, .timing-grid .rowhead {
    font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
    color: var(--muted); background: var(--accent-soft); font-weight: 600;
  }
  .timing-grid .rowhead { display: flex; align-items: center; }
  .timing-cell .tell { font-weight: 600; font-size: 14px; margin-bottom: 2px; }
  .timing-cell .read { color: var(--muted); font-size: 12.5px; line-height: 1.4; }
  .timing-cell .n { font-size: 11px; color: var(--muted); margin-top: 6px; }
  .timing-cell.thin .tell { color: var(--muted); font-weight: 500; font-size: 12.5px; }
  .narration { margin-top: 10px; max-width: none; }
  .narration.hidden { display: none; }
  .narration blockquote {
    margin: 0 0 6px; padding: 0 0 0 13px; border-left: 2px solid var(--red);
  }
  .narration ul.suggested { margin: 4px 0; padding-left: 0; list-style: none; }
  .narration ul.suggested li {
    position: relative; padding: 0 0 0 16px; margin: 0 0 10px; max-width: none;
  }
  .narration ul.suggested li::before {
    content: "\2013"; position: absolute; left: 0; color: var(--red);
  }
  .narrate-actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .rank { font-variant-numeric: tabular-nums; color: var(--muted); width: 22px; }
  .ev { display: grid; grid-template-columns: minmax(200px, auto) 1fr auto; gap: 12px;
        align-items: center; padding: 8px 0 8px 8px;
        border-bottom: 1px solid var(--line);
        border-left: 2px solid transparent; cursor: pointer; }
  /* A hand that moved the numerator gets a rail, not the word "counted": the
     board and the action already say what happened. */
  .ev.counted { border-left-color: var(--red); }
  .ev[hidden] { display: none; }
  .ev-summary { display: block; }
  .ev-when { display: block; font-size: 11px; }
  .ev-net { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .ev-net.lost { color: var(--muted); }
  .ev-verdict { margin: 0 0 6px; font-size: 14px; color: var(--ink); max-width: 70ch; }
  .onlyhits { display: inline-flex; gap: 6px; align-items: center;
              margin-left: 10px; cursor: pointer; white-space: nowrap; }
  #modal2 .veil { z-index: 60; }
  .cards-row.hole { opacity: .75; margin-left: 8px; padding-left: 8px;
                    border-left: 1px solid var(--line); }
  .seatline { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }
  .seatchunk { display: inline-flex; align-items: center; gap: 4px; }
  .ev:hover { background: var(--accent-soft); }
  .ev:last-child { border-bottom: 0; }
  .mono { font-family: var(--mono); font-size: .95em; }
  .street { border-top: 1px solid var(--line); padding: 10px 0; }
  .street:first-child { border-top: 0; }
  .street h4 { margin: 0 0 6px; font-size: 11px; text-transform: uppercase;
               letter-spacing: .06em; color: var(--muted); font-weight: 600;
               display: flex; gap: 10px; align-items: baseline; }
  .street .act { display: grid; grid-template-columns: 44px 1fr auto auto;
         gap: 10px; padding: 3px 0; font-size: 14px; }
  .street .act.focus { font-weight: 600; }
  .street .act.focus .who::before { content: "\25B8 "; color: var(--accent); }
  .street .act .amt { font-variant-numeric: tabular-nums; text-align: right; color: var(--muted); }
  .act.post { color: var(--muted); font-size: 12.5px; }
  /* review dialog */
  .veil {
    position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 30;
    display: flex; align-items: center; justify-content: center; padding: 20px;
  }
  .sheet {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    max-width: 640px; width: 100%; max-height: 86vh; overflow-y: auto; padding: 22px;
  }
  /* Keeps Close reachable over a long hand or evidence list instead of
     scrolling off with the content. */
  .sheet > .spread:first-child {
    position: sticky; top: 0; z-index: 1; background: var(--panel);
    padding-bottom: 10px; margin-bottom: 4px;
  }
  .q { border-top: 1px solid var(--line); padding: 14px 0; }
  .q:first-of-type { border-top: 0; }
  .q-prompt { font-weight: 600; }
  .sides { display: flex; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
  .side {
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px;
    font-size: 14px; flex: 1; min-width: 180px;
  }
  .choice { display: flex; gap: 8px; margin-top: 8px; }
  .choice label {
    border: 1px solid var(--edge); border-radius: 999px; padding: 4px 12px;
    font-size: 14px; cursor: pointer;
  }
  .choice input { margin-right: 6px; }
  .choice label:has(input:checked) { border-color: var(--accent); color: var(--ink); font-weight: 600; }
  .choice.namechoice { align-items: baseline; }
  .choice .namelabel { align-self: center; }
  .choice.disabled { opacity: .4; }
  .choice.disabled label { cursor: not-allowed; }
  .choice.disabled label:has(input:checked) {
    border-color: var(--line); color: var(--muted); font-weight: 400;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1><a href="#/">Villain<span class="dot"></span></a></h1>
    <span class="sub" id="meta"></span>
    <span style="flex:1"></span>
    <button class="iconbtn" id="theme" title="light / dark" aria-label="switch theme"></button>
  </header>
  <nav>
    <button data-tab="players" class="on">Database</button>
    <button data-tab="sessions">Sessions</button>
    <button data-tab="hero">Hero</button>
  </nav>
  <div id="view"></div>
</div>
<div class="tip" id="tip"></div>
<div id="modal"></div>
<div id="modal2"></div>
<script>
const $ = (s, r) => (r || document).querySelector(s);
const fmtPct = v => (100 * v).toFixed(0) + "%";
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const SVG = "http://www.w3.org/2000/svg";
const state = {tab: "players", session: null, player: null, glossary: null,
               sessionId: null};

/* An "i" that explains a term on hover. Everything the tool says in shorthand
   gets one, because a number nobody can interpret is worse than no number. */
function info(html) {
  const span = document.createElement("button");
  span.type = "button";
  span.className = "info"; span.textContent = "i";
  span.setAttribute("aria-label", "what this means");
  bindTip(span, html);
  return span;
}
function termTip(term) {
  const g = state.glossary;
  const text = g && g.terms[term];
  return text ? `<span class="hl">${esc(term)}</span><br>${esc(text)}` : esc(term);
}
function withInfo(node, html) {
  const wrap = document.createElement("span");
  wrap.style.cssText = "display:inline-flex;align-items:baseline";
  wrap.appendChild(node);
  wrap.appendChild(info(html));
  return wrap;
}
/* Full explanation of a statistic: what it counts, and whether *this* player
   is over or under the field -- with the matching play implication. */
function statTip(stat, label, row) {
  const g = state.glossary;
  const h = g && (g.stats[stat] || g.stats[stat.split(":")[0]]);
  if (!h) return esc(label || stat);
  let direction = "";
  if (row && row.population != null && row.value != null) {
    const delta = row.value - row.population;
    if (delta > 0.03) {
      direction = `<div class="dir"><b>High</b> vs field \u2014 ${esc(h.high)}</div>`;
    } else if (delta < -0.03) {
      direction = `<div class="dir"><b>Low</b> vs field \u2014 ${esc(h.low)}</div>`;
    } else {
      direction = `<div class="dir"><b>Near</b> the field \u2014 neither direction is clear yet.</div>`;
    }
  } else {
    direction = `<div class="dir"><b>High</b> ${esc(h.high)}</div>
      <div class="dir"><b>Low</b> ${esc(h.low)}</div>`;
  }
  return `<span class="hl">${esc(label || stat)}</span><br>${esc(h.what)}
    ${direction}`;
}

function fieldRead(row) {
  const g = state.glossary;
  const h = g && (g.stats[row.stat] || g.stats[row.stat.split(":")[0]]);
  if (!h || row.population == null) return "";
  const delta = row.value - row.population;
  if (delta > 0.03) return `<br><span class="hl">Over the field</span> \u2014 ${esc(h.high)}`;
  if (delta < -0.03) return `<br><span class="hl">Under the field</span> \u2014 ${esc(h.low)}`;
  return `<br><span class="muted">Near the field \u2014 neither over nor under yet.</span>`;
}

/* ---- tooltip ---- */
const tip = $("#tip");
function bindTip(el, html) {
  const place = (x, y) => {
    tip.innerHTML = html; tip.classList.add("on");
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + pad, top = y + pad;
    if (left + w > innerWidth - 8) left = x - w - pad;
    if (top + h > innerHeight - 8) top = y - h - pad;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  };
  const hide = () => tip.classList.remove("on");
  // Keyboard and touch reach it too: anchored to the element's own box, since
  // there is no cursor to hang it off.
  const anchor = () => {
    const r = el.getBoundingClientRect();
    place(r.left + r.width / 2, r.bottom - 6);
  };
  el.addEventListener("focus", anchor);
  el.addEventListener("blur", hide);
  el.addEventListener("touchstart", e => { e.preventDefault(); anchor(); }, {passive: false});
  el.addEventListener("mousemove", e => {
    tip.innerHTML = html; tip.classList.add("on");
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + pad, y = e.clientY + pad;
    if (x + w > innerWidth - 8) x = e.clientX - w - pad;
    if (y + h > innerHeight - 8) y = e.clientY - h - pad;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  });
  el.addEventListener("mouseleave", () => tip.classList.remove("on"));
}
function el(tag, attrs, parent) {
  const node = document.createElementNS(SVG, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}

/* ---- the one mark this tool needs, over and over ----
   Interval band for the credible range, dot for the estimate, hairline tick
   for the field, warm tick for breakeven. Everything here is a frequency with
   uncertainty measured against a threshold, so it is all the same picture. */
function statRow(row) {
  const W = 300, H = 34, mid = 17, r = 5;
  const x = v => Math.max(0, Math.min(1, v)) * W;
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${row.label}: ${fmtPct(row.value)}, range ${fmtPct(row.lo)} to ${fmtPct(row.hi)}`});
  el("line", {x1: 0, y1: mid, x2: W, y2: mid, stroke: "var(--grid)", "stroke-width": 1}, svg);
  el("rect", {x: x(row.lo), y: mid - 6, width: Math.max(2, x(row.hi) - x(row.lo)),
              height: 12, rx: 4, fill: "var(--band)"}, svg);
  el("line", {x1: x(row.population), y1: mid - 10, x2: x(row.population), y2: mid + 10,
              stroke: "var(--axis)", "stroke-width": 1}, svg);
  if (row.breakeven != null) {
    el("line", {x1: x(row.breakeven), y1: mid - 11, x2: x(row.breakeven), y2: mid + 11,
                stroke: "var(--warn)", "stroke-width": 2, "stroke-linecap": "round"}, svg);
  }
  el("circle", {cx: x(row.value), cy: mid, r: r + 2, fill: "var(--panel)"}, svg);
  el("circle", {cx: x(row.value), cy: mid, r: r, fill: "var(--mark-3)"}, svg);
  const hit = el("rect", {x: 0, y: 0, width: W, height: H, fill: "transparent"}, svg);
  bindTip(hit, `<b>${esc(row.label)}</b> \u2014 ${fmtPct(row.value)}<br>
    <span class="muted">95% range ${fmtPct(row.lo)}\u2013${fmtPct(row.hi)}</span><br>
    raw ${row.raw == null ? "\u2014" : fmtPct(row.raw)} of ${row.opps} ${esc(row.denominator)}<br>
    field ${fmtPct(row.population)}${row.breakeven != null
      ? `<br><span style="color:var(--warn)">${esc(row.breakeven_label)} ${fmtPct(row.breakeven)}</span>` : ""}
    ${fieldRead(row)}`);
  return svg;
}

function bar(value, max, color, width) {
  const W = width || 150, H = 14;
  const svg = el("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`});
  el("rect", {x: 0, y: 3, width: W, height: 8, rx: 4, fill: "var(--grid)"}, svg);
  const w = max > 0 ? Math.max(3, (value / max) * W) : 3;
  el("rect", {x: 0, y: 3, width: w, height: 8, rx: 4, fill: color}, svg);
  return svg;
}
const TIER_COLOR = {strong: "var(--mark-3)", likely: "var(--mark-2)", tentative: "var(--mark-1)"};

async function get(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
  return res.json();
}
async function post(url, body) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
                                body: JSON.stringify(body || {})});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

/* ---- shared renderers ---- */
function rosterTable(players, opts) {
  const wrap = document.createElement("div");
  wrap.className = "scroller";
  wrap.innerHTML = `<table><thead><tr>
      <th data-k="name">player</th>
      <th data-k="hands" class="num">hands</th><th data-k="archetype">read</th>
      <th data-k="skill" class="num">skill</th>
      <th data-k="exploitability" class="num">worth bb/100</th>
      <th data-k="top_leak">biggest leak</th>
    </tr></thead><tbody></tbody></table>`;
  const body = $("tbody", wrap);
  let sort = {key: "skill", dir: -1};
  function draw() {
    const rows = [...players].sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      const cmp = (typeof x === "number" && typeof y === "number")
        ? x - y : String(x).localeCompare(String(y));
      return cmp * sort.dir;
    });
    body.innerHTML = "";
    for (const p of rows) {
      const tr = document.createElement("tr");
      const isHero = opts && opts.heroId != null && p.player_id === opts.heroId;
      if (isHero) tr.className = "hero-row-marker hero-scope";
      if (opts && opts.onClick && p.player_id != null) {
        tr.className = (tr.className ? tr.className + " " : "") + "clickable";
        tr.tabIndex = 0;
        tr.setAttribute("role", "button");
        tr.onkeydown = e => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); opts.onClick(p); }
        };
        tr.onclick = () => opts.onClick(p);
      }
      const shown = (p.session_names && p.session_names.length && p.db_name)
        ? p.session_names.join(" / ") : p.name;
      const linkBits = [];
      if (p.db_name) linkBits.push(`database: ${esc(p.db_name)}`);
      else if (p.session_names && p.session_names.length)
        linkBits.push(`as ${p.session_names.map(n => `\u201c${esc(n)}\u201d`).join(", ")}`);
      tr.innerHTML = `
        <td><span class="name">${esc(shown)}</span>${
            isHero ? '<span class="tag hero-tag">you</span>' : ""}
            ${linkBits.length
              ? `<div class="small muted">${linkBits.join(" \u00b7 ")}</div>` : ""}</td>
        <td class="num">${p.hands}</td>
        <td><span class="tag arch ${p.confidence >= 0.5 ? "on" : ""}">${esc(p.archetype)}</span>
            <div class="small muted">${fmtPct(p.confidence)} sure</div></td>
        <td class="num"></td>
        <td class="num worth${p.exploitability > 10 ? " big" : ""}">${
          p.exploitability ? p.exploitability.toFixed(1) : "\u2014"}</td>
        <td class="small leakcell">${p.top_leak ? esc(p.top_leak)
          : '<span class="muted">nothing yet</span>'}</td>`;
      const holder = document.createElement("div");
      holder.style.cssText = "display:flex;gap:8px;align-items:center;justify-content:flex-end";
      const label = document.createElement("span");
      label.textContent = p.skill.toFixed(0);
      // Mapped to the observed domain, not 0-100: anchored at zero the whole
    // roster looked equally full -- p10 to p90 differed by 14px of bar.
    holder.append(bar(Math.max(0, p.skill - 40), 55, "var(--mark-3)", 66), label);
      // Sample quality moved onto the hands count as a tooltip: it qualifies
      // that number and nothing else, so it does not need its own line on
      // every row.
      const hcell = tr.children[1];
      if (hcell) {
        hcell.classList.add("q-" + String(p.sample_quality).split(" ")[0]);
        bindTip(hcell, `<b>${esc(p.sample_quality)}</b><br>${termTip(p.sample_quality)}`);
      }
      /* An unconfirmed read still belongs in the column -- "none clears the
         bar" left the weakest player at the table looking like the safest.
         The marker says which kind of claim it is. */
      if (p.top_leak && p.top_leak_status !== "confirmed") {
        const flag = document.createElement("span");
        flag.className = "flag";
        flag.textContent = "!";
        bindTip(flag, `<span class="hl">${p.top_leak_status === "watch"
          ? "not confirmed" : "from the rating, not a frequency"}</span><br>
          ${esc(p.top_leak_note)}`);
        $(".leakcell", tr).appendChild(flag);
      } else if (p.top_leak) {
        $(".leakcell", tr).appendChild(info(esc(p.top_leak_note)));
      }
      tr.children[3].appendChild(holder);
      bindTip(holder, `<b>${esc(p.skill_tier)}</b> ${p.skill.toFixed(0)}/100<br>
        <span class="muted">confidence ${fmtPct(p.skill_confidence)}</span>`);
      body.appendChild(tr);
    }
    wrap.querySelectorAll("th").forEach(th => {
      th.classList.toggle("sorted", th.dataset.k === sort.key);
      th.classList.toggle("desc", th.dataset.k === sort.key && sort.dir < 0);
    });
  }
  wrap.querySelectorAll("th").forEach(th => th.onclick = () => {
    const k = th.dataset.k;
    sort = {key: k, dir: sort.key === k ? -sort.dir : (k === "name" ? 1 : -1)};
    draw();
  });
  draw();
  return wrap;
}

function profileCard(p, opts) {
  opts = opts || {};
  const isHero = opts.heroId != null && p.player_id === opts.heroId;
  const card = document.createElement("div");
  card.className = isHero ? "dash hero-scope" : "dash";
  const leaks = p.leaks;
  // Full-width tiles are placed last: dropped mid-grid they force a row break
  // and strand whatever tile precedes them alone on a line.
  const wideTiles = [];

  const head = document.createElement("div");
  head.className = "panel wide";
  head.innerHTML = `
    <div class="hero-row">
      <div class="hero-name">
        <div class="hero">${esc(p.name || p.archetype)}</div>
        <div class="hero-sub">${esc(p.archetype)}</div>
      </div>
      <div class="skill-badge" id="skill-badge">
        <span class="score">${p.skill.score.toFixed(0)}</span>
        <span class="of">/100</span>
      </div>
    </div>
    <div class="read-meta" id="read-meta"></div>
    <div class="read-copy">
      <p class="summary">${esc(p.summary)}</p>
      <p class="plan">${esc(p.plan)}</p>
    </div>`;
  if (isHero) $(".hero-sub", head).insertAdjacentHTML(
    "afterbegin", '<span class="hero-badge">you</span> ');
  card.appendChild(head);

  const cols = document.createElement("div");
  cols.className = "dash-cols wide";
  card.appendChild(cols);

  const skillBox = document.createElement("div");
  skillBox.className = "panel";
  skillBox.innerHTML =
    `<h2>Skill breakdown <span class="muted" style="font-weight:400">\u00b7 ` +
    `${esc(p.skill.tier)}</span></h2><div class="skill-side" id="skill-side"></div>`;

  const badge = $("#skill-badge", head);
  bindTip(badge, `<b>${esc(p.skill.tier)}</b> ${p.skill.score.toFixed(0)}/100<br>
    <span class="muted">confidence ${fmtPct(p.skill.confidence)}</span>
    ${p.skill.observed_bb100 == null ? ""
      : `<br><span class="muted">${p.skill.observed_bb100.toFixed(1)} bb/100 observed</span>`}`);

  const meta = $("#read-meta", head);
  const line = document.createElement("span");
  line.append(document.createTextNode(`${p.hands} hands`));
  line.appendChild(info(`<b>${esc(p.sample_quality)}</b><br>${termTip(p.sample_quality)}`));
  const conf = document.createElement("span");
  conf.style.marginLeft = "12px";
  conf.append(document.createTextNode(
    `${esc(p.archetype)} ${fmtPct(p.archetype_confidence)} sure`));
  conf.appendChild(info(`${termTip("confidence")}<br><br>
    <span class="hl">also plausibly</span><br>${
      p.archetype_mix.slice(1, 4).map(([n, v]) => `${esc(n)} ${fmtPct(v)}`).join("<br>")}`));
  meta.append(line, conf);
  if (p.contributions && Object.keys(p.contributions).length > 1) {
    const where = document.createElement("span");
    where.style.marginLeft = "12px";
    where.textContent = p.regime_label;
    where.appendChild(info(`<span class="hl">one read, several table sizes</span><br>
      Each table's hands are measured against that table's own norms, then
      pooled. Shown on ${esc(p.regime_label)} terms, where they play most.<br>
      ${esc(p.table_mix || "")}`));
    meta.appendChild(where);
  }

  // Seven bars spanning 77-100 rank a player without telling you anything.
  // Each component now carries what it measures, the figure behind it, and
  // whether it is the part of their game that is actually costing them.
  const skillSide = $("#skill-side", skillBox);
  skillSide.innerHTML = "";
  const comps = p.skill_components || p.skill.components;
  for (const c of [...comps].sort((a, b) => a.score - b.score)) {
    const row = document.createElement("div");
    row.className = "comp" + (c.weak ? " weak" : "");
    row.innerHTML = `
      <div class="comp-head">
        <span class="comp-name">${esc(c.name)}</span>
        <span class="comp-score">${c.score.toFixed(0)}</span>
      </div>
      <div class="comp-bar"></div>
      ${c.note ? `<div class="small muted comp-note">${esc(c.note)}</div>` : ""}
      ${c.meaning ? `<div class="small comp-why">${esc(c.meaning)}</div>` : ""}
      <div class="small comp-ev"></div>`;
    $(".comp-bar", row).appendChild(
      bar(c.score, 100, c.weak ? "var(--red)" : "var(--mark-1)", 999));
    if (c.measures) {
      $(".comp-name", row).appendChild(info(
        `<b>${esc(c.name)}</b><br>${esc(c.measures)}<br>
         <span class="muted">counts ${c.weight}x toward the rating</span>`));
    }
    // The same evidence route the leaks use: a rating you cannot check against
    // the hands is just an opinion with a number on it.
    const ev = $(".comp-ev", row);
    for (const st of (c.stats || [])) {
      if (p.player_id == null) break;
      const b = document.createElement("button");
      b.className = "linkbtn";
      const label = statLabel(st, p.rows);
      b.textContent = label;
      b.onclick = () => showEvidence(p.player_id, st, `${c.name} \u00b7 ${label}`);
      if (ev.childNodes.length) ev.appendChild(document.createTextNode(" \u00b7 "));
      ev.appendChild(b);
    }
    skillSide.appendChild(row);
  }

  const doBox = document.createElement("div");
  doBox.className = "panel";
  doBox.innerHTML = `<div class="spread"><h2>What to Do</h2>
      <span class="small muted worth"></span></div><div class="leaks"></div>`;
  const worthLabel = $(".worth", doBox);
  worthLabel.append(document.createTextNode(
    leaks.length ? `${p.skill.exploitability_bb100.toFixed(1)} bb/100 available` : ""));
  if (leaks.length) worthLabel.appendChild(info(termTip("available")));
  cols.appendChild(doBox);

  const leakBox = $(".leaks", doBox);
  if (!leaks.length) {
    const nothing = (p.watchlist || []).length || (p.weak_spots || []).length
      ? `<div class="empty">Nothing clears the evidence bar yet. Below is
         what the numbers point at so far.</div>`
      : `<div class="empty">Nothing stands out yet. Play them straight and
         collect more hands.</div>`;
    leakBox.innerHTML = nothing;
  }
  for (const l of leaks) {
    const div = document.createElement("div");
    div.className = "leak";
    div.innerHTML = `
      <div class="leak-head">
        <div class="headline"><b>${esc(l.headline)}</b>
          <span class="tag tier">${esc(l.tier)}</span></div>
        <div class="num small muted">${l.severity_bb100.toFixed(2)} bb/100</div></div>
      <div class="leak-advice">${esc(l.do)}</div>
      <div class="small muted numbers"></div>`;
    $(".tier", div).after(info(`${termTip(l.tier)}<br><br>${esc(l.priority)}`));

    const numbers = $(".numbers", div);
    numbers.appendChild(document.createTextNode(
      l.in_words.replace(/seen about .*$/, "").trim()));
    if (p.player_id != null) {
      const link = document.createElement("button");
      link.className = "linkbtn";
      link.textContent = `seen about ${Math.round(l.sample)} times`;
      link.title = "show the hands behind this";
      link.onclick = () => showEvidence(p.player_id, l.stat, l.headline);
      numbers.appendChild(document.createTextNode(" "));
      numbers.appendChild(link);
    } else {
      numbers.appendChild(document.createTextNode(
        ` seen about ${Math.round(l.sample)} times`));
    }
    numbers.appendChild(info(statTip(l.stat, l.headline)));

    // A popup, not an inline <details>: expanding it in place grew this panel
    // after the columns were balanced, which is what threw the layout off. Same
    // sheet the hands open in.
    const whydont = [["Why", l.why], ["Do not", l.dont]].filter(([, t]) => t);
    if (whydont.length) {
      const link = document.createElement("button");
      link.className = "linkbtn how-link";
      link.textContent = "Why, and what not to do";
      link.onclick = () => {
        const modal = $("#modal");
        modal.innerHTML = `<div class="veil"><div class="sheet">
          <div class="spread"><h2 style="margin:0">${esc(l.headline)}</h2>
            <button class="act" id="close">Close</button></div>
          <div class="how-body"></div></div></div>`;
        const how = $(".how-body", modal);
        for (const [label, text] of whydont) {
          const block = document.createElement("div");
          block.className = "howblock";
          block.innerHTML = `<div class="howlabel">${esc(label)}</div>
            <div>${esc(text)}</div>`;
          how.appendChild(block);
        }
        $("#close").onclick = () => { modal.innerHTML = ""; };
      };
      div.appendChild(link);
    }
    leakBox.appendChild(div);
  }

  for (const w of (p.watchlist || [])) {
    const div = document.createElement("div");
    div.className = "leak watch";
    div.innerHTML = `
      <div class="leak-head">
        <div class="headline"><b>${esc(w.headline)}</b>
          <span class="tag tier">watch</span></div>
        <div class="num small muted">${fmtPct(w.confidence)} sure</div></div>
      <div class="small muted numbers">${esc(w.in_words)}</div>`;
    $(".tier", div).after(info(termTip("watch")));
    leakBox.appendChild(div);
  }

  if (false && (p.weak_spots || []).length) {
    const weak = document.createElement("div");
    weak.className = "leak weakspots";
    weak.innerHTML = `<div class="headline"><b>Weakest parts of their game</b>
        <span class="tag">from the rating</span></div>
      <div class="small muted" style="margin:4px 0 8px">
        Not priced leaks \u2014 where their game is thinnest.</div>`;
    for (const spot of p.weak_spots) {
      const row = document.createElement("div");
      row.className = "metric";
      const label = document.createElement("span");
      label.className = "small";
      label.textContent = spot.name + (spot.note ? ` \u2014 ${spot.note}` : "");
      const val = document.createElement("span");
      val.className = "small muted"; val.style.textAlign = "right";
      val.textContent = spot.score.toFixed(0);
      row.append(label, bar(spot.score, 100, "var(--mark-2)", 150), val);
      if (spot.meaning) bindTip(row, `<b>${esc(spot.name)}</b><br>${esc(spot.meaning)}`);
      weak.appendChild(row);
    }
    leakBox.appendChild(weak);
  }

  for (const c of (p.combinations || [])) {
    const block = document.createElement("div");
    block.className = "leak";
    block.innerHTML = `<div class="headline"><b>${esc(c.headline)}</b>
      <span class="tag">these compound</span></div>
      <div class="leak-advice">${esc(c.body)}</div>`;
    leakBox.appendChild(block);
  }

  // How they play *you*. Rendered only when there is something in it: most
  // players against most opponents have no adjustment, and an empty panel
  // takes a column off a screen that is meant to be read mid-hand.
  const adjustments = p.adjustments || [];
  if (adjustments.length) {
    const adjBox = document.createElement("div");
    // Full width, below the two columns rather than inside the height-packed
    // masonry: a short reads-only panel could never balance against the tall
    // skill breakdown without leaving dead space on one side or the other.
    adjBox.className = "panel wide adjust hero-scope";
    adjBox.innerHTML = `<h2 style="margin-bottom:6px">Against You<span class="hero-badge">hero</span></h2>
      <div class="small muted" style="margin:0 0 14px">
        Measured against how they play everybody else, not against the
        field.</div>`;
    // The mark rides the title itself, like every other panel header (see the
    // hero dashboard): .info's own 5px offset places it, no wrapper needed.
    $("h2", adjBox).appendChild(info(termTip("adjustment")));
    const adjGrid = document.createElement("div");
    adjGrid.className = "adjust-grid";
    adjBox.appendChild(adjGrid);
    for (const a of adjustments) {
      const div = document.createElement("div");
      div.className = "leak";
      div.innerHTML = `
        <div class="leak-head">
          <div class="headline"><b>${esc(a.behaviour)}</b></div>
          <div class="num small muted">${fmtPct(Math.min(a.confidence, 0.99))} sure</div>
        </div>
        <div class="small muted numbers"></div>`;
      for (const [label, value, color, term] of [
            ["against you", a.versus, "var(--mark-3)", "against you"],
            ["otherwise", a.baseline, "var(--mark-1)", "otherwise"]]) {
        const row = document.createElement("div");
        row.className = "metric";
        const name = document.createElement("span");
        name.className = "small"; name.textContent = label;
        const val = document.createElement("span");
        val.className = "small muted"; val.style.textAlign = "right";
        val.textContent = fmtPct(value);
        row.append(name, bar(value, 1, color, 150), val);
        bindTip(row, termTip(term));
        div.insertBefore(row, $(".numbers", div));
      }
      const numbers = $(".numbers", div);
      const seen = `${Math.round(a.sample)} against you`;
      if (p.player_id != null) {
        // Opens on the against-you slice rather than its parent, so the hands
        // shown are the ones the read is actually about.
        const link = document.createElement("button");
        link.className = "linkbtn";
        link.textContent = seen;
        link.title = "show the hands behind this";
        link.onclick = () => showEvidence(p.player_id, a.evidence_stat, a.behaviour);
        numbers.appendChild(link);
      } else {
        numbers.appendChild(document.createTextNode(seen));
      }
      numbers.appendChild(document.createTextNode(
        ` · ${Math.round(a.baseline_sample)} against everybody else`));
      numbers.appendChild(info(statTip(a.stat, a.behaviour)));
      adjGrid.appendChild(div);
    }
    wideTiles.push(adjBox);
  }

  if (opts.narrate) {
    const suggestBox = document.createElement("div");
    suggestBox.className = "panel wide";
    suggestBox.innerHTML = `<h2>suggested exploits</h2>
      <div class="small muted" style="margin:-6px 0 12px">
        Not measured reads \u2014 check them against the hands.</div>`;
    const narrateBox = document.createElement("div");
    narrateBox.className = "narrate";
    suggestBox.appendChild(narrateBox);
    wideTiles.push(suggestBox);
    buildNarrator(narrateBox, p);
  }

  const tells = p.timing_tells || [];
  // Render only when some cell actually has a tell -- a grid of "not enough
  // data" is noise wearing a panel's clothes.
  if (tells.some(c => c.n >= 5 && !/no clear tell|not enough/i.test(c.label || ""))) {
    const timingBox = document.createElement("div");
    timingBox.className = "panel wide";
    const headRow = document.createElement("div");
    headRow.className = "spread";
    const title = document.createElement("div");
    title.className = "headline";
    const h2 = document.createElement("h2");
    h2.style.margin = "0";
    h2.textContent = "Timing Tells";
    const flag = document.createElement("span");
    flag.className = "flag";
    flag.textContent = "!";
    bindTip(flag, `<span class="hl">use with caution</span><br>
      Timing is noisy online. Each cell is the <em>share</em> of that action
      at this pace, plus whether they won / went to showdown / folded next
      <em>differently</em> than after the same action at normal pace. Use it
      to break ties \u2014 never as the whole basis of a decision.`);
    title.append(h2, flag);
    headRow.appendChild(title);
    const note = document.createElement("span");
    note.className = "small muted";
    note.textContent = "share of action + outcome vs normal pace";
    headRow.appendChild(note);
    timingBox.appendChild(headRow);

    const byKey = Object.fromEntries(
      tells.map(c => [`${c.pace}:${c.street}:${c.action}`, c]));
    for (const street of ["flop", "turn"]) {
      const block = document.createElement("div");
      block.className = "timing-street";
      block.innerHTML = `<div class="street-label">${street}</div>`;
      const grid = document.createElement("div");
      grid.className = "timing-grid";
      grid.innerHTML = `<div class="corner"></div>
        <div class="colhead">check</div>
        <div class="colhead">call</div>
        <div class="colhead">raise</div>`;
      for (const pace of ["snap", "tank"]) {
        const rowhead = document.createElement("div");
        rowhead.className = "rowhead";
        rowhead.textContent = pace;
        grid.appendChild(rowhead);
        for (const action of ["check", "call", "aggro"]) {
          const cell = byKey[`${pace}:${street}:${action}`] || {
            n: 0, total: 0, share: null, label: "Not enough data",
            read: "Need more timed actions."};
          const div = document.createElement("div");
          const quiet = cell.n < 5 || /no clear tell/i.test(cell.label || "");
          div.className = "timing-cell" + (quiet ? " thin" : "");
          const share = cell.share == null ? ""
            : `${Math.round(100 * cell.share)}% of ${cell.total}`;
          const nLine = cell.n
            ? `${cell.n} timed${share ? ` \u00b7 ${share}` : ""}`
            : "no data yet";
          div.innerHTML = `<div class="tell">${esc(cell.label)}</div>
            <div class="read">${esc(cell.read)}</div>
            <div class="n">${nLine}</div>`;
          grid.appendChild(div);
        }
      }
      block.appendChild(grid);
      timingBox.appendChild(block);
    }
    wideTiles.push(timingBox);
  }

  cols.appendChild(skillBox);

  const detail = document.createElement("div");
  detail.className = "panel";
  detail.innerHTML = `<h2>Key numbers</h2><div class="detail-body"></div>`;
  cols.appendChild(detail);
  for (const tile of wideTiles) card.appendChild(tile);
  requestAnimationFrame(() => balanceColumns(cols));
  const body = $(".detail-body", detail);

  // Six headline numbers, the rest one click away. Rendering all seventeen
  // open made this panel taller than the read, the plan and the exploits put
  // together, and it was the whole reason the page scrolled.
  const HEADLINE = ["vpip", "pfr", "three_bet", "fold_to_three_bet",
                    "cbet:flop", "fold_vs_bet:flop"];
  const rank = (r) => {
    const i = HEADLINE.indexOf(r.stat);
    return i === -1 ? HEADLINE.length : i;
  };
  const TIMING = /^(tank_fold|snap_call)(:|$)/;
  const ordered = [...p.rows]
    .filter((r) => !TIMING.test(r.stat))
    .sort((a, b) => rank(a) - rank(b));
  const headline = ordered.filter((r) => HEADLINE.includes(r.stat));
  const rest = ordered.filter((r) => !HEADLINE.includes(r.stat));

  const makeTable = () => {
    const table = document.createElement("div");
    table.className = "scroller";
    table.innerHTML = `<table><thead><tr><th>stat</th>
        <th style="width:40%">0% \u2014 100%</th>
        <th class="num">estimate</th><th class="num">sample</th></tr></thead>
      <tbody></tbody></table>`;
    return table;
  };
  const fill = (table, rows) => {
    const tbody = $("tbody", table);
    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="label"></td><td></td>
                      <td class="num">${fmtPct(row.value)}</td>
                      <td class="num small muted">${Math.round(row.opps)}</td>`;
      const label = $(".label", tr);
      label.appendChild(document.createTextNode(row.label));
      label.appendChild(info(statTip(row.stat, row.label, row)));
      tr.children[1].appendChild(statRow(row));
      tbody.appendChild(tr);
    }
  };

  const first = makeTable();
  fill(first, headline.length ? headline : ordered);
  body.appendChild(first);
  if (rest.length) {
    // The rest open in the sheet popup rather than an inline <details>, so
    // this panel never changes height after the columns are balanced.
    const link = document.createElement("button");
    link.className = "linkbtn how-link";
    link.textContent = `See all ${ordered.length} numbers`;
    link.onclick = () => {
      const modal = $("#modal");
      modal.innerHTML = `<div class="veil"><div class="sheet">
        <div class="spread"><h2 style="margin:0">Key numbers</h2>
          <button class="act" id="close">Close</button></div>
        <div class="modal-numbers"></div></div></div>`;
      const full = makeTable();
      fill(full, ordered);
      $(".modal-numbers", modal).appendChild(full);
      $("#close").onclick = () => { modal.innerHTML = ""; };
    };
    body.appendChild(link);
  }
  return card;
}

function buildNarrator(box, profile) {
  const actions = document.createElement("div");
  actions.className = "narrate-actions";
  const button = document.createElement("button");
  button.className = "act small";
  button.textContent = "Generate additional exploits";
  const toggle = document.createElement("button");
  toggle.className = "act small";
  toggle.disabled = true;
  toggle.textContent = "Hide";
  const out = document.createElement("div");
  out.className = "narration";
  actions.append(button, toggle);
  box.append(actions, out);

  let visible = true;
  toggle.onclick = () => {
    visible = !visible;
    out.classList.toggle("hidden", !visible);
    toggle.textContent = visible ? "Hide" : "Show";
  };

  button.onclick = async () => {
    button.disabled = true;
    toggle.disabled = true;
    const original = button.textContent;
    button.textContent = "writing\u2026";
    out.classList.remove("hidden");
    visible = true;
    toggle.textContent = "Hide";
    try {
      const result = await post("/api/narrate", {profile: profile});
      out.innerHTML = `${renderBullets(result.text)}
        <div class="small muted" style="margin-top:8px">suggested by
          ${esc(result.model)} from the numbers on this page \u2014 it is given
          the computed profile and cannot state a figure the profile did not
          produce. These are not measured reads: check them against the hands
          before trusting them.</div>`;
      button.textContent = "Generate again";
      toggle.disabled = false;
    } catch (err) {
      out.innerHTML = `<div class="small err">${esc(err.message)}</div>`;
      button.textContent = original;
      toggle.disabled = true;
    }
    button.disabled = false;
  };
}

/* The model returns bullets. Render them as a list rather than a wall of
   text, and fall back to paragraphs if it ignored the instruction. */
function renderBullets(text) {
  const lines = text.split("\n").map(l => l.trim()).filter(Boolean);
  const bullets = lines.filter(l => /^[-*\u2022]\s+/.test(l));
  if (bullets.length < 2) {
    return lines.map(l => `<p>${esc(l)}</p>`).join("");
  }
  const items = bullets
    .map(l => `<li>${esc(l.replace(/^[-*\u2022]\s+/, ""))}</li>`).join("");
  return `<ul class="suggested">${items}</ul>`;
}

/* ---- a strip of player tabs over one profile at a time ---- */
function playerTabs(profiles, container, opts) {
  container.innerHTML = "";
  if (!profiles.length) {
    container.innerHTML = `<div class="panel"><div class="empty">No profiles.</div></div>`;
    return;
  }
  if (profiles.length === 1) {          // a strip of one is just clutter
    container.appendChild(profileCard(profiles[0], opts));
    return;
  }
  const strip = document.createElement("div");
  strip.className = "ptabs";
  const body = document.createElement("div");
  container.append(strip, body);

  let current = 0;
  function show(i) {
    current = i;
    [...strip.children].forEach((b, j) => b.classList.toggle("on", i === j));
    body.innerHTML = "";
    body.appendChild(profileCard(profiles[i], opts));
  }
  profiles.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "ptab";
    const shown = (p.session_names && p.session_names.length && p.db_name)
      ? p.session_names.join(" / ") : p.name;
    const meta = [esc(p.regime_label), `${p.hands}h`];
    if (p.db_name) meta.push(`database: ${esc(p.db_name)}`);
    b.innerHTML = `<span>${esc(shown)}</span>
      <span class="meta">${meta.join(" \u00b7 ")}</span>`;
    b.onclick = () => show(i);
    strip.appendChild(b);
  });
  show(0);
}

async function readFiles(list) {
  const payload = [];
  for (const f of [...list]) payload.push({name: f.name, content: await f.text()});
  return payload;
}

/* Import straight into the database: one session for the whole batch, the
   identity questions asked once across all of it, then a single commit. */
function showBusy(text) {
  const modal = $("#modal");
  modal.innerHTML = `<div class="veil busy"><div class="sheet busy-sheet">
    <div class="spinner" aria-hidden="true"></div>
    <div><b id="busy-text"></b></div>
  </div></div>`;
  $("#busy-text").textContent = text;
  return (next) => { const el = $("#busy-text"); if (el) el.textContent = next; };
}

async function importFiles(list, status, done) {
  const files = [...list];
  if (!files.length) return;
  status.textContent = "";
  const setBusy = showBusy(`Reading ${files.length} file(s)\u2026`);
  try {
    const payload = await readFiles(files);
    setBusy(`Parsing ${files.length} file(s)\u2026`);
    const data = await post("/api/upload", {files: payload});
    const skipped = (data.rejected || []).length
      ? ` \u00b7 skipped ${data.rejected.map(r => r.name).join(", ")}` : "";
    setBusy(`Parsed ${data.hands} hands \u2014 saving and rebuilding profiles\u2026`);
    const finish = async (answers) => {
      setBusy("Saving and rebuilding profiles\u2026");
      const r = await post(`/api/session/${data.token}/commit`,
                           answers ? {answers} : {});
      $("#modal").innerHTML = "";
      // Inline, not a modal: after a batch you want to be looking at the
      // roster you just changed, not dismissing a box in front of it.
      const bits = [`${r.hands_new} new hand(s) stored`];
      if (r.duplicates) bits.push(`${r.duplicates} already known`);
      if (r.unusable) bits.push(`${r.unusable} unreadable`);
      if (r.players_new) bits.push(`${r.players_new} new player(s)`);
      if (r.merged) bits.push(`${r.merged} merge(s)`);
      if (r.priors_fitted) {
        bits.push(`priors refitted from ${r.priors_fitted.players} players`);
      }
      status.innerHTML = esc(bits.join(" \u00b7 ")) + esc(skipped) +
        (r.blocked || []).map(b => `<div class="err">${esc(b)}</div>`).join("");
      if (done) done(status.innerHTML);
    };
    if (data.questions && data.questions.length && !data.answered) {
      $("#modal").innerHTML = "";
      askIdentity(data.token, data.questions, finish);
    } else {
      await finish(null);
    }
  } catch (err) {
    $("#modal").innerHTML = "";
    status.innerHTML = `<span class="err">${esc(err.message)}</span>`;
  }
}

/* Binds whichever import controls are on the page. Both states of the
   Database tab share one handler so they cannot drift apart. */
function wireImport() {
  const input = $("#db-file"), status = $("#db-status"), drop = $("#db-drop");
  if (!input || !status) return;
  const go = (files) => importFiles(files, status, async (summary) => {
    state.player = null;
    await viewPlayers();
    const after = $("#db-status");
    if (after && summary) after.innerHTML = summary;
  });
  input.onchange = () => go(input.files);
  const button = $("#db-add");
  if (button && drop) {
    button.onclick = () => { drop.hidden = false; input.click(); };
  }
  if (drop) {
    drop.onclick = () => input.click();
    drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = e => {
      e.preventDefault(); drop.classList.remove("over");
      go(e.dataTransfer.files);
    };
  }
  // Dropping anywhere on the panel works too: hunting for a target is friction
  // on the one action this tab exists for.
  const panel = status.closest(".panel");
  if (panel && drop) {
    panel.ondragover = e => { e.preventDefault(); drop.hidden = false; };
    panel.ondrop = e => {
      e.preventDefault();
      go(e.dataTransfer.files);
    };
  }
}

/* ---- sittings, derived from the database ---- */
function whenLabel(ms, withTime) {
  if (!ms) return "";
  const d = new Date(ms);
  // The year matters: these sittings span months, so "Aug 12" and "Oct 29"
  // are ambiguous without it.
  const thisYear = d.getFullYear() === new Date().getFullYear();
  const day = d.toLocaleDateString([], {weekday: "short", day: "numeric",
    month: "short", ...(thisYear ? {} : {year: "numeric"})});
  return withTime ? `${day} \u00b7 ${d.toLocaleTimeString([],
    {hour: "2-digit", minute: "2-digit"})}` : day;
}

async function viewSessions() {
  const view = $("#view");
  view.innerHTML = `<div class="panel"><div class="empty">loading\u2026</div></div>`;
  const sessions = await get("/api/sessions");
  if (!sessions.length) {
    view.innerHTML = `<div class="panel"><h2>no sittings yet</h2>
      <p class="muted">Add hand histories on the Database tab.</p></div>`;
    return;
  }
  if (state.sessionId == null) state.sessionId = sessions[0].id;
  // The list lives beside the detail, not above it: twenty sittings pushed the
  // thing you came to read off the bottom of the screen, and switching meant
  // scrolling back up every time.
  view.innerHTML = `<div class="sess-layout${state.sessListHidden ? " collapsed" : ""}"
      id="sess-layout">
      <div class="panel sess-list">
        <div class="spread"><h2 style="margin:0">sittings</h2>
          <button class="iconbtn" id="sess-toggle"
            title="hide the list">\u00ab</button></div>
        <div id="sess-rows"></div>
      </div>
      <div class="panel sess-main">
        <h2 id="sess-title">who played, and how</h2>
        <div id="sess-body"></div>
      </div>
    </div>`;
  const rows = $("#sess-rows");
  for (const sess of sessions) {
    const item = document.createElement("button");
    item.className = "sess-item" + (sess.id === state.sessionId ? " on" : "");
    const hrs = Math.floor(sess.minutes / 60), mins = sess.minutes % 60;
    item.innerHTML = `<span class="sess-when">${esc(whenLabel(sess.started_at, true))}</span>
      <span class="small muted">${hrs ? hrs + "h " : ""}${mins}m \u00b7 ${
        sess.hands} hands \u00b7 ${sess.players}p</span>`;
    item.onclick = () => { state.sessionId = sess.id; viewSessions(); };
    rows.appendChild(item);
  }
  const layout = $("#sess-layout"), toggle = $("#sess-toggle");
  toggle.onclick = () => {
    state.sessListHidden = !state.sessListHidden;
    layout.classList.toggle("collapsed", state.sessListHidden);
    toggle.textContent = state.sessListHidden ? "\u00bb" : "\u00ab";
    toggle.title = state.sessListHidden ? "show the list" : "hide the list";
  };
  if (state.sessListHidden) { toggle.textContent = "\u00bb"; }
  const chosen = sessions.find(x => x.id === state.sessionId);
  if (chosen) {
    $("#sess-title").textContent =
      `${whenLabel(chosen.started_at, true)} \u00b7 ${chosen.hands} hands`;
  }
  await drawSession(state.sessionId);
}

async function drawSession(id) {
  const body = $("#sess-body");
  body.innerHTML = `<div class="empty">reading the sitting\u2026</div>`;
  const data = await get(`/api/session-detail?id=${id}`);
  body.innerHTML = "";
  for (const p of data.players) {
    const div = document.createElement("div");
    div.className = "leak";
    const netTxt = p.net_bb > 0 ? `+${p.net_bb}` : `${p.net_bb}`;
    div.innerHTML = `<div class="leak-head">
        <div class="sess-who"><b class="linkish">${esc(p.name)}</b>
          <span class="tag arch ${p.confidence >= 0.5 ? "on" : ""}">${esc(p.archetype)}</span>
          <span class="small muted">${p.hands} hands \u00b7 ${esc(p.regime_label || "")}</span>
        </div>
        <div class="num small"><b>${netTxt}</b> <span class="muted">bb \u00b7 ${p.skill} skill</span></div></div>
      <div class="sess-deltas"></div>`;
    $("b", div).onclick = () => switchTab("players", p.player_id);
    const box = $(".sess-deltas", div);
    if (!p.deltas.length) {
      box.innerHTML = `<div class="small muted">No trend yet \u2014 not enough
        hands at this table size outside this sitting to say what is usual.</div>`;
    } else {
      // One table size at a time, picked with a tab, rather than every table
      // size stacked under its own heading. Rendering a row per (stat,
      // regime) put VPIP on screen two or three times with the table size in
      // small print, which read as a duplicate rather than as two different
      // games -- stacked headings fixed the duplication but still made you
      // scroll past every table size to find the one you sat at. A player
      // who only played one table size gets no tabs at all.
      const byRegime = new Map();
      for (const d of p.deltas) {
        const key = d.regime_label || d.regime || "";
        if (!byRegime.has(key)) byRegime.set(key, []);
        byRegime.get(key).push(d);
      }
      const regimes = [...byRegime.keys()];
      const rows = document.createElement("div");
      const drawRows = label => {
        rows.innerHTML = "";
        for (const d of byRegime.get(label)) {
          const row = document.createElement("div");
          row.className = "sess-delta";
          const up = d.delta > 0;
          row.innerHTML = `<span>${esc(statLabel(d.stat, null))}</span>
            <span class="num">${fmtPct(d.session)}</span>
            <span class="small ${up ? "up" : "down"}">${up ? "\u25b2" : "\u25bc"}${
              Math.abs(Math.round(d.delta * 100))}pp</span>
            <span class="small muted">usually ${fmtPct(d.usual)}</span>`;
          rows.appendChild(row);
        }
      };
      if (regimes.length > 1) {
        const tabs = document.createElement("div");
        tabs.className = "sess-regime-tabs";
        regimes.forEach((label, i) => {
          const b = document.createElement("button");
          b.className = "sess-regime-tab" + (i === 0 ? " on" : "");
          b.textContent = label;
          b.onclick = () => {
            tabs.querySelectorAll(".sess-regime-tab").forEach(x => x.classList.remove("on"));
            b.classList.add("on");
            drawRows(label);
          };
          tabs.appendChild(b);
        });
        box.appendChild(tabs);
      }
      box.appendChild(rows);
      drawRows(regimes[0]);
    }
    body.appendChild(div);
  }
}

/* ---- tab 3: hero ---- */
const RANK_ORDER = "AKQJT98765432";

function rangeGrid(grid) {
  const wrap = document.createElement("div");
  wrap.className = "range-grid";
  for (let i = 0; i < 13; i++) {
    for (let j = 0; j < 13; j++) {
      const hi = RANK_ORDER[i], lo = RANK_ORDER[j];
      const cls = i === j ? hi + lo : i < j ? hi + lo + "s" : lo + hi + "o";
      const g = (grid || {})[cls];
      const dealt = g ? g.dealt : 0, played = g ? g.played : 0;
      const pct = dealt ? played / dealt : 0;
      const cell = document.createElement("div");
      cell.className = "range-cell" + (pct > 0.5 ? " dark-text" : "");
      cell.style.background =
        `color-mix(in oklab, var(--ink) ${Math.round(pct * 100)}%, var(--panel))`;
      cell.textContent = cls;
      bindTip(cell, `<b>${cls}</b><br>played ${dealt ? fmtPct(pct) : "—"}` +
        (dealt ? ` (${played} of ${dealt})` : " -- never dealt"));
      wrap.appendChild(cell);
    }
  }
  return wrap;
}

const POSITION_ORDER =
  ["UTG", "UTG1", "UTG2", "MP", "MP1", "MP2", "LJ", "HJ", "CO", "BTN", "SB", "BB"];

/* A graded report (fold grades / missed value) shares one shape: graded,
   flagged, rate, by_street, by_texture, worst[]. One renderer for both. */
function renderGradedSection(el, section, opts) {
  if (!section || !section.graded) {
    el.innerHTML = `<div class="small muted">${esc(opts.emptyText)}</div>`;
    return;
  }
  const summary = document.createElement("p");
  summary.innerHTML = `<b>${section.graded}</b> ${esc(opts.noun)} graded, <b>${
      section.flagged}</b> (${fmtPct(section.rate)}) ${esc(opts.verdict)}`;
  el.appendChild(summary);

  const streets = document.createElement("p");
  streets.className = "small muted";
  streets.textContent = "by street:   " + Object.entries(section.by_street)
    .map(([s, v]) => `${s} ${fmtPct(v.flagged / v.graded)}`).join("   ");
  el.appendChild(streets);

  const textures = document.createElement("p");
  textures.className = "small muted";
  textures.textContent = "by texture:  " + Object.entries(section.by_texture)
    .map(([t, v]) => `${t} ${fmtPct(v.flagged / v.graded)}`).join("   ");
  el.appendChild(textures);

  for (const g of section.worst) {
    const row = document.createElement("div");
    row.className = "fold-row";
    row.innerHTML = `
      <span class="small muted">${esc(g.street)}</span>
      <span class="mono">${g.hole_cards.map(esc).join(" ")}</span>
      <span class="small fold-summary">${esc(g.summary)}</span>
      <span class="small muted">${esc(g.texture)} board</span>`;
    bindTip($(".fold-summary", row), esc(g.in_words));
    row.onclick = () => showReplay(g.hand_id, opts.heroId,
      `${g.street} ${opts.noun.replace(/s$/, "")} -- ${g.hole_cards.join(" ")}`);
    el.appendChild(row);
  }
}

//: Shared by sizing and timing -- both need bets/raises, so a thin sample
//: reads the same way in either one.
const TELL_EMPTY_TEXT = "Not enough postflop bets or raises with a clean line to compare yet.";

/* sizing_tell and timing_tell share a shape too: street -> {strong, weak,
   in_words}. One renderer for both. */
function renderTellSection(el, section) {
  const rows = Object.entries(section || {});
  if (!rows.length) {
    el.innerHTML = `<div class="small muted">${esc(TELL_EMPTY_TEXT)}</div>`;
    return;
  }
  for (const [street, v] of rows) {
    const row = document.createElement("div");
    row.className = "metric";
    row.innerHTML = `<span>${esc(street)}${
        v.is_tell ? '<span class="tag hero-tag" style="margin-left:8px">tell</span>' : ""
      }</span><span class="small">${esc(v.in_words || "")}</span>`;
    el.appendChild(row);
  }
}

/* Hero read through the ordinary villain machinery -- their own priced leaks
   and headline numbers, since hero is just another player in the database.
   Rendered blue (the dash is hero-scoped) and framed for self-coaching. */
function renderHeroSelf(dash, self) {
  if (!self) return;
  const leaks = self.leaks || [];
  if (leaks.length) {
    const box = document.createElement("div");
    box.className = "panel wide";
    box.innerHTML = `<h2>your biggest leaks</h2>
      <div class="small muted" style="margin:-6px 0 14px">
        What a sharp opponent studying you would target, priced in the bb/100
        they could win from it. Fix the dear ones first.</div>
      <div class="leaks"></div>`;
    const host = $(".leaks", box);
    for (const l of leaks) {
      const div = document.createElement("div");
      div.className = "leak";
      div.innerHTML = `
        <div class="leak-head">
          <div class="headline"><b>${esc(l.headline)}</b>
            <span class="tag tier">${esc(l.tier)}</span></div>
          <div class="num small muted">${l.severity_bb100.toFixed(2)} bb/100</div></div>
        <div class="small muted numbers"></div>`;
      const numbers = $(".numbers", div);
      if (self.player_id != null) {
        const link = document.createElement("button");
        link.className = "linkbtn";
        link.textContent = `seen about ${Math.round(l.sample)} times`;
        link.title = "show the hands behind this";
        link.onclick = () => showEvidence(self.player_id, l.stat, l.headline);
        numbers.appendChild(link);
      } else {
        numbers.appendChild(document.createTextNode(
          `seen about ${Math.round(l.sample)} times`));
      }
      numbers.appendChild(info(statTip(l.stat, l.headline)));
      host.appendChild(div);
    }
    dash.appendChild(box);
  }

  const rows = self.rows || [];
  if (rows.length) {
    const box = document.createElement("div");
    box.className = "panel wide";
    box.innerHTML = `<h2>your tendencies</h2>
      <div class="small muted" style="margin:-6px 0 14px">
        Your rates against the pool you were measured against -- the bar is the
        estimate, hover a row for what is typical.</div>
      <div class="scroller"><table><thead><tr><th>stat</th>
        <th style="width:40%">0% \u2014 100%</th>
        <th class="num">you</th><th class="num">sample</th></tr></thead>
        <tbody></tbody></table></div>`;
    const tbody = $("tbody", box);
    for (const row of rows) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td class="label"></td><td></td>
        <td class="num">${fmtPct(row.value)}</td>
        <td class="num small muted">${Math.round(row.opps)}</td>`;
      const label = $(".label", tr);
      label.appendChild(document.createTextNode(row.label));
      label.appendChild(info(statTip(row.stat, row.label, row)));
      tr.children[1].appendChild(statRow(row));
      tbody.appendChild(tr);
    }
    dash.appendChild(box);
  }
}

async function viewHero() {
  const view = $("#view");
  view.innerHTML = `<div class="panel"><div class="empty">reading your own hands…
    the first pass fits a model and can take a while -- after that it is
    instant until you import more hands.</div></div>`;
  let data;
  try {
    data = await get("/api/hero");
  } catch (err) {
    view.innerHTML = `<div class="panel"><h2>hero</h2>
      <p class="err">${esc(err.message)}</p></div>`;
    return;
  }

  const dash = document.createElement("div");
  dash.className = "dash hero-scope";

  const head = document.createElement("div");
  head.className = "panel wide";
  head.innerHTML = `
    <div class="hero-head">
      <h2 style="margin:0">${esc(data.name)}</h2>
      <span class="small muted">cards known on ${fmtPct(data.visibility)} of ${data.hands} hands</span>
    </div>`;
  dash.appendChild(head);

  // Grid and position breakdown side by side: two views of the same range,
  // one by hand the other by seat. dash-cols is the two-panel layout the
  // skill/read split already uses on a player's own page.
  const rangeCols = document.createElement("div");
  rangeCols.className = "dash-cols wide";
  const gridCol = document.createElement("div");
  gridCol.className = "col";
  gridCol.innerHTML = `<div class="panel">
    <h2 id="hero-range-head">preflop range</h2>
    <div id="hero-grid"></div>
    <div class="range-legend"><span>never</span><span class="ramp"></span><span>always</span></div>
  </div>`;
  const posCol = document.createElement("div");
  posCol.className = "col";
  posCol.innerHTML = `<div class="panel">
    <h2>by position</h2>
    <div id="hero-positions"></div>
  </div>`;
  rangeCols.append(gridCol, posCol);
  dash.appendChild(rangeCols);

  const gradesPanel = document.createElement("div");
  gradesPanel.className = "panel wide";
  gradesPanel.innerHTML = `
    <h2 id="hero-grades-head">fold grades &amp; missed value</h2>
    <h3>fold grades</h3>
    <div id="hero-folds"></div>
    <h3>missed value</h3>
    <div id="hero-missed"></div>`;
  dash.appendChild(gradesPanel);

  const tellsPanel = document.createElement("div");
  tellsPanel.className = "panel wide";
  tellsPanel.innerHTML = `
    <h2 id="hero-tells-head">sizing &amp; timing tells</h2>
    <h3>sizing</h3>
    <div id="hero-sizing"></div>
    <h3>timing</h3>
    <div id="hero-timing"></div>`;
  dash.appendChild(tellsPanel);

  const narrowingPanel = document.createElement("div");
  narrowingPanel.className = "panel wide";
  narrowingPanel.innerHTML = `
    <h2 id="hero-narrowing-head">range narrowing</h2>
    <div id="hero-narrowing"></div>`;
  dash.appendChild(narrowingPanel);

  // Hero through the villain machinery: your own priced leaks and tendencies.
  renderHeroSelf(dash, data.self);

  view.innerHTML = "";
  view.appendChild(dash);

  $("#hero-range-head", dash).appendChild(info(
    `Every hand you were ever dealt, not just the ones you played -- something
    only your own export can show. Darker means played (raised or called)
    more often.`));
  $("#hero-grades-head", dash).appendChild(info(
    `${termTip("percentile")}<br><br><span class="hl">fold grades</span> --
    postflop folds, graded against what a bet like that one usually turns out
    to be.<br><br><span class="hl">missed value</span> -- the mirror question,
    asked of checks that could have bet instead.`));
  $("#hero-tells-head", dash).appendChild(info(
    `Does your bet size, or think time, change with the hand behind it?
    Nobody's hand strength is known often enough to ask a villain this --
    yours is known on every bet, not just the ones that reached showdown.`));
  $("#hero-narrowing-head", dash).appendChild(info(
    `A continuing range is supposed to get stronger street by street, as the
    wide ones give up along the way. Average hand strength among hands still
    live, by street, says whether yours does.`));

  $("#hero-grid", dash).appendChild(rangeGrid(data.grid));

  const positions = $("#hero-positions", dash);
  positions.className = "hero-pos";
  const ranges = [...(data.ranges || [])].sort(
    (a, b) => POSITION_ORDER.indexOf(a.position) - POSITION_ORDER.indexOf(b.position));
  for (const r of ranges) {
    if (!r.hands) continue;
    const played = (r.raised + r.called) / r.hands;
    const row = document.createElement("div");
    row.className = "pos-row";
    row.innerHTML = `
      <span class="pos-name">${esc(r.position)}<span class="small muted"> ${r.hands}h</span></span>
      <span class="pos-bar"></span>
      <span class="pos-val small muted">${fmtPct(played)} played</span>`;
    const b = bar(played, 1, "var(--mark-3)", 150);
    b.setAttribute("preserveAspectRatio", "none");
    $(".pos-bar", row).appendChild(b);
    positions.appendChild(row);
  }

  if (data.grade_error) {
    $("#hero-folds", dash).innerHTML = `<div class="small muted">${esc(data.grade_error)}</div>`;
    $("#hero-missed", dash).innerHTML = "";
  } else {
    renderGradedSection($("#hero-folds", dash), data.fold_grades, {
      noun: "folds", heroId: data.hero_id,
      verdict: "had more edge than the bet typically shows",
      emptyText: "Not enough postflop folds with a clean line to grade yet.",
    });
    renderGradedSection($("#hero-missed", dash), data.missed_value, {
      noun: "checks", heroId: data.hero_id,
      verdict: "had more edge than the check typically shows",
      emptyText: "Not enough postflop checks with a clean line to grade yet.",
    });
  }

  renderTellSection($("#hero-sizing", dash), data.sizing);
  renderTellSection($("#hero-timing", dash), data.timing);

  const narrowing = $("#hero-narrowing", dash);
  if (!data.narrowing || !data.narrowing.length) {
    narrowing.innerHTML = `<div class="small muted">Not enough hands reaching
      each street yet.</div>`;
  } else {
    const row = document.createElement("p");
    row.textContent = data.narrowing
      .map(s => `${s.street} ${fmtPct(s.avg_strength)} (${s.hands})`).join("   ");
    narrowing.appendChild(row);
    const strengths = data.narrowing.map(s => s.avg_strength);
    if (strengths.length >= 2) {
      const monotone = strengths.every((v, i) => i === 0 || v >= strengths[i - 1]);
      const note = document.createElement("p");
      note.className = "small muted";
      note.textContent = monotone
        ? "Narrows street by street, as a continuing range should."
        : "Does not narrow monotonically -- worth a look at which street gives it back.";
      narrowing.appendChild(note);
    }
  }
}

/* ---- tab 1: session ---- */
function viewSession() {
  const view = $("#view");
  view.innerHTML = `
    <div class="panel">
      <div class="spread"><h2>read a session</h2>
        <span class="small muted">nothing is saved until you ask</span></div>
      <div class="drop" id="drop">
        <div style="font-size:15px;color:var(--ink)">Drop hand history files here</div>
        <div class="small" style="margin-top:4px">or click to choose \u00b7 PokerNow JSON exports</div>
      </div>
      <input type="file" id="file" multiple accept=".json,.txt" hidden>
      <div id="upload-status" class="small muted" style="margin-top:10px"></div>
    </div>
    <div id="session-body"></div>`;

  const drop = $("#drop"), input = $("#file"), status = $("#upload-status");
  drop.onclick = () => input.click();
  drop.ondragover = e => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = e => {
    e.preventDefault(); drop.classList.remove("over");
    handleFiles(e.dataTransfer.files);
  };
  input.onchange = () => handleFiles(input.files);

  async function handleFiles(list) {
    const files = [...list];
    if (!files.length) return;
    status.textContent = `reading ${files.length} file(s)\u2026`;
    try {
      const payload = [];
      for (const f of files) payload.push({name: f.name, content: await f.text()});
      status.textContent = "parsing\u2026";
      const data = await post("/api/upload", {files: payload});
      state.session = data;
      renderSession();
      if (data.questions && data.questions.length && !data.answered) {
        askIdentity(data.token, data.questions);
      }
      status.innerHTML = data.rejected && data.rejected.length
        ? `<span class="err">skipped: ${data.rejected.map(r => esc(r.name)).join(", ")}</span>`
        : "";
    } catch (err) {
      status.innerHTML = `<span class="err">${esc(err.message)}</span>`;
    }
  }

  if (state.session) renderSession();
}

function renderSession() {
  const data = state.session, box = $("#session-body");
  if (!box) return;
  box.innerHTML = `
    <div class="panel">
      <div class="spread">
        <div><h2>this session</h2>
          <div class="small muted">${data.hands} hands \u00b7
            ${data.files.map(f => esc(f.name)).join(", ")}</div></div>
        <div class="row">
          <span class="small muted" id="save-note">${data.saved
            ? "saved to database" : "not in the database"}</span>
          <button class="act primary" id="save" ${data.saved ? "disabled" : ""}>
            ${data.saved ? "saved" : "Add to database"}</button>
        </div>
      </div>
      ${data.auto_merged ? `<p class="small muted" style="margin:8px 0 0">
        Linked ${data.auto_merged} known player match(es) automatically
        (kept existing database names).</p>` : ""}
      <div id="session-roster" style="margin-top:12px"></div>
    </div>
    <div id="session-profiles"></div>`;
  $("#session-roster").appendChild(rosterTable(data.players, {onClick: null}));
  playerTabs(data.profiles, $("#session-profiles"));
  const save = $("#save");
  if (save && !data.saved) save.onclick = () => commit(data.token);
}

/* ---- identity, settled at upload ---- */
/* Asked when the file lands rather than when it is saved, so the session you
   are reading has already pooled the accounts. Merging also asks what to call
   the result: choosing silently files a player under a name they have stopped
   using. */
/* Accounts that might be one person arrive as pairs, but three accounts that
   are all the same human arrive as three separate questions -- and nothing
   stops you answering them inconsistently. Union the pairs into components so
   each *person* is one decision. Applying the links is still pairwise, which
   is what makes this safe: the co-occurrence guard is enforced per link, so a
   group can never smuggle through a merge the tool would refuse. */
function sideKey(side) {
  if (!side) return "";
  return side.player_id != null ? `db${side.player_id}` : `ac${side.account}`;
}
function groupQuestions(questions) {
  const parent = {};
  const find = k => { while (parent[k] !== k) k = parent[k] = parent[parent[k]]; return k; };
  const union = (a, b) => {
    parent[a] = parent[a] ?? a; parent[b] = parent[b] ?? b;
    parent[find(a)] = find(b);
  };
  for (const q of questions) {
    const a = sideKey(q.left), b = sideKey(q.right);
    parent[a] = parent[a] ?? a; parent[b] = parent[b] ?? b;
    union(a, b);
  }
  const groups = new Map();
  for (const q of questions) {
    const root = find(sideKey(q.left));
    if (!groups.has(root)) groups.set(root, {questions: [], members: new Map()});
    const g = groups.get(root);
    g.questions.push(q);
    for (const side of [q.left, q.right]) g.members.set(sideKey(side), side);
  }
  return [...groups.values()];
}

function sideMeta(side) {
  // Only "already in the database" earns a label; everything else in this
  // dialog came from the files being added, so saying so on every row is noise.
  const bits = [];
  if (side.player_id != null) bits.push("in the database");
  bits.push(`${side.hands || 0} hands`);
  const id = side.account ? String(side.account) : "";
  if (id) bits.push(`<span class="mono">${esc(id.length <= 10 ? id
    : `${id.slice(0, 6)}\u2026${id.slice(-3)}`)}</span>`);
  return bits.join(" \u00b7 ");
}

const STAT_LABELS = {
  "vpip": "VPIP", "pfr": "PFR", "three_bet": "3-bet", "limp": "limps",
  "wtsd": "went to showdown", "wsd": "won at showdown",
  "aggression:flop": "flop aggression", "aggression:turn": "turn aggression",
  "aggression:river": "river aggression",
  "fold_vs_bet:flop": "fold vs flop bet", "fold_vs_bet:turn": "fold vs turn bet",
  "fold_vs_bet:river": "fold vs river bet",
};
const SUITS = {s: "\u2660", h: "\u2665", d: "\u2666", c: "\u2663"};
/* A board is the one thing in this tool that is not a statistic, and reading
   "7s Jd 3c" as text is slower than seeing it. */
function cardsEl(list, opts) {
  const wrap = document.createElement("span");
  wrap.className = "cards-row" + ((opts && opts.small) ? " small-cards" : "");
  for (const raw of (list || [])) {
    const text = String(raw);
    const rank = text.slice(0, -1).replace("T", "10");
    const suit = text.slice(-1).toLowerCase();
    const card = document.createElement("span");
    card.className = "card " + (suit === "h" || suit === "d" ? "red" : "black");
    card.innerHTML = `<span class="r">${esc(rank)}</span><span class="s">${
      SUITS[suit] || esc(suit)}</span>`;
    wrap.appendChild(card);
  }
  return wrap;
}

/* "4 limps out of 2,945 chances" is a number; "essentially never limps, which
   is a strong player's habit" is a read. The rate decides which way, the
   glossary supplies the words. */
function evidenceVerdict(d) {
  if (!d.count) return "No hands where this could have happened yet.";
  const rate = d.rate != null ? d.rate : d.hits / d.count;
  const pop = d.population;
  const pct = Math.round(rate * 100);
  const against = d.compared_to || "the field";
  let scale;
  if (d.hits === 0) scale = "never";
  else if (rate < 0.02) scale = "almost never";
  // Without something to compare against, say the rate and stop. Claiming it
  // is "about as often as the field" when no field frequency was involved
  // describes a comparison that never happened.
  else if (pop == null) scale = "";
  else if (rate > pop * 1.35) scale = `far more than ${against}`;
  else if (rate < pop * 0.65) scale = `far less than ${against}`;
  else scale = `about as often as ${against}`;
  const head = scale ? `${d.hits} of ${d.count} \u2014 ${pct}%, ${scale}.`
                     : `${d.hits} of ${d.count} \u2014 ${pct}%.`;
  return d.reading ? `${head} ${d.reading}` : head;
}

function statLabel(stat, rows) {
  const hit = (rows || []).find(r => r.stat === stat);
  return hit ? hit.label : (STAT_LABELS[stat] || stat);
}

function normaliseName(name) {
  const text = String(name || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
  return text.replace(/\d+$/, "") || text;
}

/* The pairwise reason says "both shorten to jay", which is wrong on a group of
   three. Describe the group itself. */
function groupReason(members, questions) {
  const roots = new Set(members.map(m => normaliseName(m.name)));
  const exact = new Set(members.map(m => displayKey(m.name)));
  if (exact.size === 1) {
    return `all ${members.length} appeared as \u201c${members[0].name}\u201d`;
  }
  if (roots.size === 1) {
    return `all ${members.length} shorten to \u201c${[...roots][0]}\u201d`;
  }
  return [...new Set(questions.map(q => q.detail))].join("; ");
}

/* Longest-processing-time bin packing: tallest panel first, then each next one
   into whichever column is currently shorter. Unlike CSS columns it cannot
   strand a panel on its own. */
function balanceColumns(host) {
  if (!host) return;
  const tiles = [...host.children].filter(el => el.classList.contains("panel"));
  if (tiles.length < 2) return;
  const measured = tiles.map(el => ({el, h: el.getBoundingClientRect().height}));
  const wide = getComputedStyle(host).gridTemplateColumns.split(" ").length > 1;
  const cols = [];
  for (let i = 0; i < (wide ? 2 : 1); i++) {
    const c = document.createElement("div");
    c.className = "col";
    cols.push({node: c, h: 0});
  }
  for (const item of [...measured].sort((a, b) => b.h - a.h)) {
    const target = cols.reduce((a, b) => (a.h <= b.h ? a : b));
    target.node.appendChild(item.el);
    target.h += item.h + 14;
  }
  for (const c of cols) host.appendChild(c.node);
}

function displayKey(name) {
  return String(name || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
}
function isExactName(q) {
  return displayKey(q.left && q.left.name) === displayKey(q.right && q.right.name);
}

/* When both sides show the same screen name the name tells you nothing, so
   the account id has to be on screen to make the question answerable. */
function sideAccount(side) {
  if (!side || !side.account) return "";
  const id = String(side.account);
  const short = id.length <= 10 ? id : `${id.slice(0, 6)}\u2026${id.slice(-3)}`;
  return ` \u00b7 <span class="mono">${esc(short)}</span>`;
}

async function askIdentity(token, questions, onDone) {
  const modal = $("#modal");
  modal.innerHTML = `
    <div class="veil"><div class="sheet">
      <h2 style="margin-top:0">Same player?</h2>
      <p class="small muted" style="margin-top:0">
        Same-id accounts were merged already. These are different ids.</p>
      <label class="bulk" id="bulk-wrap" hidden>
        <input type="checkbox" id="merge-exact" checked>
        <span><b>Merge exact name matches</b>
          <span class="small muted" id="bulk-count"></span></span>
      </label>
      <div id="questions"></div>
      <div class="row" style="justify-content:flex-end;margin-top:18px">
        <button class="act" id="cancel">Keep them separate</button>
        <button class="act primary" id="confirm">Apply</button>
      </div>
    </div></div>`;
  const box = $("#questions");
  const groups = groupQuestions(questions);

  for (const g of groups.filter(x => x.questions.length > 1)) {
    const members = [...g.members.values()]
      .sort((a, b) => (b.hands || 0) - (a.hands || 0));
    const allExact = g.questions.every(isExactName);
    const div = document.createElement("div");
    div.className = "q group" + (allExact ? " exact" : "");
    div.dataset.group = g.questions.map(q => q.id).join("|");
    const names = [...new Map(members.map(m => [displayKey(m.name), m.name])).values()];
    div.innerHTML = `
      <div class="q-prompt">${members.length} accounts look like one person</div>
      <div class="small muted">${esc(groupReason(members, g.questions))}</div>
      <div class="members">${members.map(m => `
        <div class="member"><b>${esc(m.name)}</b>
          <span class="small muted">${sideMeta(m)}</span></div>`).join("")}</div>
      <div class="choice">
        <label><input type="radio" name="g-${esc(div.dataset.group)}" value="yes" checked>
          Merge all ${members.length}</label>
        <label><input type="radio" name="g-${esc(div.dataset.group)}" value="no">
          Keep all separate</label>
      </div>
      ${names.length < 2 ? "" : `<div class="choice namechoice">
        <span class="small muted namelabel">keep the name</span>${names.map(n => `
        <label><input type="radio" name="gname-${esc(div.dataset.group)}"
          value="${esc(n)}" ${n === names[0] ? "checked" : ""}>keep \u201c${esc(n)}\u201d</label>`
        ).join("")}</div>`}`;
    box.appendChild(div);
  }

  const singles = new Set(
    groups.filter(x => x.questions.length === 1).map(x => x.questions[0].id));
  for (const q of questions.filter(x => singles.has(x.id))) {
    const div = document.createElement("div");
    div.className = isExactName(q) ? "q exact" : "q";
    // Two identical strings are not a choice, so the picker goes away rather
    // than offering "keep Pratul" or "keep Pratul".
    const distinct = [...new Map((q.names || []).map(n => [displayKey(n), n])).values()];
    const names = distinct.length < 2 ? "" : distinct.map(n => `
      <label><input type="radio" name="name-${esc(q.id)}" value="${esc(n)}"
        ${n === q.default_name ? "checked" : ""}>keep \u201c${esc(n)}\u201d</label>`).join("");
    div.innerHTML = `
      <div class="q-prompt">${esc(q.prompt)}</div>
      <div class="sides">
        <div class="side"><b>${esc(q.left.name)}</b>
          <div class="small muted">${sideMeta(q.left)}</div></div>
        <div class="side"><b>${esc(q.right.name)}</b>
          <div class="small muted">${sideMeta(q.right)}</div></div>
      </div>
      <div class="small muted">${esc(q.detail)}</div>
      <div class="choice">
        <label><input type="radio" name="${esc(q.id)}" value="yes"
          ${q.default ? "checked" : ""}>Same player</label>
        <label><input type="radio" name="${esc(q.id)}" value="no"
          ${q.default ? "" : "checked"}>Different people</label>
      </div>
      ${names ? `<div class="choice namechoice">
        <span class="small muted namelabel">keep the name</span>${names}</div>` : ""}`;
    box.appendChild(div);

    // The name only means anything if they are one person, so it greys out
    // rather than disappearing -- a control that vanishes leaves you wondering
    // whether you missed something.
    const setEnabled = same => {
      const group = $(".namechoice", div);
      if (!group) return;
      group.classList.toggle("disabled", !same);
      group.querySelectorAll("input").forEach(input => { input.disabled = !same; });
    };
    setEnabled(q.default);
    div.querySelectorAll(`input[name="${CSS.escape(q.id)}"]`).forEach(radio =>
      radio.onchange = () => setEnabled(radio.value === "yes"));
  }
  // Exact-name pairs are the bulk of a batch and all get the same answer, so
  // they collapse behind one switch instead of being asked one at a time.
  const exact = questions.filter(isExactName);
  const bulk = $("#merge-exact");
  if (exact.length) {
    $("#bulk-wrap").hidden = false;
    $("#bulk-count").textContent = `\u00b7 ${exact.length} of ${questions.length}`;
    const sync = () => {
      box.querySelectorAll(".q.exact").forEach(el => { el.hidden = bulk.checked; });
    };
    bulk.onchange = sync;
    sync();
  }

  const groupAnswer = (q) => {
    const card = box.querySelector(`.q.group[data-group*="${CSS.escape(q.id)}"]`);
    if (!card) return undefined;
    const picked = card.querySelector(`input[name="g-${CSS.escape(card.dataset.group)}"]:checked`);
    const name = card.querySelector(`input[name="gname-${CSS.escape(card.dataset.group)}"]:checked`);
    return {same: picked ? picked.value === "yes" : true,
            name: name ? name.value : q.default_name};
  };

  const answerFor = (q, forceSame) => {
    if (forceSame !== undefined) return {same: forceSame, name: q.default_name};
    const picked = modal.querySelector(`input[name="${CSS.escape(q.id)}"]:checked`);
    const name = modal.querySelector(`input[name="name-${CSS.escape(q.id)}"]:checked`);
    return {same: picked ? picked.value === "yes" : q.default,
            name: name ? name.value : q.default_name};
  };

  const send = async (answers) => {
    showBusy("Applying\u2026");
    try {
      const refreshed = await post(`/api/session/${token}/identity`, {answers});
      if (state.session && state.session.token === token) {
        state.session = refreshed;
        renderSession();
      }
      if (onDone) await onDone();
      else $("#modal").innerHTML = "";
    } catch (err) {
      $("#modal").innerHTML = "";
      throw err;
    }
  };

  $("#cancel").onclick = async () => {
    // Say "no" explicitly for every question. An empty object meant
    // "unanswered", which falls back to each question's default -- and now that
    // an identical screen name defaults to *merge*, "Keep them separate" was
    // merging the very accounts it promised to leave alone.
    const answers = {};
    for (const q of questions) answers[q.id] = {same: false, name: q.default_name};
    await send(answers);
  };
  $("#confirm").onclick = async () => {
    const answers = {};
    for (const q of questions) {
      const grouped = groupAnswer(q);
      if (grouped) { answers[q.id] = grouped; continue; }
      const forced = (bulk && bulk.checked && isExactName(q)) ? true : undefined;
      answers[q.id] = answerFor(q, forced);
    }
    await send(answers);
  };
}

async function commit(token) {
  try {
    const result = await post(`/api/session/${token}/commit`, {});
    state.session.saved = true;
    renderSession();
    showResult(result);
  } catch (err) {
    showResult({error: err.message});
  }
}

function showResult(result) {
  const modal = $("#modal");
  const body = result.error
    ? `<p class="err">${esc(result.error)}</p>`
    : result.reset
    ? `<p>Removed ${result.reset.hands} hands and ${result.reset.players} players.</p>`
    : `<p>${result.hands_new} new hands stored${result.duplicates
        ? `, ${result.duplicates} already known` : ""}.</p>
       <p class="small muted">${result.players_new} new player(s)\u00b7
         ${result.merged} merge(s) applied.</p>
       ${(result.blocked || []).map(b => `<p class="err small">${esc(b)}</p>`).join("")}`;
  const heading = result.error ? "Could not save" : result.reset ? "Database reset" : "Saved";
  modal.innerHTML = `<div class="veil"><div class="sheet">
    <h2 style="margin-top:0">${heading}</h2>
    ${body}
    <div class="row" style="justify-content:flex-end;margin-top:16px">
      <button class="act" id="close">Close</button>
      ${result.error || result.reset ? "" : '<button class="act primary" id="godb">Open players</button>'}
    </div></div></div>`;
  $("#close").onclick = () => { modal.innerHTML = ""; };
  const go = $("#godb");
  if (go) go.onclick = () => { modal.innerHTML = ""; switchTab("players"); };
}

/* ---- tab 2: players ---- */
async function viewPlayers() {
  const view = $("#view");
  if (state.player) return viewPlayer(state.player);
  view.innerHTML = `<div class="panel"><div class="empty">loading\u2026</div></div>`;
  const data = await get("/api/roster");
  state.heroId = data.hero_id;
  $("#meta").textContent = `${data.hands} hands \u00b7 ${data.players.length} players`;
  if (!data.players.length) {
    view.innerHTML = `<div class="panel"><h2>nothing stored yet</h2>
      <p class="muted">Drop your hand history exports here.</p>
      <div class="drop" id="db-drop">
        <div style="font-size:15px;color:var(--ink)">Drop hand history files here</div>
        <div class="small" style="margin-top:4px">or click to choose \u00b7
          any number at once</div>
      </div>
      <input type="file" id="db-file" multiple accept=".json,.txt" hidden>
      <div id="db-status" class="small muted" style="margin-top:10px"></div></div>`;
    wireImport();
    return;
  }
  /* One table, sorted however you like. A separate leaderboard tab was the
     same rows in a different order. */
  view.innerHTML = `<div class="panel">
      <div class="spread"><h2>database</h2>
        <div class="row">
          <span class="small muted" id="db-meta">click a column to re-rank</span>
          <button class="act small nowrap" id="db-add">Add hands</button>
        </div></div>
      <div class="drop compact" id="db-drop" hidden>
        <div style="font-size:14px;color:var(--ink)">Drop hand history files here</div>
        <div class="small" style="margin-top:4px">any number at once</div>
      </div>
      <input type="file" id="db-file" multiple accept=".json,.txt" hidden>
      <div id="db-status" class="small muted" style="margin-bottom:10px"></div>
      <div id="db-roster"></div></div>
    <div id="suggest-panel" class="panel" hidden>
      <h2>possible same person</h2><div id="suggestions"></div></div>
    <p class="footnote">
      <button class="linkbtn danger-link" id="reset">reset database</button>
      <span class="muted">deletes every hand, player and merge decision</span></p>`;
  wireImport();
  $("#db-roster").appendChild(rosterTable(data.players, {
    onClick: p => { state.player = p.player_id; viewPlayer(p.player_id); },
    heroId: data.hero_id,
  }));
  $("#reset").onclick = () => confirmReset(data);

  // Priors are fitted on import now, not on request. What is worth showing is
  // which population the reads on this page were measured against -- that is
  // the one thing the automatic fit changes about how you should read them.
  const fit = data.fit_priors;
  if (fit && fit.has_fitted) {
    $("#db-meta").innerHTML =
      `measured against your own pool \u00b7 ${fit.players} players`;
  } else if (fit) {
    $("#db-meta").innerHTML =
      `measured against generic online norms \u2014 your pool takes over at 8 players`;
  }

  const suggestions = await get("/api/suggestions");
  if (suggestions.length) {
    $("#suggest-panel").hidden = false;
    $("#suggestions").innerHTML = suggestions.map(s => `
      <div class="leak"><div class="leak-head">
        <div><b>${esc(s.absorb_name)}</b> may be <b>${esc(s.keep_name)}</b>
          <div class="small muted">${esc(s.reason)}</div></div>
        <div class="row"><span class="tag">${fmtPct(s.confidence)}</span>
          <button class="act small" data-keep="${s.keep}" data-absorb="${s.absorb}">merge</button></div>
      </div></div>`).join("");
    $("#suggestions").querySelectorAll("button").forEach(b => b.onclick = async () => {
      b.disabled = true; b.textContent = "merging\u2026";
      try {
        await post("/api/link", {keep: +b.dataset.keep, absorb: +b.dataset.absorb});
        // Retire the row here rather than relying on the re-render to drop it:
        // the merge is done, and a suggestion still sitting on screen reads as
        // one that failed. If it was the last one, the panel goes too.
        b.textContent = "done";
        const row = b.closest(".leak");
        if (row) {
          row.classList.add("merged");
          setTimeout(() => {
            row.remove();
            if (!$("#suggestions").querySelector(".leak")) {
              $("#suggest-panel").hidden = true;
            }
          }, 550);
        }
        // A session on the other tab is about the same people; leaving it
        // unmerged would contradict the database it is about to be saved into.
        if (state.session && state.session.token) {
          try {
            state.session = await get(`/api/session/${state.session.token}`);
          } catch (ignored) { /* session expired; nothing to refresh */ }
        }
        await viewPlayers();
      } catch (err) {
        b.disabled = false;
        b.textContent = "merge";
        b.title = err.message || "merge failed";
        const note = b.closest(".leak");
        if (note) {
          let msg = note.querySelector(".merge-err");
          if (!msg) {
            msg = document.createElement("div");
            msg.className = "small err merge-err";
            note.appendChild(msg);
          }
          msg.textContent = err.message || "merge failed";
        }
      }
    });
  }
}

async function viewPlayer(id) {
  const view = $("#view");
  const data = await get("/api/player/" + id);
  const names = [...new Set(data.aliases.map(a => a.name))];
  $("#meta").textContent = names.length > 1
    ? `also known as ${names.slice(1).join(", ")}` : "";
  view.innerHTML = "";
  const back = document.createElement("p");
  back.innerHTML = `<button class="linkbtn" id="back">\u2190 all players</button>`;
  view.appendChild(back);
  $("#back").onclick = () => { state.player = null; viewPlayers(); };

  const holder = document.createElement("div");
  view.appendChild(holder);
  state.heroId = data.hero_id;
  playerTabs(data.profiles, holder, {narrate: true, heroId: data.hero_id});

  if (data.by_table && data.by_table.length > 1) {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h2>split by table size</h2>
      <div class="small muted" style="margin:0 0 12px">
        The read above pools these.</div>
      <div class="scroller"><table><thead><tr>
        <th>table</th><th class="num">hands</th><th>read</th>
        <th class="num">skill</th><th>biggest leak</th>
      </tr></thead><tbody></tbody></table></div>`;
    const body = $("tbody", panel);
    for (const t of data.by_table) {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${esc(t.regime_label)}</td>
        <td class="num">${t.hands}</td>
        <td>${esc(t.archetype)} <span class="small muted">${fmtPct(t.archetype_confidence)}</span></td>
        <td class="num">${t.skill.score.toFixed(0)}</td>
        <td class="small">${t.leaks.length ? esc(t.leaks[0].headline)
          : '<span class="muted">none</span>'}</td>`;
      body.appendChild(tr);
    }
    view.appendChild(panel);
  }
}

/* Destructive and irreversible, so it asks for the words rather than a click:
   a stray Enter on a normal dialog should not cost a season of hands. */
function confirmReset(data) {
  const modal = $("#modal");
  modal.innerHTML = `<div class="veil"><div class="sheet">
    <h2 style="margin-top:0">Reset the database?</h2>
    <p>This deletes <b>${data.hands} hands</b> and
       <b>${data.players.length} profiles</b>, along with every merge and rename
       decision you have made.</p>
    <p class="small muted">The original export files are untouched \u2014 you can
       import them again, but the identity decisions are gone for good.</p>
    <p class="small">Type <b>delete everything</b> to confirm:</p>
    <input id="confirm-text" style="width:100%;padding:8px 10px;border-radius:8px;
      border:1px solid var(--line);background:transparent;color:var(--ink);font:inherit">
    <div class="row" style="justify-content:flex-end;margin-top:16px">
      <button class="act" id="cancel">Cancel</button>
      <button class="act danger" id="go" disabled>Reset</button>
    </div></div></div>`;
  const text = $("#confirm-text"), go = $("#go");
  text.focus();
  text.oninput = () => { go.disabled = text.value.trim().toLowerCase() !== "delete everything"; };
  $("#cancel").onclick = () => { modal.innerHTML = ""; };
  go.onclick = async () => {
    go.disabled = true; go.textContent = "resetting\u2026";
    try {
      const result = await post("/api/reset", {confirm: "delete everything"});
      modal.innerHTML = "";
      state.player = null; state.session = null;
      viewPlayers();
      showResult({reset: result});
    } catch (err) { showResult({error: err.message}); }
  };
}

/* ---- evidence: the hands behind a number ---- */
async function showEvidence(playerId, stat, headline) {
  const modal = $("#modal");
  const sc = String(stat).startsWith("vs:") ? " hero-scope" : "";
  modal.innerHTML = `<div class="veil"><div class="sheet${sc}">
    <h2 style="margin-top:0">${esc(headline)}</h2>
    <div class="empty">finding the hands\u2026</div></div></div>`;
  let data;
  try {
    data = await get(`/api/evidence?player=${playerId}&stat=${encodeURIComponent(stat)}`);
  } catch (err) {
    modal.innerHTML = `<div class="veil"><div class="sheet${sc}">
      <p class="err">${esc(err.message)}</p>
      <div class="row" style="justify-content:flex-end"><button class="act" id="close">Close</button></div>
      </div></div>`;
    $("#close").onclick = () => { modal.innerHTML = ""; };
    return;
  }
  modal.innerHTML = `<div class="veil"><div class="sheet${sc}">
    <div class="spread"><h2 style="margin:0">${esc(headline)}</h2>
      <button class="act" id="close">Close</button></div>
    <p class="ev-verdict">${esc(evidenceVerdict(data))}</p>
    <p class="small muted">${data.hits
      ? `Showing the most recent \u2014 click one to replay it.`
      : `Never, in ${data.count} chance${data.count === 1 ? "" : "s"}.`}
      ${data.count > data.hits
      ? `<label class="onlyhits"><input type="checkbox" id="show-all">
           show the ones where it did not</label>` : ""}</p>
    <div id="evlist"></div>
    </div></div>`;
  $("#close").onclick = () => { modal.innerHTML = ""; };

  const list = $("#evlist");
  // The hands that moved the number are what you opened this to check; the
  // denominator is one click away, because 19 of 60 and 19 of 20 are different
  // players and hiding that would misrepresent the rate.
  const showAll = $("#show-all");
  if (showAll) {
    showAll.onchange = () => list.querySelectorAll(".ev").forEach(el => {
      el.hidden = !showAll.checked && !el.classList.contains("counted");
    });
  }
  for (const h of data.hands) {
    const row = document.createElement("div");
    row.className = "ev";
    const when = h.started_at ? new Date(h.started_at).toLocaleString() : "";
    row.innerHTML = `
      <span class="ev-board"></span>
      <span class="ev-what"><span class="ev-summary">${esc(h.summary)}</span>
        <span class="small muted ev-when">${esc(when)}</span></span>
      <span class="ev-net ${h.net_bb < 0 ? "lost" : ""}">${
        h.net_bb > 0 ? "+" : ""}${h.net_bb} bb</span>`;
    const boardCell = $(".ev-board", row);
    if ((h.board || []).length) boardCell.appendChild(cardsEl(h.board, {small: true}));
    else boardCell.innerHTML = `<span class="small muted">no flop</span>`;
    if ((h.hole_cards || []).length) {
      const hole = cardsEl(h.hole_cards, {small: true});
      hole.classList.add("hole");
      boardCell.appendChild(hole);
    }
    row.classList.toggle("counted", !!h.hit);
    if (!h.hit) row.hidden = true;
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.onkeydown = e => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault(); showReplay(h.hand_id, playerId, headline);
      }
    };
    row.onclick = () => showReplay(h.hand_id, playerId, headline);
    list.appendChild(row);
  }
}

async function showReplay(handId, playerId, headline) {
  // Its own layer: the replay used to append under a list you had scrolled
  // through, so the hand you clicked ended up off screen.
  const layer = $("#modal2");
  layer.innerHTML = `<div class="veil"><div class="sheet">
    <div class="spread"><h2 style="margin:0">Hand replay</h2>
      <button class="act" id="close-replay">Back</button></div>
    ${headline ? `<p class="small muted" style="margin:4px 0 0">${esc(headline)}</p>` : ""}
    <div id="replay"><div class="empty">loading hand\u2026</div></div>
  </div></div>`;
  $("#close-replay").onclick = () => { layer.innerHTML = ""; };
  const box = $("#replay", layer);
  const r = await get(`/api/hand/${handId}?focus=${playerId}`);
  const seatLine = document.createElement("div");
  seatLine.className = "small muted seatline";
  for (const st of r.seats) {
    const chunk = document.createElement("span");
    chunk.className = "seatchunk";
    chunk.textContent = `${st.position} ${st.name} `;
    if (st.hole_cards.length) chunk.appendChild(cardsEl(st.hole_cards, {small: true}));
    seatLine.appendChild(chunk);
  }
  box.innerHTML = `<div class="panel" style="margin-top:14px">
    <div class="small muted">${r.pot_bb} bb pot \u00b7 won by ${esc(r.winners.join(", ") || "\u2014")}</div>
    <div id="seats"></div>
    <div id="streets"></div></div>`;
  $("#seats").appendChild(seatLine);
  const streets = $("#streets");
  for (const st of r.streets) {
    const div = document.createElement("div");
    div.className = "street";
    div.innerHTML = `<h4>${esc(st.name)}</h4>`;
    if ((st.new_cards || []).length) $("h4", div).appendChild(cardsEl(st.new_cards));
    for (const a of st.actions) {
      const line = document.createElement("div");
      line.className = "act" + (a.focus ? " focus" : "") + (a.post ? " post" : "");
      const amount = a.act.startsWith("check") || a.act.startsWith("fold")
        ? "" : `${a.to_bb} bb`;
      line.innerHTML = `<span class="small muted">${esc(a.position)}</span>
        <span class="who">${esc(a.name)} ${esc(a.act)}</span>
        <span class="amt">${amount}</span>
        <span class="amt small">pot ${a.pot_bb}</span>`;
      div.appendChild(line);
    }
    streets.appendChild(div);
  }

}

/* ---- tabs ---- */
function switchTab(tab, playerId) {
  state.tab = tab;
  // Bare "go to the database tab" clears which player was open; a link that
  // names a specific player (from Sessions, say) opens straight to them
  // instead of dropping back to the general roster.
  if (tab === "players") state.player = playerId != null ? playerId : null;
  document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  for (const id of ["#modal2", "#modal"]) {
    const layer = $(id);
    if (layer && layer.innerHTML.trim()) { layer.innerHTML = ""; return; }
  }
});
document.querySelectorAll("nav button").forEach(b =>
    b.classList.toggle("on", b.dataset.tab === tab));
  $("#meta").textContent = "";
  render();
}
async function render() {
  try {
    if (!state.glossary) state.glossary = await get("/api/glossary");
    if (state.tab === "session") viewSession();
    else if (state.tab === "sessions") await viewSessions();
    else if (state.tab === "hero") await viewHero();
    else await viewPlayers();
  } catch (err) {
    $("#view").innerHTML = `<div class="panel err">${esc(err.message)}</div>`;
  }
}
document.querySelectorAll("nav button").forEach(b =>
  b.onclick = () => switchTab(b.dataset.tab));

/* Sun when it is dark (click for light), moon when it is light. The icon
   shows what you get, not what you have. */
const SUN = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round">
  <circle cx="12" cy="12" r="4.2"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2
  M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>`;
const MOON = `<svg width="17" height="17" viewBox="0 0 24 24" fill="none"
  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/></svg>`;
const themeBtn = $("#theme");
function isDark() {
  const set = document.documentElement.getAttribute("data-theme");
  if (set) return set === "dark";
  return matchMedia("(prefers-color-scheme: dark)").matches;
}
function paintTheme() { themeBtn.innerHTML = isDark() ? SUN : MOON; }
themeBtn.onclick = () => {
  document.documentElement.setAttribute("data-theme", isDark() ? "light" : "dark");
  paintTheme();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paintTheme);
paintTheme();

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
