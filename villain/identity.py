"""Recognising a player you have seen before.

Home games are full of the same humans under slightly different names --
``DavidMazour`` becomes ``DavidMazour2`` after a reconnect, and a profile that
restarts each time is worthless. Two independent signals are combined:

* **Name similarity**, after stripping the noise accounts accumulate: case,
  punctuation, and trailing digits. ``Arnav`` and ``Arnav2`` normalise to the
  same string.
* **Behavioural similarity**, as a Bayes factor. For each statistic, compare
  the probability of both samples having come from one player against the
  probability of them having come from two. Summed across statistics this is a
  real test: two tight-passive players will not be merged just because they are
  both tight, since the evidence has to beat the population's own explanation.

One hard constraint overrides both: accounts dealt into the same hand are
different people, whatever their names look like. Those pairs are recorded at
import time and can never be linked.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .archetypes import _log_beta_binomial
from .priors import prior_for
from .profile import PROFILE_FEATURES
from .stats import StatBook

#: A Bayes factor above this is strong evidence for one player, below it for two.
STRONG_EVIDENCE = 4.0

#: Minimum hands each side needs before behaviour is worth testing at all.
MIN_HANDS_FOR_BEHAVIOUR = 60


@dataclass
class Suggestion:
    keep: int
    absorb: int
    keep_name: str
    absorb_name: str
    name_score: float
    behaviour_log_bf: float | None
    confidence: float
    reason: str


def normalise(name: str) -> str:
    """Strip the noise a screen name accumulates across sessions."""
    text = re.sub(r"[^a-z0-9]+", "", name.lower())
    stripped = re.sub(r"\d+$", "", text)
    return stripped or text


def name_similarity(a: str, b: str) -> float:
    """Similarity of two screen names, 0-1.

    Two measures, taking the better of them. ``SequenceMatcher`` rewards shared
    runs, which catches additions and truncations; edit distance catches
    transposed and mistyped characters, which is what actually happens to a
    name being retyped from memory -- ``DavidMazour`` reappearing as
    ``DamivDazour`` scores 0.73 on shared runs but 0.82 on edit distance.
    """
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    blocks = SequenceMatcher(None, na, nb).ratio()
    edits = 1.0 - _levenshtein(na, nb) / max(len(na), len(nb))
    return max(blocks, edits)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def behaviour_log_bf(a: StatBook, b: StatBook) -> float | None:
    """Log Bayes factor for "one player" over "two players".

    Positive favours a merge. Each statistic contributes the difference between
    the marginal likelihood of the pooled counts and that of the two samples
    scored separately, so agreeing on a rare tendency counts for far more than
    agreeing on a common one.
    """
    if a.hands < MIN_HANDS_FOR_BEHAVIOUR or b.hands < MIN_HANDS_FOR_BEHAVIOUR:
        return None
    regime = a.regime or b.regime
    total = 0.0
    used = 0
    for stat in PROFILE_FEATURES:
        ra, rb = a.ratios.get(stat), b.ratios.get(stat)
        if not ra or not rb or ra.opps < 8 or rb.opps < 8:
            continue
        mean, strength = prior_for(stat, regime)
        pooled = _log_beta_binomial(ra.hits + rb.hits, ra.opps + rb.opps, mean, strength)
        apart = (_log_beta_binomial(ra.hits, ra.opps, mean, strength)
                 + _log_beta_binomial(rb.hits, rb.opps, mean, strength))
        total += pooled - apart
        used += 1
    return total if used >= 5 else None


def suggest_links(store, min_name_score: float = 0.70) -> list[Suggestion]:
    """Candidate merges, most confident first.

    Only pairs that are not already known to be different people, and only
    those with either a plausible name match or convincing behaviour. Nothing
    is merged automatically: a wrong merge silently corrupts two profiles at
    once and is far more expensive than a missed one, so the confidence is
    reported and the decision stays with the user.
    """
    players = {int(r["id"]): r for r in store.players()}
    aliases: dict[int, list[str]] = {}
    for row in store.conn.execute("SELECT player_id, name FROM aliases"):
        aliases.setdefault(int(row["player_id"]), []).append(row["name"])

    out: list[Suggestion] = []
    ids = sorted(players)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if store.are_distinct(a, b):
                continue
            score = max((name_similarity(x, y)
                         for x in aliases.get(a, []) for y in aliases.get(b, [])),
                        default=0.0)
            books_a, books_b = store.books(a), store.books(b)
            log_bf = _best_regime_bf(books_a, books_b)

            if score < min_name_score and (log_bf is None or log_bf < STRONG_EVIDENCE):
                continue
            confidence, reason = _combine(score, log_bf)
            # The busier account keeps its identity, so the merged profile
            # keeps the name the player is better known by.
            keep, absorb = (a, b) if players[a]["hands"] >= players[b]["hands"] else (b, a)
            out.append(Suggestion(
                keep=keep, absorb=absorb,
                keep_name=players[keep]["display_name"],
                absorb_name=players[absorb]["display_name"],
                name_score=round(score, 3),
                behaviour_log_bf=None if log_bf is None else round(log_bf, 2),
                confidence=round(confidence, 3), reason=reason,
            ))
    out.sort(key=lambda s: -s.confidence)
    return out


def _best_regime_bf(books_a: dict[str, StatBook], books_b: dict[str, StatBook]) -> float | None:
    """Behaviour evidence from the regime where both have the most data."""
    best = None
    for regime, book_a in books_a.items():
        book_b = books_b.get(regime)
        if book_b is None:
            continue
        bf = behaviour_log_bf(book_a, book_b)
        if bf is not None and (best is None or bf > best):
            best = bf
    return best


def _combine(name_score: float, log_bf: float | None) -> tuple[float, str]:
    """Fold the two signals into one probability and an explanation."""
    # Name similarity as log-odds, capped: a matching name is good evidence in
    # a home game and weak evidence in a public pool, so it is not allowed to
    # carry the decision alone.
    name_odds = min(3.0, max(-2.0, 8.0 * (name_score - 0.75)))
    total = name_odds + (log_bf or 0.0)
    confidence = 1.0 / (1.0 + math.exp(-total))
    if log_bf is None:
        reason = f"names match ({name_score:.0%}); not enough hands to check behaviour"
    elif log_bf >= STRONG_EVIDENCE:
        reason = f"names match ({name_score:.0%}) and play styles agree (log BF {log_bf:+.1f})"
    elif log_bf <= -STRONG_EVIDENCE:
        reason = f"names match ({name_score:.0%}) but play styles differ (log BF {log_bf:+.1f})"
    else:
        reason = f"names match ({name_score:.0%}); behaviour inconclusive (log BF {log_bf:+.1f})"
    return confidence, reason


# ---------------------------------------------------------------------------
# import-time reconciliation
# ---------------------------------------------------------------------------
# Two things go wrong when a session is added to a database, and they are
# opposites. One account id can pick up a new display name, and two account ids
# can belong to one person. Both are asked about rather than guessed, because a
# wrong answer in either direction silently corrupts a profile -- and a profile
# is the whole product.


@dataclass
class Question:
    """Something the import needs a human to settle."""

    id: str
    kind: str                 # "rename" or "alias"
    prompt: str
    detail: str
    default: bool             # the answer to preselect
    confidence: float | None
    left: dict
    right: dict


def _account_index(hands) -> dict[tuple[str, str], dict]:
    """Every (site, account) in a batch of hands, with its name and volume."""
    out: dict[tuple[str, str], dict] = {}
    for hand in hands:
        for seat in hand.seats:
            key = (hand.site, seat.player_id)
            entry = out.setdefault(key, {"site": hand.site, "account": seat.player_id,
                                         "name": seat.name, "hands": 0})
            entry["hands"] += 1
            entry["name"] = seat.name or entry["name"]
    return out


def _incoming_co_occurrence(hands) -> set[frozenset]:
    """Account pairs dealt into the same hand -- provably different people."""
    pairs = set()
    for hand in hands:
        accounts = [(hand.site, s.player_id) for s in hand.seats]
        for i, a in enumerate(accounts):
            for b in accounts[i + 1:]:
                pairs.add(frozenset((a, b)))
    return pairs


def session_questions(store, hands, min_name_score: float = 0.70) -> list[Question]:
    """What to ask before folding a session into the database.

    Renames come first: an account id that has appeared before under a
    different display name is almost always the same person renaming
    themselves, so it defaults to yes. Alias candidates -- two different
    account ids that look like one person -- default to no, because merging
    two real players is the more expensive mistake and the evidence for it is
    weaker.
    """
    incoming = _account_index(hands)
    blocked = _incoming_co_occurrence(hands)
    questions: list[Question] = []

    stored = {
        (r["site"], r["account"]): r
        for r in store.conn.execute(
            "SELECT site, account, name, player_id, hands FROM aliases")
    }

    # 1. Same account id, different display name.
    for key, entry in sorted(incoming.items()):
        row = stored.get(key)
        if row is None or not row["name"] or row["name"] == entry["name"]:
            continue
        questions.append(Question(
            id=f"rename:{key[0]}:{key[1]}",
            kind="rename",
            prompt=f"Is “{entry['name']}” the same player as “{row['name']}”?",
            detail=(f"Both are account {key[1]} on {key[0]}. Same id usually means "
                    f"one person who renamed themselves; answer no and they are "
                    f"kept as two players from here on."),
            default=True,
            confidence=None,
            left={"name": row["name"], "hands": row["hands"], "player_id": row["player_id"],
                  "where": "already in the database"},
            right={"name": entry["name"], "hands": entry["hands"],
                   "site": key[0], "account": key[1], "where": "in this session"},
        ))

    # 2. Different account ids that look like the same person.
    db_players = {int(r["id"]): r for r in store.players()}
    db_aliases: dict[int, list[str]] = {}
    for row in store.conn.execute("SELECT player_id, name FROM aliases"):
        db_aliases.setdefault(int(row["player_id"]), []).append(row["name"])

    def add_alias_question(qid, score, left, right, log_bf=None):
        confidence, reason = _combine(score, log_bf)
        questions.append(Question(
            id=qid, kind="alias",
            prompt=f"Are “{left['name']}” and “{right['name']}” the same person?",
            detail=reason, default=False, confidence=confidence,
            left=left, right=right))

    def already_one_player(a_key, b_key) -> bool:
        """Skip pairs the database has already been told are one person."""
        a, b = stored.get(a_key), stored.get(b_key)
        return bool(a and b and a["player_id"] == b["player_id"])

    seen_pairs: set[frozenset] = set()
    items = sorted(incoming.items())
    for i, (key, entry) in enumerate(items):
        # incoming vs incoming
        for other_key, other in items[i + 1:]:
            if frozenset((key, other_key)) in blocked:
                continue
            if already_one_player(key, other_key):
                continue
            score = name_similarity(entry["name"], other["name"])
            if score < min_name_score:
                continue
            add_alias_question(
                f"alias:{key[1]}|{other_key[1]}", score,
                {"name": entry["name"], "hands": entry["hands"], "site": key[0],
                 "account": key[1], "where": "in this session"},
                {"name": other["name"], "hands": other["hands"], "site": other_key[0],
                 "account": other_key[1], "where": "in this session"})

        # incoming vs the database
        if key in stored:
            continue                    # already a known account, not a new face
        for player_id, row in db_players.items():
            best = max((name_similarity(entry["name"], n)
                        for n in db_aliases.get(player_id, [])), default=0.0)
            if best < min_name_score:
                continue
            pair = frozenset((key, ("db", str(player_id))))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            add_alias_question(
                f"alias:{key[1]}|db{player_id}", best,
                {"name": row["display_name"], "hands": row["hands"] or 0,
                 "player_id": player_id, "where": "already in the database"},
                {"name": entry["name"], "hands": entry["hands"], "site": key[0],
                 "account": key[1], "where": "in this session"})

    questions.sort(key=lambda q: (q.kind != "rename", -(q.confidence or 1.0)))
    return questions
