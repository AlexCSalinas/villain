"""The hero page: expensive to build, so cached in memory and on disk.

Hero analysis walks every hand the exporting player appears in and fits a
strength model over it -- seconds, not milliseconds. The result is cached
against the hand count that produced it and reused until the database grows.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from ..db import Store
from .payloads import profile_payload

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
    from ..hero import find_hero

    key = str(store.path)
    hand_count = _hand_count(store)
    cached = _HERO_ID_CACHE.get(key)
    if cached and cached[0] == hand_count:
        return cached[1]
    hero_id = find_hero(store)
    _HERO_ID_CACHE[key] = (hand_count, hero_id)
    return hero_id


def _hero_model(store: Store):
    from ..hero import fit_population_model

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
_HERO_CACHE_VERSION = 7


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


#: Third person to second: the profile machinery writes about "them", but the
#: hero read is about you. English second-person plural agreement matches
#: "they", so a whole-word swap reads correctly on descriptive text; the only
#: opponent-directed fields (a leak's "do", the plan) are dropped by the hero
#: UI, so mistranslating them is harmless.
_HERO_PRONOUNS = {
    "they're": "you're", "they've": "you've", "themselves": "yourself",
    "theirs": "yours", "their": "your", "them": "you", "they": "you",
}


def _second_person(text: str) -> str:
    import re

    def swap(m):
        w = m.group(0)
        repl = _HERO_PRONOUNS.get(w.lower())
        if repl is None:
            return w
        return repl.capitalize() if w[0].isupper() else repl

    return re.sub(r"[A-Za-z']+", swap, text)


def _to_you(obj):
    if isinstance(obj, str):
        return _second_person(obj)
    if isinstance(obj, list):
        return [_to_you(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_you(v) for k, v in obj.items()}
    return obj


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
    return _to_you(profile_payload(prof, hero_id))


def _build_hero_payload(store: Store, hero_id: int | None) -> dict | None:
    from ..hero import NotEnoughData, combined_grid, fold_grades, hero_visibility, missed_value, preflop_range, range_narrowing, sizing_tell, timing_tell
    from ..model import STREET_LABELS

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


