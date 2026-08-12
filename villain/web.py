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
from .exploits import RULES
from .features import record_hands
from .evidence import find as find_evidence
from .glossary import payload as glossary_payload
from .model import hand_from_dict, hand_to_dict
from .identity import session_questions, suggest_links
from .narrate import Unavailable, enabled as narrator_enabled, narrate
from .parsers import UnknownFormat, parse_file
from .priors import population_mean
from .profile import build_profiles, build_unified
from .replay import replay

# Stats worth a row in the profile view, in reading order.
DISPLAY_STATS = [
    ("vpip", "VPIP", "hands played"),
    ("pfr", "PFR", "hands raised preflop"),
    ("three_bet", "3-bet", "raises facing a raise"),
    ("fold_to_three_bet", "fold to 3-bet", "after opening"),
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
            "opps": est.opps, "weight": round(est.weight, 3),
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
                               "n": profile.means.get(f"{key}#n", 0)}
        for key in ("think:fold", "think:call", "think:check", "think:aggro")
        if profile.means.get(key)
    }
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
                "top_leak": top.headline if top else None,
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


def merged_hands(session: dict) -> list:
    """The session's hands with confirmed same-person accounts pooled.

    Applied to a copy. The stored hands must keep the account ids the site
    actually wrote, because identity is a decision layered on top of them and
    decisions get revised; the hands themselves are evidence and do not.
    """
    merges = session.get("merges") or {}
    if not merges:
        return session["hands"]
    hands = [hand_from_dict(hand_to_dict(h)) for h in session["hands"]]
    for hand in hands:
        for seat in hand.seats:
            target = merges.get((hand.site, seat.player_id))
            if target:
                seat.player_id, seat.name = target["account"], target["name"]
    return hands


def session_payload(token: str) -> dict:
    """Profiles for an uploaded session, computed without touching the store."""
    session = SESSIONS[token]
    books = record_hands(merged_hands(session))
    profiles = [p for p in (build_unified(by_regime) for by_regime in books.values())
                if p is not None]
    profiles.sort(key=lambda p: -p.hands)

    rows = []
    for profile in profiles:
        enrich(profile)
        top = profile.tags[0] if profile.tags else None
        rows.append({
            "player_id": None, "name": profile.name,
            "regime": profile.regime, "regime_label": profile.regime_label,
            "table_mix": profile.table_mix,
            "hands": profile.hands, "sample_quality": profile.sample_quality,
            "archetype": profile.archetype, "confidence": profile.archetype_confidence,
            "skill": profile.skill.score, "skill_tier": profile.skill.tier,
            "skill_confidence": profile.skill.confidence,
            "exploitability": profile.skill.exploitability,
            "top_leak": top.headline if top else None,
            "leak_count": len(profile.tags),
        })
    return {
        "token": token,
        "files": session["files"],
        "hands": len(session["hands"]),
        "players": rows,
        "profiles": [profile_payload(p) for p in profiles],
        "saved": session.get("saved", False),
        "questions": [question_payload(q) for q in session.get("questions", [])],
        "answered": bool(session.get("answers")),
        "merges": [{"from": k[1], "to": v["name"]}
                   for k, v in (session.get("merges") or {}).items()],
    }


def question_payload(question) -> dict:
    return {
        "id": question.id, "kind": question.kind, "prompt": question.prompt,
        "detail": question.detail, "default": question.default,
        "confidence": question.confidence, "left": question.left, "right": question.right,
        "names": question.names, "default_name": question.default_name,
    }


def apply_answers(session: dict, answers: dict) -> None:
    """Record identity decisions on a session and pool the merged accounts.

    Asked at upload rather than at save, so the session you are reading has
    already combined them. One player split across two names halves both
    samples exactly when sample size is the scarce thing.
    """
    session["answers"] = answers
    merges: dict[tuple[str, str], dict] = {}
    for question in session.get("questions", []):
        answer = answers.get(question.id) or {}
        if not answer.get("same"):
            continue
        keep_name = answer.get("name") or question.default_name
        sides = [side for side in (question.left, question.right) if side.get("account")]
        if not sides:
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
            if path == "/api/roster":
                with Store(self.db_path) as store:
                    return self._send(200, {
                        "players": roster_payload(store),
                        "db": str(self.db_path),
                        "hands": store.conn.execute(
                            "SELECT COUNT(*) c FROM hands").fetchone()["c"],
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
                    })
            if path.startswith("/api/session/"):
                token = path.rsplit("/", 1)[1]
                if token not in SESSIONS:
                    return self._send(404, {"error": "session expired -- upload again"})
                return self._send(200, session_payload(token))
            if path == "/api/leaderboard":
                with Store(self.db_path) as store:
                    return self._send(200, leaderboard_payload(store))
            if path == "/api/evidence":
                query = parse_qs(route.query)
                player_id = int(query.get("player", ["0"])[0])
                stat = query.get("stat", [""])[0]
                if not stat:
                    return self._send(400, {"error": "stat required"})
                with Store(self.db_path) as store:
                    hands = store.player_hands(player_id)
                found = find_evidence(hands, str(player_id), stat, limit=60)
                return self._send(200, {
                    "stat": stat, "count": len(found),
                    "hits": sum(1 for e in found if e.hit),
                    "hands": [vars(e) for e in found],
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
                        "confidence": s.confidence, "reason": s.reason,
                    } for s in suggest_links(store)])
            return self._send(404, {"error": "not found"})
        except Exception as exc:                      # keep the server alive
            return self._send(500, {"error": str(exc)})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "bad json"})
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
            if route.startswith("/api/session/") and route.endswith("/identity"):
                token = route.split("/")[3]
                if token not in SESSIONS:
                    return self._send(404, {"error": "session expired -- upload again"})
                apply_answers(SESSIONS[token], body.get("answers") or {})
                return self._send(200, session_payload(token))
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
            SESSIONS[token]["questions"] = session_questions(store, unique)
        payload = session_payload(token)
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
    --line: #e3e2df; --accent: #111111; --accent-soft: #f1f0ee;
    --warn: #b4532a;
    /* Neutral ordinal ramp, validated light->dark on the panel surface:
       light end clears 2:1, monotone lightness, visible step gaps. Shade
       carries confidence, never identity. */
    --mark-1: #a8a6a2;   /* tentative */
    --mark-2: #5c5a57;   /* likely */
    --mark-3: #111111;   /* strong */
    --band: #e8e7e4;     /* credible interval wash */
    --grid: #e6e5e2; --axis: #b9b7b2; --tick: #898781;
    --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --bg: #0d0d0d; --panel: #17181a; --ink: #f2f1ee; --muted: #98968f;
      --line: #2a2b2d; --accent: #f2f1ee; --accent-soft: #202123;
      --warn: #e5645a; --danger: #e5645a; --red: #e5645a;
      --mark-1: #787774; --mark-2: #adaba6; --mark-3: #f0efec;
      --band: #26272a;
      --grid: #232427; --axis: #3a3b3e; --tick: #898781;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0d0d0d; --panel: #17181a; --ink: #f2f1ee; --muted: #98968f;
    --line: #2a2b2d; --accent: #f2f1ee; --accent-soft: #202123;
    --warn: #e5645a; --danger: #e5645a; --red: #e5645a;
    --mark-1: #787774; --mark-2: #adaba6; --mark-3: #f0efec;
    --band: #26272a;
    --grid: #232427; --axis: #3a3b3e; --tick: #898781;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1060px; margin: 0 auto; padding: 24px 20px 90px; }
  header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
  h1 {
    font-size: 40px; margin: 0; letter-spacing: -0.05em; font-weight: 800;
    line-height: 0.95;
  }
  h1 a { color: inherit; text-decoration: none; display: inline-flex; align-items: center; }
  h1 .dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--red);
    display: inline-block; margin-left: 8px; margin-bottom: 18px;
  }
  .iconbtn {
    border: 1px solid var(--line); background: transparent; color: var(--muted);
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
  .panel {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 10px; padding: 18px; margin: 16px 0;
  }
  .panel h2 {
    font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--muted); margin: 0 0 14px; font-weight: 600;
  }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
  th { font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em;
       color: var(--muted); font-weight: 600; cursor: pointer; white-space: nowrap; }
  th.sorted::after { content: " \25BE"; color: var(--accent); }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tbody tr.clickable { cursor: pointer; }
  tbody tr.clickable:hover { background: var(--accent-soft); }
  .name { font-weight: 600; }
  .tag {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11.5px; border: 1px solid var(--line); color: var(--muted);
    white-space: nowrap;
  }
  .tag.on { border-color: var(--red); color: var(--ink); }
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
    font: inherit; font-size: 13px; line-height: 1.25;
    padding: 0.46em 0.9em; border-radius: 7px;
    border: 1px solid var(--line); background: transparent;
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
  button.act.small { font-size: 11.5px; }
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
  .drop {
    border: 1.5px dashed var(--line); border-radius: 12px; padding: 34px 20px;
    text-align: center; color: var(--muted); cursor: pointer; transition: border-color .12s;
  }
  .drop:hover, .drop.over { border-color: var(--red); color: var(--ink); }
  .leak { padding: 12px 0; border-bottom: 1px solid var(--line); }
  .leak:last-child { border-bottom: 0; }
  .leak-head { display: flex; justify-content: space-between; gap: 14px; align-items: baseline; }
  .leak-advice { color: var(--muted); font-size: 13.5px; max-width: 68ch; margin-top: 4px; }
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
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block;
            vertical-align: -1px; margin-right: 5px; border: 1px solid var(--line); }
  /* per-player tabs inside a result */
  .ptabs {
    display: flex; gap: 6px; flex-wrap: wrap; margin: 0 0 16px;
    padding-bottom: 2px;
  }
  .ptab {
    border: 1px solid var(--line); background: transparent; color: var(--muted);
    border-radius: 999px; padding: 6px 13px; font: inherit; font-size: 13px;
    cursor: pointer; display: flex; align-items: center; gap: 7px;
  }
  .ptab:hover { color: var(--ink); }
  .ptab.on { border-color: var(--red); color: var(--ink); font-weight: 600; }
  .ptab .meta { font-size: 11.5px; color: var(--muted); font-weight: 400; }
  /* the hover-for-meaning affordance */
  .info {
    display: inline-flex; align-items: center; justify-content: center;
    width: 15px; height: 15px; border-radius: 50%; border: 1px solid var(--line);
    color: var(--muted); font-size: 10px; font-weight: 700; cursor: help;
    vertical-align: 1px; margin-left: 5px; font-style: normal; flex: none;
  }
  .info:hover { color: var(--ink); border-color: var(--accent); }
  .tip .hl { color: var(--ink); font-weight: 600; }
  .tip .dir { margin-top: 6px; }
  .tip .dir b { display: inline-block; min-width: 34px; }
  details > summary {
    cursor: pointer; color: var(--muted); font-size: 12.5px; list-style: none;
    padding: 4px 0;
  }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: "\25B8 "; }
  details[open] > summary::before { content: "\25BE "; }
  details > summary:hover { color: var(--ink); }
  .headline { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }
  .plan { max-width: 62ch; }
  .meta { text-align: right; line-height: 1.7; }
  .metric { display: grid; grid-template-columns: 170px 1fr 34px; gap: 10px;
            align-items: center; margin: 4px 0; }
  .howblock { margin: 10px 0; max-width: 66ch; }
  .howlabel { font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em;
              color: var(--muted); margin-bottom: 2px; }
  .footnote { font-size: 12.5px; display: flex; gap: 8px; flex-wrap: wrap;
              align-items: baseline; margin: 22px 2px; }
  .danger-link { color: var(--danger); text-decoration-color: var(--danger); }
  .narration { margin-top: 10px; max-width: 66ch; }
  .narration blockquote {
    margin: 0 0 6px; padding: 0 0 0 13px; border-left: 2px solid var(--red);
  }
  .rank { font-variant-numeric: tabular-nums; color: var(--muted); width: 22px; }
  .ev { display: grid; grid-template-columns: 54px 1fr auto; gap: 10px;
        align-items: baseline; padding: 7px 0; border-bottom: 1px solid var(--line);
        cursor: pointer; }
  .ev:hover { background: var(--accent-soft); }
  .ev:last-child { border-bottom: 0; }
  .ev .verdict { font-size: 11.5px; text-transform: uppercase; letter-spacing: .05em; }
  .ev .verdict.hit { color: var(--red); font-weight: 600; }
  .ev .verdict.miss { color: var(--muted); }
  .cards { font-family: var(--mono); letter-spacing: .04em; }
  .street { border-top: 1px solid var(--line); padding: 10px 0; }
  .street:first-child { border-top: 0; }
  .street h4 { margin: 0 0 6px; font-size: 11.5px; text-transform: uppercase;
               letter-spacing: .06em; color: var(--muted); font-weight: 600;
               display: flex; gap: 10px; align-items: baseline; }
  .act { display: grid; grid-template-columns: 44px 1fr auto auto; gap: 10px;
         padding: 3px 0; font-size: 13.5px; }
  .act.focus { font-weight: 600; }
  .act.focus .who::before { content: "\25B8 "; color: var(--accent); }
  .act .amt { font-variant-numeric: tabular-nums; color: var(--muted); }
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
  .q { border-top: 1px solid var(--line); padding: 14px 0; }
  .q:first-of-type { border-top: 0; }
  .q-prompt { font-weight: 600; }
  .sides { display: flex; gap: 10px; margin: 8px 0; flex-wrap: wrap; }
  .side {
    border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px;
    font-size: 13px; flex: 1; min-width: 180px;
  }
  .choice { display: flex; gap: 8px; margin-top: 8px; }
  .choice label {
    border: 1px solid var(--line); border-radius: 999px; padding: 4px 12px;
    font-size: 13px; cursor: pointer;
  }
  .choice input { margin-right: 6px; }
  .choice label:has(input:checked) { border-color: var(--accent); color: var(--ink); font-weight: 600; }
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
    <button data-tab="session" class="on">Session</button>
    <button data-tab="players">Players</button>
  </nav>
  <div id="view"></div>
</div>
<div class="tip" id="tip"></div>
<div id="modal"></div>
<script>
const $ = (s, r) => (r || document).querySelector(s);
const fmtPct = v => (100 * v).toFixed(0) + "%";
const esc = s => String(s == null ? "" : s).replace(/[&<>"]/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const SVG = "http://www.w3.org/2000/svg";
const state = {tab: "session", session: null, player: null, glossary: null};

/* An "i" that explains a term on hover. Everything the tool says in shorthand
   gets one, because a number nobody can interpret is worse than no number. */
function info(html) {
  const span = document.createElement("i");
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
/* Full explanation of a statistic: what it counts, and what each direction
   means for you. Both directions matter -- they call for opposite play. */
function statTip(stat, label) {
  const g = state.glossary;
  const h = g && (g.stats[stat] || g.stats[stat.split(":")[0]]);
  if (!h) return esc(label || stat);
  return `<span class="hl">${esc(label || stat)}</span><br>${esc(h.what)}
    <div class="dir"><b>High</b> ${esc(h.high)}</div>
    <div class="dir"><b>Low</b> ${esc(h.low)}</div>`;
}

/* ---- tooltip ---- */
const tip = $("#tip");
function bindTip(el, html) {
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
      ? `<br><span style="color:var(--warn)">${esc(row.breakeven_label)} ${fmtPct(row.breakeven)}</span>` : ""}`);
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
      <th data-k="exploitability" class="num">worth to you</th>
      <th data-k="top_leak">biggest leak</th>
    </tr></thead><tbody></tbody></table>`;
  const body = $("tbody", wrap);
  let sort = {key: "hands", dir: -1};
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
      if (opts && opts.onClick && p.player_id != null) {
        tr.className = "clickable";
        tr.onclick = () => opts.onClick(p);
      }
      tr.innerHTML = `
        <td><span class="name">${esc(p.name)}</span>
            <div class="small muted quality">${esc(p.sample_quality)}</div></td>
        <td class="num">${p.hands}</td>
        <td><span class="tag ${p.confidence >= 0.5 ? "on" : ""}">${esc(p.archetype)}</span>
            <div class="small muted">${fmtPct(p.confidence)} sure</div></td>
        <td class="num"></td>
        <td class="num">${p.exploitability ? p.exploitability.toFixed(1) + " bb" : "\u2014"}</td>
        <td class="small">${p.top_leak ? esc(p.top_leak)
          : '<span class="muted">none clears the bar</span>'}</td>`;
      const holder = document.createElement("div");
      holder.style.cssText = "display:flex;gap:8px;align-items:center;justify-content:flex-end";
      const label = document.createElement("span");
      label.textContent = p.skill.toFixed(0);
      holder.append(bar(p.skill, 100, "var(--mark-3)", 66), label);
      const q = $(".quality", tr);
      if (q) q.appendChild(info(termTip(p.sample_quality)));
      if (q && p.table_mix) {
        q.appendChild(document.createTextNode(" \u00b7 " + p.regime_label));
      }
      tr.children[3].appendChild(holder);
      bindTip(holder, `<b>${esc(p.skill_tier)}</b> ${p.skill.toFixed(0)}/100<br>
        <span class="muted">confidence ${fmtPct(p.skill_confidence)}</span>`);
      body.appendChild(tr);
    }
    wrap.querySelectorAll("th").forEach(th => th.classList.toggle("sorted", th.dataset.k === sort.key));
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
  const card = document.createElement("div");
  const leaks = p.leaks;
  const maxSeverity = Math.max(0.01, ...leaks.map(l => l.severity_bb100));

  /* Two panels and one disclosure. Everything a player needs mid-hand is the
     read and the adjustment; the numbers that justify them are a different
     job, done at a different time, and stacking all of it in one column made
     the page something to scroll rather than something to use. */
  const head = document.createElement("div");
  head.className = "panel";
  head.innerHTML = `
    <div class="spread">
      <div><div class="hero">${esc(p.archetype)}</div>
           <div class="muted">${esc(p.summary)}</div></div>
      <div class="meta small muted"></div>
    </div>
    <p class="plan">${esc(p.plan)}</p>
    ${opts.narrate ? '<div class="narrate"></div>' : ""}`;
  card.appendChild(head);

  const meta = $(".meta", head);
  const line = document.createElement("div");
  line.append(document.createTextNode(`${p.hands} hands, ${p.sample_quality}`));
  line.appendChild(info(termTip(p.sample_quality)));
  const conf = document.createElement("div");
  conf.append(document.createTextNode(
    `${esc(p.archetype)} ${fmtPct(p.archetype_confidence)} sure`));
  conf.appendChild(info(`${termTip("confidence")}<br><br>
    <span class="hl">also plausibly</span><br>${
      p.archetype_mix.slice(1, 4).map(([n, v]) => `${esc(n)} ${fmtPct(v)}`).join("<br>")}`));
  const where = document.createElement("div");
  where.textContent = p.table_mix || p.regime_label;
  if (p.contributions && Object.keys(p.contributions).length > 1) {
    where.appendChild(info(`<span class="hl">one read, several table sizes</span><br>
      Each table's hands are measured against that table's own norms, then
      pooled. Shown on ${esc(p.regime_label)} terms, where they play most.`));
  }
  meta.append(line, conf, where);

  const narrateBox = $(".narrate", head);
  if (narrateBox) buildNarrator(narrateBox, p);

  /* What to do. Combinations ride along at the end rather than taking their
     own panel -- they are the same advice, sharpened. */
  const doBox = document.createElement("div");
  doBox.className = "panel";
  doBox.innerHTML = `<div class="spread"><h2>what to do</h2>
      <span class="small muted worth"></span></div><div class="leaks"></div>`;
  const worthLabel = $(".worth", doBox);
  worthLabel.append(document.createTextNode(
    leaks.length ? `${p.skill.exploitability_bb100.toFixed(1)} bb/100 available` : ""));
  if (leaks.length) worthLabel.appendChild(info(termTip("available")));
  card.appendChild(doBox);

  const leakBox = $(".leaks", doBox);
  if (!leaks.length) {
    leakBox.innerHTML = `<div class="empty">Nothing stands out yet. Play them
      straight and collect more hands.</div>`;
  }
  for (const l of leaks) {
    const div = document.createElement("div");
    div.className = "leak";
    div.innerHTML = `
      <div class="leak-head">
        <div class="headline"><b>${esc(l.headline)}</b></div>
        <div class="num small muted">${l.severity_bb100.toFixed(2)} bb/100
          <span class="tag tier">${esc(l.tier)}</span></div></div>
      <div class="small muted numbers"></div>
      <div class="leak-advice">${esc(l.do)}</div>
      <details class="how" open><summary>why, and what not to do</summary>
        <div class="how-body"></div></details>`;
    $(".tier", div).after(info(`${termTip(l.tier)}<br><br>${esc(l.priority)}`));

    const numbers = $(".numbers", div);
    numbers.appendChild(document.createTextNode(
      l.in_words.replace(/seen about .*$/, "")));
    if (p.player_id != null) {
      const link = document.createElement("button");
      link.className = "linkbtn";
      link.textContent = `seen about ${Math.round(l.sample)} times`;
      link.title = "show the hands behind this";
      link.onclick = () => showEvidence(p.player_id, l.stat, l.headline);
      numbers.appendChild(link);
    } else {
      numbers.appendChild(document.createTextNode(
        `seen about ${Math.round(l.sample)} times`));
    }
    numbers.appendChild(info(statTip(l.stat, l.headline)));

    const how = $(".how-body", div);
    for (const [label, text] of [["What they are doing", l.behaviour],
                                 ["Why it works", l.why],
                                 ["Do not", l.dont],
                                 ["How hard to lean on it", l.pressure]]) {
      if (!text) continue;
      const block = document.createElement("div");
      block.className = "howblock";
      block.innerHTML = `<div class="howlabel">${esc(label)}</div>
        <div>${esc(text)}</div>`;
      how.appendChild(block);
    }
    leakBox.appendChild(div);
  }

  for (const c of (p.combinations || [])) {
    const block = document.createElement("div");
    block.className = "leak";
    block.innerHTML = `<div class="headline"><b>${esc(c.headline)}</b>
      <span class="tag">these compound</span></div>
      <div class="leak-advice">${esc(c.body)}</div>`;
    leakBox.appendChild(block);
  }

  /* The evidence, folded away. */
  const detail = document.createElement("div");
  detail.className = "panel";
  detail.innerHTML = `<details open><summary>the numbers behind this</summary>
    <div class="detail-body"></div></details>`;
  card.appendChild(detail);
  const body = $(".detail-body", detail);

  const skill = document.createElement("div");
  skill.innerHTML = `<div class="spread" style="margin-top:12px">
      <b>skill ${p.skill.score.toFixed(0)}/100 \u2014 ${esc(p.skill.tier)}</b>
      <span class="small muted">${fmtPct(p.skill.confidence)} confident \u00b7
        ${p.skill.observed_bb100 == null ? "\u2014"
          : p.skill.observed_bb100.toFixed(1)} bb/100 observed,
        ${p.skill.adjusted_bb100 == null ? "\u2014"
          : p.skill.adjusted_bb100.toFixed(1)} adjusted</span></div>`;
  body.appendChild(skill);
  for (const c of [...p.skill.components].sort((a, b) => a.score - b.score)) {
    const row = document.createElement("div");
    row.className = "metric";
    const label = document.createElement("span");
    label.className = "small"; label.textContent = c.name;
    const val = document.createElement("span");
    val.className = "small muted"; val.style.textAlign = "right";
    val.textContent = c.score.toFixed(0);
    row.append(label, bar(c.score, 100, "var(--mark-3)", 220), val);
    bindTip(row, `<b>${esc(c.name)}</b> ${c.score.toFixed(0)}/100<br>
      <span class="muted">counts ${c.weight}x${c.note ? " \u00b7 " + esc(c.note) : ""}</span>`);
    body.appendChild(row);
  }

  const table = document.createElement("div");
  table.className = "scroller";
  table.style.marginTop = "16px";
  table.innerHTML = `<table><thead><tr><th>stat</th>
      <th style="width:310px">0% \u2014 100%</th>
      <th class="num">estimate</th><th class="num">sample</th></tr></thead>
    <tbody></tbody></table>`;
  const tbody = $("tbody", table);
  for (const row of p.rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="label"></td><td></td>
                    <td class="num">${fmtPct(row.value)}</td>
                    <td class="num small muted">${row.opps}</td>`;
    const label = $(".label", tr);
    label.appendChild(document.createTextNode(row.label));
    label.appendChild(info(statTip(row.stat, row.label)));
    tr.children[1].appendChild(statRow(row));
    tbody.appendChild(tr);
  }
  body.appendChild(table);
  return card;
}

/* A written-to-order description of this specific player, on demand.
   Only offered on a saved player: it costs a model call, and an unsaved
   session has no stable identity to attach the result to. */
function buildNarrator(box, profile) {
  const button = document.createElement("button");
  button.className = "act small";
  button.textContent = "Describe this player";
  const out = document.createElement("div");
  out.className = "narration";
  box.append(button, out);

  button.onclick = async () => {
    button.disabled = true;
    const original = button.textContent;
    button.textContent = "writing\u2026";
    out.innerHTML = "";
    try {
      const result = await post("/api/narrate", {profile: profile});
      out.innerHTML = `<blockquote>${esc(result.text)}</blockquote>
        <div class="small muted">written by ${esc(result.model)} from the
          numbers on this page \u2014 it is given the computed profile and
          cannot state a figure the profile did not produce</div>`;
      button.textContent = "Rewrite";
    } catch (err) {
      out.innerHTML = `<div class="small err">${esc(err.message)}</div>`;
      button.textContent = original;
    }
    button.disabled = false;
  };
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
    b.innerHTML = `<span>${esc(p.name)}</span>
      <span class="meta">${esc(p.regime_label)} \u00b7 ${p.hands}h</span>`;
    b.onclick = () => show(i);
    strip.appendChild(b);
  });
  show(0);
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
async function askIdentity(token, questions, onDone) {
  const modal = $("#modal");
  modal.innerHTML = `
    <div class="veil"><div class="sheet">
      <h2 style="margin-top:0">Same player?</h2>
      <p class="small muted" style="margin-top:0">
        These accounts might belong to one person. Answering now means the
        session below is read with their hands pooled. Nothing is saved either
        way \u2014 but merging two real players makes both profiles fiction, so
        the tool will not guess.</p>
      <div id="questions"></div>
      <div class="row" style="justify-content:flex-end;margin-top:18px">
        <button class="act" id="cancel">Keep them separate</button>
        <button class="act primary" id="confirm">Apply</button>
      </div>
    </div></div>`;
  const box = $("#questions");
  for (const q of questions) {
    const div = document.createElement("div");
    div.className = "q";
    const names = (q.names || []).map((n, i) => `
      <label><input type="radio" name="name-${esc(q.id)}" value="${esc(n)}"
        ${n === q.default_name ? "checked" : ""}>keep \u201c${esc(n)}\u201d</label>`).join("");
    div.innerHTML = `
      <div class="q-prompt">${esc(q.prompt)}</div>
      <div class="sides">
        <div class="side"><b>${esc(q.left.name)}</b>
          <div class="small muted">${esc(q.left.where)} \u00b7 ${q.left.hands} hands</div></div>
        <div class="side"><b>${esc(q.right.name)}</b>
          <div class="small muted">${esc(q.right.where)} \u00b7 ${q.right.hands} hands</div></div>
      </div>
      <div class="small muted">${esc(q.detail)}</div>
      <div class="choice">
        <label><input type="radio" name="${esc(q.id)}" value="yes"
          ${q.default ? "checked" : ""}>Same player</label>
        <label><input type="radio" name="${esc(q.id)}" value="no"
          ${q.default ? "" : "checked"}>Different people</label>
      </div>
      <div class="choice namechoice" ${q.default ? "" : 'hidden'}>${names}</div>`;
    box.appendChild(div);
    // The name only matters if they are the same person.
    div.querySelectorAll(`input[name="${CSS.escape(q.id)}"]`).forEach(radio =>
      radio.onchange = () => {
        $(".namechoice", div).hidden = radio.value !== "yes";
      });
  }
  $("#cancel").onclick = async () => {
    modal.innerHTML = "";
    await post(`/api/session/${token}/identity`, {answers: {}});
    if (onDone) onDone();
  };
  $("#confirm").onclick = async () => {
    const answers = {};
    for (const q of questions) {
      const picked = modal.querySelector(`input[name="${CSS.escape(q.id)}"]:checked`);
      const name = modal.querySelector(`input[name="name-${CSS.escape(q.id)}"]:checked`);
      answers[q.id] = {same: picked ? picked.value === "yes" : q.default,
                       name: name ? name.value : q.default_name};
    }
    modal.innerHTML = "";
    const refreshed = await post(`/api/session/${token}/identity`, {answers});
    state.session = refreshed;
    renderSession();
    if (onDone) onDone();
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
  $("#meta").textContent = `${data.hands} hands \u00b7 ${data.players.length} players`;
  if (!data.players.length) {
    view.innerHTML = `<div class="panel"><h2>nothing stored yet</h2>
      <p class="muted">Read a session on the first tab, then add it to the
      database.</p></div>`;
    return;
  }
  /* One table, sorted however you like. A separate leaderboard tab was the
     same rows in a different order. */
  view.innerHTML = `<div class="panel">
      <div class="spread"><h2>players</h2>
        <span class="small muted">click a column to re-rank</span></div>
      <div id="db-roster"></div></div>
    <div id="suggest-panel" class="panel" hidden>
      <h2>possible same person</h2><div id="suggestions"></div></div>
    <p class="footnote">
      <button class="linkbtn danger-link" id="reset">reset database</button>
      <span class="muted">deletes every hand, player and merge decision</span></p>`;
  $("#db-roster").appendChild(rosterTable(data.players, {
    onClick: p => { state.player = p.player_id; viewPlayer(p.player_id); },
  }));
  $("#reset").onclick = () => confirmReset(data);

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
        viewPlayers();
      } catch (err) { b.textContent = "blocked"; b.classList.add("err"); }
    });
  }
}

async function viewPlayer(id) {
  const view = $("#view");
  const data = await get("/api/player/" + id);
  $("#meta").textContent = data.aliases.map(a => a.name).join(" \u00b7 ");
  view.innerHTML = "";
  const back = document.createElement("p");
  back.innerHTML = `<button class="linkbtn" id="back">\u2190 all players</button>`;
  view.appendChild(back);
  $("#back").onclick = () => { state.player = null; viewPlayers(); };

  const holder = document.createElement("div");
  view.appendChild(holder);
  playerTabs(data.profiles, holder, {narrate: true});

  if (data.by_table && data.by_table.length > 1) {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<details open><summary>split by table size</summary>
      <div class="small muted" style="margin:6px 0 12px">
        The read above pools these. Shown separately so you can check the
        pooling is not hiding a difference.</div>
      <div class="scroller"><table><thead><tr>
        <th>table</th><th class="num">hands</th><th>read</th>
        <th class="num">skill</th><th>biggest leak</th>
      </tr></thead><tbody></tbody></table></div></details>`;
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
  modal.innerHTML = `<div class="veil"><div class="sheet">
    <h2 style="margin-top:0">${esc(headline)}</h2>
    <div class="empty">finding the hands\u2026</div></div></div>`;
  let data;
  try {
    data = await get(`/api/evidence?player=${playerId}&stat=${encodeURIComponent(stat)}`);
  } catch (err) {
    modal.innerHTML = `<div class="veil"><div class="sheet">
      <p class="err">${esc(err.message)}</p>
      <div class="row" style="justify-content:flex-end"><button class="act" id="close">Close</button></div>
      </div></div>`;
    $("#close").onclick = () => { modal.innerHTML = ""; };
    return;
  }
  modal.innerHTML = `<div class="veil"><div class="sheet">
    <div class="spread"><h2 style="margin:0">${esc(headline)}</h2>
      <button class="act" id="close">Close</button></div>
    <p class="small muted">Every hand where this came up: ${data.count} of them,
      ${data.hits} counted toward the read. Click one to replay it. The estimate
      weights hands from other table sizes less, so it can differ slightly from
      this count.</p>
    <div id="evlist"></div>
    <div id="replay"></div></div></div>`;
  $("#close").onclick = () => { modal.innerHTML = ""; };

  const list = $("#evlist");
  for (const h of data.hands) {
    const row = document.createElement("div");
    row.className = "ev";
    const when = h.started_at ? new Date(h.started_at).toLocaleString() : "";
    row.innerHTML = `
      <span class="verdict ${h.hit ? "hit" : "miss"}">${h.hit ? "counted" : "no"}</span>
      <span><span class="cards">${esc(h.board.join(" ")) || "\u2014"}</span>
        <div class="small muted">${esc(h.summary)}</div></span>
      <span class="small muted" style="text-align:right">${h.net_bb > 0 ? "+" : ""}${h.net_bb} bb
        <div style="font-size:11px">${esc(when)}</div></span>`;
    row.onclick = () => showReplay(h.hand_id, playerId);
    list.appendChild(row);
  }
}

async function showReplay(handId, playerId) {
  const box = $("#replay");
  box.innerHTML = `<div class="empty">loading hand\u2026</div>`;
  const r = await get(`/api/hand/${handId}?focus=${playerId}`);
  const seats = r.seats.map(s =>
    `${esc(s.position)} ${esc(s.name)}${s.hole_cards.length
      ? ' <span class="cards">' + esc(s.hole_cards.join(" ")) + "</span>" : ""}`
  ).join(" \u00b7 ");
  box.innerHTML = `<div class="panel" style="margin-top:14px">
    <div class="spread"><h2 style="margin:0">hand replay</h2>
      <span class="small muted">${r.pot_bb} bb pot \u00b7 won by ${esc(r.winners.join(", ") || "\u2014")}</span></div>
    <div class="small muted" style="margin-bottom:10px">${seats}</div>
    <div id="streets"></div></div>`;
  const streets = $("#streets");
  for (const st of r.streets) {
    const div = document.createElement("div");
    div.className = "street";
    div.innerHTML = `<h4>${esc(st.name)}
      <span class="cards" style="color:var(--ink)">${esc(st.new_cards.join(" "))}</span></h4>`;
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
  box.scrollIntoView({behavior: "smooth", block: "nearest"});
}

/* ---- tabs ---- */
function switchTab(tab) {
  state.tab = tab;
  if (tab === "players") state.player = null;
  document.querySelectorAll("nav button").forEach(b =>
    b.classList.toggle("on", b.dataset.tab === tab));
  $("#meta").textContent = "";
  render();
}
async function render() {
  try {
    if (!state.glossary) state.glossary = await get("/api/glossary");
    if (state.tab === "session") viewSession();
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
