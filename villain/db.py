"""Persistence: hands in, profiles out, forever.

Two decisions shape this module.

**Hands are the source of truth, statistics are a cache.** Stat definitions
change -- a c-bet gets redefined, a new leak rule needs a counter nobody was
recording -- so every hand is stored in canonical form and ``rebuild()``
recomputes every book from scratch. Without that, a definition change leaves
old players wrong until they happen to sit down again.

**Identity is separate from account.** The same human is ``DavidMazour`` at one
table and ``DavidMazour2`` at the next, and the profile is worthless if it
restarts each time. So site accounts are *aliases* pointing at an internal
player, aliases can be merged, and merging is guarded by co-occurrence: two
accounts dealt into the same hand are provably different people and can never
be linked, however similar their names look.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .features import record_hand
from .model import Hand, hand_from_dict, hand_to_dict
from .stats import Meter, Ratio, StatBook

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    notes        TEXT DEFAULT ''
);
-- One row per site account. Several may point at the same player.
-- ``account`` is normally the site's own player id. When the user says that
-- one account id is being shared by two different people -- a seat handed
-- over, a renamed guest -- the key becomes "<account>#<name>" so the two get
-- separate identities without losing the hands already attributed.
CREATE TABLE IF NOT EXISTS aliases (
    site       TEXT NOT NULL,
    account    TEXT NOT NULL,
    name       TEXT NOT NULL,
    player_id  INTEGER NOT NULL REFERENCES players(id),
    hands      INTEGER NOT NULL DEFAULT 0,
    last_seen  INTEGER,
    PRIMARY KEY (site, account)
);
CREATE INDEX IF NOT EXISTS aliases_player ON aliases(player_id);

-- Provably different people: dealt into the same hand at the same time.
CREATE TABLE IF NOT EXISTS distinct_pairs (
    a INTEGER NOT NULL, b INTEGER NOT NULL,
    PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS hands (
    hand_id    TEXT PRIMARY KEY,
    site       TEXT NOT NULL,
    table_id   TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    players    INTEGER NOT NULL,
    payload    BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS hands_time ON hands(started_at);

CREATE TABLE IF NOT EXISTS books (
    player_id  INTEGER NOT NULL REFERENCES players(id),
    regime     TEXT NOT NULL,
    hands      INTEGER NOT NULL DEFAULT 0,
    first_seen INTEGER, last_seen INTEGER,
    PRIMARY KEY (player_id, regime)
);
CREATE TABLE IF NOT EXISTS ratios (
    player_id INTEGER NOT NULL, regime TEXT NOT NULL, stat TEXT NOT NULL,
    hits REAL NOT NULL, opps REAL NOT NULL,
    PRIMARY KEY (player_id, regime, stat)
);
CREATE TABLE IF NOT EXISTS meters (
    player_id INTEGER NOT NULL, regime TEXT NOT NULL, stat TEXT NOT NULL,
    n REAL NOT NULL, total REAL NOT NULL, sumsq REAL NOT NULL,
    PRIMARY KEY (player_id, regime, stat)
);
-- Priors re-estimated from this database's own players, which is what makes a
-- home game stop being measured against an online population.
CREATE TABLE IF NOT EXISTS fitted_priors (
    regime TEXT NOT NULL, stat TEXT NOT NULL,
    mean REAL NOT NULL, strength REAL NOT NULL, players INTEGER NOT NULL,
    fitted_at INTEGER NOT NULL,
    PRIMARY KEY (regime, stat)
);
CREATE TABLE IF NOT EXISTS notes (
    player_id INTEGER NOT NULL REFERENCES players(id),
    created_at INTEGER NOT NULL,
    body TEXT NOT NULL
);
"""

DEFAULT_PATH = Path.home() / ".villain" / "villain.db"


def split_key(account: str, name: str) -> str:
    """Alias key for an account id the user has split between two people."""
    return f"{account}#{name.strip().lower()}"


def alias_key(site: str, account: str, name: str,
              name_splits: set[tuple[str, str, str]]) -> str:
    return split_key(account, name) if (site, account, name) in name_splits else account


@dataclass
class ImportReport:
    files: int = 0
    hands_seen: int = 0
    hands_new: int = 0
    duplicates: int = 0
    players_new: int = 0
    players: dict[str, int] = field(default_factory=dict)   # display name -> hands added

    def __str__(self) -> str:
        return (f"{self.hands_new} new hands from {self.files} file(s) "
                f"({self.duplicates} already known), {self.players_new} new player(s)")


class Store:
    """A villain database. Safe to open repeatedly; migrations are idempotent."""

    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.conn.commit()
        self.close()

    # -- identity --------------------------------------------------------

    def player_for(self, site: str, account: str, name: str,
                   alias_key: str | None = None) -> int:
        """Internal player id for a site account, creating one if needed.

        ``alias_key`` overrides the storage key, which is how a shared account
        id gets split into two identities.
        """
        key = alias_key or account
        row = self.conn.execute(
            "SELECT player_id FROM aliases WHERE site = ? AND account = ?",
            (site, key),
        ).fetchone()
        if row:
            return int(row["player_id"])
        cur = self.conn.execute(
            "INSERT INTO players (display_name, created_at) VALUES (?, ?)",
            (name or account, int(time.time())),
        )
        player_id = int(cur.lastrowid)
        self.conn.execute(
            "INSERT INTO aliases (site, account, name, player_id) VALUES (?, ?, ?, ?)",
            (site, key, name or account, player_id),
        )
        return player_id

    def mark_distinct(self, player_ids: Iterable[int]) -> None:
        """Record that these players sat in one hand, so they are not one person."""
        ids = sorted(set(player_ids))
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
        if pairs:
            self.conn.executemany(
                "INSERT OR IGNORE INTO distinct_pairs (a, b) VALUES (?, ?)", pairs)

    def are_distinct(self, a: int, b: int) -> bool:
        lo, hi = sorted((a, b))
        return self.conn.execute(
            "SELECT 1 FROM distinct_pairs WHERE a = ? AND b = ?", (lo, hi)
        ).fetchone() is not None

    def link(self, keep: int, absorb: int) -> None:
        """Declare two players the same human, folding ``absorb`` into ``keep``."""
        if keep == absorb:
            return
        if self.are_distinct(keep, absorb):
            raise ValueError(
                f"players {keep} and {absorb} were dealt into the same hand and "
                "cannot be the same person")
        self.conn.execute("UPDATE aliases SET player_id = ? WHERE player_id = ?",
                          (keep, absorb))
        self.conn.execute("UPDATE notes SET player_id = ? WHERE player_id = ?",
                          (keep, absorb))
        for table in ("books", "ratios", "meters"):
            self.conn.execute(f"DELETE FROM {table} WHERE player_id IN (?, ?)",
                              (keep, absorb))
        self.conn.execute("DELETE FROM players WHERE id = ?", (absorb,))
        # Inherit the absorbed player's distinctness constraints.
        self.conn.execute(
            "UPDATE OR IGNORE distinct_pairs SET a = ? WHERE a = ?", (keep, absorb))
        self.conn.execute(
            "UPDATE OR IGNORE distinct_pairs SET b = ? WHERE b = ?", (keep, absorb))
        self.conn.execute("DELETE FROM distinct_pairs WHERE a = b")
        self.conn.commit()
        self.rebuild(only=[keep])

    def players(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT p.id, p.display_name, p.notes,
                      (SELECT group_concat(name, ', ') FROM aliases a
                        WHERE a.player_id = p.id) AS aliases,
                      (SELECT COALESCE(SUM(hands), 0) FROM books b
                        WHERE b.player_id = p.id) AS hands,
                      (SELECT MAX(last_seen) FROM books b
                        WHERE b.player_id = p.id) AS last_seen
                 FROM players p ORDER BY hands DESC"""
        ).fetchall()

    def find_player(self, needle: str) -> list[sqlite3.Row]:
        """Look a player up by display name or any alias, case-insensitively."""
        like = f"%{needle.lower()}%"
        return self.conn.execute(
            """SELECT DISTINCT p.id, p.display_name FROM players p
                 LEFT JOIN aliases a ON a.player_id = p.id
                WHERE LOWER(p.display_name) LIKE ? OR LOWER(a.name) LIKE ?
                   OR LOWER(a.account) = ?""",
            (like, like, needle.lower()),
        ).fetchall()

    # -- hands -----------------------------------------------------------

    def add_hands(self, hands: Iterable[Hand], report: ImportReport | None = None,
                  name_splits: set[tuple[str, str, str]] | None = None) -> ImportReport:
        """Store hands, skipping any already seen, and update the books.

        ``name_splits`` holds ``(site, account, name)`` triples the user has
        declared to be a *different* person from whoever already owns that
        account id.
        """
        report = report or ImportReport()
        fresh: list[Hand] = []
        for hand in hands:
            report.hands_seen += 1
            known = self.conn.execute(
                "SELECT 1 FROM hands WHERE hand_id = ?", (hand.hand_id,)).fetchone()
            if known:
                report.duplicates += 1
                continue
            payload = gzip.compress(json.dumps(hand_to_dict(hand)).encode())
            self.conn.execute(
                "INSERT INTO hands (hand_id, site, table_id, started_at, players, payload)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (hand.hand_id, hand.site, hand.table_id, hand.started_at,
                 len(hand.seats), payload),
            )
            fresh.append(hand)
            report.hands_new += 1

        touched = self._ingest(fresh, report, name_splits or set())
        self.conn.commit()
        if touched:
            self.rebuild(only=sorted(touched))
        return report

    def _ingest(self, hands: list[Hand], report: ImportReport,
                name_splits: set[tuple[str, str, str]]) -> set[int]:
        """Register players and aliases for a batch of hands."""
        touched: set[int] = set()
        for hand in hands:
            ids = []
            for seat in hand.seats:
                before = self.conn.execute("SELECT COUNT(*) c FROM players").fetchone()["c"]
                key = alias_key(hand.site, seat.player_id, seat.name, name_splits)
                pid = self.player_for(hand.site, seat.player_id, seat.name, alias_key=key)
                after = self.conn.execute("SELECT COUNT(*) c FROM players").fetchone()["c"]
                if after > before:
                    report.players_new += 1
                ids.append(pid)
                touched.add(pid)
                report.players[seat.name] = report.players.get(seat.name, 0) + 1
                self.conn.execute(
                    "UPDATE aliases SET hands = hands + 1, last_seen = MAX(COALESCE(last_seen, 0), ?),"
                    " name = ? WHERE site = ? AND account = ?",
                    (hand.started_at, seat.name, hand.site, key))
            self.mark_distinct(ids)
        return touched

    def stored_hands(self, player_id: int | None = None) -> list[Hand]:
        if player_id is None:
            rows = self.conn.execute(
                "SELECT payload FROM hands ORDER BY started_at").fetchall()
        else:
            accounts = self.conn.execute(
                "SELECT site, account FROM aliases WHERE player_id = ?", (player_id,)
            ).fetchall()
            keys = {(r["site"], r["account"]) for r in accounts}
            rows = [
                r for r in self.conn.execute(
                    "SELECT site, payload FROM hands ORDER BY started_at").fetchall()
                if any((r["site"], seat["player_id"]) in keys
                       or (r["site"], split_key(seat["player_id"], seat["name"])) in keys
                       for seat in json.loads(gzip.decompress(r["payload"]))["seats"])
            ]
        return [hand_from_dict(json.loads(gzip.decompress(r["payload"]))) for r in rows]

    # -- books -----------------------------------------------------------

    def rebuild(self, only: list[int] | None = None) -> int:
        """Recompute books from stored hands. The cache is always disposable."""
        alias_map = {
            (r["site"], r["account"]): int(r["player_id"])
            for r in self.conn.execute("SELECT site, account, player_id FROM aliases")
        }

        def resolve(site: str, account: str, name: str) -> int | None:
            """Split key first, then the bare account.

            Order matters: ``"<account>#<name>"`` is the more specific claim.
            Checking the bare account first would hand every split hand back to
            whoever owned the account originally, which is exactly the merge
            the user declined.
            """
            hit = alias_map.get((site, split_key(account, name)))
            if hit is not None:
                return hit
            return alias_map.get((site, account))
        books: dict[str, dict[str, StatBook]] = {}
        names: dict[str, str] = {}
        for row in self.conn.execute("SELECT site, payload FROM hands ORDER BY started_at"):
            data = json.loads(gzip.decompress(row["payload"]))
            hand = hand_from_dict(data)
            # Re-key seats onto internal player ids so merged aliases pool.
            for seat in hand.seats:
                pid = resolve(hand.site, seat.player_id, seat.name)
                if pid is None:
                    continue
                names[str(pid)] = seat.name or names.get(str(pid), "")
                seat.player_id = str(pid)
            record_hand(hand, books)

        wanted = {str(p) for p in only} if only else None
        written = 0
        for pid, by_regime in books.items():
            if wanted is not None and pid not in wanted:
                continue
            self._write_books(int(pid), by_regime, names.get(pid, ""))
            written += 1
        self.conn.commit()
        return written

    def _write_books(self, player_id: int, by_regime: dict[str, StatBook],
                     name: str) -> None:
        for table in ("books", "ratios", "meters"):
            self.conn.execute(f"DELETE FROM {table} WHERE player_id = ?", (player_id,))
        for regime, book in by_regime.items():
            self.conn.execute(
                "INSERT INTO books (player_id, regime, hands, first_seen, last_seen)"
                " VALUES (?, ?, ?, ?, ?)",
                (player_id, regime, book.hands, book.first_seen, book.last_seen))
            self.conn.executemany(
                "INSERT INTO ratios (player_id, regime, stat, hits, opps)"
                " VALUES (?, ?, ?, ?, ?)",
                [(player_id, regime, stat, r.hits, r.opps)
                 for stat, r in book.ratios.items()])
            self.conn.executemany(
                "INSERT INTO meters (player_id, regime, stat, n, total, sumsq)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(player_id, regime, stat, m.n, m.total, m.sumsq)
                 for stat, m in book.meters.items()])
        if name:
            self.conn.execute(
                "UPDATE players SET display_name = ? WHERE id = ? AND display_name = ''",
                (name, player_id))

    def books(self, player_id: int) -> dict[str, StatBook]:
        """Every regime book for a player, read back out of the cache."""
        name_row = self.conn.execute(
            "SELECT display_name FROM players WHERE id = ?", (player_id,)).fetchone()
        name = name_row["display_name"] if name_row else str(player_id)
        out: dict[str, StatBook] = {}
        for row in self.conn.execute(
                "SELECT regime, hands, first_seen, last_seen FROM books WHERE player_id = ?",
                (player_id,)):
            out[row["regime"]] = StatBook(
                player_id=str(player_id), name=name, regime=row["regime"],
                hands=row["hands"], first_seen=row["first_seen"], last_seen=row["last_seen"])
        for row in self.conn.execute(
                "SELECT regime, stat, hits, opps FROM ratios WHERE player_id = ?",
                (player_id,)):
            book = out.get(row["regime"])
            if book is not None:
                book.ratios[row["stat"]] = Ratio(row["hits"], row["opps"])
        for row in self.conn.execute(
                "SELECT regime, stat, n, total, sumsq FROM meters WHERE player_id = ?",
                (player_id,)):
            book = out.get(row["regime"])
            if book is not None:
                book.meters[row["stat"]] = Meter(row["n"], row["total"], row["sumsq"])
        return out

    def population_samples(self, stat_filter=None) -> dict[str, dict[str, list[tuple[float, float]]]]:
        """Every player's (hits, opps) per stat, per regime -- input to a prior fit."""
        out: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for row in self.conn.execute("SELECT regime, stat, hits, opps FROM ratios"):
            if stat_filter and not stat_filter(row["stat"]):
                continue
            out[row["regime"]][row["stat"]].append((row["hits"], row["opps"]))
        return {r: dict(v) for r, v in out.items()}

    # -- fitted priors ----------------------------------------------------

    def fit_priors(self, min_players: int = 8) -> dict[str, int]:
        """Re-estimate population priors from the players in this database."""
        from .priors import fit_empirical
        fitted: dict[str, int] = {}
        now = int(time.time())
        for regime, samples in self.population_samples().items():
            result = fit_empirical(samples, min_players=min_players)
            if not result:
                continue
            players = self.conn.execute(
                "SELECT COUNT(DISTINCT player_id) c FROM books WHERE regime = ?",
                (regime,)).fetchone()["c"]
            self.conn.executemany(
                "INSERT OR REPLACE INTO fitted_priors"
                " (regime, stat, mean, strength, players, fitted_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [(regime, stat, mean, strength, players, now)
                 for stat, (mean, strength) in result.items()])
            fitted[regime] = len(result)
        self.conn.commit()
        return fitted

    def fitted_priors(self, regime: str) -> dict[str, tuple[float, float]]:
        return {
            row["stat"]: (row["mean"], row["strength"])
            for row in self.conn.execute(
                "SELECT stat, mean, strength FROM fitted_priors WHERE regime = ?",
                (regime,))
        }

    def profiles(self, player_id: int, min_hands: int = 1) -> list:
        """One profile per table size. The detailed view, not the default."""
        from .profile import build_profiles
        books = self.books(player_id)
        if not books:
            return []
        regime = max(books.values(), key=lambda b: b.hands).regime
        return build_profiles(books, min_hands=min_hands,
                              priors=self.fitted_priors(regime) or None)

    def profile(self, player_id: int):
        """The single profile for a player, pooled across table sizes.

        This is the default everywhere. Splitting by table size is how the
        statistics stay meaningful, not how anybody wants to read them.
        """
        from .profile import build_unified, primary_regime
        books = self.books(player_id)
        if not books:
            return None
        return build_unified(books,
                             priors=self.fitted_priors(primary_regime(books)) or None)

    def games(self) -> list[dict]:
        """Every table in the database, with who plays there and how often.

        A ``table_id`` is one game -- a PokerNow room, a home game that keeps
        the same link. Which one to turn up to is a database-wide question and
        the only thing the profiler is missing to answer it.
        """
        rows = self.conn.execute(
            """SELECT h.site, h.table_id,
                      COUNT(*) AS hands,
                      MIN(h.started_at) AS first_seen,
                      MAX(h.started_at) AS last_seen
                 FROM hands h GROUP BY h.site, h.table_id
                 ORDER BY last_seen DESC""").fetchall()
        alias_map = {
            (r["site"], r["account"]): int(r["player_id"])
            for r in self.conn.execute("SELECT site, account, player_id FROM aliases")
        }
        seats: dict[str, dict[int, int]] = {}
        for row in self.conn.execute("SELECT site, table_id, payload FROM hands"):
            data = json.loads(gzip.decompress(row["payload"]))
            counts = seats.setdefault(row["table_id"], {})
            for seat in data["seats"]:
                pid = (alias_map.get((row["site"], seat["player_id"]))
                       or alias_map.get((row["site"],
                                         split_key(seat["player_id"], seat["name"]))))
                if pid is not None:
                    counts[pid] = counts.get(pid, 0) + 1
        return [
            {"site": r["site"], "table_id": r["table_id"], "hands": r["hands"],
             "first_seen": r["first_seen"], "last_seen": r["last_seen"],
             "seats": seats.get(r["table_id"], {})}
            for r in rows
        ]

    def player_hands(self, player_id: int) -> list[Hand]:
        """Stored hands this player was dealt into, keyed to internal ids.

        The same re-keying ``rebuild`` does, so anything computed from these
        hands lines up with the statistics computed from them.
        """
        accounts = {
            (r["site"], r["account"]): int(r["player_id"])
            for r in self.conn.execute("SELECT site, account, player_id FROM aliases")
        }

        def resolve(site, account, name):
            return (accounts.get((site, split_key(account, name)))
                    or accounts.get((site, account)))

        out = []
        for row in self.conn.execute(
                "SELECT site, payload FROM hands ORDER BY started_at"):
            data = json.loads(gzip.decompress(row["payload"]))
            hand = hand_from_dict(data)
            ids = []
            for seat in hand.seats:
                pid = resolve(hand.site, seat.player_id, seat.name)
                seat.player_id = str(pid) if pid is not None else seat.player_id
                ids.append(pid)
            if player_id in ids:
                out.append(hand)
        return out

    def reset(self) -> dict[str, int]:
        """Empty the database, keeping the file and its schema.

        There is no undo. Hands are the source of truth for everything else, so
        once they are gone every profile, alias and merge decision goes with
        them -- re-importing the original exports rebuilds the statistics, but
        not the identity decisions made along the way.
        """
        counts = {
            "hands": self.conn.execute("SELECT COUNT(*) c FROM hands").fetchone()["c"],
            "players": self.conn.execute("SELECT COUNT(*) c FROM players").fetchone()["c"],
        }
        for table in ("ratios", "meters", "books", "notes", "distinct_pairs",
                      "aliases", "fitted_priors", "hands", "players"):
            self.conn.execute(f"DELETE FROM {table}")
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'players'")
        self.conn.commit()
        self.conn.execute("VACUUM")
        return counts

    # -- notes -----------------------------------------------------------

    def add_note(self, player_id: int, body: str) -> None:
        self.conn.execute(
            "INSERT INTO notes (player_id, created_at, body) VALUES (?, ?, ?)",
            (player_id, int(time.time()), body))
        self.conn.commit()

    def notes(self, player_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT created_at, body FROM notes WHERE player_id = ? ORDER BY created_at",
            (player_id,)).fetchall()
