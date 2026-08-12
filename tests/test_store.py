"""Persistence, identity resolution and the command line."""

import json

import pytest

from villain.cli import main
from villain.db import Store
from villain.identity import name_similarity, normalise, suggest_links
from villain.parsers import parse_file


@pytest.fixture
def store(tmp_path, hands):
    with Store(tmp_path / "v.db") as s:
        s.add_hands(hands)
        yield s


def test_import_is_idempotent(tmp_path, hands):
    with Store(tmp_path / "v.db") as s:
        first = s.add_hands(hands)
        second = s.add_hands(hands)
    assert first.hands_new == len(hands)
    assert second.hands_new == 0
    assert second.duplicates == len(hands)


def test_players_accumulate_hands(store, hands):
    rows = store.players()
    assert rows
    total = sum(r["hands"] or 0 for r in rows)
    assert total == sum(len(h.seats) for h in hands)


def test_books_are_split_by_table_size(store):
    busiest = max(store.players(), key=lambda r: r["hands"] or 0)
    books = store.books(int(busiest["id"]))
    assert books
    assert all(book.regime == key for key, book in books.items())


def test_hands_are_the_source_of_truth(store):
    """Books are a cache: deleting and rebuilding them must change nothing."""
    player = max(store.players(), key=lambda r: r["hands"] or 0)
    before = {r: b.hands for r, b in store.books(int(player["id"])).items()}
    store.conn.execute("DELETE FROM ratios")
    store.conn.execute("DELETE FROM books")
    store.rebuild()
    after = {r: b.hands for r, b in store.books(int(player["id"])).items()}
    assert before == after


def test_players_in_the_same_hand_can_never_be_merged(store, hands):
    seats = hands[0].seats
    a = store.player_for(hands[0].site, seats[0].player_id, seats[0].name)
    b = store.player_for(hands[0].site, seats[1].player_id, seats[1].name)
    assert store.are_distinct(a, b)
    with pytest.raises(ValueError, match="same hand"):
        store.link(a, b)


def test_linking_pools_two_accounts(tmp_path, hands):
    """The same human under two account ids ends up with one merged profile."""
    with Store(tmp_path / "v.db") as s:
        s.add_hands(hands)
        # Forge a second account that never shares a hand with the original.
        original = max(s.players(), key=lambda r: r["hands"] or 0)
        s.conn.execute(
            "INSERT INTO players (display_name, created_at) VALUES ('Ghost', 0)")
        ghost = s.conn.execute("SELECT MAX(id) m FROM players").fetchone()["m"]
        before = sum(b.hands for b in s.books(int(original["id"])).values())
        s.link(int(original["id"]), int(ghost))
        after = sum(b.hands for b in s.books(int(original["id"])).values())
        remaining = [r["id"] for r in s.players()]
    assert after == before
    assert ghost not in remaining


@pytest.mark.parametrize("a,b,expected", [
    ("Arnav", "Arnav2", 1.0),
    ("DavidMazour", "DavidMazour2", 1.0),
    ("player_one", "PlayerOne", 1.0),
])
def test_trailing_digits_and_punctuation_are_noise(a, b, expected):
    assert name_similarity(a, b) == expected


def test_unrelated_names_stay_apart():
    assert name_similarity("aryan", "Arnav2") < 0.8
    assert normalise("Bob!!") == "bob"


def test_suggestions_never_include_impossible_merges(store):
    for suggestion in suggest_links(store):
        assert not store.are_distinct(suggestion.keep, suggestion.absorb)


def test_fit_priors_refuses_a_thin_pool(store):
    assert store.fit_priors(min_players=8) == {}


# -- CLI --------------------------------------------------------------------

def test_cli_import_and_profile(tmp_path, capsys):
    db = tmp_path / "cli.db"
    from tests.conftest import FIXTURE
    assert main(["--db", str(db), "import", str(FIXTURE)]) == 0
    assert "new hands" in capsys.readouterr().out

    assert main(["--db", str(db), "players"]) == 0
    listing = capsys.readouterr().out
    assert "player1" in listing

    assert main(["--db", str(db), "profile", "player1"]) == 0
    card = capsys.readouterr().out
    assert "READ:" in card and "SKILL:" in card


def test_cli_json_is_machine_readable(tmp_path, capsys):
    from tests.conftest import FIXTURE
    db = tmp_path / "cli.db"
    main(["--db", str(db), "import", str(FIXTURE)])
    capsys.readouterr()
    main(["--db", str(db), "profile", "player1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload
    entry = payload[0]
    assert entry["archetype"] != "unknown"
    assert "skill" in entry and "leaks" in entry
    # Internal counters must not leak into the public export.
    assert not [k for k in entry["stats"] if k.startswith(("act:", "seat:", "saw:"))]


def test_cli_scout_needs_no_database(tmp_path, capsys):
    from tests.conftest import FIXTURE
    assert main(["--db", str(tmp_path / "unused.db"), "scout", str(FIXTURE),
                 "--min-hands", "5"]) == 0
    out = capsys.readouterr().out
    assert "profiles" in out


def test_cli_rejects_unknown_files(tmp_path, capsys):
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a hand history")
    assert main(["--db", str(tmp_path / "v.db"), "import", str(junk)]) == 1
