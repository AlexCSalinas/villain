"""The lineup briefing: who is here, what to do, and no invented win rate."""

from villain.db import Store
from villain.table import brief


def _store(tmp_path):
    from tests.conftest import FIXTURE
    from villain.parsers import parse_file
    store = Store(tmp_path / "t.db")
    store.add_hands(parse_file(FIXTURE))
    return store


def test_a_name_not_in_the_database_is_named_not_guessed(tmp_path):
    with _store(tmp_path) as store:
        out = brief(store, ["definitely-not-a-player"])
    assert out.reads == []
    assert "definitely-not-a-player" in out.missing
    assert "definitely-not-a-player" in str(out)


def test_known_players_get_a_read(tmp_path):
    with _store(tmp_path) as store:
        names = [r["display_name"] for r in store.players()][:3]
        out = brief(store, names)
    assert len(out.reads) == len(names)
    for read in out.reads:
        assert read.hands > 0
        assert read.status in ("confirmed", "watch", "rated", "none")


def test_it_never_states_an_expected_win_rate(tmp_path):
    """The handoff's constraint, asserted rather than trusted.

    Per-session results vary by ~183 bb/100, so separating two lineups needs a
    ~229 bb/100 gap. Any lineup-level EV number would be noise with a decimal
    point on it.
    """
    with _store(tmp_path) as store:
        names = [r["display_name"] for r in store.players()][:4]
        text = str(brief(store, names))
    assert "No expected win rate" in text, "the constraint has to be stated"
    # Check the briefing body, not the disclaimer that explains the absence.
    body = text.split("No expected win rate")[0].lower()
    for banned in ("expected win rate", "you will make", "expected value of this table",
                   "table ev", "lineup ev", "you should win"):
        assert banned not in body, f"the briefing claims {banned!r}"
    # And no bb/100 figure that is not attached to a single player's leak.
    for line in body.splitlines():
        if "bb/100" in line:
            assert line.strip().startswith(("->", "?", "~")), \
                f"a bb/100 number outside a per-player read: {line!r}"


def test_confirmed_reads_are_listed_before_unconfirmed(tmp_path):
    with _store(tmp_path) as store:
        names = [r["display_name"] for r in store.players()]
        out = brief(store, names)
    rank = {"confirmed": 0, "watch": 1, "rated": 2, "none": 3}
    seen = [rank[r.status] for r in out.reads]
    assert seen == sorted(seen), "act-on-this-first ordering is the whole point"
