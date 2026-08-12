"""The web layer: uploads that touch nothing, and a save that asks first."""

import copy
import json

import pytest

from villain.db import Store, split_key
from villain.identity import session_questions
from villain.web import (MIN_ROSTER_HANDS, SESSIONS, commit_session, parse_upload,
                         profile_payload, roster_payload, session_payload)


@pytest.fixture
def fixture_text():
    from tests.conftest import FIXTURE
    return FIXTURE.read_text()


@pytest.fixture
def session(fixture_text):
    hands = parse_upload("sample.json", fixture_text)
    token = "test-token"
    SESSIONS[token] = {"hands": hands, "files": [{"name": "sample.json",
                                                  "hands": len(hands)}],
                       "created": 0.0}
    yield token
    SESSIONS.pop(token, None)


def test_upload_parses_from_text(fixture_text, hands):
    assert len(parse_upload("sample.json", fixture_text)) == len(hands)


def test_unparseable_upload_raises():
    from villain.parsers import UnknownFormat
    with pytest.raises((UnknownFormat, ValueError)):
        parse_upload("notes.txt", "this is not a hand history")


def test_session_analysis_touches_no_database(session, tmp_path):
    """Reading a file must leave the database exactly as it was."""
    db = tmp_path / "v.db"
    with Store(db) as store:
        before = store.conn.execute("SELECT COUNT(*) c FROM hands").fetchone()["c"]
    payload = session_payload(session)
    assert payload["players"]
    assert payload["saved"] is False
    with Store(db) as store:
        after = store.conn.execute("SELECT COUNT(*) c FROM hands").fetchone()["c"]
    assert before == after == 0


def test_session_payload_is_json_serialisable(session):
    json.dumps(session_payload(session))


def test_commit_stores_the_session(session, tmp_path):
    with Store(tmp_path / "v.db") as store:
        result = commit_session(store, session, {})
        assert result["hands_new"] == len(SESSIONS[session]["hands"])
        assert store.conn.execute("SELECT COUNT(*) c FROM hands").fetchone()["c"] > 0


def test_committing_twice_adds_nothing(session, tmp_path):
    db = tmp_path / "v.db"
    with Store(db) as store:
        commit_session(store, session, {})
    with Store(db) as store:
        again = commit_session(store, session, {})
    assert again["hands_new"] == 0
    assert again["duplicates"] > 0


# -- the questions ----------------------------------------------------------

def test_similar_names_raise_an_alias_question(session, tmp_path):
    """Two accounts that look like one person are asked about, not merged."""
    hands = SESSIONS[session]["hands"]
    twin = copy.deepcopy(hands[:3])
    for hand in twin:
        hand.hand_id += "-twin"
        for seat in hand.seats:
            seat.player_id = seat.player_id + "X"
            seat.name = seat.name + "2"
    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, hands + twin)
    alias = [q for q in questions if q.kind == "alias"]
    assert alias
    assert all(q.default is False for q in alias), "merging must never be the default"


def test_regular_opponents_are_never_offered_as_one(session, tmp_path):
    """A pair that shares more than a glitch's worth of hands is two people.

    A single shared hand is tolerated on purpose -- a reconnect can leave a
    stale seat -- but the question then says so, so the user is never asked to
    overrule the evidence without being shown it.
    """
    from villain.db import SPURIOUS_OVERLAP
    hands = SESSIONS[session]["hands"]
    with Store(tmp_path / "v.db") as store:
        for q in session_questions(store, hands):
            if q.kind != "alias":
                continue
            names = {q.left["name"], q.right["name"]}
            shared = sum(1 for h in hands if names <= {s.name for s in h.seats})
            assert shared <= SPURIOUS_OVERLAP
            if shared:
                assert "seated together" in q.detail


def test_same_account_new_name_raises_a_rename_question(session, tmp_path):
    """The PokerNow case: one account id, a different display name."""
    db = tmp_path / "v.db"
    hands = SESSIONS[session]["hands"]
    with Store(db) as store:
        store.add_hands(hands)
        later = copy.deepcopy(hands)
        for hand in later:
            hand.hand_id += "-later"
            for seat in hand.seats:
                if seat.name == "player1":
                    seat.name = "player1 renamed"
        questions = session_questions(store, later)
    renames = [q for q in questions if q.kind == "rename"]
    assert len(renames) == 1
    assert renames[0].default is True, "a shared account id defaults to one person"
    assert "player1" in renames[0].left["name"]


def test_declining_a_rename_splits_the_identity(tmp_path, hands):
    """Answering 'different people' must keep the two apart from then on."""
    db = tmp_path / "v.db"
    token = "split-token"
    with Store(db) as store:
        store.add_hands(hands)
        later = copy.deepcopy(hands)
        account = None
        for hand in later:
            hand.hand_id += "-later"
            for seat in hand.seats:
                if seat.name == "player1":
                    account, seat.name = seat.player_id, "someone else"
        SESSIONS[token] = {"hands": later, "files": [], "created": 0.0,
                           "questions": store and session_questions(store, later)}
        rename = next(q for q in SESSIONS[token]["questions"] if q.kind == "rename")
        commit_session(store, token, {rename.id: False})
        names = {r["display_name"]: r["hands"] for r in store.players()}
    SESSIONS.pop(token, None)
    assert "someone else" in names
    assert "player1" in names
    assert names["someone else"] > 0, "the split identity must own its own hands"


def test_accepting_a_rename_pools_the_hands(tmp_path, hands):
    db = tmp_path / "v.db"
    token = "rename-token"
    with Store(db) as store:
        store.add_hands(hands)
        before = {r["display_name"]: r["hands"] for r in store.players()}
        later = copy.deepcopy(hands)
        for hand in later:
            hand.hand_id += "-later"
            for seat in hand.seats:
                if seat.name == "player1":
                    seat.name = "player1 v2"
        SESSIONS[token] = {"hands": later, "files": [], "created": 0.0,
                           "questions": session_questions(store, later)}
        rename = next(q for q in SESSIONS[token]["questions"] if q.kind == "rename")
        commit_session(store, token, {rename.id: True})
        after = {r["display_name"]: r["hands"] for r in store.players()}
    SESSIONS.pop(token, None)
    assert "player1" not in after, "the confirmed rename becomes the display name"
    assert after["player1 v2"] == before["player1"] * 2


def test_roster_hides_rounding_error_profiles(tmp_path, hands):
    """A one-hand book is a rounding error, not a profile.

    The exception is a player who has nothing bigger: they still get a single
    row, because vanishing from the roster entirely is worse than a thin read.
    """
    from collections import defaultdict
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        rows = roster_payload(store)
    assert rows
    by_player = defaultdict(list)
    for row in rows:
        by_player[row["player_id"]].append(row["hands"])
    for player, counts in by_player.items():
        assert all(c >= MIN_ROSTER_HANDS for c in counts) or len(counts) == 1


def test_profile_payload_carries_chart_references(tmp_path, hands):
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        player = max(store.players(), key=lambda r: r["hands"] or 0)
        payload = profile_payload(store.profiles(int(player["id"]))[0])
    assert payload["rows"]
    for row in payload["rows"]:
        assert 0 <= row["lo"] <= row["value"] <= row["hi"] <= 1
        assert "population" in row
    folds = [r for r in payload["rows"] if r["stat"].startswith("fold_vs_bet")]
    assert folds and all("breakeven" in r for r in folds), \
        "fold stats need the breakeven tick, it is the point of the chart"


def test_split_key_is_stable():
    assert split_key("abc", "Dave M") == split_key("abc", " dave m ")


# -- glossary and reset -----------------------------------------------------

def test_every_displayed_stat_has_an_explanation():
    """A number nobody can interpret is worse than no number."""
    from villain.exploits import RULES
    from villain.glossary import stat_help
    from villain.web import DISPLAY_STATS
    missing = [s for s, _, _ in DISPLAY_STATS if not stat_help(s)]
    missing += [r.stat for r in RULES if not stat_help(r.stat)]
    assert not missing, f"no glossary entry for {sorted(set(missing))}"


def test_stat_help_covers_both_directions():
    """Most of these are exploitable in both directions, and the two call for
    opposite play -- an explanation that only covers one is a trap."""
    from villain.glossary import STATS
    for stat, entry in STATS.items():
        assert entry["what"] and entry["high"] and entry["low"], stat
        assert entry["high"] != entry["low"], stat


def test_sample_quality_words_are_all_defined():
    from villain.glossary import TERMS
    from villain.profile import Profile
    for hands in (10, 100, 300, 900):
        quality = Profile("x", "x", hands, "hu", 2.0).sample_quality
        assert quality in TERMS, f"{quality!r} appears in the UI but is undefined"


def test_leak_tiers_are_all_defined():
    from villain.exploits import TIERS
    from villain.glossary import TERMS
    for _, label in TIERS:
        assert label in TERMS


def test_glossary_payload_is_serialisable():
    from villain.glossary import payload
    assert json.dumps(payload())


def test_reset_empties_the_database(tmp_path, hands):
    db = tmp_path / "v.db"
    with Store(db) as store:
        store.add_hands(hands)
        removed = store.reset()
        assert removed["hands"] == len(hands)
        assert store.players() == []
        assert store.conn.execute("SELECT COUNT(*) c FROM hands").fetchone()["c"] == 0


def test_database_is_reusable_after_reset(tmp_path, hands):
    db = tmp_path / "v.db"
    with Store(db) as store:
        store.add_hands(hands)
        store.reset()
        report = store.add_hands(hands)
        assert report.hands_new == len(hands), "reset must clear the dedupe index too"
        assert store.players()


# -- identity settled at upload --------------------------------------------

def test_merge_answers_pool_the_session_before_it_is_saved(session, tmp_path):
    """The point of asking at upload: the session you read is already pooled."""
    from villain.identity import session_questions
    from villain.web import SESSIONS, apply_answers, session_payload
    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, SESSIONS[session]["hands"])
    alias = [q for q in questions if q.kind == "alias"]
    if not alias:
        pytest.skip("fixture has no same-person candidates")
    SESSIONS[session]["questions"] = questions

    before = {r["name"]: r["hands"] for r in session_payload(session)["players"]}
    q = alias[0]
    apply_answers(SESSIONS[session], {q.id: {"same": True, "name": q.names[0]}})
    after = {r["name"]: r["hands"] for r in session_payload(session)["players"]}

    assert len(after) < len(before), "the two accounts should now be one player"
    assert q.names[0] in after, "the chosen name should be the one kept"
    assert sum(after.values()) == sum(before.values()), "no hands lost in the merge"


def test_the_chosen_name_is_honoured(session, tmp_path):
    from villain.identity import session_questions
    from villain.web import SESSIONS, apply_answers, session_payload
    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, SESSIONS[session]["hands"])
    alias = [q for q in questions if q.kind == "alias"]
    if not alias:
        pytest.skip("fixture has no same-person candidates")
    SESSIONS[session]["questions"] = questions
    q = alias[0]
    # Deliberately pick the name that is *not* the default.
    other = [n for n in q.names if n != q.default_name][0]
    apply_answers(SESSIONS[session], {q.id: {"same": True, "name": other}})
    names = {r["name"] for r in session_payload(session)["players"]}
    assert other in names
    assert q.default_name not in names


def test_declining_leaves_the_session_untouched(session, tmp_path):
    from villain.identity import session_questions
    from villain.web import SESSIONS, apply_answers, session_payload
    with Store(tmp_path / "v.db") as store:
        SESSIONS[session]["questions"] = session_questions(store, SESSIONS[session]["hands"])
    before = {r["name"] for r in session_payload(session)["players"]}
    apply_answers(SESSIONS[session], {})
    assert {r["name"] for r in session_payload(session)["players"]} == before


def test_stored_hands_keep_the_original_account_ids(session, tmp_path):
    """Identity is a layer on top of the hands; the hands stay as recorded."""
    from villain.identity import session_questions
    from villain.web import SESSIONS, apply_answers, commit_session
    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, SESSIONS[session]["hands"])
        SESSIONS[session]["questions"] = questions
        alias = [q for q in questions if q.kind == "alias"]
        if not alias:
            pytest.skip("fixture has no same-person candidates")
        original = {s.player_id for h in SESSIONS[session]["hands"] for s in h.seats}
        apply_answers(SESSIONS[session],
                      {alias[0].id: {"same": True, "name": alias[0].names[0]}})
        commit_session(store, session, {})
        stored = {s.player_id for h in store.stored_hands() for s in h.seats}
    assert stored == original, "merging must not rewrite the recorded account ids"


def test_questions_offer_a_name_to_keep(session, tmp_path):
    from villain.identity import session_questions
    from villain.web import SESSIONS
    with Store(tmp_path / "v.db") as store:
        for q in session_questions(store, SESSIONS[session]["hands"]):
            assert q.names, f"{q.id} offers no name choice"
            assert q.default_name in q.names
