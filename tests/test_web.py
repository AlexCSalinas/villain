"""The web layer: uploads that touch nothing, and a save that asks first."""

import copy
import json

import pytest

from villain.db import Store, split_key
from villain.identity import session_questions
from villain.web import MIN_ROSTER_HANDS, SESSIONS, commit_session, parse_upload, profile_payload, roster_payload, session_payload


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
    session_only = [q for q in alias
                    if "database" not in q.left.get("where", "")
                    and "database" not in q.right.get("where", "")]
    assert session_only, "fixture should include a session-only alias candidate"
    assert all(q.default is False for q in session_only), (
        "merging two session accounts must never be the default")
    assert all(not q.auto for q in session_only)


def test_same_name_overlapping_time_is_not_offered_as_one(session, tmp_path):
    """Same name, same months, never at the same table: one person.

    A site that hands out a fresh account id per session makes one regular
    look like dozens of accounts, and their active windows all overlap by
    construction -- they are the same person playing all season. Overlapping
    windows are therefore not evidence of two humans; being dealt into a hand
    together is, and that is tested separately below.
    """
    from villain.web import SESSIONS

    hands = copy.deepcopy(SESSIONS[session]["hands"])
    target_name = "player1"
    a_account = next(
        s.player_id for h in hands for s in h.seats if s.name == target_name)
    alt_account = a_account + "-alt"

    base_ms = 1_800_000_000_000
    for i, hand in enumerate(hands):
        hand.started_at = base_ms + i * 86_400_000  # +i days

    twin = copy.deepcopy(hands)
    for hand in twin:
        for seat in hand.seats:
            if seat.name == target_name:
                seat.player_id = alt_account

    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, hands + twin)

    runs = [q for q in questions if q.id.startswith("reconnects:")]
    covering = [q for q in runs
                if {m["account"] for m in q.members} >= {a_account, alt_account}]
    assert covering, "same-name accounts that never met should be one run"
    assert covering[0].auto, "a reconnect run applies without asking"


def test_one_run_replaces_a_question_per_pair(session, tmp_path):
    """Six accounts under one name is one decision, not fifteen."""
    from villain.web import SESSIONS

    hands = copy.deepcopy(SESSIONS[session]["hands"])
    target = "player1"
    base = next(s.player_id for h in hands for s in h.seats if s.name == target)
    batch = list(hands)
    for n in range(5):
        twin = copy.deepcopy(hands)
        for hand in twin:
            hand.hand_id = f"{hand.hand_id}-copy{n}"
            for seat in hand.seats:
                if seat.name == target:
                    seat.player_id = f"{base}-alt{n}"
        batch += twin

    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, batch)

    runs = [q for q in questions if q.id.startswith("reconnects:")]
    mine = [q for q in runs if any(m["account"] == base for m in q.members)]
    assert len(mine) == 1
    assert len(mine[0].members) == 6
    # The pairwise form would have produced C(6,2) = 15 of these.
    assert not [q for q in questions
                if q.kind == "alias" and not q.id.startswith("reconnects:")
                and {q.left.get("account"), q.right.get("account")}
                <= {base} | {f"{base}-alt{n}" for n in range(5)}]


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
    assert renames[0].auto is True
    assert renames[0].default_name == "player1"
    assert "player1" in renames[0].left["name"]


def test_declining_a_rename_splits_the_identity(tmp_path, hands):
    """Answering 'different people' must keep the two apart from then on."""
    db = tmp_path / "v.db"
    token = "split-token"
    with Store(db) as store:
        store.add_hands(hands)
        later = copy.deepcopy(hands)
        for hand in later:
            hand.hand_id += "-later"
            for seat in hand.seats:
                if seat.name == "player1":
                    seat.name = "someone else"
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
        # Default keeps the database name; hands still pool onto that player.
        commit_session(store, token, {rename.id: {"same": True, "name": rename.default_name}})
        after = {r["display_name"]: r["hands"] for r in store.players()}
    SESSIONS.pop(token, None)
    assert "player1 v2" not in after, "reconnect name must not replace the DB display"
    assert after["player1"] == before["player1"] * 2


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
    for _player, counts in by_player.items():
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


def test_roster_always_names_the_biggest_leak_it_can(tmp_path, hands):
    """"None clears the bar" left the weakest player looking like the safest.

    The column falls back through what is known -- priced leak, unconfirmed
    read, weakest rated area -- and says which kind of claim it is making.
    """
    from villain.web import roster_payload
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        rows = roster_payload(store)
    assert rows
    for row in rows:
        if row["top_leak"] is None:
            continue
        assert row["top_leak_status"] in {"confirmed", "watch", "rated"}
        assert row["top_leak_note"], "an unconfirmed claim must explain itself"
        if row["top_leak_status"] != "confirmed":
            assert "not confirmed" in row["top_leak_note"] \
                or "not a measured frequency" in row["top_leak_note"]


def test_a_rated_weakness_is_never_presented_as_a_measured_leak(tmp_path, hands):
    from villain.web import roster_payload
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        for row in roster_payload(store):
            if row["top_leak_status"] == "confirmed":
                assert row["top_leak_severity"] > 0
            elif row["top_leak_status"] is not None:
                assert row["top_leak_severity"] == 0.0


def test_a_database_merge_shows_up_in_a_loaded_session(tmp_path, hands):
    """The same people, so the same answer. A session that ignored a merge made
    in the database would contradict the database it is about to be saved into.
    """
    import copy

    from villain.web import SESSIONS, database_merges, session_payload
    token = "dbmerge"
    SESSIONS[token] = {"hands": copy.deepcopy(hands), "files": [], "created": 0.0}
    try:
        with Store(tmp_path / "v.db") as store:
            store.add_hands(hands)
            before = {r["name"]: r["hands"] for r in session_payload(token, store)["players"]}
            # Merge two players who never share a hand.
            store.conn.execute(
                "INSERT INTO players (display_name, created_at) VALUES ('Ghost', 0)")
            rows = store.conn.execute(
                "SELECT site, account, player_id FROM aliases").fetchall()
            a, b = rows[0], next(r for r in rows[1:]
                                 if not store.are_distinct(int(rows[0]["player_id"]),
                                                           int(r["player_id"])))
            store.link(int(a["player_id"]), int(b["player_id"]))
            merges = database_merges(store, SESSIONS[token]["hands"])
            after = {r["name"]: r["hands"] for r in session_payload(token, store)["players"]}
    finally:
        SESSIONS.pop(token, None)
    assert merges, "the merged accounts should be pooled in the session"
    assert len(after) < len(before)
    assert sum(after.values()) == sum(before.values()), "no hands lost"


# -- against you ------------------------------------------------------------


def _seeded(store, counts, name="villain"):
    """A player whose books say what we need. Twenty fixture hands clear no
    sample floor, and the point here is the wiring, not the arithmetic."""
    from villain.stats import StatBook

    player_id = int(store.players()[0]["id"])
    book = StatBook(player_id=str(player_id), name=name, regime="6max", hands=420)
    for stat, (hits, opps) in counts.items():
        book.ratios[stat].hits, book.ratios[stat].opps = float(hits), float(opps)
    book.meters["table_size"].add(6.0)
    store._write_books(player_id, {"6max": book}, name)
    store.conn.commit()
    return player_id


ADJUSTED = {
    "vpip": (126, 420), "pfr": (79, 420), "fold_vs_bet:river": (26, 47),
    "three_bet": (14, 150), "wtsd": (44, 160), "wsd": (26, 44),
    "vs:fold_vs_bet:river": (2, 21), "vs:three_bet": (11, 32),
}


def test_the_payload_points_evidence_at_the_slice(tmp_path, hands):
    """Not at the pooled counter: the hands shown have to be the hands the
    read is about."""
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        player_id = _seeded(store, ADJUSTED)
        payload = profile_payload(store.profile(player_id), player_id)
    assert payload["adjustments"]
    for a in payload["adjustments"]:
        assert a["evidence_stat"] == "vs:" + a["stat"]
        assert a["behavior"] != a["stat"]


def test_an_uploaded_session_shows_them_too(session):
    """A preview that drops a section the same hands produce once saved is a
    preview of something else."""
    payload = session_payload(session)
    for profile in payload["profiles"]:
        assert "adjustments" in profile


def test_auto_merges_do_not_swallow_the_questions_that_need_a_human(session, tmp_path):
    """Applying the runs must not mark the session answered.

    The UI opens the identity dialog only for an unanswered session. Auto
    answers share the dict human answers live in, so counting them as "the
    user has answered" hid every remaining question behind merges the user
    never saw.
    """
    from villain.identity import auto_answers, session_questions
    from villain.web import SESSIONS, apply_answers, session_payload

    with Store(tmp_path / "v.db") as store:
        questions = session_questions(store, SESSIONS[session]["hands"])
        SESSIONS[session]["questions"] = questions
        auto = auto_answers(questions)
        if auto:
            apply_answers(SESSIONS[session], auto)
        payload = session_payload(session, store)

    if auto:
        assert not payload["answered"], "auto merges are not a human answer"
    # And a real answer does flip it.
    human = [q for q in questions if not q.auto]
    if human:
        with Store(tmp_path / "v.db") as store:
            apply_answers(SESSIONS[session], {human[0].id: {"same": False}})
            payload = session_payload(session, store)
        assert payload["answered"]


def test_a_read_that_only_holds_at_one_table_size_survives(tmp_path):
    """Translating every table size into the home one can cancel a real read.

    One player folds to turn bets 31% of the time heads-up against the hero
    and 54% against everybody else, while six-handed the same gap runs the
    other way. Averaged they cancel and nothing is reported, which is how the
    strongest against-you read in the database went missing.
    """
    from villain.dynamics import REGIME_MIN_OPPS, adjustments
    from villain.stats import VS_HERO, Ratio, StatBook

    def book(regime, vs_hits, vs_opps, other_hits, other_opps):
        b = StatBook(player_id="1", name="x", regime=regime, hands=int(other_opps))
        b.ratios["fold_vs_bet:turn"] = Ratio(hits=vs_hits + other_hits,
                                             opps=vs_opps + other_opps)
        b.ratios[VS_HERO + "fold_vs_bet:turn"] = Ratio(hits=vs_hits, opps=vs_opps)
        return b

    n = REGIME_MIN_OPPS * 3
    by_regime = {
        # Heads-up: folds to the hero far less than to anyone else.
        "hu": book("hu", vs_hits=0.22 * n, vs_opps=n, other_hits=0.52 * n, other_opps=n),
        # Six-handed: the opposite, and enough of it to cancel the average.
        "6max": book("6max", vs_hits=0.56 * n, vs_opps=n, other_hits=0.46 * n, other_opps=n),
    }
    found = adjustments(by_regime)
    regimes = {a.regime for a in found if a.stat == "fold_vs_bet:turn"}
    assert "hu" in regimes, "the heads-up read must survive being averaged away"
    hu = next(a for a in found if a.regime == "hu")
    assert hu.gap < 0, "heads-up he folds less, not more"


def test_the_against_you_read_is_detail_only(tmp_path):
    """The roster is how everybody plays the field; one reference per list.

    Mixing "how they play the table" with "how they play you" in one column is
    how the field read stopped meaning anything in the first place.
    """
    from tests.conftest import FIXTURE
    from villain.analyze import as_dict
    from villain.parsers import parse_file
    from villain.web import roster_payload

    with Store(tmp_path / "v.db") as store:
        store.add_hands(parse_file(FIXTURE))
        rows = roster_payload(store)
        assert rows, "need at least one player to check"
        assert all("versus" not in row for row in rows)
        detail = as_dict(store.profile(rows[0]["player_id"]))
    assert "versus" in detail, "the detailed view carries it"


def test_the_against_you_archetype_is_taken_at_one_table_size():
    """Pooling is what hid the read: it must name a regime, never blend them."""
    from villain.dynamics import MIN_VERSUS_DECISIONS, versus_read
    from villain.stats import VS_HERO, Ratio, StatBook

    def book(regime, rate, n):
        b = StatBook(player_id="1", name="x", regime=regime, hands=n)
        for stat in ("fold_vs_bet:flop", "fold_vs_bet:turn", "vpip", "pfr"):
            b.ratios[VS_HERO + stat] = Ratio(hits=rate * n, opps=n)
            b.ratios[stat] = Ratio(hits=rate * n, opps=n)
        return b

    n = MIN_VERSUS_DECISIONS          # each book carries 4 counters of n
    read = versus_read({"hu": book("hu", 0.2, n), "6max": book("6max", 0.5, n // 4)})
    assert read is not None
    assert read.regime == "hu", "the table size with the most shared history"
    assert read.regime_label == "heads-up"


def test_no_against_you_read_without_enough_shared_history():
    from villain.dynamics import versus_read
    from villain.stats import VS_HERO, Ratio, StatBook

    b = StatBook(player_id="1", name="x", regime="6max", hands=20)
    b.ratios[VS_HERO + "vpip"] = Ratio(hits=5, opps=20)
    b.ratios["vpip"] = Ratio(hits=5, opps=20)
    assert versus_read({"6max": b}) is None
