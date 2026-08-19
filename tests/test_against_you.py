"""Surfacing adjustments: the words, the card and the payload.

The arithmetic is tested in ``test_dynamics``. What matters here is that
nothing reaches a reader without an explanation, that the section fits the
card it has to share, and that a player with no adjustment produces no
section rather than an empty one.
"""

import json

from villain.analyze import as_dict
from villain.dynamics import adjustments
from villain.features import record_hands
from villain.glossary import VERSUS_BEHAVIOUR, stat_help, versus_behaviour
from villain.profile import build_unified
from villain.report import WIDTH, profile_card
from villain.stats import VS_HERO, StatBook

STRONG = {
    "vpip": (126, 420), "pfr": (79, 420), "raise_share": (79, 126),
    "three_bet": (14, 150), "fold_to_three_bet": (33, 55),
    "cbet:flop": (61, 90), "fold_vs_bet:flop": (52, 118),
    "fold_vs_bet:turn": (41, 84), "fold_vs_bet:river": (26, 47),
    "fold_to_cbet:flop": (38, 80), "bb_defend": (40, 96),
    "fold_to_steal": (44, 92), "wtsd": (44, 160), "wsd": (26, 44),
    VS_HERO + "fold_vs_bet:river": (2, 21),
    VS_HERO + "three_bet": (11, 32),
    VS_HERO + "fold_to_steal": (4, 26),
    VS_HERO + "cbet:flop": (17, 22),
}


def profile_with(counts, name="seat 4"):
    book = StatBook(player_id="7", name=name, regime="6max", hands=420)
    for stat, (hits, opps) in counts.items():
        book.ratios[stat].hits, book.ratios[stat].opps = float(hits), float(opps)
    book.meters["table_size"].add(6.0)
    books = {"6max": book}
    profile = build_unified(books)
    profile.adjustments = adjustments(books)
    return profile


# -- the words --------------------------------------------------------------


def test_every_counter_that_has_a_slice_can_be_said_in_words(hands):
    """Whatever the extractor records has to be sayable, or it reaches the
    screen as a raw key. The same guarantee the displayed stats have."""
    books = record_hands(hands)
    families = {
        stat[len(VS_HERO):]
        for by_regime in books.values()
        for book in by_regime.values()
        for stat in book.ratios
        if stat.startswith(VS_HERO)
    }
    assert families, "fixture should exercise the extractor"
    missing = [f for f in families if versus_behaviour(f) == f]
    assert not missing, f"no phrasing for {sorted(missing)}"


def test_every_counter_that_has_a_slice_has_an_explanation(hands):
    books = record_hands(hands)
    families = {
        stat[len(VS_HERO):]
        for by_regime in books.values()
        for book in by_regime.values()
        for stat in book.ratios
        if stat.startswith(VS_HERO)
    }
    missing = [f for f in families if not stat_help(f)]
    assert not missing, f"no glossary entry for {sorted(missing)}"


def test_the_phrasing_names_the_reader():
    """An adjustment is a person reacting to you, not a frequency."""
    for stat in VERSUS_BEHAVIOUR:
        said = versus_behaviour(f"{stat}:river" if "{street}" in
                                VERSUS_BEHAVIOUR[stat] else stat)
        assert "you" in said or "your" in said, said


def test_an_unknown_stat_falls_back_to_its_key():
    assert versus_behaviour("not_a_stat:flop") == "not_a_stat:flop"


# -- the card ---------------------------------------------------------------


def test_the_section_appears_with_what_it_is_measured_against():
    card = profile_card(profile_with(STRONG))
    assert "AGAINST YOU" in card
    assert "Against their own game, not against the field." in card
    assert "folds to your river bets" in card


def test_no_adjustment_means_no_section():
    """Most players have none. A section that appears every time to say so
    costs a line of a card that has to fit on one screen."""
    quiet = {k: v for k, v in STRONG.items() if not k.startswith(VS_HERO)}
    assert "AGAINST YOU" not in profile_card(profile_with(quiet))


def test_verbose_says_so_instead_of_staying_silent():
    quiet = {k: v for k, v in STRONG.items() if not k.startswith(VS_HERO)}
    card = profile_card(profile_with(quiet), verbose=True)
    assert "no adjustment against you clears the bar yet" in card


def test_the_section_fits_the_card():
    card = profile_card(profile_with(STRONG)).splitlines()
    start = next(i for i, line in enumerate(card) if line.startswith("AGAINST YOU"))
    block = card[start:start + 2 + len(profile_with(STRONG).adjustments) + 1]
    for line in block:
        assert len(line) <= WIDTH, f"{len(line)} chars: {line!r}"


def test_certainty_is_not_claimed():
    """The interval comes from a normal approximation, which will happily
    print a confidence it cannot have."""
    card = profile_card(profile_with(STRONG))
    assert "100% sure" not in card


def test_a_profile_built_without_adjustments_still_renders():
    """report only ever reads the attribute; nothing requires it to be set."""
    profile = profile_with(STRONG)
    profile.adjustments = []
    assert profile_card(profile)


# -- the payload ------------------------------------------------------------


def test_the_payload_carries_them_in_words_and_numbers():
    payload = as_dict(profile_with(STRONG))
    assert payload["adjustments"]
    first = payload["adjustments"][0]
    assert first["behaviour"] != first["stat"]
    assert first["direction"] in ("more", "less")
    assert first["sample"] >= 12
    assert 0 <= first["confidence"] <= 1
    assert json.dumps(payload)


def test_the_payload_has_the_key_even_when_empty():
    quiet = {k: v for k, v in STRONG.items() if not k.startswith(VS_HERO)}
    assert as_dict(profile_with(quiet))["adjustments"] == []


# -- wiring -----------------------------------------------------------------


def test_the_store_attaches_them(tmp_path, hands):
    from villain.db import Store

    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        for row in store.players():
            pooled = store.profile(int(row["id"]))
            assert pooled.adjustments == []      # twenty hands clears no floor
            for split in store.profiles(int(row["id"])):
                assert split.adjustments == []
