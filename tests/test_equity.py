import pytest

from villain.equity import equities


def test_known_preflop_matchups():
    aa, kk = equities([["Ac", "Ad"], ["Kc", "Kd"]], [])
    assert aa == pytest.approx(0.82, abs=0.02)
    ak, deuces = equities([["Ac", "Kc"], ["2d", "2h"]], [])
    assert ak == pytest.approx(0.50, abs=0.03)


def test_set_over_aces_on_the_flop():
    aces, kings = equities([["Ac", "Ad"], ["Kc", "Kd"]], ["Ks", "7h", "2c"])
    assert kings > 0.88


def test_a_played_board_chops():
    assert equities([["2c", "3d"], ["2h", "3s"]],
                    ["As", "Ks", "Qd", "Jc", "Th"]) == [0.5, 0.5]


def test_river_equity_is_certain():
    win, lose = equities([["Ac", "Ad"], ["Kc", "Kd"]], ["Ah", "7h", "2c", "9d", "4s"])
    assert (win, lose) == (1.0, 0.0)


def test_duplicate_cards_are_rejected():
    with pytest.raises(ValueError):
        equities([["Ac", "Ad"], ["Ac", "Kd"]], [])
