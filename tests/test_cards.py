import itertools
import random

import numpy as np
import pytest

from villain.cards import (card_id, card_ids, card_text, describe, evaluate,
                           evaluate_cards)


@pytest.mark.parametrize("cards,expected", [
    (["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"], "straight flush"),
    (["7c", "7d", "7h", "7s", "2c", "2d", "Ac"], "quads"),
    (["Kc", "Kd", "Ks", "9c", "9d", "2h", "3s"], "full house"),
    (["Ac", "Kc", "9c", "5c", "2c", "7d", "8h"], "flush"),
    (["Ac", "Kd", "Qh", "Js", "Tc", "2d", "3c"], "straight"),
    (["Ac", "2d", "3h", "4s", "5c", "9d", "Kc"], "straight"),
    (["9c", "9d", "9h", "5s", "3c", "2d", "Ac"], "trips"),
    (["9c", "9d", "5h", "5s", "3c", "2d", "Ac"], "two pair"),
    (["9c", "9d", "5h", "8s", "3c", "2d", "Ac"], "pair"),
    (["Ac", "Kd", "Qh", "9s", "7c", "5d", "3c"], "high card"),
])
def test_categories(cards, expected):
    assert describe(evaluate_cards(cards)) == expected


def test_wheel_is_the_weakest_straight():
    wheel = evaluate_cards(["Ac", "2d", "3h", "4s", "5c", "9d", "Kc"])
    six_high = evaluate_cards(["6c", "2d", "3h", "4s", "5c", "9d", "Kc"])
    assert wheel < six_high


def test_seven_card_kicker_traps():
    """Quads-plus-a-pair and three-pair both mis-read if kickers come from count order."""
    assert (evaluate_cards(["7c", "7d", "7h", "7s", "2c", "2d", "Ac"])
            > evaluate_cards(["7c", "7d", "7h", "7s", "2c", "2d", "Kc"]))
    assert (evaluate_cards(["9c", "9d", "5h", "5s", "3c", "3d", "Ac"])
            > evaluate_cards(["9c", "9d", "5h", "5s", "3c", "3d", "Kc"]))


def test_matches_brute_force_best_five():
    """The 7-card path must agree with taking the best 5 of the 7."""
    rng = random.Random(11)
    for _ in range(400):
        deal = rng.sample(range(52), 7)
        direct = int(evaluate(np.array(deal, dtype=np.int64)[None, :])[0])
        best = max(int(evaluate(np.array(c, dtype=np.int64)[None, :])[0])
                   for c in itertools.combinations(deal, 5))
        assert direct == best


def test_card_text_roundtrip():
    for cid in range(52):
        assert card_id(card_text(cid)) == cid


def test_evaluate_is_vectorised():
    deals = np.array([[card_id(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2d", "3c"]],
                      [card_id(c) for c in ["2c", "7d", "9h", "Js", "4c", "5d", "8s"]]],
                     dtype=np.int64)
    scores = evaluate(deals)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]
