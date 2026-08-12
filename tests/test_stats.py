"""Statistic definitions, checked against a hand whose answers are known by hand."""

import pytest

from villain.features import record_hand, record_hands
from villain.model import Act, Action, Hand, Seat, Street
from villain.priors import regime
from villain.stats import HandView, Meter, Ratio, StatBook, size_bucket


def build_hand(actions, *, board=None, seats=None, bb=10):
    hand = Hand(hand_id="t1", site="test", table_id="t", started_at=0,
                big_blind=bb, small_blind=bb // 2)
    hand.seats = seats or [
        Seat(seat=1, player_id="a", name="A", stack=100 * bb, position="BTN"),
        Seat(seat=2, player_id="b", name="B", stack=100 * bb, position="BB"),
    ]
    hand.board = board or []
    hand.actions = actions
    return hand


def act(street, seat, kind, amount=0, to_amount=0, pot_before=0, to_call=0):
    return Action(street=street, seat=seat, act=kind, amount=amount,
                  to_amount=to_amount, pot_before=pot_before, to_call=to_call)


def test_ratio_and_meter_merge_is_additive():
    a, b = Ratio(3, 10), Ratio(2, 5)
    a.merge(b)
    assert (a.hits, a.opps, a.rate) == (5, 15, pytest.approx(1 / 3))
    m, n = Meter(), Meter()
    for v in (1.0, 2.0, 3.0):
        m.add(v)
    for v in (4.0, 5.0):
        n.add(v)
    m.merge(n)
    assert m.n == 5 and m.mean == pytest.approx(3.0)


def test_statbook_merge_pools_two_sessions():
    a = StatBook(player_id="x", regime="hu", hands=10)
    a.count("vpip", True)
    b = StatBook(player_id="x", regime="hu", hands=5)
    b.count("vpip", False)
    a.merge(b)
    assert a.hands == 15
    assert a.ratios["vpip"].hits == 1 and a.ratios["vpip"].opps == 2


@pytest.mark.parametrize("fraction,expected", [
    (0.25, "small"), (0.5, "mid"), (0.75, "big"), (1.5, "over")])
def test_size_buckets(fraction, expected):
    assert size_bucket(fraction) == expected


def test_vpip_and_pfr_on_a_known_hand():
    """BTN raises, BB folds: BTN has VPIP and PFR, BB has neither."""
    hand = build_hand([
        act(Street.PREFLOP, 1, Act.POST_SB, 5, 5),
        act(Street.PREFLOP, 2, Act.POST_BB, 10, 10, pot_before=5),
        act(Street.PREFLOP, 1, Act.RAISE, 25, 30, pot_before=15, to_call=5),
        act(Street.PREFLOP, 2, Act.FOLD, pot_before=40, to_call=20),
    ])
    books = {}
    record_hand(hand, books)
    btn = books["a"]["hu"]
    bb = books["b"]["hu"]
    assert btn.rate("vpip") == 1.0 and btn.rate("pfr") == 1.0
    assert bb.rate("vpip") == 0.0
    assert bb.rate("fold_to_steal") == 1.0


def test_cbet_and_fold_to_cbet_denominators():
    """The preflop raiser c-bets; the caller folding counts as fold-to-cbet."""
    hand = build_hand([
        act(Street.PREFLOP, 1, Act.POST_SB, 5, 5),
        act(Street.PREFLOP, 2, Act.POST_BB, 10, 10, pot_before=5),
        act(Street.PREFLOP, 1, Act.RAISE, 25, 30, pot_before=15, to_call=5),
        act(Street.PREFLOP, 2, Act.CALL, 20, 30, pot_before=40, to_call=20),
        act(Street.FLOP, 2, Act.CHECK, pot_before=60),
        act(Street.FLOP, 1, Act.BET, 30, 30, pot_before=60),
        act(Street.FLOP, 2, Act.FOLD, pot_before=90, to_call=30),
    ], board=["2c", "7d", "9h"])
    books = {}
    record_hand(hand, books)
    raiser, caller = books["a"]["hu"], books["b"]["hu"]
    assert raiser.rate("cbet:flop") == 1.0
    assert caller.rate("fold_to_cbet:flop") == 1.0
    assert caller.rate("fold_vs_bet:flop") == 1.0
    # A half-pot bet lands in the "mid" bucket, not "small".
    assert caller.rate("fold_vs_bet:flop:mid") == 1.0


def test_check_raise_requires_a_check_first():
    hand = build_hand([
        act(Street.PREFLOP, 1, Act.POST_SB, 5, 5),
        act(Street.PREFLOP, 2, Act.POST_BB, 10, 10, pot_before=5),
        act(Street.PREFLOP, 1, Act.CALL, 5, 10, pot_before=15, to_call=5),
        act(Street.PREFLOP, 2, Act.CHECK, to_amount=10, pot_before=20),
        act(Street.FLOP, 2, Act.CHECK, pot_before=20),
        act(Street.FLOP, 1, Act.BET, 10, 10, pot_before=20),
        act(Street.FLOP, 2, Act.RAISE, 30, 30, pot_before=30, to_call=10),
    ], board=["2c", "7d", "9h"])
    books = {}
    record_hand(hand, books)
    assert books["b"]["hu"].rate("check_raise:flop") == 1.0
    assert books["a"]["hu"].opps("check_raise:flop") == 0


def test_three_bet_denominator_is_facing_a_raise():
    hand = build_hand([
        act(Street.PREFLOP, 1, Act.POST_SB, 5, 5),
        act(Street.PREFLOP, 2, Act.POST_BB, 10, 10, pot_before=5),
        act(Street.PREFLOP, 1, Act.RAISE, 25, 30, pot_before=15, to_call=5),
        act(Street.PREFLOP, 2, Act.RAISE, 80, 90, pot_before=40, to_call=20),
        act(Street.PREFLOP, 1, Act.FOLD, pot_before=130, to_call=60),
    ])
    books = {}
    record_hand(hand, books)
    assert books["b"]["hu"].rate("three_bet") == 1.0
    assert books["a"]["hu"].rate("fold_to_three_bet") == 1.0
    assert books["a"]["hu"].opps("three_bet") == 0


def test_stats_are_bucketed_by_table_size(hands):
    """The same player three-handed and heads-up keeps two separate books."""
    books = record_hands(hands)
    multi_regime = [pid for pid, by in books.items() if len(by) > 1]
    assert multi_regime, "fixture should contain a player at two table sizes"
    for pid in multi_regime:
        for reg, book in books[pid].items():
            assert book.regime == reg
            assert regime(book.mean("table_size")) == reg


def test_hand_view_tracks_who_saw_each_street(hands):
    for hand in hands:
        view = HandView(hand)
        for seat, street in view.folded_on.items():
            # A player who folded on the flop never saw the turn.
            later = [s for s in Street if s > street]
            for s in later:
                assert seat not in view.saw.get(s, set())


def test_vpip_and_pfr_are_counted_once_per_hand():
    """Regression: a player who limps and then calls a raise put money in once.

    Counting each preflop decision separately inflated both numerator and
    denominator, which made samples look larger than they were and quietly
    raised the confidence attached to every read built on them.
    """
    hand = build_hand([
        act(Street.PREFLOP, 1, Act.POST_SB, 5, 5),
        act(Street.PREFLOP, 2, Act.POST_BB, 10, 10, pot_before=5),
        act(Street.PREFLOP, 1, Act.CALL, 5, 10, pot_before=15, to_call=5),
        act(Street.PREFLOP, 2, Act.RAISE, 30, 40, pot_before=20, to_call=0),
        act(Street.PREFLOP, 1, Act.CALL, 30, 40, pot_before=50, to_call=30),
    ])
    books = {}
    record_hand(hand, books)
    limper = books["a"]["hu"]
    assert limper.opps("vpip") == 1
    assert limper.rate("vpip") == 1.0
    assert limper.opps("pfr") == 1
    assert limper.rate("pfr") == 0.0


def test_a_player_who_never_acts_gets_no_vpip_opportunity():
    """A big blind that everybody folds to never had a decision to make."""
    hand = build_hand([
        act(Street.PREFLOP, 1, Act.POST_SB, 5, 5),
        act(Street.PREFLOP, 2, Act.POST_BB, 10, 10, pot_before=5),
        act(Street.PREFLOP, 1, Act.FOLD, pot_before=15, to_call=5),
    ])
    books = {}
    record_hand(hand, books)
    assert books["b"]["hu"].opps("vpip") == 0
