"""The parser is the foundation: if it mis-decodes an opcode every statistic
downstream is quietly wrong, so these tests check the money, not just the shape."""

import pytest

from villain.model import Act, Street, hand_from_dict, hand_to_dict, positions_for


def test_every_hand_balances(hands):
    """Chips in must equal chips out. This is what proves the opcodes are right."""
    for hand in hands:
        invested = sum(s.invested for s in hand.seats)
        won = sum(s.won for s in hand.seats)
        assert invested == won, f"hand {hand.hand_id} does not balance"


def test_no_unknown_opcodes(hands):
    for hand in hands:
        assert not [f for f in hand.flags if f.startswith("unknown_event")]


def test_blinds_and_positions(hands):
    for hand in hands:
        posts = [a for a in hand.actions if a.act in (Act.POST_SB, Act.POST_BB)]
        assert len(posts) >= 2
        sb = next(a for a in posts if a.act is Act.POST_SB)
        assert sb.amount == hand.small_blind


def test_heads_up_button_posts_the_small_blind(hands):
    for hand in hands:
        if len(hand.seats) != 2:
            continue
        sb = next(a for a in hand.actions if a.act is Act.POST_SB)
        assert hand.seat(sb.seat).position == "BTN"


def test_amounts_are_increments_and_totals(hands):
    """``to_amount`` is cumulative for the street, ``amount`` is what was added."""
    for hand in hands:
        running: dict[tuple[int, int], int] = {}
        for action in hand.actions:
            key = (int(action.street), action.seat)
            before = running.get(key, 0)
            assert action.to_amount == before + action.amount
            running[key] = action.to_amount


def test_raise_versus_bet_classification(hands):
    """Preflop opens are raises (blinds are already wagered); flop leads are bets."""
    for hand in hands:
        for action in hand.actions:
            if action.act is Act.BET:
                assert action.street is not Street.PREFLOP
            if action.act is Act.BET:
                assert action.to_call == 0


def test_single_card_reveals_are_kept_out_of_hole_cards(hands):
    partial = [s for h in hands for s in h.seats if len(s.revealed) == 1]
    assert partial, "fixture should contain a one-card reveal"
    for seat in partial:
        assert seat.hole_cards == () or len(seat.hole_cards) == 2
        assert all(c is not None for c in seat.revealed)


def test_board_cards_are_well_formed(hands):
    for hand in hands:
        assert len(hand.board) in (0, 3, 4, 5)
        for card in hand.board:
            assert len(card) == 2


def test_serialisation_roundtrip(hands):
    for hand in hands:
        assert hand_to_dict(hand_from_dict(hand_to_dict(hand))) == hand_to_dict(hand)


@pytest.mark.parametrize("seats,dealer,expected", [
    ([1, 2], 1, {1: "BTN", 2: "BB"}),
    ([1, 2, 3], 1, {1: "BTN", 2: "SB", 3: "BB"}),
    ([1, 2, 3, 4, 5, 6], 6, {6: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}),
])
def test_position_assignment(seats, dealer, expected):
    assert positions_for(seats, dealer) == expected


def test_dead_button_falls_forward():
    """An empty dealer seat moves the button forward, keeping positions contiguous.

    Real rooms leave the button on the dead seat and kill the small blind; this
    approximates it by advancing to the next occupied seat, which keeps every
    positional statistic well defined.
    """
    assert positions_for([2, 4, 6], 3) == {4: "BTN", 6: "SB", 2: "BB"}
