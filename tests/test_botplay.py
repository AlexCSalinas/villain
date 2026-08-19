"""The villain policy: measured frequencies actually shape how a bot plays."""

from dataclasses import dataclass

import numpy as np

from villain.botplay import decide, preflop_strength
from villain.holdem import Hand, Seat


@dataclass
class _Est:
    value: float
    opps: float = 500.0


class _Prof:
    def __init__(self, **freqs):
        self.stats = {k: _Est(v) for k, v in freqs.items()}


def _seats(*stacks):
    return [Seat(chr(65 + i), s) for i, s in enumerate(stacks)]


def _preflop_play_rate(profile, n=400, seed=0):
    """How often the bot voluntarily puts chips in from the button, unopened."""
    rng = np.random.default_rng(seed)
    plays = 0
    for _ in range(n):
        h = Hand(_seats(200, 200), button=0, sb=1, bb=2, rng=rng)
        kind = decide(h, 0, profile, rng)[0]     # button acts first preflop, unopened
        plays += kind in ("call", "raise")
    return plays / n


def test_loose_plays_far_more_hands_than_tight():
    tight = _Prof(rfi=0.12)
    loose = _Prof(rfi=0.60)
    assert _preflop_play_rate(loose) > _preflop_play_rate(tight) + 0.25


class _Sized:
    """A profile with a measured open size (means) as well as frequencies."""
    def __init__(self, rfi, open_bb):
        self.stats = {"rfi": _Est(rfi)}
        self.means = {"open_bb": open_bb, "open_bb#n": 100.0}


def test_open_size_follows_the_players_own_sizing():
    def first_open(open_bb, seed=0):
        rng = np.random.default_rng(seed)
        prof = _Sized(0.95, open_bb)          # opens almost everything
        for _ in range(60):
            h = Hand(_seats(400, 400), button=0, sb=1, bb=2, rng=rng)
            kind, amt, _ = decide(h, 0, prof, rng)
            if kind == "raise":
                return amt
        return None
    assert first_open(4.5) > first_open(2.2)   # the big opener opens bigger


def test_a_station_folds_less_to_bets_than_a_nit():
    # Facing a pot bet on the flop with a middling hand, the high-fold profile
    # should fold and the low-fold profile should not, more often than not.
    rng = np.random.default_rng(3)
    nit = _Prof(**{"fold_vs_bet:flop": 0.75})
    station = _Prof(**{"fold_vs_bet:flop": 0.15})
    nit_folds = station_folds = 0
    for k in range(200):
        h = Hand(_seats(200, 200), button=0, sb=1, bb=2, rng=np.random.default_rng(k))
        # get to the flop cheaply, then have the button face a bet
        h.act("call"); h.act("check")            # to the flop
        h.act("raise", h.legal().min_raise_to + int(0.9 * h.pot))  # BB-first bets ~pot
        nit_folds += decide(h, h.to_act, nit, rng)[0] == "fold"
        station_folds += decide(h, h.to_act, station, rng)[0] == "fold"
    assert nit_folds > station_folds


def test_full_bot_hand_stays_legal_and_conserves_chips():
    rng = np.random.default_rng(9)
    profs = [_Prof(vpip=0.4, pfr=0.25, **{"cbet:flop": 0.6, "fold_vs_bet:flop": 0.4})
             for _ in range(3)]
    h = Hand(_seats(200, 200, 200), button=0, sb=1, bb=2, rng=rng)
    total = sum(s.stack for s in h.seats) + h.pot
    guard = 0
    while not h.over:
        kind, amt, _ = decide(h, h.to_act, profs[h.to_act], rng)
        h.act(kind, amt)
        guard += 1
        assert guard < 300
    assert sum(s.stack for s in h.seats) == total


def test_preflop_strength_orders_hands():
    from villain.cards import card_id
    def hole(a, b):
        return (int(card_id(a)), int(card_id(b)))
    assert preflop_strength(hole("Ac", "Ad")) > preflop_strength(hole("Kc", "Kd"))
    assert preflop_strength(hole("Ac", "Kc")) > preflop_strength(hole("Ac", "Kd"))  # suited > offsuit
    assert preflop_strength(hole("Ac", "Ad")) > preflop_strength(hole("7c", "2d"))


def test_opening_is_position_dependent_in_frequency_and_size():
    # A profile that opens tight/small UTG and wide/big on the button. The bot
    # must reproduce both the frequency and the size, per position.
    class _Pos:
        stats = {"rfi:UTG": _Est(0.10), "rfi:BTN": _Est(0.60), "rfi": _Est(0.3)}
        means = {"open_bb:UTG": 2.2, "open_bb:UTG#n": 100.0,
                 "open_bb:BTN": 3.5, "open_bb:BTN#n": 100.0, "open_bb": 2.8, "open_bb#n": 100.0}
    from villain.botplay import _position
    def open_at(pos, n=1200):
        rng = np.random.default_rng(1); opens = tot = 0; sizes = []
        for _ in range(n):
            h = Hand([Seat(str(i), 1000) for i in range(6)], button=0, sb=1, bb=2, rng=rng)
            g = 0
            while not h.over and h.raises == 0 and _position(h, h.to_act) != pos and g < 6:
                h.act("fold"); g += 1
            if h.over or h.raises > 0 or _position(h, h.to_act) != pos:
                continue
            tot += 1
            k, amt, _ = decide(h, h.to_act, _Pos(), rng)
            if k == "raise": opens += 1; sizes.append(amt)
        return (opens / tot if tot else 0), (np.mean(sizes) if sizes else 0)
    uf, us = open_at("UTG"); bf, bs = open_at("BTN")
    assert bf > uf + 0.25          # opens far wider from the button
    assert bs > us                 # ...and to a bigger size
