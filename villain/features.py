"""Turning hands into the statistics a profile is built from.

Every counter below names its denominator, because that is where trackers
disagree. "3-bet 8%" means *of the times this player faced a single raise*, not
of hands dealt.

Three families of signal are collected:

* **Frequencies** -- the tracker stats, split by street and by the size of the
  bet faced. Size-split fold frequencies are where the money is: a player who
  folds 42% to a half-pot bet and 71% to a pot-sized bet is two different
  opponents depending on what you do.
* **Sizings and timing** -- how big they bet and how long they took. Timing is
  free information that no tracker in a home game is hiding, and it separates
  players whose bet sizes are hand-dependent from those who use one size.
* **Showdown truth** -- what they actually held. Rare (a villain shows maybe
  one hand in eight) but it is the only unbiased look at the range behind their
  frequencies, so it anchors the hand-strength model in :mod:`villain.reads`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .cards import card_ids, evaluate
from .equity import equities
from .model import Act, Hand, Street
from .priors import regime as regime_of
from .stats import HandView, StatBook, size_bucket

#: player id -> table-size regime -> book
Books = dict[str, dict[str, StatBook]]

TANK_MS = 8_000        # a fold this slow was a real decision, not a formality
SNAP_MS = 1_200        # faster than a considered action
BLUFF_PCTILE = 0.35    # showdown strength below this, in a bet, was a bluff


def record_hands(hands: Iterable[Hand], books: Books | None = None) -> Books:
    books = books if books is not None else {}
    for hand in hands:
        record_hand(hand, books)
    return books


def record_hand(hand: Hand, books: Books) -> None:
    """Fold one hand into every participating player's book for this regime."""
    if "pot_mismatch" in hand.flags or hand.big_blind <= 0:
        return
    view = HandView(hand)
    reg = regime_of(len(hand.seats))

    for seat in hand.seats:
        book = book_for(books, seat.player_id, reg, seat.name)
        book.name = seat.name or book.name
        book.hands += 1
        book.first_seen = min(book.first_seen or hand.started_at, hand.started_at)
        book.last_seen = max(book.last_seen or hand.started_at, hand.started_at)
        book.count(f"seat:{seat.position}", True)
        book.measure("table_size", len(hand.seats))
        book.measure("stack_bb", seat.stack / hand.big_blind)

    _preflop(hand, view, books, reg)
    _postflop(hand, view, books, reg)
    _results(hand, view, books, reg)


def book_for(books: Books, player_id: str, reg: str, name: str = "") -> StatBook:
    by_regime = books.setdefault(player_id, {})
    book = by_regime.get(reg)
    if book is None:
        book = by_regime[reg] = StatBook(player_id=player_id, name=name, regime=reg)
    return book


def _book(hand: Hand, books: Books, seat: int, reg: str) -> StatBook:
    s = hand.seat(seat)
    return book_for(books, s.player_id, reg, s.name)


# ---------------------------------------------------------------------------
# preflop
# ---------------------------------------------------------------------------

def _preflop(hand: Hand, view: HandView, books: Books, reg: str) -> None:
    opener: int | None = None
    three_bettor: int | None = None
    open_size = 0
    voluntary: set[int] = set()
    limpers: set[int] = set()
    cold_callers = 0
    bb = hand.big_blind
    # VPIP and PFR are per *hand*, not per decision. A player who limps and
    # then calls a raise put money in once; counting both decisions inflates
    # the denominator and makes the sample look larger than it is.
    entered: dict[int, dict[str, bool]] = {}

    for d in view.decisions():
        if d.street is not Street.PREFLOP:
            break
        a = d.action
        book = _book(hand, books, d.seat, reg)
        pos = hand.seat(d.seat).position
        raised = a.act is Act.RAISE
        called = a.act is Act.CALL
        folded = a.act is Act.FOLD

        # Recorded once per hand, after the street is over.
        seen = entered.setdefault(d.seat, {"vpip": False, "pfr": False})
        seen["vpip"] = seen["vpip"] or raised or called
        seen["pfr"] = seen["pfr"] or raised

        if d.aggression_level == 0 and not voluntary:
            # First in: nobody has voluntarily put money in yet.
            book.count("rfi", raised)
            book.count(f"rfi:{pos}", raised)
            book.count("limp", called)
            if pos in ("CO", "BTN", "SB"):
                book.count("steal", raised)
            if raised:
                book.measure("open_bb", a.to_amount / bb)
                book.measure(f"open_bb:{pos}", a.to_amount / bb)

        elif d.aggression_level == 1 and d.seat != opener:
            # Facing a single raise.
            book.count("three_bet", raised)
            if pos == "BB":
                book.count("bb_defend", raised or called)
                book.count("bb_fold_to_open", folded)
            if d.seat not in voluntary:
                book.count("cold_call", called)
            if cold_callers and d.seat not in voluntary:
                book.count("squeeze", raised)
            if opener is not None and hand.seat(opener).position in ("CO", "BTN", "SB") \
                    and pos in ("SB", "BB"):
                book.count("fold_to_steal", folded)
                book.count("three_bet_vs_steal", raised)
            if d.seat in limpers:
                book.count("limp_fold", folded)
                book.count("limp_raise", raised)
            if raised and open_size:
                book.measure("three_bet_ratio", a.to_amount / open_size)

        elif d.aggression_level == 2:
            if d.seat == opener:
                book.count("fold_to_three_bet", folded)
                book.count("four_bet", raised)
            else:
                book.count("cold_four_bet", raised)

        elif d.aggression_level >= 3 and d.seat == three_bettor:
            book.count("fold_to_four_bet", folded)
            book.count("five_bet", raised)

        _timing(book, d, "pf")

        if raised:
            if opener is None:
                opener, open_size = d.seat, a.to_amount
            elif three_bettor is None:
                three_bettor = d.seat
        if called and d.aggression_level >= 1 and d.seat not in voluntary:
            cold_callers += 1
        if called and d.aggression_level == 0:
            limpers.add(d.seat)
        if raised or called:
            voluntary.add(d.seat)

    for seat, seen in entered.items():
        book = _book(hand, books, seat, reg)
        book.count("vpip", seen["vpip"])
        book.count("pfr", seen["pfr"])


# ---------------------------------------------------------------------------
# postflop
# ---------------------------------------------------------------------------

def _postflop(hand: Hand, view: HandView, books: Books, reg: str) -> None:
    street = None
    first_bettor: int | None = None
    bettor_had_initiative = False
    checked: set[int] = set()
    declined_initiative: set[int] = set()   # aggressors who checked on an earlier street
    faced_bet_size: dict[int, float] = {}

    for d in view.decisions():
        if d.street is Street.PREFLOP:
            continue
        if d.street is not street:
            street = d.street
            first_bettor, bettor_had_initiative = None, False
            checked, faced_bet_size = set(), {}
            for seat in view.saw[street]:
                _book(hand, books, seat, reg).count(f"saw:{street.label}", True)

        a = d.action
        book = _book(hand, books, d.seat, reg)
        s = street.label
        raised = a.act is Act.RAISE
        bet = a.act is Act.BET
        called = a.act is Act.CALL
        folded = a.act is Act.FOLD
        checkd = a.act is Act.CHECK
        initiative = view.initiative_at(street)

        # Pot sizes are recorded so the exploit layer can price a leak in big
        # blinds instead of reporting an abstract severity score.
        book.measure(f"pot_bb:{s}", a.pot_before / hand.big_blind)

        # Raw action mix, for aggression frequency.
        for label, hit in (("bet", bet), ("raise", raised), ("call", called),
                           ("fold", folded), ("check", checkd)):
            book.count(f"act:{s}:{label}", hit)

        if not d.facing_bet:
            if d.has_initiative:
                # Continuation bet: they took the betting lead last street and
                # the action is on them with nothing wagered yet.
                book.count(f"cbet:{s}", bet)
                if bet:
                    book.measure(f"cbet_size:{s}", d.bet_fraction)
                else:
                    declined_initiative.add(d.seat)
            elif initiative is not None and initiative not in declined_initiative:
                # Betting into the player who holds the lead.
                book.count(f"donk:{s}", bet)
            else:
                # Nobody claimed the lead -- a probe or a stab at a dead pot.
                book.count(f"probe:{s}", bet)
            if bet:
                book.measure(f"bet_size:{s}", d.bet_fraction)
                book.count(f"overbet:{s}", d.bet_fraction > 1.0)
        else:
            frac = faced_bet_size.get(d.seat, d.bet_fraction)
            bucket = size_bucket(frac)
            book.count(f"fold_vs_bet:{s}", folded)
            book.count(f"fold_vs_bet:{s}:{bucket}", folded)
            book.count(f"raise_vs_bet:{s}", raised)
            book.count(f"call_vs_bet:{s}", called)
            if first_bettor is not None and bettor_had_initiative:
                book.count(f"fold_to_cbet:{s}", folded)
                book.count(f"raise_cbet:{s}", raised)
                book.count(f"call_cbet:{s}", called)
            if d.seat in checked:
                book.count(f"check_raise:{s}", raised)
                book.count(f"check_fold:{s}", folded)
            if raised:
                book.measure(f"raise_ratio:{s}", a.to_amount / max(a.to_call, 1))

        if a.all_in:
            book.count("all_in_action", True)

        _timing(book, d, s)

        if checkd:
            checked.add(d.seat)
        if bet or raised:
            if first_bettor is None:
                first_bettor = d.seat
                bettor_had_initiative = d.has_initiative
            for seat in view.saw[street]:
                if seat != d.seat:
                    faced_bet_size[seat] = d.bet_fraction


def _timing(book: StatBook, d, street_label: str) -> None:
    """Record how long the decision took. Timing is uncontaminated read: nobody
    in a casual game is balancing their clock."""
    ms = d.action.think_ms
    if ms is None or ms < 0 or ms > 120_000:
        return
    act = d.action.act
    kind = ("fold" if act is Act.FOLD else "call" if act is Act.CALL
            else "check" if act is Act.CHECK else "aggro")
    book.measure(f"think:{kind}", ms)
    book.measure(f"think:{street_label}", ms)
    if act is Act.FOLD:
        book.count("tank_fold", ms > TANK_MS)
    if act is Act.CALL:
        book.count("snap_call", ms < SNAP_MS)
    if act.is_aggressive:
        book.count("snap_aggro", ms < SNAP_MS)


# ---------------------------------------------------------------------------
# results and showdown truth
# ---------------------------------------------------------------------------

def _results(hand: Hand, view: HandView, books: Books, reg: str) -> None:
    bb = hand.big_blind
    showdown = view.showdown()
    complete_board = len(hand.board) >= 5

    for seat in hand.seats:
        book = book_for(books, seat.player_id, reg, seat.name)
        net_bb = seat.net / bb
        book.measure("net_bb", net_bb)
        saw_flop = seat.seat in view.saw.get(Street.FLOP, set()) and hand.reached(Street.FLOP)
        if saw_flop:
            book.count("wwsf", seat.net > 0)
            book.count("wtsd", seat.seat in showdown)
        if seat.seat in showdown:
            book.count("wsd", seat.net > 0)
            book.measure("sd_net_bb", net_bb)
        else:
            book.measure("nonsd_net_bb", net_bb)
        if seat.seat in view.folded_on:
            book.measure("fold_street", float(view.folded_on[seat.seat]))
        if seat.showed:
            # Showing a single card is a distinctive habit -- usually a bluff
            # being advertised, occasionally a slow-roll.
            book.count("shows_one_card", len(seat.revealed) == 1)

    _all_in_ev(hand, view, books, reg, showdown)

    if not complete_board:
        return

    known = {s.seat: s.hole_cards for s in hand.seats
             if len(s.hole_cards) == 2 and s.seat in showdown}
    if not known:
        return
    strengths = _showdown_strengths(hand.board, known)
    aggressors = {view.aggressor[st] for st in Street if view.aggressor[st] is not None}

    for seat, pct in strengths.items():
        book = _book(hand, books, seat, reg)
        book.measure("sd_strength", pct)
        if seat in aggressors and view.aggressor[Street.RIVER] == seat:
            book.count("river_bet_bluff", pct < BLUFF_PCTILE)
            book.measure("river_bet_strength", pct)
        elif seat not in aggressors:
            book.count("sd_light_call", pct < 0.5)
            book.measure("sd_call_strength", pct)


def _all_in_ev(hand: Hand, view: HandView, books: Books, reg: str,
               showdown: set[int]) -> None:
    """Score all-in pots by equity as well as by outcome.

    Over a few hundred hands a chip graph is mostly variance: getting it in as
    an 80% favourite and losing costs exactly as much as punting, and a rating
    that cannot tell those apart is rating luck. So when the money goes in with
    cards face up, the pot is also credited by equity at that moment.

    Side pots are not modelled -- with three or more all-in players of
    different stacks the equity share is approximate, so those hands are
    flagged rather than trusted.
    """
    all_in_actions = [a for a in hand.actions if a.all_in]
    if not all_in_actions or len(showdown) < 2:
        return
    known = {s.seat: list(s.hole_cards) for s in hand.seats
             if s.seat in showdown and len(s.hole_cards) == 2}
    if len(known) != len(showdown):
        return

    street = max(a.street for a in all_in_actions)
    board = hand.board_at(street)
    try:
        shares = equities(list(known.values()), board)
    except ValueError:
        return
    pot = sum(s.invested for s in hand.seats)
    for (seat, _), share in zip(known.items(), shares):
        book = book_for(books, hand.seat(seat).player_id, reg)
        player = hand.seat(seat)
        book.measure("ev_net_bb", (share * pot - player.invested) / hand.big_blind)
        # The realised result of the same pots, so a rating can swap one for
        # the other instead of counting the all-in twice.
        book.measure("allin_realised_bb", player.net / hand.big_blind)
        book.measure("allin_equity", share)
        book.count("all_in_pot", True)


def _showdown_strengths(board: list[str], known: dict[int, tuple[str, ...]]) -> dict[int, float]:
    """Percentile of each shown hand among every holding possible on that board.

    Reporting "two pair" says nothing without the board -- two pair on a paired
    four-flush board is a bluff-catcher. The percentile is against the full set
    of combos the board allows, which is what actually decides whether a
    showdown was strong.
    """
    board5 = board[:5]
    board_ids = card_ids(board5).astype(np.int64)
    dead = set(board_ids.tolist())
    for cards in known.values():
        dead |= set(card_ids(cards).astype(np.int64).tolist())

    live = [c for c in range(52) if c not in set(board_ids.tolist())]
    combos = np.array([(a, b) for i, a in enumerate(live) for b in live[i + 1:]], dtype=np.int64)
    seven = np.concatenate([combos, np.repeat(board_ids[None, :], len(combos), axis=0)], axis=1)
    universe = np.sort(evaluate(seven))

    out: dict[int, float] = {}
    for seat, cards in known.items():
        score = int(evaluate(np.concatenate([card_ids(cards).astype(np.int64), board_ids])[None, :])[0])
        out[seat] = float(np.searchsorted(universe, score, side="left") / len(universe))
    return out
