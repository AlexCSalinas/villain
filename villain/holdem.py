"""A small, correct no-limit Hold'em engine -- enough to get reps in.

One :class:`Hand` runs a single hand as a state machine driven from outside:
it deals, posts blinds, and then exposes whose turn it is (:attr:`to_act`),
what they may legally do (:meth:`legal`), and takes one action at a time
(:meth:`act`). It never decides an action itself -- a UI or a villain policy
does that -- so the same engine drives a human seat and an AI seat identically.

Kept deliberately lean, but correct where correctness is not optional:

* **Side pots.** An all-in for less than a bet builds a side pot the short
  stack cannot win; pots are peeled by contribution level and awarded
  separately (:meth:`_settle`).
* **Min-raise.** A raise is at least the size of the previous bet or raise; an
  all-in shorter than that does not re-open action for players already square.
* **Heads-up blinds.** The button posts the small blind and acts first
  preflop, last after.

Showdown uses :func:`villain.cards.evaluate`, the same evaluator the reads are
built on. Chips are plain integers in whatever unit the caller passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cards import evaluate

STREETS = ("preflop", "flop", "turn", "river")


@dataclass
class Seat:
    name: str
    stack: int
    hole: tuple[int, ...] = ()
    folded: bool = False
    all_in: bool = False
    street_put: int = 0        # chips committed on the current street
    hand_put: int = 0          # chips committed across the whole hand

    @property
    def live(self) -> bool:
        """Still able to act -- in the hand and with chips behind."""
        return not self.folded and not self.all_in


@dataclass
class Legal:
    """What the seat to act may do, and the amounts involved."""

    can_check: bool
    can_call: bool
    call_amount: int           # chips to put in to call (0 if checking)
    can_raise: bool
    min_raise_to: int          # smallest total this-street commitment a raise may reach
    max_raise_to: int          # largest (all-in)
    can_fold: bool


class Hand:
    def __init__(self, seats: list[Seat], button: int, sb: int, bb: int,
                 rng: np.random.Generator | None = None):
        self.seats = seats
        self.n = len(seats)
        self.button = button
        self.sb, self.bb = sb, bb
        self.rng = rng or np.random.default_rng()
        self.board: list[int] = []
        self.street = 0
        self.pot_settled = 0            # chips gathered from earlier streets
        self.log: list[str] = []
        self.winners: dict[int, int] | None = None   # seat -> chips won, when over

        deck = list(range(52))
        self.rng.shuffle(deck)
        self._deck = deck
        for s in seats:
            s.hole = (deck.pop(), deck.pop())

        self._post_blinds()
        self.bet = self.bb                        # highest street commitment so far
        self.min_raise = self.bb                  # size of the last full raise
        self.acted: set[int] = set()              # acted since the last full raise
        self.raises = 0                           # raises this street (an open is 1)
        self.last_raiser: int | None = None       # seat of the most recent aggressor
        self.initiative: int | None = None         # who bet last street -> has initiative now
        self.to_act: int | None = self._first_to_act_preflop()

    # -- setup ---------------------------------------------------------------

    def _post_blinds(self) -> None:
        if self.n == 2:
            sb_seat, bb_seat = self.button, self._next(self.button)
        else:
            sb_seat = self._next(self.button)
            bb_seat = self._next(sb_seat)
        self._commit(sb_seat, self.sb)
        self._commit(bb_seat, self.bb)
        self.log.append(f"{self.seats[sb_seat].name} posts {self.sb}, "
                        f"{self.seats[bb_seat].name} posts {self.bb}")

    def _first_to_act_preflop(self) -> int | None:
        if self.n == 2:
            start = self.button                   # SB/button acts first pre
        else:
            start = self._next(self._next(self._next(self.button)))  # UTG
            start = self._prev_wrap_start(start)
        return self._seek_actor(start)

    def _prev_wrap_start(self, idx: int) -> int:
        # _seek_actor starts *at* idx, so hand back the exact UTG seat.
        return idx

    def _first_to_act_postflop(self) -> int | None:
        start = self.button if self.n == 2 else self.button
        return self._seek_actor(self._next(start))

    # -- geometry ------------------------------------------------------------

    def _next(self, idx: int) -> int:
        return (idx + 1) % self.n

    def _seek_actor(self, start: int) -> int | None:
        """First seat from ``start`` (inclusive, clockwise) that still needs to
        act this street, or ``None`` if the round is closed."""
        for step in range(self.n):
            i = (start + step) % self.n
            if self._needs_to_act(i):
                return i
        return None

    def _needs_to_act(self, i: int) -> bool:
        s = self.seats[i]
        if not s.live:
            return False
        return s.street_put < self.bet or i not in self.acted

    # -- chips ---------------------------------------------------------------

    def _commit(self, i: int, to_total: int) -> None:
        """Move a seat's *street* commitment up to ``to_total`` (capped at all-in)."""
        s = self.seats[i]
        want = min(to_total, s.street_put + s.stack)   # cannot exceed the stack
        add = want - s.street_put
        if add <= 0:
            return
        s.stack -= add
        s.street_put += add
        s.hand_put += add
        if s.stack == 0:
            s.all_in = True

    # -- queries -------------------------------------------------------------

    @property
    def pot(self) -> int:
        return self.pot_settled + sum(s.street_put for s in self.seats)

    @property
    def over(self) -> bool:
        return self.winners is not None

    def _in_hand(self) -> list[int]:
        return [i for i, s in enumerate(self.seats) if not s.folded]

    def legal(self) -> Legal:
        if self.to_act is None:
            raise RuntimeError("no seat to act")
        i = self.to_act
        s = self.seats[i]
        owed = self.bet - s.street_put
        can_check = owed == 0
        call_amount = min(owed, s.stack)
        can_call = owed > 0 and s.stack > 0
        # A raise has to reach at least the current bet plus a full raise, but
        # never more than the seat can put in.
        max_raise_to = s.street_put + s.stack
        min_raise_to = self.bet + self.min_raise
        can_raise = max_raise_to > self.bet and s.stack > owed
        return Legal(
            can_check=can_check, can_call=can_call, call_amount=call_amount,
            can_raise=can_raise, min_raise_to=min(min_raise_to, max_raise_to),
            max_raise_to=max_raise_to, can_fold=owed > 0)

    # -- actions -------------------------------------------------------------

    def act(self, kind: str, amount: int = 0) -> None:
        """Apply the seat-to-act's decision. ``kind`` is ``fold``/``check``/
        ``call``/``raise``; ``amount`` for a raise is the total this-street
        commitment to reach (a raise *to*, not *by*)."""
        if self.to_act is None:
            raise RuntimeError("no seat to act")
        i = self.to_act
        s = self.seats[i]
        legal = self.legal()

        if kind == "fold":
            s.folded = True
            self.log.append(f"{s.name} folds")
        elif kind == "check":
            if not legal.can_check:
                raise ValueError("cannot check facing a bet")
            self.acted.add(i)
            self.log.append(f"{s.name} checks")
        elif kind == "call":
            self._commit(i, self.bet)
            self.acted.add(i)
            self.log.append(f"{s.name} calls")
        elif kind == "raise":
            if not legal.can_raise:
                raise ValueError("cannot raise")
            to = max(min(amount, legal.max_raise_to), 1)
            if to < legal.min_raise_to and to != legal.max_raise_to:
                raise ValueError("raise below the minimum")
            full = to - self.bet >= self.min_raise      # a full (re-opening) raise
            self._commit(i, to)
            if full:
                self.min_raise = self.seats[i].street_put - self.bet
                self.bet = self.seats[i].street_put
                self.acted = {i}                        # everyone else owes action again
            else:
                self.bet = max(self.bet, self.seats[i].street_put)
                self.acted.add(i)
            self.raises += 1
            self.last_raiser = i
            self.log.append(f"{s.name} raises to {self.seats[i].street_put}")
        else:
            raise ValueError(f"unknown action {kind!r}")

        self._advance(i)

    def _advance(self, from_i: int) -> None:
        # One player left un-folded: hand is over, no showdown.
        alive = self._in_hand()
        if len(alive) == 1:
            self._settle()
            return
        nxt = self._seek_actor(self._next(from_i))
        if nxt is not None:
            self.to_act = nxt
            return
        # Betting round closed. Gather the street into the pot and move on.
        self.pot_settled += sum(s.street_put for s in self.seats)
        for s in self.seats:
            s.street_put = 0
        self.acted = set()
        self.bet = 0
        self.min_raise = self.bb
        self.raises = 0
        self.initiative = self.last_raiser       # carried into the next street
        self.last_raiser = None
        # If at most one player can still act, run the board out to showdown.
        if sum(1 for i in alive if self.seats[i].live) <= 1:
            self._runout()
            return
        self._next_street()

    def _next_street(self) -> None:
        if self.street == 3:
            self._settle()
            return
        self.street += 1
        draws = 3 if self.street == 1 else 1
        for _ in range(draws):
            self.board.append(self._deck.pop())
        self.log.append(f"{STREETS[self.street]}: "
                        f"{' '.join(_card(c) for c in self.board)}")
        self.to_act = self._first_to_act_postflop()
        if self.to_act is None:                          # everyone all-in
            self._runout()

    def _runout(self) -> None:
        while len(self.board) < 5:
            self.board.append(self._deck.pop())
        self._settle()

    # -- payouts -------------------------------------------------------------

    def _settle(self) -> None:
        self.pot_settled += sum(s.street_put for s in self.seats)
        for s in self.seats:                             # sweep last street in
            s.street_put = 0
        contribs = {i: s.hand_put for i, s in enumerate(self.seats)}
        winners: dict[int, int] = {i: 0 for i in range(self.n)}

        # Peel side pots by contribution level.
        levels = sorted({c for c in contribs.values() if c > 0})
        prev = 0
        for lvl in levels:
            layer = lvl - prev
            contributors = [i for i, c in contribs.items() if c >= lvl]
            amount = layer * len(contributors)
            contenders = [i for i in contributors if not self.seats[i].folded]
            if not contenders:                           # everyone folded in -- rare
                contenders = contributors
            best = self._best(contenders)
            share, extra = divmod(amount, len(best))
            for j, w in enumerate(best):
                winners[w] += share + (1 if j < extra else 0)
            prev = lvl

        for i, won in winners.items():
            self.seats[i].stack += won
        self.winners = {i: w for i, w in winners.items() if w > 0}
        for i, w in self.winners.items():
            self.log.append(f"{self.seats[i].name} wins {w}")

    def _best(self, seats: list[int]) -> list[int]:
        """The seat(s) with the strongest seven-card hand; ties share."""
        if len(seats) == 1:
            return seats
        board = np.array(self.board, dtype=np.int64)
        scores = {}
        for i in seats:
            seven = np.concatenate([np.array(self.seats[i].hole, dtype=np.int64), board])
            scores[i] = int(evaluate(seven[None, :])[0])
        top = max(scores.values())
        return [i for i in seats if scores[i] == top]


def _card(cid: int) -> str:
    from .cards import card_text
    return card_text(cid)
