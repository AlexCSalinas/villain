"""What they actually had: a hand-strength model trained on revealed cards.

Frequencies tell you how often somebody bets. They do not tell you what they
bet *with*, and that is the question that decides whether to call. This module
learns the mapping from a line -- street, action, sizing, position, board
texture, time taken -- to the strength of the hand behind it, using the hands
where cards were revealed as labels. A player's own residual against that model
is the read: "when this player bets the river they average 20 percentile points
weaker than the field does" is directly actionable in a way that "they bet the
river 55% of the time" is not.

**The bias, stated plainly.** Villains' cards are only revealed at showdown,
and hands that reach showdown are not a random sample of hands played -- they
skew toward calling lines and away from the bluffs that took the pot down
uncontested. So a model trained purely on villain showdowns *underestimates*
how weak the betting ranges are. Two things reduce it and neither eliminates
it:

* The exporting player's own cards are visible on every hand, including hands
  they folded, so their rows are an unbiased sample and are marked as such.
* Rows are labelled with strength *at the street the action was taken*, not at
  the end, so a flop bet is scored against the flop board rather than against
  a river that had not arrived yet.

Treat the population model as a baseline and the per-player residual as the
signal. Both come with sample counts, and neither is worth anything under a few
hundred rows -- ``fit`` refuses rather than returning a model that looks
authoritative and is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cards import card_ids, evaluate
from .model import Act, Hand, Street
from .stats import HandView

#: Below this many labelled rows, a fitted model is decoration.
MIN_ROWS = 300

#: Pseudo-rows of prior for a per-player residual, shrunk toward zero.
RESIDUAL_PRIOR = 12.0

FEATURES = [
    "street", "is_bet", "is_raise", "is_call", "is_check",
    "bet_fraction", "aggression_level", "has_initiative", "in_position",
    "pot_bb", "think_s", "players_in",
    "board_paired", "board_suited", "board_connected", "board_high",
]


@dataclass
class Row:
    player_id: str
    features: list[float]
    strength: float          # percentile of their hand on that street, 0-1
    unbiased: bool           # cards known regardless of showdown
    street: int
    action: str


@dataclass
class StrengthModel:
    """Population model plus per-player residuals."""

    rows: int = 0
    unbiased_rows: int = 0
    mae: float | None = None
    residuals: dict[str, tuple[float, float]] = field(default_factory=dict)
    _model: object | None = None

    def predict(self, features: list[float]) -> float:
        if self._model is None:
            return 0.5
        return float(np.clip(self._model.predict(np.array(features)[None, :])[0], 0.0, 1.0))

    def offset(self, player_id: str) -> tuple[float, float]:
        """(shrunk residual, rows). Negative means weaker than the field."""
        return self.residuals.get(player_id, (0.0, 0.0))

    def read(self, player_id: str) -> str | None:
        offset, n = self.offset(player_id)
        if n < 6 or abs(offset) < 0.06:
            return None
        direction = "weaker" if offset < 0 else "stronger"
        advice = ("call them down wider" if offset < 0
                  else "give their bets more credit")
        return (f"shows up {abs(offset) * 100:.0f} percentile points {direction} "
                f"than the field on the lines they take ({n:.0f} revealed hands) "
                f"-- {advice}")


class NotEnoughData(ValueError):
    pass


def build_dataset(hands: list[Hand]) -> list[Row]:
    """Every action whose player's cards are known, labelled by strength."""
    rows: list[Row] = []
    for hand in hands:
        if not hand.board:
            continue
        view = HandView(hand)
        showdown = view.showdown()
        known = {s.seat: s for s in hand.seats if len(s.hole_cards) == 2}
        if not known:
            continue
        strengths = _strength_by_street(hand, known)
        for decision in view.decisions():
            seat = known.get(decision.seat)
            if seat is None or decision.street is Street.PREFLOP:
                continue
            if decision.action.act is Act.FOLD:
                continue     # a folded hand has no strength worth predicting
            strength = strengths.get((decision.seat, decision.street))
            if strength is None:
                continue
            act = decision.action.act
            rows.append(Row(
                player_id=seat.player_id,
                features=[
                    float(decision.street),
                    float(act is Act.BET), float(act is Act.RAISE),
                    float(act is Act.CALL), float(act is Act.CHECK),
                    decision.bet_fraction,
                    float(decision.aggression_level),
                    float(decision.has_initiative), float(decision.in_position),
                    decision.action.pot_before / hand.big_blind,
                    min((decision.action.think_ms or 0) / 1000.0, 60.0),
                    float(decision.players_in),
                    *_texture(hand.board_at(decision.street)),
                ],
                strength=strength,
                # Cards visible even when the hand did not go to showdown means
                # this row is not selected on the outcome.
                unbiased=seat.seat not in showdown,
                street=int(decision.street),
                action=act.name.lower(),
            ))
    return rows


def fit(rows: list[Row], random_state: int = 0) -> StrengthModel:
    """Fit the population model and each player's residual against it."""
    if len(rows) < MIN_ROWS:
        raise NotEnoughData(
            f"need {MIN_ROWS} labelled rows to fit a strength model, have {len(rows)}; "
            "keep importing sessions")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_predict

    x = np.array([r.features for r in rows], dtype=float)
    y = np.array([r.strength for r in rows], dtype=float)

    model = GradientBoostingRegressor(
        n_estimators=180, max_depth=3, learning_rate=0.06,
        subsample=0.85, random_state=random_state)
    # Residuals come from out-of-fold predictions: a player's read must not be
    # measured against a model that already memorised their hands.
    out_of_fold = cross_val_predict(model, x, y, cv=min(5, max(2, len(rows) // 60)))
    model.fit(x, y)

    residuals: dict[str, list[float]] = {}
    for row, predicted in zip(rows, out_of_fold):
        residuals.setdefault(row.player_id, []).append(row.strength - predicted)

    fitted = StrengthModel(
        rows=len(rows),
        unbiased_rows=sum(1 for r in rows if r.unbiased),
        mae=float(np.mean(np.abs(y - out_of_fold))),
        residuals={
            pid: (float(np.sum(values) / (len(values) + RESIDUAL_PRIOR)), float(len(values)))
            for pid, values in residuals.items()
        },
    )
    fitted._model = model
    return fitted


def _strength_by_street(hand: Hand, known: dict) -> dict[tuple[int, Street], float]:
    """Percentile of each known hand on each street it was live for.

    Measured against every holding the board allows, because "top pair" means
    something different on a dry board than on a four-flush one.
    """
    out: dict[tuple[int, Street], float] = {}
    for street in (Street.FLOP, Street.TURN, Street.RIVER):
        board = hand.board_at(street)
        if len(board) < 3 or not hand.reached(street):
            continue
        board_ids = card_ids(board).astype(np.int64)
        live = [c for c in range(52) if c not in set(board_ids.tolist())]
        combos = np.array([(a, b) for i, a in enumerate(live) for b in live[i + 1:]],
                          dtype=np.int64)
        seven = np.concatenate(
            [combos, np.repeat(board_ids[None, :], len(combos), axis=0)], axis=1)
        universe = np.sort(evaluate(seven))
        for seat, player in known.items():
            hole = card_ids(player.hole_cards).astype(np.int64)
            if set(hole.tolist()) & set(board_ids.tolist()):
                continue
            score = int(evaluate(np.concatenate([hole, board_ids])[None, :])[0])
            out[(seat, street)] = float(
                np.searchsorted(universe, score, side="left") / len(universe))
    return out


def _texture(board: list[str]) -> tuple[float, float, float, float]:
    """Paired, suited, connected, high -- the four things that change ranges."""
    if len(board) < 3:
        return (0.0, 0.0, 0.0, 0.0)
    ranks = [card_ids([c])[0] // 4 for c in board]
    suits = [card_ids([c])[0] % 4 for c in board]
    paired = float(len(set(ranks)) < len(ranks))
    suited = float(max(suits.count(s) for s in set(suits)) >= 3)
    spread = max(ranks) - min(ranks)
    connected = float(spread <= 4)
    high = float(max(ranks) >= 10)      # queen or better
    return (paired, suited, connected, high)
