"""Score the classifier against itself, on hands it has not seen.

The prototypes were tuned twice against a simulation that generated players
*from* the prototypes. In that world the model is correctly specified, so the
harness cannot detect prototype misfit and will always endorse more confidence
than the data supports. Hand-labeled players fail differently: six points
cannot constrain a hundred prototype constants, and once they are inside the
tuning loop they stop being a test.

This is the honest alternative and it needs no labels at all. Split a player's
hands into two disjoint halves, build a profile from each, and ask whether the
posterior computed from one half predicts what the other half actually
supports. The target is observable, the split is interleaved so a player who
drifts across sessions is not scored on the drift, and the scorer is a plain
unweighted likelihood so it can never become another tuned knob.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .archetypes import ARCHETYPES, _log_beta_binomial, match, target_frequency
from .profile import PROFILE_FEATURES, build_profile

#: A half needs this many hands before it is worth scoring at all.
MIN_HALF_HANDS = 50

#: Fixed, and deliberately not the matcher's own value: the scorer must not
#: move when the thing being scored moves.
SCORER_CONCENTRATION = 30.0


@dataclass
class Score:
    players: int
    log_loss: float
    accuracy: float
    calibration_error: float
    mean_confidence: float
    agreement: float

    def __str__(self) -> str:
        return (f"{self.players} players scored on disjoint halves\n"
                f"  log loss                {self.log_loss:.3f}\n"
                f"  top-1 accuracy          {self.accuracy:.3f}\n"
                f"  calibration error       {self.calibration_error:.3f}\n"
                f"  mean stated confidence  {self.mean_confidence:.3f}\n"
                f"  halves agree            {self.agreement:.3f}")


def _halves(store, player_id: int):
    """Two profiles from disjoint halves of a player's actual hands.

    Interleaved rather than cut down the middle, so a player who drifts across
    sessions is not scored on the drift. Splitting the *counts* instead would
    be far cheaper and completely useless: both halves would carry identical
    rates and agree with each other by construction.
    """
    from .features import record_hands
    hands = store.player_hands(player_id)
    if len(hands) < 2 * MIN_HALF_HANDS:
        return None
    key = str(player_id)
    out = []
    for parity in (0, 1):
        books = record_hands(hands[parity::2])
        by_regime = books.get(key)
        if not by_regime:
            return None
        regime, book = max(by_regime.items(), key=lambda kv: kv[1].hands)
        priors = store.fitted_priors(regime) or None
        out.append(build_profile(book, priors=priors))
    return out


def _best_supported(profile) -> str:
    """Which archetype this half's raw counts most plainly support.

    Unweighted and undiscounted on purpose. The target has to be independent of
    every constant being tuned, or the harness scores the tuning against itself.
    """
    best, best_ll = None, -math.inf
    for arch in ARCHETYPES:
        ll = 0.0
        for feature in PROFILE_FEATURES:
            est = profile.stats.get(feature)
            if est is None or not est.opps:
                continue
            target = target_frequency(arch, feature, profile.regime, profile)
            ll += _log_beta_binomial(est.raw * est.opps, est.opps, target,
                                     SCORER_CONCENTRATION)
        if ll > best_ll:
            best, best_ll = arch.name, ll
    return best or "unknown"


def score(store, min_hands: int = 2 * MIN_HALF_HANDS) -> Score | None:
    """Score every player with enough hands to halve."""
    losses, hits, confs, agree = [], [], [], []
    for row in store.players():
        pair = _halves(store, int(row["id"]))
        if pair is None:
            continue
        a, b = pair
        name, conf, mix = match(a)
        target = _best_supported(b)
        share = dict(mix).get(target, 1e-6)
        losses.append(-math.log(max(share, 1e-6)))
        hits.append(1.0 if name == target else 0.0)
        confs.append(conf)
        agree.append(1.0 if name == match(b)[0] else 0.0)
    if not losses:
        return None
    n = len(losses)
    acc = sum(hits) / n
    mean_conf = sum(confs) / n
    return Score(players=n, log_loss=sum(losses) / n, accuracy=acc,
                 calibration_error=abs(mean_conf - acc),
                 mean_confidence=mean_conf, agreement=sum(agree) / n)
