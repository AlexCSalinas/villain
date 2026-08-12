"""Exploitative buckets.

An archetype here is not a personality, it is a *plan*. "Station" is the bucket
whose plan is "value bet thinner and stop bluffing"; "nit" is the bucket whose
plan is "open every button and believe their raises". So each prototype is
defined by the frequencies that determine that plan.

Prototypes are stored as **deviations from the population in log-odds**, not as
absolute frequencies. Two reasons, and the second is the one that bites.

First, table size: 55% VPIP is a nit heads-up, a normal three-handed player and
a maniac at a full ring. Storing "vpip: +1.2 spreads above the field" means one
prototype set works everywhere, measured against each table size's own
population.

Second, frequencies live on a bounded scale and do not shift linearly. Adding
the same number of percentage points to a 70% base and a 24% base is not the
same size of change, and doing it in linear space produces a six-handed "nit"
who plays 2% of hands and a heads-up "maniac" pinned at 97%. In log-odds the
same deviation lands on 44% and 9.5% respectively -- a heads-up nit and a
full-ring nit, which is what the prototype was always supposed to mean.

Matching is done by likelihood, not by distance to the shrunk numbers. The
difference matters: shrinking a stat toward the population and *then* measuring
its distance from a prototype counts the uncertainty twice, and every thin
sample collapses onto whichever prototype sits in the middle. Instead each
archetype implies a frequency for each feature, and the raw counts are scored
against it with a Beta-Binomial likelihood -- three observations move the
posterior a little, three hundred move it a lot, with no separate confidence
fudge factor needed.

Every archetype is scored over the *same* features, which is what makes the
comparison a comparison. A prototype that says nothing about check-raising is
not abstaining -- it is predicting the population frequency, and it takes the
same penalty as anyone else when the player check-raises three times as often.
Scoring each prototype over only the features it happens to mention would hand
the win to whichever one mentioned the fewest.

The likelihood is deliberately overdispersed (``CONCENTRATION``): an archetype
is a region of strategy space, not a point, and a station who folds to 26% of
turn bets rather than the prototype's 22% is still a station. Feature
importances are shared across archetypes for the same reason the feature set
is, and a global discount accounts for these features being correlated -- VPIP
and PFR are not independent measurements, and treating them as such would
manufacture certainty.

Clustering a database of four home game players discovers nothing, so
prototypes are the default; they work from the first hand and degrade
gracefully. Once the database holds enough players, :func:`fit_clusters` learns
the groupings actually present -- but the named plan still comes from the
prototypes, because a cluster id is not a strategy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .priors import population_mean
from .profile import PROFILE_FEATURES, Profile

#: Between-player spread of each stat in log-odds, the unit deviations are
#: measured in. Wider means the population genuinely disagrees about that
#: statistic -- everyone limps at roughly no rate at all until somebody limps
#: constantly, so ``limp`` spreads far more than ``wsd`` does.
SPREAD = {
    "vpip": 0.50, "pfr": 0.55, "three_bet": 0.55, "fold_to_three_bet": 0.45,
    "cbet:flop": 0.50, "cbet:turn": 0.50, "cbet:river": 0.52,
    "fold_to_cbet:flop": 0.45, "fold_to_cbet:turn": 0.48,
    "fold_vs_bet:flop": 0.45, "fold_vs_bet:turn": 0.48, "fold_vs_bet:river": 0.50,
    "check_raise:flop": 0.70, "donk:flop": 0.70,
    "wwsf": 0.32, "wtsd": 0.42, "wsd": 0.32,
    "aggression:flop": 0.50, "aggression:turn": 0.52, "aggression:river": 0.52,
    "limp": 1.00, "bb_defend": 0.45,
}
DEFAULT_SPREAD = 0.50

#: Frequencies are clamped away from certainty before taking log-odds; a
#: measured 0% is "rarely", not "never".
EPSILON = 0.005


def _logit(p: float) -> float:
    p = min(1.0 - EPSILON, max(EPSILON, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def spread_of(feature: str, table_regime: str = "") -> float:
    """Between-player spread of a stat, in log-odds."""
    return SPREAD.get(feature, DEFAULT_SPREAD)


#: How much each feature counts toward identifying a plan. Shared by every
#: archetype: these are exponents in a likelihood, so varying them per
#: archetype would make the scores incomparable.
IMPORTANCE = {
    "vpip": 1.4, "pfr": 1.3, "three_bet": 1.1, "fold_to_three_bet": 0.9,
    "fold_vs_bet:flop": 1.4, "fold_vs_bet:turn": 1.6, "fold_vs_bet:river": 1.3,
    "fold_to_cbet:flop": 1.2, "fold_to_cbet:turn": 1.0,
    "aggression:flop": 1.4, "aggression:turn": 1.4, "aggression:river": 1.2,
    "cbet:flop": 1.1, "cbet:turn": 1.0, "cbet:river": 0.8,
    "check_raise:flop": 1.2, "donk:flop": 0.8,
    "wtsd": 1.5, "wsd": 1.0, "wwsf": 0.8, "limp": 1.5, "bb_defend": 1.0,
}
DEFAULT_IMPORTANCE = 1.0


@dataclass(frozen=True)
class Archetype:
    name: str
    summary: str
    plan: str
    traits: dict[str, float]        # feature -> deviation from population, in spreads

    def deviation(self, feature: str) -> float:
        """Unmentioned features are a prediction of population-average play."""
        return self.traits.get(feature, 0.0)


ARCHETYPES: list[Archetype] = [
    Archetype(
        "nit",
        "Far tighter than the table demands and folds rather than defends.",
        "Open every pot and steal relentlessly; they surrender their blinds. "
        "When they finally raise, believe it and fold anything marginal.",
        {"vpip": -2.2, "pfr": -1.8, "three_bet": -1.2, "fold_to_three_bet": +1.5,
         "fold_vs_bet:flop": +1.2, "fold_vs_bet:turn": +1.2, "wtsd": -1.2,
         "aggression:flop": -1.0, "bb_defend": -1.5},
    ),
    Archetype(
        "station",
        "Calls far too much and folds far too little; will not release a pair.",
        "Value bet thin and relentlessly -- three streets with any made hand, "
        "and size up, they are paying regardless. Never bluff them.",
        {"vpip": +1.2, "pfr": -1.2, "three_bet": -1.0,
         "fold_to_cbet:flop": -2.0, "fold_vs_bet:flop": -2.0, "fold_vs_bet:turn": -2.2,
         "fold_vs_bet:river": -1.8, "wtsd": +1.6, "wsd": -0.8,
         "aggression:flop": -1.4, "aggression:turn": -1.5},
    ),
    Archetype(
        "overfolder",
        "Plays plenty of pots and surrenders them the moment there is pressure.",
        "Barrel every street with any two cards; small sizings are enough. "
        "Give up only when they raise -- those raises are real hands.",
        {"vpip": +0.3, "fold_to_cbet:flop": +1.3, "fold_to_cbet:turn": +1.4,
         "fold_vs_bet:flop": +1.3, "fold_vs_bet:turn": +1.5, "fold_vs_bet:river": +1.3,
         "wtsd": -1.3, "check_raise:flop": -0.8, "wwsf": -0.8},
    ),
    Archetype(
        "maniac",
        "Relentless aggression at a frequency no range can support.",
        "Stop bluffing entirely and let them bet your good hands for you. "
        "Call down light, trap with strength, and raise only for value.",
        {"vpip": +1.6, "pfr": +1.8, "three_bet": +1.8, "cbet:flop": +1.3,
         "cbet:turn": +1.3, "aggression:flop": +1.7, "aggression:turn": +1.7,
         "aggression:river": +1.5, "wtsd": +0.4},
    ),
    Archetype(
        "lag",
        "Wide, aggressive and competent -- applies pressure with a real plan.",
        "Widen your calling ranges and let them barrel into you. Three-bet "
        "their steals; do not try to out-bluff them on later streets.",
        {"vpip": +1.0, "pfr": +1.2, "three_bet": +1.0, "cbet:flop": +0.7,
         "aggression:flop": +0.9, "aggression:turn": +0.8, "wwsf": +0.5,
         "fold_vs_bet:turn": -0.3},
    ),
    Archetype(
        "tag",
        "Solid and hard to exploit; frequencies sit close to the field.",
        "No large leak to attack. Play position, take the small edges, and "
        "look for the money at another seat.",
        {"vpip": 0.0, "pfr": +0.2, "three_bet": +0.1, "fold_to_three_bet": 0.0,
         "cbet:flop": +0.1, "fold_to_cbet:flop": 0.0, "fold_vs_bet:turn": 0.0,
         "aggression:flop": +0.2, "aggression:turn": +0.2, "wtsd": 0.0, "wsd": +0.4},
    ),
    Archetype(
        "limper",
        "Passive preflop -- limps in, calls raises, then plays fit-or-fold.",
        "Isolate their limps with a wide raising range and take it on the flop. "
        "They fold everything they miss and never raise without the goods.",
        {"vpip": +0.8, "pfr": -1.8, "limp": +2.5, "three_bet": -1.2,
         "fold_to_cbet:flop": +0.8, "aggression:flop": -1.2},
    ),
    Archetype(
        "trapper",
        "Tight and passive with a slow-play habit: checks strength, then raises.",
        "Value bet thinly but respect every check-raise. Keep pots small "
        "without a strong hand -- their passive lines are hiding made hands.",
        {"vpip": -0.8, "pfr": -0.6, "check_raise:flop": +2.0, "donk:flop": -0.5,
         "cbet:flop": -1.2, "aggression:flop": -0.8, "wtsd": +0.6, "wsd": +1.0},
    ),
]

ARCHETYPE_BY_NAME = {a.name: a for a in ARCHETYPES}

#: Beta-Binomial concentration. Low values mean an archetype tolerates a wide
#: band of frequencies; high values demand players hit the prototype exactly.
CONCENTRATION = 22.0

#: Features are correlated (VPIP with PFR, every fold stat with every other),
#: so the naive-Bayes product over-counts evidence. Discounting the total
#: log-likelihood keeps the posterior from reaching false certainty.
CORRELATION_DISCOUNT = 0.35

#: How common each archetype is in the wild -- the prior the likelihood updates.
#: With no hands on a player, this is the answer.
POPULATION_MIX = {
    "tag": 0.20, "station": 0.18, "overfolder": 0.16, "nit": 0.13,
    "lag": 0.12, "limper": 0.10, "maniac": 0.06, "trapper": 0.05,
}


def deviations(profile: Profile) -> dict[str, float]:
    """Player minus population, in spreads, for every feature with data."""
    out: dict[str, float] = {}
    for feature in PROFILE_FEATURES:
        est = profile.stats.get(feature)
        if est is None or est.opps <= 0:
            continue
        pop = population_mean(feature, profile.regime)
        out[feature] = (_logit(est.value) - _logit(pop)) / spread_of(feature)
    return out


def target_frequency(arch: Archetype, feature: str, table_regime: str) -> float:
    """The frequency this archetype implies for a feature at this table size."""
    pop = population_mean(feature, table_regime)
    return _sigmoid(_logit(pop) + arch.deviation(feature) * spread_of(feature))


def match(profile: Profile) -> tuple[str, float, list[tuple[str, float]]]:
    """Best-fitting archetype, its posterior probability, and the full mix.

    The returned confidence *is* the posterior -- no scaling afterwards. With
    no hands it equals the population prior; with a real sample it concentrates
    on whichever plan the counts support. Two archetypes that fit equally well
    produce two middling numbers, which is the honest answer: players do sit
    between buckets, and a forced label invites a plan the evidence cannot
    carry.
    """
    # One shared support set: every feature the player has real data on.
    observed = []
    for feature in PROFILE_FEATURES:
        est = profile.stats.get(feature)
        if est is not None and est.opps > 0 and est.raw is not None:
            observed.append((feature, est.raw * est.opps, est.opps))

    log_posterior = {}
    for arch in ARCHETYPES:
        total = 0.0
        for feature, hits, opps in observed:
            p = target_frequency(arch, feature, profile.regime)
            total += (IMPORTANCE.get(feature, DEFAULT_IMPORTANCE)
                      * _log_beta_binomial(hits, opps, p))
        log_posterior[arch.name] = (
            CORRELATION_DISCOUNT * total + math.log(POPULATION_MIX.get(arch.name, 0.05))
        )

    if not log_posterior:
        return "unknown", 0.0, []
    peak = max(log_posterior.values())
    weights = {name: math.exp(lp - peak) for name, lp in log_posterior.items()}
    total = sum(weights.values())
    mix = sorted(((n, w / total) for n, w in weights.items()), key=lambda kv: -kv[1])
    name, share = mix[0]
    return name, round(share, 3), [(n, round(s, 3)) for n, s in mix]


def _log_beta_binomial(hits: float, opps: float, mean: float,
                       concentration: float = CONCENTRATION) -> float:
    """Log marginal likelihood of ``hits``/``opps`` under a Beta(mean) prior.

    The binomial coefficient is dropped: it is identical across archetypes and
    only the differences matter.
    """
    a = mean * concentration
    b = (1 - mean) * concentration
    return _log_beta(a + hits, b + opps - hits) - _log_beta(a, b)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def describe(name: str) -> Archetype | None:
    return ARCHETYPE_BY_NAME.get(name)
