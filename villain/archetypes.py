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

from .priors import (DEFAULT_SPREAD, SPREAD, logit, population_mean,
                     sigmoid, spread_of)
from .profile import PROFILE_FEATURES, Profile

#: How much each feature counts toward identifying a plan. Shared by every
#: archetype: these are exponents in a likelihood, so varying them per
#: archetype would make the scores incomparable.
IMPORTANCE = {
    "vpip": 2.0, "pfr": 1.3, "raise_share": 2.6, "three_bet": 2.2, "fold_to_three_bet": 0.9,
    "fold_vs_bet:flop": 1.4, "fold_vs_bet:turn": 1.6, "fold_vs_bet:river": 3.4,
    "fold_to_cbet:flop": 1.2, "fold_to_cbet:turn": 1.0,
    "aggression:flop": 1.4, "aggression:turn": 1.4, "aggression:river": 1.2,
    "cbet:flop": 1.1, "cbet:turn": 1.0, "cbet:river": 0.8,
    "check_raise:flop": 1.2, "donk:flop": 0.8,
    "wtsd": 1.5, "wsd": 1.0, "wwsf": 0.8, "limp": 1.2, "bb_defend": 1.0,
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
        "Steal their blinds every orbit -- they fold far more than the "
        "price of a raise requires, so it prints regardless of your cards. "
        "Then get out of the way. When they raise, or call two streets, "
        "they have it, and the pots they contest are the ones to keep "
        "small. Your profit here is a lot of tiny uncontested pots, not one "
        "big one.",
        {"vpip": -2.2, "pfr": -1.8, "raise_share": +0.3, "three_bet": -1.2, "fold_to_three_bet": +1.5,
         "fold_vs_bet:flop": +1.2, "fold_vs_bet:turn": +1.2, "wtsd": -1.2,
         "aggression:flop": -1.0, "bb_defend": -1.5},
    ),
    Archetype(
        "station",
        "Calls far too much and folds far too little; will not release a pair.",
        "Bet every hand that is ahead, on all three streets, and size up as "
        "you go -- they are not folding to a price, so charge them "
        "properly. Top pair with a bad kicker is a three-street value hand "
        "here. The discipline is the other half: never bluff, not on a "
        "scare card and not with a busted draw. Checking back your air is "
        "where most of the edge against them comes from.",
        # Preflop volume is deliberately mild. "Station" is a *postflop* plan,
        # and the fold family below is its identity; hanging half the label on
        # VPIP made this a preflop-looseness detector that threw out every
        # tight player who will not fold after the flop -- which is most of
        # them. Loose entry belongs to "loose passive" and "limper". Note the
        # asymmetry: *passivity* (low PFR given entry) stays a full part of the
        # identity, because that is the plan -- it is only the volume that was
        # borrowed from a different bucket.
        {"vpip": +0.7, "pfr": -1.2, "raise_share": -1.2, "three_bet": -1.0,
         "fold_to_cbet:flop": -2.0, "fold_vs_bet:flop": -2.0, "fold_vs_bet:turn": -2.2,
         "fold_vs_bet:river": -1.8, "wtsd": +1.6, "wsd": -0.8,
         "aggression:flop": -1.4, "aggression:turn": -1.5},
    ),
    Archetype(
        "overfolder",
        "Plays plenty of pots and surrenders them the moment there is pressure.",
        "Bet at them constantly and keep the sizes small -- they are "
        "folding to the fact of the bet, not to its price, so there is no "
        "reason to risk more. Flop, then turn, and take the river too when "
        "the board misses the obvious draws. The one rule: stop the moment "
        "they do anything other than fold. Their raises and their multi-"
        "street calls are genuine, because everything weak left the hand "
        "already.",
        {"raise_share": +0.1, "vpip": +0.3, "fold_to_cbet:flop": +1.3, "fold_to_cbet:turn": +1.4,
         "fold_vs_bet:flop": +1.3, "fold_vs_bet:turn": +1.5, "fold_vs_bet:river": +1.3,
         "wtsd": -1.3, "check_raise:flop": -0.8, "wwsf": -0.8},
    ),
    Archetype(
        "maniac",
        "Relentless aggression at a frequency no range can support.",
        "Play patient and passive, and let them do the betting. Call down "
        "far lighter than feels comfortable -- middle pair is a real hand "
        "against a range this wide -- and flat rather than raise your "
        "strong hands so they keep firing into them. Never bluff and never "
        "raise as a bluff: that is the one part of their game already "
        "working. Expect variance; the money arrives in lumps.",
        {"raise_share": +0.5, "vpip": +1.6, "pfr": +1.8, "three_bet": +1.8, "cbet:flop": +1.3,
         "cbet:turn": +1.3, "aggression:flop": +1.7, "aggression:turn": +1.7,
         "aggression:river": +1.5, "wtsd": +0.4},
    ),
    Archetype(
        "lag",
        "Wide, aggressive and competent -- applies pressure with a real plan.",
        "Tighten what you open but widen everything you continue with -- "
        "against this player the money is made after the flop, not before "
        "it. Re-raise their late-position opens, and call down more on "
        "turns and rivers rather than folding to pressure. Do not start a "
        "bluffing war on the later streets; they are competent, and that "
        "part of their game works. Position matters more here than against "
        "anyone else.",
        # Identity on the axes a LAG actually shows: they enter raising, three-bet
        # hard and never limp. Softening the postflop demands made this the
        # smallest prototype in the table (L1 6.1) and therefore the new magnet
        # for every average player -- the same failure TAG had.
        {"raise_share": +0.5, "vpip": +0.9, "pfr": +1.5, "three_bet": +1.6,
         "limp": -1.2, "fold_to_three_bet": -0.6, "cbet:flop": +0.45,
         "aggression:flop": +0.6, "aggression:turn": +0.55, "wwsf": +0.5,
         "fold_vs_bet:turn": -0.3},
    ),
    Archetype(
        "tag",
        "Solid home-game reg: field-sized volume, raises when in, no cheap "
        "limp-call leaks.",
        "There is no cheap edge here, so stop looking for one -- the "
        "mistake against a solid player is inventing a read and over-"
        "adjusting to it. Play a straightforward positional game, keep pots "
        "small out of position, and take thin value where it exists. If "
        "there is a weaker player at the table, your attention belongs on "
        "them; against this one, breaking even is a fine result.",
        # Near-field volume, but a real identity rather than a near-copy of the
        # population: this prototype used to sit ~8x closer to the field than
        # any other, which made it the bucket every ambiguous player fell into.
        # The identity is discipline -- enters raising, never limps, never
        # donks -- not raw aggression frequency, which a tight winner reads
        # *below* field on. It says nothing at all about fold frequency: that
        # axis is "station"'s identity, and a TAG may be sticky or not without
        # leaving the bucket.
        # The fold signal is here, and it points the *other way* from the
        # attempt that was reverted last session. That one gave TAG a positive
        # fold deviation -- folds more than the field -- and it pushed real
        # TAGs into tight passive. Measured against four players with known
        # labels, a TAG folds rivers *less* than the field and plays tighter:
        # they get to the river with hands worth calling, so they are not the
        # ones surrendering to the last bet.
        {"vpip": -0.9, "pfr": +0.2, "raise_share": +0.70, "three_bet": +0.35,
         "limp": -1.0, "donk:flop": -0.6, "fold_to_three_bet": -0.3,
         "cbet:flop": +0.3, "fold_vs_bet:river": -1.3,
         "check_raise:flop": +0.6, "wsd": +0.3, "wtsd": -0.2},
    ),
    Archetype(
        "tight passive",
        "Below-field volume and timid after the flop -- weak-tight, not a TAG.",
        "Open wider into their blinds and keep betting when they check; they "
        "are folding more than the pot odds ask for and rarely put you in "
        "hard spots themselves. Do not pay off their rare raises -- those "
        "are the strong end of an otherwise timid range. Value is thinner "
        "than against a station because they will get away from weak pairs, "
        "so prefer bluffs and small-stab continuation bets over three-street "
        "value with mediocre holdings.",
        {"raise_share": -1.3, "vpip": -1.0, "pfr": -1.0, "three_bet": -0.8, "fold_to_three_bet": +0.6,
         "fold_vs_bet:flop": +0.6, "fold_vs_bet:turn": +0.5, "cbet:flop": -0.8,
         "aggression:flop": -1.1, "aggression:turn": -1.1, "wtsd": +0.5,
         "check_raise:flop": -0.6},
    ),
    Archetype(
        "loose passive",
        "Sees too many flops and then calls along -- passive fish, not a station.",
        "Isolate their limps and calls with a wide raising range, then bet "
        "for thin value on later streets; they came to play and will stick "
        "around with second pair. Bluff less than against a nit -- they call "
        "too often for pure air to print -- but do not turn into a station "
        "yourself: when they finally raise, give them credit. The edge is "
        "volume of small-to-medium value pots, not one heroic bluff.",
        # Elevated VPIP is the gate. Limp is optional — many fish just flat.
        {"raise_share": -1.4, "vpip": +1.7, "pfr": +0.35, "three_bet": -1.0, "limp": 0.0,
         "fold_to_cbet:flop": -0.5, "fold_vs_bet:flop": -0.4, "fold_vs_bet:turn": -0.3,
         "aggression:flop": -0.9, "aggression:turn": -0.9, "cbet:flop": -0.3,
         "wtsd": +0.8, "wsd": -0.3},
    ),
    Archetype(
        "limper",
        "Passive preflop -- limps in, calls raises, then plays fit-or-fold.",
        "Raise over every limp with a wide range, around four big blinds "
        "plus one per limper, and bet the flop whether or not you "
        "connected. Their limping range is capped and built for seeing "
        "cheap flops, so they miss constantly and give up when they do. "
        "Slow down when they call the flop -- that call means something "
        "real -- and fold to their raises without hesitation.",
        {"raise_share": -1.9, "vpip": +0.8, "pfr": -1.8, "limp": +2.5, "three_bet": -1.2,
         "fold_to_cbet:flop": +0.8, "aggression:flop": -1.2},
    ),
    Archetype(
        "trapper",
        "Tight and passive with a slow-play habit: checks strength, then raises.",
        "Bet your good hands for value but stay alert: this player checks "
        "strong hands rather than betting them, so a check is not the "
        "invitation it is against most opponents. Keep pots small with "
        "marginal holdings and treat every check-raise as the real thing -- "
        "they do not have a bluffing range there. The edge comes from not "
        "paying them off in the big pots they engineer.",
        {"raise_share": -0.5, "vpip": -0.8, "pfr": -0.6, "check_raise:flop": +2.2, "donk:flop": -0.5,
         "cbet:flop": -1.4, "aggression:flop": -0.9, "wtsd": +0.6, "wsd": +1.0},
    ),
]

ARCHETYPE_BY_NAME = {a.name: a for a in ARCHETYPES}

#: Beta-Binomial concentration. Low values mean an archetype tolerates a wide
#: band of frequencies; high values demand players hit the prototype exactly.
#: At 22 the implied tolerance was ~0.95 population spreads -- half of an
#: entire prototype signature -- which cost only the archetypes whose identity
#: *is* being far from the mean, and left the near-field ones untouched.
CONCENTRATION = 40.0

#: Features are correlated (VPIP with PFR, every fold stat with every other),
#: so the naive-Bayes product over-counts evidence. Discounting the total
#: log-likelihood keeps the posterior from reaching false certainty.
#: Measured rather than guessed. The value should be the effective number of
#: independent measurements divided by the total importance, and two methods
#: agree: the eigenvalue participation ratio of the importance-weighted
#: correlation matrix over a real pool gives n_eff 7.06 of 32.0 (0.221), and
#: held-out cross-validation -- fit on half a player's hands, score against the
#: other half -- minimises log loss at 0.15-0.25.
#:
#: Earlier values were tuned against a simulation that generated players *from*
#: the prototypes. In that world the model is correctly specified, so no such
#: harness can ever detect prototype misfit, and it will always endorse too
#: much confidence. Against disjoint halves of real players, 0.55 produced a
#: calibration error of 0.275 and nine players at 1.00; 0.20 gives 0.092 and
#: none, at no cost in accuracy.
CORRELATION_DISCOUNT = 0.20

#: How common each archetype is in the wild -- the prior the likelihood updates.
#: With no hands on a player, this is the answer.
POPULATION_MIX = {
    "tag": 0.16, "loose passive": 0.16, "station": 0.12, "overfolder": 0.11,
    "tight passive": 0.10, "lag": 0.10, "limper": 0.09, "nit": 0.08,
    "maniac": 0.04, "trapper": 0.04,
}


def deviations(profile: Profile) -> dict[str, float]:
    """Player minus population, in spreads, for every feature with data."""
    out: dict[str, float] = {}
    for feature in PROFILE_FEATURES:
        est = profile.stats.get(feature)
        if est is None or est.opps <= 0:
            continue
        pop = profile.population(feature)
        out[feature] = ((logit(est.value) - logit(pop))
                        / spread_of(feature, profile.regime, profile.priors))
    return out


def target_frequency(arch: Archetype, feature: str, table_regime: str,
                     profile: "Profile | None" = None) -> float:
    """The frequency this archetype implies for a feature at this table size.

    Pass ``profile`` to measure against that player's fitted population rather
    than the built-in one. A prototype is a deviation *from the field*, so with
    a fitted pool the same deviation has to be applied to the fitted mean or
    the label is describing a different field than the numbers are.
    """
    pop = profile.population(feature) if profile is not None \
        else population_mean(feature, table_regime)
    spread = spread_of(feature, table_regime,
                       profile.priors if profile is not None else None)
    return sigmoid(logit(pop) + arch.deviation(feature) * spread)


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
            p = target_frequency(arch, feature, profile.regime, profile)
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
                       concentration: float | None = None) -> float:
    """Log marginal likelihood of ``hits``/``opps`` under a Beta(mean) prior.

    The binomial coefficient is dropped: it is identical across archetypes and
    only the differences matter.
    """
    # Read the module constant at call time: binding it as a default would
    # freeze it at import, silently ignoring anything that tries to fit it.
    if concentration is None:
        concentration = CONCENTRATION
    a = mean * concentration
    b = (1 - mean) * concentration
    return _log_beta(a + hits, b + opps - hits) - _log_beta(a, b)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def describe(name: str) -> Archetype | None:
    return ARCHETYPE_BY_NAME.get(name)
