"""Population priors, and the shrinkage that makes small samples usable.

The central problem of profiling a home game: you have 40 hands on someone and
they folded to the only three continuation bets they faced. Their raw
fold-to-cbet is 100%. Betting every flop against them on that basis is how you
lose money to a player who is actually normal.

The fix is a Beta prior. Each frequency is estimated as

    (hits + m * prior_mean) / (opportunities + m)

where ``m`` is the prior's strength in pseudo-opportunities. With 3
observations the estimate barely moves off the population mean; with 300 the
prior is irrelevant. Every number this project reports is shrunk this way and
carries the credible interval that goes with it, so a read is never quoted
without its uncertainty.

Where the priors come from: hard-coded population defaults below, replaced by
*empirically estimated* ones as soon as the database holds enough players. The
empirical version fits a Beta-Binomial by moments -- the spread between players
in the database sets how much a new player's sample is trusted, which is
exactly the question the prior strength answers.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

# Table-size regimes. These are different games, not variations on one: a 55%
# VPIP is tight heads-up, normal three-handed, and a leak at a full table.
# Pooling them would flag every short-handed player as a maniac, so stats are
# bucketed by regime when they are *collected*, not just when they are read.
HEADS_UP, THREE, SHORT, FULL = "hu", "3max", "6max", "full"

REGIMES = (HEADS_UP, THREE, SHORT, FULL)
REGIME_LABELS = {HEADS_UP: "heads-up", THREE: "3-handed", SHORT: "short-handed (4-6)",
                 FULL: "full ring (7+)"}


def regime(table_size: float) -> str:
    """Regime of a table, by the number of players actually dealt in."""
    n = int(round(table_size))
    if n <= 2:
        return HEADS_UP
    if n == 3:
        return THREE
    if n <= 6:
        return SHORT
    return FULL


#: When a player is thin in one regime, borrow from the nearest neighbours in
#: this order rather than from an arbitrary default.
NEIGHBOURS = {HEADS_UP: (THREE, SHORT, FULL), THREE: (HEADS_UP, SHORT, FULL),
              SHORT: (THREE, FULL, HEADS_UP), FULL: (SHORT, THREE, HEADS_UP)}


# mean frequency by regime. Missing entries fall back to the 6max value, then
# to 0.5 with a weak prior.
POPULATION: dict[str, dict[str, float]] = {
    HEADS_UP: {
        "vpip": 0.70, "pfr": 0.55, "raise_share": 0.79, "rfi": 0.75, "limp": 0.06,
        "three_bet": 0.16, "fold_to_three_bet": 0.45, "four_bet": 0.12,
        "fold_to_four_bet": 0.45, "bb_defend": 0.62, "bb_fold_to_open": 0.38,
        "steal": 0.75, "fold_to_steal": 0.38, "three_bet_vs_steal": 0.16,
        "cold_call": 0.35, "squeeze": 0.10,
        "cbet:flop": 0.62, "cbet:turn": 0.52, "cbet:river": 0.45,
        "fold_to_cbet:flop": 0.42, "fold_to_cbet:turn": 0.42, "fold_to_cbet:river": 0.45,
        "fold_vs_bet:flop": 0.40, "fold_vs_bet:turn": 0.42, "fold_vs_bet:river": 0.45,
        "check_raise:flop": 0.10, "check_raise:turn": 0.08, "check_raise:river": 0.06,
        "donk:flop": 0.06, "donk:turn": 0.06, "donk:river": 0.06,
        "probe:turn": 0.30, "probe:river": 0.30,
        "wwsf": 0.47, "wtsd": 0.32, "wsd": 0.52,
        # With checks in the aggression denominator (see profile.DERIVED).
        "aggression:flop": 0.32, "aggression:turn": 0.30, "aggression:river": 0.27,
        "tank_fold": 0.12, "snap_call": 0.15, "snap_aggro": 0.10,
        "tank_fold:pf": 0.12, "tank_fold:flop": 0.12,
        "tank_fold:turn": 0.12, "tank_fold:river": 0.12,
        "snap_call:pf": 0.15, "snap_call:flop": 0.15,
        "snap_call:turn": 0.15, "snap_call:river": 0.15,
        "snap_aggro:pf": 0.10, "snap_aggro:flop": 0.10,
        "snap_aggro:turn": 0.10, "snap_aggro:river": 0.10,
        "river_bet_bluff": 0.30, "sd_light_call": 0.35,
    },
    SHORT: {
        "vpip": 0.24, "pfr": 0.19, "raise_share": 0.79, "rfi": 0.26, "limp": 0.05,
        "three_bet": 0.08, "fold_to_three_bet": 0.55, "four_bet": 0.08,
        "fold_to_four_bet": 0.50, "bb_defend": 0.42, "bb_fold_to_open": 0.58,
        "steal": 0.32, "fold_to_steal": 0.62, "three_bet_vs_steal": 0.11,
        "cold_call": 0.20, "squeeze": 0.08,
        "cbet:flop": 0.60, "cbet:turn": 0.50, "cbet:river": 0.44,
        "fold_to_cbet:flop": 0.45, "fold_to_cbet:turn": 0.45, "fold_to_cbet:river": 0.47,
        "fold_vs_bet:flop": 0.44, "fold_vs_bet:turn": 0.45, "fold_vs_bet:river": 0.47,
        "check_raise:flop": 0.09, "check_raise:turn": 0.07, "check_raise:river": 0.05,
        "donk:flop": 0.05, "donk:turn": 0.05, "donk:river": 0.05,
        "probe:turn": 0.28, "probe:river": 0.28,
        "wwsf": 0.45, "wtsd": 0.27, "wsd": 0.53,
        "aggression:flop": 0.28, "aggression:turn": 0.27, "aggression:river": 0.25,
        "tank_fold": 0.12, "snap_call": 0.15, "snap_aggro": 0.10,
        "tank_fold:pf": 0.12, "tank_fold:flop": 0.12,
        "tank_fold:turn": 0.12, "tank_fold:river": 0.12,
        "snap_call:pf": 0.15, "snap_call:flop": 0.15,
        "snap_call:turn": 0.15, "snap_call:river": 0.15,
        "snap_aggro:pf": 0.10, "snap_aggro:flop": 0.10,
        "snap_aggro:turn": 0.10, "snap_aggro:river": 0.10,
        "river_bet_bluff": 0.28, "sd_light_call": 0.33,
    },
}

# Three-handed sits between the two and is the most common short-table game
# outside of full sites: everyone is in the blinds two hands out of three, so
# defence frequencies stay near heads-up while opening ranges tighten.
POPULATION[THREE] = dict(
    POPULATION[HEADS_UP],
    vpip=0.55, pfr=0.44, rfi=0.48, limp=0.05,
    three_bet=0.13, fold_to_three_bet=0.50, four_bet=0.10,
    bb_defend=0.52, bb_fold_to_open=0.48,
    steal=0.55, fold_to_steal=0.48, three_bet_vs_steal=0.14,
    cold_call=0.26, wwsf=0.46, wtsd=0.29, wsd=0.53,
    aggression__flop=0.40,
)
POPULATION[THREE]["aggression:flop"] = 0.30
POPULATION[THREE]["aggression:turn"] = 0.28
POPULATION[THREE]["aggression:river"] = 0.26
POPULATION[THREE].pop("aggression__flop", None)
POPULATION[FULL] = dict(POPULATION[SHORT], vpip=0.20, pfr=0.15, rfi=0.20,
                        three_bet=0.06, steal=0.30, wtsd=0.25)

# How many pseudo-opportunities the prior is worth -- derived, not chosen.
#
# A prior should be trusted in proportion to how much it actually says. Where
# players cluster tightly around the population mean, that mean is a good guess
# for a newcomer and should dominate a small sample. Where players are spread
# out, the mean describes nobody and the data should win sooner. That is the
# Beta-Binomial moments relationship, and ``SPREAD`` already measures the
# spread, so the strength follows from it rather than from a table of guesses.
#
# The stat this matters most for is ``limp``. Its population is bimodal:
# competent players limp essentially never, limpers limp a third of the time,
# and the 5% average is a value almost nobody holds. Shrinking a limper toward
# it hid the most obvious leak in poker behind a number no player has.
STRENGTH_SCALE = 6.25          # calibrated so a mid-spread stat lands near 25
MIN_STRENGTH, MAX_STRENGTH = 8.0, 40.0

#: Overrides, for stats whose spread does not tell the whole story. Rare
#: events need a heavier prior than their spread implies, because a single
#: observation out of two chances should barely move anything.
STRENGTH: dict[str, float] = {
    "four_bet": 35, "fold_to_four_bet": 30, "five_bet": 35, "cold_four_bet": 35,
}
DEFAULT_STRENGTH = 25.0


def strength_for(stat: str) -> float:
    """Prior weight in pseudo-opportunities, from the between-player spread."""
    if stat in STRENGTH:
        return STRENGTH[stat]
    spread = spread_of(stat)
    return min(MAX_STRENGTH, max(MIN_STRENGTH, STRENGTH_SCALE / (spread * spread)))

# Continuous stats: (prior mean, prior strength in observations).
CONTINUOUS: dict[str, dict[str, tuple[float, float]]] = {
    HEADS_UP: {
        "open_bb": (2.6, 12), "three_bet_ratio": (3.3, 8),
        "cbet_size:flop": (0.55, 10), "cbet_size:turn": (0.62, 8),
        "cbet_size:river": (0.68, 8), "bet_size:flop": (0.55, 10),
        "bet_size:turn": (0.62, 8), "bet_size:river": (0.70, 8),
        "sd_strength": (0.60, 10), "net_bb": (0.0, 400),
    },
}
CONTINUOUS[THREE] = dict(CONTINUOUS[HEADS_UP], open_bb=(2.6, 12))
CONTINUOUS[SHORT] = dict(CONTINUOUS[HEADS_UP], open_bb=(2.5, 12))
CONTINUOUS[FULL] = dict(CONTINUOUS[SHORT])


#: Between-player spread of each stat in log-odds, the unit deviations are
#: measured in. Wider means the population genuinely disagrees about that
#: statistic -- everyone limps at roughly no rate at all until somebody limps
#: constantly, so ``limp`` spreads far more than ``wsd`` does.
SPREAD = {
    "vpip": 0.50, "pfr": 0.55, "raise_share": 0.55, "three_bet": 0.55, "fold_to_three_bet": 0.45,
    "cbet:flop": 0.50, "cbet:turn": 0.50, "cbet:river": 0.52,
    "fold_to_cbet:flop": 0.45, "fold_to_cbet:turn": 0.48,
    "fold_vs_bet:flop": 0.45, "fold_vs_bet:turn": 0.48, "fold_vs_bet:river": 0.50,
    "check_raise:flop": 0.70, "donk:flop": 0.70,
    "wwsf": 0.32, "wtsd": 0.42, "wsd": 0.32,
    "aggression:flop": 0.48, "aggression:turn": 0.50, "aggression:river": 0.50,
    "limp": 1.00, "bb_defend": 0.45,
}
DEFAULT_SPREAD = 0.50

#: Frequencies are clamped away from certainty before taking log-odds; a
#: measured 0% is "rarely", not "never".
EPSILON = 0.005


def logit(p: float) -> float:
    p = min(1.0 - EPSILON, max(EPSILON, p))
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def spread_of(feature: str, table_regime: str = "",
              priors: dict | None = None) -> float:
    """Between-player spread of a stat, in log-odds.

    Still a global constant, and the arguments are accepted but unused. That is
    a known inconsistency -- the population *mean* is fitted per regime while
    the spread it is measured in is not, and the two disagree by up to 3x in
    both directions on a real pool (fold_to_cbet:flop 0.45 built-in against a
    fitted 0.14 heads-up; donk:flop 0.70 against 2.80).

    Fitting it was tried and reverted. A Beta(mean, strength) implies a spread
    of ``1/sqrt(m(1-m)(s+1))``, which comes free from the fitted pair -- but
    where ``fit_empirical`` cannot separate the pool it returns a large
    strength, implying a tiny spread, and a tiny spread *amplifies* every
    deviation measured against it. Ten features hit that on a real pool and
    drove archetype confidence back to 1.00 by themselves.

    The change is worth making, but only against a validation harness that
    scores one half of a player's hands against the other. Tuning it against a
    handful of hand-labelled players is how the prototypes were overfitted
    twice already.
    """
    return SPREAD.get(feature, DEFAULT_SPREAD)






@dataclass(frozen=True)
class Estimate:
    """A shrunk frequency with the interval that honesty requires.

    ``alpha``/``beta`` are the full Beta posterior, kept so downstream code can
    ask probability questions -- "how sure are we they fold *more* than the
    field?" -- instead of being stuck with a point estimate and an interval.
    """

    value: float
    lo: float
    hi: float
    opps: float
    raw: float | None
    prior: float
    weight: float          # how much of the estimate came from data, 0-1
    #: Opportunities actually observed at this player's own table size, before
    #: any cross-regime pooling. ``opps`` includes borrowed pseudo-counts whose
    #: rate was already shrunk toward the prior, so anything that wants to score
    #: *observations* -- the archetype likelihood does -- has to use this
    #: instead, or it counts the same uncertainty twice.
    native_opps: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    strength: float = 0.0  # prior weight in pseudo-opportunities

    @property
    def confident(self) -> bool:
        return self.weight >= 0.5

    def prob_above(self, threshold: float) -> float:
        """Posterior probability the true frequency exceeds ``threshold``."""
        from scipy.stats import beta as _beta
        return float(_beta.sf(threshold, self.alpha, self.beta))

    def prob_below(self, threshold: float) -> float:
        from scipy.stats import beta as _beta
        return float(_beta.cdf(threshold, self.alpha, self.beta))

    def prior_prob_above(self, threshold: float) -> float:
        """What we would have believed with no hands on this player at all.

        Compared against :meth:`prob_above`, this is how much the *data* moved
        the answer -- the only part of a read that was actually earned.
        """
        from scipy.stats import beta as _beta
        a = max(self.prior * self.strength, 1e-6)
        b = max((1 - self.prior) * self.strength, 1e-6)
        return float(_beta.sf(threshold, a, b))

    def prior_prob_below(self, threshold: float) -> float:
        from scipy.stats import beta as _beta
        a = max(self.prior * self.strength, 1e-6)
        b = max((1 - self.prior) * self.strength, 1e-6)
        return float(_beta.cdf(threshold, a, b))

    def __format__(self, spec: str) -> str:
        return f"{100 * self.value:.0f}%"


def shrink(hits: float, opps: float, prior_mean: float, strength: float,
           z: float = 1.96) -> Estimate:
    """Posterior mean and interval of a Beta(prior)-Binomial."""
    a = prior_mean * strength + hits
    b = (1 - prior_mean) * strength + (opps - hits)
    total = a + b
    mean = a / total
    # Normal approximation to the Beta posterior; at these sample sizes the
    # difference from the exact quantiles is smaller than the rounding.
    sd = math.sqrt(mean * (1 - mean) / (total + 1))
    return Estimate(
        value=mean,
        lo=max(0.0, mean - z * sd),
        hi=min(1.0, mean + z * sd),
        opps=opps,
        raw=(hits / opps) if opps else None,
        prior=prior_mean,
        weight=opps / (opps + strength) if (opps + strength) else 0.0,
        alpha=a,
        beta=b,
        strength=strength,
    )


def population_mean(stat: str, table_regime: str) -> float:
    """The population frequency for a stat, with the same fallbacks as the prior."""
    return prior_for(stat, table_regime)[0]


def prior_for(stat: str, table_regime: str) -> tuple[float, float]:
    table = POPULATION.get(table_regime, POPULATION[SHORT])
    mean = table.get(stat)
    if mean is None:
        # Size-split fold stats inherit the street's overall fold frequency.
        base = stat.rsplit(":", 1)[0]
        mean = table.get(base, 0.5)
    return mean, strength_for(stat)


def fit_empirical(samples: dict[str, list[tuple[float, float]]],
                  min_players: int = 8) -> dict[str, tuple[float, float]]:
    """Beta-Binomial moments fit: (hits, opps) per player -> (mean, strength).

    The between-player spread is what sets the strength. If every player in the
    database folds to c-bets at roughly the same rate, a new player's three
    observations should barely move their estimate; if the spread is wide, the
    same three observations mean more.
    """
    out: dict[str, tuple[float, float]] = {}
    for stat, rows in samples.items():
        usable = [(h, o) for h, o in rows if o >= 5]
        if len(usable) < min_players:
            continue
        rates = [h / o for h, o in usable]
        n = len(rates)
        mean = sum(rates) / n
        var = sum((r - mean) ** 2 for r in rates) / (n - 1)
        mean_opps = sum(o for _, o in usable) / n
        # Subtract the binomial sampling noise to recover the true spread.
        within = mean * (1 - mean) / mean_opps
        between = var - within
        if between <= 1e-6 or not 0 < mean < 1:
            strength = 200.0                      # players are indistinguishable
        else:
            strength = max(2.0, mean * (1 - mean) / between - 1)
        out[stat] = (mean, min(strength, 400.0))
    return out
