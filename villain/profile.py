"""From stat books to profiles: shrunk estimates and derived features.

``PROFILE_FEATURES`` is the vector everything downstream agrees on -- archetype
matching, clustering, skill and the exploit rules all read the same numbers, so
a definition change propagates everywhere at once instead of drifting.

Estimates are shrunk through two levels. Population first: what players in
general do at this table size. Then the player themselves, across the other
table sizes they have been seen at -- somebody who never folds three-handed is
a decent prior for how they play heads-up, far better than the population is.
The discount on that second level is deliberate; the regimes are related, not
the same game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .priors import (CONTINUOUS, NEIGHBOURS, REGIME_LABELS, SHORT, Estimate, logit,
                     population_mean, prior_for, regime, shrink, sigmoid)
from .stats import Meter, Ratio, StatBook

# The features that define a player, in the order clustering expects.
PROFILE_FEATURES = [
    "vpip", "pfr", "raise_share", "three_bet", "fold_to_three_bet",
    "cbet:flop", "cbet:turn", "cbet:river",
    "fold_to_cbet:flop", "fold_to_cbet:turn",
    "fold_vs_bet:flop", "fold_vs_bet:turn", "fold_vs_bet:river",
    "check_raise:flop", "donk:flop",
    "wwsf", "wtsd", "wsd",
    "aggression:flop", "aggression:turn", "aggression:river",
    "limp", "bb_defend",
]

# Frequencies computed from other counters rather than counted directly.
# Aggression includes checks in the denominator: "of everything they do" means
# check/call/fold/bet/raise. Omitting checks made a 50/50 bet-or-check player
# look 100% aggressive and polluted archetype + skill reads.
DERIVED = {
    "aggression:flop": (("act:flop:bet", "act:flop:raise"),
                        ("act:flop:bet", "act:flop:raise", "act:flop:call",
                         "act:flop:fold", "act:flop:check")),
    "aggression:turn": (("act:turn:bet", "act:turn:raise"),
                        ("act:turn:bet", "act:turn:raise", "act:turn:call",
                         "act:turn:fold", "act:turn:check")),
    "aggression:river": (("act:river:bet", "act:river:raise"),
                         ("act:river:bet", "act:river:raise", "act:river:call",
                          "act:river:fold", "act:river:check")),
}

#: How much a player's stats in a *neighbouring* table size are worth as a
#: prior for this one. Related game, not the same game.
CROSS_REGIME_DISCOUNT = 0.35


@dataclass
class Profile:
    player_id: str
    name: str
    hands: int
    regime: str
    table_size: float
    stats: dict[str, Estimate] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)   # continuous, shrunk
    archetype: str = "unknown"
    archetype_confidence: float = 0.0
    archetype_mix: list[tuple[str, float]] = field(default_factory=list)
    tags: list = field(default_factory=list)
    skill: object | None = None
    first_seen: int | None = None
    last_seen: int | None = None
    borrowed_from: list[str] = field(default_factory=list)
    #: Empirically fitted population, when the database has enough players to
    #: support one. Carried here so everything downstream measures against the
    #: same population the shrinkage used -- the archetype label, the exploit
    #: thresholds and the rating were all still reading the built-in online
    #: defaults, so "learn from your own pool" only ever moved the number and
    #: never the read.
    priors: dict = field(default_factory=dict)

    def population(self, stat: str) -> float:
        """The population frequency this profile is measured against."""
        from .priors import population_mean
        fitted = self.priors.get(stat)
        return fitted[0] if fitted else population_mean(stat, self.regime)
    #: hands played at each table size, busiest first. Empty for a profile
    #: built from a single regime.
    contributions: dict[str, int] = field(default_factory=dict)

    @property
    def table_mix(self) -> str:
        """Plain description of where these hands came from."""
        if not self.contributions:
            return self.regime_label
        parts = [f"{n} {REGIME_LABELS.get(r, r)}" for r, n in self.contributions.items()]
        return ", ".join(parts)

    def get(self, stat: str) -> float | None:
        e = self.stats.get(stat)
        return e.value if e else None

    def opps(self, stat: str) -> float:
        e = self.stats.get(stat)
        return e.opps if e else 0.0

    @property
    def regime_label(self) -> str:
        return REGIME_LABELS.get(self.regime, self.regime)

    @property
    def winrate_bb100(self) -> float | None:
        v = self.means.get("net_bb")
        return v * 100 if v is not None else None

    @property
    def sample_quality(self) -> str:
        """Plain-language reliability, so a read is never quoted bare."""
        if self.hands >= 500:
            return "solid"
        if self.hands >= 150:
            return "usable"
        if self.hands >= 50:
            return "thin"
        return "guesswork"


def build_profile(book: StatBook, others: dict[str, StatBook] | None = None,
                  priors: dict[str, tuple[float, float]] | None = None) -> Profile:
    """Shrink one regime's book into a profile.

    ``others`` is the same player's books in other regimes, used as a personal
    prior. ``priors`` overrides the population defaults, which is how
    empirically fitted priors from the database get used.
    """
    reg = book.regime or regime(book.mean("table_size") or 6.0)
    others = {r: b for r, b in (others or {}).items() if r != reg and b.hands > 0}
    priors = priors or {}

    profile = Profile(
        player_id=book.player_id, name=book.name, hands=book.hands,
        regime=reg, table_size=book.mean("table_size") or 0.0,
        first_seen=book.first_seen, last_seen=book.last_seen,
        borrowed_from=[r for r in NEIGHBOURS.get(reg, ()) if r in others],
        priors=dict(priors),
    )

    def estimate(stat: str, hits: float, opps: float) -> Estimate:
        mean, strength = priors.get(stat) or prior_for(stat, reg)
        mean, strength = _personal_prior(stat, others, mean, strength)
        return shrink(hits, opps, mean, strength)

    for stat, ratio in book.ratios.items():
        if stat.startswith("seat:") or stat.startswith("saw:"):
            continue
        profile.stats[stat] = estimate(stat, ratio.hits, ratio.opps)

    for stat, (num_keys, den_keys) in DERIVED.items():
        hits = sum(book.ratios[k].hits for k in num_keys if k in book.ratios)
        opps = sum(book.ratios[k].hits for k in den_keys if k in book.ratios)
        if opps:
            profile.stats[stat] = estimate(stat, hits, opps)

    continuous = CONTINUOUS.get(reg, {})
    for stat, meter in book.meters.items():
        if meter.n <= 0:
            continue
        prior_mean, prior_n = continuous.get(stat, (None, 0.0))
        profile.means[stat] = (meter.mean if prior_mean is None
                               else (meter.total + prior_mean * prior_n) / (meter.n + prior_n))
        profile.means[f"{stat}#n"] = meter.n
        if meter.sd is not None:
            profile.means[f"{stat}#sd"] = meter.sd

    return profile


def _personal_prior(stat: str, others: dict[str, StatBook], pop_mean: float,
                    pop_strength: float) -> tuple[float, float]:
    """Bend the population prior toward what this player does elsewhere."""
    hits = opps = 0.0
    for book in others.values():
        ratio = book.ratios.get(stat)
        if ratio:
            hits += ratio.hits
            opps += ratio.opps
    if opps <= 0:
        return pop_mean, pop_strength
    mean = (hits + pop_mean * pop_strength) / (opps + pop_strength)
    return mean, pop_strength + CROSS_REGIME_DISCOUNT * opps


def build_profiles(by_regime: dict[str, StatBook], min_hands: int = 1,
                   priors: dict[str, tuple[float, float]] | None = None) -> list[Profile]:
    """One profile per regime the player has been seen in, busiest first."""
    profiles = [build_profile(book, others=by_regime, priors=priors)
                for reg, book in by_regime.items() if book.hands >= min_hands]
    profiles.sort(key=lambda p: -p.hands)
    return profiles


def merge_books(by_regime: dict[str, StatBook]) -> StatBook:
    """All regimes in one book. Use for lifetime totals, never for frequencies."""
    total = StatBook(regime="all")
    for book in by_regime.values():
        total.merge(book)
    return total


def feature_vector(profile: Profile) -> list[float | None]:
    return [profile.get(f) for f in PROFILE_FEATURES]


def evidence(profile: Profile) -> list[float]:
    """How much real data backs each feature, 0-1. Clustering weights by this."""
    return [profile.stats[f].weight if f in profile.stats else 0.0 for f in PROFILE_FEATURES]


# ---------------------------------------------------------------------------
# one player, one profile
# ---------------------------------------------------------------------------
# Splitting statistics by table size is a statistical necessity and a
# presentational disaster. It is necessary because 55% VPIP is tight heads-up
# and reckless at a full ring, so pooling the raw counts produces a number that
# describes neither game. It is a disaster because the person reading it wants
# to know what this opponent is like, not to hold four partial reads in their
# head and reconcile them.
#
# The fix is to pool in the right space. A player's *style* -- how far they sit
# from normal for the game they are in -- carries across table sizes even
# though their raw frequencies do not. So each table's counts are converted to
# a deviation from that table's own population, translated onto the primary
# table's scale, and only then added together.
#
# Concretely, for each statistic:
#
#   1. shrink the other table's rate toward that table's population, so a
#      3-of-4 sample does not arrive as 75%;
#   2. measure the deviation in log-odds, which is the regime-invariant part;
#   3. re-express that deviation against the primary table's population;
#   4. add it in as pseudo-counts, discounted, because related games are not
#      the same game.
#
# The result reads as a single profile measured against the game they play
# most, informed by everything else they have done.

def primary_regime(by_regime: dict[str, StatBook]) -> str:
    """The table size this player is mostly seen at."""
    live = {r: b for r, b in by_regime.items() if b.hands > 0}
    if not live:
        return SHORT
    return max(live.items(), key=lambda kv: kv[1].hands)[0]


def unified_book(by_regime: dict[str, StatBook]) -> tuple[StatBook, dict[str, int]]:
    """Fold every table size into one book on the primary table's scale."""
    live = {r: b for r, b in by_regime.items() if b.hands > 0}
    if not live:
        return StatBook(), {}

    home = primary_regime(live)
    source = live[home]
    merged = StatBook(player_id=source.player_id, name=source.name, regime=home)
    merged.hands = sum(b.hands for b in live.values())
    merged.first_seen = min((b.first_seen for b in live.values()
                             if b.first_seen is not None), default=None)
    merged.last_seen = max((b.last_seen for b in live.values()
                            if b.last_seen is not None), default=None)

    for stat, ratio in source.ratios.items():
        merged.ratios[stat].hits = ratio.hits
        merged.ratios[stat].opps = ratio.opps
    for stat, meter in source.meters.items():
        merged.meters[stat].merge(meter)

    for regime, book in live.items():
        if regime == home:
            continue
        for stat, ratio in book.ratios.items():
            if ratio.opps <= 0:
                continue
            translated = _translate_rate(stat, ratio, regime, home)
            weight = CROSS_REGIME_DISCOUNT * ratio.opps
            merged.ratios[stat].hits += translated * weight
            merged.ratios[stat].opps += weight
        for stat, meter in book.meters.items():
            if meter.n <= 0 or stat == "table_size":
                continue
            share = CROSS_REGIME_DISCOUNT
            merged.meters[stat].n += meter.n * share
            merged.meters[stat].total += meter.total * share
            merged.meters[stat].sumsq += meter.sumsq * share

    # table_size stays honest: the mean of every hand actually played, so the
    # profile can say what mix it came from.
    merged.meters["table_size"] = Meter()
    for book in live.values():
        merged.meters["table_size"].merge(book.meters.get("table_size", Meter()))

    contributions = {r: b.hands for r, b in sorted(
        live.items(), key=lambda kv: -kv[1].hands)}
    return merged, contributions


def _translate_rate(stat: str, ratio: Ratio, source: str, target: str) -> float:
    """Re-express a rate measured in one regime on another regime's scale.

    Shrunk first, because an unshrunk 0% or 100% has no finite log-odds and a
    tiny sample would translate into an extreme claim.
    """
    mean, strength = prior_for(stat, source)
    shrunk = shrink(ratio.hits, ratio.opps, mean, strength).value
    source_pop = population_mean(stat, source)
    target_pop = population_mean(stat, target)
    if source_pop == target_pop:
        return shrunk
    deviation = logit(shrunk) - logit(source_pop)
    return sigmoid(logit(target_pop) + deviation)


def build_unified(by_regime: dict[str, StatBook],
                  priors: dict[str, tuple[float, float]] | None = None) -> Profile | None:
    """One profile per player, informed by every table size they have played."""
    book, contributions = unified_book(by_regime)
    if not contributions:
        return None
    profile = build_profile(book, priors=priors)
    profile.contributions = contributions
    profile.borrowed_from = [r for r in contributions if r != profile.regime]
    return profile
