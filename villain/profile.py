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

from .priors import (CONTINUOUS, NEIGHBOURS, REGIME_LABELS, Estimate, prior_for,
                     regime, shrink)
from .stats import StatBook

# The features that define a player, in the order clustering expects.
PROFILE_FEATURES = [
    "vpip", "pfr", "three_bet", "fold_to_three_bet",
    "cbet:flop", "cbet:turn", "cbet:river",
    "fold_to_cbet:flop", "fold_to_cbet:turn",
    "fold_vs_bet:flop", "fold_vs_bet:turn", "fold_vs_bet:river",
    "check_raise:flop", "donk:flop",
    "wwsf", "wtsd", "wsd",
    "aggression:flop", "aggression:turn", "aggression:river",
    "limp", "bb_defend",
]

# Frequencies computed from other counters rather than counted directly.
DERIVED = {
    "aggression:flop": (("act:flop:bet", "act:flop:raise"),
                        ("act:flop:bet", "act:flop:raise", "act:flop:call", "act:flop:fold")),
    "aggression:turn": (("act:turn:bet", "act:turn:raise"),
                        ("act:turn:bet", "act:turn:raise", "act:turn:call", "act:turn:fold")),
    "aggression:river": (("act:river:bet", "act:river:raise"),
                         ("act:river:bet", "act:river:raise", "act:river:call", "act:river:fold")),
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
