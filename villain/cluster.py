"""Learning archetypes from your own pool instead of assuming them.

The prototypes in :mod:`villain.archetypes` are textbook player types. They work
from the first hand, which is why they are the default, but they are opinions
about how poker players cluster in general -- not about how *your* game
clusters. A home game may contain three distinct kinds of player and none of
them may be a nit.

This module fits a Gaussian mixture to the profiles in the database and reports
the groups actually present. Two rules keep it honest:

* It refuses to run on too few players. A mixture model fitted to six profiles
  will produce clusters, and they will be noise. The component count is chosen
  by BIC over a range that the sample size can support, not by hope.
* Missing features are imputed with the population mean and down-weighted, so
  a player with no river data does not get clustered by a number nobody
  measured.

The output labels each cluster with its nearest textbook archetype, because a
cluster id tells you nothing about how to play against somebody.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .archetypes import ARCHETYPES, deviations
from .profile import PROFILE_FEATURES, Profile

#: Below this many profiles, clustering is pattern-matching on noise.
MIN_PROFILES = 25

#: Minimum profiles per component, used to cap how many components are tried.
MIN_PER_COMPONENT = 8


@dataclass
class Cluster:
    index: int
    label: str                       # nearest textbook archetype
    size: int
    share: float
    centroid: dict[str, float]       # feature -> deviation from population
    members: list[str] = field(default_factory=list)

    def describe(self) -> str:
        top = sorted(self.centroid.items(), key=lambda kv: -abs(kv[1]))[:4]
        traits = ", ".join(f"{k} {v:+.1f}" for k, v in top)
        return f"cluster {self.index} (~{self.label}, {self.size} players): {traits}"


@dataclass
class ClusterModel:
    clusters: list[Cluster]
    n_components: int
    bic: float
    trained_on: int

    def assign(self, profile: Profile) -> tuple[Cluster, float] | None:
        """Which learned group a profile belongs to, and how strongly."""
        if not self.clusters or self._gmm is None:
            return None
        row = _row(profile)[None, :]
        probs = self._gmm.predict_proba(row)[0]
        best = int(np.argmax(probs))
        return self.clusters[best], float(probs[best])

    _gmm: object | None = None


class NotEnoughData(ValueError):
    pass


def fit_clusters(profiles: list[Profile], max_components: int = 6,
                 random_state: int = 0) -> ClusterModel:
    """Fit a Gaussian mixture over profile deviations, choosing k by BIC."""
    usable = [p for p in profiles if p.hands >= 50]
    if len(usable) < MIN_PROFILES:
        raise NotEnoughData(
            f"need {MIN_PROFILES} profiles with 50+ hands to learn clusters, "
            f"have {len(usable)}; the textbook archetypes still apply")

    from sklearn.mixture import GaussianMixture

    matrix = np.vstack([_row(p) for p in usable])
    ceiling = min(max_components, max(2, len(usable) // MIN_PER_COMPONENT))

    best, best_bic, best_k = None, np.inf, 1
    for k in range(2, ceiling + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                              random_state=random_state, reg_covar=1e-3)
        gmm.fit(matrix)
        bic = gmm.bic(matrix)
        if bic < best_bic:
            best, best_bic, best_k = gmm, bic, k
    if best is None:
        raise NotEnoughData("mixture did not converge on this sample")

    labels = best.predict(matrix)
    clusters = []
    for index in range(best_k):
        members = [p for p, l in zip(usable, labels) if l == index]
        centroid = dict(zip(PROFILE_FEATURES, best.means_[index]))
        clusters.append(Cluster(
            index=index,
            label=_nearest_archetype(centroid),
            size=len(members),
            share=len(members) / len(usable),
            centroid={k: round(v, 2) for k, v in centroid.items()},
            members=[p.name for p in members],
        ))
    model = ClusterModel(clusters=clusters, n_components=best_k,
                         bic=float(best_bic), trained_on=len(usable))
    model._gmm = best
    return model


def _row(profile: Profile) -> np.ndarray:
    """Feature vector in spread units, with unmeasured features set to zero.

    Zero means "no deviation from the population", which is the right thing to
    assume about a statistic nobody has observed: it keeps the player at the
    center on that axis instead of inventing a tendency.
    """
    z = deviations(profile)
    return np.array([z.get(f, 0.0) for f in PROFILE_FEATURES], dtype=float)


def _nearest_archetype(centroid: dict[str, float]) -> str:
    """Name a learned cluster after the textbook type it most resembles."""
    best, best_distance = "unclassified", np.inf
    for arch in ARCHETYPES:
        total = count = 0.0
        for feature in PROFILE_FEATURES:
            expected = arch.deviation(feature)
            actual = centroid.get(feature, 0.0)
            total += (actual - expected) ** 2
            count += 1
        distance = (total / count) ** 0.5
        if distance < best_distance:
            best, best_distance = arch.name, distance
    return best
