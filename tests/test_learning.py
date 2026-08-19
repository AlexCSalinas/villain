"""The learned layers: clusters from a pool, hand strength from revealed cards.

Both refuse to run on too little data, and that refusal is the behavior worth
testing hardest -- a model fitted to six players will produce clusters, and
they will be noise.
"""

import numpy as np
import pytest

from villain.archetypes import ARCHETYPE_BY_NAME, target_frequency
from villain.cluster import MIN_PROFILES, NotEnoughData, fit_clusters
from villain.profile import PROFILE_FEATURES, build_profile
from villain.reads import MIN_ROWS, build_dataset, fit, texture
from villain.reads import NotEnoughData as ReadsNotEnoughData
from villain.stats import StatBook


def make_pool(kinds, per_kind=20, opps=60, seed=3):
    rng = np.random.default_rng(seed)
    profiles = []
    for kind in kinds:
        arch = ARCHETYPE_BY_NAME[kind]
        for i in range(per_kind):
            book = StatBook(player_id=f"{kind}{i}", name=f"{kind}{i}",
                            regime="6max", hands=opps * 3)
            for feature in PROFILE_FEATURES:
                p = float(np.clip(target_frequency(arch, feature, "6max")
                                  + rng.normal(0, 0.05), 0.02, 0.97))
                book.ratios[feature].hits = rng.binomial(opps, p)
                book.ratios[feature].opps = opps
            book.meters["table_size"].add(6, 1)
            profiles.append(build_profile(book))
    return profiles


def test_clustering_recovers_planted_groups():
    model = fit_clusters(make_pool(["station", "nit", "maniac"]))
    assert model.n_components == 3
    for cluster in model.clusters:
        kinds = {m.rstrip("0123456789") for m in cluster.members}
        assert len(kinds) == 1, f"cluster {cluster.index} mixed {kinds}"


def test_clusters_are_named_after_the_nearest_archetype():
    model = fit_clusters(make_pool(["station", "nit", "maniac"]))
    assert {c.label for c in model.clusters} == {"station", "nit", "maniac"}


def test_clustering_refuses_a_small_pool():
    with pytest.raises(NotEnoughData, match="textbook archetypes"):
        fit_clusters(make_pool(["station"], per_kind=MIN_PROFILES - 5))


def test_assign_places_a_new_player():
    profiles = make_pool(["station", "nit", "maniac"])
    model = fit_clusters(profiles)
    newcomer = make_pool(["station"], per_kind=1, seed=99)[0]
    cluster, strength = model.assign(newcomer)
    assert cluster.label == "station"
    assert strength > 0.5


def test_strength_dataset_labels_only_known_cards(hands):
    rows = build_dataset(hands)
    assert rows
    assert all(0.0 <= r.strength <= 1.0 for r in rows)
    assert all(len(r.features) == 17 for r in rows)
    # Folds are excluded: a folded hand has no strength worth predicting.
    assert "fold" not in {r.action for r in rows}


def test_strength_model_refuses_thin_data(hands):
    rows = build_dataset(hands)
    if len(rows) < MIN_ROWS:
        with pytest.raises(ReadsNotEnoughData, match="keep importing"):
            fit(rows)


def test_strength_model_fits_and_predicts(hands):
    rows = build_dataset(hands) * 30      # shape check only, not a statistical claim
    model = fit(rows)
    assert model.rows == len(rows)
    assert 0.0 <= model.mae <= 0.5
    prediction = model.predict(rows[0].features)
    assert 0.0 <= prediction <= 1.0


def test_unbiased_rows_are_marked(hands):
    """The exporting player's cards are visible without a showdown; villains' are not."""
    rows = build_dataset(hands)
    assert any(r.unbiased for r in rows)


@pytest.mark.parametrize("board,expected", [
    (["2c", "2d", "9h"], (1.0, 0.0, 0.0, 0.0)),
    (["2c", "5c", "9c"], (0.0, 1.0, 0.0, 0.0)),
    (["7c", "8d", "9h"], (0.0, 0.0, 1.0, 0.0)),
    (["Ac", "8d", "2h"], (0.0, 0.0, 0.0, 1.0)),
])
def test_board_texture(board, expected):
    assert texture(board) == expected
