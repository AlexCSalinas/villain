"""GTO comparison and rating: fidelity tagging, sample floor, and the score."""

from dataclasses import dataclass

from villain import gto
from villain.priors import FULL, HEADS_UP, SHORT, THREE


@dataclass
class _Est:
    value: float
    opps: float


class _Profile:
    """Just enough of villain.profile.Profile for gto.compare."""

    def __init__(self, regime, stats):
        self.regime = regime
        self.stats = {k: _Est(v, n) for k, (v, n) in stats.items()}


def test_every_regime_has_targets():
    for r in (HEADS_UP, THREE, SHORT, FULL):
        assert gto.targets_for(r), r


def test_unknown_regime_falls_back_to_six_max():
    assert gto.targets_for("nonsense") is gto.GTO[SHORT]


def test_preflop_is_solver_postflop_is_benchmark():
    prof = _Profile(SHORT, {
        "three_bet": (0.09, 500),        # preflop -> solver
        "cbet:flop": (0.55, 500),        # postflop -> benchmark
    })
    fid = {r.stat: r.fidelity for r in gto.compare(prof)}
    assert fid["three_bet"] == gto.EXACT
    assert fid["cbet:flop"] == gto.BENCHMARK


def test_thin_stats_are_not_scored():
    prof = _Profile(SHORT, {"three_bet": (0.09, gto.MIN_OPPS - 1)})
    assert gto.compare(prof) == []
    assert gto.rating(gto.compare(prof)) is None


def test_matching_gto_rates_near_100():
    # A player sitting exactly on every target should rate ~100.
    stats = {k: (v, 300) for k, v in gto.GTO[SHORT].items()}
    prof = _Profile(SHORT, stats)
    assert gto.rating(gto.compare(prof)) >= 99


def test_far_from_gto_rates_low():
    # Flip every frequency to its opposite -- as wrong as it gets.
    stats = {k: (1 - v, 300) for k, v in gto.GTO[SHORT].items()}
    prof = _Profile(SHORT, stats)
    assert gto.rating(gto.compare(prof)) < 30


def test_rows_sorted_by_gap_widest_first():
    prof = _Profile(SHORT, {
        "three_bet": (0.10, 300),        # 1pp off
        "fold_to_steal": (0.80, 300),    # 32pp off
    })
    rows = gto.compare(prof)
    assert rows[0].stat == "fold_to_steal"
    assert abs(rows[0].deviation) > abs(rows[1].deviation)


def test_deviation_sign_reads_as_more_than_optimal():
    prof = _Profile(SHORT, {"three_bet": (0.20, 300)})  # target 0.09
    assert gto.compare(prof)[0].deviation > 0


def test_preflop_outweighs_postflop_in_the_score():
    # Perfect preflop, terrible postflop should beat the reverse, because the
    # exact (preflop) rows carry double weight.
    pf_good = _Profile(SHORT, {"three_bet": (0.09, 300), "wwsf": (0.90, 300)})
    pf_bad = _Profile(SHORT, {"three_bet": (0.40, 300), "wwsf": (0.48, 300)})
    assert gto.rating(gto.compare(pf_good)) > gto.rating(gto.compare(pf_bad))
