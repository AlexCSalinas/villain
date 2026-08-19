"""Whether a player treats you differently from everybody else.

The rest of this package shrinks a player toward a population: what people in
general do at this table size. That is the right prior for "what is he like".
It is the wrong one for "what is he like *against me*", because the population
has never played you. The prior that question wants is the player himself.

So an adjustment is a third level of the same arithmetic. Population, then the
player, then the player against you:

* the **baseline** is his rate against everyone else -- the pooled counter with
  the against-you slice *subtracted out*, shrunk toward the population as
  usual. Subtracting matters. The slice is inside the pooled total, so
  comparing it against the total compares a number with something that
  contains it, and the difference shrinks toward nothing exactly when the
  sample is large enough to be worth reading. :meth:`Store.session_detail`
  takes a session's baseline from the player's *other* hands for the same
  reason;
* the **against-you estimate** is the slice, shrunk toward that baseline;
* the **read** is what is left: how far the slice moved off his own baseline,
  and how sure the posterior is about the direction.

Nothing here compares him to the field. Folding 70% to your river bets is not
interesting because 70% is high; it is interesting because he folds 45% to
everyone else's.

**Table size is handled the way the rest of the package handles it.** Raw
counts are never pooled across regimes -- an opponent you mostly play heads-up
would show an against-you "adjustment" that is nothing but the table size.
Instead each regime's slice is measured against *that regime's* baseline, and
the deviation, which is the part that carries, is re-expressed on the primary
table's scale before it is added in. That is :func:`villain.profile.
_translate_rate` with the player's own baseline in place of the population.
"""

from __future__ import annotations

from dataclasses import dataclass

from .priors import Estimate, logit, prior_for, shrink, sigmoid
from .profile import CROSS_REGIME_DISCOUNT, primary_regime
from .stats import VS_HERO, StatBook

#: How strongly to believe, before seeing any of it, that he plays you the way
#: he plays everybody -- in opportunities.
#:
#: Nothing in a database can fit this. It would take pairwise samples across
#: many players to learn how much people vary between opponents, and a home
#: game has one pair worth counting. So it is a judgment call, stated here
#: rather than buried: raise it and an adjustment needs more evidence before it
#: shows, lower it and normal variation starts reading as a read.
ADJUSTMENT_PRIOR = 30.0

#: Against-you opportunities before an adjustment can be reported at all,
#: counted at the player's own table size and never including the pseudo-counts
#: borrowed from another one.
MIN_OPPS = 12

#: And decisions against *other* people, or there is no baseline to differ
#: from. This is the honest refusal in a heads-up database: if you are the only
#: person he has played, "against you" and "in general" are the same hands, and
#: the difference between them is not a read, it is a subtraction of a number
#: from itself.
MIN_BASELINE_OPPS = 12

#: Posterior probability the shift has the sign it appears to have.
MIN_CONFIDENCE = 0.85

#: And a floor on the shift itself. A 3-point difference can be arbitrarily
#: certain given enough hands and still not change a single decision.
MIN_GAP = 0.08

#: Facing a bet a player folds, calls or raises: one decision and three
#: counters that add to one. A shift in any of them is the same shift seen from
#: another side, so reporting all three would say one thing three times and
#: then sort it to the top by weight of numbers. Only the largest is kept.
#: :func:`villain.exploits.dedupe_leaks` collapses overlapping leaks for the
#: same reason.
ONE_DECISION = ("fold_vs_bet", "call_vs_bet", "raise_vs_bet")

#: Regularisation on another table size's slice before its deviation is
#: measured -- just enough to keep the log-odds finite when somebody folded to
#: none of thirty bets. Deliberately tiny: :data:`ADJUSTMENT_PRIOR` is applied
#: to the pooled result at the end, and applying it here as well shrinks the
#: borrowed deviation to nothing before the cross-regime discount has even
#: touched it, which discards the signal the borrowing exists to carry.
TRANSLATION_SMOOTHING = 2.0


@dataclass(frozen=True)
class Adjustment:
    """One statistic on which a player treats you differently."""

    stat: str                # the pooled counter, e.g. "fold_vs_bet:river"
    regime: str              # the table size this is expressed on
    versus: float            # against-you frequency, shrunk toward the baseline
    baseline: float          # their frequency against everyone else
    opps: float              # against-you decisions actually observed
    borrowed_opps: float     # pseudo-counts carried from another table size
    baseline_opps: float     # decisions against everybody else
    confidence: float        # posterior probability of the direction
    estimate: Estimate       # the full posterior, for anything that wants it

    @property
    def gap(self) -> float:
        """Signed difference: positive means they do it *more* against you."""
        return self.versus - self.baseline

    @property
    def direction(self) -> str:
        return "more" if self.gap > 0 else "less"


def adjustments(by_regime: dict[str, StatBook],
                priors: dict[str, tuple[float, float]] | None = None,
                min_opps: float = MIN_OPPS,
                min_confidence: float = MIN_CONFIDENCE) -> list[Adjustment]:
    """Every statistic where the against-you slice moved off the baseline.

    Returns nothing rather than something weak: below the sample floors, or
    inside the confidence and size floors, a statistic is simply absent. A
    player you have no read on is the normal case at these sample sizes, and
    an empty list says so more honestly than a list of coin flips.
    """
    live = {r: b for r, b in by_regime.items() if b.hands > 0}
    if not live:
        return []
    home = primary_regime(live)
    priors = priors or {}

    out: list[Adjustment] = []
    for stat in _sliced_stats(live):
        # A read that only holds at one table size is still a read. Folding
        # every regime into the home one and translating the rest loses
        # exactly the strongest cases: one player folds to turn bets 23% of
        # the time heads-up against 42% otherwise, and 6-max the same gap runs
        # the *other way* -- averaged, they cancel and nothing is reported.
        # Where a single table size carries the read on its own decisions, say
        # so and name the table size.
        for regime, book in live.items():
            native = _adjustment_within(stat, book, regime, priors,
                                        REGIME_MIN_OPPS, min_confidence)
            if native is not None:
                out.append(native)
        found = _adjustment(stat, live, home, priors, min_opps, min_confidence)
        if found is not None:
            out.append(found)
    out.sort(key=lambda a: -abs(a.gap))
    return _one_per_decision(out)


#: Against-you decisions needed *at one table size* before that table size is
#: reported on its own. Higher than MIN_OPPS: a claim this specific ("heads-up
#: he does not fold to you") should not rest on a dozen hands.
REGIME_MIN_OPPS = 35


def _adjustment_within(stat: str, book: StatBook, regime: str,
                       priors: dict[str, tuple[float, float]],
                       min_opps: float, min_confidence: float) -> Adjustment | None:
    """The against-you read at one table size, on that table size's own hands.

    Nothing is translated or borrowed here, which is the point: this is the
    slice as observed, measured against how they play everybody else *at the
    same table size*.
    """
    slice_ = book.ratios.get(VS_HERO + stat)
    if slice_ is None or slice_.opps < min_opps:
        return None
    baseline = _baseline(stat, book, regime, priors)
    if baseline is None:
        return None
    estimate = shrink(slice_.hits, slice_.opps, baseline.value, ADJUSTMENT_PRIOR)
    gap = estimate.value - baseline.value
    if abs(gap) < MIN_GAP:
        return None
    confidence = (estimate.prob_above(baseline.value) if gap > 0
                  else estimate.prob_below(baseline.value))
    if confidence < min_confidence:
        return None
    return Adjustment(
        stat=stat, regime=regime, versus=estimate.value, baseline=baseline.value,
        opps=slice_.opps, borrowed_opps=0.0, baseline_opps=baseline.opps,
        confidence=confidence, estimate=estimate,
    )


def _one_per_decision(found: list[Adjustment]) -> list[Adjustment]:
    """Keep the clearest view of each decision, drop the other sides of it.

    Keyed by table size as well as decision: "heads-up he will not fold to you"
    and "six-handed he folds normally" are two facts about one player, and
    collapsing them to one loses whichever is second.
    """
    seen: set[tuple[str, str, str]] = set()
    out = []
    for adjustment in found:                    # widest gap first
        decision, street = _decision(adjustment.stat)
        key = (decision, street, adjustment.regime)
        if key in seen:
            continue
        seen.add(key)
        out.append(adjustment)
    return out


def _decision(stat: str) -> tuple[str, str]:
    base, _, street = stat.partition(":")
    return ("vs_bet" if base in ONE_DECISION else base), street


def _sliced_stats(live: dict[str, StatBook]) -> list[str]:
    """The pooled counters that have an against-you slice anywhere."""
    stats = {
        stat[len(VS_HERO):]
        for book in live.values()
        for stat, ratio in book.ratios.items()
        if stat.startswith(VS_HERO) and ratio.opps > 0
    }
    return sorted(stats)


def _adjustment(stat: str, live: dict[str, StatBook], home: str,
                priors: dict[str, tuple[float, float]],
                min_opps: float, min_confidence: float) -> Adjustment | None:
    baseline = _baseline(stat, live.get(home), home, priors)
    if baseline is None:
        return None

    hits = opps = observed = borrowed = 0.0
    for regime, book in live.items():
        slice_ = book.ratios.get(VS_HERO + stat)
        if slice_ is None or slice_.opps <= 0:
            continue
        if regime == home:
            hits += slice_.hits
            opps += slice_.opps
            observed += slice_.opps
            continue
        # Another table size. Its rate does not transfer, but how far he sits
        # from his own baseline there does, so that is what is carried over --
        # discounted, because a related game is not the same game.
        other = _baseline(stat, book, regime, priors)
        if other is None:
            continue
        here = shrink(slice_.hits, slice_.opps, other.value, TRANSLATION_SMOOTHING)
        translated = sigmoid(logit(baseline.value) + logit(here.value) - logit(other.value))
        weight = CROSS_REGIME_DISCOUNT * slice_.opps
        hits += translated * weight
        opps += weight
        observed += slice_.opps
        borrowed += weight

    # Counted on decisions seen, never on the pseudo-counts borrowed from
    # another table size: those arrive already shrunk, and letting them clear
    # the floor would count the same uncertainty twice.
    if observed < min_opps:
        return None

    estimate = shrink(hits, opps, baseline.value, ADJUSTMENT_PRIOR)
    gap = estimate.value - baseline.value
    if abs(gap) < MIN_GAP:
        return None
    # The baseline is treated as fixed here rather than as its own posterior.
    # It is the far thicker sample of the two -- it is every decision the slice
    # is not -- so the uncertainty that decides this is the slice's.
    confidence = (estimate.prob_above(baseline.value) if gap > 0
                  else estimate.prob_below(baseline.value))
    if confidence < min_confidence:
        return None

    return Adjustment(
        stat=stat, regime=home, versus=estimate.value, baseline=baseline.value,
        opps=observed, borrowed_opps=borrowed, baseline_opps=baseline.opps,
        confidence=confidence, estimate=estimate,
    )


def _baseline(stat: str, book: StatBook | None, regime: str,
              priors: dict[str, tuple[float, float]]) -> Estimate | None:
    """Their rate at this table size against everyone who is not you.

    The against-you slice is subtracted out of the pooled counter, which is the
    whole reason this is a separate function and not ``profile.stats[stat]``.
    """
    if book is None:
        return None
    pooled = book.ratios.get(stat)
    if pooled is None:
        return None
    slice_ = book.ratios.get(VS_HERO + stat)
    hits = pooled.hits - (slice_.hits if slice_ else 0.0)
    opps = pooled.opps - (slice_.opps if slice_ else 0.0)
    if opps < MIN_BASELINE_OPPS:
        return None
    mean, strength = priors.get(stat) or prior_for(stat, regime)
    return shrink(hits, opps, mean, strength)
