"""The plain-language layer: what a leak says, and the guard on the optional model."""

import json

import pytest

from villain.archetypes import ARCHETYPES
from villain.exploits import PRESSURE, RULES, TIERS, find_leaks, size_band
from villain.narrate import Unavailable, enabled, fact_sheet, unsupported_numbers
from villain.playbook import COMBINATIONS, PLAYBOOK, combinations_for, entry_for


# -- coverage ---------------------------------------------------------------

def test_every_rule_has_a_playbook_entry():
    """A leak with no words is a number nobody can act on."""
    missing = [r.id for r in RULES if not entry_for(r.id)]
    assert not missing, f"no playbook entry for {missing}"


def test_no_orphan_playbook_entries():
    assert not set(PLAYBOOK) - {r.id for r in RULES}


def test_every_entry_answers_all_four_questions():
    for leak_id, entry in PLAYBOOK.items():
        for field in ("behaviour", "why", "do", "dont"):
            text = getattr(entry, field)
            assert text and len(text) > 40, f"{leak_id}.{field} is too thin"


def test_dont_is_a_real_counter_mistake():
    """The 'do not' field exists to stop over-adjustment; it must say so."""
    for leak_id, entry in PLAYBOOK.items():
        assert entry.dont.lower().startswith(("do not", "never", "don't")), leak_id
        assert entry.dont != entry.do, leak_id


def test_combinations_reference_real_rules():
    known = {r.id for r in RULES}
    for combo in COMBINATIONS:
        assert combo.leaks <= known, combo.headline
        assert len(combo.leaks) >= 2


def test_combinations_only_fire_when_all_parts_are_present():
    combo = COMBINATIONS[0]
    part = next(iter(combo.leaks))
    assert combo not in combinations_for({part})
    assert combo in combinations_for(combo.leaks)


def test_archetype_plans_are_substantial():
    """The plan is the headline advice; two terse sentences will not do."""
    for arch in ARCHETYPES:
        assert len(arch.plan) > 200, arch.name
        assert "  " not in arch.plan, f"{arch.name} has a formatting artefact"


# -- the derived plain-language fields --------------------------------------

@pytest.mark.parametrize("severity,expected", [
    (5.0, "big"), (1.5, "solid"), (0.5, "modest"), (0.05, "small")])
def test_size_bands(severity, expected):
    assert size_band(severity)[0] == expected


def test_pressure_covers_every_tier():
    for _, tier in TIERS:
        assert tier in PRESSURE


def test_leak_exposes_words_as_well_as_numbers(synth_profile):
    leaks = find_leaks(synth_profile("overfolder", regime="hu", opps=150))
    assert leaks
    leak = leaks[0]
    for field in ("behaviour", "why", "do", "dont", "priority", "pressure", "in_words"):
        assert getattr(leak, field), field
    assert "%" in leak.in_words
    assert leak.size in {"big", "solid", "modest", "small"}


def test_in_words_states_the_direction_correctly(synth_profile):
    """'more often than breakeven' and 'less often than' are opposite reads."""
    over = find_leaks(synth_profile("overfolder", regime="hu", opps=150))
    station = find_leaks(synth_profile("station", regime="hu", opps=150))
    high = next(l for l in over if l.direction == "high")
    low = next(l for l in station if l.direction == "low")
    assert "more often than" in high.in_words
    assert "less often than" in low.in_words


def test_analyze_export_carries_the_language(tmp_path, hands):
    from villain.analyze import as_dict
    from villain.db import Store
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        player = max(store.players(), key=lambda r: r["hands"] or 0)
        payload = as_dict(store.profiles(int(player["id"]))[0])
    assert "combinations" in payload and "plan" in payload
    json.dumps(payload)
    for leak in payload["leaks"]:
        for field in ("behaviour", "why", "do", "dont", "priority", "in_words"):
            assert leak[field], field


# -- the optional model -----------------------------------------------------

def test_narrator_is_off_unless_configured(monkeypatch):
    monkeypatch.delenv("VILLAIN_LLM_MODEL", raising=False)
    monkeypatch.delenv("VILLAIN_LLM_URL", raising=False)
    assert enabled() is False


def test_fact_sheet_contains_only_computed_values(tmp_path, hands):
    from villain.analyze import as_dict
    from villain.db import Store
    with Store(tmp_path / "v.db") as store:
        store.add_hands(hands)
        player = max(store.players(), key=lambda r: r["hands"] or 0)
        payload = as_dict(store.profiles(int(player["id"]))[0])
    sheet = fact_sheet(payload)
    assert payload["name"] in sheet
    assert str(payload["hands"]) in sheet
    assert payload["archetype"] in sheet


def test_invented_numbers_are_caught():
    """The whole point of the guard: prose that reads fine but states a figure
    the arithmetic never produced."""
    facts = "They fold 51% of rivers. Breakeven is 40%. Seen 16 times."
    assert unsupported_numbers("They fold about 51% of rivers, over the 40% breakeven.",
                               facts) == []
    assert unsupported_numbers("They fold 78% of rivers.", facts) == ["78"]


def test_rounding_is_allowed():
    facts = "Worth 0.09 big blinds per 100 hands over 183 hands."
    assert unsupported_numbers("Worth roughly 0 bb/100 across 183 hands.", facts) == []


def test_narrate_reports_why_it_could_not_run(monkeypatch):
    from villain import narrate as module
    monkeypatch.setenv("VILLAIN_LLM_URL", "http://127.0.0.1:9/none")
    with pytest.raises(Unavailable, match="could not reach"):
        module.narrate({"name": "x", "regime": "hu", "hands": 1,
                        "sample_quality": "guesswork", "archetype": "tag",
                        "archetype_confidence": 0.5, "summary": "",
                        "skill": {"score": 50, "tier": "competent"}, "leaks": []},
                       timeout=2)
