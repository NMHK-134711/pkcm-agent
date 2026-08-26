"""Champions' SP stat system."""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine.stats import (
    LEVEL,
    NATURES,
    SP_PER_STAT_CAP,
    SP_TOTAL,
    compute_stat,
    compute_stats,
    get_nature,
    is_legal_sp,
    sp_errors,
    spread_from_stats,
)


def _reference_stat(base: int, sp: int, multiplier: float, stat: Stat) -> int:
    """The long-form level-50 formula, written out independently.

    Note where SP lands: *outside* the inner floor. Champions adds it to the
    finished level-50 value rather than folding it in as EVs did, which is why
    one SP is exactly one point.
    """
    inner = ((2 * base + 31) * LEVEL) // 100
    if stat is Stat.HP:
        return inner + LEVEL + 10 + sp
    return int((inner + 5 + sp) * multiplier)


def test_there_are_exactly_21_stat_alignments():
    assert len(NATURES) == 21
    assert [n.name for n in NATURES.values() if n.is_neutral] == ["Serious"]


def test_stat_alignments_are_well_formed():
    pairs = set()
    for nature in NATURES.values():
        if nature.is_neutral:
            assert nature.hindered is None
            continue
        assert nature.boosted is not nature.hindered
        assert Stat.HP not in (nature.boosted, nature.hindered), "HP is never aligned"
        pairs.add((nature.boosted, nature.hindered))
    assert len(pairs) == 20, "every boost/hinder pair appears exactly once"


def test_nature_multipliers():
    jolly = get_nature("jolly")  # +Spe -SpA
    assert jolly.multiplier(Stat.SPE) == pytest.approx(1.1)
    assert jolly.multiplier(Stat.SPA) == pytest.approx(0.9)
    assert jolly.multiplier(Stat.ATK) == 1.0
    assert jolly.multiplier(Stat.HP) == 1.0, "HP is immune to alignment"

    serious = get_nature("serious")
    assert all(serious.multiplier(stat) == 1.0 for stat in Stat)


def test_unknown_alignment_is_rejected():
    with pytest.raises(KeyError):
        get_nature("hardy")  # a neutral nature Champions removed


def test_one_sp_is_exactly_one_point():
    base, nature = 100, get_nature("serious")
    for stat in Stat:
        without = compute_stat(base, 0, nature, stat)
        with_one = compute_stat(base, 1, nature, stat)
        assert with_one - without == 1


def test_collapsed_formula_matches_the_long_form():
    """Guards the ``Base + 75 + SP`` / ``(Base + 20 + SP) * n`` simplification."""
    for base in range(1, 256):
        for sp in (0, 1, 2, 31, 32):
            for nature_id in ("serious", "adamant", "modest"):
                nature = get_nature(nature_id)
                for stat in Stat:
                    assert compute_stat(base, sp, nature, stat) == _reference_stat(
                        base, sp, nature.multiplier(stat), stat
                    )


def test_compute_stats_on_a_real_species():
    dex = load_dex()
    garchomp = dex.species["garchomp"]  # 108/130/95/80/85/102
    sp = spread_from_stats(("atk", "spe"), (32, 32))
    stats = compute_stats(garchomp.base_stats, sp, "jolly")

    assert stats[Stat.HP] == 108 + 75 + 0
    assert stats[Stat.ATK] == int((130 + 20 + 32) * 1.0)
    assert stats[Stat.SPA] == int((80 + 20 + 0) * 0.9)
    assert stats[Stat.SPE] == int((102 + 20 + 32) * 1.1)


def test_off_level_is_refused():
    with pytest.raises(ValueError):
        compute_stat(100, 0, get_nature("serious"), Stat.ATK, level=100)


def test_sp_budget_rules():
    assert is_legal_sp(spread_from_stats(("atk", "spe"), (32, 32)))  # 64 of 66
    assert is_legal_sp((11, 11, 11, 11, 11, 11))  # exactly 66
    assert is_legal_sp((0, 0, 0, 0, 0, 0))  # spending nothing is legal

    over_cap = sp_errors(spread_from_stats(("atk",), (SP_PER_STAT_CAP + 1,)))
    assert any("per-stat cap" in e for e in over_cap)

    over_budget = sp_errors((12, 12, 12, 12, 12, 12))  # 72
    assert any("budget" in e for e in over_budget)

    assert any("negative" in e for e in sp_errors((-1, 0, 0, 0, 0, 0)))
    assert sp_errors((0, 0, 0)) == ["expected 6 SP values, got 3"]


def test_budget_cannot_max_three_stats():
    """66 total against a 32 cap: two stats can be maxed, a third gets the crumbs."""
    assert 2 * SP_PER_STAT_CAP <= SP_TOTAL < 3 * SP_PER_STAT_CAP
