"""The evolutionary search over parties, without paying for a battle.

Everything here is about the shape of the search -- what an operator may and
may not do to a team -- so the fitness is stubbed. What a floor is worth is
``test_party_floor``'s question.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.legality import ranker_parties, team_errors
from pkcm.engine.rng import Rng
from pkcm.train.party_evolve import (Candidate, EvolveConfig, breed, carrying,
                                     cross, seed_population)
from pkcm.train.party_floor import Floor, FloorConfig


FIELD = "data/champions/parties_field.json"


@pytest.fixture(scope="module")
def dex():
    return load_dex()


def _config(core="garchomp", **kw):
    return EvolveConfig(core=core, parties=FIELD,
                        floor=FloorConfig(parties=FIELD), **kw)


def _graded(teams, scores):
    """Candidates with a fitness, without running a single battle."""
    return [Candidate(team, "test",
                      Floor(cvar=value, mean=value, minimum=value,
                            low=value, high=value, games=10, matchups=()))
            for team, value in zip(teams, scores)]


def test_the_field_has_something_to_start_from(dex):
    parties = ranker_parties(FIELD)
    around = carrying(parties, "garchomp")
    assert len(around) > 20, "the warm start needs real teams to begin with"
    assert all(any(one.species == "garchomp" for one in party.team)
               for party in around)


def test_a_core_that_cannot_be_fielded_is_refused(dex):
    """A core the field has never used is grafted; an illegal one is refused.

    The two are different failures and only one of them is a dead end. A
    Salamence nobody has built around yet gets a slot in somebody else's team;
    a Magikarp is not in the regulation at all, so no team can hold it and
    there is nothing to search.
    """
    regulation = dex.regulation("m_c")
    config = _config(core="magikarp", population=4)
    with pytest.raises(ValueError, match="cannot be put into any party"):
        seed_population(dex, regulation, config, Rng.from_seed(1).cursor())


def test_the_population_keeps_the_core_and_stays_legal(dex):
    regulation = dex.regulation("m_b")
    config = _config(population=12)
    population = seed_population(dex, regulation, config,
                                 Rng.from_seed(2).cursor())
    assert len(population) == 12
    for one in population:
        assert any(slot.species == "garchomp" for slot in one.team), one.origin
        assert not team_errors(dex, regulation, one.team, "singles"), one.origin


def test_breeding_never_loses_the_core(dex):
    """The whole search is about the five slots around a fixed one."""
    regulation = dex.regulation("m_b")
    config = _config(population=10, elites=2)
    cursor = Rng.from_seed(3).cursor()
    population = seed_population(dex, regulation, config, cursor)
    graded = _graded([one.team for one in population],
                     [i / 10 for i in range(len(population))])

    for _ in range(5):
        children = breed(dex, regulation, config, graded, cursor)
        assert len(children) == config.population
        for child in children:
            assert any(slot.species == "garchomp" for slot in child.team), \
                f"the core was lost by {child.origin}"
            assert not team_errors(dex, regulation, child.team, "singles"), \
                f"{child.origin} produced an illegal team"
        graded = _graded([one.team for one in children],
                         [i / 10 for i in range(len(children))])


def test_a_crossover_is_refused_rather_than_repaired(dex):
    """Two of a base species is illegal, and the answer is to drop the child.

    A repair would be a mutation nobody asked for, and it would be attributed
    to the crossover -- which is the one thing this search is arranged to keep
    separate.
    """
    regulation = dex.regulation("m_b")
    config = _config(population=8)
    cursor = Rng.from_seed(4).cursor()
    parties = carrying(ranker_parties(FIELD), "garchomp")
    refused = made = 0
    for first in parties[:6]:
        for second in parties[:6]:
            child = cross(dex, regulation, config, first.team, second.team,
                          cursor)
            if child is None:
                refused += 1
                continue
            made += 1
            assert any(one.species == "garchomp" for one in child)
            assert not team_errors(dex, regulation, child, "singles")
    assert made, "no crossover ever succeeded"
    assert refused, "no crossover was ever refused; the check is not firing"


def test_elites_come_through_ungraded(dex):
    """They keep their team and lose their score, so the new games decide."""
    regulation = dex.regulation("m_b")
    config = _config(population=8, elites=3)
    cursor = Rng.from_seed(5).cursor()
    population = seed_population(dex, regulation, config, cursor)
    graded = _graded([one.team for one in population],
                     [i / 10 for i in range(len(population))])
    best = sorted(graded, key=lambda one: -one.fitness)[:3]

    children = breed(dex, regulation, config, graded, cursor)
    assert [one.origin for one in children[:3]] == ["elite"] * 3
    assert [one.team for one in children[:3]] == [one.team for one in best]
    assert all(one.floor is None for one in children[:3])
