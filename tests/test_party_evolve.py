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


def test_the_notebook_keeps_what_the_race_drops(dex, tmp_path):
    """Every graded candidate is written, not only the survivors.

    The point of the book: a generation grades twenty-four and carries three,
    and the twenty-one that were paid for should not vanish.
    """
    from pkcm.train.party_evolve import Notebook, basis_of, signature

    config = _config(population=6)
    book = Notebook.open(tmp_path / "book.jsonl")
    teams = [one.team for one in
             seed_population(dex, dex.regulation("m_b"), config,
                             Rng.from_seed(6).cursor())]
    graded = _graded(teams, [i / 10 for i in range(len(teams))])
    for one in graded:
        assert book.record(one, basis_of(config), "g1r1")

    assert len(book.seen) == len({signature(team) for team in teams})
    lines = (tmp_path / "book.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == book.written


def test_a_second_run_adds_to_the_first(dex, tmp_path):
    """Reopening reads what is there, and a better-measured grading wins."""
    from pkcm.train.party_evolve import Notebook, basis_of, signature

    config = _config(population=4)
    team = seed_population(dex, dex.regulation("m_b"), config,
                           Rng.from_seed(7).cursor())[0].team
    first = Candidate(team, "field", Floor(cvar=0.4, mean=0.4, minimum=0.4,
                                           low=0.3, high=0.5, games=20,
                                           matchups=()))
    Notebook.open(tmp_path / "book.jsonl").record(first, basis_of(config), "g1r1")

    again = Notebook.open(tmp_path / "book.jsonl")
    assert len(again.seen) == 1
    # Fewer games behind it, so it is not an improvement even though it scored
    # higher -- taking the maximum would be the winner's curse again.
    lucky = replace_floor(first, cvar=0.9, low=0.6, games=8)
    assert not again.record(lucky, basis_of(config), "g2r1")
    measured = replace_floor(first, cvar=0.35, low=0.32, games=200)
    assert again.record(measured, basis_of(config), "g2r4")
    assert again.seen[(basis_of(config), signature(team))]["games"] == 200


def test_bases_are_not_sorted_against_each_other(dex, tmp_path):
    """A judged floor and a searched floor are different measurements."""
    from pkcm.train.party_evolve import Notebook

    config = _config(population=4)
    teams = [one.team for one in
             seed_population(dex, dex.regulation("m_b"), config,
                             Rng.from_seed(8).cursor())]
    book = Notebook.open(tmp_path / "book.jsonl")
    book.record(_graded(teams[:1], [0.30])[0], "search", "g1r1")
    book.record(_graded(teams[1:2], [0.90])[0], "judge", "g8judge")

    assert book.bases() == ["judge", "search"]
    assert len(book.seen) == 2
    assert [one["floor"] for one in book.best(5, basis="search")] == [0.30]
    assert [one["floor"] for one in book.best(5, basis="judge")] == [0.90]


def replace_floor(candidate, **kw):
    from dataclasses import replace as _replace
    return _replace(candidate, floor=_replace(candidate.floor, **kw))


def test_a_judged_row_does_not_delete_the_searched_one(dex, tmp_path):
    """Two measurements of one party, not one measurement overwritten.

    The judge always has more games behind it, so keying the book by the
    party alone let every judged finalist quietly delete its own row from the
    searched list -- which is the list you read to find what to judge next.
    """
    from pkcm.train.party_evolve import Notebook, signature

    config = _config(population=4)
    team = seed_population(dex, dex.regulation("m_b"), config,
                           Rng.from_seed(9).cursor())[0].team
    book = Notebook.open(tmp_path / "book.jsonl")
    cheap = Candidate(team, "elite", Floor(cvar=0.25, mean=0.3, minimum=0.0,
                                           low=0.05, high=0.7, games=40,
                                           matchups=()))
    judged = replace_floor(cheap, cvar=0.44, low=0.20, games=400)
    assert book.record(cheap, "search", "g8r3")
    assert book.record(judged, "judge", "g8judge")

    assert len(book.seen) == 2
    assert [one["games"] for one in book.best(5, basis="search")] == [40]
    assert [one["games"] for one in book.best(5, basis="judge")] == [400]
    assert all(one["signature"] == signature(team) for one in book.seen.values())


def test_two_machines_books_merge_on_the_basis(dex, tmp_path):
    """The point of running two PCs: one list at the end, not two.

    And the merge is only allowed to join rows measured the same way, which
    is what makes matching the flags across the machines a requirement rather
    than a nicety.
    """
    from pkcm.train.party_evolve import Notebook

    config = _config(population=6)
    teams = [one.team for one in
             seed_population(dex, dex.regulation("m_b"), config,
                             Rng.from_seed(10).cursor())]
    here, there = Notebook.open(tmp_path / "a.jsonl"), Notebook.open(tmp_path / "b.jsonl")
    here.record(_graded(teams[:1], [0.30])[0], "same", "g1r1")
    there.record(_graded(teams[1:3], [0.40, 0.50])[0], "same", "g1r1")
    there.record(_graded(teams[3:4], [0.60])[0], "different flags", "g1r1")

    assert here.absorb(there) == 2
    assert len(here.best(9, basis="same")) == 2
    assert len(here.best(9, basis="different flags")) == 1
    # Reading did not write: each machine keeps its own file, which is what
    # keeps two appenders out of each other's way in git.
    assert len((tmp_path / "a.jsonl").read_text(encoding="utf-8").splitlines()) == 1
