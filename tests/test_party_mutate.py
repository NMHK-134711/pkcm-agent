"""One legal change at a time, to a party somebody already built."""

from __future__ import annotations

import collections

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.legality import ranker_parties, ranker_slots, team_errors
from pkcm.engine.rng import Rng
from pkcm.engine.stats import SP_PER_STAT_CAP, SP_TOTAL
from pkcm.train import party_mutate as mut


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def regulation(dex):
    return dex.regulation("m_b")


@pytest.fixture(scope="module")
def parties():
    return ranker_parties()


def cursor(seed=1):
    return Rng.from_seed(seed).cursor()


def test_every_operator_leaves_a_legal_team(dex, regulation, parties):
    """The property the whole module rests on. An illegal team does not fail
    loudly later -- ``new_battle`` builds it and the search plans with a
    Pokemon that cannot exist."""
    draw = cursor(7)
    for round_number in range(300):
        team = parties[round_number % len(parties)].team
        name, out = mut.mutate(dex, regulation, team, draw)
        assert not team_errors(dex, regulation, out), (name, team_errors(dex, regulation, out))


def test_an_operator_that_cannot_move_returns_what_it_was_given(dex, regulation, parties):
    """Better than an illegal team or an exception: the caller sees nothing
    changed and can spend its budget elsewhere."""
    team = parties[0].team
    # Gengar's only registrable ability in Champions is Cursed Body, so the
    # ability operator has nowhere to go on a team of them.
    single = tuple(one for one in ranker_slots() if one.species == "gengar")[:1]
    if single:
        out = mut.mutate_ability(dex, regulation, single, cursor(3))
        assert out == single

    out = mut.mutate_move(dex, regulation, team, cursor(3))
    assert len(out) == len(team)


def test_a_mutation_changes_one_pokemon_and_no_others(dex, regulation, parties):
    draw = cursor(11)
    for round_number in range(80):
        team = parties[round_number % len(parties)].team
        _, out = mut.mutate(dex, regulation, team, draw)
        differing = [i for i, (a, b) in enumerate(zip(team, out)) if a != b]
        assert len(differing) <= 1, differing


def test_the_spread_templates_are_legal_and_come_from_the_sets(dex):
    """136 million spreads satisfy the rules. The ranker sets say which shapes
    people use, and the templates are those shapes over every assignment."""
    templates = mut.spread_templates()
    assert len(templates) > 100
    for spread in templates:
        assert sum(spread) == SP_TOTAL
        assert all(0 <= value <= SP_PER_STAT_CAP for value in spread)

    shapes = {tuple(sorted((v for v in one.sp if v), reverse=True))
              for one in ranker_slots()}
    made = {tuple(sorted((v for v in spread if v), reverse=True))
            for spread in templates}
    assert made <= shapes, "no shape is invented that nobody used"


def test_every_operator_does_something_most_of_the_time(dex, regulation, parties):
    """A silent no-op operator would waste a generation's budget without
    showing up anywhere."""
    draw = cursor(5)
    tried = collections.Counter()
    moved = collections.Counter()
    for round_number in range(400):
        team = parties[round_number % len(parties)].team
        name, out = mut.mutate(dex, regulation, team, draw)
        tried[name] += 1
        moved[name] += out != team

    for name, _, _ in mut.OPERATORS:
        assert tried[name] > 0, f"{name} was never drawn"
        # Ability is the exception worth allowing: plenty of species have only
        # one the regulation will register.
        floor = 0.6 if name == "ability" else 0.9
        assert moved[name] >= floor * tried[name], (name, moved[name], tried[name])


def test_a_species_swap_does_not_break_the_species_clause(dex, regulation, parties):
    draw = cursor(13)
    for round_number in range(60):
        team = parties[round_number % len(parties)].team
        out = mut.mutate_species(dex, regulation, team, draw)
        errors = [e for e in team_errors(dex, regulation, out) if "species clause" in e]
        assert not errors, errors


def test_an_item_swap_does_not_break_the_item_clause(dex, regulation, parties):
    draw = cursor(17)
    for round_number in range(60):
        team = parties[round_number % len(parties)].team
        out = mut.mutate_item(dex, regulation, team, draw)
        errors = [e for e in team_errors(dex, regulation, out) if "item clause" in e]
        assert not errors, errors
