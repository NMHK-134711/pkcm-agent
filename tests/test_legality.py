"""Team legality, with the Species Clause's forme collision as the centerpiece."""

from __future__ import annotations

import pytest

from pkcm.data.dex import load_dex
from pkcm.engine.legality import (
    base_species_of,
    is_legal_team,
    learnable_moves,
    random_team,
    set_errors,
    team_errors,
)
from pkcm.engine.pokemon import PokemonSet, compile_team
from pkcm.engine.rng import Rng


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def regulation(dex):
    return dex.regulation("m_b")


def make_set(dex, species_id: str, **overrides) -> PokemonSet:
    """A minimal legal set for ``species_id``, before overrides."""
    species = dex.species[species_id]
    pool = sorted(learnable_moves(dex, species_id))
    defaults = dict(
        species=species_id,
        ability=species.abilities[0],
        moves=tuple(pool[:4]),
        nature="serious",
        sp=(0, 0, 0, 0, 0, 0),
        item=None,
    )
    return PokemonSet(**{**defaults, **overrides})


def filler(dex, count: int, exclude: set[str]) -> list[PokemonSet]:
    """Legal, distinct-base-species padding so a team reaches six."""
    picks = []
    for species_id in ("pikachu", "snorlax", "gyarados", "dragonite", "tyranitar", "skarmory",
                       "gengar", "starmie"):
        if len(picks) == count:
            break
        if base_species_of(dex, species_id) in exclude:
            continue
        exclude.add(base_species_of(dex, species_id))
        picks.append(make_set(dex, species_id))
    assert len(picks) == count
    return picks


# --------------------------------------------------------------------------- #
# Species Clause
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "first,second",
    [
        ("goodra", "goodrahisui"),          # the case that prompted this rule
        ("arcanine", "arcaninehisui"),
        ("taurospaldeacombat", "taurospaldeablaze"),
        ("meowstic", "meowsticf"),
        ("gourgeist", "gourgeistsuper"),
        ("lycanroc", "lycanrocdusk"),
    ],
)
def test_regional_and_alternate_formes_collide(dex, regulation, first, second):
    assert base_species_of(dex, first) == base_species_of(dex, second)

    team = tuple(
        [make_set(dex, first), make_set(dex, second)]
        + filler(dex, 4, {base_species_of(dex, first)})
    )
    errors = team_errors(dex, regulation, team)
    assert any("species clause" in e for e in errors), errors


def test_exact_duplicate_is_caught(dex, regulation):
    team = tuple([make_set(dex, "snorlax"), make_set(dex, "snorlax")] + filler(dex, 4, {"snorlax"}))
    errors = team_errors(dex, regulation, team)
    assert any("appears twice" in e for e in errors), errors


def test_different_base_species_coexist(dex, regulation):
    team = tuple(filler(dex, 6, set()))
    assert is_legal_team(dex, regulation, team), team_errors(dex, regulation, team)


def test_species_clause_groups_agree_with_dex_numbers(dex, regulation):
    """Base species and National Dex number must partition the roster identically."""
    by_base, by_num = {}, {}
    for species_id in regulation.legal_species:
        by_base.setdefault(base_species_of(dex, species_id), set()).add(species_id)
        by_num.setdefault(dex.species[species_id].dex_num, set()).add(species_id)
    assert sorted(by_base.values(), key=sorted) == sorted(by_num.values(), key=sorted)


def test_the_roster_actually_contains_colliding_formes(dex, regulation):
    groups: dict[str, list[str]] = {}
    for species_id in regulation.legal_species:
        groups.setdefault(base_species_of(dex, species_id), []).append(species_id)
    colliding = {k: v for k, v in groups.items() if len(v) > 1}
    assert len(colliding) == 18, "M-B has 18 base species with multiple legal formes"
    assert set(colliding["goodra"]) == {"goodra", "goodrahisui"}


# --------------------------------------------------------------------------- #
# Other clauses and per-set rules
# --------------------------------------------------------------------------- #


def test_item_clause(dex, regulation):
    team = list(filler(dex, 6, set()))
    team[0] = team[0].replace(item="leftovers")
    team[1] = team[1].replace(item="leftovers")
    errors = team_errors(dex, regulation, tuple(team))
    assert any("item clause" in e for e in errors), errors


def test_holding_nothing_may_repeat(dex, regulation):
    team = tuple(filler(dex, 6, set()))  # every slot holds nothing
    assert not any("item clause" in e for e in team_errors(dex, regulation, team))


def test_team_size(dex, regulation):
    team = tuple(filler(dex, 5, set()))
    assert any("registers 6" in e for e in team_errors(dex, regulation, team))


def test_unlearnable_move(dex, regulation):
    pool = learnable_moves(dex, "snorlax")
    unlearnable = next(
        move_id
        for move_id, move in sorted(dex.moves.items())
        if move.raw.get("isNonstandard") is None and move_id not in pool
    )
    bad = make_set(dex, "snorlax", moves=("bodyslam", unlearnable))
    errors = set_errors(dex, regulation, bad)
    assert any("cannot learn" in e and unlearnable in e for e in errors), errors


def test_learnsets_are_the_all_generation_union(dex):
    """Documented approximation: Champions' own learnsets are not published.

    Snorlax learned Hydro Pump from a Gen 1 TM and never since, yet we accept it.
    This test pins the permissiveness so tightening it is a deliberate change,
    not an accident.
    """
    assert "hydropump" in learnable_moves(dex, "snorlax")


def test_illegal_ability(dex, regulation):
    bad = make_set(dex, "snorlax", ability="levitate")
    assert any("ability" in e for e in set_errors(dex, regulation, bad))


def test_sp_over_budget(dex, regulation):
    bad = make_set(dex, "snorlax", sp=(32, 32, 32, 0, 0, 0))
    assert any("budget" in e for e in set_errors(dex, regulation, bad))


def test_mega_formes_cannot_be_registered(dex, regulation):
    bad = make_set(dex, "venusaur").replace(species="venusaurmega")
    errors = set_errors(dex, regulation, bad)
    assert any("Mega formes are reached in battle" in e for e in errors), errors


def test_ineligible_species(dex, regulation):
    assert "mewtwo" not in regulation.legal_species
    bad = make_set(dex, "mewtwo")
    assert any("not eligible" in e for e in set_errors(dex, regulation, bad))


def test_learnsets_fall_back_to_base_forme(dex):
    """Gourgeist's size formes carry no learnset of their own."""
    for species_id in ("gourgeistlarge", "gourgeistsmall"):
        assert species_id not in dex.learnsets
        assert len(learnable_moves(dex, species_id)) > 50


def test_nonstandard_moves_are_excluded(dex):
    """Z-Moves and cut moves must not reach a team builder."""
    pool = learnable_moves(dex, "pikachu")
    assert "thunderbolt" in pool
    assert "hiddenpower" not in pool
    assert all(dex.moves[m].raw.get("isNonstandard") is None for m in pool)


# --------------------------------------------------------------------------- #
# Random teams
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(25))
def test_random_teams_are_always_legal(dex, regulation, seed):
    team = random_team(dex, regulation, Rng.from_seed(seed).cursor())
    assert is_legal_team(dex, regulation, team), team_errors(dex, regulation, team)


def test_random_teams_are_reproducible(dex, regulation):
    a = random_team(dex, regulation, Rng.from_seed(42).cursor())
    b = random_team(dex, regulation, Rng.from_seed(42).cursor())
    c = random_team(dex, regulation, Rng.from_seed(43).cursor())
    assert a == b
    assert a != c


def test_random_teams_compile_and_can_attack(dex, regulation):
    team = random_team(dex, regulation, Rng.from_seed(1).cursor())
    for pokemon in compile_team(dex, team):
        assert pokemon.max_hp > 0
        assert 1 <= len(pokemon.moves) <= 4
        assert any(move.base_power > 0 for move in pokemon.moves), "battles must be able to end"


# --------------------------------------------------------------------------- #
# Learnset inheritance -- two shapes that look alike and must not be conflated
# --------------------------------------------------------------------------- #


def test_changesfrom_formes_inherit_their_source(dex):
    """Rotom-Heat's own entry holds one move; the rest comes from Rotom."""
    assert dex.species["rotomheat"].changes_from == "rotom"
    assert len(dex.learnsets["rotomheat"]["learnset"]) == 1

    pool = learnable_moves(dex, "rotomheat")
    assert "overheat" in pool, "its signature move"
    assert "thunderbolt" in pool, "inherited from Rotom"
    assert len(pool) > 40


def test_regional_formes_do_not_inherit(dex):
    """Alolan Raichu carries a complete pool and must not gain Kantonian moves."""
    assert dex.species["raichualola"].changes_from is None
    kantonian_only = learnable_moves(dex, "raichu") - learnable_moves(dex, "raichualola")
    assert kantonian_only, "the two pools must differ, i.e. no union happened"


def test_terastal_moves_are_never_legal(dex, regulation):
    """Champions has no Terastallization, so Tera Blast cannot exist."""
    assert "terablast" in dex.moves
    for species_id in ("garchomp", "pikachu", "snorlax"):
        assert "terablast" not in learnable_moves(dex, species_id)
        assert "terastarstorm" not in learnable_moves(dex, species_id)


# --------------------------------------------------------------------------- #
# Generator respects the engine's capabilities
# --------------------------------------------------------------------------- #


def test_generated_moves_are_all_executable(dex, regulation):
    from pkcm.engine.scope import is_supported

    for seed in range(10):
        for pokemon_set in random_team(dex, regulation, Rng.from_seed(seed).cursor()):
            for move_id in pokemon_set.moves:
                assert is_supported(dex.moves[move_id]), f"{move_id} cannot be executed yet"


def test_species_with_nothing_playable_are_skipped(dex, regulation):
    """Ditto is legal in M-B and learns only Transform."""
    from pkcm.engine.legality import usable_moves

    assert "ditto" in regulation.legal_species
    assert usable_moves(dex, "ditto") == []

    drawn = {
        pokemon_set.species
        for seed in range(40)
        for pokemon_set in random_team(dex, regulation, Rng.from_seed(seed).cursor())
    }
    assert "ditto" not in drawn


# --------------------------------------------------------------------------- #
# Teams built out of what people actually brought
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(12))
def test_ranker_teams_are_always_legal(dex, regulation, seed):
    """Species and Item Clause are held while drawing, not checked after."""
    from pkcm.engine.legality import make_team

    team = make_team(dex, regulation, Rng.from_seed(seed).cursor(),
                     "singles", "ranker")
    assert team_errors(dex, regulation, team, "singles") == []


def test_the_ranker_pool_is_a_different_distribution(dex, regulation):
    """The point of the pool, in the measure that motivated it.

    A randomly generated Pokemon carries no same-type attack 37.5% of the
    time -- its moves were drawn without reference to its types, and its item
    without reference to anything. These are sets people built. If this ever
    stops being true the pool has been rebuilt from something else.
    """
    from pkcm.engine.legality import make_team

    def without_stab(source: str) -> float:
        total = missing = 0
        for seed in range(40):
            team = make_team(dex, regulation, Rng.from_seed(seed).cursor(),
                             "singles", source)
            for pokemon in team:
                entry = dex.species[pokemon.species]
                attacks = [dex.moves[m] for m in pokemon.moves
                           if dex.moves[m].base_power > 0]
                if not attacks:
                    continue
                total += 1
                if not any(move.type in entry.types for move in attacks):
                    missing += 1
        return missing / max(total, 1)

    assert without_stab("ranker") < 0.15
    assert without_stab("random") > 0.25


def test_an_unknown_team_source_is_refused(dex, regulation):
    """Silently falling back to random would make a run's distribution a
    typo away from being something else entirely."""
    from pkcm.engine.legality import make_team

    with pytest.raises(ValueError):
        make_team(dex, regulation, Rng.from_seed(0).cursor(), "singles", "rankers")


def test_a_party_source_repeats_its_teams(dex, regulation):
    """The curriculum's whole point: matchups have to recur.

    ``ranker`` draws six slots out of a hundred and twenty, so a run never sees
    the same team twice and the policy head is asked to generalise across the
    team space before it can learn to play in any of it. ``parties`` hands back
    whole imported parties, so positions repeat and there is something to
    accumulate.
    """
    from pkcm.engine.legality import make_team, ranker_parties

    def distinct(source, draws=200):
        return {tuple(sorted(pokemon.species for pokemon in make_team(
            dex, regulation, Rng.from_seed(seed).cursor(), "singles", source)))
            for seed in range(draws)}

    assert len(distinct("parties:10,14,17,7")) == 4
    # Pinned to however many are imported, not to a number written here: the
    # pool grows as parties are transcribed, and a literal would make that
    # growth look like a regression.
    assert len(distinct("parties")) <= len(ranker_parties())
    assert len(distinct("ranker")) > 150


def test_a_party_subset_is_checked_when_it_is_written(dex, regulation):
    """Not once a worker is three minutes into a run."""
    from pkcm.engine.legality import parse_team_source, ranker_parties

    assert parse_team_source("parties:0,1") == "parties:0,1"
    assert parse_team_source("ranker") == "ranker"
    for bad in ("parties:%d" % len(ranker_parties()), "parties:x", "rankers"):
        with pytest.raises(ValueError):
            parse_team_source(bad)
