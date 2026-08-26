"""The places Champions differs from the mainline series.

Every assertion here was wrong in this engine until Showdown's Champions mod
(``data/mods/champions/``) was read. They are the cases that cannot be guessed
from knowing Pokemon -- which is exactly why they are pinned.
"""

from __future__ import annotations

import pytest

ROOT = __import__('pathlib').Path(__file__).resolve().parents[1]

from pkcm.data.dex import CHAMPIONS_MAX_BASE_PP, load_dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import make_context, step
from pkcm.engine.conditions import FREEZE_DURATION, PARALYSIS_CHANCE, SLEEP_DURATIONS, THAW_CHANCE
from pkcm.engine.legality import clause_violation, learnable_moves, set_errors
from pkcm.engine.pokemon import PokemonSet, max_pp
from pkcm.engine.stats import get_nature
from pkcm.engine.state import BattleConfig, new_battle
from pkcm.data.dex import Stat


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def regulation(dex):
    return dex.regulation("m_b")


# --------------------------------------------------------------------------- #
# The move pool
# --------------------------------------------------------------------------- #


def test_champions_cuts_a_quarter_of_the_move_list(dex):
    pool = [m for m in dex.moves.values() if dex.exists_in_champions(m)]
    assert len(pool) == 500, "base data calls 685 standard; Champions has 500"


@pytest.mark.parametrize("move_id", ["tackle", "growl", "ember", "crushgrip", "scratch"])
def test_moves_champions_removed_do_not_exist(dex, move_id):
    assert not dex.exists_in_champions(dex.moves[move_id])


def test_removed_moves_are_not_learnable(dex):
    assert "tackle" not in learnable_moves(dex, "snorlax")
    assert "bodyslam" in learnable_moves(dex, "snorlax")


def test_champions_renumbered_moves(dex):
    """A sample of the ~30 moves whose numbers Champions changed."""
    assert dex.moves["beakblast"].base_power == 120
    assert dex.moves["firstimpression"].base_power == 100
    assert dex.moves["geargrind"].base_power == 60
    assert dex.moves["geargrind"].accuracy == 90
    assert dex.moves["crabhammer"].accuracy == 95
    assert dex.moves["growth"].type == "grass", "Normal in the mainline games"


# --------------------------------------------------------------------------- #
# PP
# --------------------------------------------------------------------------- #


def test_base_pp_is_capped_at_twenty(dex):
    assert max(move.pp for move in dex.moves.values()) == CHAMPIONS_MAX_BASE_PP
    assert dex.moves["thunderbolt"].pp == 15
    assert dex.moves["protect"].pp == 5, "Champions gives Protect 5 PP, not 10"


def test_pp_formula_is_not_the_series_one(dex):
    """``(base // 5 + 1) * 4``, on a base capped at 20 (hk, confirmed)."""
    """Champions: (pp/5 + 1) * 4. The mainline games: pp * 8/5."""
    assert max_pp(15) == 16, "the series would give 24"
    assert max_pp(10) == 12, "the series would give 16"
    assert max_pp(20) == 20, "the series would give 32"
    assert max_pp(5) == 8


# --------------------------------------------------------------------------- #
# Status conditions
# --------------------------------------------------------------------------- #


def test_paralysis_is_one_in_eight(dex):
    assert PARALYSIS_CHANCE == (1, 8), "the series uses 1/4"


def test_sleep_lasts_two_or_three_turns(dex):
    assert SLEEP_DURATIONS == (2, 3, 3), "the series rolls 1-3"


def test_freeze_has_a_hard_timer(dex):
    assert FREEZE_DURATION == 3
    assert THAW_CHANCE == (1, 4), "the series uses 1/5 with no timer"


def test_a_frozen_pokemon_always_thaws_within_three_attempts(dex):
    """The timer, not the 1/4 roll, is what guarantees the thaw.

    Driven through the hook rather than through battles: whether a frozen
    Pokemon survives long enough to thaw depends on the battle, and that is not
    what this is testing.
    """
    from pkcm.engine import effects as fx

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"))
    team = tuple(
        PokemonSet(species=s, ability="__test__", moves=("bodyslam",))
        for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam", "dragonite")
    )
    move = dex.moves["bodyslam"]

    for seed in range(50):
        state = new_battle(config, (team, team), seed=seed)
        state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
        ctx = make_context(state)
        state.sides[0].status[0] = "frz"
        state.sides[0].status_data[0] = {"turns": FREEZE_DURATION}

        attempts = 0
        while state.sides[0].status[0] == "frz":
            fx.allows(ctx, "try_move", (0, 0), move=move)
            attempts += 1
            assert attempts <= FREEZE_DURATION, f"seed {seed}: never thawed"


# --------------------------------------------------------------------------- #
# Clauses from mods/champions/rulesets.ts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "move_id,clause",
    [
        ("spore", "sleep moves clause"),
        ("hypnosis", "sleep moves clause"),
        ("sheercold", "OHKO clause"),
        ("fissure", "OHKO clause"),
        ("doubleteam", "evasion clause"),
        ("minimize", "evasion clause"),
    ],
)
def test_banned_move_categories(dex, move_id, clause):
    assert clause_violation(dex.moves[move_id]) == clause


def test_ordinary_moves_pass_the_clauses(dex):
    for move_id in ("thunderbolt", "swordsdance", "protect", "willowisp", "reflect"):
        assert clause_violation(dex.moves[move_id]) is None


def test_a_banned_move_makes_a_set_illegal(dex, regulation):
    bad = PokemonSet(species="venusaur", ability="overgrow", moves=("sludgebomb", "spore"))
    errors = set_errors(dex, regulation, bad)
    assert any("sleep moves clause" in e for e in errors), errors


def test_a_removed_move_makes_a_set_illegal(dex, regulation):
    bad = PokemonSet(species="snorlax", ability="thickfat", moves=("bodyslam", "tackle"))
    errors = set_errors(dex, regulation, bad)
    assert any("does not exist in Champions" in e for e in errors), errors


# --------------------------------------------------------------------------- #
# Stat arithmetic
# --------------------------------------------------------------------------- #


def test_nature_uses_integer_arithmetic(dex):
    """Showdown does ``trunc(stat * 110 / 100)``; a float multiply can disagree."""
    from pkcm.engine.stats import compute_stat

    jolly = get_nature("jolly")
    for base in range(1, 256):
        for sp in (0, 16, 32):
            expected = (base + 20 + sp) * 110 // 100
            assert compute_stat(base, sp, jolly, Stat.SPE) == expected


# --------------------------------------------------------------------------- #
# The Champions learnset table
#
# Built from the 포케챔스 dex, which hk found. Everything before it derived
# learnsets from Showdown's all-generations record of the main series -- the
# wrong table, and wrong in both directions.
# --------------------------------------------------------------------------- #


def test_every_legal_species_uses_the_champions_table(dex):
    """No species may quietly keep the old union."""
    from pkcm.engine.legality import _champions_entry

    regulation = dex.regulation("m_b")
    missing = [species for species in sorted(regulation.legal_species | regulation.legal_megas)
               if _champions_entry(dex, species) is None]
    assert not missing, f"still on the all-generations union: {missing}"


def test_clefable_lost_its_gen_one_tm_moves(dex):
    """The clearest case: Soft-Boiled has no legal user, so Champions has no
    Soft-Boiled. Chansey and Blissey are not in the roster (hk), and Clefable
    only ever had it from a TM the game no longer offers."""
    from pkcm.engine.legality import learnable_moves

    assert "softboiled" not in learnable_moves(dex, "clefable")
    for gone in ("seismictoss", "zapcannon", "dynamicpunch", "toxic"):
        assert gone not in learnable_moves(dex, "clefable"), gone


def test_the_table_also_adds_what_the_union_missed(dex):
    """It is not a subset. Egg and tutor moves the union never carried."""
    from pkcm.engine.legality import learnable_moves

    clefable = learnable_moves(dex, "clefable")
    for gained in ("wish", "healpulse", "airslash", "tickle"):
        assert gained in clefable, gained

    # Sing is on its row too, and the sleep clause still keeps it off the team --
    # which is the separation the next test is about.
    from pkcm.engine.legality import champions_learnsets

    assert "sing" in champions_learnsets()["clefable"]
    assert "sing" not in clefable


def test_clauses_still_apply_on_top_of_the_table(dex):
    """Taught and allowed are different questions, and the table answers one.

    Hypnosis is on plenty of rows; the sleep clause keeps it off every team.
    """
    from pkcm.engine.legality import champions_learnsets, learnable_moves

    table = champions_learnsets()
    teaches_hypnosis = [s for s, moves in table.items() if "hypnosis" in moves]
    assert teaches_hypnosis, "the table should carry banned moves too"
    for species in teaches_hypnosis[:5]:
        if species in dex.species:
            assert "hypnosis" not in learnable_moves(dex, species)


def test_no_species_learns_nothing(dex):
    """A row that came back empty means a mapping went wrong upstream."""
    from pkcm.engine.legality import learnable_moves

    regulation = dex.regulation("m_b")
    empty = [s for s in regulation.legal_species if not learnable_moves(dex, s)]
    assert not empty, empty


def test_battle_bond_is_banned_rather_than_absent(dex):
    """hk: the data has it, the ruleset forbids it.

    That distinction decides which layer the fix belongs in. Deleting it from
    ``Species.abilities`` would put a *rule* in the data layer, and the engine
    would then have no way to say why the ability is unavailable -- the same
    separation the move clauses keep (docs/DESIGN.md §1g).
    """
    from pkcm.engine.legality import ability_clause, registrable_abilities

    assert "battlebond" in dex.species["greninja"].abilities, "the data still has it"
    assert ability_clause("battlebond") == "battle bond clause"
    assert registrable_abilities(dex.species["greninja"]) == ("torrent", "protean")


def test_a_battle_bond_greninja_is_an_illegal_set(dex):
    from pkcm.engine.legality import set_errors
    from pkcm.engine.pokemon import PokemonSet

    regulation = dex.regulation("m_b")
    bonded = PokemonSet(species="greninja", ability="battlebond", moves=("surf",),
                        item=None, nature="serious", sp=(0, 0, 0, 0, 0, 0))
    errors = set_errors(dex, regulation, bonded)
    assert any("banned" in error for error in errors), errors


def test_our_abilities_match_the_pokechams_dex_once_bans_are_taken_out(dex):
    """The two readings agree on all 316 species, Battle Bond aside.

    That is the check that made the ban worth believing: one disagreement out
    of 316 says their list includes hidden abilities rather than omitting them.
    """
    from pkcm.data.dex import champions_species_abilities
    from pkcm.engine.legality import registrable_abilities

    theirs = champions_species_abilities()
    assert theirs, "run scripts/build_champions_learnsets.py"
    mismatched = []
    for species_id, their_abilities in theirs.items():
        if species_id not in dex.species:
            continue
        ours = set(registrable_abilities(dex.species[species_id]))
        if ours != set(their_abilities):
            mismatched.append((species_id, sorted(ours), sorted(their_abilities)))
    assert not mismatched, mismatched


def test_no_random_team_carries_an_ability_the_game_lacks(dex):
    from pkcm.engine.legality import random_team, set_errors
    from pkcm.engine.rng import Rng

    regulation = dex.regulation("m_b")
    for seed in range(60):
        for pokemon in random_team(dex, regulation, Rng.from_seed(seed).cursor()):
            assert pokemon.ability in dex.species[pokemon.species].abilities, pokemon
            assert not set_errors(dex, regulation, pokemon), pokemon


def test_the_two_slug_alias_tables_agree():
    """The builder and the report each carry one; they have to say the same."""
    import importlib.util

    def load(name):
        path = ROOT / "scripts" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    builder = load("build_champions_learnsets")
    report = load("compare_pokechams")
    assert builder.SLUG_ALIASES == report.SLUG_ALIASES
