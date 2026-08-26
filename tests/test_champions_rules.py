"""The places Champions differs from the mainline series.

Every assertion here was wrong in this engine until Showdown's Champions mod
(``data/mods/champions/``) was read. They are the cases that cannot be guessed
from knowing Pokemon -- which is exactly why they are pinned.
"""

from __future__ import annotations

import pytest

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
