"""The moves Showdown keeps in handler code rather than in fields.

A sample across the shapes: healing, stat manipulation, conditions that stop a
Pokemon doing something, types, abilities, items and the field.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.battle import make_context, step
from pkcm.engine.moves import use_move
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, legal_actions, new_battle

RED, BLUE = (0, 0), (1, 0)


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")


def a_set(species, ability="__none__", moves=("bodyslam",), item=None, **kwargs):
    return PokemonSet(species=species, ability=ability, moves=tuple(moves), item=item,
                      **{"nature": "serious", "sp": (0, 0, 0, 0, 0, 0), **kwargs})


def build(config, red, blue):
    filler = [a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(config, (tuple([red] + filler), tuple([blue] + filler)), seed=7)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def cast(ctx, dex, move_id, attacker=RED, defender=BLUE):
    use_move(ctx, attacker, defender, dex.moves[move_id])


# --------------------------------------------------------------------------- #
# Healing
# --------------------------------------------------------------------------- #


def test_synthesis_heals_more_in_sun(dex, config):
    def healed(weather):
        state = build(config, a_set("venusaur", "overgrow", ("synthesis",)), a_set("snorlax"))
        state.field.weather = weather
        state.sides[0].hp[0] = 1
        ctx = make_context(state)
        cast(ctx, dex, "synthesis")
        return state.sides[0].hp[0]

    assert healed("sunnyday") > healed(None) > healed("raindance")


def test_rest_heals_fully_and_puts_you_to_sleep(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("rest",)), a_set("pikachu"))
    full = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = 10
    ctx = make_context(state)
    cast(ctx, dex, "rest")
    assert state.sides[0].hp[0] == full
    assert state.sides[0].status[0] == "slp"
    assert state.sides[0].status_data[0]["turns"] == 3


def test_pain_split_averages_both(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("painsplit",)),
                  a_set("snorlax", "thickfat"))
    state.sides[0].hp[0] = 10
    state.sides[1].hp[0] = 200
    ctx = make_context(state)
    cast(ctx, dex, "painsplit")
    assert state.sides[0].hp[0] == 105
    assert state.sides[1].hp[0] == 105


def test_wish_arrives_a_turn_later(dex, config):
    state = build(config, a_set("clefable", "unaware", ("wish", "protect")),
                  a_set("pikachu", "static", ("splash",)))
    full = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = 10

    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] == 10, "nothing yet"

    state, log = step(state, Action.move(1), Action.move(0))
    assert state.sides[0].hp[0] == 10 + full // 2
    assert any(e.kind == "heal" and e.detail == "wish" for e in log)


def test_strength_sap_heals_by_the_targets_attack(dex, config):
    state = build(config, a_set("shiinotic", "effectspore", ("strengthsap",)),
                  a_set("garchomp", "roughskin"))
    state.sides[0].hp[0] = 10
    ctx = make_context(state)
    attack = mutate.effective_stat(ctx, BLUE, Stat.ATK)
    cast(ctx, dex, "strengthsap")
    assert state.sides[0].hp[0] == min(state.pokemon(0, 0).max_hp, 10 + attack)
    assert state.sides[1].boost(0, "atk") == -1


def test_roost_sheds_the_flying_type_for_the_turn(dex, config):
    state = build(config, a_set("skarmory", "sturdy", ("roost",)),
                  a_set("pikachu", "static", ("thunderbolt",)))
    assert "flying" in state.types(0, 0)
    state.sides[0].hp[0] = 10

    ctx = make_context(state)
    cast(ctx, dex, "roost")
    assert "flying" not in state.types(0, 0), "grounded while roosting"

    state, _ = step(state, Action.move(0), Action.move(0))
    assert "flying" in state.types(0, 0), "and back to normal next turn"


# --------------------------------------------------------------------------- #
# Stat stages
# --------------------------------------------------------------------------- #


def test_haze_wipes_both_sides(dex, config):
    state = build(config, a_set("gyarados", "intimidate", ("haze",)), a_set("snorlax"))
    ctx = make_context(state)
    mutate.boost(ctx, RED, {"atk": 3})
    mutate.boost(ctx, BLUE, {"def": -2})
    cast(ctx, dex, "haze")
    assert state.sides[0].boost(0, "atk") == 0
    assert state.sides[1].boost(0, "def") == 0


def test_belly_drum_maxes_attack_at_half_your_health(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("bellydrum",)), a_set("pikachu"))
    full = state.pokemon(0, 0).max_hp
    ctx = make_context(state)
    cast(ctx, dex, "bellydrum")
    assert state.sides[0].boost(0, "atk") == 6
    assert state.sides[0].hp[0] == full - full // 2


def test_belly_drum_fails_below_half(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("bellydrum",)), a_set("pikachu"))
    state.sides[0].hp[0] = 10
    ctx = make_context(state)
    cast(ctx, dex, "bellydrum")
    assert state.sides[0].boost(0, "atk") == 0
    assert any(e.kind == "move_failed" for e in ctx.log)


def test_psych_up_copies_and_topsy_turvy_inverts(dex, config):
    state = build(config, a_set("alakazam", "synchronize", ("psychup", "topsyturvy")),
                  a_set("snorlax"))
    ctx = make_context(state)
    mutate.boost(ctx, BLUE, {"atk": 2, "def": -1})

    cast(ctx, dex, "psychup")
    assert state.sides[0].boost(0, "atk") == 2
    assert state.sides[0].boost(0, "def") == -1

    cast(ctx, dex, "topsyturvy")
    assert state.sides[1].boost(0, "atk") == -2
    assert state.sides[1].boost(0, "def") == 1


def test_stockpile_and_swallow(dex, config):
    state = build(config, a_set("swalot", "liquidooze", ("stockpile", "swallow")),
                  a_set("pikachu"))
    full = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = 10
    ctx = make_context(state)

    for expected in (1, 2, 3):
        cast(ctx, dex, "stockpile")
        assert mutate.volatile(state, RED, "stockpile")["layers"] == expected
    assert state.sides[0].boost(0, "def") == 3

    cast(ctx, dex, "swallow")
    assert state.sides[0].hp[0] == min(full, 10 + full)
    assert mutate.volatile(state, RED, "stockpile") is None
    assert state.sides[0].boost(0, "def") == 0, "the layers are paid back"


# --------------------------------------------------------------------------- #
# Conditions that stop you doing something
# --------------------------------------------------------------------------- #


def test_taunt_blocks_status_moves(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("taunt",)),
                  a_set("snorlax", "thickfat", ("swordsdance", "bodyslam")))
    state, _ = step(state, Action.move(0), Action.move(1))
    assert state.sides[1].has_volatile(0, "taunt")

    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "cant_move" and e.detail == "taunt" for e in log), log
    assert state.sides[1].boost(0, "atk") == 0


def test_encore_locks_the_last_move_in(dex, config):
    """Encore needs the target to have moved, so it is used the turn after."""
    state = build(config, a_set("clefable", "unaware", ("encore", "protect")),
                  a_set("snorlax", "thickfat", ("swordsdance", "bodyslam")))
    state, _ = step(state, Action.move(1), Action.move(0))   # let Snorlax move
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].has_volatile(0, "encore")


def test_disable_takes_a_move_away(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("disable", "protect")),
                  a_set("snorlax", "thickfat", ("bodyslam", "rest")))
    state, _ = step(state, Action.move(1), Action.move(0))   # let Snorlax move
    state, _ = step(state, Action.move(0), Action.move(0))
    moves = [a.index for a in legal_actions(state, 1) if a.kind is ActionKind.MOVE]
    assert 0 not in moves, "Body Slam is disabled"
    assert 1 in moves


def test_yawn_puts_them_to_sleep_next_turn(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("yawn", "protect")),
                  a_set("pikachu", "static", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].has_volatile(0, "yawn")
    assert state.sides[1].status[0] is None

    state, _ = step(state, Action.move(1), Action.move(0))
    assert state.sides[1].status[0] == "slp"


def test_perish_song_counts_both_sides_down(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("perishsong", "protect")),
                  a_set("snorlax", "thickfat", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].has_volatile(0, "perishsong")
    assert state.sides[1].has_volatile(0, "perishsong")

    for _ in range(3):
        if state.finished:
            break
        state, _ = step(state, Action.move(1), Action.move(0))
    assert state.sides[0].is_fainted(0) or state.sides[1].is_fainted(0)


def test_curse_is_two_moves_wearing_one_name(dex, config):
    ghost = build(config, a_set("gengar", "cursedbody", ("curse",)), a_set("snorlax"))
    full = ghost.pokemon(0, 0).max_hp
    ctx = make_context(ghost)
    cast(ctx, dex, "curse")
    assert ghost.sides[1].has_volatile(0, "curse"), "a Ghost curses the target"
    assert ghost.sides[0].hp[0] == full - full // 2, "at half its own HP"

    other = build(config, a_set("snorlax", "thickfat", ("curse",)), a_set("pikachu"))
    ctx = make_context(other)
    cast(ctx, dex, "curse")
    assert other.sides[0].boost(0, "atk") == 1, "anything else just boosts"
    assert other.sides[0].boost(0, "spe") == -1
    assert not other.sides[1].has_volatile(0, "curse")


def test_destiny_bond_takes_the_killer_with_it(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("destinybond",)),
                  a_set("garchomp", "roughskin", ("earthquake",)))
    state.sides[0].hp[0] = 1
    ctx = make_context(state)
    cast(ctx, dex, "destinybond")
    state.rng = ctx.cursor.seal()

    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].is_fainted(0)
    assert state.sides[1].is_fainted(0), "and so does whoever did it"


def test_endure_survives_on_one(dex, config):
    state = build(config, a_set("pikachu", "static", ("endure",)),
                  a_set("garchomp", "roughskin", ("earthquake",)))
    ctx = make_context(state)
    cast(ctx, dex, "endure")
    cast(ctx, dex, "earthquake", attacker=BLUE, defender=RED)
    assert state.sides[0].hp[0] == 1


# --------------------------------------------------------------------------- #
# Types, abilities, items
# --------------------------------------------------------------------------- #


def test_soak_makes_them_water(dex, config):
    state = build(config, a_set("starmie", "illuminate", ("soak",)), a_set("snorlax"))
    ctx = make_context(state)
    cast(ctx, dex, "soak")
    assert state.types(1, 0) == ("water",)


def test_trick_or_treat_adds_a_type(dex, config):
    """Ghost-type status move on a Normal target: the chart does not stop it."""
    state = build(config, a_set("gourgeist", "frisk", ("trickortreat",)),
                  a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "trickortreat")
    assert state.types(1, 0) == ("normal", "ghost")


def test_skill_swap_exchanges_abilities(dex, config):
    state = build(config, a_set("alakazam", "synchronize", ("skillswap",)),
                  a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "skillswap")
    assert state.ability_id(0, 0) == "thickfat"
    assert state.ability_id(1, 0) == "synchronize"


def test_gastro_acid_switches_an_ability_off(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("gastroacid",)),
                  a_set("garchomp", "roughskin"))
    ctx = make_context(state)
    assert ctx.ability_of(BLUE) == "roughskin"
    cast(ctx, dex, "gastroacid")
    assert ctx.ability_of(BLUE) is None


def test_trick_swaps_the_held_items(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("trick",), item="lifeorb"),
                  a_set("snorlax", "thickfat", item="leftovers"))
    ctx = make_context(state)
    cast(ctx, dex, "trick")
    assert state.item_id(0, 0) == "leftovers"
    assert state.item_id(1, 0) == "lifeorb"


def test_sticky_hold_refuses_the_trade(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("trick",), item="lifeorb"),
                  a_set("muk", "stickyhold", item="leftovers"))
    ctx = make_context(state)
    cast(ctx, dex, "trick")
    assert state.item_id(0, 0) == "lifeorb", "nothing moved"


def test_fling_throws_the_item(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("fling",), item="ironball"),
                  a_set("snorlax", "thickfat"))
    full = state.pokemon(1, 0).max_hp
    ctx = make_context(state)
    cast(ctx, dex, "fling")
    assert state.sides[1].hp[0] < full
    assert state.item_id(0, 0) is None, "and is gone"


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #


def test_defog_clears_hazards_and_screens(dex, config):
    state = build(config, a_set("crobat", "innerfocus", ("defog",)), a_set("snorlax"))
    state.sides[0].conditions["stealthrock"] = 1
    state.sides[1].conditions["lightscreen"] = 5
    state.sides[1].conditions["spikes"] = 2
    ctx = make_context(state)
    cast(ctx, dex, "defog")
    assert "stealthrock" not in state.sides[0].conditions, "our own side is cleared too"
    assert "lightscreen" not in state.sides[1].conditions
    assert state.sides[1].boost(0, "evasion") == -1


def test_safeguard_blocks_status_from_the_other_side(dex, config):
    """Gengar is faster, so Safeguard has to go up a turn ahead of the burn."""
    state = build(config, a_set("clefable", "unaware", ("safeguard", "protect")),
                  a_set("gengar", "cursedbody", ("splash", "willowisp")))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert "safeguard" in state.sides[0].conditions

    state, _ = step(state, Action.move(1), Action.move(1))
    assert state.sides[0].status[0] is None, "the burn was refused"


def test_gravity_grounds_a_flying_type(dex, config):
    from pkcm.engine.conditions import is_grounded

    state = build(config, a_set("skarmory", "sturdy", ("smackdown",)),
                  a_set("crobat", "innerfocus"))
    assert not is_grounded(state, BLUE)
    ctx = make_context(state)
    cast(ctx, dex, "smackdown")
    assert is_grounded(state, BLUE), "Smack Down pins it down"


def test_magnet_rise_lifts_you_off_the_ground(dex, config):
    from pkcm.engine.conditions import is_grounded

    state = build(config, a_set("magnezone", "sturdy", ("magnetrise",)), a_set("snorlax"))
    assert is_grounded(state, RED)
    ctx = make_context(state)
    cast(ctx, dex, "magnetrise")
    assert not is_grounded(state, RED)


def test_heal_bell_cures_the_whole_team(dex, config):
    state = build(config, a_set("clefable", "unaware", ("healbell",)), a_set("snorlax"))
    state.sides[0].status = ["brn", "par", "slp"]
    ctx = make_context(state)
    cast(ctx, dex, "healbell")
    assert state.sides[0].status == [None, None, None]


def test_ally_only_moves_fail_in_singles(dex, config):
    """Implemented as failing, which is not the same as unimplemented."""
    state = build(config, a_set("clefable", "unaware", ("helpinghand",)), a_set("snorlax"))
    ctx = make_context(state)
    cast(ctx, dex, "helpinghand")
    assert any(e.kind == "move_failed" for e in ctx.log)
    assert not any(e.kind == "unimplemented" for e in ctx.log)
