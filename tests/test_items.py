"""Held items, from the roster op.gg says Champions actually has."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import Action
from pkcm.engine.battle import make_context, step
from pkcm.engine.effects import REGISTRY, registered
from pkcm.engine.items import champions_items
from pkcm.engine.legality import is_legal_team, random_team, set_errors, team_errors
from pkcm.engine.moves import compute_damage, use_move
from pkcm.engine.mutate import effective_stat
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, legal_actions, new_battle

RED, BLUE = (0, 0), (1, 0)


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")


@pytest.fixture(scope="module")
def regulation(dex):
    return dex.regulation("m_b")


def a_set(species, ability="__none__", moves=("bodyslam",), item=None, **kwargs):
    return PokemonSet(species=species, ability=ability, moves=tuple(moves), item=item,
                      **{"nature": "serious", "sp": (0, 0, 0, 0, 0, 0), **kwargs})


def build(config, red, blue):
    bench = [a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(config, (tuple([red] + bench), tuple([blue] + bench)), seed=7)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def cast(ctx, dex, move_id, attacker=RED, defender=BLUE):
    use_move(ctx, attacker, dex.moves[move_id], defender=defender)


# --------------------------------------------------------------------------- #
# The roster itself
# --------------------------------------------------------------------------- #


def test_champions_kept_only_one_choice_item(dex):
    """A striking cut, and both of our sources agree on it."""
    roster = champions_items()
    assert "choicescarf" in roster
    assert "choiceband" not in roster
    assert "choicespecs" not in roster


@pytest.mark.parametrize("item_id", [
    "assaultvest", "eviolite", "rockyhelmet", "airballoon", "flameorb", "toxicorb",
    "heavydutyboots", "weaknesspolicy", "boosterenergy", "covertcloak",
])
def test_items_champions_does_not_have(dex, item_id):
    assert item_id in dex.items, "the item exists upstream"
    assert item_id not in champions_items(), "but not in Champions"


def test_holding_a_cut_item_is_illegal(dex, regulation):
    bad = a_set("snorlax", "thickfat", ("bodyslam",), item="choiceband")
    errors = set_errors(dex, regulation, bad)
    assert any("does not exist in Champions" in e for e in errors), errors


def test_every_roster_item_is_implemented(dex):
    """Mega Stones included -- they are registered even though Megas are not."""
    missing = sorted(champions_items() - set(registered("item")))
    assert missing == []


def test_no_item_is_registered_without_doing_anything(dex):
    """Except Shed Shell, which the turn loop reads directly."""
    silent = sorted(
        item_id for (kind, item_id), effect in REGISTRY.items()
        if kind == "item" and not effect.handlers
        and item_id in champions_items() and not dex.items[item_id].mega_stone
    )
    assert silent == ["shedshell"]


# --------------------------------------------------------------------------- #
# Power items
# --------------------------------------------------------------------------- #


def test_type_boosting_item_only_boosts_its_type(dex, config):
    def damage(item, move_id):
        state = build(config, a_set("garchomp", "roughskin", ("earthquake", "dragonclaw"),
                                    item=item), a_set("snorlax"))
        return compute_damage(make_context(state), RED, BLUE, dex.moves[move_id],
                              crit=False)[0]

    assert damage("softsand", "earthquake") > damage(None, "earthquake")
    assert damage("softsand", "dragonclaw") == damage(None, "dragonclaw")


def test_life_orb_boosts_and_costs_hp(dex, config):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",), item="lifeorb"),
                  a_set("snorlax"))
    plain = build(config, a_set("garchomp", "roughskin", ("earthquake",)), a_set("snorlax"))
    assert compute_damage(make_context(state), RED, BLUE, dex.moves["earthquake"], False)[0] > \
        compute_damage(make_context(plain), RED, BLUE, dex.moves["earthquake"], False)[0]

    full = state.pokemon(0, 0).max_hp
    ctx = make_context(state)
    cast(ctx, dex, "earthquake")
    assert state.sides[0].hp[0] == full - max(1, full // 10)


def test_expert_belt_only_on_super_effective(dex, config):
    def damage(item, move_id, defender_species):
        state = build(config, a_set("garchomp", "roughskin", ("earthquake", "dragonclaw"),
                                    item=item), a_set(defender_species))
        return compute_damage(make_context(state), RED, BLUE, dex.moves[move_id],
                              crit=False)[0]

    # Ground is super effective on Nidoking (Poison/Ground), neutral on Snorlax.
    assert damage("expertbelt", "earthquake", "nidoking") > \
        damage(None, "earthquake", "nidoking")
    assert damage("expertbelt", "earthquake", "snorlax") == \
        damage(None, "earthquake", "snorlax")


def test_light_ball_is_pikachu_only(dex, config):
    charged = build(config, a_set("pikachu", "static", ("thunderbolt",), item="lightball"),
                    a_set("snorlax"))
    plain = build(config, a_set("pikachu", "static", ("thunderbolt",)), a_set("snorlax"))
    ctx, ctx2 = make_context(charged), make_context(plain)
    assert effective_stat(ctx, RED, Stat.SPA) == effective_stat(ctx2, RED, Stat.SPA) * 2

    other = build(config, a_set("raichu", "static", ("thunderbolt",), item="lightball"),
                  a_set("snorlax"))
    bare = build(config, a_set("raichu", "static", ("thunderbolt",)), a_set("snorlax"))
    assert effective_stat(make_context(other), RED, Stat.SPA) == \
        effective_stat(make_context(bare), RED, Stat.SPA)


# --------------------------------------------------------------------------- #
# Choice Scarf
# --------------------------------------------------------------------------- #


def test_choice_scarf_raises_speed_and_locks_the_move(dex, config):
    state = build(config,
                  a_set("garchomp", "roughskin", ("earthquake", "dragonclaw"), item="choicescarf"),
                  a_set("snorlax"))
    plain = build(config, a_set("garchomp", "roughskin", ("earthquake", "dragonclaw")),
                  a_set("snorlax"))
    assert effective_stat(make_context(state), RED, Stat.SPE) > \
        effective_stat(make_context(plain), RED, Stat.SPE)

    assert len([a for a in legal_actions(state, 0) if a.kind.name == "MOVE"]) == 2
    state, _ = step(state, Action.move(0), Action.move(0))
    moves = [a for a in legal_actions(state, 0) if a.kind.name == "MOVE"]
    assert [a.index for a in moves] == [0], "locked into the move it chose"


def test_the_choice_lock_lifts_on_a_switch(dex, config):
    state = build(config,
                  a_set("garchomp", "roughskin", ("earthquake", "dragonclaw"), item="choicescarf"),
                  a_set("snorlax"))
    state, _ = step(state, Action.move(0), Action.move(0))
    state, _ = step(state, Action.switch(1), Action.move(0))
    state, _ = step(state, Action.switch(0), Action.move(0))
    moves = [a for a in legal_actions(state, 0) if a.kind.name == "MOVE"]
    assert [a.index for a in moves] == [0, 1], "free again"


# --------------------------------------------------------------------------- #
# Survival
# --------------------------------------------------------------------------- #


def test_focus_sash_survives_once_and_is_used_up(dex, config):
    state = build(config, a_set("alakazam", "synchronize", ("psychic",), item="focussash"),
                  a_set("garchomp", "roughskin", ("earthquake",)))
    ctx = make_context(state)
    cast(ctx, dex, "earthquake", attacker=BLUE, defender=RED)

    assert state.sides[0].hp[0] == 1, "survives on one HP"
    assert state.item_id(0, 0) is None, "and the sash is gone"
    assert any(e.kind == "item_used" for e in ctx.log)


def test_focus_sash_does_nothing_below_full_health(dex, config):
    state = build(config, a_set("alakazam", "synchronize", ("psychic",), item="focussash"),
                  a_set("garchomp", "roughskin", ("earthquake",)))
    state.sides[0].hp[0] -= 1
    ctx = make_context(state)
    cast(ctx, dex, "earthquake", attacker=BLUE, defender=RED)
    assert state.sides[0].hp[0] == 0


def test_leftovers_heals_at_end_of_turn(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("protect",), item="leftovers"),
                  a_set("pikachu", "static", ("protect",)))
    full = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = full // 2
    before = state.sides[0].hp[0]
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] == before + max(1, full // 16)
    assert any(e.kind == "heal" and e.detail == "leftovers" for e in log)


# --------------------------------------------------------------------------- #
# Berries
# --------------------------------------------------------------------------- #


def test_resist_berry_halves_one_super_effective_hit(dex, config):
    """Yache Berry against Ice, then gone."""
    def damage(item):
        state = build(config, a_set("weavile", "pressure", ("icebeam",)),
                      a_set("garchomp", "roughskin", ("earthquake",), item=item))
        ctx = make_context(state)
        cast(ctx, dex, "icebeam")
        return state.pokemon(1, 0).max_hp - state.sides[1].hp[0], state, ctx

    with_berry, state, ctx = damage("yacheberry")
    without, _, _ = damage(None)
    assert with_berry < without * 0.6
    assert state.item_id(1, 0) is None, "eaten"


def test_resist_berry_ignores_a_neutral_hit(dex, config):
    def damage(item):
        state = build(config, a_set("garchomp", "roughskin", ("dragonclaw",)),
                      a_set("snorlax", "thickfat", ("bodyslam",), item=item))
        ctx = make_context(state)
        return compute_damage(ctx, RED, BLUE, dex.moves["dragonclaw"], crit=False)[0]

    assert damage("chopleberry") == damage(None)


def test_status_berry_cures_its_status(dex, config):
    # Not Protect: it would block Will-O-Wisp and there would be no burn to cure.
    state = build(config, a_set("snorlax", "thickfat", ("bodyslam",), item="rawstberry"),
                  a_set("gengar", "cursedbody", ("willowisp",)))
    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "status" and e.detail == "brn" for e in log), "it did get burned"
    assert state.sides[0].status[0] is None, "the burn was cured"
    assert state.item_id(0, 0) is None
    assert any(e.kind == "item_used" and e.detail == "rawstberry" for e in log)


def test_sitrus_berry_heals_at_half_health(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("protect",), item="sitrusberry"),
                  a_set("pikachu", "static", ("protect",)))
    full = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = full // 2 - 1
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] > full // 2
    assert state.item_id(0, 0) is None


# --------------------------------------------------------------------------- #
# Field extenders and herbs
# --------------------------------------------------------------------------- #


def test_weather_rock_extends_only_its_weather(dex, config):
    state = build(config, a_set("politoed", "drizzle", ("raindance", "sunnyday"),
                                item="damprock"), a_set("snorlax"))
    ctx = make_context(state)
    cast(ctx, dex, "raindance")
    assert state.field.weather_turns == 8, "5 + 3"

    state.field.weather = None
    cast(ctx, dex, "sunnyday")
    assert state.field.weather_turns == 5, "Damp Rock is for rain only"


def test_light_clay_extends_screens(dex, config):
    state = build(config, a_set("clefable", "unaware", ("lightscreen",), item="lightclay"),
                  a_set("snorlax"))
    ctx = make_context(state)
    cast(ctx, dex, "lightscreen")
    assert state.sides[0].conditions["lightscreen"] == 8


def test_white_herb_restores_lowered_stats(dex, config):
    state = build(config, a_set("typhlosion", "blaze", ("overheat",), item="whiteherb"),
                  a_set("snorlax"))
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].boost(0, "spa") == 0, "Overheat's drop was undone"
    assert state.item_id(0, 0) is None


# --------------------------------------------------------------------------- #
# Team building
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(15))
def test_random_teams_hold_legal_distinct_items(dex, regulation, seed):
    team = random_team(dex, regulation, Rng.from_seed(seed).cursor())
    assert is_legal_team(dex, regulation, team), team_errors(dex, regulation, team)

    items = [pokemon.item for pokemon in team]
    assert all(item is not None for item in items), "every slot holds something"
    assert len(set(items)) == len(items), "item clause"
    assert all(item in champions_items() for item in items)

    # A Mega Stone is only ever handed to a Pokemon that can actually use it.
    for pokemon in team:
        if dex.items[pokemon.item].mega_stone:
            assert dex.mega_evolution(pokemon.species, pokemon.item) is not None, (
                f"{pokemon.species} cannot use {pokemon.item}"
            )


def test_zoom_lens_only_helps_when_moving_second(dex, config):
    """The op.gg item scrape does not list it, so it was never implemented.

    The pokechams dex does list it, which is the whole argument for keeping two
    sources: one of them being short an item is invisible until something else
    counts them.
    """
    from pkcm.engine.moves import _both_sides

    state = build(config, a_set("pikachu", "static", ("thunder",), item="zoomlens"),
                  a_set("snorlax", "thickfat"))
    move = dex.moves["thunder"]

    early = make_context(state)
    plain = _both_sides(early, "modify_accuracy", float(move.accuracy), RED, BLUE, move)

    late = make_context(state)
    late.acted.add(BLUE)          # the target has already taken its turn
    boosted = _both_sides(late, "modify_accuracy", float(move.accuracy), RED, BLUE, move)

    assert boosted > plain
    assert plain == float(move.accuracy), "no bonus while the target still has its turn"
