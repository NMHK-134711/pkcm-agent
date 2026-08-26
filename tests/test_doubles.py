"""Doubles 6->4: everything that only exists because there are two of them.

The singles suite already covers what a move does. This one covers what having
a partner changes -- targeting, spread damage, redirection, the abilities and
moves that were registered and inert until now, and the turn order questions
that only arise with four Pokemon in the queue.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import TARGET_ALLY, TARGET_SELF, Action, ActionKind
from pkcm.engine.battle import IllegalActionError, make_context, step
from pkcm.engine.moves import use_move
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle

#: Field positions, as refs into a freshly built battle: the leads are party
#: slots 0 and 1 on each side.
RED_A, RED_B = (0, 0), (0, 1)
BLUE_A, BLUE_B = (1, 0), (1, 1)


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="doubles")


@pytest.fixture(scope="module")
def singles(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")


def a_set(species, ability="__none__", moves=("bodyslam",), item=None, **kwargs):
    return PokemonSet(species=species, ability=ability, moves=tuple(moves), item=item,
                      **{"nature": "serious", "sp": (0, 0, 0, 0, 0, 0), **kwargs})


def build(config, red, red2=None, blue=None, blue2=None):
    """A doubles battle with the four named Pokemon already on the field."""
    red2 = red2 or a_set("snorlax", "thickfat")
    blue = blue or a_set("snorlax", "thickfat")
    blue2 = blue2 or a_set("snorlax", "thickfat")
    filler = [a_set(s) for s in ("pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(
        config,
        (tuple([red, red2] + filler), tuple([blue, blue2] + filler)),
        seed=7,
    )
    return step(state, Action.select(0, 1, 2, 3), Action.select(0, 1, 2, 3))[0]


def cast(ctx, dex, move_id, attacker=RED_A, defender=None, target_code=0):
    use_move(ctx, attacker, dex.moves[move_id], target_code=target_code, defender=defender)


# --------------------------------------------------------------------------- #
# The shape of a doubles battle
# --------------------------------------------------------------------------- #


def test_four_are_brought_and_two_stand(config):
    assert config.registered == 6 and config.brought == 4
    assert config.active_count == 2 and config.is_doubles

    state = build(config, a_set("garchomp", "roughskin"))
    assert state.sides[0].active == [0, 1]
    assert len(state.active_refs(0)) == 2
    assert len(state.active_refs(1)) == 2


def test_everyone_can_name_everyone(config):
    state = build(config, a_set("garchomp", "roughskin"))
    assert state.ally(RED_A) == RED_B
    assert state.ally(RED_B) == RED_A
    assert state.foes(RED_A) == [BLUE_A, BLUE_B]
    assert len(state.everyone()) == 4


def test_singles_has_no_ally(singles):
    state = new_battle(
        singles,
        tuple(tuple(a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar",
                                       "alakazam", "garchomp")) for _ in (0, 1)),
        seed=1,
    )
    state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
    assert state.ally((0, 0)) is None
    assert state.sides[0].active == [0]


# --------------------------------------------------------------------------- #
# Targeting
# --------------------------------------------------------------------------- #


def test_a_single_target_move_offers_every_target(config):
    """Two foes and a partner: three ways to point Thunderbolt."""
    state = build(config, a_set("pikachu", "static", ("thunderbolt",)))
    targets = {a.target for a in legal_actions(state, 0, 0)
               if a.kind is ActionKind.MOVE}
    assert targets == {0, 1, TARGET_ALLY}


def test_singles_offers_exactly_one(singles):
    state = new_battle(
        singles,
        (tuple([a_set("pikachu", "static", ("thunderbolt",))]
               + [a_set(s) for s in ("snorlax", "starmie", "gengar", "alakazam", "garchomp")]),
         tuple(a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar",
                                  "alakazam", "garchomp"))),
        seed=1,
    )
    state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
    moves = [a for a in legal_actions(state, 0) if a.kind is ActionKind.MOVE]
    assert len(moves) == 1 and moves[0].target == 0, "nothing to choose between"


def test_a_spread_move_offers_no_choice(config):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)))
    moves = [a for a in legal_actions(state, 0, 0) if a.kind is ActionKind.MOVE]
    assert len(moves) == 1, "Earthquake hits everyone; there is nothing to aim"


def test_the_target_code_picks_the_right_foe(config, dex):
    state = build(config, a_set("pikachu", "static", ("thunderbolt",)),
                  blue=a_set("snorlax", "thickfat"), blue2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderbolt", target_code=1)
    assert state.sides[1].hp[0] == state.pokemon(1, 0).max_hp, "the first one is untouched"
    assert state.sides[1].hp[1] < state.pokemon(1, 1).max_hp, "the second one took it"


def test_you_may_aim_at_your_own_partner(config, dex):
    """Rarely wise, occasionally exactly right, and always legal."""
    state = build(config, a_set("pikachu", "static", ("thunderbolt",)))
    ctx = make_context(state)
    cast(ctx, dex, "thunderbolt", target_code=TARGET_ALLY)
    assert state.sides[0].hp[1] < state.pokemon(0, 1).max_hp


# --------------------------------------------------------------------------- #
# Spread damage
# --------------------------------------------------------------------------- #


def test_earthquake_hits_both_foes_and_the_partner(config, dex):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)))
    ctx = make_context(state)
    cast(ctx, dex, "earthquake")
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp
    assert state.sides[1].hp[1] < state.pokemon(1, 1).max_hp
    assert state.sides[0].hp[1] < state.pokemon(0, 1).max_hp, "and its own partner"
    assert state.sides[0].hp[0] == state.pokemon(0, 0).max_hp, "but not itself"


def test_a_spread_move_takes_a_quarter_off(config, dex):
    from pkcm.engine.moves import activate, compute_damage

    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)))
    ctx = make_context(state)
    move = activate(ctx, RED_A, BLUE_A, dex.moves["earthquake"])

    move.spread = False
    alone, _ = compute_damage(ctx, RED_A, BLUE_A, move, crit=False)
    move.spread = True
    shared, _ = compute_damage(ctx, RED_A, BLUE_A, move, crit=False)
    assert shared < alone
    # 0.75 lands before the crit multiplier and the roll, so the two are not a
    # clean ratio -- only the direction and the rough size are testable here.
    assert 0.7 < shared / alone < 0.8


def test_the_penalty_needs_two_targets(config, dex):
    """A spread move with one Pokemon left in front of it does full damage."""
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)))
    state.sides[1].hp[1] = 0
    state.sides[0].hp[1] = 0
    ctx = make_context(state)
    cast(ctx, dex, "earthquake")
    assert not any(e.kind == "unimplemented" for e in ctx.log)
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp


def test_dazzling_gleam_spares_the_partner(config, dex):
    """allAdjacentFoes, not allAdjacent -- the difference is the partner."""
    state = build(config, a_set("clefable", "unaware", ("dazzlinggleam",)))
    ctx = make_context(state)
    cast(ctx, dex, "dazzlinggleam")
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp
    assert state.sides[1].hp[1] < state.pokemon(1, 1).max_hp
    assert state.sides[0].hp[1] == state.pokemon(0, 1).max_hp


# --------------------------------------------------------------------------- #
# Redirection
# --------------------------------------------------------------------------- #


def test_follow_me_takes_the_hit(config, dex):
    state = build(config, a_set("pikachu", "static", ("thunderbolt",)),
                  blue=a_set("snorlax", "thickfat"),
                  blue2=a_set("clefable", "unaware", ("followme",)))
    ctx = make_context(state)
    cast(ctx, dex, "followme", attacker=BLUE_B)
    cast(ctx, dex, "thunderbolt", target_code=0)
    assert state.sides[1].hp[0] == state.pokemon(1, 0).max_hp, "aimed here, but pulled away"
    assert state.sides[1].hp[1] < state.pokemon(1, 1).max_hp


def test_follow_me_does_not_pull_a_spread_move(config, dex):
    state = build(config, a_set("clefable", "unaware", ("dazzlinggleam",)),
                  blue2=a_set("clefable", "unaware", ("followme",)))
    ctx = make_context(state)
    cast(ctx, dex, "followme", attacker=BLUE_B)
    cast(ctx, dex, "dazzlinggleam")
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp, "both still take it"
    assert state.sides[1].hp[1] < state.pokemon(1, 1).max_hp


def test_rage_powder_misses_a_grass_type(config, dex):
    state = build(config, a_set("venusaur", "overgrow", ("bodyslam",)),
                  blue=a_set("snorlax", "thickfat"),
                  blue2=a_set("clefable", "unaware", ("ragepowder",)))
    ctx = make_context(state)
    cast(ctx, dex, "ragepowder", attacker=BLUE_B)
    cast(ctx, dex, "bodyslam", target_code=0)
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp, "Grass ignores a powder"


def test_lightning_rod_pulls_electric_off_the_partner(config, dex):
    state = build(config, a_set("pikachu", "static", ("thunderbolt",)),
                  blue=a_set("snorlax", "thickfat"),
                  blue2=a_set("marowak", "lightningrod"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderbolt", target_code=0)
    assert state.sides[1].hp[0] == state.pokemon(1, 0).max_hp
    assert state.sides[1].hp[1] == state.pokemon(1, 1).max_hp, "and absorbs it"
    assert state.sides[1].boost(1, "spa") == 1


def test_stalwart_hits_what_it_aimed_at(config, dex):
    state = build(config, a_set("duraludon", "stalwart", ("thunderbolt",)),
                  blue=a_set("snorlax", "thickfat"),
                  blue2=a_set("clefable", "unaware", ("followme",)))
    ctx = make_context(state)
    cast(ctx, dex, "followme", attacker=BLUE_B)
    cast(ctx, dex, "thunderbolt", target_code=0)
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp, "Follow Me did not move it"


# --------------------------------------------------------------------------- #
# The abilities that needed a partner
# --------------------------------------------------------------------------- #


def test_friend_guard_softens_what_the_partner_takes(config, dex):
    def damage(partner_ability):
        state = build(config, a_set("garchomp", "roughskin", ("bodyslam",)),
                      blue=a_set("snorlax", "thickfat"),
                      blue2=a_set("clefable", partner_ability))
        ctx = make_context(state)
        cast(ctx, dex, "bodyslam", target_code=0)
        return state.pokemon(1, 0).max_hp - state.sides[1].hp[0]

    assert damage("friendguard") < damage("unaware")


def test_friend_guard_does_not_protect_its_own_holder(config, dex):
    def damage(ability):
        state = build(config, a_set("garchomp", "roughskin", ("bodyslam",)),
                      blue=a_set("clefable", ability))
        ctx = make_context(state)
        cast(ctx, dex, "bodyslam", target_code=0)
        return state.pokemon(1, 0).max_hp - state.sides[1].hp[0]

    assert damage("friendguard") == damage("unaware")


def test_telepathy_ignores_the_partners_attack(config, dex):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)),
                  red2=a_set("musharna", "telepathy"))
    ctx = make_context(state)
    cast(ctx, dex, "earthquake")
    assert state.sides[0].hp[1] == state.pokemon(0, 1).max_hp
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp, "the foes still take it"


def test_plus_and_minus_need_each_other(config):
    def spa(partner_ability):
        state = build(config, a_set("plusle", "plus"),
                      red2=a_set("minun", partner_ability))
        return mutate.effective_stat(make_context(state), RED_A, Stat.SPA)

    assert spa("minus") > spa("unaware")


def test_battery_boosts_only_special_moves(config, dex):
    from pkcm.engine.moves import _both_sides

    def power(move_id, partner_ability):
        state = build(config, a_set("garchomp", "roughskin", (move_id,)),
                      red2=a_set("charjabug", partner_ability))
        ctx = make_context(state)
        move = dex.moves[move_id]
        return _both_sides(ctx, "modify_base_power", move.base_power,
                           RED_A, BLUE_A, move)

    assert power("dragonpulse", "battery") > power("dragonpulse", "unaware")
    assert power("earthquake", "battery") == power("earthquake", "unaware")


def test_healer_can_cure_the_partner(config):
    """Three turns in ten, so it is asked until it happens rather than once."""
    from pkcm.engine import effects as fx

    from pkcm.engine.rng import Rng

    fired = 0
    for seed in range(60):
        state = build(config, a_set("blissey", "healer"), red2=a_set("snorlax", "thickfat"))
        state.sides[0].status[1] = "brn"
        state.rng = Rng.from_seed(seed)
        ctx = make_context(state)
        fx.notify(ctx, "residual", RED_A)
        fired += state.sides[0].status[1] is None
    assert 5 < fired < 40, f"fired {fired}/60; it should be about 3 in 10"


def test_hospitality_heals_the_partner_on_entry(config):
    state = build(config, a_set("sinistcha", "hospitality"), red2=a_set("snorlax", "thickfat"))
    state.sides[0].hp[1] = 1
    ctx = make_context(state)
    from pkcm.engine import effects as fx

    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[0].hp[1] > 1


def test_curious_medicine_wipes_the_partners_boosts(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("slowking", "curiousmedicine"), red2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    mutate.boost(ctx, RED_B, {"atk": 2}, source=RED_B)
    assert state.sides[0].boost(1, "atk") == 2
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[0].boost(1, "atk") == 0, "the good ones go too"


def test_costar_copies_the_partners_boosts(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("flamigo", "costar"), red2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    mutate.boost(ctx, RED_B, {"atk": 2, "spe": -1}, source=RED_B)
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[0].boost(0, "atk") == 2
    assert state.sides[0].boost(0, "spe") == -1, "the bad ones as well"


def test_sweet_veil_keeps_the_partner_awake(config, dex):
    state = build(config, a_set("venusaur", "overgrow", ("spore",)),
                  blue=a_set("snorlax", "thickfat"),
                  blue2=a_set("swirlix", "sweetveil"))
    ctx = make_context(state)
    cast(ctx, dex, "spore", target_code=0)
    assert state.sides[1].status[0] is None


# --------------------------------------------------------------------------- #
# The moves that needed a partner
# --------------------------------------------------------------------------- #


def test_helping_hand_boosts_the_partners_move(config, dex):
    from pkcm.engine.moves import _both_sides

    state = build(config, a_set("clefable", "unaware", ("helpinghand",)),
                  red2=a_set("garchomp", "roughskin", ("earthquake",)))
    ctx = make_context(state)
    move = dex.moves["earthquake"]
    plain = _both_sides(ctx, "modify_base_power", move.base_power, RED_B, BLUE_A, move)
    cast(ctx, dex, "helpinghand")
    boosted = _both_sides(ctx, "modify_base_power", move.base_power, RED_B, BLUE_A, move)
    assert boosted > plain


def test_helping_hand_fails_alone(singles, dex):
    state = new_battle(
        singles,
        (tuple([a_set("clefable", "unaware", ("helpinghand",))]
               + [a_set(s) for s in ("snorlax", "starmie", "gengar", "alakazam", "garchomp")]),
         tuple(a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar",
                                  "alakazam", "garchomp"))),
        seed=1,
    )
    state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
    ctx = make_context(state)
    use_move(ctx, (0, 0), dex.moves["helpinghand"])
    assert any(e.kind == "move_failed" for e in ctx.log)
    assert not any(e.kind == "unimplemented" for e in ctx.log)


def test_coaching_boosts_the_partner(config, dex):
    """A plain ``boosts`` move -- the declarative executor already ran it."""
    state = build(config, a_set("falinks", "defiant", ("coaching",)),
                  red2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "coaching")
    assert state.sides[0].boost(1, "atk") == 1
    assert state.sides[0].boost(1, "def") == 1
    assert state.sides[0].boost(0, "atk") == 0, "not the user"


def test_decorate_can_be_pointed_at_the_partner(config, dex):
    state = build(config, a_set("alcremie", "sweetveil", ("decorate",)),
                  red2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "decorate", target_code=TARGET_ALLY)
    assert state.sides[0].boost(1, "atk") == 2
    assert state.sides[0].boost(1, "spa") == 2


def test_ally_switch_trades_places(config, dex):
    state = build(config, a_set("hatterene", "magicbounce", ("allyswitch",)),
                  red2=a_set("snorlax", "thickfat"))
    assert state.sides[0].active == [0, 1]
    ctx = make_context(state)
    cast(ctx, dex, "allyswitch")
    assert state.sides[0].active == [1, 0], "the same two, standing the other way round"


def test_magnetic_flux_only_helps_plus_and_minus(config, dex):
    state = build(config, a_set("magnezone", "plus", ("magneticflux",)),
                  red2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "magneticflux")
    assert state.sides[0].boost(0, "def") == 1, "the user runs Plus"
    assert state.sides[0].boost(1, "def") == 0, "the partner does not"


def test_instruct_works_on_an_opponent(config, dex):
    """It was failing as ally-only, and it never needed an ally at all."""
    state = build(config, a_set("oranguru", "innerfocus", ("instruct",)),
                  blue=a_set("snorlax", "thickfat", ("swordsdance",)))
    ctx = make_context(state)
    use_move(ctx, BLUE_A, dex.moves["swordsdance"], move_index=0)
    assert state.sides[1].boost(0, "atk") == 2
    cast(ctx, dex, "instruct", target_code=0)
    assert state.sides[1].boost(0, "atk") == 4, "it swung again"


# --------------------------------------------------------------------------- #
# Turn order with four in the queue
# --------------------------------------------------------------------------- #


def test_after_you_moves_the_target_up(config, dex):
    state = build(config, a_set("clefable", "unaware", ("afteryou",)),
                  red2=a_set("shuckle", "sturdy", ("swordsdance",)))
    state.turn_queue = [(0, 0), (1, 0), (1, 1), (0, 1)]
    state.turn_actions = ((Action.move(0), Action.move(0)), (Action.PASS, Action.PASS))
    ctx = make_context(state)
    cast(ctx, dex, "afteryou", target_code=TARGET_ALLY)
    assert state.turn_queue[0] == (0, 1), "the slow partner goes next"


def test_quash_pushes_the_target_down(config, dex):
    state = build(config, a_set("clefable", "unaware", ("quash",)))
    state.turn_queue = [(1, 0), (0, 1), (1, 1)]
    ctx = make_context(state)
    cast(ctx, dex, "quash", target_code=0)
    assert state.turn_queue[-1] == (1, 0), "it now goes last"


def test_all_four_act_in_speed_order(config):
    state = build(
        config,
        a_set("ninjask", "speedboost", ("bodyslam",)),      # fastest
        red2=a_set("shuckle", "sturdy", ("bodyslam",)),     # slowest
        blue=a_set("alakazam", "synchronize", ("bodyslam",)),
        blue2=a_set("snorlax", "thickfat", ("bodyslam",)),
    )
    state, log = step(state,
                      (Action.move(0, target=0), Action.move(0, target=0)),
                      (Action.move(0, target=0), Action.move(0, target=0)))
    order = [(e.side, e.slot) for e in log if e.kind == "move_used"]
    assert order[0] == (0, 0), "Ninjask first"
    assert order[-1] == (0, 1), "Shuckle last"
    assert len(order) == 4


def test_both_sides_may_switch_at_once(config):
    state = build(config, a_set("garchomp", "roughskin"))
    state, _ = step(state,
                    (Action.switch(2), Action.switch(3)),
                    (Action.switch(2), Action.switch(3)))
    assert state.sides[0].active == [2, 3]
    assert state.sides[1].active == [2, 3]


def test_the_same_pokemon_cannot_go_to_both_positions(config):
    state = build(config, a_set("garchomp", "roughskin"))
    with pytest.raises(IllegalActionError, match="same Pokemon"):
        step(state, (Action.switch(2), Action.switch(2)), (Action.PASS, Action.PASS))


def test_a_side_down_to_one_plays_with_one(config):
    """Rather than deadlocking on a replacement it cannot supply.

    Blue has one Pokemon on the bench and both of its actives about to fall to
    the same Earthquake. Only one position can be refilled; the other is
    emptied, and the battle carries on three-handed.
    """
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)))
    blue = state.sides[1]
    blue.hp[3] = 0          # one left on the bench, not two
    blue.hp[0] = 1
    blue.hp[1] = 1

    state, log = step(state,
                      (Action.move(0), Action.move(0, target=0)),
                      (Action.move(0, target=0), Action.move(0, target=0)))
    assert state.sides[1].is_fainted(0) and state.sides[1].is_fainted(1)
    assert state.phase is Phase.FORCED_SWITCH
    assert sum(state.sides[1].must_switch) == 1, "one replacement, because one is left"
    assert any(e.kind == "position_empty" for e in log), "the other position closes"

    # And the mask agrees: the emptied position has nothing to decide.
    owed = state.sides[1].must_switch.index(True)
    other = 1 - owed
    assert legal_actions(state, 1, other) == (Action.PASS,)


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #


def test_nothing_is_inert_that_has_a_partner_to_act_on(dex):
    """``SINGLES_INERT`` is what is left after doubles, and it should be small."""
    from pkcm.engine import abilities

    assert abilities.SINGLES_INERT <= {"sweetveil2"}, (
        f"still inert with a partner available: {sorted(abilities.SINGLES_INERT)}"
    )


def test_magic_bounce_does_not_ping_pong(config, dex):
    """Two holders facing each other used to bounce the same move forever.

    Singles could never arrange it -- only one of them is ever in front of the
    other -- so the missing guard sat there until a doubles field found it.
    """
    state = build(config, a_set("hatterene", "magicbounce", ("thunderwave",)),
                  blue=a_set("espeon", "magicbounce"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderwave", target_code=0)
    bounces = [e for e in ctx.log if e.kind == "ability_block" and e.detail == "magicbounce"]
    assert len(bounces) == 1, "it comes back once and stops"


# --------------------------------------------------------------------------- #
# Effects that were registered and read by nobody
#
# Wide Guard is a doubles staple; Magic Room and Wonder Room are not, but they
# turned up in the same sweep and had the same problem.
# --------------------------------------------------------------------------- #


def test_wide_guard_stops_a_spread_move(config, dex):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)),
                  blue=a_set("hariyama", "thickfat", ("wideguard",)))
    ctx = make_context(state)
    cast(ctx, dex, "wideguard", attacker=BLUE_A)
    cast(ctx, dex, "earthquake")
    assert state.sides[1].hp[0] == state.pokemon(1, 0).max_hp
    assert state.sides[1].hp[1] == state.pokemon(1, 1).max_hp, "the partner too"


def test_wide_guard_does_not_stop_a_single_target_move(config, dex):
    state = build(config, a_set("garchomp", "roughskin", ("bodyslam",)),
                  blue=a_set("hariyama", "thickfat", ("wideguard",)))
    ctx = make_context(state)
    cast(ctx, dex, "wideguard", attacker=BLUE_A)
    cast(ctx, dex, "bodyslam", target_code=0)
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp


def test_magic_room_makes_items_inert(config, dex):
    """Still held -- ``state.item_id`` says so -- and doing nothing."""
    state = build(config, a_set("garchomp", "roughskin", ("bodyslam",)),
                  blue=a_set("snorlax", "thickfat", ("bodyslam",), item="leftovers"))
    state.sides[1].hp[0] //= 2
    hurt = state.sides[1].hp[0]

    state.field.rooms["magicroom"] = 5
    ctx = make_context(state)
    assert ctx.item_of(BLUE_A) is None
    assert state.item_id(*BLUE_A) == "leftovers", "held, just inert"
    mutate.check_item_triggers(ctx, BLUE_A)
    from pkcm.engine import effects as fx

    fx.notify(ctx, "residual", BLUE_A)
    assert state.sides[1].hp[0] == hurt, "no Leftovers tick"

    del state.field.rooms["magicroom"]
    ctx = make_context(state)
    fx.notify(ctx, "residual", BLUE_A)
    assert state.sides[1].hp[0] > hurt, "and it works again once the room ends"


def test_wonder_room_swaps_the_defences(config):
    state = build(config, a_set("garchomp", "roughskin"),
                  blue=a_set("blissey", "naturalcure"))
    ctx = make_context(state)
    physical = mutate.effective_stat(ctx, BLUE_A, Stat.DEF)
    special = mutate.effective_stat(ctx, BLUE_A, Stat.SPD)
    assert physical != special, "Blissey is the clearest case there is"

    state.field.rooms["wonderroom"] = 5
    ctx = make_context(state)
    assert mutate.effective_stat(ctx, BLUE_A, Stat.DEF) == special
    assert mutate.effective_stat(ctx, BLUE_A, Stat.SPD) == physical


def test_gravity_brings_a_flying_type_down(config):
    from pkcm.engine.conditions import is_grounded

    state = build(config, a_set("garchomp", "roughskin"),
                  blue=a_set("skarmory", "sturdy"))
    assert not is_grounded(state, BLUE_A)
    state.field.rooms["gravity"] = 5
    assert is_grounded(state, BLUE_A), "nothing floats under Gravity"


def test_gravity_refuses_a_jumping_move(config, dex):
    state = build(config, a_set("hawlucha", "unburden", ("fly",)))
    state.field.rooms["gravity"] = 5
    ctx = make_context(state)
    cast(ctx, dex, "fly", target_code=0)
    assert any(e.kind == "cant_move" and e.detail == "gravity" for e in ctx.log)


# --------------------------------------------------------------------------- #
# Abilities that quietly assumed one opponent
#
# Every one of these read ``foe.active[0]`` and stopped. Singles has nothing
# else there, so none of them were wrong until now.
# --------------------------------------------------------------------------- #


def test_intimidate_drops_both_foes(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("gyarados", "intimidate"))
    ctx = make_context(state)
    state.sides[1].boosts[0][0] = 0
    state.sides[1].boosts[1][0] = 0
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[1].boost(0, "atk") == -1
    assert state.sides[1].boost(1, "atk") == -1, "the partner too"


def test_a_substitute_shields_only_the_one_behind_it(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("gyarados", "intimidate"))
    mutate.add_volatile(make_context(state), BLUE_A, "substitute", hp=50)
    state.sides[1].boosts[0][0] = 0
    state.sides[1].boosts[1][0] = 0
    ctx = make_context(state)
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[1].boost(0, "atk") == 0, "behind a Substitute"
    assert state.sides[1].boost(1, "atk") == -1, "its partner is not"


def test_download_adds_up_both_defences(config):
    """It picks the attack that works on the side, not on one Pokemon."""
    from pkcm.engine import effects as fx

    def boosted(partner):
        state = build(config, a_set("porygonz", "download"),
                      blue=a_set("blissey", "naturalcure"), blue2=a_set(partner))
        ctx = make_context(state)
        fx.notify(ctx, "switch_in", RED_A)
        return "atk" if state.sides[0].boost(0, "atk") else "spa"

    # Blissey is paper-thin physically and a wall specially, so with a partner
    # in the same shape it invites a physical attack. Steelix is the mirror
    # image, and between them the side's Defence total wins.
    assert boosted("chansey") == "atk"
    assert boosted("steelix") == "spa", "the partner's Defence counts too"


def test_magnet_pull_holds_the_steel_type_only(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("magnezone", "magnetpull"),
                  blue=a_set("skarmory", "sturdy"), blue2=a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[1].has_volatile(0, "trapped"), "Skarmory is Steel"
    assert not state.sides[1].has_volatile(1, "trapped"), "Snorlax may leave"


def test_shadow_tag_holds_both(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("gothitelle", "shadowtag"))
    ctx = make_context(state)
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[1].has_volatile(0, "trapped")
    assert state.sides[1].has_volatile(1, "trapped")


def test_imposter_copies_diagonally(config):
    """Not the one across -- the one on the far side.

    Showdown indexes the foe side backwards from the copier's own position, so
    the left Ditto becomes the right-hand foe. Singles has one of each and the
    distinction cannot show up.
    """
    from pkcm.engine import effects as fx

    state = build(config, a_set("ditto", "imposter"),
                  blue=a_set("snorlax", "thickfat"), blue2=a_set("garchomp", "roughskin"))
    ctx = make_context(state)
    fx.notify(ctx, "switch_in", RED_A)
    assert state.species_id(0, 0) == "garchomp", "the far one, not the near one"


def test_supersweet_syrup_drops_both_foes_once(config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("dipplin", "supersweetsyrup"))
    ctx = make_context(state)
    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[1].boost(0, "evasion") == -1
    assert state.sides[1].boost(1, "evasion") == -1

    fx.notify(ctx, "switch_in", RED_A)
    assert state.sides[1].boost(0, "evasion") == -1, "and only once a battle"
