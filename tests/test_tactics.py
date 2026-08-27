"""Moves whose effect is structural rather than declarative.

Forcing a switch, U-turning out, answering a hit with Counter -- ordinary
competitive play that none of the declarative fields can express.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.battle import make_context, step
from pkcm.engine.moves import use_move
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, Phase, legal_actions, new_battle

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


def build(config, red, blue, red_bench=None, blue_bench=None):
    filler = [a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    team_red = tuple([red] + (red_bench or filler))
    team_blue = tuple([blue] + (blue_bench or filler))
    state = new_battle(config, (team_red, team_blue), seed=7)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def cast(ctx, dex, move_id, attacker=RED, defender=BLUE):
    use_move(ctx, attacker, dex.moves[move_id], defender=defender)


# --------------------------------------------------------------------------- #
# Forcing a switch
# --------------------------------------------------------------------------- #


def test_roar_drags_in_someone_else(dex, config):
    state = build(config, a_set("skarmory", "sturdy", ("roar",)), a_set("snorlax"))
    before = list(state.sides[1].active)
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].active != before
    assert any(e.kind == "dragged_out" for e in log), log


def test_roar_undoes_a_setup_turn(dex, config):
    """The reason the move exists: stat stages do not follow you out."""
    state = build(config, a_set("skarmory", "sturdy", ("roar",)),
                  a_set("snorlax", "thickfat", ("swordsdance",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].boost(0, "atk") == 0, "the +2 left with it"


def test_dragon_tail_hurts_before_it_drags(dex, config):
    """Dragon Tail is 90% accurate, so the case is found rather than assumed."""
    for seed in range(20):
        state = build(config, a_set("garchomp", "sandveil", ("dragontail",)),
                      a_set("snorlax", "thickfat", ("splash",)))
        state.rng = state.rng.__class__(state.rng.state + seed)
        full = state.pokemon(1, 0).max_hp
        state, log = step(state, Action.move(0), Action.move(0))
        if any(e.kind == "missed" for e in log):
            continue
        assert state.sides[1].hp[0] < full, "damage lands"
        assert state.sides[1].active != [0], "and then it is dragged out"
        return
    pytest.fail("Dragon Tail never connected in 20 tries")


def test_forcing_a_switch_fails_with_nobody_left(dex, config):
    state = build(config, a_set("skarmory", "sturdy", ("roar",)), a_set("snorlax"))
    state.sides[1].hp[1] = 0
    state.sides[1].hp[2] = 0
    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "move_failed" for e in log), log


# --------------------------------------------------------------------------- #
# Switching yourself out
# --------------------------------------------------------------------------- #


def test_u_turn_suspends_the_turn(dex, config):
    """The replacement has to arrive before the opponent moves. That is the move."""
    fast = a_set("weavile", "pressure", ("uturn",))
    slow = a_set("snorlax", "thickfat", ("bodyslam",))
    state = build(config, fast, slow)
    assert state.speed(0) > state.speed(1)

    state, log = step(state, Action.move(0), Action.move(0))
    assert state.phase is Phase.MID_TURN_SWITCH, "the turn stopped to ask"
    assert state.sides[0].owes_switch()
    assert not any(e.kind == "move_used" and e.side == 1 for e in log), \
        "Snorlax has not moved yet"

    options = legal_actions(state, 0)
    assert all(a.kind is ActionKind.SWITCH for a in options)
    assert legal_actions(state, 1) == (Action.PASS,)

    state, log = step(state, Action.switch(1), Action.PASS)
    assert state.phase is Phase.BATTLE
    assert state.sides[0].active == [1], "the replacement is in"
    assert any(e.kind == "move_used" and e.side == 1 for e in log), \
        "and only now does the opponent move"


def test_the_replacement_takes_the_hit(dex, config):
    state = build(config, a_set("weavile", "pressure", ("uturn",)),
                  a_set("snorlax", "thickfat", ("bodyslam",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    state, log = step(state, Action.switch(1), Action.PASS)
    hits = [e for e in log if e.kind == "damage" and e.side == 0]
    assert hits and hits[0].slot == 1, "the incoming Pokemon is what got hit"


def test_u_turn_with_an_empty_bench_just_attacks(dex, config):
    state = build(config, a_set("weavile", "pressure", ("uturn",)), a_set("snorlax"))
    state.sides[0].hp[1] = 0
    state.sides[0].hp[2] = 0
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.phase is Phase.BATTLE, "nothing to switch to, so the turn ran on"


# --------------------------------------------------------------------------- #
# Answering damage
# --------------------------------------------------------------------------- #


def test_counter_returns_double_the_physical_hit(dex, config):
    """Counter is slower than everything, so it always answers rather than opens."""
    state = build(config, a_set("wobbuffet", "shadowtag", ("counter",)),
                  a_set("garchomp", "roughskin", ("earthquake",)))
    full = state.pokemon(1, 0).max_hp
    state, log = step(state, Action.move(0), Action.move(0))

    taken = next(e for e in log if e.kind == "damage" and e.side == 0)
    returned = next(e for e in log if e.kind == "damage" and e.side == 1)
    # Damage dealt is capped at what the target had left, so 2x can come back
    # as a knockout rather than as the literal number.
    assert returned.amount == min(taken.amount * 2, full)
    assert state.sides[1].hp[0] == full - returned.amount


def test_counter_ignores_a_special_hit(dex, config):
    state = build(config, a_set("wobbuffet", "shadowtag", ("counter",)),
                  a_set("alakazam", "synchronize", ("psychic",)))
    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "move_failed" for e in log), "Counter answers physical only"


def test_mirror_coat_answers_the_special_one(dex, config):
    state = build(config, a_set("wobbuffet", "shadowtag", ("mirrorcoat",)),
                  a_set("alakazam", "synchronize", ("psychic",)))
    state, log = step(state, Action.move(0), Action.move(0))
    taken = next(e for e in log if e.kind == "damage" and e.side == 0)
    returned = next(e for e in log if e.kind == "damage" and e.side == 1)
    assert returned.amount == min(taken.amount * 2, state.pokemon(1, 0).max_hp)


def test_counter_forgets_last_turn(dex, config):
    state = build(config, a_set("wobbuffet", "shadowtag", ("counter", "splash")),
                  a_set("garchomp", "roughskin", ("earthquake", "swordsdance")))
    state, _ = step(state, Action.move(1), Action.move(0))    # take a hit
    state, log = step(state, Action.move(0), Action.move(1))  # counter, unhit
    assert any(e.kind == "move_failed" for e in log), "the ledger is per turn"


def test_endeavor_levels_the_hp(dex, config):
    state = build(config, a_set("wobbuffet", "shadowtag", ("endeavor",)), a_set("snorlax"))
    state.sides[0].hp[0] = 20
    ctx = make_context(state)
    cast(ctx, dex, "endeavor")
    assert state.sides[1].hp[0] == 20


def test_super_fang_halves(dex, config):
    state = build(config, a_set("raticate", "guts", ("superfang",)), a_set("snorlax"))
    full = state.pokemon(1, 0).max_hp
    ctx = make_context(state)
    cast(ctx, dex, "superfang")
    assert state.sides[1].hp[0] == full - full // 2


# --------------------------------------------------------------------------- #
# Two-turn moves
# --------------------------------------------------------------------------- #


def test_solar_beam_charges_then_fires(dex, config):
    # The opponent must not Protect on the second turn, or nothing lands and
    # the test proves nothing.
    state = build(config, a_set("venusaur", "overgrow", ("solarbeam", "bodyslam")),
                  a_set("starmie", "illuminate", ("splash",)))
    full = state.pokemon(1, 0).max_hp

    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "charging" for e in log), log
    assert state.sides[1].hp[0] == full, "nothing happened yet"

    # The charge locks the choice in: only Solar Beam is legal now.
    assert [a.index for a in legal_actions(state, 0) if a.kind is ActionKind.MOVE] == [0]
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].hp[0] < full, "and now it lands"


def test_dig_makes_the_user_untouchable(dex, config):
    state = build(config, a_set("garchomp", "roughskin", ("dig",)),
                  a_set("snorlax", "thickfat", ("bodyslam", "earthquake")))
    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "avoided" for e in log), "Body Slam cannot reach underground"
    assert state.sides[0].hp[0] == state.pokemon(0, 0).max_hp


def test_earthquake_still_reaches_underground(dex, config):
    state = build(config, a_set("garchomp", "roughskin", ("dig",)),
                  a_set("snorlax", "thickfat", ("earthquake", "bodyslam")))
    state, log = step(state, Action.move(0), Action.move(0))
    assert not any(e.kind == "avoided" for e in log)
    assert state.sides[0].hp[0] < state.pokemon(0, 0).max_hp


def test_power_herb_skips_the_charge(dex, config):
    state = build(config,
                  a_set("venusaur", "overgrow", ("solarbeam",), item="powerherb"),
                  a_set("starmie", "illuminate", ("splash",)))
    full = state.pokemon(1, 0).max_hp
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].hp[0] < full, "it fired on the first turn"
    assert state.item_id(0, 0) is None, "and the herb is gone"


def test_hyper_beam_needs_a_turn_to_recharge(dex, config):
    state = build(config, a_set("snorlax", "thickfat", ("hyperbeam",)),
                  a_set("skarmory", "sturdy", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].has_volatile(0, "mustrecharge")
    assert legal_actions(state, 0) == (Action.PASS,), "nothing else it can do"

    state, log = step(state, Action.PASS, Action.move(0))
    assert any(e.kind == "recharging" for e in log), log
    assert not state.sides[0].has_volatile(0, "mustrecharge"), "free again"


# --------------------------------------------------------------------------- #
# Trapping, locking, and going out with a bang
# --------------------------------------------------------------------------- #


def test_fire_spin_traps_and_chips(dex, config):
    state = build(config, a_set("arcanine", "intimidate", ("firespin",)),
                  a_set("snorlax", "thickfat", ("bodyslam",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].has_volatile(0, "partiallytrapped")
    assert all(a.kind is not ActionKind.SWITCH for a in legal_actions(state, 1)), "stuck"

    before = state.sides[1].hp[0]
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].hp[0] < before


def test_outrage_locks_the_user_in(dex, config):
    state = build(config, a_set("dragonite", "multiscale", ("outrage", "bodyslam")),
                  a_set("snorlax", "thickfat", ("protect",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert [a.index for a in legal_actions(state, 0) if a.kind is ActionKind.MOVE] == [0], \
        "no choice while raging"


def test_explosion_takes_the_user_with_it(dex, config):
    state = build(config, a_set("electrode", "static", ("explosion",)), a_set("snorlax"))
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].is_fainted(0), "the user faints"
    assert any(e.kind == "self_destruct" for e in log)


def test_final_gambit_trades_the_users_remaining_hp(dex, config):
    state = build(config, a_set("staraptor", "intimidate", ("finalgambit",)), a_set("snorlax"))
    state.sides[0].hp[0] = 77
    full = state.pokemon(1, 0).max_hp
    ctx = make_context(state)
    cast(ctx, dex, "finalgambit")
    assert state.sides[1].hp[0] == full - 77
    assert state.sides[0].is_fainted(0)


def test_memento_drops_the_opponent_on_the_way_out(dex, config):
    state = build(config, a_set("gengar", "cursedbody", ("memento",)),
                  a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "memento")
    assert state.sides[1].boost(0, "atk") == -2
    assert state.sides[1].boost(0, "spa") == -2
    assert state.sides[0].is_fainted(0)


def test_a_slow_u_turn_takes_the_hit_before_leaving(dex, config):
    """U-turn has no priority, so Speed decides -- and being slow costs you.

    Fast: switch first, the replacement eats the attack.
    Slow: take the attack yourself, then leave; the replacement comes in clean.
    """
    slow_user = a_set("snorlax", "thickfat", ("uturn",))
    fast_foe = a_set("weavile", "pressure", ("nightslash",))
    state = build(config, slow_user, fast_foe)
    assert state.speed(0) < state.speed(1)

    state, log = step(state, Action.move(0), Action.move(0))
    order = [e.side for e in log if e.kind == "move_used"]
    assert order == [1, 0], "the faster Pokemon moved first"

    hits = [e for e in log if e.kind == "damage" and e.side == 0]
    assert hits and hits[0].slot == 0, "the U-turn user took it, not the replacement"
    assert state.phase is Phase.MID_TURN_SWITCH

    state, log = step(state, Action.switch(1), Action.PASS)
    assert state.sides[0].active == [1]
    assert not [e for e in log if e.kind == "damage"], "nobody is left to act"


def test_volt_switch_and_flip_turn_behave_the_same_way(dex, config):
    for species, ability, move_id in (("rotomwash", "levitate", "voltswitch"),
                                      ("barraskewda", "swiftswim", "flipturn")):
        state = build(config, a_set(species, ability, (move_id,)),
                      a_set("snorlax", "thickfat", ("splash",)))
        state, _ = step(state, Action.move(0), Action.move(0))
        assert state.phase is Phase.MID_TURN_SWITCH, move_id


# --------------------------------------------------------------------------- #
# Running out the clock
# --------------------------------------------------------------------------- #


def _timed_out(config, red_hp, blue_hp, red_pp=None, blue_pp=None):
    """A battle taken straight to the turn limit with the HP set by hand."""
    from pkcm.engine.battle import _decide_by_attrition

    filler = [a_set(s) for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    team = tuple([a_set("garchomp", "roughskin", ("earthquake",))] + filler)
    state = new_battle(config, (team, team), seed=1)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
    state.sides[0].hp = list(red_hp)
    state.sides[1].hp = list(blue_hp)
    if red_pp:
        state.sides[0].pp = [list(slot) for slot in red_pp]
    if blue_pp:
        state.sides[1].pp = [list(slot) for slot in blue_pp]
    return _decide_by_attrition(state)


def test_time_over_counts_pokemon_first(config):
    assert _timed_out(config, [1, 1, 1], [200, 200, 0]) == 0, "three standing beats two"


def test_time_over_then_compares_hp_as_a_share(config):
    """Same number standing, so the share of their own maximum decides.

    Garchomp is 183 and Snorlax is 235: equal *amounts* of HP are not equal
    shares, and the share is what is asked second.
    """
    # Both sides bring the same three, so the totals match; only the split moves.
    assert _timed_out(config, [180, 0, 0], [90, 0, 0]) == 0


def test_time_over_falls_through_to_absolute_hp(config):
    """Identical shares, so the raw number breaks the tie.

    Both sides untouched, so both hold 100% of their own maximum -- an exact
    tie on tier two. The teams differ, so the amounts do not.
    """
    from pkcm.engine.battle import _decide_by_attrition, _hp_absolute, _hp_share

    filler = [a_set(s) for s in ("pikachu", "starmie", "gengar", "alakazam", "skarmory")]
    bulky = tuple([a_set("snorlax", "thickfat", ("bodyslam",))] + filler)
    frail = tuple([a_set("alakazam", "synchronize", ("psychic",))] + filler)
    state = new_battle(config, (bulky, frail), seed=1)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    assert _hp_share(state, 0) == _hp_share(state, 1) == 1.0, "shares are level"
    assert _hp_absolute(state, 0) > _hp_absolute(state, 1), "amounts are not"
    assert _decide_by_attrition(state) == 0


def test_time_over_ends_on_pp(config):
    """Everything else level, so whoever has moves left wins."""
    assert _timed_out(config, [100, 100, 100], [100, 100, 100],
                      red_pp=[[10], [10], [10]], blue_pp=[[1], [1], [1]]) == 0


def test_time_over_can_still_be_a_draw(config):
    assert _timed_out(config, [100, 100, 100], [100, 100, 100]) is None


# --------------------------------------------------------------------------- #
# The scripted family
# --------------------------------------------------------------------------- #


def test_temperament_changes_the_choice(dex):
    """The whole idea: one calculator, read with different attitudes.

    A certain 60% against a 60% chance to knock out is the case that separates
    them, and ``GreedyPolicy``'s fixed ``ko_chance * 10`` always takes the roll.
    """
    from pkcm.search.scripted import BY_NAME

    safe, gambler = BY_NAME["safe"], BY_NAME["gambler"]
    assert gambler.ko_weight > safe.ko_weight * 5
    # A certain 60% scores 0.6 for both; a 60% roll at a knockout scores
    # ko_weight * 0.6, so only the low weight prefers the certain damage.
    assert safe.ko_weight * 0.6 < 0.6 * 4
    assert gambler.ko_weight * 0.6 > 0.6 * 4


def test_the_family_plays_legally_and_differs(dex):
    from pkcm.engine.legality import make_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.policy import play_out
    from pkcm.search.scripted import TACTICS, TacticPolicy

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(make_team(dex, config.regulation,
                            Rng.from_seed(300 + offset).cursor(), "singles", "ranker")
                  for offset in (1, 2))

    endings = set()
    for tactic in TACTICS:
        state = play_out(new_battle(config, teams, seed=9),
                         (TacticPolicy.seeded(tactic.name, 9),
                          TacticPolicy.seeded("greedy", 77)))
        assert state.finished or state.turn > config.turn_limit
        endings.add((state.winner, state.turn))
    assert len(endings) > 1, "every temperament played the same battle"


def test_the_oracle_is_not_something_to_imitate(dex):
    """It reads the opponent exactly, which is why it is a sparring partner and
    never a behaviour-cloning target: the actions are a function of information
    a policy cannot see, so cloning them teaches the conditional mean."""
    from pkcm.engine.legality import make_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import BattleConfig, new_battle
    from pkcm.search.scripted import _oracle
    from pkcm.envs.observation import Observation

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    teams = tuple(make_team(dex, config.regulation,
                            Rng.from_seed(310 + offset).cursor(), "singles", "ranker")
                  for offset in (1, 2))
    state = new_battle(config, teams, seed=11)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    honest = Observation.of(state, 0)
    cheating = _oracle(state, 0)
    assert any(known.species_id is None for known in honest.foe), (
        "this position was meant to have something unrevealed")
    assert all(known.species_id is not None for known in cheating.foe)
    assert all(known.item_known for known in cheating.foe)
