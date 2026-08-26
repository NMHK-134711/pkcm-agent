"""Status conditions, stat stages, weather, screens, hazards and move effects.

These exercise the hook system rather than the turn loop: each case sets up a
position, runs one move, and checks what the mechanic did.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import Action
from pkcm.engine.battle import make_context, step
from pkcm.engine.effects import registered
from pkcm.engine.moves import use_move
from pkcm.engine.mutate import effective_stat
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, new_battle

RED, BLUE = (0, 0), (1, 0)


@pytest.fixture(scope="module")
def dex():
    return load_dex()


@pytest.fixture(scope="module")
def config(dex):
    return BattleConfig(dex=dex, regulation=dex.regulation("m_b"), battle_format="singles")


def a_set(species: str, moves: tuple[str, ...], **kwargs) -> PokemonSet:
    defaults = dict(ability="__test__", nature="serious", sp=(0, 0, 0, 0, 0, 0))
    return PokemonSet(species=species, moves=moves, **{**defaults, **kwargs})


def build(config, red, blue):
    bench = [a_set(s, ("tackle",)) for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    team_red = tuple([red] + bench)
    team_blue = tuple([blue] + bench)
    state = new_battle(config, (team_red, team_blue), seed=7)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def cast(ctx, dex, move_id: str, attacker=RED, defender=BLUE):
    use_move(ctx, attacker, dex.moves[move_id], defender=defender)


# --------------------------------------------------------------------------- #
# Stat stages
# --------------------------------------------------------------------------- #


def test_swords_dance_raises_attack_two_stages(dex, config):
    state = build(config, a_set("garchomp", ("swordsdance",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    before = effective_stat(ctx, RED, Stat.ATK)

    cast(ctx, dex, "swordsdance")

    assert state.sides[0].boost(0, "atk") == 2
    assert effective_stat(ctx, RED, Stat.ATK) == int(before * 2.0), "+2 doubles the stat"


def test_stat_stages_clamp_at_six(dex, config):
    state = build(config, a_set("garchomp", ("swordsdance",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    for _ in range(4):
        cast(ctx, dex, "swordsdance")
    assert state.sides[0].boost(0, "atk") == 6
    assert any(e.kind == "boost_failed" for e in ctx.log)


def test_stages_are_lost_on_switch_out(dex, config):
    state = build(config, a_set("garchomp", ("swordsdance",)), a_set("snorlax", ("tackle",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].boost(0, "atk") == 2

    state, _ = step(state, Action.switch(1), Action.move(0))
    assert state.sides[0].boost(0, "atk") == 0


# --------------------------------------------------------------------------- #
# Status conditions
# --------------------------------------------------------------------------- #


def test_burn_halves_attack_and_chips_hp(dex, config):
    state = build(config, a_set("snorlax", ("tackle",)), a_set("garchomp", ("willowisp",)))
    ctx = make_context(state)
    unburned = effective_stat(ctx, RED, Stat.ATK)

    cast(ctx, dex, "willowisp", attacker=BLUE, defender=RED)
    assert state.sides[0].status[0] == "brn"
    assert effective_stat(ctx, RED, Stat.ATK) == unburned // 2

    before = state.sides[0].hp[0]
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] < before, "burn damage at end of turn"


def test_paralysis_halves_speed(dex, config):
    # Dragonite, not a Ground type -- Ground is immune to Electric moves entirely.
    state = build(config, a_set("dragonite", ("tackle",)), a_set("snorlax", ("thunderwave",)))
    ctx = make_context(state)
    fast = effective_stat(ctx, RED, Stat.SPE)

    cast(ctx, dex, "thunderwave", attacker=BLUE, defender=RED)
    assert state.sides[0].status[0] == "par"
    assert effective_stat(ctx, RED, Stat.SPE) == fast // 2


def test_electric_types_cannot_be_paralyzed_by_thunder_wave(dex, config):
    """Thunder Wave is Electric, so the type chart makes it miss entirely."""
    state = build(config, a_set("pikachu", ("tackle",)), a_set("snorlax", ("thunderwave",)))
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].status[0] is None
    assert any(e.kind == "status_immune" for e in log)


def test_ground_types_are_untouched_by_thunder_wave(dex, config):
    """A status move still has a type, and the chart still applies."""
    state = build(config, a_set("garchomp", ("tackle",)), a_set("snorlax", ("thunderwave",)))
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].status[0] is None
    assert any(e.kind == "immune" for e in log)


def test_toxic_damage_grows_each_turn(dex, config):
    state = build(config, a_set("snorlax", ("protect",)), a_set("garchomp", ("toxic",)))
    ctx = make_context(state)
    cast(ctx, dex, "toxic", attacker=BLUE, defender=RED)
    assert state.sides[0].status[0] == "tox"

    losses = []
    for _ in range(3):
        before = state.sides[0].hp[0]
        state, _ = step(state, Action.move(0), Action.move(0))
        losses.append(before - state.sides[0].hp[0])
    assert losses == sorted(losses) and losses[0] < losses[-1], losses


def test_a_second_status_cannot_be_applied(dex, config):
    state = build(config, a_set("snorlax", ("tackle",)), a_set("garchomp", ("willowisp", "toxic")))
    ctx = make_context(state)
    cast(ctx, dex, "willowisp", attacker=BLUE, defender=RED)
    cast(ctx, dex, "toxic", attacker=BLUE, defender=RED)
    assert state.sides[0].status[0] == "brn"


def test_status_survives_switching_out(dex, config):
    state = build(config, a_set("snorlax", ("tackle",)), a_set("garchomp", ("willowisp",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].status[0] == "brn"

    state, _ = step(state, Action.switch(1), Action.move(0))
    state, _ = step(state, Action.switch(0), Action.move(0))
    assert state.sides[0].status[0] == "brn", "major status is not a field-only condition"


# --------------------------------------------------------------------------- #
# Move effects read straight off the data
# --------------------------------------------------------------------------- #


def test_drain_heals_the_user(dex, config):
    state = build(config, a_set("venusaur", ("gigadrain",)), a_set("starmie", ("tackle",)))
    state.sides[0].hp[0] = 50
    ctx = make_context(state)
    cast(ctx, dex, "gigadrain")
    assert state.sides[0].hp[0] > 50
    assert any(e.kind == "heal" and e.detail == "drain" for e in ctx.log)


def test_recoil_hurts_the_user(dex, config):
    state = build(config, a_set("snorlax", ("doubleedge",)), a_set("starmie", ("tackle",)))
    ctx = make_context(state)
    cast(ctx, dex, "doubleedge")
    recoil = [e for e in ctx.log if e.kind == "recoil"]
    assert len(recoil) == 1
    damage = next(e for e in ctx.log if e.kind == "damage")
    numerator, denominator = dex.moves["doubleedge"].raw["recoil"]  # [33, 100]
    assert recoil[0].amount == max(1, damage.amount * numerator // denominator)


def test_multihit_moves_hit_two_to_five_times(dex, config):
    counts = set()
    for seed in range(30):
        state = build(config, a_set("cinccino", ("bulletseed",)), a_set("snorlax", ("tackle",)))
        state.rng = state.rng.__class__(state.rng.state + seed)
        ctx = make_context(state)
        cast(ctx, dex, "bulletseed")
        hits = [e for e in ctx.log if e.kind == "damage"]
        counts.add(len(hits))
    assert counts <= {2, 3, 4, 5}
    assert len(counts) > 1, "the hit count must actually vary"


def test_fixed_damage_ignores_stats(dex, config):
    state = build(config, a_set("alakazam", ("seismictoss",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    cast(ctx, dex, "seismictoss")
    damage = next(e for e in ctx.log if e.kind == "damage")
    assert damage.amount == 50, "Seismic Toss deals the user's level"


def test_self_boost_after_attacking(dex, config):
    """Overheat carries ``self: {boosts: {spa: -2}}``."""
    state = build(config, a_set("typhlosion", ("overheat",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    cast(ctx, dex, "overheat")
    assert state.sides[0].boost(0, "spa") == -2


# --------------------------------------------------------------------------- #
# Protect and Substitute
# --------------------------------------------------------------------------- #


def test_protect_blocks_the_incoming_move(dex, config):
    state = build(config, a_set("snorlax", ("protect",)), a_set("garchomp", ("earthquake",)))
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] == state.pokemon(0, 0).max_hp
    assert any(e.kind == "protected" for e in log)


def test_protect_expires_after_its_turn(dex, config):
    state = build(config, a_set("snorlax", ("protect", "tackle")), a_set("garchomp", ("earthquake",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    state, log = step(state, Action.move(1), Action.move(0))
    assert not any(e.kind == "protected" for e in log)
    assert state.sides[0].hp[0] < state.pokemon(0, 0).max_hp


def test_substitute_absorbs_damage(dex, config):
    state = build(config, a_set("snorlax", ("substitute",)), a_set("pikachu", ("tackle",)))
    ctx = make_context(state)
    cast(ctx, dex, "substitute")
    full = state.pokemon(0, 0).max_hp
    cost = full // 4
    assert state.sides[0].hp[0] == full - cost
    assert mutate.volatile(state, RED, "substitute")["hp"] == cost

    after_setup = state.sides[0].hp[0]
    cast(ctx, dex, "tackle", attacker=BLUE, defender=RED)
    assert state.sides[0].hp[0] == after_setup, "the substitute took it"


# --------------------------------------------------------------------------- #
# Field and side conditions
# --------------------------------------------------------------------------- #


def test_rain_boosts_water_and_damps_fire(dex, config):
    state = build(config, a_set("starmie", ("surf", "raindance")), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    dry = sum(_surf_damage(ctx, dex) for _ in range(200))

    cast(ctx, dex, "raindance")
    assert state.field.weather == "raindance"
    wet = sum(_surf_damage(ctx, dex) for _ in range(200))
    assert wet > dry * 1.3


def _surf_damage(ctx, dex) -> int:
    from pkcm.engine.moves import compute_damage

    return compute_damage(ctx, RED, BLUE, dex.moves["surf"], crit=False)[0]


def test_reflect_halves_physical_damage_only(dex, config):
    from pkcm.engine.moves import compute_damage

    state = build(config, a_set("garchomp", ("earthquake",)), a_set("snorlax", ("reflect",)))
    ctx = make_context(state)
    physical = compute_damage(ctx, RED, BLUE, dex.moves["earthquake"], crit=False)[0]
    special = compute_damage(ctx, RED, BLUE, dex.moves["dragonpulse"], crit=False)[0]

    cast(ctx, dex, "reflect", attacker=BLUE, defender=RED)
    assert "reflect" in state.sides[1].conditions

    screened = [compute_damage(ctx, RED, BLUE, dex.moves["earthquake"], crit=False)[0]
                for _ in range(50)]
    unscreened = [compute_damage(ctx, RED, BLUE, dex.moves["dragonpulse"], crit=False)[0]
                  for _ in range(50)]
    assert max(screened) < physical
    assert max(unscreened) >= special * 0.8, "Reflect must not touch special moves"


def test_stealth_rock_hurts_on_entry_by_type(dex, config):
    state = build(config, a_set("charizard", ("tackle",)), a_set("snorlax", ("stealthrock",)))
    ctx = make_context(state)
    cast(ctx, dex, "stealthrock", attacker=BLUE, defender=RED)
    assert "stealthrock" in state.sides[0].conditions

    state, log = step(state, Action.switch(1), Action.move(0))
    state, log = step(state, Action.switch(0), Action.move(0))
    hazard = [e for e in log if e.kind == "hazard_damage"]
    assert hazard, "Charizard is 4x weak to Rock and must be hurt coming in"
    assert hazard[0].amount == max(1, state.pokemon(0, 0).max_hp // 2)


def test_spikes_skip_flying_types(dex, config):
    state = build(config, a_set("skarmory", ("tackle",)), a_set("snorlax", ("spikes",)))
    ctx = make_context(state)
    cast(ctx, dex, "spikes", attacker=BLUE, defender=RED)

    state, _ = step(state, Action.switch(1), Action.move(0))
    state, log = step(state, Action.switch(0), Action.move(0))
    assert not [e for e in log if e.kind == "hazard_damage"], "Skarmory flies over Spikes"


def test_trick_room_reverses_speed_order(dex, config):
    fast = a_set("garchomp", ("earthquake",))
    slow = a_set("snorlax", ("trickroom", "bodyslam"))
    state = build(config, fast, slow)

    state, log = step(state, Action.move(0), Action.move(0))
    assert "trickroom" in state.field.rooms

    state, log = step(state, Action.move(0), Action.move(1))
    movers = [e.side for e in log if e.kind == "move_used"]
    assert movers[0] == 1, "under Trick Room the slower Pokemon acts first"


# --------------------------------------------------------------------------- #
# Honesty about what is wired up
# --------------------------------------------------------------------------- #


def test_conditions_claimed_implemented_are_all_registered():
    from pkcm.engine import conditions

    assert conditions.IMPLEMENTED_STATUSES <= set(registered("status"))
    assert conditions.IMPLEMENTED_VOLATILES <= set(registered("volatile"))
    assert conditions.IMPLEMENTED_SIDE_CONDITIONS <= set(registered("side"))
    assert conditions.IMPLEMENTED_WEATHER <= set(registered("weather"))
    assert conditions.IMPLEMENTED_TERRAIN <= set(registered("terrain"))
    assert conditions.IMPLEMENTED_ROOMS <= set(registered("room"))


def test_moves_setting_unwired_conditions_are_not_claimed(dex, monkeypatch):
    """The Safeguard case: declarative, but nothing reads what it writes.

    Safeguard is wired up now, so the check is shown by unwiring it: the value
    matters, not just the presence of the field. Writing a condition name into
    the state and having nobody consult it is a move that does nothing.
    """
    from pkcm.engine import conditions
    from pkcm.engine import moveeffects
    from pkcm.engine.scope import move_support

    assert move_support(dex.moves["safeguard"]) is None

    hidden = dict(moveeffects.SPECIAL_MOVES)
    hidden.pop("safeguard")
    monkeypatch.setattr(moveeffects, "SPECIAL_MOVES", hidden)
    monkeypatch.setattr(conditions, "IMPLEMENTED_SIDE_CONDITIONS",
                        conditions.IMPLEMENTED_SIDE_CONDITIONS - {"safeguard"})

    assert move_support(dex.moves["safeguard"]) == "unhandled side condition: safeguard"

    assert move_support(dex.moves["reflect"]) is None
    assert move_support(dex.moves["spikes"]) is None
    assert move_support(dex.moves["willowisp"]) is None


def test_prankster_cannot_reach_a_dark_type(dex, config):
    """A status move boosted by Prankster fails on Dark, gen 6 onward."""
    state = build(config, a_set("thundurus", ("thunderwave",), ability="prankster"),
                  a_set("umbreon", ("bodyslam",), ability="synchronize"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderwave")
    assert state.sides[1].status[0] is None
    assert any(e.kind == "immune" for e in ctx.log)


def test_without_prankster_the_same_move_lands(dex, config):
    state = build(config, a_set("thundurus", ("thunderwave",), ability="defiant"),
                  a_set("umbreon", ("bodyslam",), ability="synchronize"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderwave")
    assert state.sides[1].status[0] == "par"


def test_prankster_still_reaches_its_own_side(dex, config):
    """Only the other side is protected; a self-target is unaffected."""
    state = build(config, a_set("thundurus", ("swordsdance",), ability="prankster"),
                  a_set("umbreon", ("bodyslam",), ability="synchronize"))
    ctx = make_context(state)
    cast(ctx, dex, "swordsdance")
    assert state.sides[0].boost(0, "atk") == 2
