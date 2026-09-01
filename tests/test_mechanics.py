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
    """Champions nerfed the *chance* to 1/8 and left this alone (hk, confirmed)."""
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


# --------------------------------------------------------------------------- #
# Stat stages never leave [-6, +6]
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("battle_format", ("singles", "doubles"))
def test_stat_stages_stay_inside_the_cap(dex, battle_format):
    """The clamp is in ``mutate.boost``, and seven places write ``boosts[]``
    without going through it: Belly Drum sets Attack to six outright,
    Topsy-Turvy negates, Psych Up and two abilities copy, White Herb clears the
    negatives, and one effect resets to zero.

    Every one of those is safe by construction -- negating a number in the
    range keeps it in the range, copying a clamped value copies a clamped
    value. Reading them says so. Nothing checked it until this, and the next
    move that writes the list directly will not be safe by inspection.
    """
    from pkcm.engine.battle import step
    from pkcm.engine.legality import random_team
    from pkcm.engine.rng import Rng
    from pkcm.engine.state import (
        MAX_BOOST,
        MIN_BOOST,
        BattleConfig,
        new_battle,
    )
    from pkcm.search.policy import RandomPolicy

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format=battle_format)
    for seed in range(12):
        teams = tuple(
            random_team(dex, config.regulation,
                        Rng.from_seed(4000 + seed * 2 + offset).cursor(),
                        battle_format)
            for offset in (1, 2)
        )
        state = new_battle(config, teams, seed=4000 + seed)
        policies = (RandomPolicy(Rng.from_seed(seed).cursor()),
                    RandomPolicy(Rng.from_seed(seed + 555).cursor()))
        while not state.finished and state.turn <= config.turn_limit:
            choices = tuple(policies[player].act(state, player)
                            for player in (0, 1))
            state, _ = step(state, choices[0], choices[1])
            for side_index, side in enumerate(state.sides):
                for slot, stages in enumerate(side.boosts):
                    for stage in stages:
                        assert MIN_BOOST <= stage <= MAX_BOOST, (
                            f"{battle_format} seed {seed} turn {state.turn}: "
                            f"side {side_index} slot {slot} at stage {stage}")


def test_belly_drum_lands_on_six_and_not_past_it(dex):
    """The one effect that writes the stage directly rather than adding to it.
    From +2 it must land on +6, not +8."""
    from pkcm.engine.state import BOOST_INDEX, MAX_BOOST

    # The rule, stated where a reader will find it: Belly Drum maximises
    # Attack, it does not add six stages to whatever is there.
    assert MAX_BOOST == 6
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "src" / "pkcm"
              / "engine" / "moveeffects.py").read_text(encoding="utf-8")
    assert 'BOOST_INDEX["atk"]] = 6' in source, (
        "Belly Drum no longer assigns the cap outright; if it now adds stages "
        "it has to go through mutate.boost so the clamp applies")
    assert BOOST_INDEX["atk"] == 0


# --------------------------------------------------------------------------- #
# Moves that read a different stat than their category says
# --------------------------------------------------------------------------- #


def _damage_to(state, side=1):
    hp = state.sides[side].hp[0]
    maximum = state.pokemon(side, 0).max_hp
    return maximum - hp


def test_body_press_reads_defense_not_attack(dex, config):
    """Corviknight's whole design: Bulk Up raises Defense, Body Press swings
    with it. Priced off Attack, the pool's five Bulk Up Corviknight sets are
    all playing a different game than the one the ranker built."""
    from pkcm.engine.moves import compute_damage

    state = build(config, a_set("corviknight", ("bodypress",)),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    plain, _ = compute_damage(ctx, RED, BLUE, dex.moves["bodypress"], crit=False)

    # Two stages of Defense must move Body Press the way two stages of Attack
    # move Tackle: doubled, before rolls.
    mutate.boost(ctx, RED, {"def": 2})
    boosted, _ = compute_damage(ctx, RED, BLUE, dex.moves["bodypress"], crit=False)
    assert boosted > int(plain * 1.7), (plain, boosted)

    # ...and Attack stages must not touch it at all. The roll is drawn per
    # call, so allow the 85-100 spread and nothing more.
    mutate.boost(ctx, RED, {"atk": 6})
    still, _ = compute_damage(ctx, RED, BLUE, dex.moves["bodypress"], crit=False)
    assert still <= int(boosted * 100 / 85) + 1, (boosted, still)


def test_psyshock_hits_the_physical_wall_on_its_defense(dex, config):
    """A special move that targets Defense. Against Snorlax (base 65 Defense,
    110 Special Defense) it must out-damage Psychic despite ten less power:
    the stat ratio is 110/65 = 1.69 against a power ratio of 90/80 = 1.13,
    so anything much past 1.2x says the right stat is being read. Computed
    off Special Defense it would come out *under* Psychic instead."""
    from pkcm.engine.moves import compute_damage

    state = build(config, a_set("alakazam", ("psyshock", "psychic")),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    shock = max(compute_damage(ctx, RED, BLUE, dex.moves["psyshock"], crit=False)[0]
                for _ in range(6))
    psychic = max(compute_damage(ctx, RED, BLUE, dex.moves["psychic"], crit=False)[0]
                  for _ in range(6))
    assert shock > psychic * 1.2, (shock, psychic)


def test_foul_play_swings_with_the_targets_attack(dex, config):
    """Foul Play reads the target's Attack, boosts included. A target that has
    Sworded twice must take about twice the hit."""
    from pkcm.engine.moves import compute_damage

    state = build(config, a_set("umbreon", ("foulplay",)),
                  a_set("garchomp", ("swordsdance",)))
    ctx = make_context(state)
    plain = max(compute_damage(ctx, RED, BLUE, dex.moves["foulplay"], crit=False)[0]
                for _ in range(6))
    mutate.boost(ctx, BLUE, {"atk": 2})
    boosted = max(compute_damage(ctx, RED, BLUE, dex.moves["foulplay"], crit=False)[0]
                  for _ in range(6))
    assert boosted > int(plain * 1.7), (plain, boosted)


# --------------------------------------------------------------------------- #
# The audit of 2026-09-01: behaviors the data declared and nothing read
# --------------------------------------------------------------------------- #


def test_scale_shot_pays_and_earns_its_stages(dex, config):
    """+1 Spe / -1 Def after it lands -- the reason ten Garchomp sets run it."""
    state = build(config, a_set("garchomp", ("scaleshot",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    cast(ctx, dex, "scaleshot")
    assert state.sides[0].boost(0, "spe") == 1
    assert state.sides[0].boost(0, "def") == -1


def test_hex_doubles_into_a_status(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("gengar", ("hex",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    plain = VARIABLE_POWER["hex"](ctx, RED, BLUE, dex.moves["hex"])
    state.sides[1].status[0] = "brn"
    doubled = VARIABLE_POWER["hex"](ctx, RED, BLUE, dex.moves["hex"])
    assert doubled == plain * 2 == 130


def test_stored_power_prices_every_stage(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("clefable", ("storedpower", "calmmind")),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    assert VARIABLE_POWER["storedpower"](ctx, RED, BLUE, dex.moves["storedpower"]) == 20
    cast(ctx, dex, "calmmind")
    cast(ctx, dex, "calmmind")
    # +2 SpA and +2 SpD is four stages: 20 + 4 * 20.
    assert VARIABLE_POWER["storedpower"](ctx, RED, BLUE, dex.moves["storedpower"]) == 100


def test_avalanche_doubles_after_taking_the_hit(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("snorlax", ("tackle",)), a_set("garchomp", ("avalanche",)))
    ctx = make_context(state)
    move = dex.moves["avalanche"]
    assert VARIABLE_POWER["avalanche"](ctx, BLUE, RED, move) == move.base_power
    cast(ctx, dex, "tackle", attacker=RED, defender=BLUE)
    assert VARIABLE_POWER["avalanche"](ctx, BLUE, RED, move) == move.base_power * 2


def test_last_respects_counts_the_fallen(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("basculegion", ("lastrespects",)),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    move = dex.moves["lastrespects"]
    assert VARIABLE_POWER["lastrespects"](ctx, RED, BLUE, move) == 50
    state.sides[0].hp[1] = 0
    state.sides[0].hp[2] = 0
    assert VARIABLE_POWER["lastrespects"](ctx, RED, BLUE, move) == 150


def test_sacred_sword_walks_past_the_stages(dex, config):
    from pkcm.engine.moves import compute_damage

    state = build(config, a_set("samurotthisui", ("sacredsword",)),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    before = max(compute_damage(ctx, RED, BLUE, dex.moves["sacredsword"], crit=False)[0]
                 for _ in range(6))
    mutate.boost(ctx, BLUE, {"def": 6})
    after = max(compute_damage(ctx, RED, BLUE, dex.moves["sacredsword"], crit=False)[0]
                for _ in range(6))
    # +6 Defense would quarter it; ignored, only the rolls separate the two.
    assert after > before * 0.8, (before, after)


def test_triple_axel_climbs(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("meowscarada", ("tripleaxel",)),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    from types import SimpleNamespace

    powers = [VARIABLE_POWER["tripleaxel"](ctx, RED, BLUE,
                                           SimpleNamespace(hit_index=index))
              for index in range(3)]   # what the hit loop stamps per hit
    assert powers == [20, 40, 60]


def test_scald_thaws_what_it_hits(dex, config):
    state = build(config, a_set("milotic", ("scald",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    state.sides[1].status[0] = "frz"
    cast(ctx, dex, "scald")
    assert state.sides[1].status[0] != "frz"


def test_acrobatics_doubles_empty_handed(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("gyarados", ("acrobatics",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    move = dex.moves["acrobatics"]
    assert VARIABLE_POWER["acrobatics"](ctx, RED, BLUE, move) == move.base_power * 2


def test_temper_flare_reads_the_failure_flag(dex, config):
    from pkcm.engine.moves import VARIABLE_POWER

    state = build(config, a_set("garchomp", ("temperflare",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    move = dex.moves["temperflare"]
    assert VARIABLE_POWER["temperflare"](ctx, RED, BLUE, move) == move.base_power
    state.sides[0].volatiles[0]["lastmovefailed"] = True
    assert VARIABLE_POWER["temperflare"](ctx, RED, BLUE, move) == move.base_power * 2


def test_steel_beam_pays_in_blood(dex, config):
    state = build(config, a_set("archaludon", ("steelbeam",)), a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    full = state.sides[0].hp[0]
    cast(ctx, dex, "steelbeam")
    assert state.sides[0].hp[0] <= full - full // 2 + 1, "half the user's max HP"


def test_a_crash_move_that_misses_hurts_its_user(dex, config):
    state = build(config, a_set("meowscarada", ("highjumpkick",)),
                  a_set("snorlax", ("tackle",)))
    ctx = make_context(state)
    mutate.boost(ctx, BLUE, {"evasion": 6})
    full = state.sides[0].hp[0]
    for _ in range(12):
        cast(ctx, dex, "highjumpkick")
        if any(e.kind == "crash" for e in ctx.log):
            break
    assert any(e.kind == "crash" for e in ctx.log), "no miss in twelve casts at +6 evasion?"
    assert state.sides[0].hp[0] < full
