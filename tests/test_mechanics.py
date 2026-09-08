"""Status conditions, stat stages, weather, screens, hazards and move effects.

These exercise the hook system rather than the turn loop: each case sets up a
position, runs one move, and checks what the mechanic did.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.battle import make_context, step
from pkcm.engine.effects import registered
from pkcm.engine.moves import use_move
from pkcm.engine.mutate import effective_stat
from pkcm.engine.pokemon import PokemonSet
from pkcm.engine.state import BattleConfig, legal_actions, new_battle

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


# --------------------------------------------------------------------------- #
# Aurora Veil, and the four things it shares with the other two screens
#
# hk asked one question -- "snow only, and does it halve both kinds?" -- and it
# halved both kinds in clear weather, in rain, in sand and in sun. Four more
# came out of the same look, and three of them were Reflect and Light Screen's
# too. Every one of them is written down in the move's own shipped data.
# --------------------------------------------------------------------------- #


def _snowed(config, ours=("auroraveil", "splash", "reflect", "protect")):
    state = build(config, a_set("ninetalesalola", ours),
                  a_set("garchomp", ("bodyslam", "shadowball", "splash",
                                     "focusenergy")))
    state.field.weather = "snowscape"
    state.field.weather_turns = 12
    return state


def test_aurora_veil_needs_snow(dex, config):
    for weather in (None, "raindance", "sandstorm", "sunnyday"):
        state = _snowed(config)
        state.field.weather = weather
        state.field.weather_turns = 8 if weather else 0
        state, _ = step(state, Action.move(0), Action.move(2))
        assert "auroraveil" not in state.sides[0].conditions, weather

    state, _ = step(_snowed(config), Action.move(0), Action.move(2))
    assert "auroraveil" in state.sides[0].conditions


def test_aurora_veil_halves_both_kinds(dex, config):
    from pkcm.engine.moves import compute_damage

    state = _snowed(config)
    ctx = make_context(state)
    bare = [compute_damage(ctx, BLUE, RED, dex.moves[m], crit=False)[0]
            for m in ("bodyslam", "shadowball")]

    state, _ = step(state, Action.move(0), Action.move(2))
    assert "auroraveil" in state.sides[0].conditions
    ctx = make_context(state)
    veiled = [compute_damage(ctx, BLUE, RED, dex.moves[m], crit=False)[0]
              for m in ("bodyslam", "shadowball")]

    for kind, before, after in zip(("physical", "special"), bare, veiled):
        assert 0.45 <= after / before <= 0.55, f"{kind}: {before} -> {after}"


def test_a_critical_hit_ignores_every_screen(dex, config):
    """The damage path passes ``crit`` and no screen was reading it."""
    from pkcm.engine.moves import compute_damage

    for screen in ("auroraveil", "reflect"):
        state = _snowed(config)
        state.sides[0].conditions[screen] = 5
        ctx = make_context(state)
        plain = compute_damage(ctx, BLUE, RED, dex.moves["bodyslam"], crit=False)[0]
        crit = compute_damage(ctx, BLUE, RED, dex.moves["bodyslam"], crit=True)[0]
        assert crit > plain * 1.4, f"{screen} softened a critical hit"


def test_aurora_veil_and_reflect_do_not_compound(dex, config):
    from pkcm.engine.moves import compute_damage

    def physical(*screens):
        state = _snowed(config)
        for screen in screens:
            state.sides[0].conditions[screen] = 5
        return compute_damage(make_context(state), BLUE, RED,
                              dex.moves["bodyslam"], crit=False)[0]

    assert physical("auroraveil", "reflect") == physical("auroraveil")
    assert physical("auroraveil") < physical()


def test_screens_are_two_thirds_in_doubles(dex):
    """The move's own data: 0.66x in a Double Battle, and it was 0.5x."""
    from pkcm.engine.moves import compute_damage

    doubles = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                           battle_format="doubles")
    bench = [a_set(s, ("tackle",)) for s in
             ("snorlax", "pikachu", "starmie", "gengar")]
    team = tuple([a_set("ninetalesalola", ("splash",)),
                  a_set("snorlax", ("splash",))] + bench)
    foes = tuple([a_set("garchomp", ("bodyslam",)),
                  a_set("snorlax", ("splash",))] + bench)
    state = new_battle(doubles, (team, foes), seed=7)
    state = step(state, Action.select(0, 1, 2, 3), Action.select(0, 1, 2, 3))[0]

    bare = compute_damage(make_context(state), BLUE, RED,
                          dex.moves["bodyslam"], crit=False)[0]
    state.sides[0].conditions["reflect"] = 5
    screened = compute_damage(make_context(state), BLUE, RED,
                              dex.moves["bodyslam"], crit=False)[0]
    assert 0.63 <= screened / bare <= 0.70, f"{bare} -> {screened}"


@pytest.mark.parametrize("breaker", ["brickbreak", "psychicfangs", "ragingbull"])
def test_the_screen_breakers_come_through_rather_than_under(dex, config, breaker):
    """Raging Bull ran from ``_after_effects``: after its own damage."""
    def swing(screen):
        state = build(config, a_set("garchomp", (breaker,)),
                      a_set("snorlax", ("splash",)))
        if screen:
            state.sides[1].conditions[screen] = 5
        whole = state.sides[1].hp[0]
        ctx = make_context(state)
        use_move(ctx, RED, dex.moves[breaker], defender=BLUE)
        return whole - state.sides[1].hp[0], dict(state.sides[1].conditions)

    clear, _ = swing(None)
    for screen in ("reflect", "auroraveil"):
        took, left = swing(screen)
        assert screen not in left, f"{breaker} left {screen} standing"
        assert took == clear, f"{breaker} was softened by the {screen} it broke"


# --------------------------------------------------------------------------- #
# What the descriptions said and the engine did not
#
# ``scripts/clause_check.py`` reads the 621 distinct sentences in the 497 move
# descriptions and holds each against every move that carries it. These are the
# ones it caught: each was a sentence in ``desc`` with no field behind it, which
# is exactly the shape the field-driven sweep cannot see.
# --------------------------------------------------------------------------- #


def test_ingrain_restores_a_sixteenth_every_turn(dex, config):
    """The volatile existed for the trapping and had no residual at all."""
    state = build(config, a_set("mew", ("ingrain", "splash")),
                  a_set("snorlax", ("splash",)))
    whole = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = whole // 2
    before = state.sides[0].hp[0]

    state, _ = step(state, Action.move(0), Action.move(0))

    assert state.sides[0].hp[0] - before == whole // 16
    assert state.sides[0].has_volatile(0, "ingrain"), "and it still holds"


def test_spirit_shackle_holds_what_it_hits(dex, config):
    """Same first sentence as Block and Mean Look, and no implementation."""
    state = build(config, a_set("mew", ("spiritshackle", "splash")),
                  a_set("milotic", ("splash",)))
    ctx = make_context(state)
    cast(ctx, dex, "spiritshackle")
    assert state.sides[1].has_volatile(0, "trapped")


def test_big_root_pays_more_than_the_drain_moves(dex, config):
    """Aqua Ring, Leech Seed and Strength Sap all name it; none of them asked."""
    def gained(move_id, item, target="snorlax"):
        state = build(config, a_set("mew", (move_id, "splash"), item=item),
                      a_set(target, ("splash",)))
        state.sides[0].hp[0] = 1
        state, _ = step(state, Action.move(0), Action.move(0))
        if move_id in ("aquaring",):          # the ring pays at end of turn
            return state.sides[0].hp[0] - 1
        return state.sides[0].hp[0] - 1

    for move_id in ("aquaring", "strengthsap"):
        plain, rooted = gained(move_id, None), gained(move_id, "bigroot")
        assert 1.25 <= rooted / plain <= 1.35, f"{move_id}: {plain} -> {rooted}"


def test_supercell_slam_flattens_a_minimized_target(dex, config):
    """The seventh of seven, and the only one left off the list."""
    from pkcm.engine.moveeffects import MINIMIZE_PUNISHERS

    assert "supercellslam" in MINIMIZE_PUNISHERS

    def damage(move_id):
        state = build(config, a_set("mew", (move_id, "splash")),
                      a_set("snorlax", ("minimize",)))
        ctx = make_context(state)
        cast(ctx, dex, "minimize", attacker=BLUE, defender=BLUE)
        plain = build(config, a_set("mew", (move_id, "splash")),
                      a_set("snorlax", ("splash",)))
        from pkcm.engine.moves import compute_damage
        return (compute_damage(ctx, RED, BLUE, dex.moves[move_id], crit=False)[0],
                compute_damage(make_context(plain), RED, BLUE,
                               dex.moves[move_id], crit=False)[0])

    small, upright = damage("supercellslam")
    assert 1.9 <= small / upright <= 2.1, f"{upright} -> {small}"


def test_the_binding_moves_run_four_or_five_turns(dex, config):
    """The roll was four to seven, and the data says four or five."""
    from pkcm.engine import tactics

    lengths = set()
    for seed in range(24):
        state = build(config, a_set("mew", ("wrap", "splash")),
                      a_set("snorlax", ("splash",)))
        state = new_battle(config, (tuple([a_set("mew", ("wrap", "splash"))]
                                          + [a_set(s, ("tackle",)) for s in
                                             ("pikachu", "starmie")]),
                                    tuple([a_set("snorlax", ("splash",))]
                                          + [a_set(s, ("tackle",)) for s in
                                             ("pikachu", "starmie")])), seed=seed)
        state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
        state, _ = step(state, Action.move(0), Action.move(0))
        if not state.sides[1].has_volatile(0, "partiallytrapped"):
            continue
        ticks = 1
        while state.sides[1].has_volatile(0, "partiallytrapped") and ticks < 9:
            state, _ = step(state, Action.move(1), Action.move(0))
            if state.sides[1].has_volatile(0, "partiallytrapped"):
                ticks += 1
        lengths.add(ticks)
    assert lengths and lengths <= {4, 5}, lengths


def test_grip_claw_holds_for_seven_and_reads_the_binders_hand(dex, config):
    """It was read off the target, and it gave five turns rather than seven."""
    from pkcm.engine import tactics
    from pkcm.engine.battle import make_context

    state = build(config, a_set("mew", ("wrap", "splash"), item="gripclaw"),
                  a_set("snorlax", ("splash",)))
    ctx = make_context(state)
    tactics.start_trapping(ctx, RED, BLUE, dex.moves["wrap"])
    assert state.sides[1].volatiles[0]["partiallytrapped"]["turns"] == 8

    on_the_target = build(config, a_set("mew", ("wrap", "splash")),
                          a_set("snorlax", ("splash",), item="gripclaw"))
    ctx = make_context(on_the_target)
    tactics.start_trapping(ctx, RED, BLUE, dex.moves["wrap"])
    held = on_the_target.sides[1].volatiles[0]["partiallytrapped"]["turns"]
    assert held in (5, 6), "the target's own Grip Claw does nothing"


def test_binding_band_takes_a_sixth_instead_of_an_eighth(dex, config):
    from pkcm.engine import tactics
    from pkcm.engine.battle import make_context

    def per_turn(item):
        state = build(config, a_set("mew", ("wrap", "splash"), item=item),
                      a_set("snorlax", ("splash",)))
        ctx = make_context(state)
        tactics.start_trapping(ctx, RED, BLUE, dex.moves["wrap"])
        whole = state.sides[1].hp[0]
        state, _ = step(state, Action.move(1), Action.move(0))
        return whole - state.sides[1].hp[0], state.pokemon(1, 0).max_hp

    took, whole = per_turn("bindingband")
    plain, _ = per_turn(None)
    assert took == whole // 6 and plain == whole // 8, f"{took} vs {plain} of {whole}"


def test_dire_claw_and_tri_attack_pick_a_status(dex, config):
    """Both export a bare chance and keep the pick in code; both did nothing."""
    from pkcm.engine.moves import RANDOM_SECONDARY_STATUS

    for move_id, wanted in RANDOM_SECONDARY_STATUS.items():
        seen = set()
        for seed in range(60):
            state = new_battle(config,
                               (tuple([a_set("mew", (move_id, "splash"))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")]),
                                tuple([a_set("milotic", ("splash",))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")])), seed=seed)
            state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
            state, _ = step(state, Action.move(0), Action.move(0))
            if state.sides[1].status[0]:
                seen.add(state.sides[1].status[0])
        assert seen, f"{move_id} never inflicted anything in sixty casts"
        assert seen <= set(wanted), f"{move_id} inflicted {seen}, not {wanted}"


def test_thunder_and_hurricane_read_the_sky(dex, config):
    """Two sentences each, and nothing behind either: 70% in every weather."""
    from pkcm.engine.moves import WEATHER_ACCURACY, connects

    for move_id in ("thunder", "hurricane"):
        assert move_id in WEATHER_ACCURACY
        state = build(config, a_set("mew", (move_id,)), a_set("snorlax", ("splash",)))
        ctx = make_context(state)

        state.field.weather, state.field.weather_turns = "raindance", 8
        assert all(connects(ctx, RED, BLUE, dex.moves[move_id]) for _ in range(30)), \
            "in rain it does not check accuracy"

        state.field.weather = "sunnyday"
        hits = sum(connects(ctx, RED, BLUE, dex.moves[move_id]) for _ in range(200))
        assert 0.3 <= hits / 200 <= 0.7, f"in sun it is 50%, got {hits}/200"


def test_the_protect_chain_is_rolled_once_per_cast(dex, config):
    """The variants spent two stalls on one cast: their handler and the branch
    in ``_apply_status_move`` both called ``_apply_protect``."""
    state = build(config, a_set("mew", ("banefulbunker", "splash")),
                  a_set("snorlax", ("bodyslam",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    stall = state.sides[0].volatiles[0].get("stall")
    assert stall == {"count": 1}, f"one cast, one stall: {stall}"


def test_endure_and_the_guards_are_in_the_same_chain(dex, config):
    """All eight name the other seven; three of them never rolled."""
    for move_id in ("endure", "quickguard", "wideguard"):
        worked = 0
        for seed in range(40):
            state = new_battle(config,
                               (tuple([a_set("mew", (move_id, "splash"))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")]),
                                tuple([a_set("snorlax", ("bodyslam",))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")])), seed=seed)
            state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
            state, _ = step(state, Action.move(0), Action.move(0))
            state, log = step(state, Action.move(0), Action.move(0))
            worked += not any(e.kind == "move_failed" and (e.side or 0) == 0
                              for e in log)
        assert 3 <= worked <= 25, f"{move_id} worked {worked} of 40 second casts"


def test_a_one_hit_ko_cannot_touch_sturdy(dex, config):
    def landed(ability):
        for seed in range(30):
            state = new_battle(config,
                               (tuple([a_set("mew", ("fissure",))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")]),
                                tuple([a_set("snorlax", ("splash",), ability=ability)]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")])), seed=seed)
            state = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
            state, _ = step(state, Action.move(0), Action.move(0))
            if state.sides[1].hp[0] == 0:
                return True
        return False

    assert landed("__none__"), "it lands on something ordinary"
    assert not landed("sturdy")


def test_battle_armor_stops_an_always_critical_move(dex, config):
    from pkcm.engine.moves import rolls_crit

    for ability, expected in (("__none__", True), ("battlearmor", False),
                              ("shellarmor", False)):
        state = build(config, a_set("mew", ("frostbreath",)),
                      a_set("snorlax", ("splash",), ability=ability))
        ctx = make_context(state)
        assert rolls_crit(ctx, RED, BLUE, dex.moves["frostbreath"]) is expected, ability


def test_crash_damage_is_rounded_down(dex, config):
    """Half of 207 is 103, and it was taking 104."""
    from pkcm.engine.state import BOOST_INDEX

    for move_id in ("highjumpkick", "axekick", "supercellslam"):
        state = build(config, a_set("mew", (move_id, "splash")),
                      a_set("snorlax", ("splash",)))
        state.sides[1].boosts[0][BOOST_INDEX["evasion"]] = 6
        whole = state.pokemon(0, 0).max_hp
        for seed in range(20):
            trial = new_battle(config,
                               (tuple([a_set("mew", (move_id, "splash"))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")]),
                                tuple([a_set("snorlax", ("splash",))]
                                      + [a_set(s, ("tackle",)) for s in
                                         ("pikachu", "starmie")])), seed=seed)
            trial = step(trial, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]
            trial.sides[1].boosts[0][BOOST_INDEX["evasion"]] = 6
            before = trial.sides[0].hp[0]
            trial, log = step(trial, Action.move(0), Action.move(0))
            if any(e.kind == "crash" for e in log):
                assert before - trial.sides[0].hp[0] == whole // 2, move_id
                break
        else:
            raise AssertionError(f"{move_id} never missed in twenty tries")


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


# --------------------------------------------------------------------------- #
# Sucker Punch
# --------------------------------------------------------------------------- #


def _sucker_turn(config, their_moves, their_index, their_sp):
    """One turn of Mawile's Sucker Punch into whatever Primarina picked."""
    ours = a_set("mawile", ("suckerpunch", "swordsdance"), sp=(0, 32, 0, 0, 0, 0))
    theirs = a_set("primarina", their_moves, sp=their_sp)
    state = build(config, ours, theirs)
    _, events = step(state, Action.move(0), Action.move(their_index))
    return [(e.kind, getattr(e, "detail", None)) for e in events]


def test_sucker_punch_loses_to_an_equal_priority_move(config):
    """Reported from a real game, and the reason it was unplayable there.

    Aqua Jet is +1 like Sucker Punch, so Speed breaks the tie and the faster
    Primarina moves first. By the time Sucker Punch runs its target has
    already gone, which is exactly when the move does nothing. The engine had
    it landing every time -- 70 base power at +1 that never fails is a much
    better move than the one in the game.
    """
    events = _sucker_turn(config, ("aquajet", "moonblast"), 0, (0, 0, 0, 0, 0, 32))
    assert ("move_failed", "target has already moved") in events


def test_sucker_punch_fails_against_a_status_move(config):
    """Whoever is faster: the target has to be attacking."""
    events = _sucker_turn(config, ("calmmind", "moonblast"), 0, (0, 0, 0, 0, 0, 0))
    assert ("move_failed", "target is not attacking") in events


def test_sucker_punch_fails_against_a_switch(config):
    ours = a_set("mawile", ("suckerpunch", "swordsdance"), sp=(0, 32, 0, 0, 0, 0))
    theirs = a_set("primarina", ("moonblast", "aquajet"))
    state = build(config, ours, theirs)

    _, events = step(state, Action.move(0), Action.switch(1))

    assert ("move_failed", "target is not attacking") in \
        [(e.kind, getattr(e, "detail", None)) for e in events]


def test_sucker_punch_still_lands_on_a_slower_attacker(config):
    """The other half. A move that always failed would be as wrong as one that
    always landed."""
    events = _sucker_turn(config, ("moonblast", "aquajet"), 0, (0, 0, 0, 0, 0, 0))
    kinds = [kind for kind, _ in events]
    assert "damage" in kinds
    assert not [d for kind, d in events if kind == "move_failed"]


def test_a_failed_sucker_punch_still_costs_its_pp(config):
    """It fails after being used, not before -- the same as in the game."""
    ours = a_set("mawile", ("suckerpunch", "swordsdance"), sp=(0, 32, 0, 0, 0, 0))
    theirs = a_set("primarina", ("calmmind", "moonblast"))
    state = build(config, ours, theirs)
    before = state.sides[0].pp[0][0]

    state, _ = step(state, Action.move(0), Action.move(0))

    assert state.sides[0].pp[0][0] == before - 1


# --------------------------------------------------------------------------- #
# The rest of the family: conditions Showdown keeps in handler code, which the
# data-driven audit in ``move_support`` cannot see. Every one of these made its
# move strictly better than the game's version.
# --------------------------------------------------------------------------- #


def _turn(config, red, blue, ours=0, theirs=0, warmup=0):
    state = build(config, red, blue)
    for _ in range(warmup):
        state, _ = step(state, Action.move(1), Action.move(theirs))
    state, events = step(state, Action.move(ours), Action.move(theirs))
    return state, [(e.kind, getattr(e, "detail", None)) for e in events]


def test_fake_out_only_works_on_the_turn_it_came_in(config):
    """A free flinch every turn is one of the best moves in the game. It is
    supposed to be an opening."""
    user = a_set("kangaskhan", ("fakeout", "tackle"), sp=(0, 32, 0, 0, 0, 32))
    foe = a_set("snorlax", ("rest",))

    _, first = _turn(config, user, foe)
    _, later = _turn(config, user, foe, warmup=1)

    assert ("cant_move", "flinch") in first, "it works the turn it arrives"
    assert ("move_failed", "not the turn it came in") in later


def test_first_impression_only_works_on_the_turn_it_came_in(config):
    user = a_set("golisopod", ("firstimpression", "tackle"), sp=(0, 32, 0, 0, 0, 32))
    foe = a_set("snorlax", ("rest",))

    _, first = _turn(config, user, foe)
    _, later = _turn(config, user, foe, warmup=1)

    assert ("damage", None) in first, "ninety power at +2, once"
    assert ("move_failed", "not the turn it came in") in later


def test_upper_hand_only_interrupts_a_priority_attack(config):
    """Narrower than Sucker Punch: the target has to be moving early, not just
    attacking."""
    user = a_set("hitmonchan", ("upperhand", "tackle"), sp=(0, 32, 0, 0, 0, 32))
    foe = a_set("snorlax", ("tackle", "quickattack"))

    _, plain = _turn(config, user, foe, theirs=0)
    _, priority = _turn(config, user, foe, theirs=1)

    assert ("move_failed", "target is not moving first") in plain
    assert ("damage", None) in priority


def test_snore_needs_its_user_asleep(config):
    user = a_set("snorlax", ("snore", "tackle"), sp=(0, 32, 0, 0, 0, 32))
    foe = a_set("pikachu", ("rest",))

    state = build(config, user, foe)
    _, awake = step(state, Action.move(0), Action.move(0))
    assert ("move_failed", "user is awake") in \
        [(e.kind, getattr(e, "detail", None)) for e in awake]

    state = build(config, user, foe)
    state.sides[0].status[0] = "slp"
    state.sides[0].status_data[0] = {"sleep": 3}
    _, asleep = step(state, Action.move(0), Action.move(0))
    assert ("damage", None) in \
        [(e.kind, getattr(e, "detail", None)) for e in asleep]


def test_last_resort_waits_for_every_other_move(config):
    user = a_set("zangoose", ("lastresort", "tackle"), sp=(0, 32, 0, 0, 0, 32))
    foe = a_set("snorlax", ("rest",))

    _, straight_away = _turn(config, user, foe)
    _, after_tackle = _turn(config, user, foe, warmup=1)

    assert ("move_failed", "still has moves it has not used") in straight_away
    assert ("damage", None) in after_tackle


def test_fake_out_into_last_resort_off_two_moves(config):
    """The line the pair of conditions exists for.

    Mega Kangaskhan carrying nothing but Fake Out and Last Resort: the Fake
    Out opens, and having been used it is the only other move Last Resort was
    waiting on, so from the second turn Last Resort is live. Both halves have
    to be right for this to work and for it not to work a turn early.
    """
    user = a_set("kangaskhan", ("fakeout", "lastresort"), ability="scrappy",
                 item="kangaskhanite", sp=(0, 32, 0, 0, 0, 32))
    state = build(config, user, a_set("snorlax", ("rest",)))

    state, opening = step(state, Action.move(0, mega=True), Action.move(0))
    marks = [(e.kind, getattr(e, "detail", None)) for e in opening]
    assert ("cant_move", "flinch") in marks, "Fake Out opens"
    assert not [d for kind, d in marks if kind == "move_failed"]

    state, second = step(state, Action.move(1), Action.move(0))
    marks = [(e.kind, getattr(e, "detail", None)) for e in second]
    assert ("damage", None) in marks, "and Last Resort is live from here"
    assert not [d for kind, d in marks if kind == "move_failed"]


def test_last_resort_alone_has_nothing_to_wait_for(config):
    """One move and it is Last Resort: it never comes on, rather than always
    being on."""
    user = a_set("zangoose", ("lastresort",), sp=(0, 32, 0, 0, 0, 32))
    state = build(config, user, a_set("snorlax", ("rest",)))

    _, events = step(state, Action.move(0), Action.move(0))

    assert ("move_failed", "nothing else to run out of") in \
        [(e.kind, getattr(e, "detail", None)) for e in events]


def test_the_counters_reset_when_the_user_leaves_the_field(config):
    """Both conditions are per-visit: Fake Out works again after a switch, and
    Last Resort has to earn it again."""
    user = a_set("kangaskhan", ("fakeout", "tackle"), sp=(0, 32, 0, 0, 0, 32))
    state = build(config, user, a_set("snorlax", ("rest",)))
    state, _ = step(state, Action.move(1), Action.move(0))     # tackle, so it has moved
    state, _ = step(state, Action.switch(1), Action.move(0))
    state, _ = step(state, Action.switch(0), Action.move(0))

    _, events = step(state, Action.move(0), Action.move(0))

    assert ("cant_move", "flinch") in \
        [(e.kind, getattr(e, "detail", None)) for e in events]


# --------------------------------------------------------------------------- #
# Order inside a priority bracket
# --------------------------------------------------------------------------- #


class _SureThing:
    """The real cursor with every chance roll succeeding."""

    def __init__(self, inner):
        self._inner = inner

    def chance(self, num, den):
        return True

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _order(config, red, blue, red_move=0, blue_move=0, sure=False):
    """Who is named first in the turn's move_used events."""
    from pkcm.engine.battle import make_context, _move_order, _actors

    state = build(config, red, blue)
    ctx = make_context(state)
    if sure:
        ctx.cursor = _SureThing(ctx.cursor)
    choices = ((Action.move(red_move),), (Action.move(blue_move),))
    return _move_order(ctx, choices, _actors(state))[0][0]


def test_stall_moves_last_inside_its_bracket(config):
    """Sableye's drawback ability was in the inert list -- registered as doing
    nothing -- so the engine ran it at its own Speed, which is a straight
    upgrade over the ability the game gives it."""
    quick = a_set("sableye", ("tackle",), ability="stall", sp=(0, 0, 0, 0, 0, 32))
    slow = a_set("snorlax", ("tackle",), sp=(0, 0, 0, 0, 0, 0))

    assert _order(config, quick, slow) == 1, "the slower Snorlax goes first"

    without = a_set("sableye", ("tackle",), ability="keeneye", sp=(0, 0, 0, 0, 0, 32))
    assert _order(config, without, slow) == 0, "and without Stall it does not"


def test_stall_does_not_drop_out_of_its_bracket(config):
    """Last in its bracket, not last overall: a Sucker Punch still beats a
    Tackle."""
    staller = a_set("sableye", ("suckerpunch", "tackle"), ability="stall",
                    sp=(0, 0, 0, 0, 0, 0))
    faster = a_set("snorlax", ("tackle",), sp=(0, 0, 0, 0, 0, 32))

    assert _order(config, staller, faster) == 0


def test_quick_claw_does_not_beat_a_priority_move(config):
    """It moves its holder to the front of its own bracket. It was adding to
    the priority itself, which let a Tackle go before an Aqua Jet."""
    holder = a_set("snorlax", ("tackle",), item="quickclaw", sp=(0, 0, 0, 0, 0, 0))
    priority_user = a_set("pikachu", ("quickattack", "tackle"), sp=(0, 0, 0, 0, 0, 32))

    assert _order(config, holder, priority_user, blue_move=0, sure=True) == 1, \
        "Quick Attack is a bracket above, and a Quick Claw does not reach it"
    assert _order(config, holder, priority_user, blue_move=1, sure=True) == 0, \
        "in the same bracket it does move first, slower or not"


# --------------------------------------------------------------------------- #
# Sure hits, which are not the same as a hundred percent
# --------------------------------------------------------------------------- #


def _lands(config, attacker, defender, move_id, evasion, tries=400,
           minimised=False):
    from pkcm.engine.battle import make_context
    from pkcm.engine.moves import connects
    from pkcm.engine.state import BOOST_INDEX

    state = build(config, attacker, defender)
    state.sides[1].boosts[0] = list(state.sides[1].boosts[0])
    state.sides[1].boosts[0][BOOST_INDEX["evasion"]] = evasion
    if minimised:
        # The stage and the volatile are different things: Double Team raises
        # one, Minimize raises both, and it is the volatile that says a Body
        # Slam cannot miss.
        state.sides[1].volatiles[0]["minimize"] = True
    ctx = make_context(state)
    return sum(connects(ctx, RED, BLUE, ctx.state.config.dex.moves[move_id])
               for _ in range(tries)), tries


def test_no_guard_ignores_evasion(config):
    """It returned 100 from modify_accuracy, and the evasion stages were then
    multiplied over the top of it -- so a No Guard Dynamic Punch landed a third
    of the time against a Minimize. Nothing caught it because the format was
    thought to ban evasion moves, so nothing ever raised the stage."""
    hits, tries = _lands(config,
                         a_set("machamp", ("dynamicpunch",), ability="noguard"),
                         a_set("pikachu", ("minimize",)), "dynamicpunch", 6)
    assert hits == tries, f"{hits}/{tries}"


def test_a_minimised_target_cannot_dodge_what_flattens_it(config):
    """Double damage and a sure hit are one rule. Only the damage was here."""
    hits, tries = _lands(config, a_set("snorlax", ("bodyslam",)),
                         a_set("pikachu", ("minimize",)), "bodyslam", 4,
                         minimised=True)
    assert hits == tries, f"{hits}/{tries}"

    # and something not on the list still has to roll for it
    misses, tries = _lands(config, a_set("snorlax", ("focusblast",)),
                           a_set("pikachu", ("minimize",)), "focusblast", 4,
                           minimised=True)
    assert misses < tries * 0.6, f"{misses}/{tries}"


def test_ordinary_accuracy_still_answers_to_evasion(config):
    """The sure-hit path must not swallow the ordinary one."""
    high, tries = _lands(config, a_set("snorlax", ("focusblast",)),
                         a_set("pikachu", ("tackle",)), "focusblast", 0)
    low, _ = _lands(config, a_set("snorlax", ("focusblast",)),
                    a_set("pikachu", ("tackle",)), "focusblast", 6)
    assert high > low * 2, (high, low)


def test_smack_down_lets_ground_moves_reach_a_flying_type(dex, config):
    """Grounding read as grounded, and then the type chart said 0 anyway.

    ``type_effectiveness`` asked ``is_grounded`` first, which correctly
    answered yes for a Corviknight that Smack Down had pinned -- and then
    handed the chart the Flying type it still carried, which returns 0 for
    Ground. The volatile was set for the rest of the battle and changed
    nothing, on a move whose whole purpose is that one interaction.
    """
    state = build(config, a_set("garchomp", ("smackdown", "earthquake")),
                  a_set("corviknight", ("roost",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[1].has_volatile(0, "smackdown")

    before = state.sides[1].hp[0]
    state, log = step(state, Action.move(1), Action.move(0))
    assert not any(e.kind == "immune" for e in log), log
    assert state.sides[1].hp[0] < before, "Earthquake reaches it now"


def test_a_flying_type_is_still_immune_to_ground_when_nothing_holds_it_down(dex, config):
    """The other half: the fix must not hand Ground moves a free pass."""
    state = build(config, a_set("garchomp", ("earthquake",)),
                  a_set("corviknight", ("roost",)))
    before = state.sides[1].hp[0]
    state, log = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "immune" for e in log), log
    assert state.sides[1].hp[0] == before


def test_burn_up_can_be_used_by_a_fire_type(dex, config):
    """The precondition compared a capitalised name against lower-case types.

    ``state.types`` answers ``('fire',)``; the check asked for ``"Fire"``. So
    Burn Up refused itself on every Pokemon that could use it and logged
    "not Fire" while standing there as one, and the handler's filter would
    have left the type on even if the move had run.
    """
    state = build(config, a_set("cinderace", ("burnup", "pyroball")),
                  a_set("snorlax", ("splash",)))
    assert "fire" in state.types(0, 0)
    state, log = step(state, Action.move(0), Action.move(0))
    assert not any(e.kind == "move_failed" and e.detail == "not Fire"
                   for e in log), log
    assert "fire" not in state.types(0, 0)


def test_a_move_stopped_by_protect_does_not_count_as_having_failed(dex, config):
    """This test used to assert the opposite, and the moves that read the flag
    carve Protect out by name: "A move that was blocked by Baneful Bunker,
    Detect, King's Shield, Protect, Spiky Shield, Crafty Shield, Mat Block,
    Quick Guard, or Wide Guard will not double this move's power." Showdown
    has a sentinel for exactly this. Marking it failed doubled Temper Flare
    off the turn it should have left alone -- 92 against 47."""
    state = build(config, a_set("cinderace", ("pyroball", "temperflare")),
                  a_set("snorlax", ("protect",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].volatiles[0].get("lastmovefailed") is not True

    # A refusal that is not a block still counts: Thunder Wave at a Ground type.
    ground = build(config, a_set("pikachu", ("thunderwave", "tackle")),
                   a_set("garchomp", ("splash",)))
    ground, _ = step(ground, Action.move(0), Action.move(0))
    assert ground.sides[0].volatiles[0].get("lastmovefailed") is True


def test_assurance_sees_damage_that_was_not_a_move(dex, config):
    """Its ledger was Counter's, which holds move hits alone -- and in singles
    a target is hurt before our turn by recoil, hazards or our own Rocky
    Helmet, never by a move."""
    state = build(config, a_set("kingambit", ("assurance",)),
                  a_set("staraptor", ("bravebird",)))
    state.sides[1].hp[0] -= 20
    state.sides[1].volatiles[0]["tookdamagethisturn"] = True
    ctx = make_context(state)
    from pkcm.engine.moves import _target_took_damage
    assert _target_took_damage(ctx, (0, 0), (1, 0))


def test_fairy_lock_holds_both_sides(dex, config):
    """It held the caster alone, and permanently.

    ``SPECIAL_MOVES["fairylock"] = _apply_volatile("trapped")`` put the hold on
    the user rather than on the field, and a ``trapped`` volatile with no
    source never expires -- so the move's own user was the only Pokemon it ever
    stopped, for the rest of the battle. The pseudo-weather it also sets was
    read by nobody.
    """
    state = build(config, a_set("clefable", ("fairylock", "moonblast")),
                  a_set("snorlax", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert "fairylock" in state.field.rooms

    for player in (0, 1):
        switches = [one for one in legal_actions(state, player)
                    if one.kind is ActionKind.SWITCH]
        assert not switches, f"side {player} can still leave"


def test_a_pseudo_weather_lasts_as_long_as_its_data_says(dex, config):
    """Five was hardcoded, which is right for Trick Room and wrong for Fairy
    Lock, whose own condition asks for two."""
    assert dex.moves["fairylock"].raw["condition"]["duration"] == 2
    state = build(config, a_set("clefable", ("fairylock", "moonblast")),
                  a_set("snorlax", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.field.rooms["fairylock"] == 1, "two turns, one of them spent"

    state, _ = step(state, Action.move(1), Action.move(0))
    assert "fairylock" not in state.field.rooms


def test_the_guards_and_safeguard_run_out(dex, config):
    """They were layers, like Spikes, so they went up once and stayed up.

    ``SIDE_CONDITION_DURATION`` held only the screens and Tailwind; anything
    else reaching ``add_side_condition`` was counted as a layer and never
    ticked. A permanent Safeguard is immunity to every status the opponent
    has for the rest of the battle, and a permanent Wide Guard turns off
    spread moves for good.
    """
    from pkcm.engine.conditions import SIDE_CONDITION_DURATION

    for name, turns in (("safeguard", 5), ("quickguard", 1), ("wideguard", 1)):
        assert SIDE_CONDITION_DURATION[name] == turns
        assert dex.moves[name].raw["condition"]["duration"] == turns

    state = build(config, a_set("clefable", ("safeguard", "splash")),
                  a_set("snorlax", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].conditions["safeguard"] == 4, "one of five spent"
    for _ in range(4):
        state, _ = step(state, Action.move(1), Action.move(0))
    assert "safeguard" not in state.sides[0].conditions


def test_wide_guard_is_gone_the_turn_after(dex, config):
    state = build(config, a_set("clefable", ("wideguard", "splash")),
                  a_set("snorlax", ("splash",)))
    state, _ = step(state, Action.move(0), Action.move(0))
    assert "wideguard" not in state.sides[0].conditions, "one turn only"
