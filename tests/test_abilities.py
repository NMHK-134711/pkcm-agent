"""Abilities, with the three that bent the framework first.

Mold Breaker, Corrosion and Poison Heal each break a naive implementation in a
different way, so each gets tested for the thing that would go wrong.
"""

from __future__ import annotations

import pytest

from pkcm.data.dex import Stat, load_dex
from pkcm.engine import mutate
from pkcm.engine.actions import Action
from pkcm.engine.battle import make_context, step
from pkcm.engine.effects import registered
from pkcm.engine.moves import chain_modify, compute_damage, use_move, X1_5
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


def a_set(species: str, ability: str, moves=("bodyslam",), **kwargs) -> PokemonSet:
    defaults = dict(nature="serious", sp=(0, 0, 0, 0, 0, 0))
    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      **{**defaults, **kwargs})


def build(config, red, blue):
    bench = [a_set(s, "__none__") for s in ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(config, (tuple([red] + bench), tuple([blue] + bench)), seed=7)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def cast(ctx, dex, move_id, attacker=RED, defender=BLUE):
    use_move(ctx, attacker, defender, dex.moves[move_id])


# --------------------------------------------------------------------------- #
# Mold Breaker
# --------------------------------------------------------------------------- #


def test_mold_breaker_makes_levitate_stop_floating(dex, config):
    """The failure mode: checking for Mold Breaker at some immunity tests only."""
    normal = build(config, a_set("garchomp", "roughskin", ("earthquake",)),
                   a_set("rotomheat", "levitate"))
    ctx = make_context(normal)
    _, effectiveness = compute_damage(ctx, RED, BLUE, dex.moves["earthquake"], crit=False)
    assert effectiveness == 0.0, "Levitate floats over Ground"

    breaker = build(config, a_set("garchomp", "moldbreaker", ("earthquake",)),
                    a_set("rotomheat", "levitate"))
    state, log = step(breaker, Action.move(0), Action.move(0))
    assert any(e.kind == "ability_suppressed" for e in log), log
    assert state.sides[1].hp[0] < state.pokemon(1, 0).max_hp, "Earthquake should land"


def test_mold_breaker_ignores_a_defensive_damage_modifier(dex, config):
    """Thick Fat halves Fire damage -- unless the attacker breaks moulds."""
    def fire_damage(attacker_ability: str) -> int:
        state = build(config, a_set("typhlosion", attacker_ability, ("flamethrower",)),
                      a_set("snorlax", "thickfat"))
        ctx = make_context(state)
        ctx.suppressed_abilities.clear()
        from pkcm.engine.moves import ignores_target_ability

        if ignores_target_ability(ctx, RED, dex.moves["flamethrower"]):
            ctx.suppressed_abilities.add(BLUE)
        return compute_damage(ctx, RED, BLUE, dex.moves["flamethrower"], crit=False)[0]

    assert fire_damage("moldbreaker") > fire_damage("blaze") * 1.7


def test_suppression_is_lifted_after_the_move(dex, config):
    state = build(config, a_set("garchomp", "moldbreaker", ("earthquake",)),
                  a_set("rotomheat", "levitate"))
    after, _ = step(state, Action.move(0), Action.move(0))
    ctx = make_context(after)
    assert ctx.ability_of(BLUE) == "levitate", "suppression must not outlive the move"


# --------------------------------------------------------------------------- #
# Corrosion
# --------------------------------------------------------------------------- #


def test_corrosion_poisons_a_steel_type(dex, config):
    state = build(config, a_set("salazzle", "corrosion", ("toxic",)),
                  a_set("skarmory", "sturdy"))
    ctx = make_context(state)
    cast(ctx, dex, "toxic")
    assert state.sides[1].status[0] == "tox"


def test_without_corrosion_steel_is_immune(dex, config):
    """Two separate walls stand between Toxic and a Steel type.

    Poison does not touch Steel on the type chart, and Steel cannot be poisoned
    even if something gets past that. Without Corrosion the first one stops it,
    so the log says ``immune`` rather than ``status_immune``.
    """
    state = build(config, a_set("salazzle", "oblivious", ("toxic",)),
                  a_set("skarmory", "sturdy"))
    ctx = make_context(state)
    cast(ctx, dex, "toxic")
    assert state.sides[1].status[0] is None
    assert any(e.kind == "immune" for e in ctx.log), ctx.log


def test_corrosion_has_to_clear_both_walls(dex, config):
    """The type chart one and the status one, or Toxic never lands on Steel."""
    from pkcm.engine.moves import type_effectiveness

    state = build(config, a_set("salazzle", "corrosion", ("toxic",)),
                  a_set("skarmory", "sturdy"))
    ctx = make_context(state)
    assert type_effectiveness(ctx, RED, BLUE, dex.moves["toxic"]) == 1.0

    plain = build(config, a_set("salazzle", "oblivious", ("toxic",)),
                  a_set("skarmory", "sturdy"))
    assert type_effectiveness(make_context(plain), RED, BLUE, dex.moves["toxic"]) == 0.0


def test_corrosion_does_not_hand_out_other_immunities(dex, config):
    """It empties the list for poison only -- a Fire type still cannot be burned."""
    state = build(config, a_set("salazzle", "corrosion", ("willowisp",)),
                  a_set("typhlosion", "blaze"))
    ctx = make_context(state)
    cast(ctx, dex, "willowisp")
    assert state.sides[1].status[0] is None


def test_corrosion_belongs_to_the_poisoner_not_the_poisoned(dex, config):
    """A Corrosion holder is not itself easier to poison."""
    state = build(config, a_set("skarmory", "sturdy", ("toxic",)),
                  a_set("salazzle", "corrosion"))
    ctx = make_context(state)
    cast(ctx, dex, "toxic")
    assert state.sides[1].status[0] is None, "Salazzle is Poison-type; still immune"


# --------------------------------------------------------------------------- #
# Poison Heal
# --------------------------------------------------------------------------- #


def test_poison_heal_turns_poison_damage_into_healing(dex, config):
    state = build(config, a_set("gliscor", "poisonheal", ("protect",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    mutate.set_status(ctx, RED, "tox")
    state.sides[0].hp[0] = 100
    state.rng = ctx.cursor.seal()

    before = state.sides[0].hp[0]
    state, log = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] > before, "poison should be healing it"
    assert any(e.kind == "heal" and e.detail == "poisonheal" for e in log), log
    assert not any(e.kind == "status_damage" for e in log)


def test_poison_heal_keeps_the_status(dex, config):
    state = build(config, a_set("gliscor", "poisonheal", ("protect",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    mutate.set_status(ctx, RED, "psn")
    state.rng = ctx.cursor.seal()
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].status[0] == "psn", "it is still poisoned, just not hurt by it"


def test_poison_heal_does_not_absorb_unrelated_damage(dex, config):
    state = build(config, a_set("gliscor", "poisonheal", ("protect",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    state.sides[0].hp[0] = 100
    mutate.apply_damage(ctx, RED, 20, "hazard_damage", detail="spikes")
    assert state.sides[0].hp[0] == 80


def test_magic_guard_blocks_all_indirect_damage(dex, config):
    state = build(config, a_set("alakazam", "magicguard", ("protect",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    state.sides[0].hp[0] = 100
    for kind in ("status_damage", "weather_damage", "hazard_damage", "recoil"):
        mutate.apply_damage(ctx, RED, 20, kind, detail=kind)
    assert state.sides[0].hp[0] == 100


# --------------------------------------------------------------------------- #
# A sample of the rest
# --------------------------------------------------------------------------- #


def test_intimidate_drops_attack_on_entry(dex, config):
    state = build(config, a_set("arcanine", "intimidate"), a_set("snorlax", "__none__"))
    assert state.sides[1].boost(0, "atk") == -1


def test_intimidate_is_blocked_by_a_substitute(dex, config):
    from pkcm.engine import effects as fx

    state = build(config, a_set("snorlax", "__none__", ("bodyslam",)),
                  a_set("arcanine", "intimidate"))
    # Intimidate already fired once when Arcanine entered during setup.
    before = state.sides[0].boost(0, "atk")

    ctx = make_context(state)
    mutate.add_volatile(ctx, RED, "substitute", hp=50)
    fx.notify(ctx, "switch_in", BLUE)

    assert state.sides[0].boost(0, "atk") == before, "the substitute takes it"
    assert any(e.kind == "immune" and e.detail == "substitute" for e in ctx.log)


def test_pinch_abilities_need_low_hp_and_the_right_type(dex, config):
    state = build(config, a_set("typhlosion", "blaze", ("flamethrower",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    healthy = effective_stat(ctx, RED, Stat.SPA, move=dex.moves["flamethrower"])

    state.sides[0].hp[0] = state.pokemon(0, 0).max_hp // 4
    hurt = effective_stat(ctx, RED, Stat.SPA, move=dex.moves["flamethrower"])
    assert hurt == chain_modify(healthy, X1_5)

    other_type = effective_stat(ctx, RED, Stat.SPA, move=dex.moves["shadowball"])
    assert other_type == healthy, "Blaze is Fire only"


def test_technician_reads_power_not_the_move(dex, config):
    state = build(config, a_set("scizor", "technician", ("bulletpunch", "xscissor")),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    weak = compute_damage(ctx, RED, BLUE, dex.moves["bulletpunch"], crit=False)[0]
    ctx2 = make_context(state)
    plain = build(config, a_set("scizor", "swarm", ("bulletpunch",)), a_set("snorlax", "__none__"))
    reference = compute_damage(make_context(plain), RED, BLUE, dex.moves["bulletpunch"],
                               crit=False)[0]
    assert weak > reference


def test_sturdy_survives_a_lethal_hit_from_full(dex, config):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)),
                  a_set("skarmory", "sturdy"))
    state.sides[1].hp[0] = state.pokemon(1, 0).max_hp
    ctx = make_context(state)
    from pkcm.engine.moves import compute_damage as damage

    result = damage(ctx, RED, BLUE, dex.moves["dragonclaw"], crit=False)[0]
    assert result < state.sides[1].hp[0], "Sturdy caps damage below the current HP"


def test_regenerator_heals_a_third_on_switch_out(dex, config):
    state = build(config, a_set("slowbro", "regenerator", ("bodyslam",)),
                  a_set("snorlax", "__none__"))
    full = state.pokemon(0, 0).max_hp
    state.sides[0].hp[0] = full // 3
    before = state.sides[0].hp[0]

    after, log = step(state, Action.switch(1), Action.move(0))
    assert after.sides[0].hp[0] == min(full, before + full // 3)
    assert any(e.kind == "heal" and e.detail == "regenerator" for e in log)


def test_natural_cure_clears_status_on_switch_out(dex, config):
    state = build(config, a_set("starmie", "naturalcure", ("bodyslam",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    mutate.set_status(ctx, RED, "brn")
    state.rng = ctx.cursor.seal()
    assert state.sides[0].status[0] == "brn"

    after, _ = step(state, Action.switch(1), Action.move(0))
    assert after.sides[0].status[0] is None


def test_moxie_boosts_on_a_knockout(dex, config):
    state = build(config, a_set("gyarados", "moxie", ("bodyslam",)),
                  a_set("snorlax", "__none__"))
    state.sides[1].hp[0] = 1
    after, log = step(state, Action.move(0), Action.move(0))
    assert after.sides[0].boost(0, "atk") == 1
    assert any(e.kind == "faint" and e.side == 1 for e in log)


def test_clear_body_refuses_a_drop_from_the_opponent(dex, config):
    state = build(config, a_set("metagross", "clearbody"), a_set("arcanine", "intimidate"))
    assert state.sides[0].boost(0, "atk") == 0


def test_contact_abilities_need_contact(dex, config):
    hit = build(config, a_set("garchomp", "roughskin"), a_set("snorlax", "__none__", ("bodyslam",)))
    ctx = make_context(hit)
    cast(ctx, dex, "bodyslam", attacker=BLUE, defender=RED)
    assert hit.sides[1].hp[0] < hit.pokemon(1, 0).max_hp, "contact move, Rough Skin bites"

    ranged = build(config, a_set("garchomp", "roughskin"),
                   a_set("snorlax", "__none__", ("hyperbeam",)))
    ctx = make_context(ranged)
    cast(ctx, dex, "shadowball", attacker=BLUE, defender=RED)
    assert ranged.sides[1].hp[0] == ranged.pokemon(1, 0).max_hp, "no contact, no recoil"


# --------------------------------------------------------------------------- #
# Honesty
# --------------------------------------------------------------------------- #


def test_abilities_with_no_handlers_are_deliberate(dex):
    """Every handler-free ability must be engine-side or genuinely inert."""
    from pkcm.engine import abilities
    from pkcm.engine.effects import REGISTRY
    from pkcm.engine.moves import MOLD_BREAKER_ABILITIES

    engine_side = {"levitate", "corrosion", "telepathy"} | set(MOLD_BREAKER_ABILITIES)
    for (kind, ability_id), effect in REGISTRY.items():
        if kind != "ability" or effect.handlers:
            continue
        assert ability_id in engine_side or ability_id in abilities.INERT, (
            f"{ability_id} is registered with no handlers and no reason given"
        )


def test_roster_coverage_is_reported(dex):
    from pkcm.engine.effects import registered

    regulation = dex.regulation("m_b")
    roster = {
        ability
        for species_id in regulation.legal_species | regulation.legal_megas
        for ability in dex.species[species_id].abilities
    }
    done = roster & set(registered("ability"))
    assert len(done) >= 130, f"only {len(done)} of {len(roster)} roster abilities implemented"
