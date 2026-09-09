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
from pkcm.engine.state import BattleConfig, legal_actions, new_battle

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
    use_move(ctx, attacker, dex.moves[move_id], defender=defender)


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
    """Steel cannot be poisoned, and that is the wall Corrosion breaks.

    The type chart is not involved: Toxic is a status move, and those ignore
    type immunity. The refusal comes from ``set_status``, so the log says
    ``status_immune``.
    """
    state = build(config, a_set("salazzle", "oblivious", ("toxic",)),
                  a_set("skarmory", "sturdy"))
    ctx = make_context(state)
    cast(ctx, dex, "toxic")
    assert state.sides[1].status[0] is None
    assert any(e.kind == "status_immune" for e in ctx.log), ctx.log


def test_a_status_move_reaches_through_the_type_chart(dex, config):
    """Curse is Ghost and Normal types are immune to Ghost -- and it still lands."""
    state = build(config, a_set("gengar", "cursedbody", ("curse",)),
                  a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "curse")
    assert state.sides[1].has_volatile(0, "curse")


def test_thunder_wave_is_the_exception(dex, config):
    """It sets ``ignoreImmunity: false``, which puts the chart back in play."""
    assert dex.moves["thunderwave"].raw.get("ignoreImmunity") is False
    state = build(config, a_set("pikachu", "static", ("thunderwave",)),
                  a_set("garchomp", "roughskin"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderwave")
    assert state.sides[1].status[0] is None
    assert any(e.kind == "immune" for e in ctx.log)


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

    from pkcm.engine.abilities import IGNORES_REDIRECTION

    # Suction Cups is read straight out of ``tactics.force_switch``, beside
    # Ingrain, because refusing a drag is that function's decision to make.
    # Read where the decision they change is taken, rather than through a
    # hook: Sticky Hold at the one gate every item-taker goes through, Ripen
    # where a berry works out what it is worth.
    engine_side = ({"levitate", "corrosion", "suctioncups", "stickyhold",
                    "ripen", "klutz",
                    # Guard Dog's drag refusal is read by tactics.force_switch
                    # beside Suction Cups, and its Intimidate half lives in
                    # _drop_on_every_foe. Liquid Ooze is asked by
                    # moves._apply_drain, where the drain is resolved.
                    "guarddog", "liquidooze"}
                   | set(MOLD_BREAKER_ABILITIES) | set(IGNORES_REDIRECTION))
    accounted = engine_side | abilities.INERT | abilities.SINGLES_INERT
    for (kind, ability_id), effect in REGISTRY.items():
        if kind != "ability" or effect.handlers:
            continue
        assert ability_id in accounted, (
            f"{ability_id} is registered with no handlers and no reason given. "
            f"Either implement it, or say why it does nothing: engine-side, "
            f"inert in battle, or inert in singles."
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
    assert len(done) >= 190, f"only {len(done)} of {len(roster)} roster abilities implemented"


# --------------------------------------------------------------------------- #
# The rest of the roster
# --------------------------------------------------------------------------- #


def test_imposter_transforms_on_entry(dex, config):
    """hk's report: it copies the current state, keeps its own HP.

    The Mega half is still pending; everything else is here.
    """
    ditto = a_set("ditto", "imposter", ("transform",))
    # Night Shade deals a flat 50, so Ditto survives the turn it arrives and
    # there is something left to inspect.
    target = a_set("gengar", "cursedbody", ("nightshade", "sludgebomb"))
    bench = [a_set(s, "__none__") for s in ("snorlax", "pikachu", "starmie", "alakazam", "skarmory")]
    state = new_battle(config, (tuple([bench[0], ditto] + bench[1:]), tuple([target] + bench)),
                       seed=3)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    # Give the target something to copy, then bring Ditto in.
    ctx = make_context(state)
    mutate.boost(ctx, BLUE, {"spa": 2})
    state.rng = ctx.cursor.seal()

    ditto_hp = state.sides[0].hp[1]
    state, log = step(state, Action.switch(1), Action.move(0))
    ditto_hp -= 50  # Night Shade

    assert state.species_id(0, 1) == "gengar", "copied the forme"
    assert state.sides[0].active == [1]
    assert state.ability_id(0, 1) == "cursedbody", "copied the ability"
    assert state.types(0, 1) == ("ghost", "poison"), "copied the types"
    assert state.sides[0].boost(1, "spa") == 2, "copied the stat stages"
    assert [m.id for m in state.moves(0, 1)] == ["nightshade", "sludgebomb"]
    assert state.sides[0].pp[1] == [5, 5], "copied moves get 5 PP each"
    assert any(e.kind == "transform" for e in log), log

    gengar_stats = state.pokemon(1, 0).stats
    assert state.stats(0, 1)[Stat.SPA] == gengar_stats[Stat.SPA], "copied the stats"
    assert state.stats(0, 1)[Stat.HP] != gengar_stats[Stat.HP], "but not HP"
    assert state.sides[0].hp[1] == ditto_hp, "and keeps its own current HP"


def test_pixilate_changes_type_and_adds_power(dex, config):
    from pkcm.engine.moves import activate

    state = build(config, a_set("sylveon", "pixilate", ("hypervoice",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    active = activate(ctx, RED, BLUE, dex.moves["hypervoice"])
    assert dex.moves["hypervoice"].type == "normal"
    assert active.type == "fairy"
    assert active.base_power == chain_modify(dex.moves["hypervoice"].base_power, 4915)


def test_protean_retypes_the_user_once(dex, config):
    from pkcm.engine.moves import activate

    state = build(config, a_set("greninja", "protean", ("surf", "shadowball")),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    activate(ctx, RED, BLUE, dex.moves["surf"])
    assert state.types(0, 0) == ("water",)

    activate(ctx, RED, BLUE, dex.moves["shadowball"])
    assert state.types(0, 0) == ("water",), "gen 9 allows it once per switch-in"


def test_skill_link_maxes_the_hit_count(dex, config):
    for seed in range(5):
        state = build(config, a_set("cinccino", "skilllink", ("bulletseed",)),
                      a_set("snorlax", "__none__"))
        state.rng = state.rng.__class__(state.rng.state + seed)
        ctx = make_context(state)
        cast(ctx, dex, "bulletseed")
        assert len([e for e in ctx.log if e.kind == "damage"]) == 5


def test_long_reach_removes_contact(dex, config):
    """Rough Skin should not answer a Long Reach user."""
    state = build(config, a_set("decidueye", "longreach", ("leafblade",)),
                  a_set("garchomp", "roughskin"))
    ctx = make_context(state)
    cast(ctx, dex, "leafblade")
    assert state.sides[0].hp[0] == state.pokemon(0, 0).max_hp


def test_infiltrator_goes_through_a_substitute(dex, config):
    state = build(config, a_set("noivern", "infiltrator", ("airslash",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    mutate.add_volatile(ctx, BLUE, "substitute", hp=50)
    before = state.sides[1].hp[0]
    cast(ctx, dex, "airslash")
    assert state.sides[1].hp[0] < before, "the substitute did not stop it"

    walled = build(config, a_set("snorlax", "__none__", ("airslash",)),
                   a_set("snorlax", "__none__"))
    ctx = make_context(walled)
    mutate.add_volatile(ctx, BLUE, "substitute", hp=50)
    intact = walled.sides[1].hp[0]
    cast(ctx, dex, "airslash")
    assert walled.sides[1].hp[0] == intact, "without Infiltrator the substitute holds"


def test_sheer_force_trades_secondaries_for_power(dex, config):
    from pkcm.engine.moves import activate

    state = build(config, a_set("darmanitan", "sheerforce", ("ironhead",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    assert dex.moves["ironhead"].raw.get("secondary")
    active = activate(ctx, RED, BLUE, dex.moves["ironhead"])
    assert active.secondaries == [], "the flinch chance is given up"


def test_unaware_ignores_the_opponents_stages(dex, config):
    state = build(config, a_set("garchomp", "roughskin", ("earthquake",)),
                  a_set("clefable", "unaware"))
    # Fresh contexts off the same unsealed RNG replay identical damage rolls,
    # so the only thing that can differ is the stage handling.
    def damage(target_state) -> int:
        return compute_damage(make_context(target_state), RED, BLUE,
                              dex.moves["earthquake"], crit=False)[0]

    plain = damage(state)
    mutate.boost(make_context(state), RED, {"atk": 6})
    assert damage(state) == plain, "Unaware should not see +6 Attack"

    aware = build(config, a_set("garchomp", "roughskin", ("earthquake",)),
                  a_set("clefable", "magicguard"))
    baseline = damage(aware)
    mutate.boost(make_context(aware), RED, {"atk": 6})
    assert damage(aware) > baseline * 3, "and everyone else certainly does"


def test_synchronize_hands_the_status_back(dex, config):
    state = build(config, a_set("umbreon", "synchronize"),
                  a_set("gengar", "__none__", ("willowisp",)))
    ctx = make_context(state)
    cast(ctx, dex, "willowisp", attacker=BLUE, defender=RED)
    assert state.sides[0].status[0] == "brn"
    assert state.sides[1].status[0] == "brn", "and back to the sender"


def test_light_metal_halves_weight_for_weight_moves(dex, config):
    from pkcm.engine.moves import base_power

    heavy = build(config, a_set("garchomp", "roughskin", ("grassknot",)),
                  a_set("aggron", "sturdy"))
    light = build(config, a_set("garchomp", "roughskin", ("grassknot",)),
                  a_set("aggron", "lightmetal"))
    assert base_power(make_context(heavy), RED, BLUE, dex.moves["grassknot"]) > \
        base_power(make_context(light), RED, BLUE, dex.moves["grassknot"])


def test_sniper_boosts_only_criticals(dex, config):
    state = build(config, a_set("kingdra", "sniper", ("dracometeor",)),
                  a_set("snorlax", "__none__"))
    ctx = make_context(state)
    normal = compute_damage(ctx, RED, BLUE, dex.moves["dracometeor"], crit=False)[0]
    critical = compute_damage(ctx, RED, BLUE, dex.moves["dracometeor"], crit=True)[0]
    assert critical > normal * 2, "1.5x crit and 1.5x Sniper on top"


def test_disguise_eats_the_first_hit(dex, config):
    state = build(config, a_set("mimikyu", "disguise"),
                  a_set("garchomp", "roughskin", ("earthquake",)))
    full = state.pokemon(0, 0).max_hp
    ctx = make_context(state)
    cast(ctx, dex, "earthquake", attacker=BLUE, defender=RED)

    assert "busted" in state.species_id(0, 0)
    assert state.sides[0].hp[0] == full - max(1, full // 8), "only the disguise cost"


def test_only_item_abilities_remain(dex):
    """Nothing on the roster is pending any more.

    This asserted a pending pair for months -- Ripen and Sticky Hold, both
    waiting on the seam where an item changes hands. Sticky Hold turned out
    to be implemented already, at ``_holds_removable``, and only unregistered;
    Ripen needed the two lines where a berry decides how much it is worth.
    """
    from pkcm.engine.effects import registered

    regulation = dex.regulation("m_b")
    roster = {
        ability
        for species_id in regulation.legal_species | regulation.legal_megas
        for ability in dex.species[species_id].abilities
    }
    missing = roster - set(registered("ability"))
    assert missing == set(), (
        f"every ability on the roster is implemented; got {sorted(missing)} "
        f"pending"
    )


def test_nothing_is_called_inert_that_a_battle_can_notice(dex):
    """``INERT`` is a claim, and it was wrong about most of its members.

    Registering an ability with no handlers keeps it out of the coverage
    report's missing list, which is the point -- "implemented as nothing" is
    not "forgotten". It also hides an ability that should be doing something,
    and that is what had happened: fourteen of the eighteen entries had real
    battle effects, five of them on parties in the field. Unburden was
    doubling nothing on a Sneasler built to lose its berry; Magician was two
    Delphox with no ability at all.

    Seven of the eleven that were left went the same way once the differ in
    ``scripts/effect_check.py`` got a control that was honest about gender:
    Aroma Veil, Flower Veil, Cheek Pouch, Klutz, Pickpocket, Pickup and
    Rattled all had real effects and no code. What is left is four, and every
    one of them names the reason a battle cannot see it.
    """
    from pkcm.engine.abilities import INERT

    assert INERT == {
        # outside battle entirely, and none of the three has a legal holder
        "honeygather", "runaway", "ballfetch",
        # moves a berry's trigger from a quarter of HP to a half, and this
        # format has no berry with a quarter threshold. Not inert in general.
        "gluttony",
    }, "add a reason before adding a name"


# --------------------------------------------------------------------------- #
# The seven that INERT was wrong about, and the two that were half written
#
# All nine came out of the same run: ``scripts/effect_check.py`` plays every
# ability with and without and compares the traces, and its control used to
# differ from the real one in two fields rather than one. Every ability passed
# on that difference alone, so nothing below was being asked anything.
# --------------------------------------------------------------------------- #


def _both_ways(config, ability, red_moves, blue, script, read, red="snorlax",
               **red_kwargs):
    """Play the same turns with the ability and without, and read one thing."""
    def played(which):
        state = build(config, a_set(red, which, red_moves, **red_kwargs), blue)
        for ours, theirs in script:
            state, _ = step(state, ours, theirs)
        return read(state)

    return played(ability), played("__none__")


def test_aroma_veil_refuses_taunt(dex, config):
    with_it, without = _both_ways(
        config, "aromaveil", ("splash", "bodyslam"),
        a_set("gengar", "__none__", ("taunt", "splash")),
        [(Action.move(0), Action.move(0))],
        lambda state: "taunt" in state.sides[0].volatiles[0])
    assert without, "Taunt lands on anything else"
    assert not with_it


def test_cheek_pouch_heals_on_top_of_the_berry(dex, config):
    def hp_after(ability):
        state = build(config, a_set("snorlax", ability, ("splash", "bodyslam"),
                                    item="sitrusberry"),
                      a_set("snorlax", "__none__", ("splash", "bodyslam")))
        state.sides[0].hp[0] = state.pokemon(0, 0).max_hp * 2 // 5
        state, _ = step(state, Action.move(0), Action.move(1))
        return state.sides[0].hp[0]

    assert hp_after("cheekpouch") > hp_after("__none__"), "the berry, and more"


def test_klutz_turns_the_holders_own_item_off(dex, config):
    def hp_after(ability):
        state = build(config, a_set("snorlax", ability, ("splash", "bodyslam"),
                                    item="leftovers"),
                      a_set("snorlax", "__none__", ("splash", "bodyslam")))
        state.sides[0].hp[0] = state.pokemon(0, 0).max_hp // 2
        before = state.sides[0].hp[0]
        state, _ = step(state, Action.move(0), Action.move(0))
        return state.sides[0].hp[0] - before

    assert hp_after("__none__") > 0, "Leftovers heals a Pokemon that can use it"
    assert hp_after("klutz") == 0


def test_pickpocket_takes_the_item_off_whatever_touched_it(dex, config):
    def held(ability):
        state = build(config, a_set("snorlax", ability, ("splash", "bodyslam")),
                      a_set("garchomp", "__none__", ("dragonclaw", "splash"),
                            item="leftovers"))
        state, _ = step(state, Action.move(0), Action.move(0))
        return state.item_id(0, 0), state.item_id(1, 0)

    assert held("pickpocket") == ("leftovers", None)
    assert held("__none__") == (None, "leftovers"), "and stays put otherwise"


def test_pickup_takes_up_what_somebody_else_spent(dex, config):
    def held(ability):
        state = build(config, a_set("snorlax", ability, ("bodyslam", "splash")),
                      a_set("garchomp", "__none__", ("splash", "dragonclaw"),
                            item="sitrusberry"))
        state.sides[1].hp[0] = state.pokemon(1, 0).max_hp * 2 // 5
        state, _ = step(state, Action.move(0), Action.move(0))
        return state.item_id(0, 0), state.item_id(1, 0)

    assert held("pickup") == ("sitrusberry", None), "spent, and picked up"
    assert held("__none__") == (None, None), "otherwise it is simply gone"


def test_rattled_runs_from_a_dark_move(dex, config):
    with_it, without = _both_ways(
        config, "rattled", ("splash", "bodyslam"),
        a_set("weavile", "__none__", ("knockoff", "splash")),
        [(Action.move(0), Action.move(0))],
        lambda state: state.sides[0].boost(0, "spe"))
    assert with_it == 1 and without == 0


def test_rattled_runs_from_an_intimidate(dex, config):
    """Not a move at all, which is why it needs its own hook."""
    def speed(ability):
        state = build(config, a_set("snorlax", ability, ("splash", "bodyslam")),
                      a_set("garchomp", "intimidate", ("splash", "dragonclaw")))
        return state.sides[0].boost(0, "spe")

    assert speed("rattled") == 1 and speed("__none__") == 0


def test_purifying_salt_refuses_every_status(dex, config):
    with_it, without = _both_ways(
        config, "purifyingsalt", ("splash", "bodyslam"),
        a_set("gengar", "__none__", ("willowisp", "splash")),
        [(Action.move(0), Action.move(0))] * 3,
        lambda state: state.sides[0].status[0])
    assert without == "brn"
    assert with_it is None


def test_purifying_salt_still_halves_ghost_damage(dex, config):
    def damage(ability):
        state = build(config, a_set("milotic", ability, ("splash", "bodyslam")),
                      a_set("gengar", "__none__", ("shadowball", "splash")))
        ctx = make_context(state)
        cast(ctx, dex, "shadowball", attacker=BLUE, defender=RED)
        return state.pokemon(0, 0).max_hp - state.sides[0].hp[0]

    assert damage("purifyingsalt") * 2 <= damage("__none__") + 1


def test_solar_power_is_paid_for_every_turn_of_sun(dex, config):
    """The Special Attack was implemented and the sunburn was not."""
    from pkcm.engine.abilities import SOLAR_POWER_FRACTION

    def lost(ability):
        state = build(config, a_set("charizard", ability, ("splash", "bodyslam")),
                      a_set("snorlax", "__none__", ("splash", "bodyslam")))
        state.field.weather = "sunnyday"
        state.field.weather_turns = 8
        whole = state.pokemon(0, 0).max_hp
        state, _ = step(state, Action.move(0), Action.move(0))
        return whole - state.sides[0].hp[0], whole

    burnt, whole = lost("solarpower")
    assert lost("__none__")[0] == 0
    assert abs(burnt - whole // SOLAR_POWER_FRACTION) <= 1


def test_solar_power_costs_nothing_out_of_the_sun(dex, config):
    state = build(config, a_set("charizard", "solarpower", ("splash", "bodyslam")),
                  a_set("snorlax", "__none__", ("splash", "bodyslam")))
    whole = state.pokemon(0, 0).max_hp
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.sides[0].hp[0] == whole


def test_leaf_guard_refuses_a_status_only_in_the_sun(dex, config):
    def status(weather):
        state = build(config, a_set("meganium", "leafguard", ("splash", "bodyslam")),
                      a_set("gengar", "__none__", ("willowisp", "splash")))
        if weather:
            state.field.weather = weather
            state.field.weather_turns = 8
        for _ in range(3):
            state, _ = step(state, Action.move(0), Action.move(0))
        return state.sides[0].status[0]

    assert status("sunnyday") is None
    assert status(None) == "brn"


# --------------------------------------------------------------------------- #
# How long a change to what a Pokemon *is* lasts
# --------------------------------------------------------------------------- #


def _wall_team():
    return tuple(a_set(s, "sturdy", ("protect",)) for s in
                 ("skarmory", "aggron", "steelix", "forretress", "bastiodon", "registeel"))


def test_protean_fires_once_per_entry_and_recharges_on_a_switch(dex, config):
    """Reported by hk: gen 9 allows it once, and leaving the field refills it.

    Showdown stores the flag in the ability's ``effectState``, which is per
    activation -- switching out ends the activation.
    """
    greninja = a_set("greninja", "protean", ("surf", "shadowball"))
    bench = [a_set(s, "__none__", ("bodyslam",)) for s in
             ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(config, (tuple([greninja] + bench), _wall_team()), seed=1)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))
    assert state.types(0, 0) == ("water", "dark")

    state, _ = step(state, Action.move(0), Action.move(0))       # Surf
    assert state.types(0, 0) == ("water",)

    state, _ = step(state, Action.move(1), Action.move(0))       # Shadow Ball
    assert state.types(0, 0) == ("water",), "only once per entry"

    state, _ = step(state, Action.switch(1), Action.move(0))     # out
    assert state.types(0, 0) == ("water", "dark"), "the retype reverts on the way out"

    state, _ = step(state, Action.switch(0), Action.move(0))     # back in
    state, _ = step(state, Action.move(1), Action.move(0))       # Shadow Ball
    assert state.types(0, 0) == ("ghost",), "and it is charged again"


def test_transform_also_reverts_on_switching_out(dex, config):
    ditto = a_set("ditto", "imposter", ("transform",))
    bench = [a_set(s, "__none__", ("bodyslam",)) for s in
             ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    state = new_battle(config, (tuple([bench[0], ditto] + bench[1:]), _wall_team()), seed=2)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    state, _ = step(state, Action.switch(1), Action.move(0))
    assert state.species_id(0, 1) == "skarmory"

    state, _ = step(state, Action.switch(0), Action.move(0))
    assert state.species_id(0, 1) == "ditto", "Ditto is itself again on the bench"


def test_a_busted_disguise_stays_busted(dex, config):
    """The contrast: some changes are meant to outlive a switch."""
    mimikyu = a_set("mimikyu", "disguise", ("bodyslam",))
    bench = [a_set(s, "__none__", ("bodyslam",)) for s in
             ("snorlax", "pikachu", "starmie", "gengar", "alakazam")]
    attacker = tuple([a_set("garchomp", "roughskin", ("earthquake",))]
                     + [a_set(s, "__none__", ("bodyslam",)) for s in
                        ("snorlax", "pikachu", "starmie", "gengar", "alakazam")])
    state = new_battle(config, (tuple([mimikyu] + bench), attacker), seed=3)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    state, _ = step(state, Action.move(0), Action.move(0))
    assert "busted" in state.species_id(0, 0)

    state, _ = step(state, Action.switch(1), Action.move(0))
    state, _ = step(state, Action.switch(0), Action.move(0))
    assert "busted" in state.species_id(0, 0), "the disguise does not come back"


# --------------------------------------------------------------------------- #
# Mega Evolution
# --------------------------------------------------------------------------- #


def _mega_team(config, holder="gengar", stone="gengarite"):
    lead = a_set(holder, "cursedbody", ("shadowball", "sludgebomb"))
    lead = lead.replace(item=stone)
    bench = [a_set(s, "__none__", ("bodyslam",)) for s in
             ("snorlax", "pikachu", "starmie", "alakazam", "skarmory")]
    walls = tuple(a_set(s, "sturdy", ("protect",)) for s in
                  ("aggron", "steelix", "forretress", "bastiodon", "registeel", "probopass"))
    state = new_battle(config, (tuple([lead] + bench), walls), seed=1)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def test_mega_evolution_is_offered_only_when_it_is_possible(dex, config):
    ready = _mega_team(config)
    assert any(a.mega for a in legal_actions(ready, 0)), "stone held, not yet used"

    bare = build(config, a_set("gengar", "cursedbody", ("shadowball",)),
                 a_set("snorlax", "thickfat"))
    assert not any(a.mega for a in legal_actions(bare, 0)), "no stone, no offer"

    wrong = build(config,
                  a_set("snorlax", "thickfat", ("bodyslam",)).replace(item="gengarite"),
                  a_set("pikachu", "static"))
    assert not any(a.mega for a in legal_actions(wrong, 0)), "wrong holder"


def test_mega_evolution_changes_forme_ability_and_stats(dex, config):
    state = _mega_team(config)
    before = state.stats(0, 0)

    state, log = step(state, Action.move(0, mega=True), Action.move(0))

    assert state.species_id(0, 0) == "gengarmega"
    assert state.ability_id(0, 0) == "shadowtag", "the Mega forme's ability"
    assert state.stats(0, 0)[Stat.SPA] > before[Stat.SPA]
    assert state.stats(0, 0)[Stat.SPE] > before[Stat.SPE]
    assert any(e.kind == "mega_evolve" for e in log), log


def test_the_new_ability_activates_immediately(dex, config):
    """Mega Gengar's Shadow Tag traps the moment it appears."""
    state = _mega_team(config)
    assert not state.sides[1].has_volatile(0, "trapped")
    state, _ = step(state, Action.move(0, mega=True), Action.move(0))
    assert state.sides[1].has_volatile(0, "trapped")


def test_only_one_mega_per_battle(dex, config):
    state = _mega_team(config)
    assert state.mega_used == [False, False]

    state, _ = step(state, Action.move(0, mega=True), Action.move(0))
    assert state.mega_used == [True, False]
    assert not any(a.mega for a in legal_actions(state, 0)), "spent"
    assert state.mega_target(0, 0) is None, "and it cannot Mega again"


def test_mega_evolution_survives_switching_and_fainting(dex, config):
    """Champions does not revert it, unlike the mainline games."""
    state = _mega_team(config)
    state, _ = step(state, Action.move(0, mega=True), Action.move(0))
    assert state.species_id(0, 0) == "gengarmega"

    # Shadow Tag traps the opponent, not us; we can still leave.
    state, _ = step(state, Action.switch(1), Action.move(0))
    assert state.species_id(0, 0) == "gengarmega", "still Mega on the bench"

    state, _ = step(state, Action.switch(0), Action.move(0))
    assert state.species_id(0, 0) == "gengarmega"


def test_mega_speed_decides_the_turn_order(dex, config):
    """The Mega resolves before move order is worked out, so its Speed counts."""
    # Gengar sits at 130 Speed and Mega Gengar at 150; Weavile is 145, so it
    # outruns the one and not the other.
    slower = a_set("gengar", "cursedbody", ("shadowball",)).replace(item="gengarite")
    faster = a_set("weavile", "pressure", ("nightslash",))  # not Ice Shard: priority would decide it instead
    bench = [a_set(s, "__none__", ("bodyslam",)) for s in
             ("snorlax", "pikachu", "starmie", "alakazam", "skarmory")]
    state = new_battle(config, (tuple([slower] + bench), tuple([faster] + bench)), seed=4)
    state, _ = step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))

    assert state.stats(0, 0)[Stat.SPE] < state.stats(1, 0)[Stat.SPE], "slower to begin with"
    _, log = step(state, Action.move(0, mega=True), Action.move(0))
    movers = [e.side for e in log if e.kind == "move_used"]
    assert movers[0] == 0, "Mega Gengar outspeeds Weavile once it has evolved"


def test_parental_bond_hits_twice_with_the_second_at_quarter_power(dex, config):
    state = build(config, a_set("kangaskhan", "parentalbond", ("bodyslam",)),
                  a_set("snorlax", "thickfat"))
    ctx = make_context(state)
    cast(ctx, dex, "bodyslam")
    hits = [e for e in ctx.log if e.kind == "damage"]
    assert len(hits) == 2
    assert hits[1].amount < hits[0].amount // 2, "the second hit is a quarter"


def test_shell_armor_and_battle_armor_refuse_critical_hits(dex, config):
    """They were registered, then silently registered over by the INERT list.

    Nothing failed while they did nothing: an ability that only shows up on a
    crit roll looks the same as one that is merely rarely relevant.
    """
    from pkcm.engine.moves import rolls_crit

    def crits_in(ability, tries=200):
        state = build(config, a_set("garchomp", "roughskin", ("slash",)),
                      a_set("cloyster", ability))
        ctx = make_context(state)
        return sum(rolls_crit(ctx, RED, BLUE, dex.moves["slash"]) for _ in range(tries))

    # Slash has a raised crit ratio, so an unprotected target is hit often.
    assert crits_in("skilllink") > 10, "the control has to actually crit"
    assert crits_in("shellarmor") == 0
    assert crits_in("battlearmor") == 0


def test_a_bounced_move_counts_as_an_ability_effect(dex, config):
    """hk's report: a reflected status move cannot be reflected again.

    The game treats the bounce itself as an ability effect, the same way it
    treats an Intimidate drop or a Static paralysis -- and ability effects are
    not reflectable. So between two Magic Bounce holders it comes back exactly
    once, and only the Pokemon that used the move suffers.
    """
    state = build(config, a_set("hatterene", "magicbounce", ("thunderwave",)),
                  a_set("espeon", "magicbounce"))
    ctx = make_context(state)
    cast(ctx, dex, "thunderwave")
    assert state.sides[0].status[0] == "par", "the user eats its own move"
    assert state.sides[1].status[0] is None
    assert sum(e.kind == "ability_block" and e.detail == "magicbounce"
               for e in ctx.log) == 1


def test_magic_bounce_does_not_reflect_an_ability(dex, config):
    """Intimidate and Static are not moves; there is nothing to send back."""
    from pkcm.engine import effects as fx

    state = build(config, a_set("gyarados", "intimidate"), a_set("espeon", "magicbounce"))
    ctx = make_context(state)
    ctx.log.clear()
    fx.notify(ctx, "switch_in", RED)
    assert state.sides[1].boost(0, "atk") < 0, "the drop lands"
    assert state.sides[0].boost(0, "atk") == 0, "and does not come back"


def _berry_position(dex, ours, theirs, seed=5):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle

    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    filler = [PokemonSet(species=name, ability="__none__", moves=("tackle",),
                         item=None, nature="serious", sp=(0,) * 6)
              for name in ("pikachu", "alakazam", "machamp")]
    state = new_battle(config, (tuple(list(ours) + filler),
                                tuple(list(theirs) + filler)), seed=seed)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def _a_set(species, ability, moves, item=None, sp=(0,) * 6, nature="serious"):
    from pkcm.engine.pokemon import PokemonSet

    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=item, nature=nature, sp=sp)


def test_cud_chew_eats_the_same_berry_at_the_end_of_the_next_turn(dex):
    """Champions' dex, verbatim: 나무열매를 먹으면, 다음 턴 종료 시 같은
    나무열매를 한 번 더 먹는다. Not the same turn -- the next one."""
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    ours = [_a_set("farigiraf", "cudchew", ("psychic", "crunch", "rest", "yawn"),
                   "sitrusberry", (32, 0, 0, 30, 4, 0), "modest")]
    # A harmless attacker: the point is what happens at the end of two turns,
    # and Farigiraf has to live through both of them to show it.
    theirs = [_a_set("snorlax", "thickfat", ("tackle",), None,
                     (32, 0, 32, 0, 2, 0), "impish")]
    state = _berry_position(dex, ours, theirs)
    slot = state.sides[0].active[0]
    state.sides[0].hp[slot] = int(state.pokemon(0, slot).max_hp * 0.45)

    # Yawn both turns: nothing may faint, on either side, or the end of turn
    # that is under test never arrives.
    state, events = step(state, Action.move(3), Action.move(0))
    assert state.item_id(0, slot) is None, "the berry was eaten"
    assert state.sides[0].has_volatile(slot, "cudchew"), (
        "it has to be remembered; the re-eat is a turn away")
    assert not any(e.kind == "ability" and e.detail == "cudchew"
                   for e in events), "not on the turn it was eaten"

    before = state.sides[0].hp[slot]
    state, events = step(state, Action.move(3), Action.move(0))
    assert any(e.kind == "ability" and e.detail == "cudchew" for e in events), (
        "the end of the next turn is when it eats it again")
    assert not state.sides[0].has_volatile(slot, "cudchew"), "and only once"
    assert state.item_id(0, slot) is None, (
        "Cud Chew eats it again; it does not give it back")
    assert state.sides[0].hp[slot] > before, "the second Sitrus healed it"


def test_cud_chew_forgets_the_berry_when_it_leaves_the_field(dex):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    ours = [_a_set("farigiraf", "cudchew", ("psychic", "crunch", "rest", "yawn"),
                   "sitrusberry", (32, 0, 0, 30, 4, 0), "modest"),
            _a_set("corviknight", "pressure",
                   ("bravebird", "bodypress", "roost", "ironhead"),
                   "leftovers", (32, 0, 32, 0, 2, 0), "impish"),
            _a_set("primarina", "torrent",
                   ("moonblast", "surf", "psychic", "calmmind"),
                   "sitrusberry", (4, 0, 0, 32, 30, 0), "modest")]
    theirs = [_a_set("snorlax", "thickfat", ("tackle",), None,
                     (32, 0, 32, 0, 2, 0), "impish")]
    state = _berry_position(dex, ours, theirs)
    slot = state.sides[0].active[0]
    state.sides[0].hp[slot] = int(state.pokemon(0, slot).max_hp * 0.45)
    state, _ = step(state, Action.move(3), Action.move(0))
    assert state.sides[0].has_volatile(slot, "cudchew")

    state, events = step(state, Action.switch(1), Action.move(0))
    assert not state.sides[0].has_volatile(slot, "cudchew"), (
        "leaving the field ends it")
    assert not any(e.kind == "ability" and e.detail == "cudchew"
                   for e in events)


def test_harvest_grows_the_berry_back_and_only_a_spent_one(dex):
    """Champions' dex: 사용한 나무열매를 턴 종료 시 50% 확률로 만들어 낸다.
    쾌청 상태일 때는 반드시 만들어 낸다. The sun is the certain half, so that
    is the half a test can assert without leaning on a roll.

    The berry comes back and is then eaten again in the same end of turn while
    the holder is still under half, which is the Harvest loop working rather
    than a double-trigger: ``item_restored`` is the event under test.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    ours = [_a_set("tropius", "harvest",
                   ("airslash", "gigadrain", "sunnyday", "synthesis"),
                   "sitrusberry", (32, 0, 0, 30, 4, 0), "modest")]
    theirs = [_a_set("pikachu", "static", ("tackle",), None, (0,) * 6)]
    state = _berry_position(dex, ours, theirs)
    slot = state.sides[0].active[0]

    state, events = step(state, Action.move(2), Action.move(0))
    assert state.field.weather == "sunnyday"
    assert not any(e.kind == "item_restored" for e in events), (
        "nothing has been spent, so there is nothing to grow back")

    state.sides[0].hp[slot] = int(state.pokemon(0, slot).max_hp * 0.45)
    state, events = step(state, Action.move(0), Action.move(0))
    assert any(e.kind == "item_restored" and e.detail == "sitrusberry"
               for e in events), (
        "the sun makes the regrowth certain, and it did not happen")


def test_harvest_grows_back_nothing_when_nothing_was_spent(dex):
    """It reads the engine's own shape for a spent item -- overridden to None
    with the set keeping the original -- so a Pokemon that simply holds nothing
    must not have one invented for it."""
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    ours = [_a_set("tropius", "harvest",
                   ("airslash", "gigadrain", "sunnyday", "synthesis"),
                   None, (32, 0, 0, 30, 4, 0), "modest")]
    theirs = [_a_set("pikachu", "static", ("tackle",), None, (0,) * 6)]
    state = _berry_position(dex, ours, theirs)
    slot = state.sides[0].active[0]
    state, _ = step(state, Action.move(2), Action.move(0))
    state, events = step(state, Action.move(0), Action.move(0))
    assert not any(e.kind == "item_restored" for e in events)
    assert state.item_id(0, slot) is None


def test_ripen_doubles_what_a_berry_is_worth(dex, config):
    """Two halves and two heals: the only berries in this format with a size."""
    from pkcm.engine import items

    given = []
    for ability in ("thickfat", "ripen"):
        # Nobody attacks: an Earthquake big enough to put it under half was
        # also big enough to finish it, and then no berry was eaten at all.
        state = build(config, a_set("snorlax", ability, ("splash",),
                                    item="sitrusberry"),
                      a_set("garchomp", "roughskin", ("splash",)))
        state.sides[0].hp[0] = state.pokemon(0, 0).max_hp // 2 - 1
        before = state.sides[0].hp[0]
        state, _ = step(state, Action.move(0), Action.move(0))
        given.append(state.sides[0].hp[0] - before)
    assert given[0] > 0, "the plain Sitrus Berry was eaten"
    assert given[1] == given[0] * 2, f"plain {given[0]}, ripened {given[1]}"


def test_sticky_hold_keeps_the_item_through_every_way_of_taking_it(dex, config):
    """One gate, ``_holds_removable``, and every taker goes through it."""
    from pkcm.engine.moveeffects import _holds_removable
    from pkcm.engine.battle import make_context

    state = build(config, a_set("weavile", "pressure",
                                ("knockoff", "thief", "trick")),
                  a_set("snorlax", "stickyhold", ("splash",),
                        item="leftovers"))
    ctx = make_context(state)
    assert _holds_removable(ctx, (1, 0)) is None, "nothing may take it"

    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.item_id(1, 0) == "leftovers", "Knock Off left it there"


# --------------------------------------------------------------------------- #
# The abilities the 2026-09-09 patch brings in
# --------------------------------------------------------------------------- #


def _one_on_one(dex, ours, theirs, seed=4):
    """A battle with one chosen Pokemon a side and two bodies behind it."""
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.pokemon import PokemonSet
    from pkcm.engine.state import BattleConfig, new_battle

    filler = (PokemonSet(species="pikachu", ability="__none__",
                         moves=("tackle",), nature="serious", sp=(0,) * 6),
              PokemonSet(species="alakazam", ability="__none__",
                         moves=("tackle",), nature="serious", sp=(0,) * 6))
    config = BattleConfig(dex=dex, regulation=dex.regulation("m_b"),
                          battle_format="singles")
    state = new_battle(config, (tuple(ours) + filler, tuple(theirs) + filler),
                       seed=seed)
    return step(state, Action.select(0, 1, 2), Action.select(0, 1, 2))[0]


def _set(species, ability, moves, item=None, nature="serious", sp=(0,) * 6):
    from pkcm.engine.pokemon import PokemonSet

    return PokemonSet(species=species, ability=ability, moves=tuple(moves),
                      item=item, nature=nature, sp=sp)


def _damage_to(events, side, move):
    """How much a named move dealt to one side. Side 0 has no ``side=`` tag."""
    import re

    for event in events:
        text = str(event)
        if not text.startswith("damage(") or move not in text:
            continue
        head = text.split("damage(")[1][:8]
        mine = "side=" not in head
        if (side == 0) == mine:
            return int(re.search(r"amount=(\d+)", text).group(1))
    return 0


def test_liquid_ooze_makes_the_drain_hurt(dex):
    """The sign of the transaction flips, not its size.

    Bulbapedia: the attacker takes damage equal to the amount of HP that
    would have been recovered.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    seen = {}
    for ability in ("liquidooze", "sturdy"):
        state = _one_on_one(
            dex,
            [_set("meowscarada", "protean",
                  ("gigadrain", "knockoff", "uturn", "flowertrick"),
                  nature="modest", sp=(0, 0, 2, 32, 0, 32))],
            [_set("swalot", ability,
                  ("sludgebomb", "toxic", "yawn", "seismictoss"),
                  sp=(32, 0, 32, 0, 2, 0))])
        state, _ = step(state, Action.move(1), Action.move(0))
        before = state.sides[0].hp[state.sides[0].active[0]]
        state, _ = step(state, Action.move(0), Action.move(1))
        seen[ability] = state.sides[0].hp[state.sides[0].active[0]] - before

    assert seen["sturdy"] > 0, "Giga Drain did not heal without the ability"
    assert seen["liquidooze"] < 0, f"Liquid Ooze still healed: {seen}"


def test_grass_pelt_only_helps_on_grass(dex):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    seen = {}
    for ability in ("grasspelt", "sapsipper"):
        state = _one_on_one(
            dex,
            [_set("gogoat", ability,
                  ("grassyterrain", "hornleech", "earthquake", "bulkup"),
                  sp=(32, 0, 32, 0, 2, 0))],
            [_set("garchomp", "roughskin",
                  ("dragonclaw", "earthquake", "firefang", "stoneedge"),
                  nature="jolly", sp=(0, 32, 2, 0, 0, 32))])
        # The terrain has to be standing before the hit lands: Garchomp is
        # faster, so a first-turn measurement is of no terrain at all.
        state, _ = step(state, Action.move(0), Action.move(0))
        state, events = step(state, Action.move(3), Action.move(0))
        seen[ability] = _damage_to(events, 0, "dragonclaw")

    assert seen["grasspelt"] and seen["sapsipper"]
    ratio = seen["grasspelt"] / seen["sapsipper"]
    assert 0.6 < ratio < 0.72, f"Defence was not half again: {seen}, {ratio:.2f}"


def test_steely_spirit_powers_up_steel(dex):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    seen = {}
    for ability in ("steelyspirit", "battlearmor"):
        state = _one_on_one(
            dex,
            [_set("perrserker", ability,
                  ("ironhead", "uturn", "protect", "closecombat"),
                  nature="adamant", sp=(0, 32, 2, 0, 0, 32))],
            [_set("blissey", "naturalcure",
                  ("softboiled", "toxic", "protect", "seismictoss"),
                  sp=(32, 0, 32, 0, 2, 0))])
        state, events = step(state, Action.move(0), Action.move(0))
        seen[ability] = _damage_to(events, 1, "ironhead")

    ratio = seen["steelyspirit"] / seen["battlearmor"]
    assert 1.4 < ratio < 1.6, f"not half again: {seen}, {ratio:.2f}"


def test_seed_sower_answers_a_hit_with_terrain(dex):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    state = _one_on_one(
        dex,
        [_set("arboliva", "seedsower",
              ("gigadrain", "earthpower", "leechseed", "synthesis"),
              nature="modest", sp=(32, 0, 32, 32, 2, 0))],
        [_set("garchomp", "roughskin",
              ("dragonclaw", "earthquake", "firefang", "stoneedge"),
              nature="jolly", sp=(0, 32, 2, 0, 0, 32))])
    assert state.field.terrain is None
    state, _ = step(state, Action.move(0), Action.move(0))
    assert state.field.terrain == "grassyterrain"


def test_stakeout_doubles_on_a_fresh_arrival(dex):
    """Against the replacement, not against the leads.

    Both arms face the same species, so the only difference between them is
    whether the target arrived this turn.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    seen = {}
    for switched in (True, False):
        state = _one_on_one(
            dex,
            [_set("thievul", "stakeout",
                  ("darkpulse", "uturn", "protect", "shadowball"),
                  nature="modest", sp=(0, 0, 2, 32, 0, 32))],
            [_set("blissey", "naturalcure",
                  ("softboiled", "toxic", "protect", "seismictoss"),
                  sp=(32, 0, 32, 0, 2, 0)),
             _set("blissey", "naturalcure",
                  ("softboiled", "toxic", "protect", "seismictoss"),
                  sp=(32, 0, 32, 0, 2, 0))])
        state, _ = step(state, Action.move(2), Action.move(0))
        state, events = step(state, Action.move(0),
                             Action.switch(1) if switched else Action.move(0))
        seen[switched] = _damage_to(events, 1, "darkpulse")

    ratio = seen[True] / seen[False]
    assert 1.9 < ratio < 2.1, f"not doubled against the replacement: {seen}"


def test_guard_dog_turns_intimidate_round_and_will_not_be_dragged(dex):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.state import BOOST_INDEX

    dog = [_set("mabosstiff", "guarddog",
                ("crunch", "protect", "roar", "uturn"),
                nature="adamant", sp=(32, 32, 2, 0, 0, 0)),
           _set("garchomp", "roughskin",
                ("earthquake", "dragonclaw", "firefang", "uturn"),
                nature="jolly", sp=(0, 32, 2, 0, 0, 32))]

    state = _one_on_one(
        dex, dog,
        [_set("incineroar", "intimidate",
              ("knockoff", "flareblitz", "uturn", "protect"),
              nature="adamant", sp=(32, 32, 2, 0, 0, 0))])
    ours = state.sides[0]
    assert ours.boosts[ours.active[0]][BOOST_INDEX["atk"]] == 1, \
        "Intimidate should have raised Attack, not lowered it"

    state = _one_on_one(
        dex, dog,
        [_set("blissey", "naturalcure",
              ("whirlwind", "softboiled", "toxic", "protect"),
              sp=(32, 0, 32, 0, 2, 0))])
    state, _ = step(state, Action.move(1), Action.move(0))
    assert state.species_id(0, state.sides[0].active[0]) == "mabosstiff", \
        "Whirlwind dragged it out"


def test_emergency_exit_leaves_when_it_crosses_half(dex):
    """The crossing can be recoil, so it watches the HP bar, not a move.

    Golisopod's own Aqua Jet into Rough Skin is what takes it under half here.
    On after_damage the ability never saw that and stayed in.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step
    from pkcm.engine.state import Phase

    seen = {}
    for ability in ("emergencyexit", "sturdy"):
        state = _one_on_one(
            dex,
            [_set("golisopod", ability,
                  ("aquajet", "liquidation", "protect", "xscissor"),
                  nature="adamant", sp=(0, 32, 2, 0, 0, 32)),
             _set("garchomp", "roughskin",
                  ("earthquake", "dragonclaw", "firefang", "uturn"),
                  nature="jolly", sp=(0, 32, 2, 0, 0, 32))],
            [_set("garchomp", "roughskin",
                  ("dragonclaw", "earthquake", "firefang", "stoneedge"),
                  nature="jolly", sp=(0, 32, 2, 0, 0, 32))])
        for _ in range(3):
            if state.phase is not Phase.BATTLE:
                break
            state, _ = step(state, Action.move(0), Action.move(0))
        seen[ability] = state.phase

    assert seen["emergencyexit"] is Phase.MID_TURN_SWITCH, \
        f"it did not bail: {seen}"
    assert seen["sturdy"] is not Phase.MID_TURN_SWITCH


def test_aura_guard_halves_contact_and_nothing_else(dex):
    """파동의방호, on Mega Lucario Z: contact damage from the opponent, halved.

    Neither side may be knocked out on the way: apply_damage returns what the
    HP bar could absorb, so a control that kills reads as the same number as
    a Pokemon at full health and the ratio comes out at 0.7 instead of 0.5.
    Hence no Attack investment and a Blissey to hit.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    seen = {}
    for ability in ("auraguard", "steadfast"):
        for move, index in (("knockoff", 2), ("aurasphere", 1)):
            state = _one_on_one(
                dex,
                [_set("machamp", "guts",
                      ("closecombat", "aurasphere", "knockoff", "protect"),
                      sp=(32, 0, 32, 0, 2, 0))],
                [_set("blissey", ability,
                      ("softboiled", "toxic", "protect", "seismictoss"),
                      nature="bold", sp=(32, 0, 32, 0, 2, 0))])
            state, events = step(state, Action.move(index), Action.move(0))
            seen[ability, move] = _damage_to(events, 1, move)

    assert all(seen.values()), f"something failed to connect: {seen}"
    contact = seen["auraguard", "knockoff"] / seen["steadfast", "knockoff"]
    ranged = seen["auraguard", "aurasphere"] / seen["steadfast", "aurasphere"]
    assert 0.45 < contact < 0.55, f"contact was not halved: {seen}"
    assert ranged == 1.0, f"a non-contact move was softened too: {seen}"


def test_a_powder_does_not_reach_a_grass_type(dex):
    """Every move carrying the powder flag, not just Rage Powder.

    _powder_reaches had the rule from the start and only Rage Powder's
    redirection ever asked it, so Spore put a Meowscarada to sleep. The
    2026-09-09 patch writes the immunity into the move text.
    """
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    # In the format only: the move actually called Powder is isNonstandard
    # 'Past' and the engine reports it unimplemented, which is correct and is
    # not what this is testing. Rage Powder is a redirection and already had
    # the rule.
    powders = [move.id for move in dex.moves.values()
               if "powder" in move.flags and move.id != "ragepowder"
               and move.raw.get("isNonstandard") is None]
    assert len(powders) >= 6, f"expected the powder family, found {powders}"

    for move_id in powders:
        landed = {}
        # Blissey is the control rather than a Steel type: Poison Powder is
        # immune against Steel for its own reason, and that reads here exactly
        # like the powder rule working.
        for foe, ability, moves in (
                # Not U-turn at index zero: Meowscarada is faster, so it
                # would leave and the powder would land on the Electric-type
                # body behind it instead.
                ("meowscarada", "overgrow",
                 ("knockoff", "protect", "uturn", "flowertrick")),
                ("blissey", "naturalcure",
                 ("softboiled", "protect", "toxic", "seismictoss"))):
            state = _one_on_one(
                dex,
                [_set("amoonguss", "effectspore",
                      (move_id, "gigadrain", "protect", "sludgebomb"))],
                [_set(foe, ability, moves, sp=(32, 0, 32, 0, 2, 0))], seed=7)
            state, events = step(state, Action.move(0), Action.move(0))
            landed[foe] = not any("immune" in str(one) for one in events)
        assert not landed["meowscarada"], f"{move_id} reached a Grass type"
        assert landed["blissey"], f"{move_id} was blocked by nothing at all"


def test_overcoat_and_safety_goggles_stop_a_powder_too(dex):
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    slept = {}
    for ability, item, label in (("overcoat", None, "overcoat"),
                                 ("supremeoverlord", "safetygoggles", "goggles"),
                                 ("supremeoverlord", None, "neither")):
        state = _one_on_one(
            dex,
            [_set("amoonguss", "effectspore",
                  ("spore", "gigadrain", "protect", "sludgebomb"))],
            [_set("kingambit", ability,
                  ("irondefense", "protect", "uturn", "roost"),
                  item=item, sp=(32, 0, 32, 0, 2, 0))], seed=7)
        state, _ = step(state, Action.move(0), Action.move(0))
        side = state.sides[1]
        slept[label] = side.status[side.active[0]] == "slp"

    assert slept["neither"], "Spore did not land on anything"
    assert not slept["overcoat"] and not slept["goggles"], slept


def test_effect_spore_is_a_powder_as_well(dex):
    """It reads as a chance, so it is measured as one: sixty contacts a side."""
    from pkcm.engine.actions import Action
    from pkcm.engine.battle import step

    caught = {}
    for species, ability, item, label in (
            ("meowscarada", "overgrow", None, "grass"),
            ("kingambit", "overcoat", None, "overcoat"),
            ("kingambit", "supremeoverlord", "safetygoggles", "goggles"),
            ("kingambit", "supremeoverlord", None, "neither")):
        hits = 0
        for seed in range(60):
            state = _one_on_one(
                dex,
                [_set("amoonguss", "effectspore",
                      ("protect", "gigadrain", "spore", "sludgebomb"),
                      sp=(32, 0, 32, 0, 2, 0))],
                [_set(species, ability,
                      ("knockoff", "irondefense", "uturn", "roost"),
                      item=item, sp=(32, 32, 2, 0, 0, 0))], seed=seed)
            state, _ = step(state, Action.move(1), Action.move(0))
            side = state.sides[1]
            hits += side.status[side.active[0]] is not None
        caught[label] = hits

    assert caught["neither"] > 0, "Effect Spore never fired at all"
    for label in ("grass", "overcoat", "goggles"):
        assert caught[label] == 0, f"{label} was caught by Effect Spore: {caught}"
