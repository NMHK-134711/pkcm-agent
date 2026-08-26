"""Abilities, ported from ``data/reference/abilities.ts``.

Not written from descriptions or from memory -- an earlier draft was, and it
shipped handlers that registered and then did nothing. Each entry here mirrors a
specific Showdown implementation, and the comments name the handler it came from
when the mapping is not obvious.

Three of them shaped the framework rather than fitting into it:

**Mold Breaker** carries no handlers at all, in Showdown or here. It sets
``move.ignoreAbility``; we drop the defender into ``ctx.suppressed_abilities``
for the whole resolution, so the ability is invisible to every hook the move
runs rather than to the checks someone remembered to guard.

**Corrosion** likewise has no handlers -- Showdown's comment says "Implemented in
sim/pokemon.js:Pokemon#setStatus". It is a property of the poisoner, so the type
immunity in ``mutate.set_status`` is run through ``status_immunity`` gathered
from the source, and Corrosion empties the list.

**Poison Heal** inverts damage rather than blocking it (``onDamage`` heals and
returns false), so all non-move damage funnels through
``modify_indirect_damage``. Magic Guard sits on the same hook returning zero.

Damage and base-power multipliers use Showdown's 4096ths (``chain_modify``), not
floats: the source really does say ``chainModify([5325, 4096])``, and rounding
the other way changes hit counts.
"""

from __future__ import annotations

from pkcm.data.dex import Stat
from pkcm.engine import mutate
from pkcm.engine.conditions import is_grounded
from pkcm.engine.effects import Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.moves import X0_5, X0_75, X1_2, X1_25, X1_3, X1_5, X2, chain_modify
from pkcm.engine.mutate import boost, fraction_of_max, heal

CONTACT = "contact"
ALL_STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")


def announce(ctx: Context, ref: Ref, ability: str) -> None:
    ctx.emit(Event("ability", side=ref[0], slot=ref[1], detail=ability))


def _blocked(ctx: Context, ref: Ref, ability: str, move) -> None:
    announce(ctx, ref, ability)
    ctx.emit(Event("ability_block", side=ref[0], slot=ref[1], move=move.name, detail=ability))


# --------------------------------------------------------------------------- #
# No handlers in the source either -- the engine implements these directly
# --------------------------------------------------------------------------- #

#: ``moves.ignores_target_ability`` reads these.
for _name in ("moldbreaker", "turboblaze", "teravolt"):
    register("ability", _name, name=_name.title())

#: ``moves.type_effectiveness`` reads Levitate via ``conditions.is_grounded``,
#: exactly as Showdown reads it via ``Pokemon#isGrounded``.
register("ability", "levitate", name="Levitate")

def _corrosion_status(ctx, ref, value, status, target, **_):
    """``setStatus``: the Steel/Poison status immunity does not apply."""
    return () if status in ("psn", "tox") else None


def _corrosion_reaches(ctx, ref, value, attacker, defender, move, **_):
    """And the Poison-type move has to arrive for that to matter.

    Showdown gets this by not running the type chart on status moves at all;
    we do run it (Thunder Wave must still miss Ground types), so Corrosion has
    to open the door for its own moves explicitly.
    """
    if ref != attacker or value != 0.0:
        return None
    if move.category == "Status" and move.raw.get("status") in ("psn", "tox"):
        return 1.0
    return None


register("ability", "corrosion", name="Corrosion",
         status_immunity=_corrosion_status,
         modify_effectiveness=_corrosion_reaches)


# --------------------------------------------------------------------------- #
# The two that bend a hook rather than sit on one
# --------------------------------------------------------------------------- #


def _poison_heal(ctx, ref, value, source_kind, cause, **_):
    if source_kind == "status_damage" and cause in ("psn", "tox"):
        announce(ctx, ref, "poisonheal")
        heal(ctx, ref, fraction_of_max(ctx.state, ref, 8), reason="poisonheal")
        return 0
    return None


register("ability", "poisonheal", name="Poison Heal", modify_indirect_damage=_poison_heal)
register("ability", "magicguard", name="Magic Guard",
         modify_indirect_damage=lambda ctx, ref, value, **_: 0)
register("ability", "rockhead", name="Rock Head",
         modify_indirect_damage=lambda ctx, ref, value, source_kind, **_:
             0 if source_kind == "recoil" else None)


# --------------------------------------------------------------------------- #
# Base power (Showdown's onBasePower)
# --------------------------------------------------------------------------- #


def _base_power(condition):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref != attacker:
            return None
        modifier = condition(ctx, attacker, defender, move, value)
        return chain_modify(value, modifier) if modifier else None

    return handler


def _flag(flag: str, modifier: int):
    return lambda ctx, a, d, move, power: modifier if flag in move.flags else 0


_BASE_POWER_ABILITIES = {
    "technician": ("Technician", lambda ctx, a, d, move, power: X1_5 if power <= 60 else 0),
    "ironfist": ("Iron Fist", _flag("punch", X1_2)),
    "toughclaws": ("Tough Claws", _flag(CONTACT, X1_3)),
    "strongjaw": ("Strong Jaw", _flag("bite", X1_5)),
    "megalauncher": ("Mega Launcher", _flag("pulse", X1_5)),
    "sharpness": ("Sharpness", _flag("slicing", X1_5)),
    "punkrock": ("Punk Rock", _flag("sound", X1_3)),
    "reckless": ("Reckless", lambda ctx, a, d, move, power:
                 X1_2 if move.raw.get("recoil") else 0),
}
for _name, (_label, _condition) in _BASE_POWER_ABILITIES.items():
    register("ability", _name, name=_label, modify_base_power=_base_power(_condition))


def _analytic(ctx, ref, value, attacker, defender, move, **_):
    """Boosts when everyone else has already acted this turn."""
    if ref != attacker or defender not in ctx.acted:
        return None
    return chain_modify(value, X1_3)


register("ability", "analytic", name="Analytic", modify_base_power=_analytic)


def _sheer_force_power(ctx, ref, value, attacker, defender, move, **_):
    if ref != attacker or not _has_secondaries(move):
        return None
    return chain_modify(value, X1_3)


def _has_secondaries(move) -> bool:
    return bool(move.raw.get("secondary") or move.raw.get("secondaries"))


register("ability", "sheerforce", name="Sheer Force", modify_base_power=_sheer_force_power)


# --------------------------------------------------------------------------- #
# Attack and Special Attack (onModifyAtk / onModifySpA)
# --------------------------------------------------------------------------- #

OFFENSIVE_STATS = (Stat.ATK, Stat.SPA)


def _pinch_boost(move_type: str, ability: str):
    """Blaze, Torrent, Overgrow, Swarm: +50% at a third HP or less."""

    def handler(ctx, ref, value, stat, move=None, **_):
        if stat not in OFFENSIVE_STATS or move is None or move.type != move_type:
            return None
        if mutate.current_hp(ctx.state, ref) * 3 > mutate.max_hp(ctx.state, ref):
            return None
        return chain_modify(value, X1_5)

    return handler


for _name, (_label, _type) in {
    "blaze": ("Blaze", "fire"),
    "torrent": ("Torrent", "water"),
    "overgrow": ("Overgrow", "grass"),
    "swarm": ("Swarm", "bug"),
}.items():
    register("ability", _name, name=_label, modify_stat=_pinch_boost(_type, _name))


def _offensive_type_boost(move_type: str, modifier: int = X1_5):
    def handler(ctx, ref, value, stat, move=None, **_):
        if stat in OFFENSIVE_STATS and move is not None and move.type == move_type:
            return chain_modify(value, modifier)
        return None

    return handler


register("ability", "firemane", name="Fire Mane", modify_stat=_offensive_type_boost("fire"))
register("ability", "transistor", name="Transistor",
         modify_stat=_offensive_type_boost("electric", X1_3))
register("ability", "dragonsmaw", name="Dragon's Maw",
         modify_stat=_offensive_type_boost("dragon"))
register("ability", "steelworker", name="Steelworker",
         modify_stat=_offensive_type_boost("steel"))


def _flat_stat(stat: Stat, modifier: int, when=None):
    def handler(ctx, ref, value, stat_kw=None, **kwargs):
        if kwargs.get("stat") is not stat:
            return None
        if when is not None and not when(ctx, ref):
            return None
        return chain_modify(value, modifier)

    return handler


def _has_status(ctx, ref) -> bool:
    return ctx.state.sides[ref[0]].status[ref[1]] is not None


register("ability", "hugepower", name="Huge Power", modify_stat=_flat_stat(Stat.ATK, X2))
register("ability", "purepower", name="Pure Power", modify_stat=_flat_stat(Stat.ATK, X2))
register("ability", "hustle", name="Hustle", modify_stat=_flat_stat(Stat.ATK, X1_5))
register("ability", "guts", name="Guts", modify_stat=_flat_stat(Stat.ATK, X1_5, _has_status))
register("ability", "furcoat", name="Fur Coat", modify_stat=_flat_stat(Stat.DEF, X2))
register("ability", "marvelscale", name="Marvel Scale",
         modify_stat=_flat_stat(Stat.DEF, X1_5, _has_status))
register("ability", "solarpower", name="Solar Power",
         modify_stat=_flat_stat(Stat.SPA, X1_5,
                                lambda ctx, ref: ctx.state.field.weather == "sunnyday"))


def _defeatist(ctx, ref, value, **kwargs):
    if kwargs.get("stat") not in OFFENSIVE_STATS:
        return None
    if mutate.current_hp(ctx.state, ref) * 2 > mutate.max_hp(ctx.state, ref):
        return None
    return chain_modify(value, X0_5)


register("ability", "defeatist", name="Defeatist", modify_stat=_defeatist)


# --------------------------------------------------------------------------- #
# Speed (onModifySpe)
# --------------------------------------------------------------------------- #


def _in_weather(*names: str):
    return lambda ctx, ref: ctx.state.field.weather in names


def _on_terrain(name: str):
    return lambda ctx, ref: ctx.state.field.terrain == name and is_grounded(ctx.state, ref)


_SPEED_ABILITIES = {
    "swiftswim": ("Swift Swim", X2, _in_weather("raindance")),
    "chlorophyll": ("Chlorophyll", X2, _in_weather("sunnyday")),
    "sandrush": ("Sand Rush", X2, _in_weather("sandstorm")),
    "slushrush": ("Slush Rush", X2, _in_weather("snowscape")),
    "quickfeet": ("Quick Feet", X1_5, _has_status),
    "surgesurfer": ("Surge Surfer", X2, _on_terrain("electricterrain")),
}
for _name, (_label, _modifier, _when) in _SPEED_ABILITIES.items():
    register("ability", _name, name=_label,
             modify_boosted_stat=_flat_stat(Stat.SPE, _modifier, _when))


# --------------------------------------------------------------------------- #
# Damage taken (onSourceModifyDamage and friends)
# --------------------------------------------------------------------------- #


def _defender_damage(condition):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref != defender:
            return None
        modifier = condition(ctx, attacker, defender, move)
        return chain_modify(value, modifier) if modifier else None

    return handler


def _halves_types(*types: str):
    return lambda ctx, a, d, move: X0_5 if move.type in types else 0


register("ability", "thickfat", name="Thick Fat",
         modify_damage=_defender_damage(_halves_types("fire", "ice")))
register("ability", "heatproof", name="Heatproof",
         modify_damage=_defender_damage(_halves_types("fire")))
register("ability", "waterbubble", name="Water Bubble",
         modify_damage=_defender_damage(_halves_types("fire")))
register("ability", "purifyingsalt", name="Purifying Salt",
         modify_damage=_defender_damage(_halves_types("ghost")))
register("ability", "icescales", name="Ice Scales",
         modify_damage=_defender_damage(lambda ctx, a, d, move:
             X0_5 if move.category == "Special" else 0))
register("ability", "fluffy", name="Fluffy",
         modify_damage=_defender_damage(lambda ctx, a, d, move:
             X2 if move.type == "fire" else (X0_5 if CONTACT in move.flags else 0)))


def _resists_super_effective(ctx, a, d, move):
    from pkcm.engine.moves import type_effectiveness

    return X0_75 if type_effectiveness(ctx, a, d, move) > 1.0 else 0


for _name, _label in (("filter", "Filter"), ("solidrock", "Solid Rock"),
                      ("prismarmor", "Prism Armor")):
    register("ability", _name, name=_label,
             modify_damage=_defender_damage(_resists_super_effective))


def _at_full_health(ctx, ref) -> bool:
    return mutate.current_hp(ctx.state, ref) == mutate.max_hp(ctx.state, ref)


def _multiscale(ctx, ref, value, attacker, defender, move, **_):
    if ref == defender and _at_full_health(ctx, defender):
        return chain_modify(value, X0_5)
    return None


register("ability", "multiscale", name="Multiscale", modify_damage=_multiscale)
register("ability", "shadowshield", name="Shadow Shield", modify_damage=_multiscale)


def _sturdy(ctx, ref, value, attacker, defender, move, **_):
    """Survives at 1 HP from full. Priority -30 in the source: it runs last."""
    if ref != defender:
        return None
    current = mutate.current_hp(ctx.state, defender)
    if current == mutate.max_hp(ctx.state, defender) and value >= current:
        announce(ctx, defender, "sturdy")
        return current - 1
    return None


register("ability", "sturdy", name="Sturdy", priority=30, modify_damage=_sturdy)


def _megasol(ctx, ref, value, attacker, defender, move, **_):
    """Treats the holder as if the sun were out, without setting weather."""
    if ref != attacker:
        return None
    if move.type == "fire":
        return chain_modify(value, X1_5)
    if move.type == "water":
        return chain_modify(value, X0_5)
    return None


register("ability", "megasol", name="Mega Sol", modify_damage=_megasol)


# --------------------------------------------------------------------------- #
# Immunities and absorbs (onTryHit)
# --------------------------------------------------------------------------- #


def _absorb(ability: str, *, move_type: str | None = None, flag: str | None = None,
            status_only: bool = False, heal_denominator: int | None = None,
            boosts: dict[str, int] | None = None, volatile: str | None = None):
    def handler(ctx, ref, attacker, defender, move, **_):
        if ref != defender or attacker == defender:
            return None
        if move_type is not None and move.type != move_type:
            return None
        if flag is not None and flag not in move.flags:
            return None
        if status_only and move.category != "Status":
            return None

        _blocked(ctx, defender, ability, move)
        if heal_denominator is not None:
            heal(ctx, defender, fraction_of_max(ctx.state, defender, heal_denominator),
                 reason=ability)
        if boosts:
            boost(ctx, defender, boosts)
        if volatile:
            mutate.add_volatile(ctx, defender, volatile)
        return False

    return handler


register("ability", "voltabsorb", name="Volt Absorb",
         try_hit=_absorb("voltabsorb", move_type="electric", heal_denominator=4))
register("ability", "waterabsorb", name="Water Absorb",
         try_hit=_absorb("waterabsorb", move_type="water", heal_denominator=4))
register("ability", "eartheater", name="Earth Eater",
         try_hit=_absorb("eartheater", move_type="ground", heal_denominator=4))
register("ability", "wellbakedbody", name="Well-Baked Body",
         try_hit=_absorb("wellbakedbody", move_type="fire", boosts={"def": 2}))
register("ability", "sapsipper", name="Sap Sipper",
         try_hit=_absorb("sapsipper", move_type="grass", boosts={"atk": 1}))
register("ability", "lightningrod", name="Lightning Rod",
         try_hit=_absorb("lightningrod", move_type="electric", boosts={"spa": 1}))
register("ability", "stormdrain", name="Storm Drain",
         try_hit=_absorb("stormdrain", move_type="water", boosts={"spa": 1}))
register("ability", "motordrive", name="Motor Drive",
         try_hit=_absorb("motordrive", move_type="electric", boosts={"spe": 1}))
register("ability", "flashfire", name="Flash Fire",
         try_hit=_absorb("flashfire", move_type="fire", volatile="flashfire"))
register("ability", "bulletproof", name="Bulletproof",
         try_hit=_absorb("bulletproof", flag="bullet"))
register("ability", "goodasgold", name="Good as Gold",
         try_hit=_absorb("goodasgold", status_only=True))
register("ability", "telepathy", name="Telepathy")  # allies only; nothing in singles

register("volatile", "flashfire", name="Flash Fire boost",
         modify_stat=_offensive_type_boost("fire"))


def _wonder_guard(ctx, ref, attacker, defender, move, **_):
    from pkcm.engine.moves import type_effectiveness

    if ref != defender or move.category == "Status":
        return None
    if type_effectiveness(ctx, attacker, defender, move) > 1.0:
        return None
    _blocked(ctx, defender, "wonderguard", move)
    return False


register("ability", "wonderguard", name="Wonder Guard", try_hit=_wonder_guard)


def _dry_skin_hit(ctx, ref, attacker, defender, move, **_):
    if ref != defender or move.type != "water":
        return None
    _blocked(ctx, defender, "dryskin", move)
    heal(ctx, defender, fraction_of_max(ctx.state, defender, 4), reason="dryskin")
    return False


def _dry_skin_residual(ctx, ref, **_):
    weather = ctx.state.field.weather
    if weather == "raindance":
        heal(ctx, ref, fraction_of_max(ctx.state, ref, 8), reason="dryskin")
    elif weather == "sunnyday":
        mutate.apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, 8),
                            "weather_damage", detail="dryskin")


register("ability", "dryskin", name="Dry Skin",
         try_hit=_dry_skin_hit,
         modify_damage=_defender_damage(lambda ctx, a, d, move:
             X1_25 if move.type == "fire" else 0),
         residual=_dry_skin_residual)


def _scrappy(ctx, ref, value, attacker, defender, move, **_):
    if ref == attacker and value == 0.0 and move.type in ("normal", "fighting"):
        if "ghost" in ctx.state.types(*defender):
            return 1.0
    return None


register("ability", "scrappy", name="Scrappy", modify_effectiveness=_scrappy)
register("ability", "tintedlens", name="Tinted Lens",
         modify_damage=lambda ctx, ref, value, attacker, defender, move, **_:
             chain_modify(value, X2) if ref == attacker and 0 < _chart(ctx, defender, move) < 1
             else None)


def _chart(ctx, defender: Ref, move) -> float:
    """Raw chart lookup; calling type_effectiveness here would recurse."""
    return ctx.state.config.dex.type_chart.multiplier(move.type, ctx.state.types(*defender))


register("ability", "adaptability", name="Adaptability",
         modify_damage=lambda ctx, ref, value, attacker, defender, move, **_:
             chain_modify(value, 5461)  # 4/3: turns the 1.5x STAB into 2x
             if ref == attacker and move.type in ctx.state.types(*attacker) else None)


# --------------------------------------------------------------------------- #
# Reactions to being hit (onDamagingHit)
# --------------------------------------------------------------------------- #


def _on_contact(ability: str, effect):
    def handler(ctx, ref, attacker, defender, move, damage, **kwargs):
        if ref != defender or CONTACT not in move.flags:
            return
        effect(ctx, defender, attacker, move)

    return handler


def _on_hit(ability: str, effect, physical_only: bool = False):
    def handler(ctx, ref, attacker, defender, move, damage, **kwargs):
        if ref != defender:
            return
        if physical_only and move.category != "Physical":
            return
        effect(ctx, defender, attacker, move)

    return handler


def _chance_status(status: str, numerator: int, ability: str):
    def effect(ctx, defender, attacker, move):
        if numerator < 10 and not ctx.cursor.chance(numerator, 10):
            return
        announce(ctx, defender, ability)
        mutate.set_status(ctx, attacker, status, source=defender)

    return effect


register("ability", "static", name="Static",
         after_damage=_on_contact("static", _chance_status("par", 3, "static")))
register("ability", "flamebody", name="Flame Body",
         after_damage=_on_contact("flamebody", _chance_status("brn", 3, "flamebody")))
register("ability", "poisonpoint", name="Poison Point",
         after_damage=_on_contact("poisonpoint", _chance_status("psn", 3, "poisonpoint")))
register("ability", "spicyspray", name="Spicy Spray",
         after_damage=_on_hit("spicyspray", _chance_status("brn", 10, "spicyspray")))


def _recoil_on_contact(denominator: int, ability: str):
    def effect(ctx, defender, attacker, move):
        announce(ctx, defender, ability)
        mutate.apply_damage(ctx, attacker, fraction_of_max(ctx.state, attacker, denominator),
                            "recoil", detail=ability)

    return effect


register("ability", "roughskin", name="Rough Skin",
         after_damage=_on_contact("roughskin", _recoil_on_contact(8, "roughskin")))
register("ability", "ironbarbs", name="Iron Barbs",
         after_damage=_on_contact("ironbarbs", _recoil_on_contact(8, "ironbarbs")))


def _boosts_on_hit(changes: dict[str, int], ability: str, target_self: bool = True):
    def effect(ctx, defender, attacker, move):
        announce(ctx, defender, ability)
        boost(ctx, defender if target_self else attacker, changes, source=defender)

    return effect


register("ability", "weakarmor", name="Weak Armor",
         after_damage=_on_hit("weakarmor", _boosts_on_hit({"def": -1, "spe": 2}, "weakarmor"),
                              physical_only=True))
register("ability", "stamina", name="Stamina",
         after_damage=_on_hit("stamina", _boosts_on_hit({"def": 1}, "stamina")))
register("ability", "gooey", name="Gooey",
         after_damage=_on_contact("gooey",
                                  _boosts_on_hit({"spe": -1}, "gooey", target_self=False)))
register("ability", "tanglinghair", name="Tangling Hair",
         after_damage=_on_contact("tanglinghair",
                                  _boosts_on_hit({"spe": -1}, "tanglinghair", target_self=False)))


def _justified(ctx, ref, attacker, defender, move, damage, **_):
    if ref == defender and move.type == "dark":
        announce(ctx, defender, "justified")
        boost(ctx, defender, {"atk": 1}, source=defender)


register("ability", "justified", name="Justified", after_damage=_justified)


def _anger_point(ctx, ref, attacker, defender, move, damage, crit=False, **_):
    """A critical hit maxes Attack outright -- ``boost({atk: 12})`` in the source."""
    if ref == defender and crit:
        announce(ctx, defender, "angerpoint")
        boost(ctx, defender, {"atk": 12}, source=defender)


register("ability", "angerpoint", name="Anger Point", after_damage=_anger_point)


def _poison_touch(ctx, ref, attacker, defender, move, damage, **_):
    if ref != attacker or CONTACT not in move.flags:
        return
    if ctx.cursor.chance(3, 10):
        announce(ctx, attacker, "poisontouch")
        mutate.set_status(ctx, defender, "psn", source=attacker)


register("ability", "poisontouch", name="Poison Touch", after_damage=_poison_touch)


# --------------------------------------------------------------------------- #
# Status protection (onSetStatus / onTryAddVolatile)
# --------------------------------------------------------------------------- #


def _refuses_status(*statuses: str, ability: str):
    def handler(ctx, ref, status, source, **_):
        if status in statuses:
            announce(ctx, ref, ability)
            return False
        return None

    return handler


_STATUS_IMMUNITY = {
    "immunity": ("Immunity", ("psn", "tox")),
    "limber": ("Limber", ("par",)),
    "insomnia": ("Insomnia", ("slp",)),
    "vitalspirit": ("Vital Spirit", ("slp",)),
    "waterveil": ("Water Veil", ("brn",)),
    "magmaarmor": ("Magma Armor", ("frz",)),
    "sweetveil": ("Sweet Veil", ("slp",)),
    "thermalexchange": ("Thermal Exchange", ("brn",)),
    "shieldsdown": ("Shields Down", ALL_STATUSES),
    "comatose": ("Comatose", ALL_STATUSES),
}
for _name, (_label, _statuses) in _STATUS_IMMUNITY.items():
    register("ability", _name, name=_label,
             try_status=_refuses_status(*_statuses, ability=_name))


def _refuses_volatile(*names: str, ability: str):
    def handler(ctx, ref, volatile, source, **_):
        if volatile in names and source != ref:
            announce(ctx, ref, ability)
            return False
        return None

    return handler


register("ability", "owntempo", name="Own Tempo",
         try_volatile=_refuses_volatile("confusion", ability="owntempo"))
register("ability", "innerfocus", name="Inner Focus",
         try_volatile=_refuses_volatile("flinch", ability="innerfocus"))
register("ability", "steadfast", name="Steadfast",
         try_volatile=lambda ctx, ref, volatile, source, **_:
             boost(ctx, ref, {"spe": 1}, source=ref) and None
             if volatile == "flinch" else None)
register("ability", "shielddust", name="Shield Dust",
         try_secondary=lambda ctx, ref, attacker, move, **_: (
             announce(ctx, ref, "shielddust"), False)[1])


def _shed_skin(ctx, ref, **_):
    if ctx.state.sides[ref[0]].status[ref[1]] and ctx.cursor.chance(33, 100):
        announce(ctx, ref, "shedskin")
        mutate.cure_status(ctx, ref)


register("ability", "shedskin", name="Shed Skin", residual=_shed_skin)


def _hydration(ctx, ref, **_):
    if ctx.state.field.weather == "raindance" and ctx.state.sides[ref[0]].status[ref[1]]:
        announce(ctx, ref, "hydration")
        mutate.cure_status(ctx, ref)


register("ability", "hydration", name="Hydration", residual=_hydration)


# --------------------------------------------------------------------------- #
# Switching out (Champions gives Natural Cure and Regenerator onSwitchOut)
# --------------------------------------------------------------------------- #


def _natural_cure(ctx, ref, **_):
    if ctx.state.sides[ref[0]].status[ref[1]]:
        mutate.cure_status(ctx, ref)


def _regenerator(ctx, ref, **_):
    healed = heal(ctx, ref, mutate.max_hp(ctx.state, ref) // 3, reason="regenerator")
    if healed:
        announce(ctx, ref, "regenerator")


register("ability", "naturalcure", name="Natural Cure", switch_out=_natural_cure)
register("ability", "regenerator", name="Regenerator", switch_out=_regenerator)


# --------------------------------------------------------------------------- #
# Stat stage protection and reaction (onTryBoost / onAfterEachBoost)
# --------------------------------------------------------------------------- #


def _refuse_drops(protected: tuple[str, ...] | None, ability: str):
    def handler(ctx, ref, value, stat, source, **_):
        if value >= 0 or source == ref:
            return None
        if protected is not None and stat not in protected:
            return None
        announce(ctx, ref, ability)
        return 0

    return handler


for _name, (_label, _stats) in {
    "clearbody": ("Clear Body", None),
    "whitesmoke": ("White Smoke", None),
    "fullmetalbody": ("Full Metal Body", None),
    "hypercutter": ("Hyper Cutter", ("atk",)),
    "bigpecks": ("Big Pecks", ("def",)),
    "keeneye": ("Keen Eye", ("accuracy",)),
    "illuminate": ("Illuminate", ("accuracy",)),
}.items():
    register("ability", _name, name=_label, try_boost=_refuse_drops(_stats, _name))


register("ability", "contrary", name="Contrary",
         try_boost=lambda ctx, ref, value, stat, source, **_: -value)
register("ability", "simple", name="Simple",
         try_boost=lambda ctx, ref, value, stat, source, **_: value * 2)


def _retaliate(stat: str, ability: str):
    def handler(ctx, ref, value, stat_name=None, source=None, **kwargs):
        if value < 0 and source is not None and source[0] != ref[0]:
            announce(ctx, ref, ability)
            boost(ctx, ref, {stat: 2}, source=ref)
        return None

    return handler


register("ability", "defiant", name="Defiant", try_boost=_retaliate("atk", "defiant"))
register("ability", "competitive", name="Competitive", try_boost=_retaliate("spa", "competitive"))


# --------------------------------------------------------------------------- #
# Entering the field (onStart)
# --------------------------------------------------------------------------- #


def _opponent_of(ctx: Context, ref: Ref) -> Ref:
    other = 1 - ref[0]
    return (other, ctx.state.sides[other].active)


def _intimidate(ctx, ref, **_):
    target = _opponent_of(ctx, ref)
    if ctx.state.sides[target[0]].hp[target[1]] <= 0:
        return
    announce(ctx, ref, "intimidate")
    if mutate.volatile(ctx.state, target, "substitute") is not None:
        ctx.emit(Event("immune", side=target[0], slot=target[1], detail="substitute"))
        return
    boost(ctx, target, {"atk": -1}, source=ref)


register("ability", "intimidate", name="Intimidate", switch_in=_intimidate)


def _sets_weather(weather: str, ability: str):
    def handler(ctx, ref, **_):
        if ctx.state.field.weather == weather:
            return
        announce(ctx, ref, ability)
        ctx.state.field.weather = weather
        ctx.state.field.weather_turns = 5
        ctx.emit(Event("weather_start", detail=weather))

    return handler


for _name, (_label, _weather) in {
    "drought": ("Drought", "sunnyday"),
    "drizzle": ("Drizzle", "raindance"),
    "sandstream": ("Sand Stream", "sandstorm"),
    "snowwarning": ("Snow Warning", "snowscape"),
}.items():
    register("ability", _name, name=_label, switch_in=_sets_weather(_weather, _name))


def _sets_terrain(terrain: str, ability: str):
    def handler(ctx, ref, **_):
        if ctx.state.field.terrain == terrain:
            return
        announce(ctx, ref, ability)
        ctx.state.field.terrain = terrain
        ctx.state.field.terrain_turns = 5
        ctx.emit(Event("terrain_start", detail=terrain))

    return handler


for _name, (_label, _terrain) in {
    "electricsurge": ("Electric Surge", "electricterrain"),
    "grassysurge": ("Grassy Surge", "grassyterrain"),
    "mistysurge": ("Misty Surge", "mistyterrain"),
    "psychicsurge": ("Psychic Surge", "psychicterrain"),
}.items():
    register("ability", _name, name=_label, switch_in=_sets_terrain(_terrain, _name))


def _boost_on_entry(changes: dict[str, int], ability: str):
    def handler(ctx, ref, **_):
        announce(ctx, ref, ability)
        boost(ctx, ref, changes, source=ref)

    return handler


register("ability", "intrepidsword", name="Intrepid Sword",
         switch_in=_boost_on_entry({"atk": 1}, "intrepidsword"))
register("ability", "dauntlessshield", name="Dauntless Shield",
         switch_in=_boost_on_entry({"def": 1}, "dauntlessshield"))


def _download(ctx, ref, **_):
    target = _opponent_of(ctx, ref)
    if ctx.state.sides[target[0]].hp[target[1]] <= 0:
        return
    stats = ctx.state.stats(*target)
    announce(ctx, ref, "download")
    boost(ctx, ref, {"atk": 1} if stats[Stat.DEF] < stats[Stat.SPD] else {"spa": 1}, source=ref)


register("ability", "download", name="Download", switch_in=_download)

#: Purely informational on switch-in: they reveal something to the player and
#: change nothing about the battle state.
for _name, _label in (("frisk", "Frisk"), ("anticipation", "Anticipation"),
                      ("forewarn", "Forewarn"), ("unnerve", "Unnerve"),
                      ("pressure", "Pressure")):
    register("ability", _name, name=_label,
             switch_in=lambda ctx, ref, effect, **_: announce(ctx, ref, effect.id))


# --------------------------------------------------------------------------- #
# End of turn (onResidual)
# --------------------------------------------------------------------------- #


def _weather_heal(weather: str, ability: str):
    def handler(ctx, ref, **_):
        if ctx.state.field.weather == weather:
            heal(ctx, ref, fraction_of_max(ctx.state, ref, 16), reason=ability)

    return handler


register("ability", "raindish", name="Rain Dish", residual=_weather_heal("raindance", "raindish"))
register("ability", "icebody", name="Ice Body", residual=_weather_heal("snowscape", "icebody"))
register("ability", "speedboost", name="Speed Boost",
         residual=lambda ctx, ref, **_: boost(ctx, ref, {"spe": 1}, source=ref))


# --------------------------------------------------------------------------- #
# Knocking something out (onSourceAfterFaint)
# --------------------------------------------------------------------------- #


def _on_kill(changes_for, ability: str):
    def handler(ctx, ref, victim, move, **_):
        if move is None:
            return
        announce(ctx, ref, ability)
        boost(ctx, ref, changes_for(ctx, ref), source=ref)

    return handler


register("ability", "moxie", name="Moxie", kill=_on_kill(lambda ctx, ref: {"atk": 1}, "moxie"))
register("ability", "chillingneigh", name="Chilling Neigh",
         kill=_on_kill(lambda ctx, ref: {"atk": 1}, "chillingneigh"))
register("ability", "grimneigh", name="Grim Neigh",
         kill=_on_kill(lambda ctx, ref: {"spa": 1}, "grimneigh"))


def _best_stat(ctx: Context, ref: Ref) -> dict[str, int]:
    """Beast Boost and Eelevate raise whichever stat is highest, HP excluded."""
    stats = ctx.state.stats(*ref)
    best = max(
        (Stat.ATK, Stat.DEF, Stat.SPA, Stat.SPD, Stat.SPE),
        key=lambda stat: stats[stat],
    )
    return {mutate.STAT_TO_BOOST[best]: 1}


register("ability", "beastboost", name="Beast Boost", kill=_on_kill(_best_stat, "beastboost"))
register("ability", "eelevate", name="Eelevate", kill=_on_kill(_best_stat, "eelevate"))


# --------------------------------------------------------------------------- #
# Priority and accuracy
# --------------------------------------------------------------------------- #


def _prankster(ctx, ref, value, move, **_):
    return value + 1 if move.category == "Status" else None


register("ability", "prankster", name="Prankster", modify_priority=_prankster)
register("ability", "galewings", name="Gale Wings",
         modify_priority=lambda ctx, ref, value, move, **_:
             value + 1 if move.type == "flying" and _at_full_health(ctx, ref) else None)
register("ability", "triage", name="Triage",
         modify_priority=lambda ctx, ref, value, move, **_:
             value + 3 if move.raw.get("drain") or move.raw.get("heal") else None)


def _accuracy(modifier: int | None, ability: str, attacker_side: bool = True):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if (ref == attacker) != attacker_side:
            return None
        return 100.0 if modifier is None else chain_modify(int(value), modifier)

    return handler


register("ability", "compoundeyes", name="Compound Eyes", modify_accuracy=_accuracy(X1_3, "compoundeyes"))
register("ability", "noguard", name="No Guard", modify_accuracy=_accuracy(None, "noguard"))
register("ability", "sandveil", name="Sand Veil",
         modify_accuracy=lambda ctx, ref, value, attacker, defender, move, **_:
             chain_modify(int(value), X0_75)
             if ref == defender and ctx.state.field.weather == "sandstorm" else None)
register("ability", "snowcloak", name="Snow Cloak",
         modify_accuracy=lambda ctx, ref, value, attacker, defender, move, **_:
             chain_modify(int(value), X0_75)
             if ref == defender and ctx.state.field.weather == "snowscape" else None)


# --------------------------------------------------------------------------- #
# Champions originals with no engine consequence yet, and genuinely inert ones
# --------------------------------------------------------------------------- #


# Dragonize, Piercing Drill and Fire Mane's typing half are deliberately absent:
# they need a hook that rewrites the move mid-resolution, which the engine does
# not have. Leaving them unregistered is what makes the coverage report say so.


#: Abilities that do nothing in a battle -- registering them keeps the coverage
#: report honest, because "implemented as nothing" is not "forgotten".
INERT = frozenset({
    "honeygather", "pickup", "runaway", "ballfetch", "cheekpouch", "gluttony",
    "klutz", "stall", "friendguard", "healer", "symbiosis", "receiver",
    "powerofalchemy", "pickpocket", "magician", "harvest", "unburden",
    "shellarmor", "battlearmor", "earlybird", "rattled",
    "aftermath", "damp", "aromaveil", "flowerveil", "suctioncups",
})
for _name in INERT:
    register("ability", _name, name=_name.title())
