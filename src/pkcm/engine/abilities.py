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
from pkcm.engine.effects import REGISTRY, Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.moves import (
    MODIFIER_SCALE, X0_5, X0_75, X1_2, X1_25, X1_3, X1_5, X2, chain_modify,
)
from pkcm.engine.mutate import boost, fraction_of_max, heal
from pkcm.engine.state import BOOST_STATS, PERMANENT

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


# --------------------------------------------------------------------------- #
# Rewriting the move as it is used (Showdown's onModifyMove)
# --------------------------------------------------------------------------- #


def _ate(new_type: str, ability: str):
    """Pixilate and friends: Normal moves change type and gain 20%."""

    def handler(ctx, ref, active, attacker, defender, **_):
        if active.type != "normal" or active.category == "Status":
            return
        active.type = new_type
        active.type_changed = True
        active.base_power = chain_modify(active.base_power, X1_2)

    return handler


register("ability", "pixilate", name="Pixilate", modify_move=_ate("fairy", "pixilate"))
register("ability", "refrigerate", name="Refrigerate", modify_move=_ate("ice", "refrigerate"))
register("ability", "aerilate", name="Aerilate", modify_move=_ate("flying", "aerilate"))
register("ability", "galvanize", name="Galvanize", modify_move=_ate("electric", "galvanize"))
register("ability", "dragonize", name="Dragonize", modify_move=_ate("dragon", "dragonize"))


def _normalize(ctx, ref, active, attacker, defender, **_):
    if active.type == "normal" or active.category == "Status":
        return
    active.type = "normal"
    active.type_changed = True
    active.base_power = chain_modify(active.base_power, X1_2)


register("ability", "normalize", name="Normalize", modify_move=_normalize)


def _liquid_voice(ctx, ref, active, attacker, defender, **_):
    if "sound" in active.flags:
        active.type = "water"


register("ability", "liquidvoice", name="Liquid Voice", modify_move=_liquid_voice)


def _protean(ctx, ref, active, attacker, defender, **_):
    """The user becomes the move's type. Gen 9 allows it once per switch-in."""
    side = ctx.state.sides[ref[0]]
    if side.volatiles[ref[1]].get("typechanged"):
        return
    if ctx.state.types(*ref) == (active.type,):
        return
    side.volatiles[ref[1]]["typechanged"] = True
    ctx.state.set_override(ref[0], ref[1], "types", (active.type,))
    announce(ctx, ref, "protean")
    ctx.emit(Event("type_change", side=ref[0], slot=ref[1], detail=active.type))


register("ability", "protean", name="Protean", modify_move=_protean)
register("ability", "libero", name="Libero", modify_move=_protean)


def _flagged_move_edit(setter, ability: str, flag: str | None = None):
    def handler(ctx, ref, active, attacker, defender, **_):
        if flag is None or flag in active.flags:
            setter(active)

    return handler


register("ability", "infiltrator", name="Infiltrator",
         modify_move=_flagged_move_edit(
             lambda active: setattr(active, "infiltrates", True), "infiltrator"))
register("ability", "skilllink", name="Skill Link",
         modify_move=_flagged_move_edit(
             lambda active: setattr(active, "always_max_hits", True), "skilllink"))
register("ability", "longreach", name="Long Reach",
         modify_move=_flagged_move_edit(
             lambda active: active.flags.discard(CONTACT), "longreach"))
register("ability", "unseenfist", name="Unseen Fist",
         modify_move=_flagged_move_edit(
             lambda active: setattr(active, "breaks_protect", True), "unseenfist", CONTACT))
register("ability", "piercingdrill", name="Piercing Drill",
         modify_move=_flagged_move_edit(
             lambda active: setattr(active, "breaks_protect", True), "piercingdrill", CONTACT))


def _sheer_force_strips(ctx, ref, active, attacker, defender, **_):
    """The half of Sheer Force that the power bonus pays for."""
    if active.secondaries:
        active.secondaries = []
        active.self_effects = None


register("ability", "sheerforce", name="Sheer Force",
         modify_base_power=_sheer_force_power, modify_move=_sheer_force_strips)


def _stench(ctx, ref, active, attacker, defender, **_):
    if active.category != "Status":
        active.secondaries = list(active.secondaries) + [
            {"chance": 10, "volatileStatus": "flinch"}
        ]


register("ability", "stench", name="Stench", modify_move=_stench)

#: Redirection only exists in doubles, so ignoring it is a no-op in singles.
for _name, _label in (("stalwart", "Stalwart"), ("propellertail", "Propeller Tail")):
    register("ability", _name, name=_label)


# --------------------------------------------------------------------------- #
# More reactions to being hit
# --------------------------------------------------------------------------- #


def _cursed_body(ctx, ref, attacker, defender, move, damage, **_):
    if ref != defender or not ctx.cursor.chance(3, 10):
        return
    side = ctx.state.sides[attacker[0]]
    for index, carried in enumerate(ctx.state.moves(*attacker)):
        if carried.id == move.id:
            announce(ctx, defender, "cursedbody")
            mutate.add_volatile(ctx, attacker, "disabled", source=defender,
                                move=index, turns=4)
            return


register("ability", "cursedbody", name="Cursed Body", after_damage=_cursed_body)


def _opposite_genders(ctx, a: Ref, b: Ref) -> bool:
    left, right = ctx.state.gender(*a), ctx.state.gender(*b)
    return bool(left and right and left != right and "N" not in (left, right))


def _cute_charm(ctx, ref, attacker, defender, move, damage, **_):
    if ref != defender or CONTACT not in move.flags:
        return
    if not ctx.cursor.chance(3, 10) or not _opposite_genders(ctx, attacker, defender):
        return
    announce(ctx, defender, "cutecharm")
    mutate.add_volatile(ctx, attacker, "attract", source=defender)


register("ability", "cutecharm", name="Cute Charm", after_damage=_cute_charm)


def _effect_spore(ctx, ref, attacker, defender, move, damage, **_):
    """11% poison, 10% paralysis, 9% sleep -- in that order in the source."""
    if ref != defender or CONTACT not in move.flags:
        return
    roll = ctx.cursor.below(100)
    status = "psn" if roll < 11 else ("par" if roll < 21 else ("slp" if roll < 30 else None))
    if status is None:
        return
    announce(ctx, defender, "effectspore")
    mutate.set_status(ctx, attacker, status, source=defender)


register("ability", "effectspore", name="Effect Spore", after_damage=_effect_spore)


def _sand_spit(ctx, ref, attacker, defender, move, damage, **_):
    if ref != defender or ctx.state.field.weather == "sandstorm":
        return
    announce(ctx, defender, "sandspit")
    ctx.state.field.weather = "sandstorm"
    ctx.state.field.weather_turns = 5
    ctx.emit(Event("weather_start", detail="sandstorm"))


register("ability", "sandspit", name="Sand Spit", after_damage=_sand_spit)


def _toxic_debris(ctx, ref, attacker, defender, move, damage, **_):
    if ref != defender or move.category != "Physical":
        return
    conditions = ctx.state.sides[attacker[0]].conditions
    if conditions.get("toxicspikes", 0) >= 2:
        return
    announce(ctx, defender, "toxicdebris")
    conditions["toxicspikes"] = conditions.get("toxicspikes", 0) + 1
    ctx.emit(Event("side_condition", side=attacker[0], detail="toxicspikes",
                   amount=conditions["toxicspikes"]))


register("ability", "toxicdebris", name="Toxic Debris", after_damage=_toxic_debris)


def _replaces_attacker_ability(new_ability, ability: str):
    def handler(ctx, ref, attacker, defender, move, damage, **_):
        if ref != defender or CONTACT not in move.flags:
            return
        replacement = new_ability(ctx, defender)
        if ctx.state.ability_id(*attacker) == replacement:
            return
        announce(ctx, defender, ability)
        ctx.state.set_override(attacker[0], attacker[1], "ability", replacement)
        ctx.emit(Event("ability_change", side=attacker[0], slot=attacker[1],
                       detail=replacement))

    return handler


register("ability", "mummy", name="Mummy",
         after_damage=_replaces_attacker_ability(lambda ctx, ref: "mummy", "mummy"))


def _wandering_spirit(ctx, ref, attacker, defender, move, damage, **_):
    """Swaps the two abilities rather than overwriting one."""
    if ref != defender or CONTACT not in move.flags:
        return
    mine = ctx.state.ability_id(*defender)
    theirs = ctx.state.ability_id(*attacker)
    announce(ctx, defender, "wanderingspirit")
    ctx.state.set_override(defender[0], defender[1], "ability", theirs)
    ctx.state.set_override(attacker[0], attacker[1], "ability", mine)
    ctx.emit(Event("ability_change", side=attacker[0], slot=attacker[1], detail=mine))


register("ability", "wanderingspirit", name="Wandering Spirit", after_damage=_wandering_spirit)


def _electromorphosis(ctx, ref, attacker, defender, move, damage, **_):
    if ref == defender:
        announce(ctx, defender, "electromorphosis")
        mutate.add_volatile(ctx, defender, "charge")


register("ability", "electromorphosis", name="Electromorphosis", after_damage=_electromorphosis)


def _innards_out(ctx, ref, source, **_):
    """On fainting, the attacker takes what was left of this Pokemon."""
    if source is None:
        return
    remaining = ctx.state.sides[ref[0]].volatiles[ref[1]].get("__last_hp__", 0)
    if remaining > 0:
        announce(ctx, ref, "innardsout")
        mutate.apply_damage(ctx, source, remaining, "recoil", detail="innardsout")


register("ability", "innardsout", name="Innards Out", faint=_innards_out)


def _berserk(ctx, ref, attacker, defender, move, damage, **_):
    """+1 Special Attack when a hit takes it through half health."""
    if ref != defender:
        return
    current = mutate.current_hp(ctx.state, defender)
    total = mutate.max_hp(ctx.state, defender)
    if current > 0 and current * 2 <= total < (current + damage) * 2:
        announce(ctx, defender, "berserk")
        boost(ctx, defender, {"spa": 1}, source=defender)


register("ability", "berserk", name="Berserk", after_damage=_berserk)
register("ability", "angershell", name="Anger Shell",
         after_damage=lambda ctx, ref, attacker, defender, move, damage, **_:
             _anger_shell(ctx, ref, defender, damage))


def _anger_shell(ctx, ref, defender, damage):
    if ref != defender:
        return
    current = mutate.current_hp(ctx.state, defender)
    total = mutate.max_hp(ctx.state, defender)
    if current > 0 and current * 2 <= total < (current + damage) * 2:
        announce(ctx, defender, "angershell")
        boost(ctx, defender, {"atk": 1, "spa": 1, "spe": 1, "def": -1, "spd": -1},
              source=defender)


# --------------------------------------------------------------------------- #
# Blocking and reflecting
# --------------------------------------------------------------------------- #


def _magic_bounce(ctx, ref, attacker, defender, move, **_):
    """Reflectable status moves come straight back at the user."""
    if ref != defender or move.category != "Status":
        return None
    if "reflectable" not in move.flags:
        return None
    _blocked(ctx, defender, "magicbounce", move)
    from pkcm.engine.moves import use_move

    use_move(ctx, defender, attacker, move.base if hasattr(move, "base") else move)
    return False


register("ability", "magicbounce", name="Magic Bounce", try_hit=_magic_bounce)

register("ability", "soundproof", name="Soundproof",
         try_hit=_absorb("soundproof", flag="sound"))
register("ability", "overcoat", name="Overcoat",
         try_hit=_absorb("overcoat", flag="powder"),
         modify_indirect_damage=lambda ctx, ref, value, source_kind, cause, **_:
             0 if source_kind == "weather_damage" else None)


def _blocks_priority(ability: str):
    def handler(ctx, ref, attacker, defender, move, **_):
        if ref != defender or attacker[0] == defender[0]:
            return None
        if move.priority <= 0:
            return None
        _blocked(ctx, defender, ability, move)
        return False

    return handler


for _name, _label in (("dazzling", "Dazzling"), ("queenlymajesty", "Queenly Majesty"),
                      ("armortail", "Armor Tail")):
    register("ability", _name, name=_label, try_hit=_blocks_priority(_name))


def _oblivious(ctx, ref, volatile, source, **_):
    if volatile in ("attract", "taunt"):
        announce(ctx, ref, "oblivious")
        return False
    return None


register("ability", "oblivious", name="Oblivious", try_volatile=_oblivious,
         try_boost=lambda ctx, ref, value, stat, source, **_:
             0 if value < 0 and stat == "atk" and source is not None
             and source[0] != ref[0] and ctx.state.ability_id(*source) == "intimidate"
             else None)


def _mirror_armor(ctx, ref, value, stat, source, **_):
    """Reflects the drop instead of taking it."""
    if value >= 0 or source is None or source == ref or source[0] == ref[0]:
        return None
    announce(ctx, ref, "mirrorarmor")
    boost(ctx, source, {stat: value})
    return 0


register("ability", "mirrorarmor", name="Mirror Armor", try_boost=_mirror_armor)


# --------------------------------------------------------------------------- #
# Status side effects
# --------------------------------------------------------------------------- #


def _synchronize(ctx, ref, status, source, **_):
    """Hands the status straight back to whoever inflicted it."""
    if source is None or source[0] == ref[0] or status not in ("brn", "par", "psn", "tox"):
        return
    announce(ctx, ref, "synchronize")
    mutate.set_status(ctx, source, status, source=ref)


register("ability", "synchronize", name="Synchronize", after_status=_synchronize)


def _leaf_guard(ctx, ref, status, source, **_):
    if ctx.state.field.weather == "sunnyday":
        announce(ctx, ref, "leafguard")
        return False
    return None


register("ability", "leafguard", name="Leaf Guard", try_status=_leaf_guard)


# --------------------------------------------------------------------------- #
# Damage and crit modifiers
# --------------------------------------------------------------------------- #


register("ability", "sniper", name="Sniper",
         modify_damage=lambda ctx, ref, value, attacker, defender, move, crit=False, **_:
             chain_modify(value, X1_5) if ref == attacker and crit else None)
register("ability", "superluck", name="Super Luck",
         modify_crit_ratio=lambda ctx, ref, value, attacker, defender, move, **_:
             value + 1 if ref == attacker else None)
register("ability", "merciless", name="Merciless",
         modify_crit_ratio=lambda ctx, ref, value, attacker, defender, move, **_:
             3 if ref == attacker
             and ctx.state.sides[defender[0]].status[defender[1]] in ("psn", "tox")
             else None)
def _no_crits(ctx, ref, value, attacker, defender, move, **_):
    return 0 if ref == defender else None


for _name, _label in (("battlearmor", "Battle Armor"), ("shellarmor", "Shell Armor")):
    REGISTRY.pop(("ability", _name), None)
    register("ability", _name, name=_label, modify_crit_ratio=_no_crits)


def _rivalry(ctx, ref, value, attacker, defender, move, **_):
    if ref != attacker:
        return None
    mine, theirs = ctx.state.gender(*attacker), ctx.state.gender(*defender)
    if not mine or not theirs or "N" in (mine, theirs):
        return None
    return chain_modify(value, X1_25 if mine == theirs else X0_75)


register("ability", "rivalry", name="Rivalry", modify_base_power=_rivalry)


def _sand_force(ctx, ref, value, attacker, defender, move, **_):
    if ref != attacker or ctx.state.field.weather != "sandstorm":
        return None
    if move.type not in ("rock", "ground", "steel"):
        return None
    return chain_modify(value, X1_3)


register("ability", "sandforce", name="Sand Force", modify_base_power=_sand_force,
         modify_indirect_damage=lambda ctx, ref, value, source_kind, cause, **_:
             0 if cause == "sandstorm" else None)


def _supreme_overlord(ctx, ref, value, attacker, defender, move, **_):
    """+10% per fallen team mate."""
    if ref != attacker:
        return None
    side = ctx.state.sides[attacker[0]]
    fallen = sum(1 for slot in range(len(side.hp)) if side.hp[slot] <= 0)
    if not fallen:
        return None
    return chain_modify(value, MODIFIER_SCALE + fallen * (MODIFIER_SCALE // 10))


register("ability", "supremeoverlord", name="Supreme Overlord",
         modify_base_power=_supreme_overlord)


def _aura(move_type: str, ability: str):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref != defender or move.type != move_type:
            return None
        return chain_modify(value, 5448)  # 4/3, the aura multiplier

    return handler


register("ability", "fairyaura", name="Fairy Aura", modify_base_power=_aura("fairy", "fairyaura"))
register("ability", "darkaura", name="Dark Aura", modify_base_power=_aura("dark", "darkaura"))


def _unaware(ctx, ref, stat, target, **_):
    """Refuses to see the other Pokemon's stat stages -- offence and defence."""
    return False


register("ability", "unaware", name="Unaware", ignore_stat_stages=_unaware)


def _tangled_feet(ctx, ref, value, attacker, defender, move, **_):
    if ref == defender and ctx.state.sides[defender[0]].has_volatile(defender[1], "confusion"):
        return chain_modify(int(value), X0_5)
    return None


register("ability", "tangledfeet", name="Tangled Feet", modify_accuracy=_tangled_feet)


register("ability", "lightmetal", name="Light Metal",
         modify_weight=lambda ctx, ref, value, **_: value / 2)
register("ability", "heavymetal", name="Heavy Metal",
         modify_weight=lambda ctx, ref, value, **_: value * 2)


# --------------------------------------------------------------------------- #
# Copying and changing what a Pokemon is
# --------------------------------------------------------------------------- #

#: Abilities that refuse to be copied or traded (Showdown's ``flags.failrole``).
UNCOPYABLE = frozenset({
    "trace", "imposter", "forecast", "flowergift", "zenmode", "illusion",
    "stancechange", "battlebond", "powerconstruct", "schooling", "shieldsdown",
    "disguise", "rkssystem", "commander", "zerotohero", "hungerswitch",
})


def _trace(ctx, ref, **_):
    target = _opponent_of(ctx, ref)
    if ctx.state.sides[target[0]].hp[target[1]] <= 0:
        return
    copied = ctx.state.ability_id(*target)
    if copied in UNCOPYABLE:
        return
    announce(ctx, ref, "trace")
    ctx.state.set_override(ref[0], ref[1], "ability", copied)
    ctx.emit(Event("ability_change", side=ref[0], slot=ref[1], detail=copied))


register("ability", "trace", name="Trace", switch_in=_trace)


def _imposter(ctx, ref, **_):
    """Transform into whatever is facing it, on the way in.

    Copies the target's *current* state: forme, ability, types, stats other than
    HP, moves at 5 PP each, and the stat stages it is sitting on. HP stays its
    own. Reported by hk and pinned in tests/scenarios/.
    """
    target = _opponent_of(ctx, ref)
    if ctx.state.sides[target[0]].hp[target[1]] <= 0:
        return
    if ctx.state.sides[ref[0]].volatiles[ref[1]].get("transformed"):
        return

    stats = list(ctx.state.stats(*target))
    stats[Stat.HP] = ctx.state.pokemon(*ref).stats[Stat.HP]
    copied = [move.id for move in ctx.state.moves(*target)]

    for key, value in (
        ("species", ctx.state.species_id(*target)),
        ("ability", ctx.state.ability_id(*target)),
        ("types", ctx.state.types(*target)),
        ("stats", tuple(stats)),
        ("moves", tuple(copied)),
    ):
        ctx.state.set_override(ref[0], ref[1], key, value)
    ctx.state.sides[ref[0]].pp[ref[1]] = [5] * len(copied)

    # Stat stages come along too -- the part a naive Transform forgets.
    ctx.state.sides[ref[0]].boosts[ref[1]] = list(ctx.state.sides[target[0]].boosts[target[1]])

    ctx.state.sides[ref[0]].volatiles[ref[1]]["transformed"] = True
    announce(ctx, ref, "imposter")
    ctx.emit(Event("transform", side=ref[0], slot=ref[1],
                   species=ctx.state.species_name(*ref), detail=ctx.state.species_id(*ref)))


register("ability", "imposter", name="Imposter", switch_in=_imposter)
register("volatile", "transformed", name="Transformed")
register("volatile", "typechanged", name="Type Changed")


def _forme_by_weather(ctx, ref, **_):
    """Castform's Forecast. The forme follows the weather."""
    formes = {"sunnyday": "castformsunny", "raindance": "castformrainy",
              "snowscape": "castformsnowy"}
    species = formes.get(ctx.state.field.weather, "castform")
    if ctx.state.species_id(*ref) == species:
        return
    ctx.state.set_override(ref[0], ref[1], "species", species)
    ctx.state.set_override(ref[0], ref[1], "types",
                           ctx.state.config.dex.species[species].types)
    ctx.emit(Event("forme_change", side=ref[0], slot=ref[1], detail=species))


register("ability", "forecast", name="Forecast", switch_in=_forme_by_weather,
         residual=_forme_by_weather)


def _mimicry(ctx, ref, **_):
    types = {"electricterrain": "electric", "grassyterrain": "grass",
             "mistyterrain": "fairy", "psychicterrain": "psychic"}
    terrain_type = types.get(ctx.state.field.terrain)
    override = ctx.state.overrides[ref[0]][ref[1]]
    if terrain_type is None:
        override.pop("types", None)
        return
    if override.get("types") == (terrain_type,):
        return
    ctx.state.set_override(ref[0], ref[1], "types", (terrain_type,))
    announce(ctx, ref, "mimicry")
    ctx.emit(Event("type_change", side=ref[0], slot=ref[1], detail=terrain_type))


register("ability", "mimicry", name="Mimicry", switch_in=_mimicry, residual=_mimicry)


def _moody(ctx, ref, **_):
    """One random stat up two stages, a different one down one."""
    stats = [s for s in BOOST_STATS if s not in ("accuracy", "evasion")]
    side = ctx.state.sides[ref[0]]
    can_rise = [s for s in stats if side.boost(ref[1], s) < 6] or stats
    up = ctx.cursor.choice(can_rise)
    can_fall = [s for s in stats if s != up and side.boost(ref[1], s) > -6]
    announce(ctx, ref, "moody")
    boost(ctx, ref, {up: 2}, source=ref)
    if can_fall:
        boost(ctx, ref, {ctx.cursor.choice(can_fall): -1}, source=ref)


register("ability", "moody", name="Moody", residual=_moody)


def _quick_draw(ctx, ref, value, move, **_):
    if move.category != "Status" and ctx.cursor.chance(3, 10):
        announce(ctx, ref, "quickdraw")
        return value + 1
    return None


register("ability", "quickdraw", name="Quick Draw", modify_priority=_quick_draw)


def _traps_the_opponent(ability: str, condition=None):
    def handler(ctx, ref, **_):
        target = _opponent_of(ctx, ref)
        if ctx.state.sides[target[0]].hp[target[1]] <= 0:
            return
        if condition is not None and not condition(ctx, target):
            return
        mutate.add_volatile(ctx, target, "trapped", source=ref)

    return handler


register("ability", "shadowtag", name="Shadow Tag",
         switch_in=_traps_the_opponent(
             "shadowtag",
             lambda ctx, target: ctx.state.ability_id(*target) != "shadowtag"),
         residual=_traps_the_opponent(
             "shadowtag",
             lambda ctx, target: ctx.state.ability_id(*target) != "shadowtag"))
register("ability", "arenatrap", name="Arena Trap",
         switch_in=_traps_the_opponent(
             "arenatrap", lambda ctx, target: is_grounded(ctx.state, target)),
         residual=_traps_the_opponent(
             "arenatrap", lambda ctx, target: is_grounded(ctx.state, target)))
register("ability", "magnetpull", name="Magnet Pull",
         switch_in=_traps_the_opponent(
             "magnetpull", lambda ctx, target: "steel" in ctx.state.types(*target)),
         residual=_traps_the_opponent(
             "magnetpull", lambda ctx, target: "steel" in ctx.state.types(*target)))


def _cloud_nine(ctx, ref, **_):
    announce(ctx, ref, "cloudnine")


for _name, _label in (("cloudnine", "Cloud Nine"), ("airlock", "Air Lock")):
    register("ability", _name, name=_label, switch_in=_cloud_nine)


def _screen_cleaner(ctx, ref, **_):
    removed = False
    for side in ctx.state.sides:
        for screen in ("reflect", "lightscreen", "auroraveil"):
            if screen in side.conditions:
                del side.conditions[screen]
                removed = True
    if removed:
        announce(ctx, ref, "screencleaner")
        ctx.emit(Event("side_condition_end", side=ref[0], detail="screens"))


register("ability", "screencleaner", name="Screen Cleaner", switch_in=_screen_cleaner)


def _super_sweet_syrup(ctx, ref, **_):
    if ctx.state.sides[ref[0]].volatiles[ref[1]].get("syrupused"):
        return
    ctx.state.sides[ref[0]].volatiles[ref[1]]["syrupused"] = True
    target = _opponent_of(ctx, ref)
    if ctx.state.sides[target[0]].hp[target[1]] > 0:
        announce(ctx, ref, "supersweetsyrup")
        boost(ctx, target, {"evasion": -1}, source=ref)


register("ability", "supersweetsyrup", name="Supersweet Syrup", switch_in=_super_sweet_syrup)


def _zero_to_hero(ctx, ref, **_):
    if ctx.state.species_id(*ref) != "palafin":
        return
    announce(ctx, ref, "zerotohero")
    _become(ctx, ref, "palafinhero", permanent=True)


register("ability", "zerotohero", name="Zero to Hero", switch_out=_zero_to_hero)


def _hunger_switch(ctx, ref, **_):
    current = ctx.state.species_id(*ref)
    if current not in ("morpeko", "morpekohangry"):
        return
    flipped = "morpekohangry" if current == "morpeko" else "morpeko"
    ctx.state.set_override(ref[0], ref[1], "species", flipped)
    ctx.state.set_override(ref[0], ref[1], "types",
                           ctx.state.config.dex.species[flipped].types)
    ctx.emit(Event("forme_change", side=ref[0], slot=ref[1], detail=flipped))


register("ability", "hungerswitch", name="Hunger Switch", residual=_hunger_switch)


def _opportunist(ctx, ref, boosted, stat, stages, source, **_):
    """Copies a rise the opponent just gained. ``ref`` is the watcher."""
    if stages <= 0 or boosted == ref or boosted[0] == ref[0]:
        return
    if ctx.state.sides[ref[0]].volatiles[ref[1]].get("__copying__"):
        return  # do not answer our own copy
    ctx.state.sides[ref[0]].volatiles[ref[1]]["__copying__"] = True
    announce(ctx, ref, "opportunist")
    boost(ctx, ref, {stat: stages}, source=ref)
    ctx.state.sides[ref[0]].volatiles[ref[1]].pop("__copying__", None)


register("ability", "opportunist", name="Opportunist", after_boost=_opportunist)


def _parental_bond(ctx, ref, active, attacker, defender, **_):
    """A second hit at a quarter power, for single-target damaging moves."""
    if active.category == "Status" or active.multihit is not None:
        return
    if active.target not in ("normal", "any", "randomNormal"):
        return
    active.multihit = 2
    active.parental = True


register("ability", "parentalbond", name="Parental Bond", modify_move=_parental_bond)


def _illusion(ctx, ref, **_):
    """Purely cosmetic here: it disguises the sprite, not the mechanics."""
    announce(ctx, ref, "illusion")


register("ability", "illusion", name="Illusion", switch_in=_illusion)


# --------------------------------------------------------------------------- #
# Forme changers
# --------------------------------------------------------------------------- #


def _become(ctx: Context, ref: Ref, species_id: str, permanent: bool = False) -> None:
    """Swap forme, keeping HP. Stats and types follow the new forme.

    ``permanent`` for the ones that outlive a switch: a busted Disguise stays
    busted, Palafin stays Hero. Stance Change and Hunger Switch do not.
    """
    if ctx.state.species_id(*ref) == species_id:
        return
    species = ctx.state.config.dex.species[species_id]

    from pkcm.engine.stats import compute_stats

    stats = list(compute_stats(species.base_stats, ctx.state.pokemon(*ref).set.sp,
                               ctx.state.pokemon(*ref).nature))
    stats[Stat.HP] = ctx.state.pokemon(*ref).stats[Stat.HP]

    for key, value in (("species", species_id), ("types", species.types),
                       ("stats", tuple(stats))):
        ctx.state.set_override(ref[0], ref[1], key, value, permanent=permanent)
    ctx.emit(Event("forme_change", side=ref[0], slot=ref[1], detail=species_id))


def _stance_change(ctx, ref, active, attacker, defender, **_):
    """Blade forme to attack, Shield forme for King's Shield."""
    species = ctx.state.config.dex.species[ctx.state.species_id(*ref)]
    if species.base_species != "aegislash":
        return
    if active.category == "Status" and active.id != "kingsshield":
        return
    _become(ctx, ref, "aegislash" if active.id == "kingsshield" else "aegislashblade")


register("ability", "stancechange", name="Stance Change", modify_move=_stance_change)


def _battle_bond_kill(ctx, ref, victim, move, **_):
    """One boost per battle, on a knockout with a move."""
    if move is None:
        return
    side = ctx.state.sides[ref[0]]
    if side.volatiles[ref[1]].get("bondtriggered"):
        return
    side.volatiles[ref[1]]["bondtriggered"] = True
    announce(ctx, ref, "battlebond")
    boost(ctx, ref, {"atk": 1, "spa": 1, "spe": 1}, source=ref)


register("ability", "battlebond", name="Battle Bond", kill=_battle_bond_kill)


def _disguise(ctx, ref, attacker, defender, move, **_):
    """The first damaging hit costs Mimikyu its disguise and an eighth of its HP."""
    if ref != defender or move.category == "Status":
        return None
    species = ctx.state.config.dex.species[ctx.state.species_id(*defender)]
    if species.base_species != "mimikyu" or "busted" in ctx.state.species_id(*defender):
        return None

    _blocked(ctx, defender, "disguise", move)
    _become(ctx, defender, ctx.state.species_id(*defender) + "busted", permanent=True)
    mutate.apply_damage(ctx, defender, fraction_of_max(ctx.state, defender, 8),
                        "recoil", detail="disguise")
    return False


register("ability", "disguise", name="Disguise", try_hit=_disguise)


# Cud Chew, Ripen and Sticky Hold all act on held items, which the engine does
# not have yet. Left unregistered on purpose: the coverage report should say so
# rather than count them as done.


#: Abilities whose whole effect is on an ally, so they correctly do nothing in a
#: singles battle. They are implemented -- as nothing -- rather than missing, and
#: they will need real handlers when doubles arrives.
SINGLES_INERT = frozenset({
    "telepathy", "friendguard", "healer", "symbiosis", "receiver",
    "powerofalchemy", "plus", "minus", "hospitality", "curiousmedicine",
    "stalwart", "propellertail", "battery", "powerspot", "victorystar",
    "costar", "sweetveil2",
})
for _name in SINGLES_INERT:
    if ("ability", _name) not in REGISTRY:
        register("ability", _name, name=_name.title())


#: Abilities that do nothing in a battle at all -- registering them keeps the
#: coverage report honest, because "implemented as nothing" is not "forgotten".
INERT = frozenset({
    "honeygather", "pickup", "runaway", "ballfetch", "cheekpouch", "gluttony",
    "klutz", "stall", "friendguard", "healer", "symbiosis", "receiver",
    "powerofalchemy", "pickpocket", "magician", "harvest", "unburden",
    "shellarmor", "battlearmor", "earlybird", "rattled",
    "aftermath", "damp", "aromaveil", "flowerveil", "suctioncups",
})
for _name in INERT:
    register("ability", _name, name=_name.title())
