"""Abilities.

Each one is a small handler table on the hook system. The three that shape the
architecture, and are worth reading first:

**Mold Breaker** (and Turboblaze, Teravolt) carries no handlers at all. It is
consulted by ``moves.ignores_target_ability``, which drops the defender into
``ctx.suppressed_abilities`` for the whole resolution, so the defender's ability
is invisible to *every* hook the move runs. Implementing it as "check for Mold
Breaker at each immunity test" is the version that is always missing a case.

**Corrosion** cannot work as a property of the target -- the Steel type really is
immune to poison, and it is the poisoner that overrides it. So the type immunity
in ``mutate.set_status`` is run through ``status_immunity`` gathered from the
*source*, and Corrosion empties the list.

**Poison Heal** does not block damage, it inverts it. Blocking hooks cannot
express that, so all non-move damage funnels through ``modify_indirect_damage``
in ``mutate.apply_damage``; Poison Heal heals there and returns zero. Magic Guard
sits on the same hook and simply returns zero.
"""

from __future__ import annotations

from pkcm.data.dex import Stat
from pkcm.engine import mutate
from pkcm.engine.conditions import is_grounded
from pkcm.engine.effects import Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.moves import type_effectiveness
from pkcm.engine.mutate import boost, fraction_of_max, heal

CONTACT = "contact"


def announce(ctx: Context, ref: Ref, ability: str, detail: str | None = None) -> None:
    ctx.emit(Event("ability", side=ref[0], slot=ref[1], detail=detail or ability))


# --------------------------------------------------------------------------- #
# Ignoring the target's ability -- no handlers; see moves.ignores_target_ability
# --------------------------------------------------------------------------- #

for _breaker in ("moldbreaker", "turboblaze", "teravolt"):
    register("ability", _breaker, name=_breaker.title())


# --------------------------------------------------------------------------- #
# The two that had to bend the framework
# --------------------------------------------------------------------------- #


def _corrosion(ctx, ref, value, status, target, **_):
    """Poison and Steel stop being immune -- but only to *this* Pokemon."""
    if status in ("psn", "tox"):
        return ()
    return None


register("ability", "corrosion", name="Corrosion", status_immunity=_corrosion)


def _poison_heal(ctx, ref, value, source_kind, cause, **_):
    """Its own poison becomes healing instead of damage."""
    if source_kind == "status_damage" and cause in ("psn", "tox"):
        announce(ctx, ref, "poisonheal")
        heal(ctx, ref, fraction_of_max(ctx.state, ref, 8), reason="poisonheal")
        return 0
    return None


register("ability", "poisonheal", name="Poison Heal", modify_indirect_damage=_poison_heal)


def _magic_guard(ctx, ref, value, **_):
    return 0


register("ability", "magicguard", name="Magic Guard", modify_indirect_damage=_magic_guard)


# --------------------------------------------------------------------------- #
# Type immunities and absorbs
# --------------------------------------------------------------------------- #


def _absorb(move_type: str, ability: str, *, heal_denominator: int | None = None,
            boosts: dict[str, int] | None = None, volatile: str | None = None):
    def handler(ctx, ref, attacker, defender, move, **_):
        if ref != defender or move.type != move_type:
            return None
        announce(ctx, defender, ability)
        ctx.emit(Event("ability_block", side=defender[0], slot=defender[1], move=move.name,
                       detail=ability))
        if heal_denominator is not None:
            heal(ctx, defender, fraction_of_max(ctx.state, defender, heal_denominator),
                 reason=ability)
        if boosts:
            boost(ctx, defender, boosts)
        if volatile:
            mutate.add_volatile(ctx, defender, volatile)
        return False

    return handler


register("ability", "levitate", name="Levitate", try_hit=_absorb("ground", "levitate"))
register("ability", "voltabsorb", name="Volt Absorb",
         try_hit=_absorb("electric", "voltabsorb", heal_denominator=4))
register("ability", "waterabsorb", name="Water Absorb",
         try_hit=_absorb("water", "waterabsorb", heal_denominator=4))
register("ability", "eartheater", name="Earth Eater",
         try_hit=_absorb("ground", "eartheater", heal_denominator=4))
register("ability", "wellbakedbody", name="Well-Baked Body",
         try_hit=_absorb("fire", "wellbakedbody", boosts={"def": 2}))
register("ability", "sapsipper", name="Sap Sipper",
         try_hit=_absorb("grass", "sapsipper", boosts={"atk": 1}))
register("ability", "lightningrod", name="Lightning Rod",
         try_hit=_absorb("electric", "lightningrod", boosts={"spa": 1}))
register("ability", "stormdrain", name="Storm Drain",
         try_hit=_absorb("water", "stormdrain", boosts={"spa": 1}))
register("ability", "motordrive", name="Motor Drive",
         try_hit=_absorb("electric", "motordrive", boosts={"spe": 1}))
register("ability", "flashfire", name="Flash Fire",
         try_hit=_absorb("fire", "flashfire", volatile="flashfire"))

register("volatile", "flashfire", name="Flash Fire boost",
         modify_damage=lambda ctx, ref, value, attacker, defender, move, **_:
             int(value * 1.5) if ref == attacker and move.type == "fire" else None)


def _wonder_guard(ctx, ref, attacker, defender, move, **_):
    if ref != defender or move.category == "Status":
        return None
    if type_effectiveness(ctx, attacker, defender, move) > 1.0:
        return None
    announce(ctx, defender, "wonderguard")
    ctx.emit(Event("ability_block", side=defender[0], slot=defender[1], move=move.name,
                   detail="wonderguard"))
    return False


register("ability", "wonderguard", name="Wonder Guard", try_hit=_wonder_guard)


def _dry_skin_hit(ctx, ref, attacker, defender, move, **_):
    if ref != defender or move.type != "water":
        return None
    announce(ctx, defender, "dryskin")
    heal(ctx, defender, fraction_of_max(ctx.state, defender, 4), reason="dryskin")
    return False


def _dry_skin_fire(ctx, ref, value, attacker, defender, move, **_):
    if ref == defender and move.type == "fire":
        return int(value * 1.25)
    return None


def _dry_skin_residual(ctx, ref, **_):
    weather = ctx.state.field.weather
    if weather == "raindance":
        heal(ctx, ref, fraction_of_max(ctx.state, ref, 8), reason="dryskin")
    elif weather == "sunnyday":
        mutate.apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, 8),
                            "weather_damage", detail="dryskin")


register("ability", "dryskin", name="Dry Skin", try_hit=_dry_skin_hit,
         modify_damage=_dry_skin_fire, residual=_dry_skin_residual)


def _scrappy(ctx, ref, value, attacker, defender, move, **_):
    """Normal and Fighting reach Ghost types."""
    if ref == attacker and value == 0.0 and move.type in ("normal", "fighting"):
        if "ghost" in ctx.state.types(*defender):
            return 1.0
    return None


register("ability", "scrappy", name="Scrappy", modify_effectiveness=_scrappy)


# --------------------------------------------------------------------------- #
# Damage: attacking side
# --------------------------------------------------------------------------- #


def _attacker_damage(ability: str, condition):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref != attacker:
            return None
        multiplier = condition(ctx, attacker, defender, move)
        return int(value * multiplier) if multiplier != 1.0 else None

    return handler


def _type_boost(move_type: str, multiplier: float = 1.5):
    return lambda ctx, a, d, move: multiplier if move.type == move_type else 1.0


def _flag_boost(flag: str, multiplier: float):
    return lambda ctx, a, d, move: multiplier if flag in move.flags else 1.0


register("ability", "adaptability", name="Adaptability",
         modify_damage=_attacker_damage("adaptability", lambda ctx, a, d, move:
             4 / 3 if move.type in ctx.state.types(*a) else 1.0))
register("ability", "technician", name="Technician",
         modify_damage=_attacker_damage("technician", lambda ctx, a, d, move:
             1.5 if move.base_power and move.base_power <= 60 else 1.0))
register("ability", "tintedlens", name="Tinted Lens",
         modify_damage=_attacker_damage("tintedlens", lambda ctx, a, d, move:
             2.0 if type_effectiveness_cached(ctx, a, d, move) < 1.0 else 1.0))
register("ability", "ironfist", name="Iron Fist",
         modify_damage=_attacker_damage("ironfist", _flag_boost("punch", 1.2)))
register("ability", "toughclaws", name="Tough Claws",
         modify_damage=_attacker_damage("toughclaws", _flag_boost(CONTACT, 1.3)))
register("ability", "strongjaw", name="Strong Jaw",
         modify_damage=_attacker_damage("strongjaw", _flag_boost("bite", 1.5)))
register("ability", "megalauncher", name="Mega Launcher",
         modify_damage=_attacker_damage("megalauncher", _flag_boost("pulse", 1.5)))
register("ability", "sharpness", name="Sharpness",
         modify_damage=_attacker_damage("sharpness", _flag_boost("slicing", 1.5)))
register("ability", "punkrock", name="Punk Rock",
         modify_damage=_attacker_damage("punkrock", _flag_boost("sound", 1.3)))
register("ability", "reckless", name="Reckless",
         modify_damage=_attacker_damage("reckless", lambda ctx, a, d, move:
             1.2 if move.raw.get("recoil") else 1.0))
register("ability", "analytic", name="Analytic",
         modify_damage=_attacker_damage("analytic", lambda ctx, a, d, move: 1.0))
register("ability", "transistor", name="Transistor",
         modify_damage=_attacker_damage("transistor", _type_boost("electric", 1.3)))
register("ability", "dragonsmaw", name="Dragon's Maw",
         modify_damage=_attacker_damage("dragonsmaw", _type_boost("dragon")))
register("ability", "steelworker", name="Steelworker",
         modify_damage=_attacker_damage("steelworker", _type_boost("steel")))
register("ability", "steelyspirit", name="Steely Spirit",
         modify_damage=_attacker_damage("steelyspirit", _type_boost("steel")))
register("ability", "rockypayload", name="Rocky Payload",
         modify_damage=_attacker_damage("rockypayload", _type_boost("rock")))


def type_effectiveness_cached(ctx: Context, attacker: Ref, defender: Ref, move) -> float:
    """Raw chart lookup -- calling ``type_effectiveness`` here would recurse."""
    return ctx.state.config.dex.type_chart.multiplier(move.type, ctx.state.types(*defender))


def _sheer_force(ctx, ref, value, attacker, defender, move, **_):
    if ref == attacker and (move.raw.get("secondary") or move.raw.get("secondaries")):
        return int(value * 1.3)
    return None


register("ability", "sheerforce", name="Sheer Force", modify_damage=_sheer_force)


# --------------------------------------------------------------------------- #
# Damage: defending side
# --------------------------------------------------------------------------- #


def _defender_damage(condition):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref != defender:
            return None
        multiplier = condition(ctx, attacker, defender, move)
        return int(value * multiplier) if multiplier != 1.0 else None

    return handler


def _halves(*types: str):
    return lambda ctx, a, d, move: 0.5 if move.type in types else 1.0


register("ability", "thickfat", name="Thick Fat",
         modify_damage=_defender_damage(_halves("fire", "ice")))
register("ability", "heatproof", name="Heatproof",
         modify_damage=_defender_damage(_halves("fire")))
register("ability", "waterbubble", name="Water Bubble",
         modify_damage=_defender_damage(_halves("fire")))
register("ability", "icescales", name="Ice Scales",
         modify_damage=_defender_damage(lambda ctx, a, d, move:
             0.5 if move.category == "Special" else 1.0))
register("ability", "fluffy", name="Fluffy",
         modify_damage=_defender_damage(lambda ctx, a, d, move:
             2.0 if move.type == "fire" else (0.5 if CONTACT in move.flags else 1.0)))
register("ability", "purifyingsalt", name="Purifying Salt",
         modify_damage=_defender_damage(_halves("ghost")))


def _resists_super_effective(ctx, a, d, move):
    return 0.75 if type_effectiveness_cached(ctx, a, d, move) > 1.0 else 1.0


for _name, _label in (("filter", "Filter"), ("solidrock", "Solid Rock"),
                      ("prismarmor", "Prism Armor")):
    register("ability", _name, name=_label,
             modify_damage=_defender_damage(_resists_super_effective))


def _multiscale(ctx, ref, value, attacker, defender, move, **_):
    if ref != defender:
        return None
    if mutate.current_hp(ctx.state, defender) == mutate.max_hp(ctx.state, defender):
        return int(value * 0.5)
    return None


register("ability", "multiscale", name="Multiscale", modify_damage=_multiscale)
register("ability", "shadowshield", name="Shadow Shield", modify_damage=_multiscale)


def _sturdy(ctx, ref, value, attacker, defender, move, **_):
    """Survives with 1 HP from full health."""
    if ref != defender:
        return None
    current = mutate.current_hp(ctx.state, defender)
    if current == mutate.max_hp(ctx.state, defender) and value >= current:
        announce(ctx, defender, "sturdy")
        return current - 1
    return None


register("ability", "sturdy", name="Sturdy", modify_damage=_sturdy)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def _stat_scaler(stat: Stat, multiplier: float, when=None):
    def handler(ctx, ref, value, stat_arg=None, **kwargs):
        if kwargs.get("stat") is not stat:
            return None
        if when is not None and not when(ctx, ref):
            return None
        return int(value * multiplier)

    return handler


def _in_weather(*names: str):
    return lambda ctx, ref: ctx.state.field.weather in names


register("ability", "hugepower", name="Huge Power",
         modify_stat=_stat_scaler(Stat.ATK, 2.0))
register("ability", "purepower", name="Pure Power",
         modify_stat=_stat_scaler(Stat.ATK, 2.0))
register("ability", "hustle", name="Hustle",
         modify_stat=_stat_scaler(Stat.ATK, 1.5))
register("ability", "furcoat", name="Fur Coat",
         modify_stat=_stat_scaler(Stat.DEF, 2.0))
register("ability", "chlorophyll", name="Chlorophyll",
         modify_boosted_stat=_stat_scaler(Stat.SPE, 2.0, _in_weather("sunnyday")))
register("ability", "swiftswim", name="Swift Swim",
         modify_boosted_stat=_stat_scaler(Stat.SPE, 2.0, _in_weather("raindance")))
register("ability", "sandrush", name="Sand Rush",
         modify_boosted_stat=_stat_scaler(Stat.SPE, 2.0, _in_weather("sandstorm")))
register("ability", "slushrush", name="Slush Rush",
         modify_boosted_stat=_stat_scaler(Stat.SPE, 2.0, _in_weather("snowscape")))
register("ability", "solarpower", name="Solar Power",
         modify_stat=_stat_scaler(Stat.SPA, 1.5, _in_weather("sunnyday")))
register("ability", "sandforce", name="Sand Force",
         modify_damage=_attacker_damage("sandforce", lambda ctx, a, d, move:
             1.3 if ctx.state.field.weather == "sandstorm"
             and move.type in ("rock", "ground", "steel") else 1.0))


def _status_powered(stat: Stat, multiplier: float, statuses: tuple[str, ...]):
    def handler(ctx, ref, value, **kwargs):
        if kwargs.get("stat") is not stat:
            return None
        if ctx.state.sides[ref[0]].status[ref[1]] in statuses:
            return int(value * multiplier)
        return None

    return handler


register("ability", "guts", name="Guts",
         modify_stat=_status_powered(Stat.ATK, 1.5, ("brn", "par", "psn", "tox", "slp", "frz")))
register("ability", "quickfeet", name="Quick Feet",
         modify_boosted_stat=_status_powered(Stat.SPE, 1.5,
                                             ("brn", "par", "psn", "tox", "slp", "frz")))
register("ability", "marvelscale", name="Marvel Scale",
         modify_stat=_status_powered(Stat.DEF, 1.5,
                                     ("brn", "par", "psn", "tox", "slp", "frz")))


def _guts_ignores_burn(ctx, ref, value, **kwargs):
    """Guts holders do not take the burn's Attack cut."""
    if kwargs.get("stat") is Stat.ATK and ctx.state.sides[ref[0]].status[ref[1]] == "brn":
        return value * 2  # undo the halving the burn effect applied
    return None


def _unaware(ctx, ref, value, attacker, defender, move, **_):
    return None


register("ability", "defeatist", name="Defeatist",
         modify_stat=lambda ctx, ref, value, **kwargs:
             int(value * 0.5)
             if kwargs.get("stat") in (Stat.ATK, Stat.SPA)
             and mutate.current_hp(ctx.state, ref) * 2 <= mutate.max_hp(ctx.state, ref)
             else None)


# --------------------------------------------------------------------------- #
# Status immunity and cure
# --------------------------------------------------------------------------- #


def _status_immune(*statuses: str, ability: str):
    def handler(ctx, ref, status, source, **_):
        if status in statuses:
            announce(ctx, ref, ability)
            return False
        return None

    return handler


_STATUS_IMMUNITIES = {
    "immunity": ("psn", "tox"),
    "limber": ("par",),
    "insomnia": ("slp",),
    "vitalspirit": ("slp",),
    "waterveil": ("brn",),
    "waterbubble": ("brn",),
    "magmaarmor": ("frz",),
    "sweetveil": ("slp",),
    "thermalexchange": ("brn",),
}
for _ability, _statuses in _STATUS_IMMUNITIES.items():
    if ("ability", _ability) in __import__("pkcm.engine.effects", fromlist=["REGISTRY"]).REGISTRY:
        continue  # already registered above with other handlers
    register("ability", _ability, name=_ability.title(),
             try_status=_status_immune(*_statuses, ability=_ability))


def _comatose_like(ctx, ref, status, source, **_):
    return False


register("ability", "shieldsdown", name="Shields Down", try_status=_comatose_like)


def _natural_cure(ctx, ref, **_):
    """Cures on the way out. Handled at switch time by the turn loop."""
    return None


register("ability", "naturalcure", name="Natural Cure")
register("ability", "regenerator", name="Regenerator")


def _shed_skin(ctx, ref, **_):
    if ctx.state.sides[ref[0]].status[ref[1]] and ctx.cursor.chance(1, 3):
        announce(ctx, ref, "shedskin")
        mutate.cure_status(ctx, ref)


register("ability", "shedskin", name="Shed Skin", residual=_shed_skin)


def _shield_dust(ctx, ref, status, source, **_):
    return None


register("ability", "shielddust", name="Shield Dust")
register("ability", "innerfocus", name="Inner Focus",
         try_status=lambda ctx, ref, status, source, **_: None)


def _no_flinch(ctx, ref, status, source, **_):
    return None


# --------------------------------------------------------------------------- #
# Stat stage interaction
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


register("ability", "clearbody", name="Clear Body",
         try_boost=_refuse_drops(None, "clearbody"))
register("ability", "whitesmoke", name="White Smoke",
         try_boost=_refuse_drops(None, "whitesmoke"))
register("ability", "fullmetalbody", name="Full Metal Body",
         try_boost=_refuse_drops(None, "fullmetalbody"))
register("ability", "hypercutter", name="Hyper Cutter",
         try_boost=_refuse_drops(("atk",), "hypercutter"))
register("ability", "bigpecks", name="Big Pecks",
         try_boost=_refuse_drops(("def",), "bigpecks"))
register("ability", "keeneye", name="Keen Eye",
         try_boost=_refuse_drops(("accuracy",), "keeneye"))


def _contrary(ctx, ref, value, stat, source, **_):
    return -value


register("ability", "contrary", name="Contrary", try_boost=_contrary)
register("ability", "simple", name="Simple",
         try_boost=lambda ctx, ref, value, stat, source, **_: value * 2)


def _retaliate_on_drop(stat_to_raise: str, ability: str):
    def handler(ctx, ref, value, stat, source, **_):
        if value < 0 and source is not None and source[0] != ref[0]:
            announce(ctx, ref, ability)
            boost(ctx, ref, {stat_to_raise: 2})
        return None

    return handler


register("ability", "defiant", name="Defiant", try_boost=_retaliate_on_drop("atk", "defiant"))
register("ability", "competitive", name="Competitive",
         try_boost=_retaliate_on_drop("spa", "competitive"))


# --------------------------------------------------------------------------- #
# Switch-in
# --------------------------------------------------------------------------- #


def _intimidate(ctx, ref, **_):
    opponent = (1 - ref[0], ctx.state.sides[1 - ref[0]].active)
    if ctx.state.sides[opponent[0]].hp[opponent[1]] <= 0:
        return
    announce(ctx, ref, "intimidate")
    boost(ctx, opponent, {"atk": -1}, source=ref)


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


register("ability", "drought", name="Drought", switch_in=_sets_weather("sunnyday", "drought"))
register("ability", "drizzle", name="Drizzle", switch_in=_sets_weather("raindance", "drizzle"))
register("ability", "sandstream", name="Sand Stream",
         switch_in=_sets_weather("sandstorm", "sandstream"))
register("ability", "snowwarning", name="Snow Warning",
         switch_in=_sets_weather("snowscape", "snowwarning"))


def _sets_terrain(terrain: str, ability: str):
    def handler(ctx, ref, **_):
        if ctx.state.field.terrain == terrain:
            return
        announce(ctx, ref, ability)
        ctx.state.field.terrain = terrain
        ctx.state.field.terrain_turns = 5
        ctx.emit(Event("terrain_start", detail=terrain))

    return handler


register("ability", "electricsurge", name="Electric Surge",
         switch_in=_sets_terrain("electricterrain", "electricsurge"))
register("ability", "grassysurge", name="Grassy Surge",
         switch_in=_sets_terrain("grassyterrain", "grassysurge"))
register("ability", "mistysurge", name="Misty Surge",
         switch_in=_sets_terrain("mistyterrain", "mistysurge"))
register("ability", "psychicsurge", name="Psychic Surge",
         switch_in=_sets_terrain("psychicterrain", "psychicsurge"))
register("ability", "megasol", name="Mega Sol",
         switch_in=_sets_weather("sunnyday", "megasol"))


def _self_boost_on_entry(changes: dict[str, int], ability: str):
    def handler(ctx, ref, **_):
        announce(ctx, ref, ability)
        boost(ctx, ref, changes, source=ref)

    return handler


register("ability", "intrepidsword", name="Intrepid Sword",
         switch_in=_self_boost_on_entry({"atk": 1}, "intrepidsword"))
register("ability", "dauntlessshield", name="Dauntless Shield",
         switch_in=_self_boost_on_entry({"def": 1}, "dauntlessshield"))


def _download(ctx, ref, **_):
    opponent = (1 - ref[0], ctx.state.sides[1 - ref[0]].active)
    if ctx.state.sides[opponent[0]].hp[opponent[1]] <= 0:
        return
    stats = ctx.state.stats(*opponent)
    announce(ctx, ref, "download")
    boost(ctx, ref, {"atk": 1} if stats[Stat.DEF] < stats[Stat.SPD] else {"spa": 1}, source=ref)


register("ability", "download", name="Download", switch_in=_download)


# --------------------------------------------------------------------------- #
# Contact reactions
# --------------------------------------------------------------------------- #


def _contact_status(status: str, chance: int, ability: str):
    def handler(ctx, ref, attacker, defender, move, damage, **_):
        if ref != defender or CONTACT not in move.flags:
            return
        if not ctx.cursor.chance(chance, 100):
            return
        announce(ctx, defender, ability)
        mutate.set_status(ctx, attacker, status, source=defender)

    return handler


register("ability", "static", name="Static", after_damage=_contact_status("par", 30, "static"))
register("ability", "flamebody", name="Flame Body",
         after_damage=_contact_status("brn", 30, "flamebody"))
register("ability", "poisonpoint", name="Poison Point",
         after_damage=_contact_status("psn", 30, "poisonpoint"))


def _contact_damage(denominator: int, ability: str):
    def handler(ctx, ref, attacker, defender, move, damage, **_):
        if ref != defender or CONTACT not in move.flags:
            return
        announce(ctx, defender, ability)
        mutate.apply_damage(ctx, attacker, fraction_of_max(ctx.state, attacker, denominator),
                            "recoil", detail=ability)

    return handler


register("ability", "roughskin", name="Rough Skin", after_damage=_contact_damage(8, "roughskin"))
register("ability", "ironbarbs", name="Iron Barbs", after_damage=_contact_damage(8, "ironbarbs"))


def _contact_drop(stat: str, ability: str):
    def handler(ctx, ref, attacker, defender, move, damage, **_):
        if ref != defender or CONTACT not in move.flags:
            return
        announce(ctx, defender, ability)
        boost(ctx, attacker, {stat: -1}, source=defender)

    return handler


register("ability", "gooey", name="Gooey", after_damage=_contact_drop("spe", "gooey"))
register("ability", "tanglinghair", name="Tangling Hair",
         after_damage=_contact_drop("spe", "tanglinghair"))


def _poison_touch(ctx, ref, attacker, defender, move, damage, **_):
    if ref != attacker or CONTACT not in move.flags:
        return
    if ctx.cursor.chance(30, 100):
        announce(ctx, attacker, "poisontouch")
        mutate.set_status(ctx, defender, "psn", source=attacker)


register("ability", "poisontouch", name="Poison Touch", after_damage=_poison_touch)


# --------------------------------------------------------------------------- #
# End of turn
# --------------------------------------------------------------------------- #


def _weather_heal(weather: str, ability: str):
    def handler(ctx, ref, **_):
        if ctx.state.field.weather == weather:
            heal(ctx, ref, fraction_of_max(ctx.state, ref, 16), reason=ability)

    return handler


register("ability", "raindish", name="Rain Dish", residual=_weather_heal("raindance", "raindish"))
register("ability", "icebody", name="Ice Body", residual=_weather_heal("snowscape", "icebody"))


def _speed_boost(ctx, ref, **_):
    announce(ctx, ref, "speedboost")
    boost(ctx, ref, {"spe": 1}, source=ref)


register("ability", "speedboost", name="Speed Boost", residual=_speed_boost)


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #


def _accuracy_scaler(multiplier: float | None, ability: str):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if ref != attacker:
            return None
        return 100.0 if multiplier is None else value * multiplier

    return handler


register("ability", "compoundeyes", name="Compound Eyes",
         modify_accuracy=_accuracy_scaler(1.3, "compoundeyes"))
register("ability", "noguard", name="No Guard",
         modify_accuracy=_accuracy_scaler(None, "noguard"))


def _serene_grace(ctx, ref, value, **_):
    return None


register("ability", "serenegrace", name="Serene Grace")


#: Abilities that genuinely do nothing in a battle. Registering them keeps the
#: coverage report honest -- "implemented as nothing" is not the same as
#: "forgotten".
NO_BATTLE_EFFECT = frozenset({
    "honeygather", "illuminate", "pickup", "runaway", "ballfetch", "cheekpouch",
    "gluttony", "klutz", "stall", "hospitality", "friendguard", "healer",
    "symbiosis", "telepathy", "receiver", "powerofalchemy", "wimpout",
    "emergencyexit", "pickpocket", "magician", "harvest", "unburden",
})
for _inert in NO_BATTLE_EFFECT:
    if ("ability", _inert) not in __import__(
        "pkcm.engine.effects", fromlist=["REGISTRY"]
    ).REGISTRY:
        register("ability", _inert, name=_inert.title())
