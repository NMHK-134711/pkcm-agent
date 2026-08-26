"""Status conditions, volatiles, weather, terrain, screens and hazards.

Every one of these is an ``Effect`` registered on the hook system, which is what
lets them stay this small. Burn is four lines because "halve physical Attack"
and "lose 1/16 at end of turn" are each one hook, rather than two edits inside
the damage formula and the turn loop.

Damage-related hooks all receive ``attacker``, ``defender`` and ``move``, plus
``ref`` for whose effect is running -- comparing ``ref`` against those tells a
handler which side of the interaction it is on.
"""

from __future__ import annotations

from pkcm.data.dex import Stat
from pkcm.engine import mutate
from pkcm.engine.effects import Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.mutate import apply_damage, fraction_of_max, heal

CONFUSION_SELF_HIT_POWER = 40
CONFUSION_CHANCE = (1, 3)

# Champions retunes all three of these away from the mainline series
# (mods/champions/conditions.ts). Ported from that file, not from the games.
#: Full paralysis is 1/8 here, not the series' 1/4.
PARALYSIS_CHANCE = (1, 8)
#: Sleep runs 2 or 3 turns -- ``sample([2, 3, 3])``, so 3 turns two times in three.
SLEEP_DURATIONS = (2, 3, 3)
#: Freeze has a hard 3-turn timer *and* a 1/4 thaw roll each turn.
FREEZE_DURATION = 3
THAW_CHANCE = (1, 4)

SCREEN_MULTIPLIER = 0.5
WEATHER_BOOST, WEATHER_DAMP = 1.5, 0.5
TERRAIN_BOOST = 1.3


def is_grounded(state, ref: Ref, ctx: Context | None = None) -> bool:
    """Flying types and Levitate float; hazards, terrain and Ground moves skip them.

    Pass ``ctx`` when a move is resolving, so a suppressed Levitate (Mold
    Breaker) correctly stops floating.
    """
    side_index, slot = ref
    # Gravity pulls everything down: Flying types, Levitate, Air Balloon and a
    # Magnet Rise that is already up.
    if "gravity" in state.field.rooms:
        return True
    volatiles = state.sides[side_index].volatiles
    if slot < len(volatiles):
        # Smack Down pins a Pokemon to the ground whatever it is; Magnet Rise
        # lifts one that would otherwise be standing on it.
        if "smackdown" in volatiles[slot] or "ingrain" in volatiles[slot]:
            return True
        if "magnetrise" in volatiles[slot]:
            return False
    if "flying" in state.types(side_index, slot):
        return False
    ability = ctx.ability_of(ref) if ctx is not None else state.ability_id(side_index, slot)
    if ability == "levitate":
        return False
    if state.item_id(side_index, slot) == "airballoon":
        return False
    return True


# --------------------------------------------------------------------------- #
# Major status conditions
# --------------------------------------------------------------------------- #


def _burn_halves_attack(ctx, ref, value, stat, **_):
    return value // 2 if stat is Stat.ATK else None


def _burn_residual(ctx, ref, **_):
    apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, 16), "status_damage", detail="brn")


register("status", "brn", name="Burn",
         modify_stat=_burn_halves_attack, residual=_burn_residual)


def _paralysis_halves_speed(ctx, ref, value, stat, **_):
    return value // 2 if stat is Stat.SPE else None


def _paralysis_may_stop(ctx, ref, move, **_):
    if ctx.cursor.chance(*PARALYSIS_CHANCE):
        ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="par"))
        return False
    return None


register("status", "par", name="Paralysis",
         modify_stat=_paralysis_halves_speed, try_move=_paralysis_may_stop)


def _poison_residual(ctx, ref, **_):
    apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, 8), "status_damage", detail="psn")


register("status", "psn", name="Poison", residual=_poison_residual)


def _toxic_residual(ctx, ref, **_):
    data = ctx.state.sides[ref[0]].status_data[ref[1]]
    data["stage"] = min(15, data.get("stage", 0) + 1)
    amount = fraction_of_max(ctx.state, ref, 16) * data["stage"]
    apply_damage(ctx, ref, amount, "status_damage", detail="tox")


register("status", "tox", name="Bad Poison", residual=_toxic_residual)


def _sleep_blocks_move(ctx, ref, move, **_):
    data = ctx.state.sides[ref[0]].status_data[ref[1]]
    data["turns"] = data.get("turns", SLEEP_DURATIONS[-1]) - 1
    if data["turns"] <= 0:
        mutate.cure_status(ctx, ref)
        return None
    ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="slp"))
    return False


register("status", "slp", name="Sleep", try_move=_sleep_blocks_move)


def _freeze_blocks_move(ctx, ref, move, **_):
    # Moves flagged ``defrost`` thaw the user and go through.
    if "defrost" in move.flags:
        mutate.cure_status(ctx, ref)
        return None
    data = ctx.state.sides[ref[0]].status_data[ref[1]]
    data["turns"] = data.get("turns", FREEZE_DURATION) - 1
    if data["turns"] <= 0 or ctx.cursor.chance(*THAW_CHANCE):
        mutate.cure_status(ctx, ref)
        return None
    ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="frz"))
    return False


register("status", "frz", name="Freeze", try_move=_freeze_blocks_move)


# --------------------------------------------------------------------------- #
# Volatile conditions
# --------------------------------------------------------------------------- #


def _flinch_blocks_move(ctx, ref, move, **_):
    ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="flinch"))
    return False


register("volatile", "flinch", name="Flinch", try_move=_flinch_blocks_move)


def _confusion_may_self_hit(ctx, ref, move, **_):
    data = ctx.state.sides[ref[0]].volatiles[ref[1]]["confusion"]
    data["turns"] = data.get("turns", 1) - 1
    if data["turns"] <= 0:
        mutate.remove_volatile(ctx, ref, "confusion")
        return None

    if not ctx.cursor.chance(*CONFUSION_CHANCE):
        return None

    # A typeless physical hit against the confused Pokemon's own defence.
    attack = mutate.effective_stat(ctx, ref, Stat.ATK)
    defense = mutate.effective_stat(ctx, ref, Stat.DEF)
    damage = ((2 * 50 // 5 + 2) * CONFUSION_SELF_HIT_POWER * attack // defense) // 50 + 2
    damage = damage * ctx.cursor.between(85, 100) // 100
    ctx.emit(Event("confused", side=ref[0], slot=ref[1]))
    apply_damage(ctx, ref, max(1, damage), "damage", detail="confusion")
    return False


register("volatile", "confusion", name="Confusion", try_move=_confusion_may_self_hit)


def _protect_blocks(ctx, ref, attacker, defender, move, **_):
    if ref != defender or attacker == defender:
        return None
    if getattr(move, "breaks_protect", False):
        return None
    if "protect" in move.flags:
        ctx.emit(Event("protected", side=defender[0], slot=defender[1], move=move.id))
        return False
    return None


register("volatile", "protect", name="Protect", try_hit=_protect_blocks)

#: Set when Protect succeeds; each consecutive use is likelier to fail.
register("volatile", "stall", name="Stall")

register("volatile", "trapped", name="Trapped")


def _disabled_expires(ctx, ref, **_):
    data = ctx.state.sides[ref[0]].volatiles[ref[1]].get("disabled")
    if data is None:
        return
    data["turns"] -= 1
    if data["turns"] <= 0:
        mutate.remove_volatile(ctx, ref, "disabled")


#: One move is unusable for a few turns. ``state.legal_actions`` reads it.
register("volatile", "disabled", name="Disabled", residual=_disabled_expires)


def _attract_may_stop(ctx, ref, move, **_):
    if ctx.cursor.chance(1, 2):
        ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="attract"))
        return False
    return None


register("volatile", "attract", name="Attract", try_move=_attract_may_stop)

#: Electromorphosis and Charge: the next Electric move is doubled.
register("volatile", "charge", name="Charge",
         modify_base_power=lambda ctx, ref, value, attacker, defender, move, **_:
             value * 2 if ref == attacker and move.type == "electric" else None)
register("volatile", "substitute", name="Substitute")


def _leech_seed_residual(ctx, ref, **_):
    opponent = ctx.state.foes(ref)[0] if ctx.state.foes(ref) else ref
    if ctx.state.sides[opponent[0]].hp[opponent[1]] <= 0:
        return
    drained = apply_damage(
        ctx, ref, fraction_of_max(ctx.state, ref, 8), "status_damage", detail="leechseed"
    )
    heal(ctx, opponent, drained, reason="leechseed")


register("volatile", "leechseed", name="Leech Seed", residual=_leech_seed_residual)


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #


def _weather_damage_scaler(boosted: str, damped: str):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if move.type == boosted:
            return int(value * WEATHER_BOOST)
        if move.type == damped:
            return int(value * WEATHER_DAMP)
        return None

    return handler


register("weather", "sunnyday", name="Harsh Sunlight",
         modify_damage=_weather_damage_scaler("fire", "water"))
register("weather", "raindance", name="Rain",
         modify_damage=_weather_damage_scaler("water", "fire"))


SANDSTORM_IMMUNE_TYPES = ("rock", "ground", "steel")


def _sandstorm_residual(ctx, ref, **_):
    if set(ctx.state.types(*ref)) & set(SANDSTORM_IMMUNE_TYPES):
        return
    apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, 16), "weather_damage",
                 detail="sandstorm")


def _sandstorm_boosts_rock_spd(ctx, ref, value, stat, **_):
    if stat is Stat.SPD and "rock" in ctx.state.types(*ref):
        return int(value * 1.5)
    return None


register("weather", "sandstorm", name="Sandstorm",
         residual=_sandstorm_residual, modify_boosted_stat=_sandstorm_boosts_rock_spd)


def _snow_boosts_ice_def(ctx, ref, value, stat, **_):
    if stat is Stat.DEF and "ice" in ctx.state.types(*ref):
        return int(value * 1.5)
    return None


register("weather", "snowscape", name="Snow", modify_boosted_stat=_snow_boosts_ice_def)


# --------------------------------------------------------------------------- #
# Terrain -- only affects grounded Pokemon
# --------------------------------------------------------------------------- #


def _terrain_boost(move_type: str):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if move.type == move_type and is_grounded(ctx.state, attacker):
            return int(value * TERRAIN_BOOST)
        return None

    return handler


register("terrain", "electricterrain", name="Electric Terrain",
         modify_damage=_terrain_boost("electric"))
register("terrain", "grassyterrain", name="Grassy Terrain",
         modify_damage=_terrain_boost("grass"))
register("terrain", "psychicterrain", name="Psychic Terrain",
         modify_damage=_terrain_boost("psychic"))


def _misty_damps_dragon(ctx, ref, value, attacker, defender, move, **_):
    if move.type == "dragon" and is_grounded(ctx.state, defender):
        return int(value * 0.5)
    return None


register("terrain", "mistyterrain", name="Misty Terrain",
         modify_damage=_misty_damps_dragon)


register("room", "trickroom", name="Trick Room")


# --------------------------------------------------------------------------- #
# Side conditions
# --------------------------------------------------------------------------- #


def _screen(category: str | None):
    def handler(ctx, ref, value, attacker, defender, move, **_):
        if attacker[0] == defender[0]:
            return None
        if getattr(move, "infiltrates", False):
            return None
        if category is not None and move.category != category:
            return None
        return int(value * SCREEN_MULTIPLIER)

    return handler


register("side", "reflect", name="Reflect", modify_damage=_screen("Physical"))
register("side", "lightscreen", name="Light Screen", modify_damage=_screen("Special"))
register("side", "auroraveil", name="Aurora Veil", modify_damage=_screen(None))


def _tailwind_doubles_speed(ctx, ref, value, stat, **_):
    return value * 2 if stat is Stat.SPE else None


register("side", "tailwind", name="Tailwind", modify_boosted_stat=_tailwind_doubles_speed)

#: Entry hazards carry no hooks -- they fire from ``apply_entry_hazards`` on
#: switch-in, which is a turn-loop event rather than a per-Pokemon one.
for hazard in ("spikes", "toxicspikes", "stealthrock", "stickyweb"):
    register("side", hazard, name=hazard.title())


SPIKES_FRACTION = {1: 8, 2: 6, 3: 4}

#: Screens and Tailwind last a set number of turns and do not stack; hazards
#: stack up to a cap and never expire. Getting this wrong is invisible in tests
#: and obvious in a battle log -- Light Screen was reaching "x2".
SIDE_CONDITION_DURATION = {"reflect": 5, "lightscreen": 5, "auroraveil": 5, "tailwind": 4}
SIDE_CONDITION_LAYERS = {"spikes": 3, "toxicspikes": 2, "stealthrock": 1, "stickyweb": 1}


def apply_entry_hazards(ctx: Context, ref: Ref) -> None:
    """Everything that greets a Pokemon as it lands on the field."""
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    grounded = is_grounded(ctx.state, ref)

    if "stealthrock" in side.conditions:
        chart = ctx.state.config.dex.type_chart
        effectiveness = chart.multiplier("rock", ctx.state.types(side_index, slot))
        amount = max(1, int(mutate.max_hp(ctx.state, ref) * effectiveness / 8))
        apply_damage(ctx, ref, amount, "hazard_damage", detail="stealthrock")
        if side.hp[slot] <= 0:
            return

    if not grounded:
        return

    if "spikes" in side.conditions:
        layers = min(3, side.conditions["spikes"])
        apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, SPIKES_FRACTION[layers]),
                     "hazard_damage", detail="spikes")
        if side.hp[slot] <= 0:
            return

    if "toxicspikes" in side.conditions:
        if "poison" in ctx.state.types(side_index, slot):
            del side.conditions["toxicspikes"]
            ctx.emit(Event("hazard_absorbed", side=side_index, detail="toxicspikes"))
        elif "steel" not in ctx.state.types(side_index, slot):
            status = "tox" if side.conditions["toxicspikes"] >= 2 else "psn"
            mutate.set_status(ctx, ref, status)

    if "stickyweb" in side.conditions:
        mutate.boost(ctx, ref, {"spe": -1})


# --------------------------------------------------------------------------- #
# What is actually wired up
#
# A move setting a condition we never registered a handler for would look
# supported -- the executor happily writes the name into the state -- and then
# do nothing at all. Safeguard was exactly that: `sideCondition: "safeguard"` is
# declarative, so the naive check passed it, and the move silently did nothing.
# `move_support` consults these sets so such a move is named unsupported instead.
# --------------------------------------------------------------------------- #

IMPLEMENTED_STATUSES = frozenset({"brn", "par", "psn", "tox", "slp", "frz"})
IMPLEMENTED_VOLATILES = frozenset({
    "confusion", "flinch", "leechseed", "protect", "substitute", "trapped",
    "partiallytrapped", "lockedmove", "twoturn", "mustrecharge", "invulnerable",
    "disabled", "attract", "charge", "ingrain",
})
IMPLEMENTED_SIDE_CONDITIONS = frozenset({"reflect", "lightscreen", "auroraveil", "tailwind",
                                         "spikes", "toxicspikes", "stealthrock", "stickyweb",
                                         "healingwish"})
IMPLEMENTED_WEATHER = frozenset({"sunnyday", "raindance", "sandstorm", "snowscape"})
IMPLEMENTED_TERRAIN = frozenset({"electricterrain", "grassyterrain", "mistyterrain",
                                 "psychicterrain"})
IMPLEMENTED_ROOMS = frozenset({"trickroom"})
