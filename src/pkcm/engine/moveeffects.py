"""The moves whose effect the data does not describe at all.

Showdown keeps these in handler functions rather than in fields, so nothing
survives the trip through the client JSON: Haze is an empty shell, Trick is an
empty shell, Rest is an empty shell. Ninety-odd moves that a competitive player
would call basic.

Each entry here is ``(ctx, user, target, move) -> bool``, returning whether it
did anything -- a move that changes nothing has failed, and the log should say
so. The declarative parts of a move still run; this is what runs alongside.

Moves that only matter with an ally on the field (Helping Hand, Follow Me,
After You, Ally Switch, Quash, Magnetic Flux, Instruct) are listed in
``ALLY_ONLY``. In singles they correctly fail, which is not the same as being
unimplemented, and the coverage report should not confuse the two.
"""

from __future__ import annotations

from typing import Callable

from pkcm.data.dex import Stat
from pkcm.engine import mutate
from pkcm.engine.effects import Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.moves import X1_5, chain_modify
from pkcm.engine.mutate import boost, current_hp, fraction_of_max, heal, max_hp
from pkcm.engine.state import BOOST_INDEX, BOOST_STATS

Handler = Callable[[Context, Ref, Ref, object], bool]

SPECIAL_MOVES: dict[str, Handler] = {}


def special(*move_ids: str):
    def register_handler(handler: Handler) -> Handler:
        for move_id in move_ids:
            SPECIAL_MOVES[move_id] = handler
        return handler

    return register_handler


def _fail(ctx: Context, user: Ref, reason: str) -> bool:
    ctx.emit(Event("move_failed", side=user[0], slot=user[1], detail=reason))
    return False


def _volatiles(ctx: Context, ref: Ref) -> dict:
    return ctx.state.sides[ref[0]].volatiles[ref[1]]


# --------------------------------------------------------------------------- #
# Healing
# --------------------------------------------------------------------------- #

#: Moonlight, Morning Sun and Synthesis heal by the weather.
WEATHER_HEAL = {"sunnyday": (2, 3), "raindance": (1, 4), "sandstorm": (1, 4),
                "snowscape": (1, 4)}


@special("moonlight", "morningsun", "synthesis")
def _weather_heal(ctx, user, target, move) -> bool:
    numerator, denominator = WEATHER_HEAL.get(ctx.state.field.weather, (1, 2))
    amount = max(1, max_hp(ctx.state, user) * numerator // denominator)
    return bool(heal(ctx, user, amount, reason=move.id))


@special("rest")
def _rest(ctx, user, target, move) -> bool:
    if current_hp(ctx.state, user) == max_hp(ctx.state, user):
        return _fail(ctx, user, "already at full health")
    ctx.state.sides[user[0]].status[user[1]] = None
    mutate.set_status(ctx, user, "slp")
    ctx.state.sides[user[0]].status_data[user[1]]["turns"] = 3
    heal(ctx, user, max_hp(ctx.state, user), reason="rest")
    return True


@special("healpulse")
def _heal_pulse(ctx, user, target, move) -> bool:
    return bool(heal(ctx, target, max_hp(ctx.state, target) // 2, reason=move.id))


@special("painsplit")
def _pain_split(ctx, user, target, move) -> bool:
    """Both end on the average of the two current HP values."""
    average = max(1, (current_hp(ctx.state, target) + current_hp(ctx.state, user)) // 2)
    for ref in (user, target):
        side = ctx.state.sides[ref[0]]
        side.hp[ref[1]] = min(average, max_hp(ctx.state, ref))
        ctx.emit(Event("set_hp", side=ref[0], slot=ref[1], hp=side.hp[ref[1]],
                       max_hp=max_hp(ctx.state, ref), detail=move.id))
    return True


@special("strengthsap")
def _strength_sap(ctx, user, target, move) -> bool:
    """Heals by the target's Attack, then drops it."""
    if ctx.state.sides[target[0]].boost(target[1], "atk") == -6:
        return _fail(ctx, user, "Attack is already as low as it goes")
    attack = mutate.effective_stat(ctx, target, Stat.ATK)
    heal(ctx, user, attack, reason=move.id)
    boost(ctx, target, {"atk": -1}, source=user)
    return True


@special("wish")
def _wish(ctx, user, target, move) -> bool:
    conditions = ctx.state.sides[user[0]].conditions
    if "wish" in conditions:
        return _fail(ctx, user, "a wish is already pending")
    conditions["wish"] = max_hp(ctx.state, user) // 2
    ctx.emit(Event("side_condition", side=user[0], detail="wish", amount=1))
    return True


def resolve_wish(ctx: Context, player: int) -> None:
    """Fires at the end of the turn after the one it was made on."""
    conditions = ctx.state.sides[player].conditions
    amount = conditions.pop("wish", None)
    if amount is None:
        return
    side = ctx.state.sides[player]
    for slot in side.active_slots():
        heal(ctx, (player, slot), amount, reason="wish")


register("side", "wish", name="Wish")


@special("roost")
def _roost(ctx, user, target, move) -> bool:
    """The heal is declarative; losing Flying for the turn is not."""
    if "flying" in ctx.state.types(*user):
        mutate.add_volatile(ctx, user, "roost")
    return True


register("volatile", "roost", name="Roosting")


# --------------------------------------------------------------------------- #
# Stat stages
# --------------------------------------------------------------------------- #


@special("haze")
def _haze(ctx, user, target, move) -> bool:
    for side in ctx.state.sides:
        for slot in range(len(side.hp)):
            side.boosts[slot] = [0] * len(BOOST_STATS)
    ctx.emit(Event("boosts_cleared", detail=move.id))
    return True


@special("bellydrum")
def _belly_drum(ctx, user, target, move) -> bool:
    cost = max_hp(ctx.state, user) // 2
    if current_hp(ctx.state, user) <= cost:
        return _fail(ctx, user, "not enough HP")
    if ctx.state.sides[user[0]].boost(user[1], "atk") >= 6:
        return _fail(ctx, user, "Attack is already maxed")
    mutate.apply_damage(ctx, user, cost, "damage", detail=move.id)
    ctx.state.sides[user[0]].boosts[user[1]][BOOST_INDEX["atk"]] = 6
    ctx.emit(Event("boost", side=user[0], slot=user[1], detail="atk", amount=6, hp=6))
    return True


@special("psychup")
def _psych_up(ctx, user, target, move) -> bool:
    ctx.state.sides[user[0]].boosts[user[1]] = list(
        ctx.state.sides[target[0]].boosts[target[1]]
    )
    ctx.emit(Event("boosts_copied", side=user[0], slot=user[1]))
    return True


@special("topsyturvy")
def _topsy_turvy(ctx, user, target, move) -> bool:
    boosts = ctx.state.sides[target[0]].boosts[target[1]]
    if not any(boosts):
        return _fail(ctx, user, "nothing to invert")
    ctx.state.sides[target[0]].boosts[target[1]] = [-stage for stage in boosts]
    ctx.emit(Event("boosts_inverted", side=target[0], slot=target[1]))
    return True


def _swap_boosts(stats: tuple[str, ...], label: str):
    def handler(ctx, user, target, move) -> bool:
        mine = ctx.state.sides[user[0]].boosts[user[1]]
        theirs = ctx.state.sides[target[0]].boosts[target[1]]
        for name in stats:
            index = BOOST_INDEX[name]
            mine[index], theirs[index] = theirs[index], mine[index]
        ctx.emit(Event("boosts_swapped", side=user[0], slot=user[1], detail=label))
        return True

    return handler


SPECIAL_MOVES["powerswap"] = _swap_boosts(("atk", "spa"), "offence")
SPECIAL_MOVES["guardswap"] = _swap_boosts(("def", "spd"), "defence")
SPECIAL_MOVES["speedswap"] = _swap_boosts(("spe",), "speed")


def _split_stats(stats: tuple[Stat, ...], label: str):
    """Power Split and Guard Split: both sides end on the average."""

    def handler(ctx, user, target, move) -> bool:
        for stat in stats:
            average = (mutate.raw_stat(ctx.state, user, stat)
                       + mutate.raw_stat(ctx.state, target, stat)) // 2
            for ref in (user, target):
                stats_now = list(ctx.state.stats(*ref))
                stats_now[stat] = average
                ctx.state.set_override(ref[0], ref[1], "stats", tuple(stats_now))
        ctx.emit(Event("stats_split", side=user[0], slot=user[1], detail=label))
        return True

    return handler


SPECIAL_MOVES["powersplit"] = _split_stats((Stat.ATK, Stat.SPA), "offence")
SPECIAL_MOVES["guardsplit"] = _split_stats((Stat.DEF, Stat.SPD), "defence")


def _swap_own_stats(first: Stat, second: Stat, volatile: str):
    def handler(ctx, user, target, move) -> bool:
        stats = list(ctx.state.stats(*user))
        stats[first], stats[second] = stats[second], stats[first]
        ctx.state.set_override(user[0], user[1], "stats", tuple(stats))
        if volatile in _volatiles(ctx, user):
            mutate.remove_volatile(ctx, user, volatile)
        else:
            mutate.add_volatile(ctx, user, volatile)
        return True

    return handler


SPECIAL_MOVES["powertrick"] = _swap_own_stats(Stat.ATK, Stat.DEF, "powertrick")
SPECIAL_MOVES["powershift"] = _swap_own_stats(Stat.DEF, Stat.SPD, "powershift")
register("volatile", "powertrick", name="Power Trick")
register("volatile", "powershift", name="Power Shift")


@special("acupressure")
def _acupressure(ctx, user, target, move) -> bool:
    side = ctx.state.sides[target[0]]
    raisable = [name for name in BOOST_STATS if side.boost(target[1], name) < 6]
    if not raisable:
        return _fail(ctx, user, "nothing left to raise")
    return bool(boost(ctx, target, {ctx.cursor.choice(raisable): 2}, source=user))


@special("noretreat")
def _no_retreat(ctx, user, target, move) -> bool:
    if "noretreat" in _volatiles(ctx, user):
        return _fail(ctx, user, "already committed")
    mutate.add_volatile(ctx, user, "noretreat")
    mutate.add_volatile(ctx, user, "trapped")
    boost(ctx, user, {"atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1}, source=user)
    return True


register("volatile", "noretreat", name="No Retreat")


@special("stockpile")
def _stockpile(ctx, user, target, move) -> bool:
    data = _volatiles(ctx, user).get("stockpile")
    if data and data["layers"] >= 3:
        return _fail(ctx, user, "already at three")
    if data is None:
        mutate.add_volatile(ctx, user, "stockpile", layers=1)
    else:
        data["layers"] += 1
    boost(ctx, user, {"def": 1, "spd": 1}, source=user)
    return True


def _spend_stockpile(ctx: Context, user: Ref) -> int:
    data = _volatiles(ctx, user).pop("stockpile", None)
    if data is None:
        return 0
    layers = data["layers"]
    boost(ctx, user, {"def": -layers, "spd": -layers}, source=user)
    return layers


@special("swallow")
def _swallow(ctx, user, target, move) -> bool:
    layers = _spend_stockpile(ctx, user)
    if not layers:
        return _fail(ctx, user, "nothing stockpiled")
    fraction = {1: 4, 2: 2, 3: 1}[layers]
    return bool(heal(ctx, user, max_hp(ctx.state, user) // fraction, reason=move.id))


register("volatile", "stockpile", name="Stockpile")


@special("focusenergy", "dragoncheer")
def _focus_energy(ctx, user, target, move) -> bool:
    who = user if move.id == "focusenergy" else target
    if "focusenergy" in _volatiles(ctx, who):
        return _fail(ctx, user, "already focused")
    return mutate.add_volatile(ctx, who, "focusenergy")


register("volatile", "focusenergy", name="Focus Energy",
         modify_crit_ratio=lambda ctx, ref, value, attacker, defender, move, **_:
             value + 2 if ref == attacker else None)


# --------------------------------------------------------------------------- #
# Conditions that stop the target doing something
# --------------------------------------------------------------------------- #


def _apply_volatile(name: str, to_target: bool = True, **data):
    def handler(ctx, user, target, move) -> bool:
        who = target if to_target else user
        if name in _volatiles(ctx, who):
            return _fail(ctx, user, f"already affected by {name}")
        payload = dict(data)
        if "turns" in payload and callable(payload["turns"]):
            payload["turns"] = payload["turns"](ctx)
        return mutate.add_volatile(ctx, who, name, source=user, **payload)

    return handler


SPECIAL_MOVES["taunt"] = _apply_volatile("taunt", turns=4)
SPECIAL_MOVES["torment"] = _apply_volatile("torment")
SPECIAL_MOVES["imprison"] = _apply_volatile("imprison", to_target=False)
SPECIAL_MOVES["destinybond"] = _apply_volatile("destinybond", to_target=False)
SPECIAL_MOVES["magnetrise"] = _apply_volatile("magnetrise", to_target=False, turns=5)
SPECIAL_MOVES["minimize"] = _apply_volatile("minimize", to_target=False)
SPECIAL_MOVES["aquaring"] = _apply_volatile("aquaring", to_target=False)
SPECIAL_MOVES["lockon"] = _apply_volatile("lockon", to_target=False, turns=2)
SPECIAL_MOVES["smackdown"] = _apply_volatile("smackdown")
SPECIAL_MOVES["electrify"] = _apply_volatile("electrify")
SPECIAL_MOVES["saltcure"] = _apply_volatile("saltcure")
SPECIAL_MOVES["syrupbomb"] = _apply_volatile("syrupbomb", turns=4)
SPECIAL_MOVES["gastroacid"] = _apply_volatile("abilitysuppressed")
SPECIAL_MOVES["psychicnoise"] = _apply_volatile("healblock", turns=3)
SPECIAL_MOVES["block"] = _apply_volatile("trapped")
SPECIAL_MOVES["meanlook"] = _apply_volatile("trapped")
SPECIAL_MOVES["fairylock"] = _apply_volatile("trapped")


def _taunt_blocks_status(ctx, ref, move, **_):
    if move.category == "Status":
        ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="taunt"))
        return False
    return None


def _tick_down(name: str):
    def handler(ctx, ref, **_):
        data = mutate.volatile(ctx.state, ref, name)
        if data is None or "turns" not in data:
            return
        data["turns"] -= 1
        if data["turns"] <= 0:
            mutate.remove_volatile(ctx, ref, name)

    return handler


register("volatile", "taunt", name="Taunt",
         try_move=_taunt_blocks_status, residual=_tick_down("taunt"))


def _torment_blocks_repeats(ctx, ref, move, **_):
    if _volatiles(ctx, ref).get("lastmove") == move.id:
        ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="torment"))
        return False
    return None


register("volatile", "torment", name="Torment", try_move=_torment_blocks_repeats)


def _encore_locks(ctx, ref, **_):
    data = mutate.volatile(ctx.state, ref, "encore")
    if data is None:
        return
    data["turns"] -= 1
    if data["turns"] <= 0:
        mutate.remove_volatile(ctx, ref, "encore")


@special("encore")
def _encore(ctx, user, target, move) -> bool:
    last = _volatiles(ctx, target).get("lastmove")
    if last is None:
        return _fail(ctx, user, "it has not moved yet")
    for index, carried in enumerate(ctx.state.moves(*target)):
        if carried.id == last:
            return mutate.add_volatile(ctx, target, "encore", move=index, turns=4)
    return _fail(ctx, user, "it is not carrying that move")


register("volatile", "encore", name="Encore", residual=_encore_locks)


@special("disable")
def _disable(ctx, user, target, move) -> bool:
    last = _volatiles(ctx, target).get("lastmove")
    if last is None:
        return _fail(ctx, user, "it has not moved yet")
    for index, carried in enumerate(ctx.state.moves(*target)):
        if carried.id == last:
            return mutate.add_volatile(ctx, target, "disabled", move=index, turns=5)
    return _fail(ctx, user, "it is not carrying that move")


@special("spite")
def _spite(ctx, user, target, move) -> bool:
    last = _volatiles(ctx, target).get("lastmove")
    side = ctx.state.sides[target[0]]
    for index, carried in enumerate(ctx.state.moves(*target)):
        if carried.id == last and side.pp[target[1]][index] > 0:
            lost = min(4, side.pp[target[1]][index])
            side.pp[target[1]][index] -= lost
            ctx.emit(Event("pp_lost", side=target[0], slot=target[1],
                           move=carried.id, amount=lost))
            return True
    return _fail(ctx, user, "nothing to sap")


@special("yawn")
def _yawn(ctx, user, target, move) -> bool:
    if ctx.state.sides[target[0]].status[target[1]] is not None:
        return _fail(ctx, user, "it already has a status")
    return mutate.add_volatile(ctx, target, "yawn", turns=2, source=user)


def _yawn_resolves(ctx, ref, **_):
    data = mutate.volatile(ctx.state, ref, "yawn")
    if data is None:
        return
    data["turns"] -= 1
    if data["turns"] <= 0:
        mutate.remove_volatile(ctx, ref, "yawn")
        mutate.set_status(ctx, ref, "slp")


register("volatile", "yawn", name="Yawn", residual=_yawn_resolves)


@special("curse")
def _curse(ctx, user, target, move) -> bool:
    """Two different moves wearing one name, split by the user's type."""
    if "ghost" not in ctx.state.types(*user):
        return bool(boost(ctx, user, {"spe": -1, "atk": 1, "def": 1}, source=user))
    if "curse" in _volatiles(ctx, target):
        return _fail(ctx, user, "already cursed")
    mutate.apply_damage(ctx, user, max_hp(ctx.state, user) // 2, "damage", detail=move.id)
    return mutate.add_volatile(ctx, target, "curse", source=user)


register("volatile", "curse", name="Curse",
         residual=lambda ctx, ref, **_: mutate.apply_damage(
             ctx, ref, fraction_of_max(ctx.state, ref, 4), "status_damage", detail="curse"))


@special("perishsong")
def _perish_song(ctx, user, target, move) -> bool:
    applied = False
    for player in (0, 1):
        side = ctx.state.sides[player]
        for slot in side.active_slots():
            ref = (player, slot)
            if "perishsong" in _volatiles(ctx, ref):
                continue
            mutate.add_volatile(ctx, ref, "perishsong", turns=4)
            applied = True
    return applied or _fail(ctx, user, "everyone is already counting")


def _perish_count(ctx, ref, **_):
    data = mutate.volatile(ctx.state, ref, "perishsong")
    if data is None:
        return
    data["turns"] -= 1
    ctx.emit(Event("perish_count", side=ref[0], slot=ref[1], amount=data["turns"]))
    if data["turns"] <= 0:
        mutate.apply_damage(ctx, ref, current_hp(ctx.state, ref), "status_damage",
                            detail="perishsong")


register("volatile", "perishsong", name="Perish Song", residual=_perish_count)


register("volatile", "aquaring", name="Aqua Ring",
         residual=lambda ctx, ref, **_: heal(
             ctx, ref, fraction_of_max(ctx.state, ref, 16), reason="aquaring"))
register("volatile", "magnetrise", name="Magnet Rise", residual=_tick_down("magnetrise"))
register("volatile", "syrupbomb", name="Syrup Bomb",
         residual=lambda ctx, ref, **_: (
             boost(ctx, ref, {"spe": -1}, source=ref), _tick_down("syrupbomb")(ctx, ref=ref))[1])
register("volatile", "abilitysuppressed", name="Ability suppressed")


def _salt_cure_residual(ctx, ref, **_):
    types = ctx.state.types(*ref)
    denominator = 4 if {"water", "steel"} & set(types) else 8
    mutate.apply_damage(ctx, ref, fraction_of_max(ctx.state, ref, denominator),
                        "status_damage", detail="saltcure")


register("volatile", "saltcure", name="Salt Cure", residual=_salt_cure_residual)


@special("endure")
def _endure(ctx, user, target, move) -> bool:
    return mutate.add_volatile(ctx, user, "endure")


def _endure_survives(ctx, ref, value, attacker, defender, move, **_):
    if ref != defender:
        return None
    remaining = current_hp(ctx.state, defender)
    if value >= remaining:
        ctx.emit(Event("endured", side=defender[0], slot=defender[1]))
        return remaining - 1
    return None


register("volatile", "endure", name="Endure", priority=30, modify_damage=_endure_survives)


# --------------------------------------------------------------------------- #
# Protection variants
# --------------------------------------------------------------------------- #

#: Protect with a sting: the flavour differs, the blocking does not.
PROTECT_VARIANTS = {
    "kingsshield": ("kingsshield", {"atk": -1}),
    "banefulbunker": ("banefulbunker", None),
    "spikyshield": ("spikyshield", None),
    "silktrap": ("silktrap", {"spe": -1}),
    "obstruct": ("obstruct", {"def": -2}),
    "burningbulwark": ("burningbulwark", None),
}


def _protect_variant(name: str):
    def handler(ctx, user, target, move) -> bool:
        from pkcm.engine.moves import _apply_protect

        if not _apply_protect(ctx, user, move):
            return False
        mutate.add_volatile(ctx, user, name)
        return True

    return handler


for _move_id, (_volatile, _drop) in PROTECT_VARIANTS.items():
    SPECIAL_MOVES[_move_id] = _protect_variant(_volatile)


def _punishing_shield(volatile: str, drop, damage_fraction: int | None, status: str | None):
    def handler(ctx, ref, attacker, defender, move, **_):
        if ref != defender or "protect" not in _volatiles(ctx, defender):
            return None
        if "contact" not in move.flags:
            return None
        if drop:
            boost(ctx, attacker, drop, source=defender)
        if damage_fraction:
            mutate.apply_damage(ctx, attacker,
                                fraction_of_max(ctx.state, attacker, damage_fraction),
                                "recoil", detail=volatile)
        if status:
            mutate.set_status(ctx, attacker, status, source=defender)
        return None

    return handler


register("volatile", "kingsshield", name="King's Shield",
         after_damage=_punishing_shield("kingsshield", {"atk": -1}, None, None))
register("volatile", "silktrap", name="Silk Trap",
         after_damage=_punishing_shield("silktrap", {"spe": -1}, None, None))
register("volatile", "obstruct", name="Obstruct",
         after_damage=_punishing_shield("obstruct", {"def": -2}, None, None))
register("volatile", "spikyshield", name="Spiky Shield",
         after_damage=_punishing_shield("spikyshield", None, 8, None))
register("volatile", "banefulbunker", name="Baneful Bunker",
         after_damage=_punishing_shield("banefulbunker", None, None, "psn"))
register("volatile", "burningbulwark", name="Burning Bulwark",
         after_damage=_punishing_shield("burningbulwark", None, None, "brn"))


# --------------------------------------------------------------------------- #
# Types and abilities
# --------------------------------------------------------------------------- #


def _set_types(types: tuple[str, ...]):
    def handler(ctx, user, target, move) -> bool:
        if ctx.state.types(*target) == types:
            return _fail(ctx, user, "no change")
        ctx.state.set_override(target[0], target[1], "types", types)
        ctx.emit(Event("type_change", side=target[0], slot=target[1], detail=types[0]))
        return True

    return handler


SPECIAL_MOVES["soak"] = _set_types(("water",))
SPECIAL_MOVES["magicpowder"] = _set_types(("psychic",))


def _add_type(extra: str):
    def handler(ctx, user, target, move) -> bool:
        types = ctx.state.types(*target)
        if extra in types:
            return _fail(ctx, user, "it already has that type")
        ctx.state.set_override(target[0], target[1], "types", types + (extra,))
        ctx.emit(Event("type_added", side=target[0], slot=target[1], detail=extra))
        return True

    return handler


SPECIAL_MOVES["forestscurse"] = _add_type("grass")
SPECIAL_MOVES["trickortreat"] = _add_type("ghost")


@special("reflecttype")
def _reflect_type(ctx, user, target, move) -> bool:
    ctx.state.set_override(user[0], user[1], "types", ctx.state.types(*target))
    ctx.emit(Event("type_change", side=user[0], slot=user[1],
                   detail=ctx.state.types(*target)[0]))
    return True


@special("conversion")
def _conversion(ctx, user, target, move) -> bool:
    moves = ctx.state.moves(*user)
    if not moves:
        return _fail(ctx, user, "no moves to copy")
    ctx.state.set_override(user[0], user[1], "types", (moves[0].type,))
    ctx.emit(Event("type_change", side=user[0], slot=user[1], detail=moves[0].type))
    return True


#: Abilities that refuse to be traded, copied or overwritten.
UNTOUCHABLE_ABILITIES = frozenset({
    "trace", "imposter", "forecast", "flowergift", "zenmode", "illusion",
    "stancechange", "battlebond", "powerconstruct", "schooling", "shieldsdown",
    "disguise", "rkssystem", "commander", "zerotohero", "hungerswitch",
    "multitype", "comatose", "neutralizinggas", "asoneglastrier", "asonespectrier",
})


def _set_ability(chooser, to_target: bool):
    def handler(ctx, user, target, move) -> bool:
        who = target if to_target else user
        current = ctx.state.ability_id(*who)
        new = chooser(ctx, user, target)
        if new is None or current in UNTOUCHABLE_ABILITIES or new == current:
            return _fail(ctx, user, "that ability cannot be changed")
        ctx.state.set_override(who[0], who[1], "ability", new)
        ctx.emit(Event("ability_change", side=who[0], slot=who[1], detail=new))
        return True

    return handler


SPECIAL_MOVES["worryseed"] = _set_ability(lambda ctx, u, t: "insomnia", True)
SPECIAL_MOVES["simplebeam"] = _set_ability(lambda ctx, u, t: "simple", True)
SPECIAL_MOVES["entrainment"] = _set_ability(
    lambda ctx, u, t: ctx.state.ability_id(*u), True)
SPECIAL_MOVES["roleplay"] = _set_ability(
    lambda ctx, u, t: ctx.state.ability_id(*t), False)


@special("skillswap")
def _skill_swap(ctx, user, target, move) -> bool:
    mine, theirs = ctx.state.ability_id(*user), ctx.state.ability_id(*target)
    if {mine, theirs} & UNTOUCHABLE_ABILITIES or mine == theirs:
        return _fail(ctx, user, "those abilities will not swap")
    ctx.state.set_override(user[0], user[1], "ability", theirs)
    ctx.state.set_override(target[0], target[1], "ability", mine)
    ctx.emit(Event("ability_swapped", side=user[0], slot=user[1]))
    return True


@special("transform")
def _transform(ctx, user, target, move) -> bool:
    from pkcm.engine.abilities import _imposter

    if _volatiles(ctx, user).get("transformed"):
        return _fail(ctx, user, "already transformed")
    _imposter(ctx, ref=user)
    return True


# --------------------------------------------------------------------------- #
# Items
# --------------------------------------------------------------------------- #


@special("trick", "switcheroo")
def _trick(ctx, user, target, move) -> bool:
    mine, theirs = ctx.state.item_id(*user), ctx.state.item_id(*target)
    if mine is None and theirs is None:
        return _fail(ctx, user, "neither is holding anything")
    if ctx.state.ability_id(*target) == "stickyhold":
        return _fail(ctx, user, "Sticky Hold")
    ctx.state.set_override(user[0], user[1], "item", theirs, permanent=True)
    ctx.state.set_override(target[0], target[1], "item", mine, permanent=True)
    ctx.emit(Event("items_swapped", side=user[0], slot=user[1]))
    return True


def _holds_removable(ctx, target) -> str | None:
    """The target's item, if it can be taken off it.

    Champions on Sticky Hold: 지니고 있는 도구를 상대에게 빼앗기거나
    잃어버리지 않는다. It guards this whole family, and it is checked here
    rather than in five places.
    """
    if ctx.state.ability_id(*target) == "stickyhold":
        return None
    return ctx.state.item_id(*target)


@special("knockoff")
def _knock_off(ctx, user, target, move) -> bool:
    """Champions: 상대가 도구를 지니고 있으면 위력이 1.5배가 된다. 상대의
    도구를 없앤다.

    The power half is in ``moves.VARIABLE_POWER``, because it has to be known
    before the damage is rolled and this runs after it.
    """
    item = _holds_removable(ctx, target)
    if item is None:
        return False
    mutate.consume_item(ctx, target, move.id)
    return True


@special("thief", "covet")
def _steal_item(ctx, user, target, move) -> bool:
    """Champions, for both: 자신이 도구를 지니고 있지 않은 경우 상대의 도구를
    빼앗는다.

    Both conditions matter and both are easy to drop. Holding anything at all
    means no steal -- not "steal if it is better" -- and a target holding
    nothing is not a failure worth announcing, the move simply did its damage.
    """
    if ctx.state.item_id(*user) is not None:
        return False
    item = _holds_removable(ctx, target)
    if item is None:
        return False
    # Off them and onto us, rather than consumed: it is the same item.
    ctx.state.set_override(target[0], target[1], "item", None, permanent=True)
    ctx.state.set_override(user[0], user[1], "item", item, permanent=True)
    ctx.emit(Event("item_stolen", side=user[0], slot=user[1], detail=item))
    return True


@special("bugbite", "pluck")
def _eat_their_berry(ctx, user, target, move) -> bool:
    """Champions: 상대가 나무열매를 지니고 있으면 그 나무열매를 대신 먹고
    자신이 효과를 받는다.

    The effect lands on *us* -- which is why berries needed their trigger and
    their effect separated for Cud Chew, and why this can reuse it: eating a
    Sitrus Berry off somebody else heals us whatever our own HP is doing.
    """
    from pkcm.engine.items import eat_berry

    item = _holds_removable(ctx, target)
    if item is None or not ctx.state.config.dex.items[item].raw.get("isBerry"):
        return False
    mutate.consume_item(ctx, target, move.id)
    return eat_berry(ctx, user, item)


@special("icespinner")
def _ice_spinner(ctx, user, target, move) -> bool:
    """Champions: 필드를 해제한다."""
    if ctx.state.field.terrain is None:
        return False
    ended = ctx.state.field.terrain
    ctx.state.field.terrain = None
    ctx.state.field.terrain_turns = 0
    ctx.emit(Event("terrain_end", detail=ended))
    return True


@special("ragingbull")
def _raging_bull(ctx, user, target, move) -> bool:
    """Champions: 상대 필드의 리플렉터, 빛의장막, 오로라베일 상태를 해제하고
    공격한다.

    Screens only. The hazards stay -- this is not a Defog either.
    """
    conditions = ctx.state.sides[target[0]].conditions
    broke = False
    for name in SCREENS:
        if conditions.pop(name, None) is not None:
            broke = True
            ctx.emit(Event("side_condition_end", side=target[0], detail=name))
    return broke


@special("fellstinger")
def _fell_stinger(ctx, user, target, move) -> bool:
    """Champions: 이 기술로 상대를 쓰러뜨리면 공격이 3단계 올라간다.

    Runs from ``_after_effects`` once the damage has landed, so "did it knock
    the target out" is a question the state can already answer.
    """
    if not ctx.state.sides[target[0]].is_fainted(target[1]):
        return False
    return bool(boost(ctx, user, {"atk": 3}, source=user))


@special("corrosivegas")
def _corrosive_gas(ctx, user, target, move) -> bool:
    if ctx.state.item_id(*target) is None:
        return _fail(ctx, user, "it is not holding anything")
    if ctx.state.ability_id(*target) == "stickyhold":
        return _fail(ctx, user, "Sticky Hold")
    mutate.consume_item(ctx, target, move.id)
    return True


@special("recycle")
def _recycle(ctx, user, target, move) -> bool:
    override = ctx.state.override(*user)
    if override.get("item") is not None or "item" not in override:
        return _fail(ctx, user, "nothing to get back")
    original = ctx.state.pokemon(*user).item
    if original is None:
        return _fail(ctx, user, "nothing to get back")
    ctx.state.set_override(user[0], user[1], "item", original, permanent=True)
    ctx.emit(Event("item_restored", side=user[0], slot=user[1], detail=original))
    return True


@special("stuffcheeks")
def _stuff_cheeks(ctx, user, target, move) -> bool:
    item = ctx.state.item_id(*user)
    if item is None or not ctx.state.config.dex.items[item].raw.get("isBerry"):
        return _fail(ctx, user, "no berry to eat")
    mutate.consume_item(ctx, user, move.id)
    boost(ctx, user, {"def": 2}, source=user)
    return True


@special("teatime")
def _teatime(ctx, user, target, move) -> bool:
    eaten = False
    for ref in ctx.state.everyone():
        item = ctx.state.item_id(*ref)
        if item and ctx.state.config.dex.items[item].raw.get("isBerry"):
            mutate.consume_item(ctx, ref, move.id)
            eaten = True
    return eaten or _fail(ctx, user, "nobody had a berry")


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #


def _room(name: str):
    def handler(ctx, user, target, move) -> bool:
        rooms = ctx.state.field.rooms
        if name in rooms:
            del rooms[name]
            ctx.emit(Event("room_end", detail=name))
        else:
            rooms[name] = 5
            ctx.emit(Event("room_start", detail=name))
        return True

    return handler


@special("steelroller")
def _steel_roller(ctx, user, target, move) -> bool:
    """Tears up the terrain. Refusing to run without one is a precondition,
    checked in ``moves.MOVE_PRECONDITIONS`` before any PP is spent."""
    terrain = ctx.state.field.terrain
    if terrain is None:
        return False
    ctx.state.field.terrain = None
    ctx.state.field.terrain_turns = 0
    ctx.emit(Event("terrain_end", detail=terrain))
    return True


SPECIAL_MOVES["gravity"] = _room("gravity")
SPECIAL_MOVES["magicroom"] = _room("magicroom")
SPECIAL_MOVES["wonderroom"] = _room("wonderroom")

#: Gravity steadies everyone's aim by a third (5/3 in Showdown).
GRAVITY_ACCURACY = (5, 3)

#: Moves that leave the ground, and so cannot be used while Gravity holds.
GROUNDED_MOVES = frozenset({
    "fly", "bounce", "highjumpkick", "jumpkick", "splash", "magnetrise",
    "telekinesis", "flyingpress", "skydrop",
})


def _gravity_steadies_aim(ctx, ref, value, attacker, defender, move, **_):
    if ref != attacker:
        return None
    return value * GRAVITY_ACCURACY[0] / GRAVITY_ACCURACY[1]


def _gravity_grounds_the_move(ctx, ref, move, **_):
    if move.id not in GROUNDED_MOVES:
        return None
    ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="gravity"))
    return False


# Gravity also pulls everything down to earth -- that half is in
# ``conditions.is_grounded``, which is the one place that answers the question.
register("room", "gravity", name="Gravity",
         modify_accuracy=_gravity_steadies_aim,
         try_move=_gravity_grounds_the_move)

# Magic Room and Wonder Room are read where the thing they suppress is read:
# ``effects.Context.item_of`` for the items, ``mutate.raw_stat`` for the swap.
# Both were registered and consulted by nobody until now.
register("room", "magicroom", name="Magic Room")
register("room", "wonderroom", name="Wonder Room")

HAZARDS = ("spikes", "toxicspikes", "stealthrock", "stickyweb")
SCREENS = ("reflect", "lightscreen", "auroraveil")


@special("defog")
def _defog(ctx, user, target, move) -> bool:
    cleared = False
    boost(ctx, target, {"evasion": -1}, source=user)
    for player, names in ((user[0], HAZARDS), (target[0], HAZARDS + SCREENS)):
        conditions = ctx.state.sides[player].conditions
        for name in names:
            if conditions.pop(name, None) is not None:
                cleared = True
                ctx.emit(Event("side_condition_end", side=player, detail=name))
    return True


@special("tidyup")
def _tidy_up(ctx, user, target, move) -> bool:
    for player in (0, 1):
        conditions = ctx.state.sides[player].conditions
        for name in HAZARDS:
            if conditions.pop(name, None) is not None:
                ctx.emit(Event("side_condition_end", side=player, detail=name))
        for slot in range(len(ctx.state.sides[player].hp)):
            ctx.state.sides[player].volatiles[slot].pop("substitute", None)
    boost(ctx, user, {"atk": 1, "spe": 1}, source=user)
    return True


def _side_condition(name: str, turns: int, on_own_side: bool = True):
    def handler(ctx, user, target, move) -> bool:
        player = user[0] if on_own_side else target[0]
        conditions = ctx.state.sides[player].conditions
        if name in conditions:
            return _fail(ctx, user, f"{name} is already up")
        conditions[name] = turns
        ctx.emit(Event("side_condition", side=player, detail=name, amount=turns))
        return True

    return handler


SPECIAL_MOVES["safeguard"] = _side_condition("safeguard", 5)
SPECIAL_MOVES["quickguard"] = _side_condition("quickguard", 1)
SPECIAL_MOVES["wideguard"] = _side_condition("wideguard", 1)

register("side", "safeguard", name="Safeguard",
         try_status=lambda ctx, ref, status, source, **_:
             False if source is not None and source[0] != ref[0] else None)
register("side", "quickguard", name="Quick Guard",
         try_hit=lambda ctx, ref, attacker, defender, move, **_:
             False if ref == defender and attacker[0] != defender[0]
             and move.priority > 0 else None)
def _wide_guard(ctx, ref, attacker, defender, move, **_):
    """Blocks a spread move for the whole side. The defining doubles wall.

    Registered with no handler until doubles existed, which made it a move that
    quietly did nothing -- the one failure this project keeps finding.
    """
    if ref != defender or attacker[0] == defender[0]:
        return None
    if move.target not in ("allAdjacent", "allAdjacentFoes"):
        return None
    if getattr(move, "breaks_protect", False):
        return None
    ctx.emit(Event("protected", side=ref[0], slot=ref[1], move=move.id,
                   detail="wideguard"))
    return False


register("side", "wideguard", name="Wide Guard", try_hit=_wide_guard)


@special("healbell")
def _heal_bell(ctx, user, target, move) -> bool:
    side = ctx.state.sides[user[0]]
    cured = False
    for slot in range(len(side.hp)):
        if side.status[slot] is not None:
            side.status[slot] = None
            side.status_data[slot] = {}
            cured = True
    if cured:
        ctx.emit(Event("team_cured", side=user[0], detail=move.id))
    return cured or _fail(ctx, user, "nobody is affected")


@special("sparklingaria")
def _sparkling_aria(ctx, user, target, move) -> bool:
    """A damaging move whose point is that it puts the burn out."""
    if ctx.state.sides[target[0]].status[target[1]] != "brn":
        return False
    mutate.cure_status(ctx, target)
    return True


# --------------------------------------------------------------------------- #
# Copying and calling other moves
# --------------------------------------------------------------------------- #


@special("copycat")
def _copycat(ctx, user, target, move) -> bool:
    """Repeat the last move used. Not every move can be repeated.

    ``failcopycat`` is in the data and was going unread, which mattered because
    Copycat carries the flag itself: two holders facing each other copied each
    other's Copycat until the stack ran out. Singles can arrange that as easily
    as doubles -- it just needs both sides to have brought one.
    """
    foes = ctx.state.foes(user)
    if not foes:
        return _fail(ctx, user, "nothing to copy")
    last = _volatiles(ctx, foes[0]).get("lastmove")
    if last is None:
        return _fail(ctx, user, "nothing to copy")
    copied = ctx.state.config.dex.moves.get(last)
    if copied is None or "failcopycat" in copied.flags:
        return _fail(ctx, user, f"{last} cannot be copied")
    return _call_move(ctx, user, target, last)


@special("sleeptalk")
def _sleep_talk(ctx, user, target, move) -> bool:
    if ctx.state.sides[user[0]].status[user[1]] != "slp":
        return _fail(ctx, user, "only works while asleep")
    options = [m.id for m in ctx.state.moves(*user)
               if m.id != "sleeptalk" and "charge" not in m.flags]
    if not options:
        return _fail(ctx, user, "nothing to talk about")
    return _call_move(ctx, user, target, ctx.cursor.choice(options))


#: How deep one move may call another. Sleep Talk into Copycat into something
#: else is three, and there is no legitimate chain longer than a handful.
#:
#: This is a backstop, not the mechanism. Each calling move refuses the moves it
#: is not allowed to call (``failcopycat`` and friends), and that is what makes
#: the behaviour right. This is what makes a *missed* guard fail loudly instead
#: of exhausting the stack -- which is how the Copycat cycle showed up, in a
#: rollout, as a RecursionError with no clue in it.
MAX_CALL_DEPTH = 4


def _call_move(ctx: Context, user: Ref, target: Ref, move_id: str) -> bool:
    """Use another move as though it had been chosen.

    ``target`` is *not* passed on. Sleep Talk and Copycat both target ``self``,
    so inheriting their target aimed the called move at its own user -- Snorlax
    Body Slamming itself for 84. The called move resolves its own target, which
    is the only reading that makes sense: it is a Body Slam, and Body Slam has
    an opinion about who it hits.
    """
    from pkcm.engine.moves import use_move

    called = ctx.state.config.dex.moves.get(move_id)
    if called is None:
        return False
    depth = getattr(ctx, "call_depth", 0)
    if depth >= MAX_CALL_DEPTH:
        ctx.emit(Event("move_failed", side=user[0], move=move_id,
                       detail="called too deep"))
        return False
    ctx.emit(Event("called_move", side=user[0], slot=user[1], move=move_id))
    ctx.call_depth = depth + 1
    try:
        use_move(ctx, user, called)
    finally:
        ctx.call_depth = depth
    return True


# --------------------------------------------------------------------------- #
# Only meaningful with an ally on the field
# --------------------------------------------------------------------------- #

#: Moves that do nothing without a partner standing next to you. They are
#: implemented -- they fail *because there is no ally*, which is a different
#: thing from being unimplemented, and in doubles they stop failing.
#:
#: Coaching, Decorate and Helping Hand are not here: the first two are plain
#: ``boosts`` moves the declarative executor already runs, and all three fail
#: on their own when target resolution finds no partner. Instruct is not here
#: either -- it works perfectly well on an opponent.
ALLY_ONLY = frozenset({"followme", "ragepowder", "allyswitch", "afteryou", "quash"})


def _no_ally(ctx: Context, user: Ref) -> bool:
    return _fail(ctx, user, "there is no ally in a single battle")


# --------------------------------------------------------------------------- #
# Taking the hit for your partner
# --------------------------------------------------------------------------- #


@special("followme", "ragepowder")
def _draw_fire(ctx, user, target, move) -> bool:
    """Every single-target move from the other side comes here instead."""
    if ctx.state.ally(user) is None:
        return _no_ally(ctx, user)
    return mutate.add_volatile(ctx, user, move.id, source=user)


def _pull_moves_to_me(name: str, powder: bool):
    def handler(ctx, ref, value, attacker, move, **_):
        if ref == value:
            return None
        if powder and not _powder_reaches(ctx, attacker):
            return None
        ctx.emit(Event("redirected", side=ref[0], slot=ref[1], detail=name))
        return ref

    return handler


def _powder_reaches(ctx: Context, ref: Ref) -> bool:
    """Grass types, Overcoat and Safety Goggles all ignore a powder."""
    if "grass" in ctx.state.types(*ref):
        return False
    if ctx.ability_of(ref) == "overcoat":
        return False
    return ctx.state.item_id(*ref) != "safetygoggles"


register("volatile", "followme", name="Follow Me",
         redirect_target=_pull_moves_to_me("followme", powder=False))
register("volatile", "ragepowder", name="Rage Powder",
         redirect_target=_pull_moves_to_me("ragepowder", powder=True))


@special("helpinghand")
def _helping_hand(ctx, user, target, move) -> bool:
    """Half again on the partner's move, and it stacks with a second one."""
    data = mutate.volatile(ctx.state, target, "helpinghand")
    if data is not None:
        data["stacks"] = data.get("stacks", 1) + 1
        return True
    return mutate.add_volatile(ctx, target, "helpinghand", source=user, stacks=1)


def _helping_hand_power(ctx, ref, value, attacker, defender, move, **_):
    if ref != attacker:
        return None
    data = mutate.volatile(ctx.state, attacker, "helpinghand")
    if data is None:
        return None
    for _ in range(data.get("stacks", 1)):
        value = chain_modify(value, X1_5)
    return value


register("volatile", "helpinghand", name="Helping Hand",
         modify_base_power=_helping_hand_power)


# --------------------------------------------------------------------------- #
# Damaging moves that also lay something down
#
# Showdown keeps these in ``secondary.onHit`` JavaScript, so what we import
# carries an empty ``secondary: {}`` and the declarative path lays nothing. The
# move still hits for its damage, which is why nothing ever failed: docs are
# explicit that **a damaging move missing its conditional half just looks like a
# weak move**. Found by playing one, not by a test.
# --------------------------------------------------------------------------- #

#: move id -> the hazard it leaves on the target's side.
HAZARD_ON_HIT = {"ceaselessedge": "spikes", "stoneaxe": "stealthrock"}


@special("rapidspin", "mortalspin")
def _spin_free(ctx, user, target, move) -> bool:
    """Champions, for both:

        자신의 바인드, 씨뿌리기 상태와 같은 편 필드의 압정뿌리기, 독압정,
        끈적끈적네트, 스텔스록 상태를 해제한다.

    Nothing in the move data says any of this -- the speed boost rides in as a
    secondary and the poison as a status, so both of those already worked and
    the moves looked like they were doing their job. The clearing is script in
    the reference implementation and there was no handler here, so a spin was
    damage and nothing else.

    Runs from ``_after_effects``, which for a damaging move happens once the
    damage has landed, so the damage is untouched.

    **Our own side only.** Defog takes the opponent's hazards too; a spin does
    not, and clearing theirs would quietly turn every spinner into a Defog.
    """
    side = ctx.state.sides[user[0]]
    for name in HAZARDS:
        if side.conditions.pop(name, None) is not None:
            ctx.emit(Event("side_condition_end", side=user[0], detail=name))

    mutate.remove_volatile(ctx, user, "leechseed")
    # Binding moves only. ``trapped`` also comes from Shadow Tag and from Mean
    # Look, and a spin frees from neither -- so it goes only alongside the
    # ``partiallytrapped`` that a binding move sets with it.
    if mutate.volatile(ctx.state, user, "partiallytrapped") is not None:
        mutate.remove_volatile(ctx, user, "partiallytrapped")
        mutate.remove_volatile(ctx, user, "trapped", quiet=True)
    return True


@special("ceaselessedge", "stoneaxe")
def _hazard_on_hit(ctx, user, target, move) -> bool:
    from pkcm.engine.conditions import add_side_condition

    return add_side_condition(ctx, target[0], HAZARD_ON_HIT[move.id], user)


@special("burnup")
def _burn_up(ctx, user, target, move) -> bool:
    """The user's Fire type burns away, leaving whatever else it had.

    ``MOVE_PRECONDITIONS`` refuses the move outright unless the user is Fire, so
    by the time this runs there is always a type to remove.
    """
    remaining = tuple(name for name in ctx.state.types(*user) if name != "Fire")
    ctx.state.set_override(user[0], user[1], "types", remaining)
    ctx.emit(Event("type_change", side=user[0], slot=user[1],
                   detail="/".join(remaining) or "typeless"))
    return True


# --------------------------------------------------------------------------- #
# Rearranging the turn
#
# After You and Quash reach into the queue the turn loop is working through.
# It holds (player, position) pairs, so moving one is a list operation -- which
# is exactly why the queue lives on the state rather than in a local.
# --------------------------------------------------------------------------- #


def _queued_position(ctx: Context, ref: Ref) -> tuple[int, int] | None:
    position = ctx.state.sides[ref[0]].position_of(ref[1])
    if position is None:
        return None
    actor = (ref[0], position)
    return actor if actor in ctx.state.turn_queue else None


@special("afteryou")
def _after_you(ctx, user, target, move) -> bool:
    """The target moves next, whatever its Speed said."""
    if ctx.state.ally(user) is None:
        return _no_ally(ctx, user)
    actor = _queued_position(ctx, target)
    if actor is None:
        return _fail(ctx, user, "it has already moved")
    ctx.state.turn_queue.remove(actor)
    ctx.state.turn_queue.insert(0, actor)
    ctx.emit(Event("move_order", side=target[0], slot=target[1], detail="afteryou"))
    return True


@special("quash")
def _quash(ctx, user, target, move) -> bool:
    """The target moves last instead."""
    if ctx.state.ally(user) is None:
        return _no_ally(ctx, user)
    actor = _queued_position(ctx, target)
    if actor is None:
        return _fail(ctx, user, "it has already moved")
    ctx.state.turn_queue.remove(actor)
    ctx.state.turn_queue.append(actor)
    ctx.emit(Event("move_order", side=target[0], slot=target[1], detail="quash"))
    return True


@special("allyswitch")
def _ally_switch(ctx, user, target, move) -> bool:
    """Trade places with your partner. Nothing else about either changes."""
    side = ctx.state.sides[user[0]]
    partner = ctx.state.ally(user)
    if partner is None:
        return _no_ally(ctx, user)
    here, there = side.position_of(user[1]), side.position_of(partner[1])
    if here is None or there is None:
        return _fail(ctx, user, "nobody to swap with")
    side.active[here], side.active[there] = side.active[there], side.active[here]
    ctx.emit(Event("ally_switch", side=user[0], slot=user[1]))
    return True


@special("magneticflux")
def _magnetic_flux(ctx, user, target, move) -> bool:
    """Defences up, but only for the ones running Plus or Minus."""
    boosted = False
    for ref in ctx.state.allies_and_self(user):
        if ctx.ability_of(ref) in ("plus", "minus"):
            boosted |= bool(mutate.boost(ctx, ref, {"def": 1, "spd": 1}, source=user))
    return boosted or _fail(ctx, user, "nobody here runs Plus or Minus")


#: Moves Instruct refuses to repeat -- the ones whose state it cannot restore.
INSTRUCT_REFUSES = frozenset({"instruct", "struggle", "transform", "mimic", "sketch",
                              "kingsshield", "beakblast", "focuspunch", "shelltrap"})


@special("instruct")
def _instruct(ctx, user, target, move) -> bool:
    """Make the target use its last move again, right now.

    Works on an opponent as readily as on a partner, which is why this is not
    an ally-only move -- singles gets it too.
    """
    last = _volatiles(ctx, target).get("lastmove")
    if last is None or last in INSTRUCT_REFUSES:
        return _fail(ctx, user, "there is nothing to repeat")
    repeated = ctx.state.config.dex.moves[last]
    if "charge" in repeated.flags or "recharge" in repeated.flags:
        return _fail(ctx, user, "that move cannot be repeated")

    ctx.emit(Event("instructed", side=target[0], slot=target[1], move=last))
    from pkcm.engine.moves import use_move

    foes = ctx.state.foes(target)
    use_move(ctx, target, repeated, defender=foes[0] if foes else None)
    return True


# --------------------------------------------------------------------------- #
# Variable power that needs more than the move
# --------------------------------------------------------------------------- #


def beat_up_hits(ctx: Context, user: Ref) -> list[int]:
    """One hit per healthy team mate, each at that Pokemon's base Attack."""
    side = ctx.state.sides[user[0]]
    powers = []
    for slot in range(len(side.hp)):
        if side.hp[slot] <= 0 or (slot != user[1] and side.status[slot] is not None):
            continue
        base = ctx.state.pokemon(user[0], slot).species.base_stats[Stat.ATK]
        powers.append(5 + base // 10)
    return powers


def fling_power(ctx: Context, user: Ref) -> int | None:
    item_id = ctx.state.item_id(*user)
    if item_id is None:
        return None
    fling = ctx.state.config.dex.items[item_id].raw.get("fling")
    return fling.get("basePower") if fling else None


def spit_up_power(ctx: Context, user: Ref) -> int:
    data = mutate.volatile(ctx.state, user, "stockpile")
    return 100 * data["layers"] if data else 0


# --------------------------------------------------------------------------- #
# The volatiles that need someone to read them
#
# A volatile nobody consults is a move that quietly does nothing -- the failure
# this project keeps guarding against. Each of these has its reader here.
# --------------------------------------------------------------------------- #


def _destiny_bond_takes_you_with_it(ctx, ref, source, **_):
    """If the bonded Pokemon faints to a move, whoever did it faints too."""
    if source is None or source == ref or source[0] == ref[0]:
        return
    ctx.emit(Event("destiny_bond", side=ref[0], slot=ref[1]))
    mutate.apply_damage(ctx, source, current_hp(ctx.state, source), "damage",
                        detail="destinybond")


register("volatile", "destinybond", name="Destiny Bond",
         faint=_destiny_bond_takes_you_with_it)


# Imprison and Uproar are read engine-side, not here. Both ask about a volatile
# that sits on the *other* Pokemon -- Imprison on the opponent, Uproar on
# whoever is making the noise -- and effect gathering only ever reaches the
# Pokemon the hook is running for. ``state.imprisoned_moves`` and
# ``state.uproar_in_progress`` answer them, called from ``use_move`` and
# ``set_status`` respectively. Registered here so the volatile still has a name.
register("volatile", "imprison", name="Imprison")


def _lock_on_never_misses(ctx, ref, value, attacker, defender, move, **_):
    if ref == attacker and "lockon" in _volatiles(ctx, attacker):
        return 100.0
    return None


register("volatile", "lockon", name="Lock-On",
         modify_accuracy=_lock_on_never_misses, residual=_tick_down("lockon"))


def _minimize_doubles_stomping_moves(ctx, ref, value, attacker, defender, move, **_):
    if ref == defender and move.id in MINIMIZE_PUNISHERS:
        return value * 2
    return None


#: Moves that flatten a minimized target for double damage.
MINIMIZE_PUNISHERS = frozenset({
    "stomp", "bodyslam", "flyingpress", "dragonrush", "heatcrash", "heavyslam",
    "maliciousmoonsault",
})

register("volatile", "minimize", name="Minimize",
         modify_base_power=_minimize_doubles_stomping_moves)


# Heal Block's refusal lives in ``mutate.heal``, for the same reason: the
# healing it has to stop can come from anywhere, not just from a hook it sees.
register("volatile", "healblock", name="Heal Block",
         residual=_tick_down("healblock"))


UPROAR_TURNS = 3


@special("uproar")
def _uproar(ctx: Context, user: Ref, target: Ref, move) -> bool:
    """Three turns of noise. Wakes the field and keeps it awake.

    Refreshing rather than failing on the second turn is the point -- the user
    is locked into the move (``state.legal_actions``), so a plain
    ``_apply_volatile`` would report a failure every turn after the first.
    """
    for player in (0, 1):
        side = ctx.state.sides[player]
        for slot in side.active_slots():
            if side.status[slot] == "slp":
                mutate.cure_status(ctx, (player, slot))

    volatiles = _volatiles(ctx, user)
    if "uproar" in volatiles:
        return True  # already going; the residual tick owns the countdown
    return mutate.add_volatile(ctx, user, "uproar", source=user, turns=UPROAR_TURNS)


register("volatile", "uproar", name="Uproar", residual=_tick_down("uproar"))


def _electrify_retypes(ctx, ref, active, attacker, defender, **_):
    if "electrify" in _volatiles(ctx, attacker):
        active.type = "electric"
        mutate.remove_volatile(ctx, attacker, "electrify", quiet=True)


register("volatile", "electrify", name="Electrify", modify_move=_electrify_retypes)
register("volatile", "smackdown", name="Smack Down")
