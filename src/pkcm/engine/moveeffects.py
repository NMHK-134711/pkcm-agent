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


def _opponent(ctx: Context, ref: Ref) -> Ref:
    other = 1 - ref[0]
    return (other, ctx.state.sides[other].active)


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
    if not side.is_fainted(side.active):
        heal(ctx, (player, side.active), amount, reason="wish")


register("side", "wish", name="Wish")


@special("roost")
def _roost(ctx, user, target, move) -> bool:
    """The heal is declarative; losing Flying for the turn is not."""
    if "flying" in ctx.state.types(*user):
        mutate.add_volatile(ctx, user, "roost")
    return True


def _roost_types(ctx, ref, value, **_):
    return tuple(t for t in value if t != "flying") or ("normal",)


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
SPECIAL_MOVES["uproar"] = _apply_volatile("uproar", to_target=False, turns=3)
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
        ref = (player, side.active)
        if side.is_fainted(side.active) or "perishsong" in _volatiles(ctx, ref):
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
    for player in (0, 1):
        side = ctx.state.sides[player]
        ref = (player, side.active)
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


SPECIAL_MOVES["gravity"] = _room("gravity")
SPECIAL_MOVES["magicroom"] = _room("magicroom")
SPECIAL_MOVES["wonderroom"] = _room("wonderroom")

register("room", "gravity", name="Gravity")
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
register("side", "wideguard", name="Wide Guard")


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
    last = ctx.state.sides[1 - user[0]].volatiles[
        ctx.state.sides[1 - user[0]].active].get("lastmove")
    if last is None:
        return _fail(ctx, user, "nothing to copy")
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


def _call_move(ctx: Context, user: Ref, target: Ref, move_id: str) -> bool:
    from pkcm.engine.moves import use_move

    called = ctx.state.config.dex.moves.get(move_id)
    if called is None:
        return False
    ctx.emit(Event("called_move", side=user[0], slot=user[1], move=move_id))
    use_move(ctx, user, target, called)
    return True


# --------------------------------------------------------------------------- #
# Only meaningful with an ally on the field
# --------------------------------------------------------------------------- #

ALLY_ONLY = frozenset({
    "helpinghand", "followme", "ragepowder", "allyswitch", "afteryou", "quash",
    "magneticflux", "instruct", "aromatherapy", "coaching", "decorate",
})


def _needs_an_ally(ctx, user, target, move) -> bool:
    return _fail(ctx, user, "there is no ally in a single battle")


for _move_id in ALLY_ONLY:
    SPECIAL_MOVES[_move_id] = _needs_an_ally


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


def _imprison_blocks_shared_moves(ctx, ref, move, **_):
    """Cannot use a move the imprisoning Pokemon also knows."""
    opponent = _opponent(ctx, ref)
    if "imprison" not in _volatiles(ctx, opponent):
        return None
    if move.id in {m.id for m in ctx.state.moves(*opponent)}:
        ctx.emit(Event("cant_move", side=ref[0], slot=ref[1], detail="imprison"))
        return False
    return None


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


def _heal_block_stops_healing(ctx, ref, value, **_):
    return 0


register("volatile", "healblock", name="Heal Block",
         residual=_tick_down("healblock"))


def _uproar_prevents_sleep(ctx, ref, status, source, **_):
    """Nobody sleeps while an Uproar is going on -- either side."""
    for player in (0, 1):
        side = ctx.state.sides[player]
        if side.hp and "uproar" in side.volatiles[side.active]:
            if status == "slp":
                ctx.emit(Event("status_immune", side=ref[0], slot=ref[1], detail="uproar"))
                return False
    return None


register("volatile", "uproar", name="Uproar", residual=_tick_down("uproar"))


def _electrify_retypes(ctx, ref, active, attacker, defender, **_):
    if "electrify" in _volatiles(ctx, attacker):
        active.type = "electric"
        mutate.remove_volatile(ctx, attacker, "electrify", quiet=True)


register("volatile", "electrify", name="Electrify", modify_move=_electrify_retypes)
register("volatile", "smackdown", name="Smack Down")
