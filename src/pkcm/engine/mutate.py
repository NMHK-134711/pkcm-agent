"""The primitive state changes, each of which runs its hooks.

Nothing in the engine writes HP, a stat stage, or a status directly. It goes
through here, because every one of those changes is something a mechanic may
want to intercept -- Substitute absorbs damage, Clear Body refuses a drop,
Limber refuses paralysis, Magic Guard refuses everything indirect. Putting the
hook call next to the mutation is what makes those one-liners later instead of
edits scattered through the turn loop.
"""

from __future__ import annotations

from pkcm.data.dex import Stat
from pkcm.engine import effects as fx
from pkcm.engine import events as ev
from pkcm.engine.effects import Context, Ref
from pkcm.engine.events import Event
from pkcm.engine.state import BOOST_INDEX, BOOST_STATS, MAX_BOOST, MIN_BOOST

#: A type cannot be given the status it embodies. This is a rule of the game
#: rather than an ability, so it lives here rather than as a registered effect.
STATUS_TYPE_IMMUNITY = {
    "par": ("electric",),
    "brn": ("fire",),
    "frz": ("ice",),
    "psn": ("poison", "steel"),
    "tox": ("poison", "steel"),
}

#: Stat -> stat stage name. HP has no stage.
STAT_TO_BOOST = {
    Stat.ATK: "atk",
    Stat.DEF: "def",
    Stat.SPA: "spa",
    Stat.SPD: "spd",
    Stat.SPE: "spe",
}


def stage_multiplier(stage: int, accuracy_like: bool = False) -> float:
    """The series' stat stage table, as the ratio it actually is.

    Battle stats step by halves from a base of 2; accuracy and evasion step by
    thirds from a base of 3, which is why they need their own branch.
    """
    base = 3 if accuracy_like else 2
    if stage >= 0:
        return (base + stage) / base
    return base / (base - stage)


def raw_stat(state, ref: Ref, stat: Stat) -> int:
    return state.stats(ref[0], ref[1])[stat]


def effective_stat(ctx: Context, ref: Ref, stat: Stat) -> int:
    """A stat as the damage formula should see it: stages, then hooks."""
    side_index, slot = ref
    value = fx.modify(ctx, "modify_stat", raw_stat(ctx.state, ref, stat), ref, stat=stat)

    boost_name = STAT_TO_BOOST.get(stat)
    if boost_name is not None:
        stage = ctx.state.sides[side_index].boost(slot, boost_name)
        value = int(value * stage_multiplier(stage))

    value = fx.modify(ctx, "modify_boosted_stat", value, ref, stat=stat)
    return max(1, int(value))


def max_hp(state, ref: Ref) -> int:
    return state.pokemon(ref[0], ref[1]).stats[Stat.HP]


def current_hp(state, ref: Ref) -> int:
    return state.sides[ref[0]].hp[ref[1]]


# --------------------------------------------------------------------------- #
# HP
# --------------------------------------------------------------------------- #


def apply_damage(
    ctx: Context,
    ref: Ref,
    amount: int,
    kind: str = "damage",
    **event_fields,
) -> int:
    """Subtract HP, emit the event, check for a faint. Returns damage dealt."""
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0 or amount <= 0:
        return 0

    dealt = min(amount, side.hp[slot])
    side.hp[slot] -= dealt

    ctx.emit(
        Event(
            kind,
            side=side_index,
            slot=slot,
            amount=dealt,
            hp=side.hp[slot],
            max_hp=max_hp(ctx.state, ref),
            **event_fields,
        )
    )
    check_faint(ctx, ref)
    return dealt


def heal(ctx: Context, ref: Ref, amount: int, reason: str | None = None) -> int:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0:
        return 0

    total = max_hp(ctx.state, ref)
    healed = min(max(0, amount), total - side.hp[slot])
    if healed == 0:
        return 0
    side.hp[slot] += healed
    ctx.emit(
        Event("heal", side=side_index, slot=slot, amount=healed,
              hp=side.hp[slot], max_hp=total, detail=reason)
    )
    return healed


def fraction_of_max(state, ref: Ref, denominator: int) -> int:
    """``max_hp / denominator``, never less than 1 -- the series' rounding."""
    return max(1, max_hp(state, ref) // denominator)


def check_faint(ctx: Context, ref: Ref) -> bool:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] > 0:
        return False
    if side.volatiles[slot].get("__fainted__"):
        return True
    side.volatiles[slot]["__fainted__"] = True
    ctx.emit(ev.faint(side_index, slot, ctx.state.species_name(side_index, slot)))
    fx.notify(ctx, "faint", ref)
    return True


# --------------------------------------------------------------------------- #
# Stat stages
# --------------------------------------------------------------------------- #


def boost(ctx: Context, ref: Ref, changes: dict[str, int], source: Ref | None = None) -> dict[str, int]:
    """Apply stat stage changes. Returns what actually landed."""
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0:
        return {}

    applied: dict[str, int] = {}
    for name, stages in changes.items():
        if name not in BOOST_INDEX or stages == 0:
            continue

        stages = fx.modify(ctx, "try_boost", stages, ref, stat=name, source=source)
        if stages == 0:
            continue

        index = BOOST_INDEX[name]
        before = side.boosts[slot][index]
        after = max(MIN_BOOST, min(MAX_BOOST, before + stages))
        if after == before:
            ctx.emit(
                Event("boost_failed", side=side_index, slot=slot, detail=name,
                      amount=stages)
            )
            continue

        side.boosts[slot][index] = after
        applied[name] = after - before
        ctx.emit(
            Event("boost", side=side_index, slot=slot, detail=name,
                  amount=after - before, hp=after)
        )
    return applied


def clear_boosts(ctx: Context, ref: Ref) -> None:
    side_index, slot = ref
    ctx.state.sides[side_index].boosts[slot] = [0] * len(BOOST_STATS)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def set_status(ctx: Context, ref: Ref, status: str, source: Ref | None = None) -> bool:
    """Apply a major status. Fails if one is already present or a hook vetoes."""
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0 or side.status[slot] is not None:
        return False

    immune_types = STATUS_TYPE_IMMUNITY.get(status, ())
    if set(ctx.state.types(side_index, slot)) & set(immune_types):
        ctx.emit(Event("status_immune", side=side_index, slot=slot, detail=status))
        return False

    if not fx.allows(ctx, "try_status", ref, status=status, source=source):
        return False

    side.status[slot] = status
    side.status_data[slot] = {}
    if status == "slp":
        # 1-3 turns of sleep, decremented on each attempt to move.
        side.status_data[slot]["turns"] = ctx.cursor.between(1, 3)
    elif status == "tox":
        side.status_data[slot]["stage"] = 0

    ctx.emit(Event("status", side=side_index, slot=slot, detail=status))
    return True


def cure_status(ctx: Context, ref: Ref) -> bool:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.status[slot] is None:
        return False
    ctx.emit(Event("cure_status", side=side_index, slot=slot, detail=side.status[slot]))
    side.status[slot] = None
    side.status_data[slot] = {}
    return True


# --------------------------------------------------------------------------- #
# Volatiles
# --------------------------------------------------------------------------- #


def add_volatile(ctx: Context, ref: Ref, name: str, **data) -> bool:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0 or name in side.volatiles[slot]:
        return False
    side.volatiles[slot][name] = dict(data)
    ctx.emit(Event("volatile_start", side=side_index, slot=slot, detail=name))
    return True


def remove_volatile(ctx: Context, ref: Ref, name: str, quiet: bool = False) -> bool:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if name not in side.volatiles[slot]:
        return False
    del side.volatiles[slot][name]
    if not quiet:
        ctx.emit(Event("volatile_end", side=side_index, slot=slot, detail=name))
    return True


def volatile(state, ref: Ref, name: str) -> dict | None:
    return state.sides[ref[0]].volatiles[ref[1]].get(name)
