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
from pkcm.engine.state import (
    BOOST_INDEX,
    BOOST_STATS,
    MAX_BOOST,
    MIN_BOOST,
    uproar_in_progress,
)

#: Damage that did not come from a move connecting. Magic Guard ignores all of
#: it; Poison Heal turns its own share of it into healing. Routing every such
#: source through one funnel is what keeps those two abilities from having to be
#: special-cased in the status handler, the weather handler and the hazard code
#: separately.
INDIRECT_DAMAGE_KINDS = frozenset(
    {"status_damage", "weather_damage", "hazard_damage", "recoil"}
)

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


#: Wonder Room swaps these two. Stat *stages* are not swapped with them,
#: which is why this is done here rather than in the boost table.
WONDER_ROOM_SWAP = {Stat.DEF: Stat.SPD, Stat.SPD: Stat.DEF}


def raw_stat(state, ref: Ref, stat: Stat) -> int:
    if "wonderroom" in state.field.rooms:
        stat = WONDER_ROOM_SWAP.get(stat, stat)
    return state.stats(ref[0], ref[1])[stat]


def effective_stat(
    ctx: Context,
    ref: Ref,
    stat: Stat,
    move=None,
    opponent: Ref | None = None,
) -> int:
    """A stat as the damage formula should see it: stages, then hooks.

    ``move`` matters more than it looks: Blaze, Torrent and Flash Fire are all
    Attack modifiers that only apply to moves of a particular type, so a stat
    hook that cannot see the move cannot express them.
    """
    side_index, slot = ref
    extra = {"move": move, "opponent": opponent}
    value = fx.modify(ctx, "modify_stat", raw_stat(ctx.state, ref, stat), ref,
                      stat=stat, **extra)

    boost_name = STAT_TO_BOOST.get(stat)
    if boost_name is not None:
        stage = ctx.state.sides[side_index].boost(slot, boost_name)
        # Unaware sits on the *other* Pokemon and refuses to see these stages,
        # so the question has to be asked from over there.
        if opponent is not None and not fx.allows(
            ctx, "ignore_stat_stages", opponent, scope="self", stat=stat, target=ref
        ):
            stage = 0
        value = int(value * stage_multiplier(stage))

    value = fx.modify(ctx, "modify_boosted_stat", value, ref, stat=stat, **extra)
    return max(1, int(value))


def weight_kg(ctx: Context, ref: Ref) -> float:
    """Weight as Low Kick and Heavy Slam see it, after Light/Heavy Metal."""
    species = ctx.state.config.dex.species[ctx.state.species_id(*ref)]
    return max(0.1, fx.modify(ctx, "modify_weight", species.weight_kg, ref, scope="self"))


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

    source: Ref | None = event_fields.pop("__source__", None)
    culprit_move = event_fields.pop("__move__", None)
    cause = event_fields.get("detail")
    if kind in INDIRECT_DAMAGE_KINDS or cause == "confusion":
        amount = fx.modify(ctx, "modify_indirect_damage", amount, ref,
                           source_kind=kind, cause=cause)
        if amount is None or amount <= 0:
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
    check_faint(ctx, ref, source=source, move=culprit_move)
    return dealt


def heal(ctx: Context, ref: Ref, amount: int, reason: str | None = None) -> int:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0:
        return 0
    if "healblock" in side.volatiles[slot]:
        ctx.emit(Event("heal_blocked", side=side_index, slot=slot))
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


def check_faint(ctx: Context, ref: Ref, source: Ref | None = None, move=None) -> bool:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] > 0:
        return False
    if side.volatiles[slot].get("__fainted__"):
        return True
    side.volatiles[slot]["__fainted__"] = True
    ctx.emit(ev.faint(side_index, slot, ctx.state.species_id(side_index, slot)))
    fx.notify(ctx, "faint", ref, source=source)
    # Moxie and friends hang off the *killer*, not the victim.
    if source is not None and source != ref:
        fx.notify(ctx, "kill", source, scope="self", victim=ref, move=move)
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
        if applied[name] < 0:
            # Lash Out asks whether this one has been dropped *this turn*, and
            # the answer has to be recorded where the drop happens. Set rather
            # than added, because it is bookkeeping and not a condition anyone
            # announces; it goes when the Pokemon leaves, which is right.
            side.volatiles[slot]["statdropped"] = {"turn": ctx.state.turn}
        ctx.emit(
            Event("boost", side=side_index, slot=slot, detail=name,
                  amount=after - before, hp=after)
        )
        for watcher in [ref, *ctx.state.foes(ref)]:
            if ctx.state.sides[watcher[0]].hp[watcher[1]] > 0:
                fx.notify(ctx, "after_boost", watcher, scope="self", boosted=ref,
                          stat=name, stages=after - before, source=source)
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

    # An Uproar anywhere on the field refuses sleep to everyone, which is why
    # this is asked here and not through ``try_status``: the volatile is on the
    # Pokemon making the noise, not on the one being put to sleep.
    if status == "slp" and uproar_in_progress(ctx.state):
        ctx.emit(Event("status_immune", side=side_index, slot=slot, detail="uproar"))
        return False

    # The source gets to say which types are immune, because Corrosion is a
    # property of the poisoner, not of the poisoned.
    immune_types = STATUS_TYPE_IMMUNITY.get(status, ())
    if source is not None:
        immune_types = fx.modify(ctx, "status_immunity", immune_types, source,
                                 scope="self", status=status, target=ref)

    if set(ctx.state.types(side_index, slot)) & set(immune_types):
        ctx.emit(Event("status_immune", side=side_index, slot=slot, detail=status))
        return False

    if not fx.allows(ctx, "try_status", ref, status=status, source=source):
        return False

    side.status[slot] = status
    side.status_data[slot] = {}
    if status == "slp":
        from pkcm.engine.conditions import SLEEP_DURATIONS

        side.status_data[slot]["turns"] = ctx.cursor.choice(SLEEP_DURATIONS)
    elif status == "frz":
        from pkcm.engine.conditions import FREEZE_DURATION

        side.status_data[slot]["turns"] = FREEZE_DURATION
    elif status == "tox":
        side.status_data[slot]["stage"] = 0

    ctx.emit(Event("status", side=side_index, slot=slot, detail=status))
    fx.notify(ctx, "after_status", ref, scope="self", status=status, source=source)
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


def add_volatile(ctx: Context, ref: Ref, name: str, source: Ref | None = None, **data) -> bool:
    side_index, slot = ref
    side = ctx.state.sides[side_index]
    if side.hp[slot] <= 0 or name in side.volatiles[slot]:
        return False
    if not fx.allows(ctx, "try_volatile", ref, volatile=name, source=source):
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


def consume_item(ctx: Context, ref: Ref, reason: str | None = None) -> str | None:
    """Use up the held item. Gone for the rest of the battle, not just the turn."""
    item_id = ctx.state.item_id(*ref)
    if item_id is None:
        return None
    ctx.state.set_override(ref[0], ref[1], "item", None, permanent=True)
    ctx.emit(Event("item_used", side=ref[0], slot=ref[1], detail=item_id,
                   move=reason))
    fx.notify(ctx, "after_use_item", ref, scope="self", item=item_id)
    return item_id


def check_item_triggers(ctx: Context, ref: Ref) -> None:
    """Give held items a chance to react to whatever just happened."""
    if ctx.state.sides[ref[0]].hp[ref[1]] > 0:
        fx.notify(ctx, "update", ref, scope="self")


def volatile(state, ref: Ref, name: str) -> dict | None:
    return state.sides[ref[0]].volatiles[ref[1]].get(name)
