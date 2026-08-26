"""The hook system every mechanic plugs into.

Principle (d) in docs/DESIGN.md. Abilities, items, status conditions, volatile
conditions, side conditions and weather all want to interfere with the *same*
handful of computations -- what a stat is, whether a move connects, how much
damage it does, what happens at the end of the turn. Written as branches, that
becomes an unmaintainable chain of special cases inside the damage formula.
Written as hooks, each mechanic is a small self-contained table.

An ``Effect`` is anything with handlers: an ability, an item, a status, a
volatile, a side condition, a weather. They are registered once at import time
and are immutable, so nothing here is cloned with the battle state.

Two shapes of hook:

``modify``  a value flows through every handler in priority order, each free to
            change it. Used for stats, accuracy, damage, priority.
``notify``  handlers run for their side effects only. Used for switch-in,
            after-damage, end-of-turn.

Handler signatures are keyword-only and documented per event in ``EVENTS``.
A ``modify`` handler returns the new value, or ``None`` to leave it alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from pkcm.engine.events import Event
from pkcm.engine.rng import RngCursor
from pkcm.engine.state import BattleState

#: (side, slot). Identifies one Pokemon in the battle.
Ref = tuple[int, int]

Handler = Callable[..., Any]

#: Every hook point, with what it is for. Handlers take ``ctx`` plus the
#: keywords listed here, and ``modify`` handlers also take ``value``.
EVENTS: dict[str, str] = {
    # notify
    "switch_in": "ref -- a Pokemon has just come onto the field",
    "switch_out": "ref -- a Pokemon is leaving the field, before its state is wiped",
    "modify_move": (
        "active, attacker, defender -- rewrite the move for this use only. "
        "Showdown's onModifyMove; the -ate abilities, Protean, Skill Link, "
        "Long Reach, Infiltrator and Sheer Force all live here."
    ),
    "after_damage": "ref, target, move, damage -- a damaging move connected",
    "after_move": "ref, move -- the move finished resolving",
    "residual": "ref -- end of turn, in Speed order",
    "faint": "ref, source -- this Pokemon fainted; source is who caused it",
    "kill": "ref, victim, move -- this Pokemon knocked something out (Moxie)",
    "after_status": "ref, status, source -- a status has just been applied (Synchronize)",
    "after_boost": (
        "ref, boosted, stat, stages, source -- a stat stage changed. Fires on "
        "both sides: Opportunist is watching from across the field, so a hook "
        "that only ran on the boosted Pokemon could never see it."
    ),
    # modify
    "modify_stat": "ref, stat -- raw stat before stat stages",
    "modify_boosted_stat": "ref, stat -- stat after stat stages",
    "modify_priority": "ref, move -- move priority",
    "modify_accuracy": "ref, target, move -- percentage chance to hit",
    "modify_base_power": (
        "attacker, defender, move -- the move's power before the damage formula. "
        "Showdown's onBasePower, and a distinct step from modify_damage: "
        "Technician's <=60 test reads the power *after* other base-power "
        "modifiers, which only works if this is its own pass."
    ),
    "modify_damage": "ref, target, move, crit -- final damage",
    "ignore_stat_stages": (
        "ref (the OPPONENT), stat, target -- may the target's stat stages count? "
        "Unaware answers no, and it has to be asked from the other side."
    ),
    "modify_weight": "ref -- the Pokemon's weight in kg (Light Metal, Heavy Metal)",
    "modify_effectiveness": "ref, target, move -- type multiplier",
    "modify_crit_ratio": "ref, target, move -- denominator of the crit chance",
    "modify_indirect_damage": (
        "ref, source_kind, cause -- damage from anything but a move connecting: "
        "status, weather, hazards, recoil, confusion. Magic Guard zeroes it; "
        "Poison Heal turns its own poison damage into healing."
    ),
    "status_immunity": (
        "ref (the SOURCE), status, target -- the types immune to this status. "
        "Corrosion empties it so Steel and Poison can be poisoned."
    ),
    # veto: return False to prevent
    "try_move": "ref, move -- may this Pokemon act at all",
    "try_status": "ref, status, source -- may this status be applied",
    "try_boost": "ref, stat, stages -- stat stage change, return new stages",
    "try_hit": "ref, attacker, defender, move -- may this move hit at all",
    "try_volatile": "ref, volatile, source -- may this volatile condition be added",
    "try_secondary": "ref, attacker, move -- may a move's secondary effects land",
}


@dataclass(frozen=True, slots=True)
class Effect:
    id: str
    kind: str
    handlers: dict[str, Handler] = field(default_factory=dict)
    #: Lower runs first. Ties broken by the order effects are gathered in.
    priority: int = 0
    name: str = ""


#: (kind, id) -> Effect. Populated at import time by ``register``.
REGISTRY: dict[tuple[str, str], Effect] = {}


def register(kind: str, effect_id: str, *, priority: int = 0, name: str = "", **handlers: Handler) -> Effect:
    """Register one mechanic. Unknown event names are a typo and must not pass."""
    unknown = set(handlers) - set(EVENTS)
    if unknown:
        raise KeyError(f"{kind}:{effect_id} registers unknown events {sorted(unknown)}")
    effect = Effect(id=effect_id, kind=kind, handlers=handlers, priority=priority, name=name)
    REGISTRY[(kind, effect_id)] = effect
    return effect


def lookup(kind: str, effect_id: str | None) -> Effect | None:
    if effect_id is None:
        return None
    return REGISTRY.get((kind, effect_id))


@dataclass(slots=True)
class Context:
    """Everything a handler may touch. One per ``step``."""

    state: BattleState
    cursor: RngCursor
    log: list[Event]
    #: Pokemon whose ability is being ignored for the moment. Mold Breaker and
    #: friends put the defender in here for the duration of one move. Gathering
    #: consults it, which is the only way suppression can be correct: the
    #: defender's ability has to be invisible to *every* hook the move runs,
    #: not just the one place someone remembered to check.
    suppressed_abilities: set[Ref] = field(default_factory=set)
    #: Who has already taken their action this turn. Analytic needs it, and
    #: nothing else can reconstruct it after the fact.
    acted: set[Ref] = field(default_factory=set)

    def emit(self, event: Event) -> None:
        self.log.append(event)

    def ability_of(self, ref: Ref) -> str | None:
        """The ability in force right now -- ``None`` while suppressed."""
        if ref in self.suppressed_abilities:
            return None
        return self.state.ability_id(*ref)


# --------------------------------------------------------------------------- #
# Gathering
# --------------------------------------------------------------------------- #


def effects_on(ctx: "Context", ref: Ref) -> Iterator[Effect]:
    """Every effect attached to one Pokemon, innermost first."""
    state = ctx.state
    side_index, slot = ref
    side = state.sides[side_index]

    ability = lookup("ability", ctx.ability_of(ref))
    if ability is not None:
        yield ability

    item = lookup("item", state.item_id(side_index, slot))
    if item is not None:
        yield item

    if slot < len(side.status):
        status = lookup("status", side.status[slot])
        if status is not None:
            yield status

    if slot < len(side.volatiles):
        for name in side.volatiles[slot]:
            volatile = lookup("volatile", name)
            if volatile is not None:
                yield volatile


def effects_on_side(state: BattleState, side_index: int) -> Iterator[Effect]:
    for name in state.sides[side_index].conditions:
        condition = lookup("side", name)
        if condition is not None:
            yield condition


def effects_on_field(state: BattleState) -> Iterator[Effect]:
    weather = lookup("weather", state.field.weather)
    if weather is not None:
        yield weather
    terrain = lookup("terrain", state.field.terrain)
    if terrain is not None:
        yield terrain
    for name in state.field.rooms:
        room = lookup("room", name)
        if room is not None:
            yield room


def _ordered(effects: list[Effect], event: str) -> list[Effect]:
    relevant = [effect for effect in effects if event in effect.handlers]
    if len(relevant) > 1:
        relevant.sort(key=lambda effect: effect.priority)
    return relevant


def _gather(ctx: "Context", ref: Ref, scope: str) -> list[Effect]:
    """``scope`` picks how wide to look: ``"self"``, ``"side"``, ``"all"``."""
    effects = list(effects_on(ctx, ref))
    if scope in ("side", "all"):
        effects.extend(effects_on_side(ctx.state, ref[0]))
    if scope == "all":
        effects.extend(effects_on_field(ctx.state))
    return effects


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def modify(
    ctx: Context,
    event: str,
    value: Any,
    ref: Ref,
    scope: str = "all",
    **kwargs: Any,
) -> Any:
    """Pass ``value`` through every handler for ``event``, in priority order."""
    for effect in _ordered(_gather(ctx, ref, scope), event):
        result = effect.handlers[event](ctx, ref=ref, value=value, effect=effect, **kwargs)
        if result is not None:
            value = result
    return value


def notify(ctx: Context, event: str, ref: Ref, scope: str = "all", **kwargs: Any) -> None:
    """Run every handler for its side effects."""
    for effect in _ordered(_gather(ctx, ref, scope), event):
        effect.handlers[event](ctx, ref=ref, effect=effect, **kwargs)


def allows(ctx: Context, event: str, ref: Ref, scope: str = "all", **kwargs: Any) -> bool:
    """``False`` as soon as any handler vetoes. Handlers returning ``None`` allow."""
    for effect in _ordered(_gather(ctx, ref, scope), event):
        if effect.handlers[event](ctx, ref=ref, effect=effect, **kwargs) is False:
            return False
    return True


def registered(kind: str) -> list[str]:
    """Every id registered under ``kind``. Used by tests and coverage reports."""
    return sorted(effect_id for (registered_kind, effect_id) in REGISTRY if registered_kind == kind)
