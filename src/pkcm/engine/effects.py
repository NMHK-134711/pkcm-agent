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

Doubles adds a third *scope* rather than a third shape. An effect registered
under ``ally_<event>`` also runs when the event is about its holder's partner
-- Showdown's ``onAllyBasePower``, ``onAllyFaint``. The handler receives both
``ref`` (who the event is about) and ``holder`` (who owns the effect), which in
singles are always the same Pokemon and in doubles are the whole point. Nothing
outside this module has to know: gathering picks the partner's effects up on
its own, so every existing dispatch site got ally scope for free.

Handler signatures are keyword-only and documented per event in ``EVENTS``.
A ``modify`` handler returns the new value, or ``None`` to leave it alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from pkcm.engine.events import Event
from pkcm.engine.rng import RngCursor
from pkcm.engine.state import BattleState
from pkcm.engine.state import Ref as StateRef

#: (side, slot). Identifies one Pokemon in the battle, wherever it stands.
#: Defined in ``state`` and re-exported here, where most handlers import it.
Ref = StateRef

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
    "redirect_target": (
        "ref (the one pulling), attacker, move -- return this Pokemon's own ref "
        "to take a single-target move aimed at its partner. Follow Me and Rage "
        "Powder claim everything; Lightning Rod and Storm Drain claim a type. "
        "Doubles only: in singles there is nothing to redirect away from."
    ),
    "modify_field_duration": (
        "ref (whoever set it), field, kind -- how long a weather, terrain or "
        "screen lasts. The weather rocks and Light Clay live here."
    ),
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
    "after_use_item": "ref, item -- the held item was just consumed",
    "commit_move": (
        "ref, move, move_index -- the holder has committed to a move this turn. "
        "Choice items lock in here."
    ),
    "dealt_damage": "ref, attacker, defender, move, damage -- our move connected",
    "modify_drain": "ref, move -- how much a draining move gives back (Big Root)",
    "update": "ref -- a checkpoint for items that watch HP or status (berries)",
    # ally-scoped: fires when the event is about the holder's partner.
    "ally_modify_base_power": "Battery, Power Spot -- boost the partner's move",
    "ally_modify_damage": "Friend Guard -- soften what the partner takes",
    "ally_modify_accuracy": "Victory Star -- steady the partner's aim",
    "ally_try_status": "Sweet Veil -- refuse a status on the partner's behalf",
    "ally_try_volatile": "Sweet Veil again, for Yawn",
    "ally_after_use_item": "Symbiosis -- hand the partner your own item",
    "ally_faint": "Receiver, Power of Alchemy -- inherit the partner's ability",
}

#: ``ally_x`` is dispatched whenever ``x`` is, for the partner's effects.
ALLY_PREFIX = "ally_"


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
    #: Who is resolving a move right now. Beat Up needs it to count its team.
    acting: Ref | None = None
    #: How many moves deep one call is. ``moveeffects._call_move`` owns it; it
    #: lives here because the chain crosses several moves and none of them is a
    #: natural place to keep it.
    call_depth: int = 0

    def emit(self, event: Event) -> None:
        self.log.append(event)
        if event.kind in REVELATION_KINDS:
            record_revelation(self.state, event)

    def item_of(self, ref: Ref) -> str | None:
        """The item in force right now -- ``None`` while Magic Room is up.

        The Pokemon is still holding it, and ``state.item_id`` still says so;
        this is the question everything that *acts* on an item should ask.
        """
        if "magicroom" in self.state.field.rooms:
            return None
        return self.state.item_id(*ref)

    def ability_of(self, ref: Ref) -> str | None:
        """The ability in force right now -- ``None`` while suppressed.

        Two things suppress: Mold Breaker for the length of one move, and
        Gastro Acid until the Pokemon leaves the field.
        """
        if ref in self.suppressed_abilities:
            return None
        side, slot = ref
        volatiles = self.state.sides[side].volatiles
        if slot < len(volatiles) and "abilitysuppressed" in volatiles[slot]:
            return None
        return self.state.ability_id(side, slot)


#: Events that make something about a Pokemon public, and what they reveal.
#: Folded into ``state.revealed`` as they are emitted, so the information set is
#: a property of the state rather than of whoever happened to watch the log.
REVEALS_SPECIES = frozenset({"switch_in", "mega_evolve", "forme_change"})
REVEALS_ITEM = frozenset({"use_item", "item_revealed", "knock_off", "item_stolen",
                          "ability"})
REVEALS_ABILITY = frozenset({"ability", "ability_block", "ability_suppressed",
                             "ability_change", "ability_swapped"})
#: Events that make the abilities of *both* Pokemon public at once. A swap is
#: announced for both halves, so watching one happen tells you what each of
#: them had -- and a record that only marked one side would have the observer
#: knowing an ability arrived from nowhere.
REVEALS_BOTH_ABILITIES = frozenset({"ability_swapped", "ability_change"})

#: Everything above, in one set. Emit tests this first: the overwhelming
#: majority of events reveal nothing, and they should cost one hash lookup.
REVELATION_KINDS = (REVEALS_SPECIES | REVEALS_ITEM | REVEALS_ABILITY
                    | {"move_used", "status", "cure_status"})


def record_revelation(state: BattleState, event: Event) -> None:
    """Fold one event into what its side has now shown the opponent."""
    side, slot = event.side, event.slot
    if side is None or slot is None or not state.revealed:
        return
    if slot >= len(state.sides[side].hp):
        return  # a side-wide event whose ``slot`` means something else
    seen = state.revealed[side]

    if event.kind in REVEALS_SPECIES:
        seen.species.add(slot)
    if event.kind == "move_used" and event.move:
        seen.species.add(slot)
        seen.saw_move(slot, event.move)
    if event.kind in REVEALS_ITEM:
        seen.items.add(slot)
    if event.kind in REVEALS_ABILITY:
        seen.abilities.add(slot)
    if event.kind in REVEALS_BOTH_ABILITIES:
        # Whoever it swapped with is on the field, and now visibly holding
        # something that was not theirs a moment ago.
        for other in state.active_refs(1 - side) + state.active_refs(side):
            if other != (side, slot):
                state.revealed[other[0]].abilities.add(other[1])
    if event.kind == "status":
        seen.status_since[slot] = state.turn
    elif event.kind == "cure_status":
        seen.status_since.pop(slot, None)


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

    item = lookup("item", ctx.item_of(ref))
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


#: (ability, item, status, volatiles) -> {event: ((effect, handler), ...)}.
#:
#: What one Pokemon carries changes a handful of times per battle; which event
#: is being asked about changes tens of thousands of times per decision. So the
#: expensive walk -- five REGISTRY lookups and a ``handlers.get`` per effect --
#: is done once per *combination* and every later ask is one dict lookup.
#: Profiled before this existed, the walk was 13% of a self-play second.
#:
#: Global rather than per-state on purpose: the search clones states by the
#: thousand, and a memo living on the state would start cold in every clone.
#: The key is only ids, never objects, so a clone hits the same entries.
#:
#: Safe to build lazily because ``REGISTRY`` is finished at import time --
#: ``register`` calls and the two Lightning Rod handler patches all run before
#: the first battle asks anything. Ordering inside an entry is the walk order
#: exactly (ability, item, status, volatiles in insertion order), which the
#: caller's stable sort by priority depends on for its tie-break.
_SELF_MEMO: dict[tuple, dict[str, tuple]] = {}
_SIDE_MEMO: dict[tuple, dict[str, tuple]] = {}
_FIELD_MEMO: dict[tuple, dict[str, tuple]] = {}


def _fold(table: dict[str, list], effect: "Effect") -> None:
    for event, handler in effect.handlers.items():
        table.setdefault(event, []).append((effect, handler))


def _build_table(memo: dict, key: tuple, kinds) -> dict[str, tuple]:
    table: dict[str, list] = {}
    for kind, name in kinds:
        effect = lookup(kind, name)
        if effect is not None:
            _fold(table, effect)
    frozen = {event: tuple(rows) for event, rows in table.items()}
    memo[key] = frozen
    return frozen


def _self_table_for(ctx: "Context", ref: Ref) -> dict[str, tuple]:
    """The event table for what this Pokemon carries right now.

    Suppression and Magic Room are folded into the key -- a suppressed ability
    keys as ``None`` and shares its entry with genuinely having none.
    """
    state = ctx.state
    side = state.sides[ref[0]]
    slot = ref[1]
    key = (
        ctx.ability_of(ref),
        ctx.item_of(ref),
        side.status[slot] if slot < len(side.status) else None,
        tuple(side.volatiles[slot]) if slot < len(side.volatiles) else (),
    )
    found = _SELF_MEMO.get(key)
    if found is not None:
        return found
    return _build_table(_SELF_MEMO, key,
                        (("ability", key[0]), ("item", key[1]),
                         ("status", key[2]),
                         *(("volatile", name) for name in key[3])))


def _relevant(ctx: "Context", event: str, ref: Ref,
              scope: str) -> list[tuple[Ref, "Effect", Any]]:
    """Everything with a handler for this event, in gathering order.

    The order is the two-step version of old exactly: self, partner, side,
    field, then a stable sort by priority. ``effects_on`` and friends are kept
    above -- they state the rule far more clearly than a memo table does, and
    the tests that pin this function against them read like the rule.
    """
    state = ctx.state
    found: list[tuple[Ref, "Effect", Any]] = []

    rows = _self_table_for(ctx, ref).get(event)
    if rows:
        for effect, handler in rows:
            found.append((ref, effect, handler))

    partner = state.ally(ref)
    if partner is not None:
        # A partner's effect answers to ``ally_<event>`` and to nothing else,
        # so an ability without an ally variant stays silent about its partner.
        rows = _self_table_for(ctx, partner).get(ALLY_PREFIX + event)
        if rows:
            for effect, handler in rows:
                found.append((partner, effect, handler))

    if scope in ("side", "all"):
        conditions = state.sides[ref[0]].conditions
        if conditions:
            key = tuple(conditions)
            table = _SIDE_MEMO.get(key)
            if table is None:
                table = _build_table(_SIDE_MEMO, key,
                                     tuple(("side", name) for name in key))
            rows = table.get(event)
            if rows:
                for effect, handler in rows:
                    found.append((ref, effect, handler))

    if scope == "all":
        field_state = state.field
        key = (field_state.weather, field_state.terrain,
               tuple(field_state.rooms))
        table = _FIELD_MEMO.get(key)
        if table is None:
            table = _build_table(
                _FIELD_MEMO, key,
                (("weather", key[0]), ("terrain", key[1]),
                 *(("room", name) for name in key[2])))
        rows = table.get(event)
        if rows:
            for effect, handler in rows:
                found.append((ref, effect, handler))

    if len(found) > 1:
        found.sort(key=lambda triple: triple[1].priority)
    return found


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
    for holder, effect, handler in _relevant(ctx, event, ref, scope):
        result = handler(ctx, ref=ref, value=value, effect=effect,
                         holder=holder, **kwargs)
        if result is not None:
            value = result
    return value


def notify(ctx: Context, event: str, ref: Ref, scope: str = "all", **kwargs: Any) -> None:
    """Run every handler for its side effects."""
    for holder, effect, handler in _relevant(ctx, event, ref, scope):
        handler(ctx, ref=ref, effect=effect, holder=holder, **kwargs)


def allows(ctx: Context, event: str, ref: Ref, scope: str = "all", **kwargs: Any) -> bool:
    """``False`` as soon as any handler vetoes. Handlers returning ``None`` allow."""
    for holder, effect, handler in _relevant(ctx, event, ref, scope):
        if handler(ctx, ref=ref, effect=effect, holder=holder,
                   **kwargs) is False:
            return False
    return True


def registered(kind: str) -> list[str]:
    """Every id registered under ``kind``. Used by tests and coverage reports."""
    return sorted(effect_id for (registered_kind, effect_id) in REGISTRY if registered_kind == kind)
