"""Turn resolution: the pure ``step`` at the center of everything.

The turn loop itself stays thin. Almost everything a mechanic wants to do is a
hook (``pkcm.engine.effects``) or a declarative field on the move
(``pkcm.engine.moves``), so what remains here is genuinely about sequencing:
who acts first, when replacements come in, when the battle is over.

``step`` is pure: it clones, mutates the clone, and returns it. The RNG is opened
as a mutable cursor for the duration of the step and sealed back into the
returned state, so purity holds exactly where search needs it -- at the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

from pkcm.data.dex import Move, Stat
from pkcm.engine import abilities  # noqa: F401  -- registers its effects on import
from pkcm.engine import conditions  # noqa: F401  -- registers its effects on import
from pkcm.engine import items  # noqa: F401  -- registers its effects on import
from pkcm.engine import moveeffects  # noqa: F401  -- registers its effects on import
from pkcm.engine import tactics  # noqa: F401  -- registers its effects on import
from pkcm.engine import effects as fx
from pkcm.engine import events as ev
from pkcm.engine import moves as mv
from pkcm.engine import mutate
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.conditions import apply_entry_hazards
from pkcm.engine.effects import Context, Ref
from pkcm.engine.events import Event
from pkcm.engine.state import (
    BOOST_STATS,
    BattleState,
    Phase,
    _bench,
    legal_actions,
)

STRUGGLE_ID = mv.STRUGGLE_ID

#: How long a weather or terrain set by a move lasts.
FIELD_DURATION = 5


class IllegalActionError(ValueError):
    pass


def make_context(state: BattleState, log: list[Event] | None = None) -> Context:
    """A context bound to ``state``. Exposed for tests and tooling."""
    return Context(state=state, cursor=state.rng.cursor(), log=log if log is not None else [])


#: One player's decisions for a step: one action per field position.
Choice = Action | Sequence[Action]


def _as_choices(choice: Choice) -> tuple[Action, ...]:
    return (choice,) if isinstance(choice, Action) else tuple(choice)


def step(
    state: BattleState,
    choice_p0: Choice,
    choice_p1: Choice,
) -> tuple[BattleState, list[Event]]:
    """Advance the battle by one decision point. Never mutates ``state``.

    Each player submits one action per field position. Singles has one, so a
    bare ``Action`` is accepted and means what it always did.
    """
    choices = (_as_choices(choice_p0), _as_choices(choice_p1))
    _validate(state, choices)

    next_state = state.clone()
    log: list[Event] = []
    ctx = Context(state=next_state, cursor=next_state.rng.cursor(), log=log)

    if state.phase is Phase.TEAM_PREVIEW:
        _resolve_team_preview(ctx, choices)
    elif state.phase is Phase.FORCED_SWITCH:
        _resolve_forced_switch(ctx, choices)
    elif state.phase is Phase.MID_TURN_SWITCH:
        _resume_turn(ctx, choices)
    elif state.phase is Phase.BATTLE:
        _resolve_turn(ctx, choices)

    next_state.rng = ctx.cursor.seal()
    return next_state, log


Choices = tuple[tuple[Action, ...], ...]


def _validate(state: BattleState, choices: Choices) -> None:
    for player, actions in enumerate(choices):
        for position, action in enumerate(actions):
            allowed = legal_actions(state, player, position)
            if action not in allowed:
                raise IllegalActionError(
                    f"player {player} position {position} submitted {action} in "
                    f"{state.phase.name}; legal: {[str(a) for a in allowed]}"
                )
        # Two positions cannot send in the same Pokemon. No per-position mask
        # can say so, because the conflict exists only between them.
        switching = [a.index for a in actions if a.kind is ActionKind.SWITCH]
        if len(switching) != len(set(switching)):
            raise IllegalActionError(
                f"player {player} sent the same Pokemon to two positions: {switching}"
            )


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def _resolve_team_preview(ctx: Context, choices: Choices) -> None:
    state = ctx.state
    positions = state.config.active_count
    for player, actions in enumerate(choices):
        side = state.sides[player]
        brought = actions[0].selection
        side.selection = brought
        side.hp = [state.parties[player][i].stats[Stat.HP] for i in brought]
        side.pp = [list(state.parties[player][i].max_pp) for i in brought]
        side.status = [None] * len(brought)
        side.status_data = [{} for _ in brought]
        side.boosts = [[0] * len(BOOST_STATS) for _ in brought]
        side.volatiles = [{} for _ in brought]
        # Lead order is selection order: slot 0 to position 0, and in doubles
        # slot 1 to position 1.
        side.active = list(range(positions))
        side.must_switch = [False] * positions
        state.overrides[player].extend({} for _ in brought)
        ctx.emit(ev.team_preview(player, brought))

    # Everyone is standing there before anyone's switch-in ability fires, and
    # the order among them is Speed order: the faster lead's Intimidate lands
    # first, and in doubles that decides which Attack drop the other one sees.
    arrivals = [(player, position) for player in (0, 1) for position in range(positions)]
    for player, position in arrivals:
        _announce_arrival(ctx, player, position)
    for player, position in _by_speed(ctx, arrivals):
        _greet_field(ctx, player, position)

    state.phase = Phase.BATTLE


def _resolve_forced_switch(ctx: Context, choices: Choices) -> None:
    _take_replacements(ctx, choices)
    if not ctx.state.finished:
        ctx.state.phase = Phase.BATTLE


def _take_replacements(ctx: Context, choices: Choices) -> None:
    """Send in everyone who owes a replacement, then let them all arrive.

    Split in two on purpose: a side can owe two at once in doubles, and both
    are on the field before either one's switch-in ability goes off.
    """
    arrived: list[tuple[int, int]] = []
    for player, actions in enumerate(choices):
        for position, action in enumerate(actions):
            if action.kind is not ActionKind.SWITCH:
                continue
            _leave_field(ctx, player, position)
            ctx.state.sides[player].active[position] = action.index
            ctx.state.sides[player].must_switch[position] = False
            _announce_arrival(ctx, player, position)
            arrived.append((player, position))

    for player, position in _by_speed(ctx, arrived):
        _greet_field(ctx, player, position)


def _resolve_turn(ctx: Context, choices: Choices) -> None:
    state = ctx.state
    state.turn += 1
    ctx.emit(ev.turn_start(state.turn))

    actors = _actors(state)

    # Recharging is spent by doing nothing, and a PASS never reaches the queue,
    # so it is settled before the queue is built.
    for player, position in actors:
        slot = state.sides[player].active[position]
        action = _action_for(choices, player, position)
        if action.kind is ActionKind.PASS and slot >= 0 \
                and state.sides[player].has_volatile(slot, "mustrecharge"):
            ctx.emit(Event("recharging", side=player, slot=slot))
            mutate.remove_volatile(ctx, (player, slot), "mustrecharge", quiet=True)

    switchers = [a for a in actors if _action_for(choices, *a).kind is ActionKind.SWITCH]
    attackers = [a for a in actors
                 if _action_for(choices, *a).kind in (ActionKind.MOVE, ActionKind.STRUGGLE)]

    # Switches always resolve before any move, fastest first.
    arrived: list[tuple[int, int]] = []
    for player, position in _by_speed(ctx, switchers):
        _leave_field(ctx, player, position)
        state.sides[player].active[position] = _action_for(choices, player, position).index
        _announce_arrival(ctx, player, position)
        arrived.append((player, position))
    for player, position in _by_speed(ctx, arrived):
        _greet_field(ctx, player, position)

    # Then Mega Evolution, before move order is worked out -- the new forme's
    # Speed is what decides who goes first.
    megas = [a for a in actors if _action_for(choices, *a).mega]
    for player, position in _by_speed(ctx, megas):
        _mega_evolve(ctx, player, position)

    state.turn_actions = choices
    state.turn_queue = _move_order(ctx, choices, attackers)
    _run_queue(ctx)


def _actors(state: BattleState) -> list[tuple[int, int]]:
    """Every field position, whether or not anyone is standing in it."""
    return [(player, position)
            for player in (0, 1)
            for position in range(len(state.sides[player].active))]


def _action_for(choices: Choices, player: int, position: int) -> Action:
    actions = choices[player]
    return actions[position] if position < len(actions) else Action.PASS


def _resume_turn(ctx: Context, choices: Choices) -> None:
    """Pick a turn back up after a mid-turn switch has been chosen."""
    _take_replacements(ctx, choices)
    if not ctx.state.finished:
        ctx.state.phase = Phase.BATTLE
        _run_queue(ctx)


def _run_queue(ctx: Context) -> None:
    """Work through whoever still has to act. May stop partway."""
    state = ctx.state
    while state.turn_queue:
        if state.finished:
            return
        player, position = state.turn_queue.pop(0)
        side = state.sides[player]
        slot = side.active[position]
        if slot < 0 or side.is_fainted(slot):
            continue  # knocked out before it could act

        _use(ctx, player, position, _action_for(state.turn_actions, player, position))
        ctx.acted.add((player, slot))
        for ref in state.everyone():
            mutate.check_item_triggers(ctx, ref)
        if _check_loss(ctx):
            return

        # A self-switch stops the turn here: the replacement has to be chosen
        # and standing before anyone else moves.
        if _owes_mid_turn_switch(ctx):
            state.phase = Phase.MID_TURN_SWITCH
            return

    _end_of_turn(ctx)


def _owes_mid_turn_switch(ctx: Context) -> bool:
    return any(side.owes_switch() for side in ctx.state.sides)


# --------------------------------------------------------------------------- #
# Switching
#
# Split into three because doubles needs the seam: everyone who is coming in
# leaves and arrives first, and only then does anyone's switch-in ability fire.
# Running them one Pokemon at a time would let the first arrival's Intimidate
# hit a partner that has not been sent out yet.
# --------------------------------------------------------------------------- #


def _leave_field(ctx: Context, player: int, position: int) -> None:
    """Take whoever is standing here off the field, keeping what survives."""
    side = ctx.state.sides[player]
    slot = side.active[position]
    if slot < 0:
        return
    # Natural Cure and Regenerator fire here, before the state is wiped.
    if not side.is_fainted(slot):
        fx.notify(ctx, "switch_out", (player, slot), scope="self")
    side.clear_on_switch_out(slot)
    ctx.state.clear_temporary_overrides(player, slot)


def _announce_arrival(ctx: Context, player: int, position: int) -> None:
    """The Pokemon is on the field and hazards have had their say."""
    side = ctx.state.sides[player]
    slot = side.active[position]
    if slot < 0:
        return
    ref: Ref = (player, slot)
    pokemon = ctx.state.pokemon(*ref)
    ctx.emit(
        ev.switch_in(player, slot, ctx.state.species_id(*ref), side.hp[slot], pokemon.max_hp)
    )
    apply_entry_hazards(ctx, ref)


def _greet_field(ctx: Context, player: int, position: int) -> None:
    """Switch-in abilities, once everyone arriving this turn has arrived."""
    side = ctx.state.sides[player]
    slot = side.active[position]
    if slot < 0 or side.hp[slot] <= 0:
        return
    ref: Ref = (player, slot)

    from pkcm.engine.tactics import _healing_wish_on_entry

    _healing_wish_on_entry(ctx, ref)
    fx.notify(ctx, "switch_in", ref)


def switch_into(ctx: Context, player: int, position: int, slot: int) -> None:
    """Replace one field position, start to finish.

    The three steps run back to back here because there is only one Pokemon
    involved. Anything sending in several at once -- team preview, a double
    knockout -- has to use the steps directly, so that all of them are standing
    before any of their switch-in abilities fire.
    """
    _leave_field(ctx, player, position)
    ctx.state.sides[player].active[position] = slot
    _announce_arrival(ctx, player, position)
    _greet_field(ctx, player, position)


def _mega_evolve(ctx: Context, player: int, position: int) -> None:
    """Spend the battle's one Mega Evolution.

    Permanent: Champions does not revert it even on fainting
    (mods/champions/scripts.ts, formeChange), so the override is marked as
    surviving a switch-out.
    """
    side = ctx.state.sides[player]
    slot = side.active[position]
    if slot < 0:
        return
    ref: Ref = (player, slot)
    target = ctx.state.mega_target(*ref)
    if target is None:
        return

    from pkcm.engine.abilities import _become

    ctx.state.mega_used[player] = True
    _become(ctx, ref, target, permanent=True)
    ctx.state.set_override(player, slot, "ability",
                           ctx.state.config.dex.species[target].abilities[0],
                           permanent=True)
    ctx.emit(
        Event("mega_evolve", side=player, slot=slot,
              species=ctx.state.species_id(*ref), detail=target)
    )
    # The new forme's ability starts now: Mega Mawile's Intimidate fires here.
    fx.notify(ctx, "switch_in", ref)


# --------------------------------------------------------------------------- #
# Ordering
#
# Four Pokemon can act in doubles, so ordering is a sort rather than the
# two-way comparison singles got away with. The random tie-break is drawn only
# for positions that are actually tied, which keeps a singles turn consuming
# exactly the RNG it used to.
# --------------------------------------------------------------------------- #

Actor = tuple[int, int]


def _speed(ctx: Context, actor: Actor) -> int:
    player, position = actor
    slot = ctx.state.sides[player].active[position]
    if slot < 0:
        return 0
    return mutate.effective_stat(ctx, (player, slot), Stat.SPE)


def _shuffle(ctx: Context, group: list[Actor]) -> list[Actor]:
    """Fisher-Yates over a speed tie. One draw for a pair, as before."""
    if len(group) < 2:
        return group
    if len(group) == 2:
        return group if ctx.cursor.chance(1, 2) else [group[1], group[0]]
    shuffled = list(group)
    for index in range(len(shuffled) - 1, 0, -1):
        pick = ctx.cursor.between(0, index)
        shuffled[index], shuffled[pick] = shuffled[pick], shuffled[index]
    return shuffled


def _order_by(ctx: Context, actors: list[Actor], key) -> list[Actor]:
    """Sort by ``key`` descending, breaking exact ties at random.

    Trick Room reverses Speed, so it is applied to the Speed component of the
    key rather than to the sort -- priority keeps pointing the same way.
    """
    if len(actors) < 2:
        return list(actors)
    graded = sorted(((key(actor), actor) for actor in actors),
                    key=lambda pair: pair[0], reverse=True)
    ordered: list[Actor] = []
    index = 0
    while index < len(graded):
        end = index + 1
        while end < len(graded) and graded[end][0] == graded[index][0]:
            end += 1
        ordered.extend(_shuffle(ctx, [actor for _, actor in graded[index:end]]))
        index = end
    return ordered


def _speed_key(ctx: Context, actor: Actor) -> int:
    speed = _speed(ctx, actor)
    return -speed if "trickroom" in ctx.state.field.rooms else speed


def _by_speed(ctx: Context, actors: list[Actor]) -> list[Actor]:
    """Fastest first -- or slowest first under Trick Room."""
    return _order_by(ctx, actors, lambda actor: _speed_key(ctx, actor))


def _move_order(ctx: Context, choices: Choices, actors: list[Actor]) -> list[Actor]:
    """Priority first, then Speed. Trick Room reverses Speed but not priority."""
    def key(actor: Actor) -> tuple[int, int]:
        priority = _priority(ctx, actor, _action_for(choices, *actor))
        return (priority, _speed_key(ctx, actor))

    return _order_by(ctx, actors, key)


def _priority(ctx: Context, actor: Actor, action: Action) -> int:
    player, position = actor
    slot = ctx.state.sides[player].active[position]
    if slot < 0:
        return 0
    move = _chosen_move(ctx.state, actor, action)
    return fx.modify(ctx, "modify_priority", move.priority, (player, slot),
                     scope="self", move=move)


def _chosen_move(state: BattleState, actor: Actor, action: Action) -> Move:
    if action.kind is ActionKind.STRUGGLE:
        return state.config.dex.moves[STRUGGLE_ID]
    player, position = actor
    slot = state.sides[player].active[position]
    return state.moves(player, slot)[action.index]


def _use(ctx: Context, player: int, position: int, action: Action) -> None:
    slot = ctx.state.sides[player].active[position]
    attacker: Ref = (player, slot)
    move = _chosen_move(ctx.state, (player, position), action)
    index = action.index if action.kind is ActionKind.MOVE else None
    mv.use_move(ctx, attacker, move, index, target_code=action.target)


# --------------------------------------------------------------------------- #
# End of turn
# --------------------------------------------------------------------------- #


def _end_of_turn(ctx: Context) -> None:
    state = ctx.state
    if state.finished:
        return

    # Everyone on the field, in Speed order across both sides -- a doubles
    # residual pass interleaves the two teams rather than doing one then the
    # other, which is what decides who a Leftovers tick outlives.
    for player, position in _by_speed(ctx, _actors(state)):
        slot = state.sides[player].active[position]
        if slot < 0 or state.sides[player].is_fainted(slot):
            continue
        fx.notify(ctx, "residual", (player, slot))
        mutate.check_item_triggers(ctx, (player, slot))
        if _check_loss(ctx):
            return

    from pkcm.engine.moveeffects import resolve_wish

    for player in (0, 1):
        if ctx.state.sides[player].conditions.get("wish_ready"):
            del ctx.state.sides[player].conditions["wish_ready"]
            resolve_wish(ctx, player)
        elif "wish" in ctx.state.sides[player].conditions:
            ctx.state.sides[player].conditions["wish_ready"] = 1

    _tick_field(ctx)
    _clear_turn_volatiles(ctx)

    needs_switch = False
    for player in (0, 1):
        side = state.sides[player]
        # A side can lose both its Pokemon in one turn and have only one left
        # to send. The first position gets it; the second is emptied for good,
        # and doubles carries on with three on the field. Handing out more
        # replacements than exist would deadlock the turn.
        available = len(_bench(state, player))
        for position, slot in enumerate(side.active):
            if slot < 0 or not side.is_fainted(slot):
                continue
            if available > 0:
                side.must_switch[position] = True
                available -= 1
                needs_switch = True
            else:
                side.active[position] = -1
                ctx.emit(Event("position_empty", side=player, slot=position))

    if needs_switch:
        state.phase = Phase.FORCED_SWITCH
        return

    if state.turn >= state.config.turn_limit:
        _finish(ctx, winner=_decide_by_attrition(state), detail="turn limit")


def _tick_field(ctx: Context) -> None:
    field = ctx.state.field
    if field.weather is not None:
        field.weather_turns -= 1
        if field.weather_turns <= 0:
            ctx.emit(Event("weather_end", detail=field.weather))
            field.weather = None
    if field.terrain is not None:
        field.terrain_turns -= 1
        if field.terrain_turns <= 0:
            ctx.emit(Event("terrain_end", detail=field.terrain))
            field.terrain = None
    for name in list(field.rooms):
        field.rooms[name] -= 1
        if field.rooms[name] <= 0:
            ctx.emit(Event("room_end", detail=name))
            del field.rooms[name]

    from pkcm.engine.conditions import SIDE_CONDITION_DURATION

    for player in (0, 1):
        conditions = ctx.state.sides[player].conditions
        for name in list(conditions):
            if name not in SIDE_CONDITION_DURATION:
                continue  # hazards are layers, not turns
            conditions[name] -= 1
            if conditions[name] <= 0:
                del conditions[name]
                ctx.emit(Event("side_condition_end", side=player, detail=name))


def _clear_turn_volatiles(ctx: Context) -> None:
    """Protect lasts one turn; the stall counter resets the turn it is not used."""
    for player in (0, 1):
        side = ctx.state.sides[player]
        for slot in range(len(side.hp)):
            volatiles = side.volatiles[slot]
            if "protect" in volatiles:
                del volatiles["protect"]
            elif "stall" in volatiles:
                del volatiles["stall"]
            volatiles.pop("flinch", None)
            volatiles.pop("roost", None)
            volatiles.pop("endure", None)
            for shield in ("kingsshield", "banefulbunker", "spikyshield",
                           "silktrap", "obstruct", "burningbulwark"):
                volatiles.pop(shield, None)
            # Counter and Mirror Coat only answer damage from this turn.
            volatiles.pop("hurtthisturn", None)
            if slot not in side.active:
                volatiles.pop("lastmove", None)


# --------------------------------------------------------------------------- #
# Ending
# --------------------------------------------------------------------------- #


def _check_loss(ctx: Context) -> bool:
    if ctx.state.finished:
        return True
    lost = [side.has_lost() for side in ctx.state.sides]
    if lost[0] and lost[1]:
        _finish(ctx, winner=None, detail="both sides fainted")
        return True
    for player in (0, 1):
        if lost[player]:
            _finish(ctx, winner=1 - player, detail="all Pokemon fainted")
            return True
    return False


def _decide_by_attrition(state: BattleState) -> int | None:
    """The official time-over ruling, in its four tiers.

    1. how many Pokemon are still standing
    2. remaining HP as a share of the HP that side started with
    3. remaining HP in absolute terms
    4. remaining PP

    Tiers two and three are not the same test. Two sides can hold the same
    *share* of their HP while holding very different amounts of it, and a
    bulky team is not entitled to win a tie on that alone -- so the ratio is
    asked first and the raw number second.
    """
    for measure in (_living_count, _hp_share, _hp_absolute, _pp_remaining):
        scores = [measure(state, player) for player in (0, 1)]
        if scores[0] != scores[1]:
            return 0 if scores[0] > scores[1] else 1
    return None


def _living_count(state: BattleState, player: int) -> int:
    return len(state.sides[player].living_slots())


def _hp_absolute(state: BattleState, player: int) -> int:
    return sum(state.sides[player].hp)


def _hp_share(state: BattleState, player: int) -> float:
    side = state.sides[player]
    total = sum(state.pokemon(player, slot).max_hp for slot in range(len(side.hp)))
    return sum(side.hp) / total if total else 0.0


def _pp_remaining(state: BattleState, player: int) -> int:
    return sum(sum(slot) for slot in state.sides[player].pp)


def _finish(ctx: Context, winner: int | None, detail: str) -> None:
    ctx.state.phase = Phase.FINISHED
    ctx.state.winner = winner
    for side in ctx.state.sides:
        side.must_switch = False
    ctx.emit(ev.battle_end(winner, detail))


#: Kept importable from here because it reads as part of the battle's surface.
compute_damage = mv.compute_damage
