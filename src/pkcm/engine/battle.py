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

from pkcm.data.dex import Move, Stat
from pkcm.engine import abilities  # noqa: F401  -- registers its effects on import
from pkcm.engine import conditions  # noqa: F401  -- registers its effects on import
from pkcm.engine import items  # noqa: F401  -- registers its effects on import
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


def step(
    state: BattleState,
    action_p0: Action,
    action_p1: Action,
) -> tuple[BattleState, list[Event]]:
    """Advance the battle by one decision point. Never mutates ``state``."""
    actions = (action_p0, action_p1)
    for player, action in enumerate(actions):
        if action not in legal_actions(state, player):
            raise IllegalActionError(
                f"player {player} submitted {action} in {state.phase.name}; "
                f"legal: {[str(a) for a in legal_actions(state, player)]}"
            )

    next_state = state.clone()
    log: list[Event] = []
    ctx = Context(state=next_state, cursor=next_state.rng.cursor(), log=log)

    if state.phase is Phase.TEAM_PREVIEW:
        _resolve_team_preview(ctx, actions)
    elif state.phase is Phase.FORCED_SWITCH:
        _resolve_forced_switch(ctx, actions)
    elif state.phase is Phase.BATTLE:
        _resolve_turn(ctx, actions)

    next_state.rng = ctx.cursor.seal()
    return next_state, log


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #


def _resolve_team_preview(ctx: Context, actions: tuple[Action, ...]) -> None:
    state = ctx.state
    for player, action in enumerate(actions):
        side = state.sides[player]
        brought = action.selection
        side.selection = brought
        side.hp = [state.parties[player][i].stats[Stat.HP] for i in brought]
        side.pp = [list(state.parties[player][i].max_pp) for i in brought]
        side.status = [None] * len(brought)
        side.status_data = [{} for _ in brought]
        side.boosts = [[0] * len(BOOST_STATS) for _ in brought]
        side.volatiles = [{} for _ in brought]
        side.active = 0
        state.overrides[player].extend({} for _ in brought)
        ctx.emit(ev.team_preview(player, brought))

    for player in (0, 1):
        _enter_field(ctx, player)

    state.phase = Phase.BATTLE


def _resolve_forced_switch(ctx: Context, actions: tuple[Action, ...]) -> None:
    for player, action in enumerate(actions):
        if action.kind is not ActionKind.SWITCH:
            continue
        _switch(ctx, player, action.index)
        ctx.state.sides[player].must_switch = False

    if not ctx.state.finished:
        ctx.state.phase = Phase.BATTLE


def _resolve_turn(ctx: Context, actions: tuple[Action, ...]) -> None:
    state = ctx.state
    state.turn += 1
    ctx.emit(ev.turn_start(state.turn))

    switchers = [p for p in (0, 1) if actions[p].kind is ActionKind.SWITCH]
    attackers = [p for p in (0, 1) if actions[p].kind in (ActionKind.MOVE, ActionKind.STRUGGLE)]

    # Switches always resolve before any move, fastest first.
    for player in _by_speed(ctx, switchers):
        _switch(ctx, player, actions[player].index)

    # Then Mega Evolution, before move order is worked out -- the new forme's
    # Speed is what decides who goes first.
    for player in _by_speed(ctx, [p for p in (0, 1) if actions[p].mega]):
        _mega_evolve(ctx, player)

    for player in _move_order(ctx, actions, attackers):
        if state.finished:
            return
        side = state.sides[player]
        if side.is_fainted(side.active):
            continue  # knocked out before it could act
        _use(ctx, player, actions[player])
        ctx.acted.add((player, side.active))
        for who in (0, 1):
            other = ctx.state.sides[who]
            if other.hp and not other.is_fainted(other.active):
                mutate.check_item_triggers(ctx, (who, other.active))
        if _check_loss(ctx):
            return

    _end_of_turn(ctx)


# --------------------------------------------------------------------------- #
# Switching
# --------------------------------------------------------------------------- #


def _switch(ctx: Context, player: int, slot: int) -> None:
    side = ctx.state.sides[player]
    if side.active >= 0:
        # Natural Cure and Regenerator fire here, before the state is wiped.
        if not side.is_fainted(side.active):
            fx.notify(ctx, "switch_out", (player, side.active), scope="self")
        side.clear_on_switch_out(side.active)
        ctx.state.clear_temporary_overrides(player, side.active)
    side.active = slot
    _enter_field(ctx, player)


def _mega_evolve(ctx: Context, player: int) -> None:
    """Spend the battle's one Mega Evolution.

    Permanent: Champions does not revert it even on fainting
    (mods/champions/scripts.ts, formeChange), so the override is marked as
    surviving a switch-out.
    """
    side = ctx.state.sides[player]
    ref: Ref = (player, side.active)
    target = ctx.state.mega_target(*ref)
    if target is None:
        return

    from pkcm.engine.abilities import _become

    ctx.state.mega_used[player] = True
    _become(ctx, ref, target, permanent=True)
    ctx.state.set_override(player, side.active, "ability",
                           ctx.state.config.dex.species[target].abilities[0],
                           permanent=True)
    ctx.emit(
        Event("mega_evolve", side=player, slot=side.active,
              species=ctx.state.species_id(*ref), detail=target)
    )
    # The new forme's ability starts now: Mega Mawile's Intimidate fires here.
    fx.notify(ctx, "switch_in", ref)


def _enter_field(ctx: Context, player: int) -> None:
    side = ctx.state.sides[player]
    ref: Ref = (player, side.active)
    pokemon = ctx.state.pokemon(*ref)
    ctx.emit(
        ev.switch_in(player, side.active, ctx.state.species_id(*ref),
                     side.hp[side.active], pokemon.max_hp)
    )
    apply_entry_hazards(ctx, ref)
    if side.hp[side.active] > 0:
        fx.notify(ctx, "switch_in", ref)


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def _speed(ctx: Context, player: int) -> int:
    side = ctx.state.sides[player]
    return mutate.effective_stat(ctx, (player, side.active), Stat.SPE)


def _by_speed(ctx: Context, players: list[int]) -> list[int]:
    """Fastest first -- or slowest first under Trick Room."""
    if len(players) < 2:
        return players
    a, b = players
    speed_a, speed_b = _speed(ctx, a), _speed(ctx, b)
    if speed_a == speed_b:
        return [a, b] if ctx.cursor.chance(1, 2) else [b, a]

    faster_first = speed_a > speed_b
    if "trickroom" in ctx.state.field.rooms:
        faster_first = not faster_first
    return [a, b] if faster_first else [b, a]


def _move_order(ctx: Context, actions: tuple[Action, ...], players: list[int]) -> list[int]:
    """Priority first, then Speed. Trick Room reverses Speed but not priority."""
    if len(players) < 2:
        return players
    a, b = players
    priority_a = _priority(ctx, a, actions[a])
    priority_b = _priority(ctx, b, actions[b])
    if priority_a != priority_b:
        return [a, b] if priority_a > priority_b else [b, a]
    return _by_speed(ctx, players)


def _priority(ctx: Context, player: int, action: Action) -> int:
    move = _chosen_move(ctx.state, player, action)
    ref: Ref = (player, ctx.state.sides[player].active)
    return fx.modify(ctx, "modify_priority", move.priority, ref, scope="self", move=move)


def _chosen_move(state: BattleState, player: int, action: Action) -> Move:
    if action.kind is ActionKind.STRUGGLE:
        return state.config.dex.moves[STRUGGLE_ID]
    side = state.sides[player]
    return state.moves(player, side.active)[action.index]


def _use(ctx: Context, player: int, action: Action) -> None:
    side = ctx.state.sides[player]
    attacker: Ref = (player, side.active)
    opponent = 1 - player
    defender: Ref = (opponent, ctx.state.sides[opponent].active)
    move = _chosen_move(ctx.state, player, action)
    index = action.index if action.kind is ActionKind.MOVE else None
    mv.use_move(ctx, attacker, defender, move, index)


# --------------------------------------------------------------------------- #
# End of turn
# --------------------------------------------------------------------------- #


def _end_of_turn(ctx: Context) -> None:
    state = ctx.state
    if state.finished:
        return

    for player in _by_speed(ctx, [0, 1]):
        side = state.sides[player]
        if side.is_fainted(side.active):
            continue
        fx.notify(ctx, "residual", (player, side.active))
        mutate.check_item_triggers(ctx, (player, side.active))
        if _check_loss(ctx):
            return

    _tick_field(ctx)
    _clear_turn_volatiles(ctx)

    needs_switch = False
    for player in (0, 1):
        side = state.sides[player]
        if side.is_fainted(side.active):
            side.must_switch = True
            needs_switch = True

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
            volatiles.pop("lastmove", None) if slot != side.active else None


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
    """Timer-out ruling: more Pokemon standing, then more total HP, else a draw."""
    remaining = [len(side.living_slots()) for side in state.sides]
    if remaining[0] != remaining[1]:
        return 0 if remaining[0] > remaining[1] else 1

    fractions = []
    for player, side in enumerate(state.sides):
        fractions.append(
            sum(side.hp[slot] / state.pokemon(player, slot).max_hp for slot in range(len(side.hp)))
        )
    if fractions[0] == fractions[1]:
        return None
    return 0 if fractions[0] > fractions[1] else 1


def _finish(ctx: Context, winner: int | None, detail: str) -> None:
    ctx.state.phase = Phase.FINISHED
    ctx.state.winner = winner
    for side in ctx.state.sides:
        side.must_switch = False
    ctx.emit(ev.battle_end(winner, detail))


#: Kept importable from here because it reads as part of the battle's surface.
compute_damage = mv.compute_damage
