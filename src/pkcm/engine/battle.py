"""Turn resolution: the pure ``step`` at the center of everything.

M0 scope. Implemented: team preview, switching, move order, the damage formula,
accuracy, criticals, the damage roll, STAB, type effectiveness, PP, Struggle,
fainting, forced replacement, and win conditions. Deliberately absent: abilities,
items, status conditions, stat stages, weather, and every secondary effect.
Those arrive in M1-M4 through the event-hook system (docs/DESIGN.md §1d), and the
resolution order here is laid out so they can slot in without a rewrite.

``step`` is pure: it clones, mutates the clone, and returns it. The RNG is opened
as a mutable cursor for the duration of the step and sealed back into the
returned state, so purity holds exactly where search needs it -- at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

from pkcm.data.dex import Move, Stat, TypeChart
from pkcm.engine import events as ev
from pkcm.engine.actions import Action, ActionKind
from pkcm.engine.events import Event
from pkcm.engine.pokemon import BattlePokemon
from pkcm.engine.rng import RngCursor
from pkcm.engine.scope import move_support
from pkcm.engine.state import BattleState, Phase, legal_actions

LEVEL = 50

CRIT_DENOMINATOR = 24  # Gen 7+ stage-0 critical hit rate
CRIT_MULTIPLIER_NUM, CRIT_MULTIPLIER_DEN = 3, 2
STAB_NUM, STAB_DEN = 3, 2
DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH = 85, 100

STRUGGLE_ID = "struggle"
STRUGGLE_RECOIL_FRACTION = 4  # 1/4 of max HP, Gen 5+


class IllegalActionError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Damage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class DamageResult:
    amount: int
    crit: bool
    effectiveness: float

    @property
    def immune(self) -> bool:
        return self.effectiveness == 0.0


def compute_damage(
    attacker: BattlePokemon,
    defender: BattlePokemon,
    move: Move,
    cursor: RngCursor,
    chart: TypeChart,
) -> DamageResult:
    """The Gen 5+ damage formula, in its documented order.

    Every step floors, and the order the floors happen in is observable -- it is
    why the same matchup can roll a different number of hits to KO. Modifiers
    that M0 does not implement (weather, screens, burn, items) would slot in at
    their own points in this chain rather than being multiplied in at the end.
    """
    # Struggle is typeless: neither STAB nor type effectiveness apply to it.
    typeless = move.id == STRUGGLE_ID
    effectiveness = 1.0 if typeless else chart.multiplier(move.type, defender.types)
    if effectiveness == 0.0:
        return DamageResult(0, False, 0.0)

    if move.category == "Physical":
        attack, defense = attacker.stats[Stat.ATK], defender.stats[Stat.DEF]
    else:
        attack, defense = attacker.stats[Stat.SPA], defender.stats[Stat.SPD]

    damage = ((2 * LEVEL // 5 + 2) * move.base_power * attack // defense) // 50 + 2

    crit = cursor.chance(1, CRIT_DENOMINATOR)
    if crit:
        damage = damage * CRIT_MULTIPLIER_NUM // CRIT_MULTIPLIER_DEN

    damage = damage * cursor.between(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH) // 100

    if not typeless and move.type in attacker.types:
        damage = damage * STAB_NUM // STAB_DEN

    damage = int(damage * effectiveness)
    return DamageResult(max(1, damage), crit, effectiveness)


# --------------------------------------------------------------------------- #
# Step
# --------------------------------------------------------------------------- #


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
    cursor = next_state.rng.cursor()
    log: list[Event] = []

    if state.phase is Phase.TEAM_PREVIEW:
        _resolve_team_preview(next_state, actions, log)
    elif state.phase is Phase.FORCED_SWITCH:
        _resolve_forced_switch(next_state, actions, cursor, log)
    elif state.phase is Phase.BATTLE:
        _resolve_turn(next_state, actions, cursor, log)

    next_state.rng = cursor.seal()
    return next_state, log


def _resolve_team_preview(state: BattleState, actions: tuple[Action, ...], log: list[Event]) -> None:
    for player, action in enumerate(actions):
        side = state.sides[player]
        side.selection = action.selection
        side.hp = [state.parties[player][i].stats[Stat.HP] for i in action.selection]
        side.pp = [list(state.parties[player][i].max_pp) for i in action.selection]
        side.active = 0
        log.append(ev.team_preview(player, action.selection))

    for player in (0, 1):
        _emit_switch_in(state, player, log)

    state.phase = Phase.BATTLE


def _resolve_forced_switch(
    state: BattleState,
    actions: tuple[Action, ...],
    cursor: RngCursor,
    log: list[Event],
) -> None:
    for player, action in enumerate(actions):
        if action.kind is not ActionKind.SWITCH:
            continue
        side = state.sides[player]
        side.active = action.index
        side.must_switch = False
        _emit_switch_in(state, player, log)

    state.phase = Phase.BATTLE


def _resolve_turn(
    state: BattleState,
    actions: tuple[Action, ...],
    cursor: RngCursor,
    log: list[Event],
) -> None:
    state.turn += 1
    log.append(ev.turn_start(state.turn))

    switchers = [p for p in (0, 1) if actions[p].kind is ActionKind.SWITCH]
    attackers = [p for p in (0, 1) if actions[p].kind in (ActionKind.MOVE, ActionKind.STRUGGLE)]

    # Switches always resolve before any move, fastest first.
    for player in _by_speed(state, switchers, cursor):
        state.sides[player].active = actions[player].index
        _emit_switch_in(state, player, log)

    for player in _move_order(state, actions, attackers, cursor):
        if state.finished:
            return
        side = state.sides[player]
        if side.is_fainted(side.active):
            continue  # knocked out before it could act
        _execute_move(state, player, actions[player], cursor, log)

    _end_of_turn(state, log)


def _emit_switch_in(state: BattleState, player: int, log: list[Event]) -> None:
    side = state.sides[player]
    pokemon = state.pokemon(player, side.active)
    log.append(
        ev.switch_in(player, side.active, pokemon.species.name, side.hp[side.active], pokemon.max_hp)
    )


def _by_speed(state: BattleState, players: list[int], cursor: RngCursor) -> list[int]:
    if len(players) < 2:
        return players
    a, b = players
    if state.speed(a) != state.speed(b):
        return [a, b] if state.speed(a) > state.speed(b) else [b, a]
    return [a, b] if cursor.chance(1, 2) else [b, a]


def _move_order(
    state: BattleState,
    actions: tuple[Action, ...],
    players: list[int],
    cursor: RngCursor,
) -> list[int]:
    """Priority first, then Speed, then a coin flip for the speed tie."""
    if len(players) < 2:
        return players
    a, b = players
    priority_a = _priority(state, a, actions[a])
    priority_b = _priority(state, b, actions[b])
    if priority_a != priority_b:
        return [a, b] if priority_a > priority_b else [b, a]
    return _by_speed(state, players, cursor)


def _priority(state: BattleState, player: int, action: Action) -> int:
    if action.kind is ActionKind.STRUGGLE:
        return 0
    return _chosen_move(state, player, action).priority


def _chosen_move(state: BattleState, player: int, action: Action) -> Move:
    if action.kind is ActionKind.STRUGGLE:
        return state.config.dex.moves[STRUGGLE_ID]
    return state.active_pokemon(player).moves[action.index]


def _execute_move(
    state: BattleState,
    player: int,
    action: Action,
    cursor: RngCursor,
    log: list[Event],
) -> None:
    side = state.sides[player]
    attacker = state.active_pokemon(player)
    move = _chosen_move(state, player, action)

    if action.kind is ActionKind.MOVE:
        side.pp[side.active][action.index] -= 1

    log.append(ev.move_used(player, side.active, attacker.species.name, move.name))

    opponent = 1 - player
    other = state.sides[opponent]
    defender = state.active_pokemon(opponent)

    if move.accuracy is not None and not cursor.percent(move.accuracy):
        log.append(ev.missed(player, side.active, move.name))
        return

    # M0 executes damaging, single-hit, fixed-power moves and nothing else. A set
    # carrying anything else still plays out -- it resolves as a no-op -- but the
    # log says exactly which mechanic was skipped rather than silently pretending
    # the move did nothing. The random generator avoids these entirely.
    unsupported = move_support(move)
    if unsupported is not None:
        log.append(
            Event("unimplemented", side=player, slot=side.active, move=move.name,
                  detail=unsupported)
        )
        return

    result = compute_damage(attacker, defender, move, cursor, state.config.dex.type_chart)
    if result.immune:
        log.append(ev.immune(opponent, other.active, move.name))
        return

    dealt = min(result.amount, other.hp[other.active])
    other.hp[other.active] -= dealt
    log.append(
        ev.damage(
            opponent, other.active, dealt, other.hp[other.active],
            defender.max_hp, result.effectiveness, result.crit,
        )
    )
    _check_faint(state, opponent, log)

    if move.id == STRUGGLE_ID:
        _apply_struggle_recoil(state, player, log)


def _apply_struggle_recoil(state: BattleState, player: int, log: list[Event]) -> None:
    side = state.sides[player]
    attacker = state.active_pokemon(player)
    amount = max(1, attacker.max_hp // STRUGGLE_RECOIL_FRACTION)
    amount = min(amount, side.hp[side.active])
    side.hp[side.active] -= amount
    log.append(ev.recoil(player, side.active, amount, side.hp[side.active], attacker.max_hp))
    _check_faint(state, player, log)


def _check_faint(state: BattleState, player: int, log: list[Event]) -> None:
    side = state.sides[player]
    if side.hp[side.active] > 0:
        return
    log.append(ev.faint(player, side.active, state.pokemon(player, side.active).species.name))
    if side.has_lost():
        _finish(state, winner=1 - player, detail="all Pokemon fainted", log=log)


def _end_of_turn(state: BattleState, log: list[Event]) -> None:
    if state.finished:
        return

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
        _finish(state, winner=_decide_by_attrition(state), detail="turn limit", log=log)


def _decide_by_attrition(state: BattleState) -> int | None:
    """Timer-out ruling: more Pokemon standing, then more total HP, else a draw."""
    remaining = [len(side.living_slots()) for side in state.sides]
    if remaining[0] != remaining[1]:
        return 0 if remaining[0] > remaining[1] else 1

    fractions = []
    for player, side in enumerate(state.sides):
        total = sum(
            side.hp[slot] / state.pokemon(player, slot).max_hp for slot in range(len(side.hp))
        )
        fractions.append(total)
    if fractions[0] == fractions[1]:
        return None
    return 0 if fractions[0] > fractions[1] else 1


def _finish(state: BattleState, winner: int | None, detail: str, log: list[Event]) -> None:
    state.phase = Phase.FINISHED
    state.winner = winner
    for side in state.sides:
        side.must_switch = False
    log.append(ev.battle_end(winner, detail))
