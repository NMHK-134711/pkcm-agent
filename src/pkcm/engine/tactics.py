"""Moves whose effect is a shape the declarative data cannot express.

These are not exotic. Forcing a switch beats a Pokemon that has spent the turn
setting up; U-turn is how momentum works at all; Counter and Mirror Coat punish
the attack you were going to take anyway. Leaving them out would leave a policy
blind to a whole layer of the game.

What they have in common is that none of them fit ``boosts``/``status``/
``secondary``. Each needs the turn loop to do something structural: replace the
target, suspend itself, look back at damage already dealt, or carry a decision
across turns.
"""

from __future__ import annotations

from pkcm.data.dex import Move, Stat
from pkcm.engine import mutate
from pkcm.engine.effects import Context, Ref, register
from pkcm.engine.events import Event
from pkcm.engine.moves import chain_modify, X1_5

# --------------------------------------------------------------------------- #
# Damage that answers damage
# --------------------------------------------------------------------------- #

#: ``move id -> (which category it answers, how much to return, in halves)``.
#: ``None`` for the category means "whatever hit last, either kind".
COUNTER_MOVES: dict[str, tuple[str | None, int]] = {
    "counter": ("Physical", 4),   # 2x
    "mirrorcoat": ("Special", 4),  # 2x
    "metalburst": (None, 3),      # 1.5x
    "comeuppance": (None, 3),     # 1.5x
}


def record_hit(ctx: Context, ref: Ref, attacker: Ref, move, damage: int) -> None:
    """Remember what just hit us, for Counter and friends to answer."""
    if damage <= 0 or move.category == "Status":
        return
    ledger = ctx.state.sides[ref[0]].volatiles[ref[1]].setdefault("hurtthisturn", {})
    ledger[move.category] = ledger.get(move.category, 0) + damage
    ledger["last"] = damage
    ledger["source"] = attacker
    # Rage Fist counts hits taken across the whole battle, switches included,
    # which is why this lives in status_data rather than the volatiles.
    counters = ctx.state.sides[ref[0]].status_data[ref[1]]
    counters["timeshit"] = counters.get("timeshit", 0) + 1


def counter_damage(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int | None:
    """How much a Counter-like move deals, or ``None`` if it fails."""
    category, halves = COUNTER_MOVES[move.id]
    ledger = ctx.state.sides[attacker[0]].volatiles[attacker[1]].get("hurtthisturn")
    if not ledger:
        return None
    taken = ledger.get(category) if category else ledger.get("last")
    if not taken:
        return None
    return max(1, taken * halves // 2)


def endeavor_damage(ctx: Context, attacker: Ref, defender: Ref) -> int | None:
    """Brings the target down to the user's own HP. Fails if it is already lower."""
    mine = mutate.current_hp(ctx.state, attacker)
    theirs = mutate.current_hp(ctx.state, defender)
    return theirs - mine if theirs > mine else None


# --------------------------------------------------------------------------- #
# Forcing a switch
# --------------------------------------------------------------------------- #


def force_switch(ctx: Context, target: Ref) -> bool:
    """Drag in a random party member. Fails when there is nobody to drag in."""
    side = ctx.state.sides[target[0]]
    candidates = [slot for slot in side.living_slots() if slot not in side.active]
    if not candidates:
        return False
    if side.has_volatile(target[1], "ingrain"):
        return False
    position = side.position_of(target[1])
    if position is None:
        return False

    from pkcm.engine.battle import switch_into

    chosen = ctx.cursor.choice(candidates)
    ctx.emit(Event("dragged_out", side=target[0], slot=target[1]))
    switch_into(ctx, target[0], position, chosen)
    return True


def self_switch(ctx: Context, user: Ref) -> bool:
    """U-turn and friends: the user leaves, and the player picks who replaces it.

    Marking ``must_switch`` is what makes the turn loop suspend -- the
    replacement has to be on the field before the opponent moves, which is the
    whole point of the move.
    """
    side = ctx.state.sides[user[0]]
    if not [slot for slot in side.living_slots() if slot not in side.active]:
        return False
    position = side.position_of(user[1])
    if position is None:
        return False
    side.must_switch[position] = True
    ctx.emit(Event("self_switch", side=user[0], slot=user[1]))
    return True


# --------------------------------------------------------------------------- #
# Two-turn moves
# --------------------------------------------------------------------------- #

#: Charging moves that also take the user off the field. The value is the flag
#: an attack needs to reach them anyway (Earthquake hits a Pokemon underground).
SEMI_INVULNERABLE = {
    "fly": "gravity",
    "bounce": "gravity",
    "dig": "underground",
    "dive": "underwater",
    "phantomforce": None,
    "shadowforce": None,
    "skydrop": "gravity",
}

#: Moves that reach a Pokemon in the matching semi-invulnerable state.
REACHES = {
    "gravity": {"gust", "twister", "thunder", "hurricane", "smackdown", "skyuppercut"},
    "underground": {"earthquake", "magnitude", "fissure"},
    "underwater": {"surf", "whirlpool"},
}


def is_charging(ctx: Context, ref: Ref) -> dict | None:
    return mutate.volatile(ctx.state, ref, "twoturn")


def start_charging(ctx: Context, user: Ref, move, move_index: int | None) -> None:
    data = {"move": move_index, "id": move.id}
    ctx.state.sides[user[0]].volatiles[user[1]]["twoturn"] = data
    if move.id in SEMI_INVULNERABLE:
        ctx.state.sides[user[0]].volatiles[user[1]]["invulnerable"] = {"move": move.id}
    ctx.emit(Event("charging", side=user[0], slot=user[1], move=move.id))


def finish_charging(ctx: Context, user: Ref) -> None:
    mutate.remove_volatile(ctx, user, "twoturn", quiet=True)
    mutate.remove_volatile(ctx, user, "invulnerable", quiet=True)


def _invulnerable_blocks(ctx, ref, attacker, defender, move, **_):
    if ref != defender:
        return None
    data = mutate.volatile(ctx.state, defender, "invulnerable")
    if data is None:
        return None
    needed = SEMI_INVULNERABLE.get(data["move"])
    if needed is not None and move.id in REACHES.get(needed, ()):
        return None
    ctx.emit(Event("avoided", side=defender[0], slot=defender[1], move=move.id))
    return False


register("volatile", "invulnerable", name="Semi-invulnerable", try_hit=_invulnerable_blocks)
register("volatile", "twoturn", name="Charging")


def _recharging(ctx, ref, move, **_):
    ctx.emit(Event("recharging", side=ref[0], slot=ref[1]))
    mutate.remove_volatile(ctx, ref, "mustrecharge", quiet=True)
    return False


register("volatile", "mustrecharge", name="Recharging", try_move=_recharging)


# --------------------------------------------------------------------------- #
# Locking moves: Outrage, and the trapping moves
# --------------------------------------------------------------------------- #


def start_locked_move(ctx: Context, user: Ref, move, move_index: int | None) -> None:
    """Outrage and friends: 2-3 turns of the same move, then confusion."""
    volatiles = ctx.state.sides[user[0]].volatiles[user[1]]
    data = volatiles.get("lockedmove")
    if data is None or "turns" not in data:
        volatiles["lockedmove"] = {"move": move_index, "id": move.id,
                                   "turns": ctx.cursor.between(2, 3) - 1}
        return
    data["turns"] -= 1
    if data["turns"] <= 0:
        del volatiles["lockedmove"]
        mutate.add_volatile(ctx, user, "confusion", turns=ctx.cursor.between(2, 5))


register("volatile", "lockedmove", name="Locked in")

TRAP_FRACTION = 8


def start_trapping(ctx: Context, target: Ref, move) -> None:
    if mutate.volatile(ctx.state, target, "partiallytrapped") is not None:
        return
    turns = 5 if ctx.state.item_id(*target) == "gripclaw" else ctx.cursor.choice((4, 4, 5, 5, 6, 7))
    mutate.add_volatile(ctx, target, "partiallytrapped", move=move.id, turns=turns)
    mutate.add_volatile(ctx, target, "trapped")


def _trapping_residual(ctx, ref, **_):
    data = mutate.volatile(ctx.state, ref, "partiallytrapped")
    if data is None:
        return
    data["turns"] -= 1
    if data["turns"] <= 0:
        mutate.remove_volatile(ctx, ref, "partiallytrapped")
        mutate.remove_volatile(ctx, ref, "trapped", quiet=True)
        return
    mutate.apply_damage(ctx, ref, mutate.fraction_of_max(ctx.state, ref, TRAP_FRACTION),
                        "status_damage", detail=data["move"])


register("volatile", "partiallytrapped", name="Trapped by a move",
         residual=_trapping_residual)


# --------------------------------------------------------------------------- #
# Moves that cost the user everything
# --------------------------------------------------------------------------- #

#: ``move id -> what happens to the user``. ``"faint"`` is the whole effect for
#: Explosion; the others hand something over first.
SELF_DESTRUCT_MOVES = {
    "explosion": "faint",
    "selfdestruct": "faint",
    "mistyexplosion": "faint",
    "memento": "memento",
    "healingwish": "healingwish",
    "finalgambit": "finalgambit",
}


def self_destruct(ctx: Context, user: Ref, defender: Ref, move) -> None:
    kind = SELF_DESTRUCT_MOVES[move.id]

    # Memento's -2/-2 comes from the move data; re-applying it here would
    # double it. Only the fainting is ours to do.
    if kind == "finalgambit":
        remaining = mutate.current_hp(ctx.state, user)
        mutate.apply_damage(ctx, defender, remaining, "damage",
                            move=move.id, effectiveness=1.0, __source__=user, __move__=move)
    elif kind == "healingwish":
        ctx.state.sides[user[0]].conditions["healingwish"] = 1
        ctx.emit(Event("side_condition", side=user[0], detail="healingwish", amount=1))

    ctx.emit(Event("self_destruct", side=user[0], slot=user[1], move=move.id))
    mutate.apply_damage(ctx, user, mutate.current_hp(ctx.state, user), "damage",
                        detail=move.id)


def _healing_wish_on_entry(ctx: Context, ref: Ref) -> None:
    """The wish is spent on whoever comes in next."""
    conditions = ctx.state.sides[ref[0]].conditions
    if "healingwish" not in conditions:
        return
    if mutate.current_hp(ctx.state, ref) == mutate.max_hp(ctx.state, ref):
        if ctx.state.sides[ref[0]].status[ref[1]] is None:
            return
    del conditions["healingwish"]
    mutate.heal(ctx, ref, mutate.max_hp(ctx.state, ref), reason="healingwish")
    mutate.cure_status(ctx, ref)


register("side", "healingwish", name="Healing Wish")
register("volatile", "ingrain", name="Ingrain")
