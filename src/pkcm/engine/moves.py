"""Executing a move, driven by the move data rather than by hand-written cases.

Showdown's move table is largely *declarative*: Swords Dance carries
``boosts: {atk: 2}``, Thunderbolt carries ``secondary: {chance: 10, status: par}``,
Giga Drain carries ``drain: [1, 2]``, Bullet Seed carries ``multihit: [2, 5]``.
One executor that understands those fields covers most of the move list at once,
which is why this file implements mechanics in bulk instead of move by move.

What genuinely cannot be read off the data -- moves whose power is computed from
battle state -- lives in ``VARIABLE_POWER``, keyed by move id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pkcm.data.dex import Move, Stat
from pkcm.engine import effects as fx
from pkcm.engine import events as ev
from pkcm.engine import mutate
from pkcm.engine.effects import Context, Ref
from pkcm.engine.events import Event
from pkcm.engine.mutate import apply_damage, effective_stat, heal, stage_multiplier
from pkcm.engine.actions import TARGET_ALLY, TARGET_SELF
from pkcm.engine.state import imprisoned_moves

LEVEL = 50

#: Showdown accumulates every damage multiplier in 4096ths and rounds half up
#: (``Battle#modify``). Using floats instead is off by a point often enough to
#: change how many hits a KO takes, so the magic numbers below are the ones the
#: source actually uses: 1.3 is 5325/4096, not 1.3.
MODIFIER_SCALE = 4096
X0_25, X0_5, X0_75, X0_9 = 1024, 2048, 3072, 3686
X1_1, X1_2, X1_25, X1_3, X1_5 = 4506, 4915, 5120, 5325, 6144
X2 = 8192


def chain_modify(value: int, modifier: int) -> int:
    """``Battle#modify``: apply a 4096ths modifier with Showdown's rounding."""
    return (value * modifier + MODIFIER_SCALE // 2 - 1) // MODIFIER_SCALE


STAB_NUM, STAB_DEN = 3, 2
DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH = 85, 100
CRIT_MULTIPLIER_NUM, CRIT_MULTIPLIER_DEN = 3, 2

#: Gen 7+ critical hit rate by crit ratio: the denominator of a 1/N chance.
CRIT_DENOMINATOR = {0: 24, 1: 24, 2: 8, 3: 2}
#: A crit ratio below zero means no critical hit is possible at all. Shell
#: Armor and Battle Armor return it. Zero will not do: it is a real ratio worth
#: 1/24, which is what those two were quietly granting.
NEVER_CRITS = -1

STRUGGLE_ID = "struggle"
STRUGGLE_RECOIL_FRACTION = 4

#: Distribution for a 2-5 hit move: **35-35-15-15** across 2, 3, 4 and 5 hits,
#: which is what Gen 5 onward uses (mods/champions/scripts.ts, hitStepMoveHitLoop).
#: The 3/8-3/8-1/8-1/8 spread that reads so much tidier is the Gen 4 one, and
#: using it makes every multi-hit move land 3.0 times instead of 3.1.
MULTIHIT_2_TO_5 = (2,) * 7 + (3,) * 7 + (4,) * 3 + (5,) * 3

#: Targets that never point at the opposing active Pokemon.
SELF_TARGETS = frozenset({"self", "adjacentAlly", "adjacentAllyOrSelf", "allies", "allySide"})
FIELD_TARGETS = frozenset({"all"})
FOE_SIDE_TARGETS = frozenset({"foeSide"})

#: Moves that hit more than one Pokemon at once. Each one takes a quarter off
#: its damage in doubles -- and only when it actually lands on more than one,
#: which is why the count is taken at resolution rather than read off the data.
SPREAD_TARGETS = frozenset({"allAdjacentFoes", "allAdjacent"})
#: 0.75 in 4096ths, applied once for a spread move with two or more targets.
SPREAD_MODIFIER = X0_75

#: Targets that mean "one Pokemon on the other side". These are the ones a
#: caller can name outright with ``use_move(defender=...)``.
SINGLE_FOE_TARGETS = frozenset({"normal", "any", "adjacentFoe", "randomNormal", "scripted"})


# --------------------------------------------------------------------------- #
# Base power
# --------------------------------------------------------------------------- #


def _weight_based(thresholds: tuple[tuple[float, int], ...]) -> Callable:
    def compute(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
        weight = mutate.weight_kg(ctx, defender)
        for limit, power in thresholds:
            if weight < limit:
                return power
        return thresholds[-1][1]

    return compute


def _relative_weight(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    ratio = mutate.weight_kg(ctx, attacker) / mutate.weight_kg(ctx, defender)
    for limit, power in ((2, 40), (3, 60), (4, 80), (5, 100)):
        if ratio < limit:
            return power
    return 120


def _gyro_ball(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    mine = max(1, effective_stat(ctx, attacker, Stat.SPE))
    theirs = effective_stat(ctx, defender, Stat.SPE)
    return max(1, min(150, 25 * theirs // mine))


def _electro_ball(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    mine = effective_stat(ctx, attacker, Stat.SPE)
    theirs = max(1, effective_stat(ctx, defender, Stat.SPE))
    ratio = mine / theirs
    for limit, power in ((1, 40), (2, 60), (3, 80), (4, 120)):
        if ratio < limit:
            return power
    return 150


def _low_hp_scaling(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    fraction = mutate.current_hp(ctx.state, attacker) * 48 // mutate.max_hp(ctx.state, attacker)
    for limit, power in ((2, 200), (5, 150), (10, 100), (17, 80), (33, 40)):
        if fraction < limit:
            return power
    return 20


def _target_hp_scaling(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    """Crush Grip / Wring Out: stronger the healthier the target is."""
    ratio = mutate.current_hp(ctx.state, defender) / mutate.max_hp(ctx.state, defender)
    return max(1, int(120 * ratio))


# --------------------------------------------------------------------------- #
# Moves that read the terrain
#
# All six are damaging moves, so ``move_support`` waved them through -- it only
# catches *status* moves with no declarative payload. They landed their damage
# and skipped everything that makes them worth using. A damaging move whose
# conditional effect lives in handler code is a blind spot that check cannot
# see, which is why they sat here working-but-wrong.
#
# Each also requires its user to be *grounded*: terrain does not reach a Flying
# type or a Levitate holder, and neither do these.
# --------------------------------------------------------------------------- #


def _stands_on(ctx: Context, ref: Ref, terrain: str) -> bool:
    from pkcm.engine.conditions import is_grounded

    return ctx.state.field.terrain == terrain and is_grounded(ctx.state, ref, ctx)


def _boosted_on(terrain: str, modifier: int):
    """Expanding Force and Misty Explosion: half again on the right ground."""
    def compute(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
        power = move.base_power
        return chain_modify(power, modifier) if _stands_on(ctx, attacker, terrain) else power

    return compute


def _rising_voltage(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    """Double -- and it is the *target* that has to be on the ground."""
    from pkcm.engine.conditions import is_grounded

    if ctx.state.field.terrain == "electricterrain" and is_grounded(ctx.state, defender, ctx):
        return move.base_power * 2
    return move.base_power


def _terrain_pulse_power(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    from pkcm.engine.conditions import is_grounded

    if ctx.state.field.terrain and is_grounded(ctx.state, attacker, ctx):
        return move.base_power * 2
    return move.base_power


#: Terrain -> the type Terrain Pulse takes while it is up.
TERRAIN_PULSE_TYPES = {
    "electricterrain": "electric",
    "grassyterrain": "grass",
    "mistyterrain": "fairy",
    "psychicterrain": "psychic",
}


def _rewrite_for_terrain(ctx: Context, active, attacker: Ref) -> None:
    """The two rewrites that must happen before the move resolves.

    Expanding Force stops being single-target on Psychic Terrain, which changes
    who it hits *and* costs it the spread quarter. Terrain Pulse changes type,
    which changes STAB and the type chart. Neither is about power, so neither
    belongs in the table above.
    """
    if active.id == "expandingforce" and _stands_on(ctx, attacker, "psychicterrain"):
        active.target = "allAdjacentFoes"
    elif active.id == "terrainpulse":
        from pkcm.engine.conditions import is_grounded

        terrain = ctx.state.field.terrain
        if terrain in TERRAIN_PULSE_TYPES and is_grounded(ctx.state, attacker, ctx):
            active.type = TERRAIN_PULSE_TYPES[terrain]


def _grassy_glide(ctx: Context, ref: Ref, priority: int) -> int:
    return priority + 1 if _stands_on(ctx, ref, "grassyterrain") else priority


#: Moves whose priority depends on the field. Consulted by ``battle._priority``
#: alongside the ability and item hooks.
MOVE_PRIORITY: dict[str, Callable[[Context, Ref, int], int]] = {
    "grassyglide": _grassy_glide,
}


def _needs_terrain(ctx: Context, ref: Ref, move: Move) -> str | None:
    return "no terrain" if ctx.state.field.terrain is None else None


#: Moves that refuse to run under some condition, checked before PP is spent.
#: The string is the reason, and it reaches the log.
MOVE_PRECONDITIONS: dict[str, Callable[[Context, Ref, Move], str | None]] = {
    "steelroller": _needs_terrain,
}


#: Moves whose base power depends on the battle rather than on a constant.
VARIABLE_POWER: dict[str, Callable[[Context, Ref, Ref, Move], int]] = {
    "expandingforce": _boosted_on("psychicterrain", X1_5),
    "mistyexplosion": _boosted_on("mistyterrain", X1_5),
    "risingvoltage": _rising_voltage,
    "terrainpulse": _terrain_pulse_power,
    "lowkick": _weight_based(((10, 20), (25, 40), (50, 60), (100, 80), (200, 100), (float("inf"), 120))),
    "grassknot": _weight_based(((10, 20), (25, 40), (50, 60), (100, 80), (200, 100), (float("inf"), 120))),
    "heavyslam": _relative_weight,
    "heatcrash": _relative_weight,
    "gyroball": _gyro_ball,
    "electroball": _electro_ball,
    "flail": _low_hp_scaling,
    "reversal": _low_hp_scaling,
    "crushgrip": _target_hp_scaling,
    "wringout": _target_hp_scaling,
    "hardpress": _target_hp_scaling,
}


def base_power(ctx: Context, attacker: Ref, defender: Ref, move) -> int:
    from pkcm.engine import moveeffects

    if move.id == "fling":
        return moveeffects.fling_power(ctx, attacker) or 0
    if move.id == "spitup":
        return moveeffects.spit_up_power(ctx, attacker)
    if move.id == "beatup":
        powers = moveeffects.beat_up_hits(ctx, attacker)
        # Each hit uses a different team mate's Attack; the hit loop tracks
        # which one, and stores it on the active move.
        index = getattr(move, "hit_index", 0)
        return powers[index] if index < len(powers) else 0

    computed = VARIABLE_POWER.get(move.id)
    if computed is not None:
        return computed(ctx, attacker, defender, move)
    return move.base_power


# --------------------------------------------------------------------------- #
# Damage
# --------------------------------------------------------------------------- #


def type_effectiveness(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> float:
    if move.id == STRUGGLE_ID:
        return 1.0

    # Ground moves cannot reach anything airborne. Showdown does this in
    # Pokemon#isGrounded rather than as a Levitate handler, which is why
    # Levitate's entry in abilities.ts has no handlers at all.
    if move.type == "ground" and move.category != "Status":
        from pkcm.engine.conditions import is_grounded

        if not is_grounded(ctx.state, defender, ctx=ctx):
            return 0.0

    value = ctx.state.config.dex.type_chart.multiplier(move.type, ctx.state.types(*defender))
    return _both_sides(ctx, "modify_effectiveness", value, attacker, defender, move)


def _both_sides(ctx: Context, event: str, value: Any, attacker: Ref, defender: Ref, move: Move) -> Any:
    """Run a hook from the attacker's effects, then the defender's and the field.

    Gathering the field from the defender's side only is deliberate: weather
    would otherwise apply twice.
    """
    value = fx.modify(ctx, event, value, attacker, scope="self",
                      attacker=attacker, defender=defender, move=move)
    return fx.modify(ctx, event, value, defender, scope="all",
                     attacker=attacker, defender=defender, move=move)


def damage_formula(
    *,
    power: int,
    attack: int,
    defense: int,
    roll: int,
    crit: bool = False,
    spread: bool = False,
    stab: bool = False,
    effectiveness: float = 1.0,
    level: int = LEVEL,
) -> int:
    """The arithmetic, with no battle attached.

    Pulled out of ``compute_damage`` so the damage *estimator* the agent gets
    to use (``pkcm.envs.analysis``) runs the same steps in the same order. Two
    copies of a formula that truncates at every step do not stay equal, and the
    estimate would drift away from the engine one roll at a time.

    Everything that needs a battle -- which stats, whose ability, what the
    terrain says -- is settled by the caller. This is only the order of
    operations, and the order is visible in the result because each step floors.
    """
    damage = ((2 * level // 5 + 2) * power * attack // defense) // 50 + 2

    # Showdown applies the spread penalty here -- before the crit multiplier and
    # before the roll, not at the end (``Battle#modifyDamage``).
    if spread:
        damage = chain_modify(damage, SPREAD_MODIFIER)
    if crit:
        damage = damage * CRIT_MULTIPLIER_NUM // CRIT_MULTIPLIER_DEN
    damage = damage * roll // 100
    if stab:
        damage = damage * STAB_NUM // STAB_DEN
    return int(damage * effectiveness)


def compute_damage(
    ctx: Context,
    attacker: Ref,
    defender: Ref,
    move: Move,
    crit: bool,
) -> tuple[int, float]:
    """The Gen 5+ formula in its documented order. Returns (damage, effectiveness)."""
    effectiveness = type_effectiveness(ctx, attacker, defender, move)
    if effectiveness == 0.0:
        return 0, 0.0

    power = base_power(ctx, attacker, defender, move)
    if power <= 0:
        return 0, effectiveness
    # A distinct pass from modify_damage, and it has to be: Technician's test is
    # on the power *after* the other base-power modifiers have run.
    power = max(1, _both_sides(ctx, "modify_base_power", power, attacker, defender, move))

    if move.category == "Physical":
        attack_stat, defense_stat = Stat.ATK, Stat.DEF
    else:
        attack_stat, defense_stat = Stat.SPA, Stat.SPD

    attack = effective_stat(ctx, attacker, attack_stat, move=move, opponent=defender)
    defense = effective_stat(ctx, defender, defense_stat, move=move, opponent=attacker)

    # A critical hit ignores the defender's positive stages and the attacker's
    # negative ones -- it recomputes without the stages that would have helped
    # the defender or hurt the attacker.
    if crit:
        attack = max(attack, _unstaged(ctx, attacker, attack_stat, move, defender, True))
        defense = min(defense, _unstaged(ctx, defender, defense_stat, move, attacker, False))

    damage = damage_formula(
        power=power,
        attack=attack,
        defense=defense,
        roll=ctx.cursor.between(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH),
        crit=crit,
        spread=bool(getattr(move, "spread", False)),
        stab=move.id != STRUGGLE_ID and move.type in ctx.state.types(*attacker),
        effectiveness=effectiveness,
    )
    damage = fx.modify(ctx, "modify_damage", damage, attacker, scope="self",
                       attacker=attacker, defender=defender, move=move, crit=crit)
    damage = fx.modify(ctx, "modify_damage", damage, defender, scope="all",
                       attacker=attacker, defender=defender, move=move, crit=crit)
    return max(1, int(damage)), effectiveness


def _unstaged(ctx: Context, ref: Ref, stat: Stat, move, opponent: Ref, keep_positive: bool) -> int:
    """The stat with unhelpful stages ignored, as criticals do."""
    stage = ctx.state.sides[ref[0]].boost(ref[1], mutate.STAT_TO_BOOST[stat])
    if (stage > 0) == keep_positive:
        return effective_stat(ctx, ref, stat, move=move, opponent=opponent)
    extra = {"move": move, "opponent": opponent}
    raw = fx.modify(ctx, "modify_stat", mutate.raw_stat(ctx.state, ref, stat), ref,
                    stat=stat, **extra)
    return max(1, int(fx.modify(ctx, "modify_boosted_stat", raw, ref, stat=stat, **extra)))


def rolls_crit(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> bool:
    if move.raw.get("willCrit"):
        return True
    ratio = move.raw.get("critRatio", 1)
    ratio = _both_sides(ctx, "modify_crit_ratio", ratio, attacker, defender, move)
    if ratio <= NEVER_CRITS:
        return False
    denominator = CRIT_DENOMINATOR.get(ratio, 1)
    if denominator <= 1:
        return True
    return ctx.cursor.chance(1, denominator)


def connects(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> bool:
    """Accuracy, including stat stages for accuracy and evasion."""
    if move.accuracy is None:
        return True

    accuracy = _both_sides(ctx, "modify_accuracy", float(move.accuracy), attacker, defender, move)
    stage = (
        ctx.state.sides[attacker[0]].boost(attacker[1], "accuracy")
        - ctx.state.sides[defender[0]].boost(defender[1], "evasion")
    )
    stage = max(-6, min(6, stage))
    accuracy *= stage_multiplier(stage, accuracy_like=True)
    return ctx.cursor.chance(min(100, max(1, int(accuracy))), 100)


# --------------------------------------------------------------------------- #
# Using a move
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class ActiveMove:
    """A move as it is being used, which hooks may rewrite for this use only.

    Showdown's ``onModifyMove`` mutates a per-use copy of the move, and a
    surprising share of the ability list needs exactly that: Pixilate rewrites
    the type, Skill Link fixes the hit count, Long Reach removes contact,
    Infiltrator lets it through screens, Sheer Force deletes the secondaries it
    trades away. Attribute names mirror ``Move`` so everything downstream reads
    the same way whether it was rewritten or not.
    """

    base: Move
    type: str
    category: str
    base_power: int
    accuracy: int | None
    priority: int
    target: str
    flags: set[str]
    secondaries: list[dict]
    self_effects: dict | None
    multihit: object
    #: Set by Pixilate and friends, which also add 20% to what they changed.
    type_changed: bool = False
    #: Infiltrator: ignores screens and Substitute.
    infiltrates: bool = False
    #: Skill Link: multi-hit moves always hit the maximum number of times.
    always_max_hits: bool = False
    #: Unseen Fist / Piercing Drill: contact moves go through Protect.
    breaks_protect: bool = False
    #: Parental Bond: the extra hit lands at a quarter power (gen 7+).
    parental_bond: bool = False
    #: Which hit of a multi-hit move is being resolved. Beat Up reads it.
    hit_index: int = 0
    #: Reflected once already, by Magic Bounce or Magic Coat. Without this two
    #: Magic Bounce holders facing each other bounce the same move forever --
    #: singles could only ever have one of them on the field to be hit.
    has_bounced: bool = False
    #: This use landed on more than one Pokemon, so each takes a quarter less.
    #: Set at resolution, not read off the data: a spread move that finds only
    #: one target left standing does full damage.
    spread: bool = False

    @property
    def id(self) -> str:
        return self.base.id

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def raw(self) -> dict:
        return self.base.raw

    @property
    def is_status(self) -> bool:
        return self.category == "Status"


def activate(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> ActiveMove:
    """Build the per-use view and let ``modify_move`` hooks rewrite it."""
    active = ActiveMove(
        base=move,
        type=move.type,
        category=move.category,
        base_power=move.base_power,
        accuracy=move.accuracy,
        priority=move.priority,
        target=move.target,
        flags=set(move.flags),
        secondaries=list(_all_secondaries(move)),
        self_effects=move.raw.get("self") if isinstance(move.raw.get("self"), dict) else None,
        multihit=move.raw.get("multihit"),
        has_bounced=getattr(move, "has_bounced", False),
    )
    _rewrite_for_terrain(ctx, active, attacker)
    fx.notify(ctx, "modify_move", attacker, scope="self",
              active=active, attacker=attacker, defender=defender)
    return active


def _respects_type_immunity(move) -> bool:
    """Damaging moves always do; status moves only when the data says so."""
    declared = move.raw.get("ignoreImmunity")
    if declared is not None:
        return declared is False
    return move.category != "Status"


def resolve_targets(ctx: Context, attacker: Ref, move, target_code: int = 0) -> list[Ref]:
    """Everyone this use of the move lands on, in field-position order.

    Singles collapses to one entry every time; the list only ever has two in it
    because doubles put a second Pokemon on each side. Empty means the move has
    nobody to hit, which is a failure rather than a miss.
    """
    from pkcm.engine.state import resolve_target_code

    state = ctx.state
    kind = move.target

    if kind in ("self", "allySide", "allyTeam"):
        return [attacker]
    if kind == "allies":
        return state.allies_and_self(attacker)
    if kind == "allAdjacentFoes" or kind == "foeSide":
        return state.foes(attacker)
    if kind == "allAdjacent":
        return state.foes(attacker) + [ref for ref in state.allies_and_self(attacker)
                                       if ref != attacker]
    if kind == "all":
        return [attacker]  # weather and rooms belong to the field, not a Pokemon
    if kind == "randomNormal":
        # Thrash and friends pick for themselves; nobody chose a target.
        foes = state.foes(attacker)
        if not foes:
            return []
        return [foes[ctx.cursor.between(0, len(foes) - 1)]]
    if kind == "adjacentAlly":
        # Never the foe, whatever the code says -- in singles there is simply
        # nobody to aim at, and the move fails rather than hitting across.
        partner = state.ally(attacker)
        return [partner] if partner is not None else []
    if kind == "adjacentAllyOrSelf":
        partner = state.ally(attacker)
        if target_code == TARGET_ALLY and partner is not None:
            return [partner]
        return [attacker]

    chosen = resolve_target_code(state, attacker, target_code)
    if chosen is None:
        # The chosen target left the field between selection and resolution.
        # Champions redirects a single-target move to whoever is left rather
        # than fizzling it, which is what makes double-target prediction work.
        remaining = state.foes(attacker)
        if kind in ("adjacentAlly", "adjacentAllyOrSelf"):
            remaining = [ref for ref in state.allies_and_self(attacker) if ref != attacker]
        if not remaining:
            return []
        chosen = remaining[0]
    return [chosen]


def redirect(ctx: Context, attacker: Ref, target: Ref, move) -> Ref:
    """Follow Me, Rage Powder, Lightning Rod, Storm Drain.

    Only single-target moves from the other side can be pulled, and only onto a
    Pokemon that is still standing. Asked as a ``modify`` hook over everyone on
    the field, because the Pokemon doing the pulling is neither the attacker
    nor the current target.
    """
    from pkcm.engine.abilities import IGNORES_REDIRECTION

    if move.target not in ("normal", "any", "adjacentFoe") or target[0] == attacker[0]:
        return target
    # Stalwart and Propeller Tail hit what they aimed at, Follow Me or no.
    if ctx.ability_of(attacker) in IGNORES_REDIRECTION:
        return target
    pulled = target
    for candidate in ctx.state.foes(attacker):
        if candidate == target:
            continue
        pulled = fx.modify(ctx, "redirect_target", pulled, candidate,
                           scope="self", attacker=attacker, move=move)
    return pulled


def use_move(
    ctx: Context,
    attacker: Ref,
    move: Move,
    move_index: int | None = None,
    target_code: int = 0,
    defender: Ref | None = None,
) -> None:
    """Run one move from start to finish.

    ``target_code`` is what the player picked (see ``pkcm.engine.actions``).
    ``defender`` names the foe instead, for callers that already know who is
    across from them -- tests, Magic Bounce, Instruct. It stands in for the
    chosen opponent only: a move that targets its own user still targets its
    own user.
    """
    side = ctx.state.sides[attacker[0]]

    if not fx.allows(ctx, "try_move", attacker, move=move):
        _clear_flinch(ctx, attacker)
        return

    # Asked here rather than through ``try_move`` because the Imprison volatile
    # is on the opponent, where the mover's own effects cannot reach it.
    if move.id != STRUGGLE_ID and move.id in imprisoned_moves(ctx.state, attacker[0]):
        ctx.emit(Event("cant_move", side=attacker[0], slot=attacker[1], detail="imprison"))
        _clear_flinch(ctx, attacker)
        return

    refusal = MOVE_PRECONDITIONS.get(move.id)
    if refusal is not None:
        reason = refusal(ctx, attacker, move)
        if reason is not None:
            ctx.emit(Event("move_failed", side=attacker[0], move=move.id, detail=reason))
            return

    if move_index is not None:
        side.pp[attacker[1]][move_index] -= 1

    ctx.acting = attacker
    ctx.emit(ev.move_used(attacker[0], attacker[1], ctx.state.species_id(*attacker), move.id))
    _clear_flinch(ctx, attacker)
    if move_index is not None:
        side.volatiles[attacker[1]]["lastmove"] = move.id
        fx.notify(ctx, "commit_move", attacker, scope="self",
                  move=move, move_index=move_index)

    # A move this executor cannot run must say so. Letting it fall through would
    # look like a move that legitimately did nothing, and a policy trained on
    # that learns the move is free (docs/DESIGN.md §1g).
    unsupported = move_support(move)
    if unsupported is not None:
        ctx.emit(
            Event("unimplemented", side=attacker[0], slot=attacker[1],
                  move=move.id, detail=unsupported)
        )
        return

    from pkcm.engine import tactics

    # A two-turn move spends its first turn charging and its second attacking.
    if "charge" in move.flags:
        if tactics.is_charging(ctx, attacker) is None:
            if ctx.state.item_id(*attacker) != "powerherb":
                tactics.start_charging(ctx, attacker, move, move_index)
                return
            mutate.consume_item(ctx, attacker, move.id)
        else:
            tactics.finish_charging(ctx, attacker)

    if "recharge" in move.flags:
        mutate.add_volatile(ctx, attacker, "mustrecharge")

    if move.raw.get("multihit") is None and _locks_in(move):
        tactics.start_locked_move(ctx, attacker, move, move_index)

    first_guess = defender or (ctx.state.foes(attacker) or [attacker])[0]
    active = activate(ctx, attacker, first_guess, move)

    if defender is not None and active.target in SINGLE_FOE_TARGETS:
        targets = [defender]
    else:
        targets = resolve_targets(ctx, attacker, active, target_code)
        targets = [redirect(ctx, attacker, ref, active) for ref in targets]

    if not targets:
        ctx.emit(Event("move_failed", side=attacker[0], move=move.id, detail="no target"))
        return

    # A spread move that finds only one target does full damage. The count is
    # taken here, after fainting and redirection have had their say.
    active.spread = active.target in SPREAD_TARGETS and len(targets) > 1

    for target in targets:
        if ctx.state.sides[target[0]].hp[target[1]] <= 0 and target != attacker:
            continue  # the first hit of a spread move knocked this one out
        _resolve_one(ctx, attacker, target, active)

    fx.notify(ctx, "after_move", attacker, move=active)


def _resolve_one(ctx: Context, attacker: Ref, target: Ref, active) -> None:
    """One move against one target, with the defender's ability suppression."""
    targets_opponent = target != attacker

    # Mold Breaker and friends blind the defender's ability for the *whole*
    # resolution -- immunity, damage modifiers, contact reactions, all of it.
    # Suppressing it here rather than at each check is the only version that
    # cannot be quietly incomplete.
    suppressing = targets_opponent and ignores_target_ability(ctx, attacker, active)
    if suppressing:
        ctx.suppressed_abilities.add(target)
        ctx.emit(Event("ability_suppressed", side=target[0], slot=target[1],
                       detail=ctx.state.ability_id(*target)))
    try:
        _resolve(ctx, attacker, target, target, targets_opponent, active)
    finally:
        if suppressing:
            ctx.suppressed_abilities.discard(target)


#: Abilities that ignore the target's ability while their holder attacks.
MOLD_BREAKER_ABILITIES = frozenset({"moldbreaker", "turboblaze", "teravolt"})
#: Moves that do the same regardless of who uses them.
MOLD_BREAKER_MOVES = frozenset({"sunsteelstrike", "moongeistbeam", "photongeyser"})


def ignores_target_ability(ctx: Context, attacker: Ref, move: Move) -> bool:
    return ctx.ability_of(attacker) in MOLD_BREAKER_ABILITIES or move.id in MOLD_BREAKER_MOVES


def _resolve(
    ctx: Context,
    attacker: Ref,
    defender: Ref,
    target: Ref,
    targets_opponent: bool,
    move: Move,
) -> None:
    if targets_opponent and ctx.state.sides[defender[0]].hp[defender[1]] <= 0:
        ctx.emit(Event("move_failed", side=attacker[0], move=move.id, detail="no target"))
        return

    if targets_opponent and not fx.allows(
        ctx, "try_hit", defender, attacker=attacker, defender=defender, move=move
    ):
        return

    # Status moves ignore the type chart unless they say otherwise. Showdown
    # defaults ``ignoreImmunity`` to true for them and Thunder Wave sets it back
    # to false, which is the whole rule: Curse and Trick-or-Treat land on Normal
    # types, Thunder Wave does not reach a Ground type.
    # A Prankster-boosted status move does not reach a Dark type (gen 6+).
    # Showdown carries this as ``move.pranksterBoosted``, set when the holder
    # uses a status move -- which is exactly the condition asked here, so the
    # flag would only be a second copy of it.
    if targets_opponent and move.category == "Status"             and ctx.ability_of(attacker) == "prankster"             and "dark" in ctx.state.types(*defender):
        ctx.emit(ev.immune(defender[0], defender[1], move.id))
        return

    if targets_opponent and _respects_type_immunity(move) \
            and type_effectiveness(ctx, attacker, defender, move) == 0.0:
        ctx.emit(ev.immune(defender[0], defender[1], move.id))
        return

    if targets_opponent and not connects(ctx, attacker, defender, move):
        ctx.emit(ev.missed(attacker[0], attacker[1], move.id))
        return

    if move.category == "Status":
        landed = _apply_status_move(ctx, attacker, target, move)
    else:
        landed = _apply_damaging_move(ctx, attacker, defender, move)

    if landed:
        _apply_self_effects(ctx, attacker, move)


def _clear_flinch(ctx: Context, ref: Ref) -> None:
    mutate.remove_volatile(ctx, ref, "flinch", quiet=True)


def _apply_damaging_move(ctx: Context, attacker: Ref, defender: Ref, move) -> bool:
    from pkcm.engine import tactics

    if move.id in tactics.COUNTER_MOVES:
        amount = tactics.counter_damage(ctx, attacker, defender, move)
        if amount is None:
            ctx.emit(Event("move_failed", side=attacker[0], move=move.id))
            return False
        apply_damage(ctx, defender, amount, "damage", move=move.id, effectiveness=1.0,
                     __source__=attacker, __move__=move)
        return True

    if move.id == "endeavor":
        amount = tactics.endeavor_damage(ctx, attacker, defender)
        if amount is None:
            ctx.emit(Event("move_failed", side=attacker[0], move=move.id))
            return False
        apply_damage(ctx, defender, amount, "damage", move=move.id, effectiveness=1.0,
                     __source__=attacker, __move__=move)
        return True

    if move.raw.get("ohko"):
        return _apply_ohko(ctx, attacker, defender, move)

    if move.id == "superfang":
        amount = max(1, mutate.current_hp(ctx.state, defender) // 2)
        if type_effectiveness(ctx, attacker, defender, move) == 0.0:
            ctx.emit(ev.immune(defender[0], defender[1], move.id))
            return False
        apply_damage(ctx, defender, amount, "damage", move=move.id, effectiveness=1.0,
                     __source__=attacker, __move__=move)
        return True

    fixed = move.raw.get("damage")
    if fixed is not None:
        amount = LEVEL if fixed == "level" else int(fixed)
        if type_effectiveness(ctx, attacker, defender, move) == 0.0:
            ctx.emit(ev.immune(defender[0], defender[1], move.id))
            return False
        apply_damage(ctx, defender, amount, "damage", move=move.id, effectiveness=1.0)
        return True

    hits = _hit_count(ctx, move)
    total = 0
    any_crit = False
    for hit_number in range(hits):
        if ctx.state.sides[defender[0]].hp[defender[1]] <= 0:
            break
        if hasattr(move, "hit_index"):
            move.hit_index = hit_number
        crit = rolls_crit(ctx, attacker, defender, move)
        any_crit = any_crit or crit
        damage, effectiveness = compute_damage(ctx, attacker, defender, move, crit)
        if getattr(move, "parental_bond", False) and hit_number > 0:
            damage = max(1, chain_modify(damage, X0_25))
        if effectiveness == 0.0:
            ctx.emit(ev.immune(defender[0], defender[1], move.id))
            return False
        dealt = _deal_or_break_substitute(ctx, attacker, defender, move, damage, effectiveness, crit)
        total += dealt

    if hits > 1:
        ctx.emit(Event("multi_hit", side=attacker[0], move=move.id, amount=hits))

    if total:
        fx.notify(ctx, "dealt_damage", attacker, scope="self", attacker=attacker,
                  defender=defender, move=move, damage=total)
        fx.notify(ctx, "after_damage", defender, attacker=attacker, defender=defender,
                  move=move, damage=total, crit=any_crit)
        fx.notify(ctx, "after_damage", attacker, scope="self", attacker=attacker,
                  defender=defender, move=move, damage=total, crit=any_crit)

    if move.id == "fling":
        mutate.consume_item(ctx, attacker, move.id)
    elif move.id == "spitup":
        from pkcm.engine.moveeffects import _spend_stockpile

        _spend_stockpile(ctx, attacker)

    _apply_drain(ctx, attacker, move, total)
    _apply_recoil(ctx, attacker, move, total)
    if total:
        _apply_secondaries(ctx, attacker, defender, move)
        if "partiallytrapped" in _volatile_names(move):
            tactics.start_trapping(ctx, defender, move)
    _after_effects(ctx, attacker, defender, move, landed=bool(total))
    return True


def _volatile_names(move) -> set[str]:
    names = {move.raw.get("volatileStatus")}
    single = move.raw.get("secondary") or {}
    names.add(single.get("volatileStatus"))
    for secondary in move.raw.get("secondaries") or ():
        names.add((secondary or {}).get("volatileStatus"))
    return {name for name in names if name}


def _after_effects(ctx: Context, attacker: Ref, defender: Ref, move, landed: bool) -> None:
    """Things a move does once its damage or status is settled."""
    from pkcm.engine import tactics
    from pkcm.engine.moveeffects import SPECIAL_MOVES

    # A damaging move can have a hand-written effect too -- Sparkling Aria puts
    # out a burn. Status moves already ran theirs on the way in.
    if landed and move.category != "Status":
        special = SPECIAL_MOVES.get(move.id)
        if special is not None:
            special(ctx, attacker, defender, move)

    if move.id in tactics.SELF_DESTRUCT_MOVES:
        tactics.self_destruct(ctx, attacker, defender, move)
        return

    if move.raw.get("forceSwitch") and landed:
        if not tactics.force_switch(ctx, defender):
            ctx.emit(Event("move_failed", side=attacker[0], move=move.id,
                           detail="nobody to drag in"))

    if move.raw.get("selfSwitch") and landed:
        tactics.self_switch(ctx, attacker)


def _deal_or_break_substitute(
    ctx: Context, attacker: Ref, defender: Ref, move: Move,
    damage: int, effectiveness: float, crit: bool,
) -> int:
    substitute = mutate.volatile(ctx.state, defender, "substitute")
    bypasses = getattr(move, "infiltrates", False) or "authentic" in move.flags
    if substitute is not None and not bypasses:
        substitute["hp"] -= damage
        if substitute["hp"] <= 0:
            mutate.remove_volatile(ctx, defender, "substitute")
        else:
            ctx.emit(Event("substitute_hit", side=defender[0], slot=defender[1],
                           amount=damage, move=move.id))
        return 0

    from pkcm.engine import tactics

    dealt = apply_damage(ctx, defender, damage, "damage", move=move.id,
                         effectiveness=effectiveness, crit=crit,
                         __source__=attacker, __move__=move)
    tactics.record_hit(ctx, defender, attacker, move, dealt)
    return dealt


def _apply_ohko(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> bool:
    if type_effectiveness(ctx, attacker, defender, move) == 0.0:
        ctx.emit(ev.immune(defender[0], defender[1], move.id))
        return False
    apply_damage(ctx, defender, mutate.max_hp(ctx.state, defender), "damage",
                 move=move.id, detail="ohko", effectiveness=1.0)
    return True


def _hit_count(ctx: Context, move) -> int:
    if move.id == "beatup":
        from pkcm.engine.moveeffects import beat_up_hits

        return max(1, len(beat_up_hits(ctx, ctx.acting or (0, 0))))
    multihit = getattr(move, "multihit", None)
    if multihit is None:
        multihit = move.raw.get("multihit")
    if multihit is None:
        return 1
    if isinstance(multihit, int):
        return multihit
    low, high = multihit
    if getattr(move, "always_max_hits", False):
        return high
    if (low, high) == (2, 5):
        return ctx.cursor.choice(MULTIHIT_2_TO_5)
    return ctx.cursor.between(low, high)


def _apply_status_move(ctx: Context, attacker: Ref, target: Ref, move) -> bool:
    from pkcm.engine.moveeffects import SPECIAL_MOVES

    raw = move.raw
    did_something = False

    # Moves Showdown keeps in handler code rather than in fields. They run
    # alongside whatever the data does describe, not instead of it.
    special = SPECIAL_MOVES.get(move.id)
    if special is not None:
        did_something |= special(ctx, attacker, target, move)

    if raw.get("stallingMove") and move.id != "endure":
        return _apply_protect(ctx, attacker, move)

    if move.id == "substitute":
        return _apply_substitute(ctx, attacker)

    # Shed Tail buys the same doll for half the HP and leaves it behind. Its
    # data says ``volatileStatus: substitute``, which the declarative path below
    # would honour by creating one with no HP at all -- and the first hit on it
    # would then raise rather than break it.
    if move.id == "shedtail":
        return _apply_substitute(ctx, attacker, SHED_TAIL_FRACTION, "shedtail")

    if "boosts" in raw and raw["boosts"]:
        did_something |= bool(mutate.boost(ctx, target, raw["boosts"], source=attacker))

    if raw.get("status"):
        did_something |= mutate.set_status(ctx, target, raw["status"], source=attacker)

    if raw.get("volatileStatus") and raw["volatileStatus"] not in TACTICS_MANAGED_VOLATILES:
        did_something |= mutate.add_volatile(ctx, target, raw["volatileStatus"],
                                             **_volatile_data(ctx, raw["volatileStatus"]))

    if raw.get("heal"):
        numerator, denominator = raw["heal"]
        did_something |= bool(
            heal(ctx, attacker, mutate.max_hp(ctx.state, attacker) * numerator // denominator,
                 reason=move.id)
        )

    did_something |= _apply_field_effects(ctx, attacker, target, move)

    from pkcm.engine import tactics

    if move.id in tactics.SELF_DESTRUCT_MOVES or move.raw.get("forceSwitch") \
            or move.raw.get("selfSwitch"):
        _after_effects(ctx, attacker, target, move, landed=True)
        return True

    if not did_something:
        ctx.emit(Event("move_failed", side=attacker[0], move=move.id))
    return did_something


#: Moves that lock their user in for a few turns and then confuse it.
LOCKING_MOVES = frozenset({"outrage", "thrash", "petaldance", "ragingfury"})


def _locks_in(move) -> bool:
    return move.id in LOCKING_MOVES


#: Volatiles that ``tactics`` owns outright. The move data mentions some of
#: them too, but it cannot supply the counters they need, and applying it a
#: second time overwrites the bookkeeping.
TACTICS_MANAGED_VOLATILES = frozenset({
    "lockedmove", "partiallytrapped", "twoturn", "mustrecharge", "invulnerable",
    # Hand-written in moveeffects, which supplies counters the data cannot.
    "curse", "taunt", "torment", "encore", "disable", "imprison", "yawn",
    "destinybond", "saltcure", "syrupbomb", "perishsong", "stockpile",
    "focusenergy", "magnetrise", "lockon", "endure", "noretreat", "roost",
    "substitute",
    "aquaring", "minimize", "healblock", "uproar", "smackdown", "electrify",
    "powertrick", "powershift", "kingsshield", "banefulbunker", "spikyshield",
    "silktrap", "obstruct", "burningbulwark", "attract",
})


def _volatile_data(ctx: Context, name: str) -> dict:
    if name == "confusion":
        return {"turns": ctx.cursor.between(2, 5)}
    return {}


def _apply_protect(ctx: Context, attacker: Ref, move: Move) -> bool:
    """Consecutive Protects get likelier to fail: 1/1, then 1/3, 1/9, ..."""
    stall = mutate.volatile(ctx.state, attacker, "stall")
    denominator = 3 ** stall["count"] if stall else 1
    if denominator > 1 and not ctx.cursor.chance(1, denominator):
        mutate.remove_volatile(ctx, attacker, "stall", quiet=True)
        ctx.emit(Event("move_failed", side=attacker[0], move=move.id, detail="stalled out"))
        return False

    mutate.add_volatile(ctx, attacker, "protect")
    if stall is None:
        ctx.state.sides[attacker[0]].volatiles[attacker[1]]["stall"] = {"count": 1}
    else:
        stall["count"] += 1
    return True


#: A Substitute costs this fraction of maximum HP and is worth that much.
#: Shed Tail pays half instead, and hands the result to a replacement.
SUBSTITUTE_FRACTION = 4
SHED_TAIL_FRACTION = 2


def _apply_substitute(ctx: Context, attacker: Ref, fraction: int = SUBSTITUTE_FRACTION,
                      move_id: str = "substitute") -> bool:
    """Pay HP for a doll that takes hits until its own HP runs out.

    The cost is also the doll's HP, which is why they are one number. Shed Tail
    calls this too: it is the same doll bought at a different price, and giving
    it its own path is how one of them ends up without the HP key that the
    damage path assumes is there.
    """
    cost = mutate.max_hp(ctx.state, attacker) // fraction
    if cost <= 0 or mutate.current_hp(ctx.state, attacker) <= cost:
        ctx.emit(Event("move_failed", side=attacker[0], move=move_id, detail="not enough HP"))
        return False
    if mutate.volatile(ctx.state, attacker, "substitute") is not None:
        ctx.emit(Event("move_failed", side=attacker[0], move=move_id, detail="already up"))
        return False
    apply_damage(ctx, attacker, cost, "damage", detail="substitute")
    mutate.add_volatile(ctx, attacker, "substitute", hp=cost)
    return True


def _apply_field_effects(ctx: Context, attacker: Ref, target: Ref, move: Move) -> bool:
    raw = move.raw
    changed = False

    weather = raw.get("weather")
    if weather:
        set_weather(ctx, _to_id(weather), attacker)
        changed = True

    terrain = raw.get("terrain")
    if terrain:
        set_terrain(ctx, _to_id(terrain), attacker)
        changed = True

    pseudo = raw.get("pseudoWeather")
    if pseudo:
        ctx.state.field.rooms[_to_id(pseudo)] = 5
        ctx.emit(Event("room_start", detail=_to_id(pseudo)))
        changed = True

    condition = raw.get("sideCondition")
    if condition:
        from pkcm.engine.conditions import SIDE_CONDITION_DURATION, SIDE_CONDITION_LAYERS

        name = _to_id(condition)
        side_index = (attacker[0] if move.target in SELF_TARGETS or move.target == "allySide"
                      else 1 - attacker[0])
        conditions = ctx.state.sides[side_index].conditions

        if name in SIDE_CONDITION_DURATION:
            if name in conditions:
                return changed  # a screen already up cannot be re-set
            conditions[name] = fx.modify(
                ctx, "modify_field_duration", SIDE_CONDITION_DURATION[name],
                attacker, scope="self", field=name, kind="side")
        else:
            cap = SIDE_CONDITION_LAYERS.get(name, 1)
            if conditions.get(name, 0) >= cap:
                return changed
            conditions[name] = conditions.get(name, 0) + 1

        ctx.emit(Event("side_condition", side=side_index, detail=name,
                       amount=conditions[name]))
        changed = True

    return changed


def _to_id(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


#: Base duration for weather, terrain and screens. The rocks and Light Clay
#: extend it through ``modify_field_duration``.
FIELD_DURATION = 5


def field_duration(ctx: Context, setter: Ref, field: str, kind: str) -> int:
    return fx.modify(ctx, "modify_field_duration", FIELD_DURATION, setter,
                     scope="self", field=field, kind=kind)


def set_weather(ctx: Context, weather: str, setter: Ref) -> bool:
    if ctx.state.field.weather == weather:
        return False
    ctx.state.field.weather = weather
    ctx.state.field.weather_turns = field_duration(ctx, setter, weather, "weather")
    ctx.emit(Event("weather_start", detail=weather, turn=ctx.state.field.weather_turns))
    return True


def set_terrain(ctx: Context, terrain: str, setter: Ref) -> bool:
    if ctx.state.field.terrain == terrain:
        return False
    ctx.state.field.terrain = terrain
    ctx.state.field.terrain_turns = field_duration(ctx, setter, terrain, "terrain")
    ctx.emit(Event("terrain_start", detail=terrain, turn=ctx.state.field.terrain_turns))
    return True


def _apply_self_effects(ctx: Context, attacker: Ref, move) -> None:
    """``self: {boosts, volatileStatus}`` -- unconditional, e.g. Overheat."""
    payload = getattr(move, "self_effects", None)
    if payload is None:
        payload = move.raw.get("self")
    if not isinstance(payload, dict):
        return
    if payload.get("boosts"):
        mutate.boost(ctx, attacker, payload["boosts"], source=attacker)
    if payload.get("volatileStatus") and payload["volatileStatus"] not in TACTICS_MANAGED_VOLATILES:
        mutate.add_volatile(ctx, attacker, payload["volatileStatus"])


def _apply_secondaries(ctx: Context, attacker: Ref, defender: Ref, move) -> None:
    secondaries = getattr(move, "secondaries", None)
    if secondaries is None:
        secondaries = _all_secondaries(move)

    if secondaries and not fx.allows(ctx, "try_secondary", defender,
                                     attacker=attacker, move=move):
        return

    for secondary in secondaries:
        if not secondary:
            continue
        chance = secondary.get("chance", 100)
        if chance < 100 and not ctx.cursor.chance(chance, 100):
            continue

        if secondary.get("status"):
            mutate.set_status(ctx, defender, secondary["status"], source=attacker)
        if (secondary.get("volatileStatus")
                and secondary["volatileStatus"] not in TACTICS_MANAGED_VOLATILES):
            mutate.add_volatile(ctx, defender, secondary["volatileStatus"],
                                **_volatile_data(ctx, secondary["volatileStatus"]))
        if secondary.get("boosts"):
            mutate.boost(ctx, defender, secondary["boosts"], source=attacker)

        own = secondary.get("self")
        if isinstance(own, dict) and own.get("boosts"):
            mutate.boost(ctx, attacker, own["boosts"], source=attacker)


def _apply_drain(ctx: Context, attacker: Ref, move: Move, damage: int) -> None:
    drain = move.raw.get("drain")
    if not drain or damage <= 0:
        return
    numerator, denominator = drain
    restored = max(1, damage * numerator // denominator)
    restored = fx.modify(ctx, "modify_drain", restored, attacker, scope="self", move=move)
    heal(ctx, attacker, restored, reason="drain")


def _apply_recoil(ctx: Context, attacker: Ref, move: Move, damage: int) -> None:
    if move.id == STRUGGLE_ID:
        amount = mutate.fraction_of_max(ctx.state, attacker, STRUGGLE_RECOIL_FRACTION)
        apply_damage(ctx, attacker, amount, "recoil")
        return

    recoil = move.raw.get("recoil")
    if not recoil or damage <= 0:
        return
    numerator, denominator = recoil
    apply_damage(ctx, attacker, max(1, damage * numerator // denominator), "recoil")


# --------------------------------------------------------------------------- #
# What this executor can and cannot run
#
# The predicate lives next to the executor it describes (and that is also what
# keeps `scope` importable from here without a cycle). `scope` re-exports it.
# --------------------------------------------------------------------------- #


VARIABLE_POWER_REASON = "variable base power"
NO_EFFECT_DATA = "effect not described by the data"
MULTI_TURN = "two-turn"
SELF_DESTRUCT = "self-destructing"
FORCE_SWITCH = "forces a switch"
SELF_SWITCH = "switches the user out"
SPECIAL_DAMAGE = "damage computed from what it was hit by"

#: A status move is executable when it carries at least one of these.
DECLARATIVE_FIELDS = (
    "boosts",
    "status",
    "volatileStatus",
    "heal",
    "weather",
    "terrain",
    "pseudoWeather",
    "sideCondition",
    "stallingMove",
)

#: Damaging moves whose damage comes from elsewhere in the battle entirely.
COUNTER_MOVES = frozenset(
    {"counter", "mirrorcoat", "metalburst", "comeuppance", "bide", "endeavor", "finalgambit"}
)

#: Handled explicitly by the executor despite carrying no declarative fields.
SPECIAL_CASED = frozenset({"substitute"})


def _unwired_condition(move: Move) -> str | None:
    """Does this move set a condition nobody handles?

    A move writing ``sideCondition: "safeguard"`` reads as declarative and would
    sail through the field check below, then do exactly nothing -- the name goes
    into the state and no handler ever looks at it. Safeguard was caught doing
    precisely that. Checking the value, not just the presence of the field, is
    the difference between "we implement this" and "we store the word".
    """
    from pkcm.engine import conditions as cond

    checks = (
        ("status", cond.IMPLEMENTED_STATUSES, "status"),
        ("volatileStatus", cond.IMPLEMENTED_VOLATILES, "volatile condition"),
        ("sideCondition", cond.IMPLEMENTED_SIDE_CONDITIONS, "side condition"),
        ("weather", cond.IMPLEMENTED_WEATHER, "weather"),
        ("terrain", cond.IMPLEMENTED_TERRAIN, "terrain"),
        ("pseudoWeather", cond.IMPLEMENTED_ROOMS, "field effect"),
    )
    for field_name, implemented, label in checks:
        value = move.raw.get(field_name)
        if value and _to_id(str(value)) not in implemented:
            return f"unhandled {label}: {_to_id(str(value))}"

    payloads = list(_all_secondaries(move))
    own = move.raw.get("self")
    if isinstance(own, dict):
        payloads.append(own)

    for payload in payloads:
        for field_name, implemented, label in checks[:2]:
            value = payload.get(field_name)
            if value and _to_id(str(value)) not in implemented:
                return f"unhandled {label}: {_to_id(str(value))}"
    return None


def _all_secondaries(move: Move) -> list[dict]:
    secondaries = move.raw.get("secondaries")
    if secondaries is None:
        single = move.raw.get("secondary")
        secondaries = [single] if single else []
    return [s for s in secondaries if s]


def move_support(move: Move) -> str | None:
    """``None`` if the engine executes this move correctly, else why it does not."""
    from pkcm.engine.tactics import COUNTER_MOVES, SELF_DESTRUCT_MOVES

    # These four used to be blanket exclusions. They are structural rather than
    # declarative, which is a reason to write real code for them, not a reason
    # to skip them -- forcing a switch, U-turning out and answering a hit with
    # Counter are all ordinary competitive play.
    if move.id in COUNTER_MOVES or move.id in SELF_DESTRUCT_MOVES:
        return None
    if move.id in ("endeavor", "superfang"):
        return None
    if move.raw.get("forceSwitch") or move.raw.get("selfSwitch"):
        return None
    if "charge" in move.flags or "recharge" in move.flags:
        return None
    if move.id in LOCKING_MOVES:
        return None

    # Asked before the condition check: a move written by hand knows what its
    # own volatile is for, and the generic check cannot see that.
    from pkcm.engine.moveeffects import SPECIAL_MOVES

    if move.id in SPECIAL_MOVES or move.id in ("beatup", "fling", "spitup"):
        return None

    unwired = _unwired_condition(move)
    if unwired is not None:
        return unwired

    if move.category == "Status":
        if move.id in SPECIAL_CASED:
            return None
        if any(move.raw.get(field) for field in DECLARATIVE_FIELDS):
            return None
        return NO_EFFECT_DATA


    if move.base_power == 0 and move.raw.get("damage") is None and not move.raw.get("ohko"):
        if move.id not in VARIABLE_POWER:
            return VARIABLE_POWER_REASON
    return None
