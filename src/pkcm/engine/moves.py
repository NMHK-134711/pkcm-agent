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


def _needs_own_type(type_name: str):
    def refuse(ctx: Context, ref: Ref, move: Move) -> str | None:
        return None if type_name in ctx.state.types(*ref) else f"not {type_name}"

    return refuse


#: Moves that refuse to run under some condition, checked before PP is spent.
#: The string is the reason, and it reaches the log.
MOVE_PRECONDITIONS: dict[str, Callable[[Context, Ref, Move], str | None]] = {
    "steelroller": _needs_terrain,
    "burnup": _needs_own_type("Fire"),
}


def _knock_off_power(ctx: Context, attacker: Ref, defender: Ref, move) -> int:
    """Champions: 상대가 도구를 지니고 있으면 위력이 1.5배가 된다.

    Asked before the damage is rolled, and the item is taken off after it --
    so this reads the item while it is still there, which is the order the
    move is written in.
    """
    from pkcm.engine.moveeffects import _holds_removable

    if _holds_removable(ctx, defender) is None:
        return move.base_power
    return chain_modify(move.base_power, X1_5)


#: Weather that lets a two-turn move fire the same turn.
#:
#: Champions, on Solar Beam: 쾌청 상태인 경우 차지 상태를 생략하고 바로 공격할
#: 수 있다. On Electro Shot: 비 상태인 경우. Neither was here, so a Solar Beam
#: in its own sun still spent a turn winding up.
CHARGE_SKIPS: dict[str, tuple[str, ...]] = {
    "solarbeam": ("sunnyday", "desolateland"),
    "solarblade": ("sunnyday", "desolateland"),
    "electroshot": ("raindance", "primordialsea"),
}

#: What a two-turn move pays its user on the turn it is used.
CHARGE_TURN_BOOST: dict[str, dict[str, int]] = {
    "electroshot": {"spa": 1},
    "meteorbeam": {"spa": 1},
}


def _stored_power(ctx: Context, attacker: Ref, defender: Ref, move) -> int:
    """Champions: 자신의 올라간 능력 변화 1단계당 이 기술의 위력이 20씩 올라간다.

    Raised stages only -- a Pokemon that has been dropped to -2 still swings
    at twenty.
    """
    boosts = ctx.state.sides[attacker[0]].boosts[attacker[1]]
    return move.base_power + 20 * sum(stage for stage in boosts if stage > 0)


def _lash_out(ctx: Context, attacker: Ref, defender: Ref, move) -> int:
    """Champions: 사용한 턴 동안 자신의 능력이 떨어진 경우 위력이 2배가 된다."""
    from pkcm.engine import mutate as _mutate

    dropped = _mutate.volatile(ctx.state, attacker, "statdropped")
    if dropped is not None and dropped.get("turn") == ctx.state.turn:
        return move.base_power * 2
    return move.base_power


def _solar_power(ctx: Context, attacker: Ref, defender: Ref, move) -> int:
    """Champions: 다른 날씨인 경우 위력이 1/2이 된다.

    "Other weather" and not "no weather": clear skies leave it alone.
    """
    weather = ctx.state.field.weather
    if weather is None or weather in CHARGE_SKIPS[move.id]:
        return move.base_power
    return max(1, move.base_power // 2)


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
    "knockoff": _knock_off_power,
    "storedpower": _stored_power,
    "powertrip": _stored_power,
    "lashout": _lash_out,
    "solarbeam": _solar_power,
    "solarblade": _solar_power,
    "electroball": _electro_ball,
    "flail": _low_hp_scaling,
    "reversal": _low_hp_scaling,
    "crushgrip": _target_hp_scaling,
    "wringout": _target_hp_scaling,
    "hardpress": _target_hp_scaling,
}


def _doubled_when(condition):
    def compute(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
        return move.base_power * 2 if condition(ctx, attacker, defender) else move.base_power

    return compute


def _boost_counting(base: int, per: int):
    """Stored Power and Power Trip: the user's positive stages, priced."""
    def compute(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
        stages = sum(max(0, value)
                     for value in ctx.state.sides[attacker[0]].boosts[attacker[1]])
        return base + per * stages

    return compute


def _hp_scaled(top: int):
    """Eruption and Water Spout: full power only at full health."""
    def compute(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
        return max(1, top * mutate.current_hp(ctx.state, attacker)
                   // mutate.max_hp(ctx.state, attacker))

    return compute


def _was_hit_by_target(ctx: Context, attacker: Ref, defender: Ref) -> bool:
    ledger = ctx.state.sides[attacker[0]].volatiles[attacker[1]].get("hurtthisturn")
    return bool(ledger) and ledger.get("source") == defender


def _target_took_damage(ctx: Context, attacker: Ref, defender: Ref) -> bool:
    return bool(ctx.state.sides[defender[0]].volatiles[defender[1]].get("hurtthisturn"))


def _target_already_moved(ctx: Context, attacker: Ref, defender: Ref) -> bool:
    return defender in ctx.acted


def _target_statused(ctx: Context, attacker: Ref, defender: Ref) -> bool:
    return ctx.state.sides[defender[0]].status[defender[1]] is not None


def _own_last_move_failed(ctx: Context, attacker: Ref, defender: Ref) -> bool:
    return bool(ctx.state.sides[attacker[0]].volatiles[attacker[1]].get("lastmovefailed"))


def _no_held_item(ctx: Context, attacker: Ref, defender: Ref) -> bool:
    return ctx.state.item_id(*attacker) is None


def _fallen_allies(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    """Last Respects: +50 for every brought party member lying down."""
    side = ctx.state.sides[attacker[0]]
    fallen = sum(1 for hp in side.hp if hp <= 0)
    return move.base_power + 50 * fallen


def _triple_axel(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    """20, then 40, then 60. The hit loop stamps ``hit_index`` for us."""
    return 20 * (getattr(move, "hit_index", 0) + 1)


def _rage_fist(ctx: Context, attacker: Ref, defender: Ref, move: Move) -> int:
    """+50 for each hit taken, capped at six, surviving switches."""
    side = ctx.state.sides[attacker[0]]
    taken = min(6, side.status_data[attacker[1]].get("timeshit", 0))
    return move.base_power + 50 * taken


VARIABLE_POWER.update({
    "hex": _doubled_when(_target_statused),
    "infernalparade": _doubled_when(_target_statused),
    "avalanche": _doubled_when(_was_hit_by_target),
    "assurance": _doubled_when(_target_took_damage),
    "payback": _doubled_when(_target_already_moved),
    "stompingtantrum": _doubled_when(_own_last_move_failed),
    "temperflare": _doubled_when(_own_last_move_failed),
    "acrobatics": _doubled_when(_no_held_item),
    "storedpower": _boost_counting(20, 20),
    "powertrip": _boost_counting(20, 20),
    "eruption": _hp_scaled(150),
    "waterspout": _hp_scaled(150),
    "lastrespects": _fallen_allies,
    "tripleaxel": _triple_axel,
    "ragefist": _rage_fist,
})


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
    return damage_from_base(
        damage_base(power=power, attack=attack, defense=defense, crit=crit,
                    spread=spread, level=level),
        roll, stab=stab, effectiveness=effectiveness)


def damage_base(
    *,
    power: int,
    attack: int,
    defense: int,
    crit: bool = False,
    spread: bool = False,
    level: int = LEVEL,
) -> int:
    """Everything above the damage roll, which the roll does not touch.

    Split out for the estimator, which needs sixteen rolls of the same move
    against the same defence and was recomputing all of this for each one --
    two multiplications and three floor divisions, sixteen times, for a number
    that could not change. The estimator asks for the base once and rolls it.

    Not a second copy of the formula: ``damage_formula`` is these two halves
    composed, so the engine and the estimator still run one set of steps in one
    order. A test holds them equal.
    """
    damage = ((2 * level // 5 + 2) * power * attack // defense) // 50 + 2

    # Showdown applies the spread penalty here -- before the crit multiplier and
    # before the roll, not at the end (``Battle#modifyDamage``).
    if spread:
        damage = chain_modify(damage, SPREAD_MODIFIER)
    if crit:
        damage = damage * CRIT_MULTIPLIER_NUM // CRIT_MULTIPLIER_DEN
    return damage


def damage_from_base(
    base: int,
    roll: int,
    *,
    stab: bool = False,
    effectiveness: float = 1.0,
) -> int:
    """The half of the formula the damage roll reaches."""
    damage = base * roll // 100
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

    # A few moves read a stat their category does not say. Body Press swings
    # with the user's Defense, Foul Play with the *target's* Attack, and the
    # Psyshock family lands on Defense while staying Special (so Special
    # Defense boosts do nothing and a burned attacker still deals full
    # damage). The dex spells all three out; the formula just has to listen.
    override = move.raw.get("overrideOffensiveStat")
    if override:
        attack_stat = _STAT_BY_NAME[override]
    if move.raw.get("overrideDefensiveStat"):
        defense_stat = _STAT_BY_NAME[move.raw["overrideDefensiveStat"]]
    striker = defender if move.raw.get("overrideOffensivePokemon") == "target" else attacker

    attack = effective_stat(ctx, striker, attack_stat, move=move, opponent=defender)
    if move.raw.get("ignoreDefensive"):
        # Sacred Sword and Darkest Lariat read the raw stat: no stages, in
        # either direction. The hooks still run -- an ability multiplier is
        # not a stage.
        raw_value = fx.modify(ctx, "modify_stat",
                              mutate.raw_stat(ctx.state, defender, defense_stat),
                              defender, stat=defense_stat, move=move,
                              opponent=attacker)
        defense = max(1, int(fx.modify(ctx, "modify_boosted_stat", raw_value,
                                       defender, stat=defense_stat, move=move,
                                       opponent=attacker)))
    else:
        defense = effective_stat(ctx, defender, defense_stat, move=move, opponent=attacker)

    # A critical hit ignores the defender's positive stages and the attacker's
    # negative ones -- it recomputes without the stages that would have helped
    # the defender or hurt the attacker.
    if crit:
        attack = max(attack, _unstaged(ctx, striker, attack_stat, move, defender, True))
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


#: How the dex names stats in its override fields.
_STAT_BY_NAME = {"atk": Stat.ATK, "def": Stat.DEF, "spa": Stat.SPA,
                 "spd": Stat.SPD, "spe": Stat.SPE}


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
    evasion = (0 if move.raw.get("ignoreEvasion")
               else ctx.state.sides[defender[0]].boost(defender[1], "evasion"))
    stage = (
        ctx.state.sides[attacker[0]].boost(attacker[1], "accuracy")
        - evasion
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
            _note_move_failed(ctx, attacker, True)
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
            # "사용한 턴에" -- the turn it is used, which is this one whether or
            # not the charge is skipped.
            paid = CHARGE_TURN_BOOST.get(move.id)
            if paid:
                mutate.boost(ctx, attacker, paid, source=attacker)
            skipped = ctx.state.field.weather in CHARGE_SKIPS.get(move.id, ())
            if not skipped:
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
        _note_move_failed(ctx, attacker, True)
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

    # Steel Beam pays its half-max-HP cost for swinging at all -- hit or
    # miss -- once a target was in front of it.
    if move.raw.get("mindBlownRecoil") and move.category != "Status":
        apply_damage(ctx, attacker, (mutate.max_hp(ctx.state, attacker) + 1) // 2,
                     "recoil", detail=move.id)

    if targets_opponent and not connects(ctx, attacker, defender, move):
        ctx.emit(ev.missed(attacker[0], attacker[1], move.id))
        # High Jump Kick's gamble: missing costs half the user's own HP.
        if move.raw.get("hasCrashDamage"):
            ctx.emit(Event("crash", side=attacker[0], slot=attacker[1], move=move.id))
            apply_damage(ctx, attacker, (mutate.max_hp(ctx.state, attacker) + 1) // 2,
                         "recoil", detail="crash")
        _note_move_failed(ctx, attacker, True)
        return

    if move.category == "Status":
        landed = _apply_status_move(ctx, attacker, target, move)
    else:
        landed = _apply_damaging_move(ctx, attacker, defender, move)

    _note_move_failed(ctx, attacker, not landed)
    if landed:
        _apply_self_effects(ctx, attacker, move)


def _note_move_failed(ctx: Context, ref: Ref, failed: bool) -> None:
    """The one bit Stomping Tantrum and Temper Flare ask about.

    An approximation of Showdown's ``pokemonLastMoveFailed``: set when the
    move missed, was blocked, or reported failure; cleared when it landed.
    The early exits in ``use_move`` (no target, precondition refusals) set it
    too, at their own emit sites.
    """
    ctx.state.sides[ref[0]].volatiles[ref[1]]["lastmovefailed"] = failed


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

    if move.id == "pollenpuff" and defender[0] == attacker[0] and defender != attacker:
        # Champions: 같은 편에게 사용하면 데미지를 주는 대신 최대 HP의 1/2만큼
        # 같은 편의 HP를 회복한다. Before this it damaged its own partner.
        healed = mutate.heal(ctx, defender,
                             mutate.fraction_of_max(ctx.state, defender, 2),
                             reason=move.id)
        if not healed:
            ctx.emit(Event("move_failed", side=attacker[0], move=move.id,
                           detail="already full"))
            return False
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
        # ``move`` is the per-use ActiveMove, so the index is ours to set.
        # Beat Up reads it to pick whose Attack swings; Triple Axel reads it
        # to ramp its power.
        move.hit_index = hit_number
        # Triple Axel and Population Bomb roll accuracy for every hit, not
        # once for the lot -- the second miss ends the move, keeping whatever
        # already landed.
        if hit_number > 0 and move.raw.get("multiaccuracy") \
                and not connects(ctx, attacker, defender, move):
            ctx.emit(ev.missed(attacker[0], attacker[1], move.id))
            break
        crit = rolls_crit(ctx, attacker, defender, move)
        any_crit = any_crit or crit
        # Judged before the number even exists: a Focus Sash fires inside
        # ``compute_damage``'s modify hooks and spends itself doing it, so a
        # check made any later finds no item and calls the truncated number
        # clean. It is not, and it once struck the true set off the pool for
        # a Moonblast the sash had quietly shaved from 194 to 184.
        clean = hits == 1 and _hit_is_recordable(ctx, attacker, defender, move, crit)
        damage, effectiveness = compute_damage(ctx, attacker, defender, move, crit)
        if getattr(move, "parental_bond", False) and hit_number > 0:
            damage = max(1, chain_modify(damage, X0_25))
        if effectiveness == 0.0:
            ctx.emit(ev.immune(defender[0], defender[1], move.id))
            return False
        dealt = _deal_or_break_substitute(ctx, attacker, defender, move, damage, effectiveness, crit)
        if clean and dealt > 0:
            _record_hit(ctx, attacker, defender, move, dealt)
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
        # Scald and friends thaw whoever they hit.
        if move.raw.get("thawsTarget") \
                and ctx.state.sides[defender[0]].status[defender[1]] == "frz":
            mutate.cure_status(ctx, defender)
        _apply_secondaries(ctx, attacker, defender, move)
        # Both machines fixed Scale Shot's unread selfBoost on the same day;
        # the merge briefly applied it twice and dropped Garchomp two stages
        # of Defence per use. One implementation stays -- the named one.
        _apply_self_boost(ctx, attacker, move)
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


#: The events whose handlers change what the analytic damage formula says.
#: ``modify_move`` belongs here even though it never touches a number
#: directly: Pixilate rewrites Hyper Voice's *type*, which rewrites STAB and
#: effectiveness, and a Sylveon priced as throwing Normal at a Dragon was
#: struck off the pool five times in a hundred and twenty battles before this
#: line existed.
_DAMAGE_EVENTS = frozenset({"modify_damage", "modify_base_power",
                            "modify_stat", "modify_boosted_stat",
                            "modify_effectiveness", "modify_attack",
                            "modify_move"})

#: Attacker items whose effect the belief pricer reproduces exactly, so a hit
#: through them is still invertible. Everything else that touches damage
#: silences the recorder.
PRICED_ITEMS = frozenset({
    "lifeorb", "expertbelt",
    "blackglasses", "spelltag", "mysticwater", "fairyfeather", "charcoal",
    "miracleseed", "magnet", "sharpbeak", "softsand", "silkscarf",
    "hardstone", "silverpowder", "dragonfang", "metalcoat", "twistedspoon",
    "nevermeltice", "poisonbarb", "blackbelt",
})


#: Attacker abilities the belief pricer reproduces exactly, so their hits are
#: still invertible and stay in the ledger. The doubled-Attack pair is flat;
#: the skin family rewrites a move's type (and power) deterministically --
#: Liquid Voice Primarina and Pixilate Sylveon are built around exactly these
#: moves, and silencing them threw away the observations that identify them.
PRICED_ATTACKER_ABILITIES = frozenset({
    "hugepower", "purepower",
    "pixilate", "refrigerate", "aerilate", "galvanize", "dragonize",
    "normalize", "liquidvoice",
})


def _touches_damage(kind: str, effect_id) -> bool:
    from pkcm.engine.effects import lookup

    effect = lookup(kind, effect_id)
    return effect is not None and bool(_DAMAGE_EVENTS & set(effect.handlers))


#: How many hits the ledger keeps. Oldest fall off; late hits carry the most
#: current information anyway, and the cap keeps clones cheap.
OBSERVED_HITS_CAP = 24


def _hit_is_recordable(ctx: Context, attacker: Ref, defender: Ref,
                       move: Move, crit: bool) -> bool:
    """Whether the analytic formula is the whole story of this hit.

    Judged **before** the damage is applied, because the things that falsify
    the formula can consume themselves in the act -- a Focus Sash has already
    left the item slot by the time the number exists. The belief inverts the
    formula to eliminate candidate sets, and a hit it cannot price must not
    be allowed to eliminate anyone; the check errs toward silence, since a
    skipped clean hit costs one observation and a recorded dirty one costs
    the pool its true set.
    """
    state = ctx.state
    if crit or getattr(move, "spread", False):
        return False
    if state.field.weather is not None or state.field.terrain is not None:
        return False
    if move.raw.get("basePowerCallback") or move.raw.get("multihit") \
            or move.raw.get("damage") or move.id in VARIABLE_POWER:
        return False
    # Body Press, Foul Play, the Psyshock family: the stat they read is not
    # the one the pricer would use, so their numbers stay out of the ledger.
    if move.raw.get("overrideOffensiveStat") \
            or move.raw.get("overrideOffensivePokemon") \
            or move.raw.get("overrideDefensiveStat"):
        return False
    physical = move.category == "Physical"
    attack_boost = "atk" if physical else "spa"
    defense_boost = "def" if physical else "spd"
    if state.sides[attacker[0]].boost(attacker[1], attack_boost):
        return False
    if state.sides[defender[0]].boost(defender[1], defense_boost):
        return False
    if physical and state.sides[attacker[0]].status[attacker[1]] == "brn":
        return False
    screens = state.sides[defender[0]].conditions
    if "reflect" in screens or "lightscreen" in screens or "auroraveil" in screens:
        return False
    # Anything in force on either side whose handlers can touch the number --
    # an ability like Multiscale, an item, a volatile like Charge -- and the
    # analytic formula is no longer the whole story. Asked of the registry
    # rather than written down as a list, so implementing a new modifier
    # cannot quietly leave this check behind. The attacker items the pricer
    # reproduces exactly stay recordable.
    attacker_ability = ctx.ability_of(attacker)
    if attacker_ability not in PRICED_ATTACKER_ABILITIES             and _touches_damage("ability", attacker_ability):
        return False
    if _touches_damage("ability", ctx.ability_of(defender)):
        return False
    if _touches_damage("item", ctx.item_of(defender)):
        return False
    attacker_item = ctx.item_of(attacker)
    if attacker_item not in PRICED_ITEMS and _touches_damage("item", attacker_item):
        return False
    for holder in (attacker, defender):
        for name in state.sides[holder[0]].volatiles[holder[1]]:
            if _touches_damage("volatile", name):
                return False
    return True


def _record_hit(ctx: Context, attacker: Ref, defender: Ref,
                move: Move, dealt: int) -> None:
    state = ctx.state
    entry = (
        defender[0], defender[1],
        attacker[0], attacker[1],
        state.species_id(*attacker),        # the forme that actually swung
        move.id,
        dealt,
        # A knockout truncates: ``apply_damage`` returns what the HP bar
        # could absorb, not what the formula rolled. The number is then a
        # floor -- the roll was *at least* this -- and the pricer has to test
        # it as one, or every finishing blow eliminates its own attacker.
        state.sides[defender[0]].hp[defender[1]] == 0,
        # The defender as it stood when the hit landed. Snapshotted here, not
        # at observation time: a defender that Mega Evolves afterwards keeps
        # its old bulk in the ledger, because that is the bulk the number
        # answers to.
        state.species_id(*defender),
        tuple(state.stats(*defender)),
        tuple(state.types(*defender)),
    )
    kept = state.observed_hits
    if len(kept) >= OBSERVED_HITS_CAP:
        kept = kept[1:]
    state.observed_hits = kept + (entry,)


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
        from pkcm.engine.conditions import add_side_condition

        side_index = (attacker[0] if move.target in SELF_TARGETS or move.target == "allySide"
                      else 1 - attacker[0])
        changed |= add_side_condition(ctx, side_index, _to_id(condition), attacker)

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


def _apply_self_boost(ctx: Context, attacker: Ref, move) -> None:
    """``selfBoost: {boosts}`` -- once, after the last hit, if the move landed.

    A separate field from ``self`` and nothing here read it, so two moves in
    M-B changed no stat at all: Scale Shot never traded Defence for Speed, and
    Clanging Scales cost nothing to use. Both are silent failures -- the move
    does its damage and the stat simply does not move.

    Its own field because of *when* it lands. ``self`` applies per hit and
    ``selfBoost`` applies once at the end, which is the difference between
    Scale Shot dropping one stage of Defence and dropping five.
    """
    payload = move.raw.get("selfBoost")
    if isinstance(payload, dict) and payload.get("boosts"):
        mutate.boost(ctx, attacker, payload["boosts"], source=attacker)


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
