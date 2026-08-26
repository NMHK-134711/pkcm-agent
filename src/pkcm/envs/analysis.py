"""The sums a strong player does in their head, from what they can actually see.

"Is this a guaranteed two-hit KO?" "Do I outspeed?" "Does this resist my STAB?"
Those questions decide most turns, and a human answers them with a dex, a
calculator and an assumption about the opponent's spread. None of it is hidden
information -- it is arithmetic over public numbers -- so a policy that has to
rediscover it from reward is being made to work for something it could be told.

Everything here takes an ``Observation``, never a ``BattleState``. That is the
guarantee worth having: a tool that could see the truth would quietly launder
it into the policy, and the policy would look like it had learned to read minds.

**The opponent's spread is unknown, so answers come as brackets.** Their
Defence is somewhere between "0 SP, hindering nature" and "32 SP, boosting
nature", and Champions' SP cap makes that a *narrow* bracket -- narrow enough
that "guaranteed 2HKO" is usually still a definite answer. Reporting a range
rather than a point estimate is what an honest calculator does, and it is what
a player means by "확정 2타".
"""

from __future__ import annotations

from dataclasses import dataclass

from pkcm.data.dex import Dex, Move, Stat
from pkcm.engine.moves import (
    DAMAGE_ROLL_HIGH,
    DAMAGE_ROLL_LOW,
    damage_formula,
)
from pkcm.engine.stats import LEVEL, NATURES, compute_stat
from pkcm.envs.observation import KnownPokemon, Observation
from pkcm.envs.reference import ReferenceSheet

#: Champions caps SP at 32 per stat, so the gap between the least and most
#: invested version of the same species is small -- much smaller than the EV
#: spreads of the main series. That is why bracketing works here.
SP_CAP = 32
BOOST_NATURE = 110
HINDER_NATURE = 90


@dataclass(frozen=True, slots=True)
class Bracket:
    """A quantity we can only bound, because it depends on a hidden spread."""

    low: int
    high: int

    @property
    def certain(self) -> bool:
        return self.low == self.high

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.low}" if self.certain else f"{self.low}-{self.high}"


#: Every damage roll, as Showdown draws it: sixteen uniform values from 85% to
#: 100%. The whole reason "확정 1타" and "난수 1타" are different words.
ROLL_COUNT = DAMAGE_ROLL_HIGH - DAMAGE_ROLL_LOW + 1


@dataclass(frozen=True, slots=True)
class DamageEstimate:
    """What one move would do to one target, as far as we can tell."""

    move_id: str
    effectiveness: float
    #: Damage as a percentage of the target's maximum HP, worst and best case
    #: across both the damage roll and the spreads they might be running.
    percent: Bracket
    #: Hits needed, worst and best case. ``high`` is the pessimistic answer.
    hits_to_ko: Bracket
    #: True when even the unluckiest roll against the bulkiest spread kills.
    guaranteed_ko: bool
    #: The probability this move knocks the target out **this turn**, counting
    #: the sixteen damage rolls, the accuracy check and the critical hit. This
    #: is the number a player means by "난수 1타 43%", and the reason a bracket
    #: alone is not enough: 3-6 hits does not say whether to go for it.
    ko_chance: float = 0.0
    #: The same probability at their bulkiest and frailest plausible spread.
    ko_chance_bracket: tuple[float, float] = (0.0, 0.0)
    #: Chance the move connects at all, after the accuracy we can see.
    hit_chance: float = 1.0
    #: The sixteen rolls themselves, at the middle of the spread bracket. A
    #: policy that wants the shape of the distribution rather than a summary.
    rolls: tuple[int, ...] = ()

    @property
    def immune(self) -> bool:
        return self.effectiveness == 0.0

    @property
    def certain_ko(self) -> bool:
        """확정타. Every roll kills, even against their bulkiest spread.

        Says nothing about connecting -- that is ``hit_chance``, and a move can
        be a 확정 1타 with 80 accuracy. Keeping them apart is the point: they
        are answered by different decisions.
        """
        return self.guaranteed_ko

    @property
    def likely_ko(self) -> bool:
        """난수타 worth taking -- more often than not, but not certain."""
        return not self.guaranteed_ko and self.ko_chance >= 0.5


def defensive_bracket(dex: Dex, species_id: str, stat: Stat) -> tuple[int, int]:
    """The least and most a species could have in one stat.

    Public: base stats are in the dex, and the SP and nature limits are the
    format's. Nothing about the specific Pokemon in front of us is used.
    """
    base = dex.species[species_id].base_stats[stat]
    if stat is Stat.HP:
        return base + 75, base + 75 + SP_CAP
    return ((base + 20) * HINDER_NATURE // 100,
            (base + 20 + SP_CAP) * BOOST_NATURE // 100)


def _effective_stat(known: KnownPokemon, dex: Dex, stat: Stat) -> int:
    """A stat with its stage applied.

    Exact when we own the Pokemon -- the observation carries our real numbers,
    because the game shows them to us. For anything else this falls back to the
    middle of the bracket, which is the same guess a player makes.
    """
    from pkcm.engine.mutate import stage_multiplier
    from pkcm.engine.state import BOOST_INDEX

    if known.stats is not None:
        raw = known.stats[stat]
    else:
        base = dex.species[known.species_id].base_stats[stat]
        raw = base + 75 + SP_CAP // 2 if stat is Stat.HP else base + 20 + SP_CAP // 2
    if stat is Stat.HP:
        return max(1, raw)
    stage = known.boosts[BOOST_INDEX[stat.name.lower()]]
    return max(1, int(raw * stage_multiplier(stage)))


def estimate_damage(
    observation: Observation,
    sheet: ReferenceSheet,
    dex: Dex,
    attacker: KnownPokemon,
    defender: KnownPokemon,
    move: Move,
    spread: bool = False,
) -> DamageEstimate | None:
    """One move against one target, bracketed over what we cannot see.

    Returns ``None`` for a move that deals no damage -- a status move has no
    number to give and pretending otherwise would put a zero where a policy
    might read it as "this move is bad".
    """
    if move.category == "Status" or attacker.species_id is None:
        return None
    if defender.species_id is None:
        return None  # never seen; there is nothing to calculate against

    defender_types = dex.species[defender.species_id].types
    effectiveness = sheet.effectiveness(move.type, defender_types)

    if effectiveness == 0.0:
        return DamageEstimate(move.id, 0.0, Bracket(0, 0), Bracket(99, 99), False,
                              ko_chance=0.0, hit_chance=0.0)

    attack_stat = Stat.ATK if move.category == "Physical" else Stat.SPA
    defense_stat = Stat.DEF if move.category == "Physical" else Stat.SPD
    attack = _effective_stat(attacker, dex, attack_stat)

    defense_low, defense_high = defensive_bracket(dex, defender.species_id, defense_stat)
    hp_low, hp_high = defensive_bracket(dex, defender.species_id, Stat.HP)
    stab = move.type in dex.species[attacker.species_id].types
    power = move.base_power or 0
    if power <= 0:
        return None

    def rolls_against(defense: int, crit: bool = False) -> tuple[int, ...]:
        return tuple(
            damage_formula(power=power, attack=attack, defense=defense, roll=roll,
                           crit=crit, spread=spread, stab=stab,
                           effectiveness=effectiveness)
            for roll in range(DAMAGE_ROLL_LOW, DAMAGE_ROLL_HIGH + 1)
        )

    defense_mid = (defense_low + defense_high) // 2
    hp_mid = (hp_low + hp_high) // 2

    bulky, frail, middle = (rolls_against(defense_high), rolls_against(defense_low),
                            rolls_against(defense_mid))
    middle_crit = rolls_against(defense_mid, crit=True)

    remaining_low = max(1, int(hp_low * defender.hp_fraction))
    remaining_high = max(1, int(hp_high * defender.hp_fraction))
    remaining_mid = max(1, int(hp_mid * defender.hp_fraction))

    hit_chance = 1.0 if move.accuracy is None else min(100, move.accuracy) / 100.0
    crit_chance = _crit_chance(move)

    def knockout(rolls: tuple[int, ...], crit_rolls: tuple[int, ...], hp: int) -> float:
        plain = sum(1 for damage in rolls if damage >= hp) / len(rolls)
        critical = sum(1 for damage in crit_rolls if damage >= hp) / len(crit_rolls)
        return hit_chance * ((1 - crit_chance) * plain + crit_chance * critical)

    return DamageEstimate(
        move_id=move.id,
        effectiveness=effectiveness,
        percent=Bracket(_percent(bulky[0], hp_high), _percent(frail[-1], hp_low)),
        hits_to_ko=Bracket(_hits(frail[-1], remaining_low),
                           _hits(bulky[0], remaining_high)),
        # 확정 1타 is a statement about the damage roll, not about accuracy.
        # A player says "확정 1타지만 명중 80" -- two facts, kept apart, because
        # they are answered by different decisions.
        guaranteed_ko=bulky[0] >= remaining_high,
        ko_chance=knockout(middle, middle_crit, remaining_mid),
        ko_chance_bracket=(
            knockout(bulky, rolls_against(defense_high, crit=True), remaining_high),
            knockout(frail, rolls_against(defense_low, crit=True), remaining_low),
        ),
        hit_chance=hit_chance,
        rolls=middle,
    )


def _crit_chance(move: Move) -> float:
    """How often this move crits, from its own crit ratio.

    Ignores Focus Energy and the like: those live on the attacker, and the
    observation does carry its volatiles, but the extra fidelity is not worth
    a second code path here -- the engine remains the authority.
    """
    from pkcm.engine.moves import CRIT_DENOMINATOR, NEVER_CRITS

    ratio = move.raw.get("critRatio", 1)
    if move.raw.get("willCrit"):
        return 1.0
    if ratio <= NEVER_CRITS:
        return 0.0
    denominator = CRIT_DENOMINATOR.get(ratio, 1)
    return 1.0 if denominator <= 1 else 1.0 / denominator


def _percent(damage: int, maximum: int) -> int:
    return min(100, round(100 * damage / maximum)) if maximum else 0


def _hits(damage: int, hp: int) -> int:
    return 99 if damage <= 0 else min(99, -(-hp // damage))


# --------------------------------------------------------------------------- #
# The whole picture, for one decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Assessment:
    """Everything the calculator can say about one of our field positions."""

    position: int
    attacker_slot: int
    #: One row per (our move, their standing Pokemon).
    damage: tuple[tuple[int, DamageEstimate], ...]
    #: What each of their revealed actives could do back, using the same
    #: bracketing -- their base stats are public, their spreads are not.
    incoming: tuple[tuple[int, DamageEstimate], ...]
    #: Whether we move first, as far as base Speed and stages can say.
    outspeeds: tuple[tuple[int, bool | None], ...]


def assess(observation: Observation, sheet: ReferenceSheet, dex: Dex,
           position: int = 0) -> Assessment | None:
    """The calculator's full answer for one position, from the observation only."""
    attacker = next((known for known in observation.own if known.position == position),
                    None)
    if attacker is None or attacker.species_id is None:
        return None

    targets = [known for known in observation.foe if known.position is not None
               and not known.fainted]

    damage: list[tuple[int, DamageEstimate]] = []
    for move_id in attacker.moves:
        move = dex.moves.get(move_id)
        if move is None:
            continue
        spread = move.target in ("allAdjacentFoes", "allAdjacent") and len(targets) > 1
        for target in targets:
            estimate = estimate_damage(observation, sheet, dex, attacker, target,
                                       move, spread=spread)
            if estimate is not None:
                damage.append((target.slot, estimate))

    incoming: list[tuple[int, DamageEstimate]] = []
    for target in targets:
        for move_id in target.moves:      # only what we have watched them use
            move = dex.moves.get(move_id)
            if move is None:
                continue
            estimate = estimate_damage(observation, sheet, dex, target, attacker, move)
            if estimate is not None:
                incoming.append((target.slot, estimate))

    return Assessment(
        position=position,
        attacker_slot=attacker.slot,
        damage=tuple(damage),
        incoming=tuple(incoming),
        outspeeds=tuple((target.slot, outspeeds(dex, attacker, target))
                        for target in targets),
    )


def outspeeds(dex: Dex, ours: KnownPokemon, theirs: KnownPokemon) -> bool | None:
    """Do we move first? ``None`` when their spread could decide it either way.

    The honest three-state answer. A player who says "I outspeed" usually means
    "I outspeed anything they could plausibly be running", and that is exactly
    the case where both ends of the bracket agree.
    """
    if ours.species_id is None or theirs.species_id is None:
        return None
    from pkcm.engine.mutate import stage_multiplier
    from pkcm.engine.state import BOOST_INDEX

    index = BOOST_INDEX["spe"]
    mine = _effective_stat(ours, dex, Stat.SPE)
    low, high = defensive_bracket(dex, theirs.species_id, Stat.SPE)
    stage = stage_multiplier(theirs.boosts[index])
    if mine > int(high * stage):
        return True
    if mine < int(low * stage):
        return False
    return None
