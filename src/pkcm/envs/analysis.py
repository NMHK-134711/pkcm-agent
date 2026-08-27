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
    #: The target's ability is unknown and one it could have would soften this.
    #: Multiscale is the case that matters: a knockout promised into a full-HP
    #: Dragonite is half a knockout if it turns out to have it.
    blunted_possible: bool = False
    #: Something could refuse the knockout outright -- Sturdy, a Focus Sash, or
    #: a Disguise still intact.
    survivable: bool = False

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


def midpoint(dex: Dex, species_id: str, stat: Stat | int) -> int:
    """The middle of ``defensive_bracket``. The guess a player makes.

    Public by construction -- base stats from the dex and the format's SP and
    nature limits, nothing about the Pokemon in front of us. Takes a plain index
    as well as a ``Stat`` so callers outside this module need not import the
    enum to ask about the opponent's bulk.
    """
    low, high = defensive_bracket(dex, species_id, Stat(stat))
    return (low + high) // 2


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

    # An ability that has announced itself is not hidden information any more,
    # so use it. One that has not is left out of the number and reported as a
    # risk instead -- see ``blunted_possible``.
    if defender.ability_known:
        modifier = defender_multiplier(defender.ability, move, defender, effectiveness)
        if modifier is None:
            return DamageEstimate(move.id, 0.0, Bracket(0, 0), Bracket(99, 99), False,
                                  ko_chance=0.0, hit_chance=0.0)
        effectiveness *= modifier

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
        blunted_possible=could_blunt(defender, dex),
        survivable=could_survive_a_kill(defender, dex),
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
        outspeeds=tuple(
            (target.slot, outspeeds(dex, attacker, target, observation))
            for target in targets
        ),
    )


# --------------------------------------------------------------------------- #
# Who moves first
#
# The hardest of these sums, and the one most worth getting right: nearly every
# turn is decided by it. The naive version -- compare base Speed -- is wrong
# about paralysis, Tailwind, a Choice Scarf, half a dozen weather abilities, and
# Trick Room, which reverses the whole thing.
#
# The modifiers are mirrored from the engine rather than invented, and a test
# compares this against ``mutate.effective_stat`` on real battles. Two
# implementations of the same arithmetic drift; a test is what keeps them
# honest, since the estimator cannot call the engine's version (it would need
# the battle state, and then it could see everything).
# --------------------------------------------------------------------------- #

#: ability -> (multiplier as a fraction, the field condition it needs).
#: From ``abilities._SPEED_ABILITIES``.
SPEED_ABILITIES = {
    "swiftswim": (2.0, ("weather", "raindance")),
    "chlorophyll": (2.0, ("weather", "sunnyday")),
    "sandrush": (2.0, ("weather", "sandstorm")),
    "slushrush": (2.0, ("weather", "snowscape")),
    "surgesurfer": (2.0, ("terrain", "electricterrain")),
    "quickfeet": (1.5, ("status", None)),
}
#: item -> multiplier.
SPEED_ITEMS = {"choicescarf": 1.5, "ironball": 0.5}
#: Paralysis halves Speed. Champions nerfed the *chance*, not this.
PARALYSIS_SPEED = 0.5
TAILWIND_SPEED = 2.0


def _field_gives(observation: Observation, requirement, known: KnownPokemon) -> bool:
    kind, value = requirement
    if kind == "weather":
        return observation.weather == value
    if kind == "terrain":
        return observation.terrain == value
    if kind == "status":
        return known.status is not None
    return False


def speed_of(observation: Observation, known: KnownPokemon, dex: Dex,
             ours: bool) -> Bracket:
    """Effective Speed, as a bracket over what we cannot see.

    Ours collapses to a point: we know the stat, the item and the ability.
    Theirs stays a range, and the range is wide when a Choice Scarf is still
    possible -- which is the honest answer, and the reason a strong player
    plays around a Scarf rather than assuming one way or the other.
    """
    from pkcm.engine.mutate import stage_multiplier
    from pkcm.engine.state import BOOST_INDEX

    if known.species_id is None:
        return Bracket(0, 0)

    if known.stats is not None:
        low = high = known.stats[Stat.SPE]
    else:
        low, high = defensive_bracket(dex, known.species_id, Stat.SPE)

    stage = stage_multiplier(known.boosts[BOOST_INDEX["spe"]])
    low, high = int(low * stage), int(high * stage)

    if known.status == "par":
        low, high = int(low * PARALYSIS_SPEED), int(high * PARALYSIS_SPEED)

    conditions = observation.own_conditions if ours else observation.foe_conditions
    if any(name == "tailwind" for name, _ in conditions):
        low, high = int(low * TAILWIND_SPEED), int(high * TAILWIND_SPEED)

    # Item and ability: exact for us, a spread of possibilities for them.
    factors = [1.0]
    if known.item_known:
        factors = [SPEED_ITEMS.get(known.item, 1.0)]
    else:
        factors = [1.0, SPEED_ITEMS["choicescarf"]]

    ability_factors = [1.0]
    if known.ability_known:
        found = SPEED_ABILITIES.get(known.ability)
        if found and _field_gives(observation, found[1], known):
            ability_factors = [found[0]]
    else:
        from pkcm.engine.legality import registrable_abilities

        for candidate in registrable_abilities(dex.species[known.species_id]):
            found = SPEED_ABILITIES.get(candidate)
            if found and _field_gives(observation, found[1], known):
                ability_factors.append(found[0])

    lowest = min(factors) * min(ability_factors)
    highest = max(factors) * max(ability_factors)
    return Bracket(int(low * lowest), int(high * highest))


def outspeeds(dex: Dex, ours: KnownPokemon, theirs: KnownPokemon,
              observation: Observation | None = None,
              our_priority: int = 0, their_priority: int = 0) -> bool | None:
    """Do we move first? ``None`` when what we cannot see could decide it.

    Priority beats Speed outright, so it is answered first. Trick Room reverses
    Speed and leaves priority alone, which is why it is applied to the
    comparison rather than to the numbers.
    """
    if our_priority != their_priority:
        return our_priority > their_priority
    if ours.species_id is None or theirs.species_id is None or observation is None:
        return None

    mine = speed_of(observation, ours, dex, ours=True)
    yours = speed_of(observation, theirs, dex, ours=False)
    reversed_order = any(name == "trickroom" for name, _ in observation.rooms)

    if mine.low > yours.high:
        return not reversed_order
    if mine.high < yours.low:
        return reversed_order
    return None


# --------------------------------------------------------------------------- #
# Abilities that change what a hit does
#
# hk asked about Mimikyu's Disguise and Dragonite's Multiscale specifically, and
# they are the two shapes: one refuses a hit outright, the other halves it under
# a condition. The engine has both right. The *estimator* did not have either,
# which meant a calculator that would happily promise a knockout into a
# Multiscale Dragonite at full HP.
#
# Two cases, and keeping them apart is the whole job:
#
#   known    -- something announced it, so apply it exactly.
#   unknown  -- it is one of the abilities that species is allowed. Do not
#               apply it, but say that the number might be wrong, because a
#               strong player thinks "얘 멀티스케일일 수도 있는데" and prices it.
#
# The tables below are the numeric effects. Which roster abilities touch
# incoming damage *at all* is read out of the effect registry instead of being
# listed here, so a newly implemented ability cannot be silently missed -- a
# test asserts every one of them is accounted for.
# --------------------------------------------------------------------------- #

#: Halves damage while the holder is untouched. Multiscale and its clone.
FULL_HP_HALVERS = frozenset({"multiscale", "shadowshield"})

#: ability -> (move types it softens, multiplier).
TYPE_SOFTENERS = {
    "thickfat": (("fire", "ice"), 0.5),
    "heatproof": (("fire",), 0.5),
    "waterbubble": (("fire",), 0.5),
    "purifyingsalt": (("ghost",), 0.5),
}

#: Quarter off anything super effective.
SUPER_EFFECTIVE_SOFTENERS = frozenset({"filter", "solidrock", "prismarmor"})

#: ability -> (category it halves).
CATEGORY_SOFTENERS = {
    "furcoat": "Physical",
    "icescales": "Special",
}

#: ability -> the move type it absorbs outright.
ABSORBS_TYPE = {
    "flashfire": "fire",
    "wellbakedbody": "fire",
    "waterabsorb": "water",
    "dryskin": "water",
    "stormdrain": "water",
    "voltabsorb": "electric",
    "lightningrod": "electric",
    "motordrive": "electric",
    "sapsipper": "grass",
    "eartheater": "ground",
    "levitate": "ground",
}

#: ability -> the move flag it refuses.
BLOCKS_FLAG = {
    "bulletproof": "bullet",
    "soundproof": "sound",
    "overcoat": "powder",
}

#: Abilities in the registry's "touches incoming damage" set that work for the
#: *attacker*, not the defender. Named so the completeness test can tell the
#: difference between "handled" and "forgotten".
ATTACKER_SIDE_ABILITIES = frozenset({
    "adaptability", "sniper", "scrappy", "megasol", "waterbubble",
    "magicbounce", "telepathy", "goodasgold", "armortail", "queenlymajesty",
})


def defender_multiplier(ability: str | None, move: Move, defender: KnownPokemon,
                        effectiveness: float) -> float | None:
    """What the defender's ability does to this hit.

    ``None`` means it stops the hit entirely. ``1.0`` means it does nothing.
    Only ever called with an ability we actually know about.
    """
    if ability is None:
        return 1.0
    if ABSORBS_TYPE.get(ability) == move.type:
        return None
    if BLOCKS_FLAG.get(ability) in move.flags:
        return None
    if ability in FULL_HP_HALVERS and defender.hp_fraction >= 1.0:
        return 0.5
    if ability in SUPER_EFFECTIVE_SOFTENERS and effectiveness > 1.0:
        return 0.75
    softened = TYPE_SOFTENERS.get(ability)
    if softened is not None and move.type in softened[0]:
        return softened[1]
    if CATEGORY_SOFTENERS.get(ability) == move.category:
        return 0.5
    if ability == "fluffy":
        if move.type == "fire":
            return 2.0
        if "contact" in move.flags:
            return 0.5
    return 1.0


def blunting_abilities() -> frozenset[str]:
    """Every roster ability that interferes with an incoming hit.

    Read out of the effect registry rather than written down, so implementing a
    new one cannot leave this list quietly behind.
    """
    from pkcm.engine import abilities as _abilities  # noqa: F401  -- fills REGISTRY
    from pkcm.engine.effects import REGISTRY

    interferes = {"modify_damage", "modify_effectiveness", "try_hit"}
    return frozenset(
        ability_id
        for (kind, ability_id), effect in REGISTRY.items()
        if kind == "ability"
        and interferes & set(effect.handlers)
        and ability_id not in ATTACKER_SIDE_ABILITIES
    )


def could_blunt(defender: KnownPokemon, dex: Dex) -> bool:
    """Might an ability we cannot see make this hit land softer than the sum says?

    True when the ability is unknown and at least one the species is allowed to
    have would interfere. This is the flag that stops a policy from trusting a
    knockout it has not earned.
    """
    if defender.ability_known or defender.species_id is None:
        return False
    from pkcm.engine.legality import registrable_abilities

    possible = registrable_abilities(dex.species[defender.species_id])
    return bool(set(possible) & blunting_abilities())


# --------------------------------------------------------------------------- #
# Losing the turn
#
# A turn taken away costs more than most damage, and every way of losing one is
# a die roll with a published number. A policy that has to learn "paralysis is
# bad" from reward will learn it eventually and expensively; the number is right
# here and it is not hidden.
#
# The engine owns these constants. They are imported rather than repeated, so a
# rules change moves them in one place -- Champions already nerfed paralysis
# from 1/4 to 1/8, and a second copy would still be saying 1/4.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TurnRisk:
    """How likely we are not to act, and why."""

    paralysis: float = 0.0
    sleep: float = 0.0
    freeze: float = 0.0
    confusion: float = 0.0
    #: Already flinched this turn -- certain, not a probability.
    flinched: float = 0.0

    @property
    def cannot_act(self) -> float:
        """Any of them happening. They are mutually exclusive in practice: a
        Pokemon has one major status, and a flinch pre-empts the rest."""
        if self.flinched:
            return 1.0
        return 1.0 - (
            (1 - self.paralysis) * (1 - self.sleep)
            * (1 - self.freeze) * (1 - self.confusion)
        )


def turn_risk(known: KnownPokemon) -> TurnRisk:
    """The chance this Pokemon loses its turn, from what we can see.

    Sleep is the one that needs the counter: a Pokemon on its last sleep turn
    always wakes, and one that just fell asleep never does. We have that number
    for our own side and never for theirs, so theirs is answered with the
    average over the durations the engine actually rolls.
    """
    from pkcm.engine.conditions import (
        CONFUSION_CHANCE,
        PARALYSIS_CHANCE,
        SLEEP_DURATIONS,
        THAW_CHANCE,
    )

    if known.fainted:
        return TurnRisk()
    if "flinch" in known.volatiles:
        return TurnRisk(flinched=1.0)

    confusion = (CONFUSION_CHANCE[0] / CONFUSION_CHANCE[1]
                 if "confusion" in known.volatiles else 0.0)

    status = known.status
    if status == "par":
        return TurnRisk(paralysis=PARALYSIS_CHANCE[0] / PARALYSIS_CHANCE[1],
                        confusion=confusion)
    if status == "frz":
        # Thawing is the *escape*; not thawing is the turn lost.
        return TurnRisk(freeze=1.0 - THAW_CHANCE[0] / THAW_CHANCE[1],
                        confusion=confusion)
    if status == "slp":
        return TurnRisk(sleep=_still_asleep(known, SLEEP_DURATIONS),
                        confusion=confusion)
    return TurnRisk(confusion=confusion)


def _still_asleep(known: KnownPokemon, durations: tuple[int, ...]) -> float:
    """Chance it is still asleep on the coming turn.

    Ours is exact -- the counter is in front of us. Theirs is conditioned on how
    long we have already watched it sleep, which is public even though the roll
    is not: after two turns of a 2-or-3 turn sleep, only the threes are left, so
    it is two in three rather than a coin flip.
    """
    if known.status_turns is not None:
        return 0.0 if known.status_turns <= 1 else 1.0
    elapsed = known.status_elapsed
    if elapsed is None:
        return 1.0
    consistent = [turns for turns in durations if turns >= elapsed]
    if not consistent:
        return 0.0
    return sum(1 for turns in consistent if turns > elapsed) / len(consistent)


#: Refuse a knockout, but only from full HP.
SURVIVES_FROM_FULL = {"sturdy": "ability", "focussash": "item"}
#: Refuses one hit at *any* HP, which is what makes Disguise different in kind
#: from a Focus Sash -- and why hk asked about it separately.
SURVIVES_AT_ANY_HP = {"disguise"}


def could_survive_a_kill(known: KnownPokemon, dex: Dex) -> bool:
    """Might this target live through a lethal hit?

    True when it is at full HP and something it could plausibly be holding
    would save it. Deliberately generous about the unknown: from across the
    field a Focus Sash looks exactly like no item at all.
    """
    if known.species_id is None:
        return False

    # Disguise first: it does not care about HP, and it is still up as long as
    # the Pokemon has not changed forme.
    if "busted" not in known.species_id:
        if known.ability_known and known.ability in SURVIVES_AT_ANY_HP:
            return True
        if not known.ability_known:
            from pkcm.engine.legality import registrable_abilities

            possible = registrable_abilities(dex.species[known.species_id])
            if set(possible) & SURVIVES_AT_ANY_HP:
                return True

    if known.hp_fraction < 1.0:
        return False
    if known.ability_known and known.ability in SURVIVES_FROM_FULL:
        return True
    if known.item_known and known.item in SURVIVES_FROM_FULL:
        return True
    if not known.ability_known:
        from pkcm.engine.legality import registrable_abilities

        possible = registrable_abilities(dex.species[known.species_id])
        if any(ability in SURVIVES_FROM_FULL for ability in possible):
            return True
    return not known.item_known


# --------------------------------------------------------------------------- #
# Team preview: who beats whom, before anyone has moved
#
# One copy, shared by ``search.policy`` (which scores picks) and
# ``envs.encoding`` (which shows the same grid to the network). Two copies of a
# formula do not stay equal, and here the drift would be silent: the policy
# would be trained to imitate a function slightly different from the one it is
# later measured against.
# --------------------------------------------------------------------------- #

#: What a Pokemon's own STAB attack is worth when we have not seen it move.
#: Real sets vary either side of this; the point is that everything gets the
#: same guess, so the comparison between them is about the Pokemon.
NOMINAL_POWER = 90.0

#: Below this the matchup is close enough that moving first decides it.
CLOSE_ENOUGH = 0.2
SPEED_EDGE = 0.1


def damage_share(power: float, offence: float, defence: float, health: float,
                 stab: float, effectiveness: float) -> float:
    """The damage formula's shape at level 50, as a share of ``health``.

    None of the modifiers that need a live field -- weather, screens, items,
    abilities. This is not trying to be the full estimator above; it is trying
    to be cheap enough to run over three hundred and sixty team picks and right
    about which of them deserve the search's budget.
    """
    damage = (22 * power * offence / max(1.0, defence)) / 50 + 2
    return min(1.0, damage * stab * effectiveness / max(1.0, health))


def our_threat(dex: Dex, moves, stats, types, foe_id: str) -> float:
    """What one of ours does to one of theirs. Our set is ours to read."""
    foe_types = dex.species[foe_id].types
    best = 0.0
    for move in moves:
        if not move.base_power:
            continue
        effectiveness = dex.type_chart.multiplier(move.type, foe_types)
        if not effectiveness:
            continue
        physical = move.category == "Physical"
        best = max(best, damage_share(
            move.base_power,
            stats[1] if physical else stats[3],
            midpoint(dex, foe_id, 2 if physical else 4),
            midpoint(dex, foe_id, 0),
            1.5 if move.type in types else 1.0,
            effectiveness))
    return best


def their_threat(dex: Dex, foe_id: str, stats, types) -> float:
    """What one of theirs does to ours, from **what preview actually shows**.

    Their species and nothing else. Their moves are hidden until we watch them
    used and their SP spread is hidden for good, so this reads base stats
    through the same public bracket the estimator uses, and assumes the one
    thing every Pokemon has: a STAB attack off its better attacking stat.

    Being wrong about their set is fine and expected. Being *told* their set
    would make the pick phase look brilliant in self-play and transfer nothing.
    """
    attack, special = midpoint(dex, foe_id, 1), midpoint(dex, foe_id, 3)
    physical = attack >= special
    offence, defence = (attack, stats[2]) if physical else (special, stats[4])
    best = 0.0
    for foe_type in dex.species[foe_id].types:
        effectiveness = dex.type_chart.multiplier(foe_type, types)
        if not effectiveness:
            continue
        best = max(best, damage_share(NOMINAL_POWER, offence, defence,
                                      stats[0], 1.5, effectiveness))
    return best


def matchup(dex: Dex, moves, stats, types, foe_id: str) -> float:
    """One of ours against one of theirs. Positive means we are ahead.

    Who threatens whom, and -- when the threats are close -- who moves first,
    because between two Pokemon that each take the other out in two hits, the
    faster one takes one hit fewer.
    """
    edge = (our_threat(dex, moves, stats, types, foe_id)
            - their_threat(dex, foe_id, stats, types))
    if abs(edge) < CLOSE_ENOUGH:
        theirs = midpoint(dex, foe_id, 5)
        if stats[5] != theirs:
            edge += SPEED_EDGE if stats[5] > theirs else -SPEED_EDGE
    return edge
