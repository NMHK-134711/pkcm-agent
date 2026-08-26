"""Turning an ``Observation`` into arrays, and an integer back into an ``Action``.

Two jobs, both boring on purpose:

**Actions** are one flat index. Singles and doubles share the layout -- doubles
simply unmasks more of it -- so a policy trained on one has the other's action
space already in front of it. The mask comes from ``state.legal_actions``, which
is the same function the engine validates against, so the two cannot drift.

**Observations** stay categorical. Species, moves, items and abilities are
emitted as integer ids for the consumer to embed, not as one-hot vectors: there
are 316 species and 500 moves, and a one-hot of that width is mostly a waste of
a matrix multiply. ``0`` means *unknown or absent* in every categorical field,
which is what makes the opponent's hidden information representable at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkcm.data.dex import Dex
from pkcm.engine.actions import TARGET_ALLY, TARGET_SELF, Action, ActionKind, team_selections
from pkcm.engine.pokemon import MAX_MOVES
from pkcm.engine.state import BOOST_STATS, MAX_BOOST
from pkcm.envs.observation import Observation

#: The most Pokemon either side can bring in any Champions format.
MAX_BROUGHT = 4
#: Field positions a move can be aimed at, in index order.
TARGET_CODES = (0, 1, TARGET_ALLY, TARGET_SELF)
MAX_TARGETS = len(TARGET_CODES)

#: Flat action layout. Singles masks most of it away and doubles does not, which
#: is the only difference between the two action spaces.
#:
#:   0 ..  31   move x target x mega
#:  32 ..  35   switch to a brought slot
#:  36         struggle
#:  37         pass
#:  38 ..      one per team-preview selection
MOVE_BLOCK = MAX_MOVES * MAX_TARGETS * 2
SWITCH_BASE = MOVE_BLOCK
STRUGGLE_INDEX = SWITCH_BASE + MAX_BROUGHT
PASS_INDEX = STRUGGLE_INDEX + 1
SELECTION_BASE = PASS_INDEX + 1


def action_space_size(registered: int, brought: int) -> int:
    return SELECTION_BASE + len(team_selections(registered, brought))


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


def encode_action(action: Action, registered: int, brought: int) -> int:
    if action.kind is ActionKind.MOVE:
        target = TARGET_CODES.index(action.target)
        return (action.index * MAX_TARGETS + target) * 2 + int(action.mega)
    if action.kind is ActionKind.SWITCH:
        return SWITCH_BASE + action.index
    if action.kind is ActionKind.STRUGGLE:
        return STRUGGLE_INDEX
    if action.kind is ActionKind.PASS:
        return PASS_INDEX
    return SELECTION_BASE + team_selections(registered, brought).index(action)


def decode_action(index: int, registered: int, brought: int) -> Action:
    if index < MOVE_BLOCK:
        mega = bool(index % 2)
        rest = index // 2
        return Action.move(rest // MAX_TARGETS, mega=mega,
                           target=TARGET_CODES[rest % MAX_TARGETS])
    if index < STRUGGLE_INDEX:
        return Action.switch(index - SWITCH_BASE)
    if index == STRUGGLE_INDEX:
        return Action.struggle()
    if index == PASS_INDEX:
        return Action.PASS
    return team_selections(registered, brought)[index - SELECTION_BASE]


def action_mask(observation: Observation, position: int,
                registered: int, brought: int) -> np.ndarray:
    """One row of the mask, straight from what the engine would accept."""
    mask = np.zeros(action_space_size(registered, brought), dtype=np.int8)
    for action in observation.action_mask(position):
        mask[encode_action(action, registered, brought)] = 1
    return mask


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Stable id -> index maps. Built once per dex, shared by every env.

    Index ``0`` is reserved everywhere for "unknown or none", so a hidden
    opponent item and an empty item slot are the same symbol -- which is
    correct, because from across the field they are indistinguishable.
    """

    species: dict[str, int]
    moves: dict[str, int]
    items: dict[str, int]
    abilities: dict[str, int]
    statuses: dict[str, int]

    @staticmethod
    def of(dex: Dex) -> "Vocabulary":
        def index(ids) -> dict[str, int]:
            return {key: number for number, key in enumerate(sorted(ids), start=1)}

        from pkcm.engine.state import MAJOR_STATUSES

        return Vocabulary(
            species=index(dex.species),
            moves=index(dex.moves),
            items=index(dex.items),
            abilities=index(dex.abilities),
            statuses=index(MAJOR_STATUSES),
        )

    def sizes(self) -> dict[str, int]:
        return {
            "species": len(self.species) + 1,
            "moves": len(self.moves) + 1,
            "items": len(self.items) + 1,
            "abilities": len(self.abilities) + 1,
            "statuses": len(self.statuses) + 1,
        }


#: Per Pokemon: HP fraction, fainted, on-field, plus one per stat stage.
PER_POKEMON_SCALARS = 3 + len(BOOST_STATS)
#: Field-wide: turn, weather turns, terrain turns, mega spent x2.
FIELD_SCALARS = 5
#: Side-wide: the hazards and screens a policy has to see.
SIDE_CONDITIONS = ("reflect", "lightscreen", "auroraveil", "tailwind",
                   "spikes", "toxicspikes", "stealthrock", "stickyweb", "safeguard")
WEATHERS = ("sunnyday", "raindance", "sandstorm", "snowscape")
TERRAINS = ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")
ROOMS = ("trickroom", "gravity", "magicroom", "wonderroom")

SCALAR_SIZE = (
    FIELD_SCALARS
    + 2 * MAX_BROUGHT * PER_POKEMON_SCALARS
    + 2 * len(SIDE_CONDITIONS)
    + len(WEATHERS) + len(TERRAINS) + len(ROOMS)
)


#: Field positions per side, at most. Doubles has two; singles uses the first.
MAX_POSITIONS = 2
#: Per (our move, their standing Pokemon): effectiveness, the damage bracket,
#: whether it is a guaranteed knockout, whether they are immune, and the two
#: numbers a player actually decides on -- the chance it knocks them out this
#: turn, and the chance it connects at all.
MATCHUP_FEATURES = 7
MATCHUP_ROWS = MAX_POSITIONS * MAX_MOVES * MAX_POSITIONS
#: Per (our position, their position): we outspeed, we are outsped. Both zero
#: means their spread could decide it either way, which is a real third answer.
SPEED_ROWS = MAX_POSITIONS * MAX_POSITIONS
SPEED_FEATURES = 2

#: Per Pokemon on the field, ours then theirs: the chance it loses the turn,
#: and the four ways that happens. A turn taken away costs more than most
#: damage, and every one of these is a published number rather than something
#: to be learned from reward.
RISK_ROWS = 2 * MAX_POSITIONS
RISK_FEATURES = 5


def encode_matchup(observation: Observation, sheet, dex) -> tuple:
    """What the calculator can work out, laid out for a policy to read.

    This is the half of the game a strong player does with a damage calculator
    and a dex, and none of it is hidden information -- it is arithmetic over
    public numbers. Making a policy rediscover "Fighting resists Dark" from
    reward is making it work for something the game already tells it.

    The damage figures are brackets, because the opponent's spread is unknown.
    See ``pkcm.envs.analysis``.
    """
    from pkcm.envs.analysis import assess

    matchup = np.zeros((MATCHUP_ROWS, MATCHUP_FEATURES), dtype=np.float32)
    speed = np.zeros((SPEED_ROWS, SPEED_FEATURES), dtype=np.float32)
    risk = np.zeros((RISK_ROWS, RISK_FEATURES), dtype=np.float32)

    from pkcm.envs.analysis import turn_risk

    for side_index, team in enumerate((observation.own, observation.foe)):
        for known in team:
            if known.position is None or known.position >= MAX_POSITIONS:
                continue
            row = side_index * MAX_POSITIONS + known.position
            found = turn_risk(known)
            risk[row] = (found.cannot_act, found.paralysis, found.sleep,
                         found.freeze, found.confusion)

    foe_position = {known.slot: known.position for known in observation.foe
                    if known.position is not None}

    for position in range(MAX_POSITIONS):
        assessment = assess(observation, sheet, dex, position)
        if assessment is None:
            continue
        attacker = next((k for k in observation.own if k.position == position), None)
        move_order = {move_id: index for index, move_id
                      in enumerate(attacker.moves[:MAX_MOVES])} if attacker else {}

        for slot, estimate in assessment.damage:
            move_index = move_order.get(estimate.move_id)
            target = foe_position.get(slot)
            if move_index is None or target is None or target >= MAX_POSITIONS:
                continue
            row = (position * MAX_MOVES + move_index) * MAX_POSITIONS + target
            # Effectiveness spans 0 to 4; log2 makes the steps even and keeps
            # "neutral" at zero, which is where a linear layer wants it.
            matchup[row, 0] = 0.0 if estimate.immune else np.log2(estimate.effectiveness)
            matchup[row, 1] = estimate.percent.low / 100.0
            matchup[row, 2] = estimate.percent.high / 100.0
            matchup[row, 3] = float(estimate.guaranteed_ko)
            matchup[row, 4] = float(estimate.immune)
            matchup[row, 5] = estimate.ko_chance
            matchup[row, 6] = estimate.hit_chance

        for slot, faster in assessment.outspeeds:
            target = foe_position.get(slot)
            if target is None or target >= MAX_POSITIONS:
                continue
            row = position * MAX_POSITIONS + target
            speed[row, 0] = float(faster is True)
            speed[row, 1] = float(faster is False)

    return matchup, speed, risk


def encode_observation(observation: Observation, vocabulary: Vocabulary,
                       sheet=None, dex=None) -> dict:
    """The observation as arrays. Categorical stays categorical.

    ``sheet`` and ``dex`` add the calculator's block. They are optional so the
    encoding stays usable without one, but a policy that has them is playing
    the game a human plays.
    """
    scalars = np.zeros(SCALAR_SIZE, dtype=np.float32)
    cursor = 0

    scalars[cursor] = min(observation.turn, 200) / 200.0
    scalars[cursor + 1] = observation.weather_turns / 8.0
    scalars[cursor + 2] = observation.terrain_turns / 8.0
    scalars[cursor + 3] = float(observation.mega_used[0])
    scalars[cursor + 4] = float(observation.mega_used[1])
    cursor += FIELD_SCALARS

    for team in (observation.own, observation.foe):
        for slot in range(MAX_BROUGHT):
            known = team[slot] if slot < len(team) else None
            if known is not None:
                scalars[cursor] = known.hp_fraction
                scalars[cursor + 1] = float(known.fainted)
                scalars[cursor + 2] = float(known.position is not None)
                for stage, value in enumerate(known.boosts):
                    scalars[cursor + 3 + stage] = value / MAX_BOOST
            cursor += PER_POKEMON_SCALARS

    for conditions in (observation.own_conditions, observation.foe_conditions):
        table = dict(conditions)
        for offset, name in enumerate(SIDE_CONDITIONS):
            scalars[cursor + offset] = min(table.get(name, 0), 5) / 5.0
        cursor += len(SIDE_CONDITIONS)

    for names, active in ((WEATHERS, observation.weather),
                          (TERRAINS, observation.terrain)):
        for offset, name in enumerate(names):
            scalars[cursor + offset] = float(active == name)
        cursor += len(names)
    rooms = dict(observation.rooms)
    for offset, name in enumerate(ROOMS):
        scalars[cursor + offset] = float(name in rooms)
    cursor += len(ROOMS)

    assert cursor == SCALAR_SIZE, (cursor, SCALAR_SIZE)

    species = np.zeros(2 * MAX_BROUGHT, dtype=np.int64)
    statuses = np.zeros(2 * MAX_BROUGHT, dtype=np.int64)
    items = np.zeros(2 * MAX_BROUGHT, dtype=np.int64)
    abilities = np.zeros(2 * MAX_BROUGHT, dtype=np.int64)
    moves = np.zeros(2 * MAX_BROUGHT * MAX_MOVES, dtype=np.int64)
    pp = np.zeros(2 * MAX_BROUGHT * MAX_MOVES, dtype=np.float32)

    for team_index, team in enumerate((observation.own, observation.foe)):
        for slot in range(MAX_BROUGHT):
            flat = team_index * MAX_BROUGHT + slot
            if slot >= len(team):
                continue
            known = team[slot]
            species[flat] = vocabulary.species.get(known.species_id, 0)
            statuses[flat] = vocabulary.statuses.get(known.status, 0)
            items[flat] = vocabulary.items.get(known.item, 0)
            abilities[flat] = vocabulary.abilities.get(known.ability, 0)
            for move_index, move_id in enumerate(known.moves[:MAX_MOVES]):
                moves[flat * MAX_MOVES + move_index] = vocabulary.moves.get(move_id, 0)
                if known.pp is not None and move_index < len(known.pp):
                    pp[flat * MAX_MOVES + move_index] = known.pp[move_index] / 20.0

    # Everything each side registered, whether or not it was brought. Public
    # from team preview, and the only thing a policy can reason about before the
    # opponent has shown anything.
    registered = np.zeros(2 * 6, dtype=np.int64)
    for team_index, team in enumerate(observation.registered):
        for slot, species_id in enumerate(team[:6]):
            registered[team_index * 6 + slot] = vocabulary.species.get(species_id, 0)

    encoded = {
        "scalars": scalars,
        "species": species,
        "status": statuses,
        "items": items,
        "abilities": abilities,
        "moves": moves,
        "pp": pp,
        "registered": registered,
    }
    if sheet is not None and dex is not None:
        matchup, speed, risk = encode_matchup(observation, sheet, dex)
        encoded["matchup"] = matchup
        encoded["speed"] = speed
        encoded["risk"] = risk
    return encoded
