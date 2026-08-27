"""The reference sheet a player has open, as arrays rather than as weights.

A network could learn that Garchomp is Dragon/Ground with 130 Attack, and that
Earthquake is 100 base power Ground. Making it learn that is a waste: it is a
fact, it never changes, and there are 316 species and 500 moves of them. Worse,
a policy that has *memorised* the dex has no way to be right about a Pokemon it
has not seen enough of during training.

So the dex ships as a lookup table indexed by the same vocabulary ids the
observation uses. The policy gathers rows for the ids it is looking at, the way
a player glances at the sheet. Nothing here is hidden information -- every field
is printed in the game's own dex.

    sheet = ReferenceSheet.of(dex, vocabulary)
    sheet.species[obs["species"]]   # -> (slots, SPECIES_FEATURES)
    sheet.moves[obs["moves"]]       # -> (slots * 4, MOVE_FEATURES)

Row 0 of every table is zeros, because id 0 means "unknown or none" -- looking
up an opponent's unrevealed item returns a blank row rather than a wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pkcm.data.dex import Dex, Stat
from pkcm.envs.encoding import Vocabulary

#: The 18 types, in a fixed order so the one-hot columns mean the same thing
#: from run to run.
TYPES = ("normal", "fire", "water", "electric", "grass", "ice", "fighting",
         "poison", "ground", "flying", "psychic", "bug", "rock", "ghost",
         "dragon", "dark", "steel", "fairy")
TYPE_INDEX = {name: number for number, name in enumerate(TYPES)}

#: Base stats are divided by this to land in roughly [0, 1.5]. Blissey's 255 HP
#: is the outlier and it is fine for it to exceed one.
STAT_SCALE = 180.0
POWER_SCALE = 150.0

#: Per species: six base stats, an 18-wide type one-hot, weight, and whether it
#: is a Mega forme.
SPECIES_FEATURES = 6 + len(TYPES) + 2
#: Per move: an 18-wide type one-hot, then power, accuracy, priority, three
#: category flags, contact -- and the move's own dice: how often its secondary
#: fires, how often that secondary is a flinch, how many times it hits, and how
#: often it crits. A strong player knows all four by heart for the moves they
#: run, and none of them is hidden.
MOVE_FEATURES = len(TYPES) + 11


@dataclass(frozen=True, slots=True)
class ReferenceSheet:
    """Static per-id feature tables. Built once, shared by every env."""

    species: np.ndarray        # (n_species + 1, SPECIES_FEATURES)
    moves: np.ndarray          # (n_moves + 1, MOVE_FEATURES)
    #: ``type_chart[attacking, defending]`` -- the full 18x18, so a policy can
    #: work out a matchup it has never been shown.
    type_chart: np.ndarray     # (18, 18)
    #: ``learnable[species_id, move_id]`` as a bit. What the opponent's
    #: unrevealed Pokemon *could* be holding, which is the pick-phase question.
    learnable: np.ndarray      # (n_species + 1, n_moves + 1), uint8
    vocabulary: Vocabulary

    @staticmethod
    def of(dex: Dex, vocabulary: Vocabulary) -> "ReferenceSheet":
        sizes = vocabulary.sizes()
        species = np.zeros((sizes["species"], SPECIES_FEATURES), dtype=np.float32)
        for species_id, row in vocabulary.species.items():
            entry = dex.species[species_id]
            species[row, :6] = [entry.base_stats[stat] / STAT_SCALE for stat in Stat]
            for type_name in entry.types:
                # Showdown's pokedex carries MissingNo. and its glitch ``bird``
                # type. Nothing in Champions has it, and the vocabulary covers
                # the whole dex rather than the roster, so skip rather than fail.
                column = TYPE_INDEX.get(type_name)
                if column is not None:
                    species[row, 6 + column] = 1.0
            species[row, 6 + len(TYPES)] = min(entry.weight_kg, 1000.0) / 1000.0
            species[row, 7 + len(TYPES)] = float(entry.is_mega)

        moves = np.zeros((sizes["moves"], MOVE_FEATURES), dtype=np.float32)
        for move_id, row in vocabulary.moves.items():
            entry = dex.moves[move_id]
            if entry.type in TYPE_INDEX:
                moves[row, TYPE_INDEX[entry.type]] = 1.0
            offset = len(TYPES)
            moves[row, offset] = (entry.base_power or 0) / POWER_SCALE
            moves[row, offset + 1] = 1.0 if entry.accuracy is None else entry.accuracy / 100.0
            moves[row, offset + 2] = entry.priority / 5.0
            moves[row, offset + 3] = float(entry.category == "Physical")
            moves[row, offset + 4] = float(entry.category == "Special")
            moves[row, offset + 5] = float(entry.category == "Status")
            moves[row, offset + 6] = float("contact" in entry.flags)
            moves[row, offset + 7] = secondary_chance(entry) / 100.0
            moves[row, offset + 8] = flinch_chance(entry) / 100.0
            moves[row, offset + 9] = expected_hits(entry) / 5.0
            moves[row, offset + 10] = crit_chance(entry)

        chart = np.ones((len(TYPES), len(TYPES)), dtype=np.float32)
        for attacking in TYPES:
            for defending in TYPES:
                chart[TYPE_INDEX[attacking], TYPE_INDEX[defending]] = \
                    dex.type_chart.multiplier(attacking, (defending,))

        return ReferenceSheet(
            species=species,
            moves=moves,
            type_chart=chart,
            learnable=_learnable_matrix(dex, vocabulary),
            vocabulary=vocabulary,
        )

    # -- the questions a player asks the sheet ------------------------------ #

    def effectiveness(self, move_type: str, defender_types: tuple[str, ...]) -> float:
        """The multiplier, straight off the chart. Public information."""
        if move_type not in TYPE_INDEX:
            return 1.0
        value = 1.0
        for defending in defender_types:
            if defending in TYPE_INDEX:
                value *= float(self.type_chart[TYPE_INDEX[move_type],
                                               TYPE_INDEX[defending]])
        return value

    def could_learn(self, species_id: str | None, move_id: str) -> bool:
        """Could this species be carrying that move?

        The pick-phase question. Team preview names the opponent's six, and what
        those six *can* learn is public -- so an unrevealed Pokemon is not a
        blank, it is a distribution over the moves its species is allowed.
        """
        if species_id is None:
            return False
        row = self.vocabulary.species.get(species_id, 0)
        column = self.vocabulary.moves.get(move_id, 0)
        return bool(self.learnable[row, column])

    def candidate_moves(self, species_id: str) -> np.ndarray:
        """The whole row: every move that species could be running."""
        return self.learnable[self.vocabulary.species.get(species_id, 0)]


def secondaries(move) -> list[dict]:
    """Every secondary a move carries, however the data spells it."""
    found: list[dict] = []
    primary = move.raw.get("secondary")
    if isinstance(primary, dict):
        found.append(primary)
    extra = move.raw.get("secondaries")
    if isinstance(extra, list):
        found.extend(entry for entry in extra if isinstance(entry, dict))
    return found


def secondary_chance(move) -> int:
    """The best chance any of its secondaries fires, as a percentage."""
    return max((entry.get("chance", 100) for entry in secondaries(move)), default=0)


def flinch_chance(move) -> int:
    """How often it makes the target flinch. Its own line because flinching is
    a *turn* taken away, which is worth more than most secondaries."""
    return max((entry.get("chance", 100) for entry in secondaries(move)
                if entry.get("volatileStatus") == "flinch"), default=0)


def status_chance(move, status: str) -> int:
    """How often it inflicts one particular status, secondary or primary."""
    if move.raw.get("status") == status:
        return move.accuracy if isinstance(move.accuracy, int) else 100
    return max((entry.get("chance", 100) for entry in secondaries(move)
                if entry.get("status") == status), default=0)


def expected_hits(move) -> float:
    """How many times it lands, on average.

    ``[2, 5]`` is not uniform: Gen 5 onward it is 35-35-15-15 across 2, 3, 4
    and 5, which averages 3.1 rather than 3.5. The engine's own table is the
    source, so this cannot drift from what actually gets rolled -- and it did
    drift once, when that table still held the Gen 4 spread.
    """
    from pkcm.engine.moves import MULTIHIT_2_TO_5

    multihit = move.raw.get("multihit")
    if multihit is None:
        return 1.0
    if isinstance(multihit, int):
        return float(multihit)
    low, high = multihit
    if (low, high) == (2, 5):
        return sum(MULTIHIT_2_TO_5) / len(MULTIHIT_2_TO_5)
    return (low + high) / 2


def crit_chance(move) -> float:
    """How often the move crits, from its own crit ratio alone."""
    from pkcm.engine.moves import CRIT_DENOMINATOR, NEVER_CRITS

    if move.raw.get("willCrit"):
        return 1.0
    ratio = move.raw.get("critRatio", 1)
    if ratio <= NEVER_CRITS:
        return 0.0
    denominator = CRIT_DENOMINATOR.get(ratio, 1)
    return 1.0 if denominator <= 1 else 1.0 / denominator


def _learnable_matrix(dex: Dex, vocabulary: Vocabulary) -> np.ndarray:
    from pkcm.engine.legality import learnable_moves

    sizes = vocabulary.sizes()
    matrix = np.zeros((sizes["species"], sizes["moves"]), dtype=np.uint8)
    for species_id, row in vocabulary.species.items():
        for move_id in learnable_moves(dex, species_id):
            column = vocabulary.moves.get(move_id)
            if column is not None:
                matrix[row, column] = 1
    return matrix


#: One sheet per dex object. Building the learnable matrix walks every species,
#: so it is worth not doing per environment.
#:
#: Keyed by ``id(dex)`` **and holding the dex alive**. Without the second half
#: the entry outlives the object it describes: CPython reuses an address once
#: the old dex is collected, and the next dex to land there silently gets the
#: previous one's sheet. That showed up as a test comparing a serial run against
#: a pooled one and disagreeing about one battle in ten -- in a worker there is
#: only ever one dex, so it could not happen there.
_CACHE: dict[int, tuple[Dex, ReferenceSheet]] = {}


def sheet_for(dex: Dex, vocabulary: Vocabulary) -> ReferenceSheet:
    key = id(dex)
    found = _CACHE.get(key)
    if found is None or found[0] is not dex:
        _CACHE[key] = (dex, ReferenceSheet.of(dex, vocabulary))
    return _CACHE[key][1]
