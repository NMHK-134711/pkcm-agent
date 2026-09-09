"""Build a party around one Pokemon, by evolution.

**Why an axis.** hk builds the way people build: pick the Pokemon the team is
for, then find the five that let it work. That is not a search over all teams,
it is a search over the five slots around a fixed one -- so the core's species
is held and every operator here works around it.

**Why evolution rather than hill climbing.** A team is not a smooth function of
its slots. Swapping a wall for a sweeper is worth nothing on its own and a lot
once the two moves that support it arrive, so a single-step climb sits in a
local minimum that a population walks out of. Crossover is the reason: two
parents that solved different halves of the field can hand their halves to one
child, which is exactly what a slot-wise exchange is.

**What fitness is, and what it is not.** The floor -- the mean of the worst
quarter of matchups, from ``party_floor`` -- because hk's goal is a party with
an answer to most of what it meets rather than one that beats the average. At
the budget a generation can pay, that number is biased low by the winner's
curse: the tail is chosen from the same games that score it. It is biased the
same way for every candidate, so the *ranking* survives and the value does not.
Do not read a generation's fitness as a win rate.

**And the agent scoring it is not neutral.** Measured on the 2026-09-09 engine,
a search that can see past its own horizon beats the material count by ten
points on screen parties and nine on setup parties, and by nothing at all on
plain offence. So the cheap agent used here systematically underrates exactly
the parties hk wants to find. The answer is not to pay fifteen times as much
for every candidate -- it is ``judge`` below: search with the cheap agent,
decide with the expensive one, on the handful that survive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterator, Sequence

from pkcm.data.dex import Dex
from pkcm.engine.legality import Party, ranker_parties, team_errors
from pkcm.engine.pokemon import PokemonSet, Team
from pkcm.engine.rng import Rng, RngCursor
from pkcm.search import SearchConfig
from pkcm.train.party_floor import Floor, FloorConfig, score
from pkcm.train.party_mutate import mutate


@dataclass(frozen=True, slots=True)
class EvolveConfig:
    """The axis, the field it is measured against, and how much to spend."""

    #: The species every candidate must carry. Its set may still change --
    #: moves, item, ability, nature, spread -- but never what it is.
    core: str = ""
    #: Where the opponents come from, and where the warm start is read from.
    parties: str | None = None
    population: int = 24
    generations: int = 8
    #: Kept unchanged into the next generation. Without this a good team can
    #: be lost to a bad mutation of itself.
    elites: int = 3
    #: How many of a child's slots come from the second parent.
    crossover_slots: int = 3
    #: Chance in a hundred that a child is a crossover rather than a mutant.
    crossover_chance: int = 40
    #: Mutations applied to a mutant child. More than one lets a change that
    #: only pays with its partner arrive in one step.
    mutations: tuple[int, ...] = (1, 1, 1, 2, 2, 3)
    #: Battles a generation may spend, shared out by racing.
    budget: int = 3000
    floor: FloorConfig = field(default_factory=FloorConfig)
    seed: int = 900_000


@dataclass(frozen=True, slots=True)
class Candidate:
    team: Team
    origin: str
    floor: Floor | None = None

    @property
    def fitness(self) -> float:
        return self.floor.cvar if self.floor is not None else -1.0


# --------------------------------------------------------------------------- #
# Building a population
# --------------------------------------------------------------------------- #


def carrying(parties: Sequence[Party], core: str) -> list[Party]:
    """The field's own teams built around this Pokemon.

    A warm start rather than a random one. Six random legal sets that happen
    to include a Garchomp is not a Garchomp team, and a generation spent
    discovering that a team needs a way to answer Steel is a generation spent
    rediscovering what the field already knows.
    """
    return [party for party in parties
            if any(one.species == core for one in party.team)]


def a_set_for(dex: Dex, core: str, cursor: RngCursor) -> PokemonSet | None:
    """A first set for a species the field has never used.

    Its own ranker set when the pool has one, and otherwise four moves it can
    learn with a template spread -- deliberately rough, because the search is
    about to spend a generation improving exactly this.
    """
    from pkcm.engine.legality import (NATURES, ranker_slots,
                                      registrable_abilities, usable_moves)
    from pkcm.train.party_mutate import spread_templates

    built = [one for one in ranker_slots() if one.species == core]
    if built:
        return built[cursor.between(0, len(built) - 1)]
    stone = mega_stone_for(dex, core)
    moves = usable_moves(dex, core)
    if len(moves) < 4:
        return None
    picked = tuple(moves[i] for i in cursor.shuffled(list(range(len(moves))))[:4])
    abilities = registrable_abilities(dex.species[core]) or ("__none__",)
    templates = spread_templates()
    return PokemonSet(species=core, ability=abilities[0], moves=picked,
                      item=stone,
                      nature=sorted(NATURES)[cursor.between(0, len(NATURES) - 1)],
                      sp=templates[cursor.between(0, len(templates) - 1)])


def mega_stone_for(dex: Dex, species: str) -> str | None:
    """The stone that Megas this species, if it has one.

    A core chosen for its Mega should start holding the stone. Mutation can
    find it -- ``mutate_item`` draws from every item in the format -- but that
    is a needle, and "Mega Salamence" is a request for the Mega rather than
    for the odds of stumbling onto Salamencite.
    """
    name = dex.species[species].name
    for item_id, item in dex.items.items():
        stone = item.raw.get("megaStone")
        if isinstance(stone, dict) and name in stone:
            return item_id
    return None


def graft(dex: Dex, regulation, config: EvolveConfig, host: Team,
          cursor: RngCursor) -> Team | None:
    """Put the core into somebody else's team, for a core nobody has used.

    Which is what a person does with a Pokemon the ladder has not seen yet:
    take a shell that works and give one slot to the new thing. The five
    around it are then a real team's five rather than a random draw, which is
    the whole reason the warm start exists.
    """
    fresh = a_set_for(dex, config.core, cursor)
    if fresh is None:
        return None
    order = cursor.shuffled(list(range(len(host))))
    for index in order:
        candidate = tuple(fresh if i == index else one
                          for i, one in enumerate(host))
        if not team_errors(dex, regulation, candidate,
                           config.floor.battle_format):
            return candidate
    return None


def seed_population(dex: Dex, regulation, config: EvolveConfig,
                    cursor: RngCursor) -> list[Candidate]:
    """Warm start from the field: teams built around the core, or grafted.

    A core the field has never used -- anything the update just added -- has
    no team to start from, so one slot of a real team is given to it instead.
    Raises only when even that fails, which means the core cannot be fielded
    at all under this regulation.
    """
    parties = ranker_parties(config.parties)
    seeds = carrying(parties, config.core)
    grafted = False
    if not seeds:
        grafted = True
        made = []
        for party in parties:
            team = graft(dex, regulation, config, party.team, cursor)
            if team is not None:
                made.append(Party(title=f"grafted onto {party.title[:34]}",
                                  rate=party.rate, rank=party.rank, team=team))
            if len(made) >= config.population:
                break
        seeds = made
    if not seeds:
        raise ValueError(
            f"{config.core!r} cannot be put into any party in the field; "
            "check that it is legal in this regulation")

    label = "graft" if grafted else "field"
    population = [Candidate(party.team, f"{label}:{party.title[:40]}")
                  for party in seeds[:config.population]]
    while len(population) < config.population:
        parent = population[cursor.between(0, len(seeds) - 1)]
        team = parent.team
        for _ in range(1 + cursor.between(0, 2)):
            _, team = _mutate_around_core(dex, regulation, config, team, cursor)
        population.append(Candidate(team, "seeded mutant"))
    return population


def _mutate_around_core(dex: Dex, regulation, config: EvolveConfig,
                        team: Team, cursor: RngCursor) -> tuple[str, Team]:
    """One mutation, refused if it would change what the core *is*.

    ``mutate_species`` is the only operator that can, and it picks its slot
    itself, so this retries rather than reaching inside it. Ten tries is
    generous: the odds of hitting the one protected slot are one in six, and
    the operator is not always a species swap.
    """
    for _ in range(10):
        name, candidate = mutate(dex, regulation, team, cursor,
                                 config.floor.battle_format)
        if _core_intact(candidate, config.core):
            return name, candidate
    return "none", team


def _core_intact(team: Team, core: str) -> bool:
    return any(one.species == core for one in team)


def cross(dex: Dex, regulation, config: EvolveConfig, first: Team,
          second: Team, cursor: RngCursor) -> Team | None:
    """Slots from two parents, or ``None`` when nothing legal comes out.

    The core is taken from whichever parent is being kept whole, so a child
    cannot lose it. Everything else is refused by ``team_errors`` -- two of a
    base species, two of an item -- and a refused child is dropped rather than
    repaired, because a repair is a mutation nobody asked for and it would be
    attributed to the crossover.
    """
    slots = list(range(len(first)))
    taken = set()
    while len(taken) < config.crossover_slots and len(taken) < len(slots):
        pick = cursor.between(0, len(slots) - 1)
        if first[pick].species != config.core:
            taken.add(pick)
        elif len(taken) + 1 >= len(slots):
            break
    child = tuple(second[index] if index in taken else first[index]
                  for index in slots)
    if not _core_intact(child, config.core):
        return None
    if team_errors(dex, regulation, child, config.floor.battle_format):
        return None
    return child


def breed(dex: Dex, regulation, config: EvolveConfig,
          survivors: Sequence[Candidate], cursor: RngCursor) -> list[Candidate]:
    """The next generation: the elites, then children of the survivors."""
    ranked = sorted(survivors, key=lambda one: -one.fitness)
    children = [replace(one, origin="elite", floor=None)
                for one in ranked[:config.elites]]

    while len(children) < config.population:
        first = ranked[cursor.between(0, len(ranked) - 1)].team
        if cursor.between(0, 99) < config.crossover_chance and len(ranked) > 1:
            second = ranked[cursor.between(0, len(ranked) - 1)].team
            child = cross(dex, regulation, config, first, second, cursor)
            if child is not None:
                children.append(Candidate(child, "crossover"))
                continue
        team = first
        count = config.mutations[cursor.between(0, len(config.mutations) - 1)]
        names = []
        for _ in range(count):
            name, team = _mutate_around_core(dex, regulation, config, team,
                                             cursor)
            names.append(name)
        children.append(Candidate(team, "+".join(names)))
    return children


# --------------------------------------------------------------------------- #
# Spending a generation's budget
# --------------------------------------------------------------------------- #


def basis_of(config: EvolveConfig) -> str:
    """What has to match for two candidates' floors to be comparable.

    The field they faced, how much of it, the agent that played, and the seed
    that drew the subset. Change any of these and the numbers are measuring
    different things, which is why the notebook keeps them apart rather than
    sorting them together.
    """
    floor = config.floor
    return (f"{Path(floor.parties).name if floor.parties else 'archive'}"
            f"/{floor.regulation}/{floor.battle_format}"
            f"/opp{floor.opponents or 'all'}"
            f"/sim{floor.search.iterations}"
            f"/rollout{floor.search.rollout_turns}"
            f"/seed{floor.seed}")


def race(config: EvolveConfig, population: Sequence[Candidate],
         workers: int | None = None,
         on_round=None, notebook: "Notebook | None" = None,
         stage: str = "") -> list[Candidate]:
    """Successive halving over the population.

    A flat budget spends most of itself separating candidates that are already
    out of the running. Every survivor gets the same number of games, the worst
    half is dropped, and the games double -- so the pair at the top is
    separated by the last round rather than by the first, which is where the
    budget is worth paying.

    ``score`` does the same thing one level down, over a candidate's opponents.
    """
    alive = list(population)
    spent = 0
    rounds = max(1, (len(alive)).bit_length() - 1)
    for step in range(rounds):
        if len(alive) <= 1:
            break
        share = max(1, (config.budget - spent) // (2 * max(1, len(alive))))
        graded = []
        for one in alive:
            floor = score(one.team, config.floor, share, workers=workers)
            scored = replace(one, floor=floor)
            graded.append(scored)
            spent += floor.games
            if notebook is not None:
                # Here rather than after the sort: a candidate is written the
                # moment its games are paid for, so a run killed mid-round
                # still keeps everything it had measured.
                notebook.record(scored, basis_of(config), f"{stage}r{step + 1}")
        graded.sort(key=lambda one: -one.fitness)
        keep = max(1, len(graded) // 2) if step < rounds - 1 else len(graded)
        alive = graded[:keep]
        if on_round:
            on_round(step, alive, spent)
        if spent >= config.budget:
            break
    return alive


def evolve(dex: Dex, regulation, config: EvolveConfig,
           workers: int | None = None,
           on_generation=None,
           notebook: "Notebook | None" = None) -> Iterator[list[Candidate]]:
    """Run the search, handing back each generation's graded survivors.

    ``notebook`` collects every candidate on the way past, including the ones
    the race drops. The generator's product is then a book of parties rather
    than one winner, which is what the run is actually for.
    """
    cursor = Rng.from_seed(config.seed).cursor()
    population = seed_population(dex, regulation, config, cursor)
    best: list[Candidate] = []
    for generation in range(config.generations):
        survivors = race(config, population, workers=workers,
                         notebook=notebook, stage=f"g{generation + 1}")
        best = sorted(survivors + best, key=lambda one: -one.fitness)
        best = best[:config.elites]
        if on_generation:
            on_generation(generation, survivors)
        yield survivors
        if generation + 1 < config.generations:
            population = breed(dex, regulation, config, survivors, cursor)


def as_party(team: Team, title: str) -> dict:
    """A candidate in the shape ``parties_field.json`` uses."""
    return {"id": "evolved", "title": title, "source": "evolve",
            "team": [{"species": one.species, "ability": one.ability,
                      "moves": list(one.moves), "item": one.item,
                      "nature": one.nature, "sp": list(one.sp)}
                     for one in team]}


# --------------------------------------------------------------------------- #
# Keeping what the search throws away
# --------------------------------------------------------------------------- #


def signature(team: Team) -> str:
    """What makes two candidates the same party.

    Slot order is not part of it -- a crossover that hands back the same six
    sets in a different order is the same team -- but the whole set is, down
    to the spread, because two Salamence that differ only in nature are two
    different parties to play.
    """
    import hashlib

    parts = sorted(
        "|".join((one.species, one.ability or "", one.item or "", one.nature,
                  ",".join(str(value) for value in one.sp),
                  ",".join(sorted(one.moves))))
        for one in team)
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Notebook:
    """Every candidate the search grades, written down as it is graded.

    A generation grades twenty-four parties and carries three of them forward.
    The other twenty-one are measured -- the games were paid for -- and then
    dropped, and a run that is stopped halfway leaves nothing at all. This
    writes each one as it is scored, so the run's product is a list of parties
    rather than a single winner, and stopping early costs only the rest of the
    search.

    **Read the interval, not the score.** A candidate eliminated in the first
    round was graded on a fifth of the games the survivors got, and it was
    eliminated partly because it was unlucky. Ranking the notebook by ``cvar``
    would put those noisy draws on top -- the same winner's curse the round
    robin had, one level down. ``best`` sorts by ``low`` instead, so a party
    has to have been measured to rank.

    **And only within a basis.** Two runs that faced different opponent
    subsets, or graded with different search settings, produce numbers that
    are not comparable. Each record carries the basis it was measured under
    and ``best`` refuses to mix them.
    """

    path: Path
    #: Records below this lower bound are not written. Zero keeps everything,
    #: which is the useful default: the file is small and a party that looks
    #: bad under the cheap agent is exactly what the judge might disagree with.
    keep: float = 0.0
    seen: dict = field(default_factory=dict)
    written: int = 0

    @classmethod
    def open(cls, path: str | Path, keep: float = 0.0) -> "Notebook":
        """Load what is already there, so a second run adds to the first."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        seen = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:      # a run killed mid-write
                    continue
                key = (record["basis"], record["signature"])
                held = seen.get(key)
                if held is None or record["games"] > held["games"]:
                    seen[key] = record
        return cls(path=path, keep=keep, seen=seen)

    def record(self, candidate: Candidate, basis: str, stage: str) -> bool:
        """Write one graded candidate. False when it was not worth keeping.

        The same party graded twice keeps whichever measurement had more
        games behind it, not whichever scored higher -- the second grading of
        a survivor is the better number, and taking the maximum instead would
        reintroduce exactly the bias this is arranged to avoid.
        """
        floor = candidate.floor
        if floor is None or floor.low < self.keep:
            return False
        # Keyed by the basis as well as the party, because the same six under
        # two agents are two measurements. Keying by the party alone let a
        # judged floor -- which has more games behind it, always -- delete the
        # search-basis row, and the searched list then quietly lost exactly
        # the parties that had done best in it.
        key = signature(candidate.team)
        held = self.seen.get((basis, key))
        if held is not None and held["games"] >= floor.games:
            return False
        record = {
            "signature": key,
            "floor": round(floor.cvar, 4),
            "low": round(floor.low, 4),
            "high": round(floor.high, 4),
            "mean": round(floor.mean, 4),
            "worst": round(floor.minimum, 4),
            "games": floor.games,
            "origin": candidate.origin,
            "stage": stage,
            "basis": basis,
            **as_party(candidate.team, f"notebook {stage}"),
        }
        self.seen[(basis, key)] = record
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        self.written += 1
        return True

    def absorb(self, other: "Notebook") -> int:
        """Take in another machine's book, keeping the better measurement.

        Two PCs running the same search from different ``--seed`` values
        produce two books over one basis, and the point of running both is to
        read them as one list. Nothing is written by this -- the merge is for
        reading, and each machine keeps writing its own file, which is what
        keeps the two out of each other's way in git.
        """
        taken = 0
        for key, record in other.seen.items():
            held = self.seen.get(key)
            if held is None or record["games"] > held["games"]:
                self.seen[key] = record
                taken += 1
        return taken

    def best(self, count: int = 20, basis: str | None = None) -> list[dict]:
        """The best-measured parties in the book, by lower bound."""
        rows = [one for one in self.seen.values()
                if basis is None or one["basis"] == basis]
        return sorted(rows, key=lambda one: (-one["low"], -one["floor"]))[:count]

    def bases(self) -> list[str]:
        return sorted({one["basis"] for one in self.seen.values()})
