"""PettingZoo adapter. Wraps the engine; it is not the engine.

Principle (f) in docs/DESIGN.md. Search rides ``pkcm.engine`` directly, where a
state is a cheap value and ``step`` is pure. This exists so the learning half
gets the standard API and the wrapper ecosystem without the search half paying
for it.

**ParallelEnv, not AEC.** Champions is genuinely simultaneous. Serialising it
into AEC would mean the second agent sees the first one's action every turn, and
every one of those leaks would have to be plugged by hand.

**One agent per player, not per field position.** In doubles a player submits
one action per position, as an array. The tempting alternative -- an agent per
position -- cannot express the one constraint that actually spans them: the same
Pokemon may not be sent to both. There would be nowhere to put it.

Turns where only one side decides (a replacement after a knockout) are handled
the way the design says: the idle side is masked down to ``PASS``, which is a
legal action rather than a special case.

**An illegal action forfeits the game** (``on_illegal="lose"``), which is what
PettingZoo's own chess environment does and what makes the standard conformance
test -- which samples the action space uniformly and ignores masks -- meaningful
rather than an immediate crash. A policy that reads ``infos[agent]["action_mask"]``
can never reach it. While developing one that should be reading the mask and is
not, ``on_illegal="raise"`` turns it back into an exception at the point of the
mistake.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from pkcm.data.dex import Dex, load_dex
from pkcm.engine.actions import Action
from pkcm.engine.battle import IllegalActionError, step
from pkcm.engine.legality import random_team
from pkcm.engine.pokemon import MAX_MOVES, Team
from pkcm.engine.rng import Rng
from pkcm.engine.state import BattleConfig, Phase, new_battle
from pkcm.envs.encoding import (
    MATCHUP_FEATURES,
    MATCHUP_ROWS,
    MAX_BROUGHT,
    SCALAR_SIZE,
    SPEED_FEATURES,
    SPEED_ROWS,
    Vocabulary,
    action_mask,
    action_space_size,
    decode_action,
    encode_observation,
)
from pkcm.envs.observation import Observation
from pkcm.envs.reference import ReferenceSheet, sheet_for

AGENTS = ("player_0", "player_1")

#: Reward is the game's own signal and nothing else: +1 for the win, -1 for the
#: loss, 0 for a draw, 0 every other turn. Shaping it with damage dealt teaches
#: trading HP for HP, which is not the same game.
WIN_REWARD = 1.0


class ChampionsEnv(ParallelEnv):
    """Champions singles 6->3 or doubles 6->4 as a PettingZoo environment."""

    metadata = {"name": "pkcm_champions_v0", "is_parallelizable": True}

    def __init__(
        self,
        battle_format: str = "singles",
        dex: Dex | None = None,
        regulation: str = "m_b",
        teams: tuple[Team, Team] | None = None,
        seed: int | None = None,
        on_illegal: str = "lose",
        with_analysis: bool = True,
    ) -> None:
        if on_illegal not in ("lose", "raise"):
            raise ValueError(f"on_illegal must be 'lose' or 'raise', not {on_illegal!r}")
        self.on_illegal = on_illegal
        self.dex = dex if dex is not None else load_dex()
        self.config = BattleConfig(
            dex=self.dex,
            regulation=self.dex.regulation(regulation),
            battle_format=battle_format,
        )
        self.vocabulary = Vocabulary.of(self.dex)
        #: The dex as lookup tables. Exposed rather than folded into the
        #: observation because it never changes -- a policy embeds it once and
        #: gathers rows, instead of being handed 316 species every step.
        self.reference: ReferenceSheet = sheet_for(self.dex, self.vocabulary)
        #: Whether the observation carries the damage calculator's block.
        self.with_analysis = with_analysis
        self.positions = self.config.active_count
        self.n_actions = action_space_size(self.config.registered, self.config.brought)

        #: Fixed teams, or ``None`` to draw a fresh random pair every episode.
        self.fixed_teams = teams
        self._seed = seed
        self.possible_agents = list(AGENTS)
        self.agents: list[str] = []
        self.state = None
        self._episode = 0
        self._last_log: list = []
        #: Set when a player forfeits by submitting something illegal.
        self._forfeited: int | None = None
        self._observation_space: spaces.Space | None = None
        self._action_space: spaces.Space | None = None

    # -- spaces ------------------------------------------------------------- #

    def observation_space(self, agent: str) -> spaces.Space:  # noqa: D401
        """Cached on the instance. ``lru_cache`` on a method would keep every
        env that ever asked alive for the life of the process."""
        if self._observation_space is None:
            self._observation_space = self._build_observation_space()
        return self._observation_space

    def _build_observation_space(self) -> spaces.Space:
        sizes = self.vocabulary.sizes()
        slots = 2 * MAX_BROUGHT
        fields = {
            "scalars": spaces.Box(-1.0, 1.0, (SCALAR_SIZE,), dtype=np.float32),
            "species": spaces.MultiDiscrete([sizes["species"]] * slots),
            "status": spaces.MultiDiscrete([sizes["statuses"]] * slots),
            "items": spaces.MultiDiscrete([sizes["items"]] * slots),
            "abilities": spaces.MultiDiscrete([sizes["abilities"]] * slots),
            "moves": spaces.MultiDiscrete([sizes["moves"]] * (slots * MAX_MOVES)),
            "pp": spaces.Box(0.0, 1.0, (slots * MAX_MOVES,), dtype=np.float32),
            "registered": spaces.MultiDiscrete([sizes["species"]] * 12),
        }
        if self.with_analysis:
            # Effectiveness runs from x0 to x4, so log2 lands in [-2, 2].
            fields["matchup"] = spaces.Box(-2.0, 2.0,
                                           (MATCHUP_ROWS, MATCHUP_FEATURES),
                                           dtype=np.float32)
            fields["speed"] = spaces.Box(0.0, 1.0, (SPEED_ROWS, SPEED_FEATURES),
                                         dtype=np.float32)
        return spaces.Dict(fields)

    def action_space(self, agent: str) -> spaces.Space:  # noqa: D401
        """One index per field position. Singles has one; doubles has two."""
        if self._action_space is None:
            self._action_space = spaces.MultiDiscrete([self.n_actions] * self.positions)
        return self._action_space

    # -- episode ------------------------------------------------------------ #

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
        episode_seed = (self._seed if self._seed is not None else 0) + self._episode
        self._episode += 1

        teams = self.fixed_teams or (
            random_team(self.dex, self.config.regulation,
                        Rng.from_seed(episode_seed * 2 + 1).cursor(),
                        self.config.battle_format),
            random_team(self.dex, self.config.regulation,
                        Rng.from_seed(episode_seed * 2 + 2).cursor(),
                        self.config.battle_format),
        )
        self.state = new_battle(self.config, teams, seed=episode_seed)
        self.agents = list(self.possible_agents)
        self._last_log = []
        self._forfeited = None
        views = self._views()
        return self._observations(views), self._infos(views)

    def step(self, actions: dict[str, Any]):
        decoded = tuple(
            self._decode(player, actions.get(AGENTS[player])) for player in (0, 1)
        )
        try:
            self.state, self._last_log = step(self.state, decoded[0], decoded[1])
        except IllegalActionError as error:
            if self.on_illegal == "raise":
                raise
            # Never substitute something legal: that would teach from a turn
            # that did not happen. Forfeit instead -- the outcome is wrong for
            # the player who broke the rules, and right for everyone else.
            self._forfeited = self._blame(error, decoded)
            self._last_log = []
            self.agents = []
            views = self._views()
            observations, infos = self._observations(views), self._infos(views)
            return (observations, self._rewards(),
                    {agent: True for agent in AGENTS},
                    {agent: False for agent in AGENTS}, infos)

        finished = self.state.phase is Phase.FINISHED
        rewards = self._rewards() if finished else {agent: 0.0 for agent in AGENTS}
        terminations = {agent: finished for agent in AGENTS}
        truncations = {agent: False for agent in AGENTS}
        views = self._views()
        observations, infos = self._observations(views), self._infos(views)
        if finished:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def _blame(self, error: IllegalActionError, decoded) -> int:
        """Which player broke the rules. The message names them; trust it, and
        fall back to checking both if the wording ever changes."""
        message = str(error)
        for player in (0, 1):
            if f"player {player}" in message:
                return player
        return 0

    # -- translation -------------------------------------------------------- #

    def _decode(self, player: int, chosen) -> tuple[Action, ...]:
        """Indices to actions, padded to one per position.

        Team preview asks for a single decision from each player, not one per
        position, so it is the one phase where the tuple is short.
        """
        if chosen is None:
            return (Action.PASS,) * self._expected(player)
        if np.isscalar(chosen) or isinstance(chosen, (int, np.integer)):
            chosen = [int(chosen)]
        indices = [int(value) for value in chosen][: self._expected(player)]
        return tuple(
            decode_action(index, self.config.registered, self.config.brought)
            for index in indices
        )

    def _expected(self, player: int) -> int:
        return 1 if self.state.phase is Phase.TEAM_PREVIEW else self.positions

    def _views(self) -> tuple[Observation, Observation]:
        """One structured observation per player, built once per step.

        Both the arrays and the masks come from it. Building it twice was a
        third of the adapter's cost, and the two copies could in principle
        disagree -- they are snapshots of a state that a step is mutating.
        """
        return tuple(Observation.of(self.state, player) for player in (0, 1))

    def _observations(self, views=None) -> dict[str, dict]:
        views = views if views is not None else self._views()
        sheet = self.reference if self.with_analysis else None
        dex = self.dex if self.with_analysis else None
        return {
            agent: encode_observation(views[player], self.vocabulary, sheet, dex)
            for player, agent in enumerate(AGENTS)
        }

    def _infos(self, views=None) -> dict[str, dict]:
        views = views if views is not None else self._views()
        infos: dict[str, dict] = {}
        for player, agent in enumerate(AGENTS):
            observation = views[player]
            masks = np.stack([
                action_mask(observation, position,
                            self.config.registered, self.config.brought)
                for position in range(self.positions)
            ])
            infos[agent] = {
                "action_mask": masks,
                "phase": self.state.phase.name,
                "turn": self.state.turn,
                # Team preview asks one question, so only the first row means
                # anything there; a policy that reads the mask cannot tell the
                # difference, because the rest is masked to PASS anyway.
                "decisions": self._expected(player),
            }
        return infos

    def _rewards(self) -> dict[str, float]:
        if self._forfeited is not None:
            return {
                agent: (-WIN_REWARD if player == self._forfeited else WIN_REWARD)
                for player, agent in enumerate(AGENTS)
            }
        winner = self.state.winner
        if winner is None:
            return {agent: 0.0 for agent in AGENTS}
        return {
            agent: (WIN_REWARD if player == winner else -WIN_REWARD)
            for player, agent in enumerate(AGENTS)
        }

    # -- extras the engine can give and PettingZoo has no slot for ----------- #

    def battle_state(self):
        """The real state. For search and for tests -- never for a policy."""
        return self.state

    def observation_of(self, player: int) -> Observation:
        """The structured observation, before it was flattened into arrays."""
        return Observation.of(self.state, player)

    def assess(self, player: int, position: int = 0):
        """The damage calculator's answer, as a structure rather than as floats.

        The same numbers the ``matchup`` block carries, in the shape a human --
        or a language model driving the engine through tools -- would want them.
        Reads the observation only, so it cannot leak what the player cannot see.
        """
        from pkcm.envs.analysis import assess as run

        return run(Observation.of(self.state, player), self.reference, self.dex,
                   position)

    def event_log(self) -> list:
        return list(self._last_log)

    def render(self) -> str | None:
        from pkcm.render.text import Renderer

        return Renderer("en", self.dex).render_log(self._last_log)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def env(**kwargs) -> ChampionsEnv:
    """PettingZoo's conventional constructor name."""
    return ChampionsEnv(**kwargs)


def legal_indices(mask: np.ndarray) -> Iterable[int]:
    """Convenience for the random policies in tests and demos."""
    return np.flatnonzero(mask)


def sample_legal(mask: np.ndarray, rng: np.random.Generator) -> int:
    choices = np.flatnonzero(mask)
    return int(rng.choice(choices)) if choices.size else 0
