"""The two heads the search is missing, in one network.

    policy   a distribution over the flat action space -> the search's prior
    value    who is winning, in [-1, 1]              -> the search's leaf

Both slots are already cut: ``search.policy.prior_over`` and
``search.evaluate.heuristic``. The ablation says what the policy slot alone is
worth -- a crude power-times-effectiveness score there took the search from
45.8% against a one-turn damage calculator to 66.4%.

**The dex is fed in, not learned.** ``pkcm.envs.reference`` already ships base
stats, types, power, accuracy and the rest as tables indexed by the same ids the
observation uses. They go in as a frozen embedding the network gathers from, so
it never has to spend capacity memorising that Garchomp is Dragon/Ground -- and
so it is not helpless in front of a species it saw twice in training. A small
learned embedding sits alongside for whatever the tables do not say.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pkcm.envs.encoding import MAX_BROUGHT, Vocabulary
from pkcm.envs.reference import ReferenceSheet
from pkcm.engine.pokemon import MAX_MOVES


@dataclass(frozen=True, slots=True)
class NetConfig:
    #: Learned embedding width per categorical field. Small on purpose: the
    #: reference tables carry the facts, and this only has to carry the rest.
    embedding: int = 24
    hidden: int = 512
    blocks: int = 3
    dropout: float = 0.0


class _Residual(nn.Module):
    """Pre-norm residual block. Depth without the training getting delicate."""

    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ChampionsNet(nn.Module):
    """Observation in, policy logits and a value out."""

    def __init__(self, vocabulary: Vocabulary, sheet: ReferenceSheet,
                 action_space: int, scalar_size: int,
                 config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
        self.action_space = action_space
        sizes = vocabulary.sizes()
        width = self.config.embedding

        self.species = nn.Embedding(sizes["species"], width, padding_idx=0)
        self.moves = nn.Embedding(sizes["moves"], width, padding_idx=0)
        self.items = nn.Embedding(sizes["items"], width, padding_idx=0)
        self.abilities = nn.Embedding(sizes["abilities"], width, padding_idx=0)
        self.status = nn.Embedding(sizes["statuses"], width, padding_idx=0)

        # The dex, frozen. Registered as buffers so they move with the model to
        # the GPU and are saved with it, but never receive a gradient.
        self.register_buffer("species_facts",
                             torch.from_numpy(sheet.species.copy()), persistent=False)
        self.register_buffer("move_facts",
                             torch.from_numpy(sheet.moves.copy()), persistent=False)
        species_facts = sheet.species.shape[1]
        move_facts = sheet.moves.shape[1]

        slots = 2 * MAX_BROUGHT
        features = (
            scalar_size
            + slots * (width + species_facts)          # species: learned + facts
            + slots * width * 3                        # item, ability, status
            + slots * MAX_MOVES * (width + move_facts)  # moves: learned + facts
            + slots * MAX_MOVES                        # pp
            # The registered six get the facts too, not just the learned id.
            # At team preview they are the *only* non-zero input -- nothing
            # has been brought, so every other array is zeros -- and the
            # handcrafted pick prior is computed from types and base stats.
            # Shown ids alone, the policy agreed with it 5.0% of the time on
            # a 24-way choice, which is exactly chance.
            + 12 * (width + species_facts)             # the registered six
            # Our own six in full. At team preview every other per-Pokemon
            # array is zeros, so without this a physical set and a special
            # set on the same species are the *same input* -- and the pick
            # they justify is different. Measured before it was here, the
            # policy agreed with the handcrafted pick 6.7% of the time on a
            # 24-way choice, against 4.2% for guessing.
            + 6 * MAX_MOVES * (width + move_facts)     # our six's moves
            + 6 * width * 2                            # our six's item, ability
            + 6 * 6                                    # our six's stats
            + 16 * 7 + 4 * 2 + 4 * 5                   # matchup, speed, risk
            + 36 * 3                                   # our six vs their six
        )
        self.stem = nn.Sequential(
            nn.Linear(features, self.config.hidden),
            nn.GELU(),
        )
        self.body = nn.Sequential(*[
            _Residual(self.config.hidden, self.config.dropout)
            for _ in range(self.config.blocks)
        ])
        self.norm = nn.LayerNorm(self.config.hidden)
        self.policy_head = nn.Linear(self.config.hidden, action_space)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden, 128), nn.GELU(), nn.Linear(128, 1), nn.Tanh(),
        )

    # -- forward ------------------------------------------------------------ #

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = batch["scalars"].shape[0]

        def flat(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(rows, -1)

        species = batch["species"]
        moves = batch["moves"]
        registered = batch["registered"]
        own_moves = batch["own_moves"]
        parts = [
            batch["scalars"],
            flat(self.species(species)),
            flat(self.species_facts[species]),
            flat(self.items(batch["items"])),
            flat(self.abilities(batch["abilities"])),
            flat(self.status(batch["status"])),
            flat(self.moves(moves)),
            flat(self.move_facts[moves]),
            batch["pp"],
            flat(self.species(registered)),
            flat(self.species_facts[registered]),
            flat(self.moves(own_moves)),
            flat(self.move_facts[own_moves]),
            flat(self.items(batch["own_items"])),
            flat(self.abilities(batch["own_abilities"])),
            batch["own_stats"],
            flat(batch["matchup"]),
            flat(batch["speed"]),
            flat(batch["risk"]),
            flat(batch["preview"]),
        ]
        hidden = self.norm(self.body(self.stem(torch.cat(parts, dim=1))))
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    # -- inference ---------------------------------------------------------- #

    @torch.no_grad()
    def evaluate(self, observations: list[dict[str, np.ndarray]],
                 device: torch.device | str = "cpu") -> tuple[np.ndarray, np.ndarray]:
        """Policy probabilities and values for a batch of encoded observations."""
        self.eval()
        batch = collate(observations, device)
        logits, value = self(batch)
        return (torch.softmax(logits, dim=1).cpu().numpy(), value.cpu().numpy())


#: Which observation fields are indices rather than numbers.
CATEGORICAL = ("species", "moves", "items", "abilities", "status", "registered",
               "own_moves", "own_items", "own_abilities")


def collate(observations: list[dict[str, np.ndarray]],
            device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Stack encoded observations into one batch on ``device``."""
    batch: dict[str, torch.Tensor] = {}
    for key in observations[0]:
        stacked = np.stack([observation[key] for observation in observations])
        dtype = torch.int64 if key in CATEGORICAL else torch.float32
        batch[key] = torch.as_tensor(stacked, dtype=dtype, device=device)
    return batch


def build(vocabulary: Vocabulary, sheet: ReferenceSheet, action_space: int,
          scalar_size: int, config: NetConfig | None = None) -> ChampionsNet:
    return ChampionsNet(vocabulary, sheet, action_space, scalar_size, config)


def pick_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
