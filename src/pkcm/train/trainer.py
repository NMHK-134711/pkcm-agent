"""Fitting the two heads to what the search found.

AlphaZero's loss, unchanged, because there is no reason to change it:

    policy   cross-entropy against the search's visit distribution
    value    mean squared error against who eventually won

The visit distribution is a *better* policy than the network that produced it --
that is the whole engine of the loop. The search improves on its own prior by
looking ahead; the network learns the improvement; the next search starts from
there.

Masking matters here in a way it does not in board games. Most of the action
space is illegal at any moment, and a policy head that spends capacity learning
"switch to a fainted Pokemon is rare" is wasting it -- the mask already says so,
for free and exactly. So the cross-entropy runs over the support of the target,
which is the set of actions the search was actually allowed to consider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pkcm.train.net import ChampionsNet, collate
from pkcm.train.samples import Sample


@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1
    #: How much the value head counts against the policy head. AlphaZero uses
    #: one; a lower number is usual when the value target is noisy, and a win
    #: or loss thirty turns away is about as noisy as they come.
    value_weight: float = 0.5
    grad_clip: float = 1.0
    #: Share of *battles* held out. Not samples -- see ``Sample.battle``.
    #:
    #: This is not optional bookkeeping. Without it the first run here reported
    #: a value error of 0.032 while scoring 0.771 on battles it had not seen,
    #: and a constant zero would have scored 1.0. The training curve looked
    #: excellent throughout.
    validation_fraction: float = 0.1
    #: How much of the value target comes from the search's root value rather
    #: than from who eventually won. Zero is AlphaZero's target unchanged.
    #:
    #: AlphaZero can afford the pure outcome because it plays millions of games
    #: from one starting position, so the noise averages out over inputs that
    #: recur. Here the teams are random and no position ever recurs, and at team
    #: preview the observation identifies the battle outright -- which is how
    #: the value head came to emit +-0.99 before anyone had moved. See
    #: ``Sample.search_value``.
    #:
    #: **Off until it is measured.** A target the network partly produced itself
    #: can collapse into predicting its own output, which is exactly the kind of
    #: change that looks like progress in every number except the arena.
    search_value_weight: float = 0.0
    #: How much of the value target is the n-step bootstrap rather than who
    #: won. One is MuZero's target; zero is AlphaZero's. See
    #: ``Sample.bootstrap``.
    #:
    #: **Off until measured.** A target the network partly wrote itself can
    #: collapse into predicting its own output, so validation scores the real
    #: outcome whatever this is set to, and a test holds that line.
    bootstrap_weight: float = 0.0


@dataclass
class Metrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    value_error: float = 0.0
    batches: int = 0

    def add(self, policy: float, value: float, error: float) -> None:
        self.policy_loss += policy
        self.value_loss += value
        self.value_error += error
        self.batches += 1

    def mean(self) -> dict[str, float]:
        count = max(1, self.batches)
        return {
            "policy_loss": self.policy_loss / count,
            "value_loss": self.value_loss / count,
            "value_mae": self.value_error / count,
        }


def split_by_battle(samples: list[Sample], fraction: float,
                    seed: int = 0) -> tuple[list[int], list[int]]:
    """Row indices for training and validation, divided by battle.

    Whole battles go to one side or the other. Splitting samples at random
    would put turn 11 in training and turn 12 in validation, and those two are
    nearly the same position with the same label.
    """
    battles = sorted({sample.battle for sample in samples})
    if fraction <= 0 or len(battles) < 4:
        return list(range(len(samples))), []
    rng = np.random.default_rng(seed)
    shuffled = list(battles)
    rng.shuffle(shuffled)
    held = set(shuffled[:max(1, int(len(shuffled) * fraction))])
    training, validation = [], []
    for index, sample in enumerate(samples):
        (validation if sample.battle in held else training).append(index)
    return training, validation


def fit(net: ChampionsNet, samples: list[Sample], device: torch.device,
        config: TrainConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    """Train, and report both what it fitted and what that is worth.

    The gap between the two is the number that matters. Training loss falls
    whenever the network has capacity; only the held-out score says whether
    anything was learned that applies to a battle it has not seen.
    """
    settings = config or TrainConfig()
    if not samples:
        return Metrics().mean()

    net.to(device).train()
    optimiser = optimizer or torch.optim.AdamW(
        net.parameters(), lr=settings.learning_rate,
        weight_decay=settings.weight_decay)

    policies = torch.as_tensor(
        np.stack([sample.policy for sample in samples]), dtype=torch.float32)
    outcomes = np.array([sample.value for sample in samples], dtype=np.float32)
    blend = settings.search_value_weight
    if blend > 0:
        rooted = np.array([sample.search_value for sample in samples],
                          dtype=np.float32)
        outcomes = (1 - blend) * outcomes + blend * rooted
    boot = settings.bootstrap_weight
    if boot > 0:
        ahead = np.array([sample.bootstrap for sample in samples],
                         dtype=np.float32)
        outcomes = (1 - boot) * outcomes + boot * ahead
    values = torch.as_tensor(outcomes, dtype=torch.float32)
    value_weights = torch.as_tensor(
        np.array([sample.value_weight for sample in samples], dtype=np.float32))
    #: Validation always scores against the real outcome, whatever the training
    #: target was blended from. Scoring against a target the network helped
    #: write would make a collapse into self-prediction look like success.
    truth = torch.as_tensor(
        np.array([sample.value for sample in samples]), dtype=torch.float32)
    training, validation = split_by_battle(samples, settings.validation_fraction)

    metrics = Metrics()
    order = np.array(training)
    for _ in range(settings.epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), settings.batch_size):
            rows = order[start:start + settings.batch_size]
            batch = collate([samples[index].observation for index in rows], device)
            target_policy = policies[rows].to(device)
            target_value = values[rows].to(device)
            weights = value_weights[rows].to(device)

            logits, value = net(batch)

            # Only over what the search could choose. Everything else is masked
            # out by the environment anyway, and training on it teaches the
            # head to reproduce the mask instead of the policy.
            support = target_policy > 0
            log_probabilities = torch.log_softmax(
                logits.masked_fill(~support, float("-inf")), dim=1)
            policy_loss = -(target_policy * log_probabilities.nan_to_num(0.0)).sum(1).mean()
            # Weighted, so a rehearsal row can train the policy head and
            # abstain on the value. Normalised by the weight actually present
            # rather than the batch size, or a batch that is mostly rehearsal
            # would quietly shrink the value gradient as well as narrowing it.
            squared = (value - target_value) ** 2 * weights
            present = weights.sum()
            value_loss = squared.sum() / present if present > 0 else squared.sum() * 0
            loss = policy_loss + settings.value_weight * value_loss

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
            optimiser.step()

            error = ((value - target_value).abs() * weights).sum()
            metrics.add(float(policy_loss), float(value_loss),
                        float(error / present) if present > 0 else 0.0)

    result = metrics.mean()
    result.update(_validate(net, samples, validation, policies, truth,
                            device, settings))
    return result


def _validate(net: ChampionsNet, samples: list[Sample], rows: list[int],
              policies: torch.Tensor, values: torch.Tensor,
              device: torch.device, settings: TrainConfig) -> dict[str, float]:
    """Score battles the network was not trained on.

    ``value_mae`` against a baseline: predicting a constant zero scores 1.0 on
    a target of plus or minus one, so anything near 1.0 has learned nothing at
    all, whatever the training curve is doing.
    """
    if not rows:
        return {}
    net.eval()
    total_policy = total_value = 0.0
    batches = 0
    with torch.no_grad():
        for start in range(0, len(rows), settings.batch_size):
            chunk = rows[start:start + settings.batch_size]
            batch = collate([samples[index].observation for index in chunk], device)
            target_policy = policies[chunk].to(device)
            target_value = values[chunk].to(device)
            logits, value = net(batch)
            support = target_policy > 0
            log_probabilities = torch.log_softmax(
                logits.masked_fill(~support, float("-inf")), dim=1)
            total_policy += float(
                -(target_policy * log_probabilities.nan_to_num(0.0)).sum(1).mean())
            total_value += float((value - target_value).abs().mean())
            batches += 1
    net.train()
    count = max(1, batches)
    return {
        "val_policy_loss": total_policy / count,
        "val_value_mae": total_value / count,
        "val_battles": float(len({samples[index].battle for index in rows})),
    }


def save(net: ChampionsNet, path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(),
                "config": net.config,
                "action_space": net.action_space,
                **(extra or {})}, path)


def load_into(net: ChampionsNet, path: Path, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    return payload
