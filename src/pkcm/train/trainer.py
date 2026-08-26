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


def fit(net: ChampionsNet, samples: list[Sample], device: torch.device,
        config: TrainConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None) -> dict[str, float]:
    """One pass (or ``epochs`` passes) over the samples. Returns the losses."""
    settings = config or TrainConfig()
    if not samples:
        return Metrics().mean()

    net.to(device).train()
    optimiser = optimizer or torch.optim.AdamW(
        net.parameters(), lr=settings.learning_rate,
        weight_decay=settings.weight_decay)

    policies = torch.as_tensor(
        np.stack([sample.policy for sample in samples]), dtype=torch.float32)
    values = torch.as_tensor(
        np.array([sample.value for sample in samples]), dtype=torch.float32)

    metrics = Metrics()
    order = np.arange(len(samples))
    for _ in range(settings.epochs):
        np.random.shuffle(order)
        for start in range(0, len(order), settings.batch_size):
            rows = order[start:start + settings.batch_size]
            batch = collate([samples[index].observation for index in rows], device)
            target_policy = policies[rows].to(device)
            target_value = values[rows].to(device)

            logits, value = net(batch)

            # Only over what the search could choose. Everything else is masked
            # out by the environment anyway, and training on it teaches the
            # head to reproduce the mask instead of the policy.
            support = target_policy > 0
            log_probabilities = torch.log_softmax(
                logits.masked_fill(~support, float("-inf")), dim=1)
            policy_loss = -(target_policy * log_probabilities.nan_to_num(0.0)).sum(1).mean()
            value_loss = nn.functional.mse_loss(value, target_value)
            loss = policy_loss + settings.value_weight * value_loss

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), settings.grad_clip)
            optimiser.step()

            metrics.add(float(policy_loss), float(value_loss),
                        float((value - target_value).abs().mean()))
    return metrics.mean()


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
