"""Turning search into training data, and training data into a better search.

The loop AlphaZero describes, with the two ends already cut to fit:

    search plays itself      -> (observation, visit distribution, outcome)
    a network learns those   -> policy prior and value estimate
    the search uses them     -> stronger search, better data

Both ends are a single function in ``pkcm.search``. ``policy.prior_over`` is the
policy slot and ``evaluate.heuristic`` is the value slot, and the ablation says
what the first one is worth: a crude power-times-effectiveness score there took
the search from 45.8% against a one-turn damage calculator to 66.4%.

The bottleneck is not the network. It is generating the games -- one self-play
battle costs about twelve seconds of search -- which is why ``parallel`` exists
before ``net`` does.
"""

from pkcm.train.samples import Sample, SelfPlayConfig, play_one
from pkcm.train.parallel import generate

__all__ = ["Sample", "SelfPlayConfig", "play_one", "generate"]
