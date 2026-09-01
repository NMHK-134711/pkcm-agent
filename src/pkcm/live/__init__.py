"""Playing a real game with the search's advice.

Everything else in this project plays both sides of a battle it owns. Here the
battle is happening in Champions, on a Switch or a PC, and we are watching it
through whatever the person at the keyboard types in. That is a different
problem, and the difference is the whole content of this package: the state is
a *guess*, it drifts, and the correction has to be cheaper than the turn timer.
"""

from pkcm.live.mirror import Mirror, MirrorError

__all__ = ["Mirror", "MirrorError"]
