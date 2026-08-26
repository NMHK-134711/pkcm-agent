"""Search that plays the game the game actually is.

Two things make Champions awkward for a textbook tree search, and both are
handled here rather than approximated away:

**It is simultaneous.** Both sides commit before either sees the other. A
sequential tree would let the second player read the first one's move, and a
search built on that advantage plays a game nobody else is playing. Each node
therefore holds *two* sets of action statistics and picks a pair.

**It is imperfect information.** The opponent's items, moves, spreads and bench
are hidden. The search never looks at them: every iteration draws a fresh
determinization from the observation (``pkcm.envs.observation.determinize``) and
searches that, so what survives many iterations is what works against the
*range* of teams consistent with what has been seen.
"""

from pkcm.search.mcts import MCTS, SearchConfig, SearchResult
from pkcm.search.policy import GreedyPolicy, Policy, RandomPolicy, SearchPolicy

__all__ = [
    "MCTS", "SearchConfig", "SearchResult",
    "Policy", "RandomPolicy", "GreedyPolicy", "SearchPolicy",
]
