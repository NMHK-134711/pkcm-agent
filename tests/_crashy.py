"""Helpers for the pool-resilience test.

Module level and importable, because a spawned worker unpickles the task
function by name and cannot reach anything defined inside a test.
"""
from __future__ import annotations

import os

_MARKER = ""


def remember(path: str) -> None:
    global _MARKER
    _MARKER = path


def crash_first_time(item: int) -> int:
    """Die once, the way a segfaulting worker does -- no exception, no unwind."""
    if item == 3 and not os.path.exists(_MARKER):
        open(_MARKER, "w").close()
        os._exit(1)
    return item * 2
