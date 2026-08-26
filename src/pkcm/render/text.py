"""Turn the engine's event log into readable lines.

A consumer of the log, not part of the engine (docs/DESIGN.md §1e). A sprite
renderer would be a sibling of this module and would need no engine changes.
"""

from __future__ import annotations

from pkcm.engine.events import Event

EFFECTIVENESS_TEXT = {
    0.0: "It doesn't affect the target...",
    0.25: "It's not very effective...",
    0.5: "It's not very effective...",
    2.0: "It's super effective!",
    4.0: "It's super effective!",
}


def _who(event: Event, names: tuple[str, str] | None) -> str:
    if names is None or event.side is None:
        return f"P{event.side}"
    return names[event.side]


def render(event: Event, names: tuple[str, str] | None = None) -> list[str]:
    """Zero or more lines for one event."""
    kind = event.kind

    if kind == "turn_start":
        return ["", f"--- Turn {event.turn} ---"]

    if kind == "team_preview":
        return [f"{_who(event, names)} brings {event.detail}"]

    if kind == "switch_in":
        return [f"{_who(event, names)} sent out {event.species}! ({event.hp}/{event.max_hp})"]

    if kind == "move_used":
        return [f"{event.species} used {event.move}!"]

    if kind == "missed":
        return ["  ...but it missed!"]

    if kind == "immune":
        return ["  It doesn't affect the target..."]

    if kind == "unimplemented":
        return [f"  (not implemented yet: {event.detail})"]

    if kind == "damage":
        lines = []
        if event.crit:
            lines.append("  A critical hit!")
        note = EFFECTIVENESS_TEXT.get(event.effectiveness)
        if note:
            lines.append(f"  {note}")
        percent = 100 * event.amount / event.max_hp
        lines.append(f"  -{event.amount} HP ({percent:.0f}%) -> {event.hp}/{event.max_hp}")
        return lines

    if kind == "recoil":
        return [f"  It was hurt by recoil! -{event.amount} -> {event.hp}/{event.max_hp}"]

    if kind == "faint":
        return [f"  {event.species} fainted!"]

    if kind == "out_of_pp":
        return [f"  {event.move} has no PP left!"]

    if kind == "battle_end":
        if event.side is None:
            return ["", f"Draw. ({event.detail})"]
        return ["", f"{_who(event, names)} wins! ({event.detail})"]

    return [repr(event)]


def render_log(log: list[Event], names: tuple[str, str] | None = None) -> str:
    lines: list[str] = []
    for event in log:
        lines.extend(render(event, names))
    return "\n".join(lines)
