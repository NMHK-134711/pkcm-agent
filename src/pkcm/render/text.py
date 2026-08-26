"""Turn the engine's event log into readable lines.

A consumer of the log, not part of the engine (docs/DESIGN.md §1e). A sprite
renderer would be a sibling of this module and would need no engine changes.
"""

from __future__ import annotations

from pkcm.engine.events import Event

EFFECTIVENESS_TEXT = {
    0.25: "It's not very effective...",
    0.5: "It's not very effective...",
    2.0: "It's super effective!",
    4.0: "It's super effective!",
}

STATUS_TEXT = {
    "brn": "was burned",
    "par": "was paralyzed",
    "psn": "was poisoned",
    "tox": "was badly poisoned",
    "slp": "fell asleep",
    "frz": "was frozen solid",
}

CANT_MOVE_TEXT = {
    "par": "is paralyzed! It can't move!",
    "slp": "is fast asleep.",
    "frz": "is frozen solid!",
    "flinch": "flinched!",
}

RESIDUAL_TEXT = {
    "brn": "was hurt by its burn",
    "psn": "was hurt by poison",
    "tox": "was hurt by poison",
    "leechseed": "had its health sapped by Leech Seed",
    "sandstorm": "was buffeted by the sandstorm",
    "spikes": "was hurt by Spikes",
    "stealthrock": "was hurt by Stealth Rock",
    "substitute": "put in a substitute",
}

FIELD_TEXT = {
    "sunnyday": "The sunlight turned harsh!",
    "raindance": "It started to rain!",
    "sandstorm": "A sandstorm kicked up!",
    "snowscape": "It started to snow!",
    "electricterrain": "An electric current ran across the battlefield!",
    "grassyterrain": "Grass grew to cover the battlefield!",
    "mistyterrain": "Mist swirled around the battlefield!",
    "psychicterrain": "The battlefield got weird!",
    "trickroom": "The dimensions were twisted!",
}


def _who(event: Event, names: tuple[str, str] | None) -> str:
    if names is None or event.side is None:
        return f"P{event.side}"
    return names[event.side]


def _hp(event: Event) -> str:
    return f"-> {event.hp}/{event.max_hp}"


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

    if kind == "protected":
        return [f"  It protected itself from {event.move}!"]

    if kind == "move_failed":
        detail = f" ({event.detail})" if event.detail else ""
        return [f"  ...but it failed!{detail}"]

    if kind == "unimplemented":
        return [f"  (not implemented yet: {event.detail})"]

    if kind == "cant_move":
        return [f"  It {CANT_MOVE_TEXT.get(event.detail, event.detail)}"]

    if kind == "confused":
        return ["  It hurt itself in its confusion!"]

    if kind == "damage":
        lines = []
        if event.crit:
            lines.append("  A critical hit!")
        note = EFFECTIVENESS_TEXT.get(event.effectiveness)
        if note:
            lines.append(f"  {note}")
        percent = 100 * event.amount / event.max_hp
        lines.append(f"  -{event.amount} HP ({percent:.0f}%) {_hp(event)}")
        return lines

    if kind in ("status_damage", "weather_damage", "hazard_damage"):
        text = RESIDUAL_TEXT.get(event.detail, event.detail)
        return [f"  It {text}. -{event.amount} {_hp(event)}"]

    if kind == "recoil":
        return [f"  It was hurt by recoil! -{event.amount} {_hp(event)}"]

    if kind == "heal":
        source = f" ({event.detail})" if event.detail else ""
        return [f"  It restored {event.amount} HP{source}. {_hp(event)}"]

    if kind == "substitute_hit":
        return [f"  The substitute took the hit! (-{event.amount})"]

    if kind == "multi_hit":
        return [f"  Hit {event.amount} times!"]

    if kind == "status":
        return [f"  It {STATUS_TEXT.get(event.detail, event.detail)}!"]

    if kind == "cure_status":
        return [f"  It shook off its {event.detail}."]

    if kind == "boost":
        direction = "rose" if event.amount > 0 else "fell"
        size = {1: "", 2: " sharply", 3: " drastically"}.get(abs(event.amount), "")
        return [f"  Its {event.detail}{size} {direction}! (now {event.hp:+d})"]

    if kind == "boost_failed":
        limit = "any higher" if event.amount > 0 else "any lower"
        return [f"  Its {event.detail} won't go {limit}!"]

    if kind == "volatile_start":
        return [f"  {event.detail} started."]

    if kind == "volatile_end":
        return [f"  {event.detail} ended."]

    if kind in ("weather_start", "terrain_start", "room_start"):
        return [f"  {FIELD_TEXT.get(event.detail, event.detail)}"]

    if kind in ("weather_end", "terrain_end", "room_end"):
        return [f"  The {event.detail} ended."]

    if kind == "mega_evolve":
        return [f"  {event.species}! It Mega Evolved!"]

    if kind == "forme_change":
        return []  # the mechanic that caused it says something more useful

    if kind == "transform":
        return [f"  It transformed into {event.detail}!"]

    if kind == "type_change":
        return [f"  Its type changed to {event.detail}!"]

    if kind == "item_used":
        return [f"  Its {event.detail} was used up."]

    if kind == "item":
        return [f"  [{event.detail}]"]

    if kind == "boost_restored":
        return ["  Its lowered stats were restored!"]

    if kind == "ability_change":
        return [f"  Its ability became {event.detail}!"]

    if kind == "ability":
        return [f"  [{event.detail}]"]

    if kind == "ability_block":
        return [f"  {event.detail} blocked {event.move}!"]

    if kind == "ability_suppressed":
        return [f"  ({event.detail} is ignored)"]

    if kind == "status_immune":
        return ["  It can't be affected by that."]

    if kind == "side_condition_end":
        return [f"  {_who(event, names)}'s {event.detail} wore off."]

    if kind == "side_condition":
        return [f"  {_who(event, names)}'s side: {event.detail} (x{event.amount})"]

    if kind == "hazard_absorbed":
        return [f"  {_who(event, names)}'s {event.detail} was absorbed."]

    if kind == "faint":
        return [f"  {event.species} fainted!"]

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
