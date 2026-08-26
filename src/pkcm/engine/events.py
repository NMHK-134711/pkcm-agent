"""The battle's output channel.

Principle (e) in docs/DESIGN.md: the engine does not know how anything is
displayed. It emits a flat, structured log and someone else turns that into
English, sprites, or a training signal.

One dataclass with a ``kind`` tag rather than a class per event: the log is
serialized, diffed against Showdown's log, and pattern-matched by renderers, and
all three are easier against a uniform record.
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True, slots=True)
class Event:
    """One thing that happened.

    ``species``, ``move`` and ``detail`` carry **ids**, not display names. The
    engine has no opinion about what language anyone reads; ``pkcm.render.names``
    turns an id into a name, and it is the only place that knows Korean exists.
    """

    kind: str
    side: int | None = None
    slot: int | None = None
    species: str | None = None
    move: str | None = None
    amount: int | None = None
    hp: int | None = None
    max_hp: int | None = None
    effectiveness: float | None = None
    crit: bool = False
    turn: int | None = None
    detail: str | None = None

    def __repr__(self) -> str:  # compact logs are worth the override
        parts = [
            f"{f.name}={value!r}"
            for f in fields(self)
            if f.name != "kind" and (value := getattr(self, f.name)) not in (None, False)
        ]
        return f"{self.kind}({', '.join(parts)})"


def turn_start(turn: int) -> Event:
    return Event("turn_start", turn=turn)


def team_preview(side: int, selection: tuple[int, ...]) -> Event:
    return Event("team_preview", side=side, detail=",".join(map(str, selection)))


def switch_in(side: int, slot: int, species: str, hp: int, max_hp: int) -> Event:
    return Event("switch_in", side=side, slot=slot, species=species, hp=hp, max_hp=max_hp)


def move_used(side: int, slot: int, species: str, move: str) -> Event:
    return Event("move_used", side=side, slot=slot, species=species, move=move)


def missed(side: int, slot: int, move: str) -> Event:
    return Event("missed", side=side, slot=slot, move=move)


def immune(side: int, slot: int, move: str) -> Event:
    """The *defending* side is unaffected."""
    return Event("immune", side=side, slot=slot, move=move)


def damage(
    side: int,
    slot: int,
    amount: int,
    hp: int,
    max_hp: int,
    effectiveness: float,
    crit: bool,
) -> Event:
    return Event(
        "damage",
        side=side,
        slot=slot,
        amount=amount,
        hp=hp,
        max_hp=max_hp,
        effectiveness=effectiveness,
        crit=crit,
    )


def recoil(side: int, slot: int, amount: int, hp: int, max_hp: int) -> Event:
    return Event("recoil", side=side, slot=slot, amount=amount, hp=hp, max_hp=max_hp)


def faint(side: int, slot: int, species: str) -> Event:
    return Event("faint", side=side, slot=slot, species=species)


def out_of_pp(side: int, slot: int, move: str) -> Event:
    return Event("out_of_pp", side=side, slot=slot, move=move)


def battle_end(winner: int | None, detail: str) -> Event:
    return Event("battle_end", side=winner, detail=detail)
