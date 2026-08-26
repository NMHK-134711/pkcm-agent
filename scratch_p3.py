import pathlib
ROOT = pathlib.Path("C:/Users/User/Desktop/intern_HK/pkcm-agent")

def patch(rel, pairs):
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in t, f"{rel}: NOT FOUND:\n{old[:300]}"
        assert t.count(old) == 1, f"{rel}: {t.count(old)} matches"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("patched", rel)


patch("src/pkcm/engine/actions.py", [
    ('''``TEAM_PREVIEW``   an ordered selection of the Pokemon to bring; index 0 leads.
``BATTLE``         a move or a switch.
``FORCED_SWITCH``  a switch, from whichever side just lost its active Pokemon.
                   The other side submits ``PASS``.''',
     '''``TEAM_PREVIEW``   an ordered selection of the Pokemon to bring; index 0 leads.
``BATTLE``         a move or a switch.
``FORCED_SWITCH``  a switch, from whichever side just lost its active Pokemon.
                   The other side submits ``PASS``.

In doubles each side decides once per *field position*, so a player submits a
tuple of actions rather than one -- see ``pkcm.engine.battle.step``. A move
action also carries a ``target``, because "the opponent" stops being a single
Pokemon the moment there are two of them.'''),

    ('''class ActionKind(IntEnum):''',
     '''#: Where a move is aimed. Foe field positions are numbered from zero, which
#: makes ``target=0`` mean "the other side's first slot" in both formats -- so a
#: singles action is a doubles action that never needed the field.
TARGET_ALLY = -1
TARGET_SELF = -2


class ActionKind(IntEnum):'''),

    ('''    #: Mega Evolve first, then use the move. Champions allows it once a battle,
    #: so it is a property of the action rather than a separate decision -- the
    #: player commits to spending it on this turn's move.
    mega: bool = False

    @staticmethod
    def move(index: int, mega: bool = False) -> "Action":
        return Action(ActionKind.MOVE, index, mega=mega)''',
     '''    #: Mega Evolve first, then use the move. Champions allows it once a battle,
    #: so it is a property of the action rather than a separate decision -- the
    #: player commits to spending it on this turn's move.
    mega: bool = False
    #: Which Pokemon a move is aimed at: a foe's field position, ``TARGET_ALLY``
    #: or ``TARGET_SELF``. Ignored by moves that do not choose (spread moves,
    #: field moves, and everything in singles, where there is one answer).
    target: int = 0

    @staticmethod
    def move(index: int, mega: bool = False, target: int = 0) -> "Action":
        return Action(ActionKind.MOVE, index, mega=mega, target=target)'''),

    ('''        prefix = "mega+" if self.mega else ""
        return f"{prefix}{self.kind.name.lower()}({self.index})"''',
     '''        prefix = "mega+" if self.mega else ""
        if self.kind is ActionKind.MOVE and self.target != 0:
            aim = {TARGET_ALLY: "ally", TARGET_SELF: "self"}.get(self.target, self.target)
            return f"{prefix}move({self.index}->{aim})"
        return f"{prefix}{self.kind.name.lower()}({self.index})"'''),
])
print("done")
