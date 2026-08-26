import pathlib
ROOT = pathlib.Path("C:/Users/User/Desktop/intern_HK/pkcm-agent")

def patch(rel, pairs):
    p = ROOT / rel
    t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert old in t, f"{rel}: NOT FOUND:\n{old[:300]}"
        assert t.count(old) == 1, f"{rel}: {t.count(old)} matches for:\n{old[:200]}"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("patched", rel)


patch("src/pkcm/engine/state.py", [
    # -- module docstring: say what changed --------------------------------- #
    ('''Two lifetimes live inside ``SideState`` and the difference matters:''',
     '''A side holds one *field position* per active Pokemon: one in singles, two in
doubles. ``SideState.active`` maps position -> party slot, so a ``Ref`` still
names a Pokemon by the slot it was brought in, not by where it happens to be
standing. That is what lets HP, PP, status, boosts and volatiles stay indexed
by party slot in both formats, and it is why doubles needed no second copy of
any of them.

Two lifetimes live inside ``SideState`` and the difference matters:'''),

    # -- config: how many positions ----------------------------------------- #
    ('''    @property
    def brought(self) -> int:
        return self.regulation.bring_select(self.battle_format)[1]''',
     '''    @property
    def brought(self) -> int:
        return self.regulation.bring_select(self.battle_format)[1]

    @property
    def active_count(self) -> int:
        """Field positions per side: 1 in singles, 2 in doubles."""
        return 2 if self.battle_format == "doubles" else 1

    @property
    def is_doubles(self) -> bool:
        return self.active_count > 1'''),

    # -- SideState.active becomes a list ------------------------------------ #
    ('''    #: Index into ``selection``; -1 before the first switch-in.
    active: int = -1
    #: Set when this side's active fainted and owes a replacement.
    must_switch: bool = False''',
     '''    #: Field position -> index into ``selection``; -1 while a position is empty.
    #: One entry in singles, two in doubles.
    active: list[int] = field(default_factory=list)
    #: Per position: this one's occupant fainted and owes a replacement.
    must_switch: list[bool] = field(default_factory=list)'''),

    ('''            selection=self.selection,
            hp=self.hp.copy(),
            pp=[slot.copy() for slot in self.pp],
            active=self.active,
            must_switch=self.must_switch,''',
     '''            selection=self.selection,
            hp=self.hp.copy(),
            pp=[slot.copy() for slot in self.pp],
            active=self.active.copy(),
            must_switch=self.must_switch.copy(),'''),

    # -- queries ------------------------------------------------------------ #
    ('''    def has_lost(self) -> bool:
        return bool(self.hp) and not self.living_slots()''',
     '''    def has_lost(self) -> bool:
        return bool(self.hp) and not self.living_slots()

    def active_slots(self) -> list[int]:
        """Party slots standing on the field right now, fainted ones dropped.

        A fainted Pokemon stays in its position until a replacement is sent,
        so that ``must_switch`` knows which position it owes. Everything that
        asks "who is out there" wants it gone, which is what this is for.
        """
        return [slot for slot in self.active if slot >= 0 and self.hp[slot] > 0]

    def position_of(self, slot: int) -> int | None:
        """Where a party slot is standing, or ``None`` if it is on the bench."""
        for position, occupant in enumerate(self.active):
            if occupant == slot:
                return position
        return None

    def owes_switch(self) -> bool:
        return any(self.must_switch)'''),
])
print("done")
