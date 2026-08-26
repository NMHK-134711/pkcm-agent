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
    ('''    def active_pokemon(self, side: int) -> BattlePokemon:
        return self.pokemon(side, self.sides[side].active)''',
     '''    def active_pokemon(self, side: int, position: int = 0) -> BattlePokemon:
        return self.pokemon(side, self.sides[side].active[position])

    # -- who is on the field ------------------------------------------------ #

    def active_refs(self, side: int) -> list[Ref]:
        """Every Pokemon this side has standing, in field-position order."""
        return [(side, slot) for slot in self.sides[side].active_slots()]

    def ref_at(self, side: int, position: int) -> Ref | None:
        """Whoever occupies one field position, or ``None`` if it is empty."""
        occupants = self.sides[side].active
        if position >= len(occupants):
            return None
        slot = occupants[position]
        if slot < 0 or self.sides[side].hp[slot] <= 0:
            return None
        return (side, slot)

    def foes(self, ref: Ref) -> list[Ref]:
        """The opposing Pokemon on the field. Both of them, in doubles."""
        return self.active_refs(1 - ref[0])

    def ally(self, ref: Ref) -> Ref | None:
        """The partner standing beside this one. Always ``None`` in singles."""
        for other in self.active_refs(ref[0]):
            if other != ref:
                return other
        return None

    def allies_and_self(self, ref: Ref) -> list[Ref]:
        return self.active_refs(ref[0])

    def everyone(self) -> list[Ref]:
        return self.active_refs(0) + self.active_refs(1)'''),

    ('''    def active_hp(self, side: int) -> int:
        return self.sides[side].hp[self.sides[side].active]

    def speed(self, side: int) -> int:
        """Raw Speed. Stage multipliers and Speed-modifying effects live in
        ``pkcm.engine.effects``; this is the unmodified number."""
        return self.stats(side, self.sides[side].active)[Stat.SPE]''',
     '''    def active_hp(self, side: int, position: int = 0) -> int:
        return self.sides[side].hp[self.sides[side].active[position]]

    def speed(self, side: int, position: int = 0) -> int:
        """Raw Speed. Stage multipliers and Speed-modifying effects live in
        ``pkcm.engine.effects``; this is the unmodified number."""
        return self.stats(side, self.sides[side].active[position])[Stat.SPE]'''),
])

# ``Ref`` is defined in effects.py, which imports state.py -- so state.py cannot
# import it back. Declare it here; effects.py keeps re-exporting the same alias.
patch("src/pkcm/engine/state.py", [
    ('''#: Key inside an override entry naming the fields that survive switching out.
PERMANENT = "__permanent__"''',
     '''#: Key inside an override entry naming the fields that survive switching out.
PERMANENT = "__permanent__"

#: (side, party slot). Identifies one Pokemon for the whole battle, wherever it
#: happens to be standing. ``pkcm.engine.effects`` re-exports this name.
Ref = tuple[int, int]'''),
])
print("done")
