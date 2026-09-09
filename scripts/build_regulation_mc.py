"""Build regulation M-C from M-B plus what the update added.

``fetch_regulation.py m_c`` is the way this should be done, and it will be:
Bulbapedia's Regulation Set pages are a Champions source and the one this
project already trusts for the roster. There is no M-C page yet -- the update
is hours old and the fetch comes back 404 -- so this stands in.

It is not a guess. The 2026-09-09 notes list thirty-two species added and none
removed, and ``mc_table.json``, read out of the game's own v1.2.0 data and
checked by hk, has exactly those thirty-two with their ids. So M-C is M-B plus
those, and the arithmetic is checked rather than asserted: twenty-six ordinary
species and six Megas, against a note that says twenty-six and six.

Replace this the moment the Bulbapedia page exists.

Usage:
    python scripts/build_regulation_mc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:                       # pragma: no cover
        pass

from pkcm.data.dex import load_dex                       # noqa: E402

OUT = ROOT / "data" / "champions"
EXPECTED_BASE, EXPECTED_MEGA = 26, 6


#: The eighteen the update adds. Six Mega Stones and twelve held items, from
#: the 2026-09-09 notes hk collected.
NEW_ITEMS = (
    "absolitez", "garchompitez", "lucarionitez", "salamencite", "golisopite",
    "baxcalibrite",
    "leek", "rockyhelmet", "airballoon", "redcard", "bindingband",
    "ejectbutton", "normalgem", "terrainextender", "electricseed",
    "psychicseed", "mistyseed", "grassyseed",
)


def open_the_items() -> None:
    """Let the eighteen new items exist.

    Move existence is answered by the ROM's own table; item existence is not,
    because ``champout/item.json`` keys everything by a numeric message label
    -- ITEMNAME_214 -- and there is no name file to resolve them with. So it
    still falls through to Showdown's ``isNonstandard``, filtered by
    overrides.json, which comes from Showdown's Champions mod and predates the
    patch. Twelve of the eighteen are marked Past there and six are not
    mentioned at all, so every one of them, Salamencite included, reads as not
    existing -- and a Mega Salamence cannot be built.

    Writing them into the overrides is the same kind of statement the file
    already makes: a declared difference between the base data and Champions.
    Replace it the day the ROM's item table can be read by name.
    """
    path = OUT / "overrides.json"
    overrides = json.loads(path.read_text(encoding="utf-8"))
    changes = overrides["items"]["changes"]
    for item_id in NEW_ITEMS:
        entry = changes.setdefault(item_id, {})
        entry["isNonstandard"] = None
    overrides.setdefault("added_by", []).append(
        "2026-09-09 M-C: the eighteen items above, from the update notes, "
        "because the item table cannot yet be read out of the ROM.")
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"  items opened: {len(NEW_ITEMS)}")


def main() -> int:
    dex = load_dex()
    base = json.loads((OUT / "regulation_m_b.json").read_text(encoding="utf-8"))
    table = json.loads((OUT / "mc_table.json").read_text(encoding="utf-8"))

    known = {one for entry in base["eligible"] + base["megas"]
             for one in entry["showdown_ids"]}
    added_base, added_mega = [], []
    for entry in table["species"]:
        if entry["id"] in known:
            continue
        species = dex.species[entry["id"]]
        row = {"dex_num": species.raw.get("num"),
               "species": species.name,
               "form": "-Mega" if "stone_ko" in entry else "",
               "showdown_ids": [entry["id"]],
               "missing": []}
        (added_mega if "stone_ko" in entry else added_base).append(row)

    if len(added_base) != EXPECTED_BASE or len(added_mega) != EXPECTED_MEGA:
        raise SystemExit(
            f"the notes say {EXPECTED_BASE} species and {EXPECTED_MEGA} Megas; "
            f"the table gives {len(added_base)} and {len(added_mega)}")

    payload = {
        "regulation": "M-C",
        "source": ("built from data/champions/mc_table.json, which is the "
                   "game's own v1.2.0 data as hk checked it, because "
                   "Bulbapedia has no Regulation Set M-C page yet. Replace "
                   "with fetch_regulation.py m_c when it does."),
        "eligible": base["eligible"] + added_base,
        "megas": base["megas"] + added_mega,
    }
    # The rule constants come across unchanged, because the update's notes
    # say nothing about them: team size, the level rule, the clauses and the
    # SP system are all as M-B had them. Recorded as inherited rather than
    # confirmed, so that a later reading of the game can contradict it.
    rules = json.loads((OUT / "ruleset_m_b.json").read_text(encoding="utf-8"))
    rules["regulation"] = "M-C"
    rules["active_from"] = "2026-09-09"
    rules["active_until"] = None
    rules["inherited_from"] = ("M-B. The 2026-09-09 notes list species, items, "
                               "abilities and moves, and no change to any rule "
                               "constant -- so these are carried over rather "
                               "than read from a source about M-C.")
    (OUT / "ruleset_m_c.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    open_the_items()

    out = OUT / "regulation_m_c.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8")

    print(f"M-B: {len(base['eligible'])} species, {len(base['megas'])} Megas")
    print(f"M-C: {len(payload['eligible'])} species, {len(payload['megas'])} Megas")
    print(f"  added: {', '.join(one['showdown_ids'][0] for one in added_base)}")
    print(f"  Megas: {', '.join(one['showdown_ids'][0] for one in added_mega)}")
    print(f"wrote {out} and ruleset_m_c.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
