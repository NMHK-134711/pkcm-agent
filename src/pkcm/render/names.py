"""Display names, in the renderer where they belong.

Ids stay English throughout the engine: they are the keys that tie us to
Showdown's data, the Champions override layer and the scraped item list, and
renaming them would break all three. What a person reads is a separate question,
answered here (docs/DESIGN.md §1e).

Korean covers everything Champions uses -- all 311 formes, all 500 moves, all 147
items -- with two abilities missing, both Champions originals that no public
table has a Korean name for yet. Those fall back to the English name rather than
to a raw id, so a log stays readable either way.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

NAMES_PATH = Path(__file__).resolve().parents[3] / "data" / "champions" / "names.json"

#: The language the renderer uses unless told otherwise.
DEFAULT_LANGUAGE = "ko"


@lru_cache(maxsize=1)
def _tables() -> dict[str, dict[str, str]]:
    if not NAMES_PATH.exists():
        return {}
    return json.loads(NAMES_PATH.read_text(encoding="utf-8"))


class Names:
    """Resolve ids to display names, falling back to English then to the id."""

    __slots__ = ("language", "_dex")

    def __init__(self, language: str = DEFAULT_LANGUAGE, dex=None) -> None:
        self.language = language
        self._dex = dex

    def _localized(self, kind: str, key: str | None) -> str | None:
        if key is None or self.language != "ko":
            return None
        return _tables().get(kind, {}).get(key)

    def _english(self, kind: str, key: str) -> str | None:
        if self._dex is None:
            return None
        table = {
            "species": self._dex.species,
            "moves": self._dex.moves,
            "abilities": self._dex.abilities,
            "items": self._dex.items,
        }.get(kind)
        entry = table.get(key) if table else None
        return getattr(entry, "name", None)

    def _resolve(self, kind: str, key: str | None) -> str:
        if key is None:
            return ""
        return self._localized(kind, key) or self._english(kind, key) or key

    def species(self, key: str | None) -> str:
        return self._resolve("species", key)

    def move(self, key: str | None) -> str:
        return self._resolve("moves", key)

    def ability(self, key: str | None) -> str:
        return self._resolve("abilities", key)

    def item(self, key: str | None) -> str:
        return self._resolve("items", key)

    def type(self, key: str | None) -> str:
        return self._resolve("types", key)


ENGLISH = Names("en")
