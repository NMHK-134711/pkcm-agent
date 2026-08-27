"""The pkmnchamps exporter, against the sample that defined its format.

``party_samples/메가아쿠스타_비파티.txt`` was made by hand, by copying a party
page's slot cards out of a browser. It is the specification for what
``scripts/export_party_samples.py`` emits, so the test is a byte comparison
against it -- CRLF, the "Mega" that is really the badge's alt text, and no final
newline included.

Skipped without the cache, which is gitignored:
    python scripts/fetch_pkmnchamps.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "pkmnchamps"
SAMPLE = ROOT / "party_samples" / "메가아쿠스타_비파티.txt"

#: The party the sample was copied from -- ピッツ's rain team, single, R2776.
SAMPLE_PARTY_ID = "09e6c48f-eed2-4814-973a-81c487229482"


def _load_exporter():
    """Import the script by path; ``scripts/`` is not a package."""
    path = ROOT / "scripts" / "export_party_samples.py"
    spec = importlib.util.spec_from_file_location("export_party_samples", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cache():
    parties = RAW_DIR / "archive_parties.json"
    names = RAW_DIR / "names_ko.json"
    if not parties.exists() or not names.exists():
        pytest.skip("no pkmnchamps cache -- run scripts/fetch_pkmnchamps.py")
    return (json.loads(parties.read_text(encoding="utf-8")),
            json.loads(names.read_text(encoding="utf-8")))


def test_sample_party_renders_byte_for_byte(cache):
    if not SAMPLE.exists():
        # The specification is a file someone made by hand, and it is not in
        # the repository. A test whose fixture is absent has nothing to say;
        # failing here would put a permanent red mark next to every run and
        # hide the next real failure behind it.
        pytest.skip(f"{SAMPLE.name} is not here -- copy it back to compare against")
    parties, tables = cache
    export = _load_exporter()
    party = next(p for p in parties if p["id"] == SAMPLE_PARTY_ID)

    rendered = export.render_party(party, export.Names(tables),
                                   with_ability=False, strict=False)

    assert rendered == SAMPLE.read_text(encoding="utf-8", newline="")


def test_every_name_resolves(cache):
    """No party falls back to an English slug.

    The fallback is deliberate -- a visible ``choice-scarf`` beats a silently
    wrong 구애스카프 -- but it should never fire on the archive as it stands. If
    it does, the site added a species or renamed a slug and the tables are stale.
    """
    parties, tables = cache
    export = _load_exporter()
    names = export.Names(tables)

    for party in parties:
        export.render_party(party, names, with_ability=True, strict=False)

    assert not names.unresolved, dict(names.unresolved)
