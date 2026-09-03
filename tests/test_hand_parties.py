"""The hand-transcribed parties the engine reads are the text hk corrected.

``party_samples/pkmnchamps/single/NN.txt`` is the source hk edits against the
real game; ``data/champions/parties_hand.json`` is what every team draw
reads. Twice on 2026-09-01/03 the text was fixed (Primarina's Hyper Voice was
really Sparkling Aria, on parties 43 and 39) and the JSON was not
regenerated, so a day of runs -- the v2 round robin, six curriculum runs,
every judgment -- played the stale set. This regenerates the JSON from the
text through the checker's own parser and diffs the two, so the drift shows
up in the suite rather than in a week of results.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "party_samples" / "pkmnchamps" / "single"
JSON_PATH = ROOT / "data" / "champions" / "parties_hand.json"


def _load_checker():
    path = ROOT / "scripts" / "check_party_text.py"
    spec = importlib.util.spec_from_file_location("check_party_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hand_party_json_matches_its_text():
    if not TEXT_DIR.exists() or not JSON_PATH.exists():
        pytest.skip("hand parties are not in this checkout")
    checker = _load_checker()
    regenerated, _, _ = checker.build_parties(TEXT_DIR)
    stored = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    by_id = {p["id"]: p for p in stored}
    assert [p["id"] for p in regenerated] == [p["id"] for p in stored], \
        "the set of hand parties differs between text and JSON"

    drift = []
    for fresh in regenerated:
        old = by_id[fresh["id"]]
        for slot, (a, b) in enumerate(zip(fresh["team"], old["team"])):
            for field in ("species", "ability", "item", "nature", "moves", "sp"):
                if a[field] != b[field]:
                    drift.append(f"party {fresh['id']} slot {slot} {a['species']}: "
                                 f"{field} text={a[field]!r} json={b[field]!r}")
    assert not drift, "regenerate with: python scripts/check_party_text.py " \
        "party_samples/pkmnchamps/single --json data/champions/parties_hand.json\n" \
        + "\n".join(drift)
