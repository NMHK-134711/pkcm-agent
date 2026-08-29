"""One party, written out the way the game's build screen asks for it.

    python scripts/team_sheet.py 1
    python scripts/team_sheet.py --best runs/tournament_800.json --record

The tournament names a winner as an index. This turns that index back into six
slot cards -- species, held item, Stat Alignment and what it moves, the Stat
Points that are not zero, four moves -- in Korean, in the order the game shows
them, so it can be typed in rather than translated first.

The ability is printed too, which the site's own slot card leaves out: it is a
choice in the build screen and there is nothing to be gained by hiding it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pkcm.data.dex import Stat, load_dex  # noqa: E402
from pkcm.engine.legality import ranker_parties  # noqa: E402
from pkcm.engine.stats import SP_TOTAL, get_nature  # noqa: E402
from pkcm.render.names import Names  # noqa: E402

#: SP order and labels as the build screen prints them.
STATS = (("hp", "체력"), ("atk", "공격"), ("def", "방어"),
         ("spa", "특공"), ("spd", "특방"), ("spe", "스피드"))


def sheet(dex, names: Names, party, index: int) -> str:
    lines = [f"파티 {index} — {party.title}"]
    if party.rate:
        lines.append(f"  레이팅 {party.rate}"
                     + (f", {party.rank}위" if party.rank else ""))
    lines.append("")
    for slot, pokemon in enumerate(party.team, 1):
        nature = get_nature(pokemon.nature)
        moved = ([] if nature.is_neutral
                 else [f"↑{_label(nature.boosted)}", f"↓{_label(nature.hindered)}"])
        spent = " ".join(f"{label}{pokemon.sp[Stat[key.upper()]]}"
                         for key, label in STATS
                         if pokemon.sp[Stat[key.upper()]])
        lines.append(f"{slot}. {names.species(pokemon.species)}")
        lines.append(f"   지닌물건  {names.item(pokemon.item) or '없음'}")
        lines.append(f"   특성      {names.ability(pokemon.ability)}")
        lines.append(f"   보정      {names.nature(pokemon.nature)}"
                     + (f"  {' '.join(moved)}" if moved else "  (무보정)"))
        lines.append(f"   스탯포인트 {spent}"
                     f"   (합 {sum(pokemon.sp)}/{SP_TOTAL})")
        lines.append("   기술      " + ", ".join(
            names.move(move) for move in pokemon.moves))
        lines.append("")
    return "\n".join(lines)


def _label(stat: Stat) -> str:
    return dict(STATS)[stat.name.lower()]


def record(payload: dict, index: int) -> str:
    """How the party did, and against whom -- the reason it is on this page."""
    standing = next((row for row in payload["standings"]
                     if row["party"] == index), None)
    if standing is None:
        return f"파티 {index}는 이 토너먼트에 참가하지 않았습니다."

    against: dict[int, list[int]] = {}
    for fixture in payload["fixtures"]:
        if index not in (fixture["a"], fixture["b"]):
            continue
        mine, theirs = ((fixture["a_wins"], fixture["b_wins"])
                        if fixture["a"] == index
                        else (fixture["b_wins"], fixture["a_wins"]))
        other = fixture["b"] if fixture["a"] == index else fixture["a"]
        tally = against.setdefault(other, [0, 0])
        tally[0] += mine
        tally[1] += theirs

    place = [row["party"] for row in payload["standings"]].index(index) + 1
    lines = [f"성적: {standing['rate']:.1%} "
             f"({standing['wins']}승 {standing['losses']}패 "
             f"{standing['draws']}무), {len(payload['standings'])}팀 중 {place}위",
             f"  {payload['iterations']} 시뮬레이션, 상대도 같은 에이전트", ""]
    ranked = sorted(against.items(),
                    key=lambda pair: -(pair[1][0] / max(sum(pair[1]), 1)))
    titles = {row["party"]: row["title"] for row in payload["standings"]}
    lines.append("  상대별 (강한 순):")
    for other, (mine, theirs) in ranked:
        total = mine + theirs
        share = mine / total if total else 0.0
        lines.append(f"    {share:>5.0%}  {mine}-{theirs}  "
                     f"파티 {other:<2} {titles.get(other, '')[:44]}")
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):  # pragma: no cover
            pass

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("party", nargs="?", type=int, default=None,
                        help="which imported party. Omit with --best")
    parser.add_argument("--best", default=None,
                        help="a tournament JSON; takes its winner")
    parser.add_argument("--record", action="store_true",
                        help="also print how it did, and against whom")
    parser.add_argument("--language", default="ko", choices=("ko", "en"))
    args = parser.parse_args()

    payload = None
    if args.best:
        payload = json.loads(Path(args.best).read_text(encoding="utf-8"))
        if not payload["standings"]:
            parser.error(f"{args.best} has no standings in it")
        if args.party is None:
            args.party = payload["standings"][0]["party"]
    if args.party is None:
        parser.error("name a party, or pass --best with a tournament JSON")

    parties = ranker_parties()
    if not 0 <= args.party < len(parties):
        parser.error(f"party {args.party} does not exist; "
                     f"there are {len(parties)}")

    dex = load_dex()
    print(sheet(dex, Names(args.language, dex), parties[args.party], args.party))
    if args.record:
        if payload is None:
            parser.error("--record needs --best to say which tournament")
        print(record(payload, args.party))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
