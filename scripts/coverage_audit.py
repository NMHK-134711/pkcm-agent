"""Which moves does a probe actually stand behind, and which does only my
judgement stand behind?

``clause_check`` reports 0 unclaimed clauses, and that is a weaker sentence
than it sounds: a clause is "claimed" when some family says it covers it, and
nine of those families cover a clause by re-running ``mechanic_check``'s check
for the same move. Re-running a real measurement is fine when the measurement
is of the thing the sentence says. Dragon Cheer is what happens when it is
not -- "raises the target's chance for a critical hit by 1 stage, or by 2
stages if the target is Dragon type" was claimed by the family that matches
"Raises the " and checks the move's declared ``boosts``, which has nothing to
say about critical hits and said it confidently.

So this walks every move in the format, drops the ones whose description is
only "No additional effect." (or is empty), drops the sentences that name
mechanics this format does not have, and sorts what is left by what stands
behind it:

  A  the delegated check only asserts the move fails in singles
  B  a hand-written check for that move exists, but not for that sentence
  C  a generated check confirms the move's own data field appears

``python scripts/coverage_audit.py`` rewrites docs/UNVERIFIED_MOVES.md.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "clause_check", ROOT / "scripts" / "clause_check.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)
mc = cc.mc

#: Families that check a clause by re-running the move's ``mechanic_check``
#: entry rather than by putting the sentence in a position.
DELEGATING = {"secondary", "declaredboosts", "screenbreakers", "plainstatus",
              "statedduration", "counterdoubles", "delegatedpower",
              "counterhalfagain", "restated"}

#: Ally-only moves whose entry says only "has no ally to use it on in
#: singles". True, and it measures nothing about what the move does.
ALLY_ONLY = {"afteryou", "allyswitch", "aromaticmist", "coaching", "quash"}

FAMILY_KO = {
    "secondary": "추가효과 확률 (데이터의 secondary 필드)",
    "declaredboosts": "선언된 랭크 변화 (boosts 필드)",
    "plainstatus": "선언된 상태이상 (status 필드)",
    "statedduration": "지속 턴 수",
    "screenbreakers": "벽 파괴",
    "delegatedpower": "위력 계산",
    "counterdoubles": "조건부 2배",
    "counterhalfagain": "조건부 1.5배",
    "restated": "설명이 기술 그 자체 (mechanic_check의 수기 검사에 통째로 위임)",
}

TIERS = {
    "ally-only stub": ("A", "위임된 검사가 '싱글에서 아군이 없어 실패한다'만 "
                            "확인합니다. 기술이 무엇을 하는지는 아무도 재지 "
                            "않았습니다 -- 드래곤옐이 여기 있었습니다."),
    "hand-written": ("B", "그 기술 전용 수기 검사가 있습니다. 다만 그 검사가 "
                          "아래 문장을 덮는다는 것은 확인된 적이 없습니다."),
    "generated": ("C", "기술 데이터의 필드(secondary/boosts/status)가 실제 "
                       "배틀에 나타나는지 자동 검사합니다. 문장이 필드보다 더 "
                       "말하는 부분은 측정되지 않습니다."),
}


def korean(names, move_id, fallback):
    entry = names.get(move_id)
    if isinstance(entry, dict):
        return entry.get("ko") or entry.get("name") or fallback
    return entry or fallback


def survey():
    """Every move, with the sentences nothing measured directly."""
    names = json.loads((ROOT / "data" / "champions" / "names.json")
                       .read_text(encoding="utf-8"))["moves"]
    source = (ROOT / "scripts" / "mechanic_check.py").read_text(encoding="utf-8")
    handwritten = (set(re.findall(r'@check\("([a-z0-9]+)"', source))
                   | set(re.findall(r'CHECKS\["([a-z0-9]+)"\] = \(', source)))

    by_clause = defaultdict(list)
    for name, entry in cc.FAMILIES.items():
        for clause in entry["clauses"]:
            by_clause[clause].append(name)

    rows, with_effect = [], 0
    for move in cc.CHAMPIONS:
        carries = []
        for sentence in cc.sentences(move):
            families = by_clause.get(sentence, [])
            if "noeffect" in families:
                continue                  # "It hits, and nothing else happens"
            # Asked of every sentence and not only the unclaimed ones: a
            # clause about Utility Umbrella is out of reach here whether or
            # not some family put its hand up for it.
            named, missing = cc.out_of_format(sentence)
            if named and len(missing) == len(named):
                continue                  # Terastal, Blue Orb, Sky Drop, ...
            carries.append((sentence, families))
        if not carries:
            continue                      # damage and nothing else
        with_effect += 1

        loose = [(one, f) for one, f in carries if not (set(f) - DELEGATING)]
        if not loose:
            continue
        backing = ("ally-only stub" if move.id in ALLY_ONLY
                   else "hand-written" if move.id in handwritten
                   else "generated" if move.id in mc.CHECKS else "none")
        rows.append({"id": move.id, "en": move.name,
                     "ko": korean(names, move.id, move.name),
                     "category": move.raw.get("category"), "backing": backing,
                     "families": sorted({f for _, ff in loose for f in ff}),
                     "sentences": [one for one, _ in loose]})
    return with_effect, rows


def pretty(families):
    return ", ".join(FAMILY_KO.get(one, one) for one in families)


def report(with_effect, rows) -> str:
    out = []
    add = out.append
    direct = with_effect - len(rows)
    add("# 효과가 있는데, 그 문장을 직접 재보지는 않은 기술")
    add("")
    add(f"`python scripts/coverage_audit.py`가 만듭니다. 포맷의 기술 "
        f"{len(cc.CHAMPIONS)}개 중 설명이 없거나 \"No additional effect.\"뿐인 "
        f"것 {len(cc.CHAMPIONS) - with_effect}개를 빼면 **{with_effect}개**가 "
        f"효과 문장을 갖고, 그 중 **{direct}개**는 모든 효과 문장마다 그 문장을 "
        f"읽고 쓴 probe가 붙어 있습니다.")
    add("")
    if not rows:
        add("**남은 것은 없습니다.** 효과 문장을 가진 기술 전부가, 문장마다 그 "
            "문장을 읽고 쓴 probe를 갖고 있습니다.")
        add("")
        add("드래곤옐이 있던 자리가 여기였습니다: 문장은 급소율을 말하는데 위임된 "
            "검사는 능력치 랭크를 재고 있었고, 잴 것이 없으니 조용히 "
            "통과시켰습니다. 그런 자리가 이제 없다는 뜻입니다 -- probe가 "
            "*맞다*는 뜻은 아닙니다. 잘못된 이유로 통과하는 probe는 여전히 "
            "보이지 않습니다.")
        add("")
        add("(테라스탈·블루오브·하늘가르기·유틸리티우산처럼 이 포맷에 없는 "
            "기전만 조건으로 삼는 문장은 도달 불가로 기록되어 있고, 여기서는 "
            "세지 않았습니다.)")
        add("")
        return "\n".join(out) + "\n"
    add(f"아래 **{len(rows)}개**가 나머지입니다. `clause_check`가 그 문장을 "
        f"다른 검사에 *위임*했고, \"그 검사가 이 문장을 덮는다\"는 건 제 "
        f"판단이지 측정이 아닙니다. 전부 초록이지만, 초록의 근거가 문장이 "
        f"아니라 제 짐작입니다.")
    add("")
    add("드래곤옐이 정확히 이 자리에 있었습니다: 문장은 급소율을 말하는데 위임된 "
        "검사는 능력치 랭크를 재고 있었고, 잴 것이 없으니 조용히 통과시켰습니다.")
    add("")
    add("(테라스탈·블루오브·하늘가르기처럼 이 포맷에 없는 기전만 언급하는 문장은 "
        "도달 불가로 기록되어 있고, 여기서는 세지 않았습니다.)")
    add("")

    grouped = defaultdict(list)
    for row in rows:
        grouped[TIERS[row["backing"]][0]].append(row)

    for tier in ("A", "B", "C"):
        block = grouped[tier]
        if not block:
            continue
        note = next(v[1] for v in TIERS.values() if v[0] == tier)
        add(f"## {tier}등급 -- {len(block)}개")
        add("")
        add(note)
        add("")
        buckets = defaultdict(list)
        for row in block:
            buckets[tuple(row["families"])].append(row)
        for families, items in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            if len(buckets) > 1:
                add(f"### {pretty(families)} -- {len(items)}개")
                add("")
            for row in sorted(items, key=lambda one: one["id"]):
                add(f"- **{row['ko']}** ({row['en']}, `{row['id']}`, "
                    f"{row['category']})")
                for sentence in row["sentences"][:3]:
                    add(f"  - {sentence}")
            add("")
    return "\n".join(out) + "\n"


def main() -> int:
    with_effect, rows = survey()
    (ROOT / "docs" / "UNVERIFIED_MOVES.md").write_text(
        report(with_effect, rows), encoding="utf-8", newline="\n")
    counts = Counter(TIERS[one["backing"]][0] for one in rows)
    print(f"{with_effect} moves carry an effect sentence; "
          f"{with_effect - len(rows)} are checked sentence by sentence")
    for tier in sorted(counts):
        print(f"  {tier}  {counts[tier]:3d}")
    print("docs/UNVERIFIED_MOVES.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
