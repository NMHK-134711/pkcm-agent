# 파티 토너먼트 — 이어받기

이 브랜치가 들고 있는 것은 두 가지다. **팀끼리 붙이는 측정 도구**와, 그 측정이
찾아낸 **프라이어 버그의 수정**. 왜 그런지는 커밋 메시지 두 개에 다 적혀 있고,
이 문서는 **다음에 뭘 돌리면 되는지**만 적는다.

## 지금 상태

- `runs/tournament_800.json` — 20팀 라운드로빈, 3800게임. **버그 수정 *전*에 잰
  것이라 아래쪽 순위는 못 믿는다.** 메가아쿠스타 팀 둘(파티 1, 6)이 에이스를
  벤치에 둔 채로 전 경기를 3대6으로 싸웠고, 그래서 19·20위다.
- 우승은 파티 14 (65.6% [60.7, 70.2]). 핸디캡 받은 팀 넷을 빼고 다시 세도
  65.2% [59.7, 70.4]로 그대로라, **우승 자체는 버그의 산물이 아니다.**
- 테스트 644개 통과.

## 돌릴 것, 순서대로

**1. 수정이 에이전트를 실제로 강하게 했는가.** 순위표와 별개로 이게 성과
지표다. 수정 전 탐색과 수정 후 탐색을 직접 붙이는 게 정석이지만 둘이 한 코드에
공존하지 않으므로, 가장 가까운 대용은 프리뷰가 걸리는 팀 분포에서 재는 것이다:

```bash
.venv/Scripts/python.exe -u scripts/judge.py --matches 200 --teams ranker
```

`--teams ranker`는 슬롯을 재조합하므로 메가 스톤 보유자가 섞여 들어온다.
수정 전 같은 명령의 기록이 있으면 그것과 비교하고, 없으면 이 커밋을 되돌린
워킹트리에서 한 번 재서 짝을 만든다.

**2. 라운드로빈 재실행.** 필드가 온전해졌으니 순위를 다시 뽑는다. 20코어에서
45분:

```bash
.venv/Scripts/python.exe -u scripts/tournament.py --repeats 10 --matrix --out runs/tournament_800_fixed.json
```

`runs/tournament_800.json`을 지우지 말 것 — **"순위가 바뀌었는가"가 다음
질문이고, 그건 옛 표가 있어야 답할 수 있다.**

**3. 상위권만 깊게.** 실제로 플레이하는 강도에서 확정한다:

```bash
.venv/Scripts/python.exe -u scripts/tournament.py --entrants <상위 5개> --repeats 40 --search-iterations 3200 --out runs/tournament_3200.json
```

## 알아둘 것

**`--repeats`는 대진당 배틀 수가 아니라 반복 수다.** 한 반복이 자리를 바꿔
두 판이므로, 팀당 게임 수는 `2 × repeats × (참가팀 - 1)`이다. 20팀 10반복이
팀당 380게임, ±5pp.

**반복 r은 모든 대진에서 같은 시드를 쓴다** (common random numbers). 일부러
그렇게 했다 — 팀을 비교하는 게 목적이라 주사위는 공유하는 쪽이 분산이 작다.

**필드가 곧 결과다.** 20팀 라운드로빈은 그 19팀에 대한 승률이지 래더에 대한
승률이 아니다. 아카이브 23개 중 3개는 임포트에서 걸러졌다
(`scripts/import_parties.py`가 이유를 출력한다).

**`fought_as`는 우리 쪽에만 적용된다.** 스톤은 도구이고 도구는 프리뷰에서
안 보인다. 상대 쪽에 같은 해석을 붙이면 그건 정보 누출이다.

우승 파티를 보려면:

```bash
.venv/Scripts/python.exe scripts/team_sheet.py --best runs/tournament_800.json --record
```

직접 둬보려면:

```bash
.venv/Scripts/python.exe scripts/play_web.py --your-party 14
```
