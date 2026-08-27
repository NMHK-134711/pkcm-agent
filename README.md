# pkcm-agent

Pokémon Champions 배틀 시뮬레이터 + 에이전트 (개인 사이드 프로젝트).

Regulation Set M-B 룰을 따르는 배틀 엔진을 직접 구현하고, 그 위에 탐색 + 학습 하이브리드
정책을 올리는 것이 목표다. 파티 구성(기술·도구·특성·Stat Alignment·SP 배분)까지 학습 대상.

- 설계 원칙과 로드맵: [docs/DESIGN.md](docs/DESIGN.md)
- 이어서 작업할 때: [docs/HANDOFF.md](docs/HANDOFF.md)

## 현재 상태

**싱글 6→3, 더블 6→4 둘 다 완성.** 챔피언스에 존재하는 기술 전부가 실행된다.

| | |
|---|---|
| 포맷 | 싱글 6→3, **더블 6→4** |
| 종족 | 235 + 메가 76 |
| 기술 | **500 / 500** |
| 특성 | 198 / 201 (남은 3종은 도구에 작용 — 되새김질·숙성·점착) |
| 도구 | **167** (op.gg 목록 + 포케챔스 도감이 채운 누락분) |
| 메커니즘 | 상태이상·랭크변화·날씨·필드·룸·벽·설치기·조이기·2턴·강제교체·카운터 계열 |
| 학습기술 | 게임 도감 기준 (전 세대 합집합 아님) |
| 환경 | PettingZoo **ParallelEnv** (싱글/더블, 행동 마스크, 정보집합 분리) |
| 에이전트 도구 | 도감 참조표 + 데미지 계산기 + 확률 (난수·명중·급소·부가효과·턴 손실) |
| 탐색 | 결정화 + 동시행동 MCTS (PUCT, IS-MCTS 단일 트리) — **greedy 상대 70.0%** [61.3, 77.5] |
| 학습 | AlphaZero 루프 (자가대전 → 정책·가치망 → 탐색). PyTorch + GPU + 15코어, wandb 로깅 |
| 더블 전용 | 타겟팅, 광범위 0.75배, 유인(따라하기·분노가루·피뢰침·저수), 동맹 특성·기술 |
| 메가진화 | 배틀당 1회, 기절해도 유지 |
| 표기 | 한국어(조사 처리 포함) / 영어 |
| 테스트 | 571 |
| 처리량 | 엔진 싱글 2,900 / 더블 680 turns/s (단일 코어), 자가대전 ~2,000 battles/h (15코어) |

미착수: 대규모 학습 실행의 **결과**, 파티 구성 학습.

**새 PC/새 세션에서 이어받는다면 → [docs/RESUME.md](docs/RESUME.md).**
환경 세팅, 옮겨야 할 파일, 다음에 돌릴 명령, 성공 판정 기준이 거기 있다.

### 측정 기록

파이프라인의 모든 숫자는 실제로 나아졌든 아니든 좋은 방향으로 움직인다. 손실은 망이
주어진 걸 맞추기 때문에 내려가고, 루트 가치는 망이 자신감을 얻기 때문에 올라간다.
**`scripts/arena.py`만 답을 안다** — 팀을 미러링하고, 양쪽 자리로 붙이고, Wilson 신뢰구간을 낸다.

| 탐색 구성 | vs greedy | |
|---|---|---|
| DUCT, 깊이 6 | 53.1% [39.4, 66.3] | 구분 안 됨 |
| DUCT + 롤아웃 | 57.1% [38.8, 74.5] | 구분 안 됨 |
| **PUCT 사전확률** | **66.4% [57.5, 74.2]** | **구분됨** |
| 사전확률만 제거 | 45.8% [37.2, 54.7] | 구분 안 됨 |
| **현재 (픽 사전확률 + min-max Q + exploration 0.1)** | **70.0% [61.3, 77.5]** | 65.0%와 구간이 겹침 |

**사전확률이 전부였다.** 4행과 3행은 같은 구성이고 그것만 다른데 20.6%p 벌어진다.
그리고 앞의 두 줄은 진전처럼 보였지만 아니었다 — 어블레이션이 더 좁은 구간으로 45.8%를
찍었으니, 사전확률 전에는 아무것도 나아지지 않았던 것이다.

### greedy 상대 승률로는 못 가른다

두 설정을 각각 greedy에 붙여 비교하면 표본이 대부분 낭비된다. exploration 0.3과 0.7은
greedy 상대로 **75.0%와 68.8%** — 6%p 차이로 보였는데, 서로 200판을 직접 붙이니
**50.3% [43.4, 57.1]**, 완전한 무승부였다. 그래서 `arena.py --exploration-b`로 두 탐색을
직접 맞붙인다.

| 직접 대결 (200판, 팀 미러링 + 양쪽 자리) | | |
|---|---|---|
| `sample_opponent` **끈 쪽** vs 켠 쪽 | **59.5% [52.6, 66.1]** | **구분됨 — 켜면 약해진다** |
| `normalize_value` 끈 쪽 vs 켠 쪽 | 54.3% [47.3, 61.0] | 구분 안 됨 |
| exploration 0.1 vs 0.7 | 53.8% [46.8, 60.6] | 구분 안 됨 |
| exploration 0.3 vs 0.7 | 51.0% [44.1, 57.8] | 구분 안 됨 |

`sample_opponent`는 **내가 측정 없이 기본값으로 넣었다가 측정하고 되돌린 것**이다.
근거로 삼은 논증은 그럴듯했고 당시 가지고 있던 측정치는 그 논증과 어긋났는데 논증 쪽을 택했다.

### 탐색이 우유부단하면 학습할 게 없다

방문분포가 균등에 가까우면 정책망의 손실은 **자기 타깃의 엔트로피**에서 멈춘다. 배울 걸
다 배운 것이고, 그 양이 0인 것이다. 첫 학습 실행이 정확히 그랬다 (손실 1.675 ≈ 타깃
엔트로피 1.678).

원인은 세 개였고 전부 탐색 쪽이었다:

1. **픽 페이즈는 선택되고 있지 않았다.** `_promise`가 `SELECT` 행동을 전부 0점으로 매겨
   정렬이 무의미했고, 상한 24개는 `permutations`가 먼저 뱉은 것들 — **전부 0번이나 1번
   포켓몬으로 시작하는 것들**이었다. 싱글 120개, 더블 360개 중에서.
2. **PUCT가 단위가 다른 두 수를 더하고 있었다.** `heuristic`은 한 마리 전멸을 -0.2로,
   체력 절반을 -0.02로 읽는다. 루트의 Q 편차는 0.037, 탐험 항은 0.20. MuZero의 min-max
   재스케일링으로 맞춘다.
3. **탐험 항이 두 개였다.** PUCT의 사전확률 항 + UCB1 항. AlphaZero에는 두 번째가 없다.

턴 4 기준 방문분포 엔트로피(균등 대비, 낮을수록 탐색이 의견을 가진 것): **94% → 78%**,
최선 행동 지분 **34% → 46%**.

지금 사전확률 자리에 있는 건 **위력 × 상성 × 자속**이라는 조잡한 곱셈이고, **정책망이
대체할 자리가 정확히 거기다.**

## 셋업

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[rl,train,dev]" json5
```

GPU를 쓰려면 CUDA 빌드로 (드라이버에 맞는 인덱스 선택):

```bash
.venv/Scripts/python.exe -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

데이터 받기 (`data/raw/`와 `data/reference/`는 gitignore, 아래로 재생성):

```bash
.venv/Scripts/python.exe scripts/fetch_showdown_data.py
```

```bash
.venv/Scripts/python.exe scripts/fetch_showdown_source.py
```

```bash
.venv/Scripts/python.exe scripts/fetch_regulation.py m_b
```

가공 (산출물은 커밋돼 있어 upstream이 바뀔 때만 필요):

```bash
.venv/Scripts/python.exe scripts/build_champions_overrides.py
```

```bash
.venv/Scripts/python.exe scripts/build_champions_data.py
```

```bash
.venv/Scripts/python.exe scripts/build_champions_items.py
```

```bash
.venv/Scripts/python.exe scripts/build_names.py
```

포챔스 도감 대조 (학습기술표의 출처):

```bash
.venv/Scripts/python.exe scripts/fetch_pokechams.py
```

```bash
.venv/Scripts/python.exe scripts/build_champions_learnsets.py
```

```bash
.venv/Scripts/python.exe scripts/compare_pokechams.py
```

## 써보기

배틀 관전 (`--lang en`으로 영어):

```bash
.venv/Scripts/python.exe scripts/selfplay_demo.py --seed 1
```

더블배틀로:

```bash
.venv/Scripts/python.exe scripts/selfplay_demo.py --format doubles --seed 3
```

PettingZoo 환경으로 (행동 마스크를 읽는 랜덤 정책):

```bash
.venv/Scripts/python.exe scripts/env_demo.py --format doubles --episodes 40 --quiet
```

계산기가 뭘 보는지 (상성·데미지 구간·확정타·선공 여부):

```bash
.venv/Scripts/python.exe scripts/env_demo.py --explain --seed 4
```

정책끼리 붙여보기 (팀을 미러링하고 신뢰구간까지 출력):

```bash
.venv/Scripts/python.exe scripts/arena.py --a greedy --b random --battles 60
```

```bash
.venv/Scripts/python.exe scripts/arena.py --a search --b greedy --battles 25
```

자가대전 데이터 생성 (전 코어):

```bash
.venv/Scripts/python.exe scripts/selfplay_gen.py --battles 200
```

학습 루프 (wandb 로깅, N iteration마다 아레나 측정):

```bash
.venv/Scripts/python.exe scripts/train_loop.py --iterations 8 --battles 250 --search-iterations 800 --out runs/first
```

wandb 대시보드를 쓰려면 **직접** 로그인할 것 (API 키는 이 저장소를 거치지 않는다):

```bash
.venv/Scripts/wandb.exe login
```

로그인 안 해도 실행은 정상 동작하고 `runs/<name>/history.json`에 기록된다.
끄고 싶으면 `--no-wandb`.

학습이 값을 했는지 확인 (**이것만이 답이다**):

```bash
.venv/Scripts/python.exe scripts/arena.py --a net --b search --battles 60 --checkpoint runs/first/net.pt
```

처리량 측정:

```bash
.venv/Scripts/python.exe scripts/selfplay_demo.py --bench 1000
```

테스트:

```bash
.venv/Scripts/python.exe -m pytest -q
```

특성 구현 현황 (형태별로 묶어서 표시, `--show <name>`으로 원본 소스 열람):

```bash
.venv/Scripts/python.exe scripts/ability_report.py
```

## 구조

```
src/pkcm/
  data/         dex 로더 — 종족·기술·특성·도구·타입표·레귤레이션
  engine/       순수 함수 배틀 코어
    effects      훅 레지스트리 (modify / notify / veto)
    mutate       원시 상태 변경, 각각이 훅을 실행
    conditions   상태이상·날씨·필드·벽·설치기
    abilities    특성 198종
    items        도구 147종
    moves        선언적 기술 실행 + 데미지 공식
    tactics      구조적 기술 (강제교체·카운터·유턴·2턴·자폭)
    moveeffects  데이터에 효과가 없는 기술 94종
    battle       턴 루프 (재개 가능, 위치당 행동)
    state        배틀 상태, legal_actions
    legality     팀 합법성 + 랜덤 팀 생성
    scope        엔진이 실행 가능한 범위 (합법성과 분리)
  envs/         PettingZoo 어댑터 — 엔진을 감싸기만 한다
    observation  정보집합 — 그 플레이어가 아는 것만 + determinize
    reference    도감 참조표 — 가중치에 외우는 대신 조회하는 시트
    analysis     데미지 계산기 — 관측만 보고, 모르는 건 구간으로
  search/       탐색 — 엔진에 직접 올라탄다 (환경을 안 거친다)
    mcts         결정화 + 동시행동 MCTS
    policy       Policy 프로토콜 — random / greedy / search, PUCT 사전확률
    evaluate     리프 평가 — 물량 휴리스틱 또는 롤아웃
  train/        AlphaZero 루프
    samples      자가대전 1판 → (관측, 방문분포, 결과)
    parallel     멀티프로세싱 (Windows spawn)
    net          정책·가치망 — 도감은 학습이 아니라 입력
    trainer      손실·옵티마이저·체크포인트
    evaluator    망을 탐색의 사전확률·리프값 자리에 끼움
    logging      history.json + wandb (둘 다 선택 아님/선택)
    interval     Wilson 신뢰구간 — Wald는 극단에서 거짓말한다
    encoding     관측 → 배열, 행동 ↔ 정수 인덱스
    champions    ParallelEnv 본체
  render/       이벤트 로그 소비자 — 텍스트 뷰어(한/영), 표시명
  testing/      시나리오 러너 — 실제 게임 관찰을 테스트로
tests/scenarios/   게임에서 본 동작 (형식은 그 안의 README)
scripts/           데이터 수집·가공, 커버리지 리포트
data/raw/          Showdown 클라이언트 데이터 (gitignored)
data/reference/    Showdown TypeScript 소스 = 메커니즘 명세 (gitignored)
data/champions/    레귤레이션·override·도구 목록·표시명 (커밋)
```

## 데이터 출처

- 수치·구현 참조: [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (MIT).
  `data/mods/champions/`가 챔피언스 구현 그 자체라 메커니즘의 명세서 역할을 한다.
- 출전 목록: [Bulbapedia — Regulation Set M-B](https://bulbapedia.bulbagarden.net/wiki/Regulation_Set_M-B)
- 도구 목록: op.gg 챔피언스 도구 화면 스크랩 (`포챔스 아이템 목록.txt`)
- **학습기술표**: [포케챔스](https://pokemon.yodams.com) 도감 (Flutter 앱, 정적 JSON 에셋).
  게임 도감에서 나온 종족별 습득 기술 목록 — Showdown의 전 세대 합집합보다 정확하다.
  `compare_pokechams.py`가 두 출처의 차이를 출력한다.
- 한국어 표기: [PokéAPI](https://github.com/PokeAPI/pokeapi) 다국어 CSV + 위 스크랩

최종 권위는 **게임 본편**이다. 관찰한 것은 `tests/scenarios/`에 적으면 영구 테스트가 된다.

## 이 프로젝트에서 되풀이해서 옳았던 것

**등록만 하고 아무도 안 읽는 효과가 최다 빈출 버그다.** 봉인, 시끄러운소리, 와이드가드,
매직룸, 원더룸, 중력, 조가비갑옷, 전투무장, 흉내내기, 잠꼬대, 코골기 — 전부 등록돼 있고
이름이 있고 로그에도 나오는데 아무도 안 읽었다. **실패하지 않아서 안 보인다.**
지금은 세 개의 가드가 지킨다 (`docs/HANDOFF.md` 참고).

**계산기는 관측만 본다.** 진실을 볼 수 있는 도구는 그걸 조용히 정책에 세탁해 넣고,
정책은 독심술을 배운 것처럼 보인다. 테스트가 상대 SP를 바꾸고 추정치가 안 움직이는지 본다.

**같은 계산의 두 벌은 반드시 어긋난다.** 데미지는 `moves.damage_formula`를 엔진과 계산기가
공유하고, 스피드는 공유가 불가능해서 테스트가 32조합에서 두 값을 대조한다.

**점추정만 보고 진전을 읽지 말 것.** 이 프로젝트는 세 번 속았다. 세 번째가 가장 깨끗한
예다 — greedy 상대 75.0% 대 68.8%가 직접 붙이면 50.3%였다.

**측정 없이 기본값을 바꾸지 말 것.** `sample_opponent`는 논증만 보고 들어갔다가 200판
직접 대결에서 59.5%로 졌다. 그때도 측정치는 있었고 논증과 어긋났는데 논증을 택했다.

**"등록만 하고 아무도 안 읽는" 것의 탐색판이 있다.** `_promise`는 `SELECT` 행동을 조용히
0점으로 매겼다. 실패하지 않고, 로그에도 안 남고, 게임에서 가장 큰 결정 하나가 사실상
무작위로 내려지고 있었다.
