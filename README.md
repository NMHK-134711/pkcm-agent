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
| 더블 전용 | 타겟팅, 광범위 0.75배, 유인(따라하기·분노가루·피뢰침·저수), 동맹 특성·기술 |
| 메가진화 | 배틀당 1회, 기절해도 유지 |
| 표기 | 한국어(조사 처리 포함) / 영어 |
| 테스트 | 514 |
| 처리량 | 싱글 2,500 / 더블 680 turns/s (단일 코어) |

미착수: 학습 루프, 파티 구성 학습, 탐색 정책.

## 셋업

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[rl,dev]" json5
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
