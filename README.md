# pkcm-agent

Pokémon Champions 배틀 시뮬레이터 + 에이전트 (개인 사이드 프로젝트).

Regulation Set M-B 룰을 따르는 배틀 엔진을 직접 구현하고, 그 위에 탐색 + 학습
하이브리드 정책을 올리는 것이 목표다. 파티 구성(기술·도구·특성·Stat Alignment·SP 배분)
까지 학습 대상이다. 설계 원칙과 로드맵은 [docs/DESIGN.md](docs/DESIGN.md).

## 현재 상태

| 단계 | 상태 |
|---|---|
| 데이터 레이어 | 완료 — 235종 + 메가 76종 전부 해석됨 |
| SP 능력치 체계 | 완료 — SP 66/32, Stat Alignment 21종, 레벨 50 공식 |
| 팀 합법성 + 랜덤 팀 생성 | 완료 — 원종 단위 종족 클로즈, 아이템 클로즈, SP 예산 |
| 챔피언스 데이터 정합 | 완료 — Showdown champions 모드 기준 override 레이어 |
| 배틀 엔진 | 챔피언스 기술 500종 중 **353종(70.6%)** 실행, 6,000 turns/s |
| 메커니즘 | 상태이상·랭크변화·날씨·필드·벽·설치기·방어·대타출동 |
| 특성 | 로스터 201종 중 **198종**, 출전 슬롯의 99% (남은 3종은 도구 대기) |
| 시나리오 검증 하네스 | 완료 — 실제 게임 관찰을 테스트로, 미구현 메커니즘은 skip |
| 도구 | 챔피언스 배틀 도구 **72종 전부** (op.gg 실측 목록 기준) |
| 메가진화 | 완료 — 배틀당 1회, 스톤 75종, 기절해도 유지 |
| 한국어 표기 | 완료 — 폼 311·기술 500·도구 147, 조사 처리 포함 |
| PettingZoo 어댑터 | 미착수 |
| 하이브리드 배틀 정책 | 미착수 |
| 파티 구성 학습 | 미착수 |

## 셋업

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[rl,dev]" json5
```

데이터 받기 (`data/raw/`는 gitignore, 아래로 재생성):

```bash
.venv/Scripts/python.exe scripts/fetch_showdown_data.py
```

```bash
.venv/Scripts/python.exe scripts/fetch_showdown_source.py
```

```bash
.venv/Scripts/python.exe scripts/build_champions_overrides.py
```

```bash
.venv/Scripts/python.exe scripts/build_champions_items.py
```

```bash
.venv/Scripts/python.exe scripts/build_names.py
```

```bash
.venv/Scripts/python.exe scripts/fetch_regulation.py m_b
```

```bash
.venv/Scripts/python.exe scripts/build_champions_data.py
```

테스트:

```bash
.venv/Scripts/python.exe -m pytest -q
```

배틀 관전 / 처리량 측정:

```bash
.venv/Scripts/python.exe scripts/selfplay_demo.py --seed 3
```

```bash
.venv/Scripts/python.exe scripts/selfplay_demo.py --bench 3000
```

## 구조

```
src/pkcm/
  data/      dex 로더 — 종족/기술/특성/도구/타입표/레귤레이션
  engine/    순수 함수 배틀 코어
             effects(훅) / conditions(상태·날씨·벽) / moves(기술 실행)
             mutate(원시 변경) / state / battle / legality / scope / rng / stats
  envs/      PettingZoo ParallelEnv 어댑터 (예정)
  render/    이벤트 로그 소비자 — 텍스트 뷰어(한/영), 표시명 테이블
  testing/   시나리오 러너 — 실제 게임 관찰을 테스트로
tests/scenarios/  게임에서 본 동작 (형식은 그 안의 README)
scripts/     데이터 수집·가공
data/raw/         Showdown 클라이언트 데이터 (gitignored)
data/champions/   레귤레이션 합법 목록 + 룰 상수 (커밋)
```

## 데이터 출처

- 종족값·기술·특성·도구·타입표: [Pokémon Showdown](https://play.pokemonshowdown.com/data/) (MIT)
- 레귤레이션 합법 목록: [Bulbapedia — Regulation Set M-B](https://bulbapedia.bulbagarden.net/wiki/Regulation_Set_M-B)
