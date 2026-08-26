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
| M0 배틀 엔진 골격 | 미착수 |
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
.venv/Scripts/python.exe scripts/fetch_regulation.py m_b
```

```bash
.venv/Scripts/python.exe scripts/build_champions_data.py
```

테스트:

```bash
.venv/Scripts/python.exe -m pytest -q
```

## 구조

```
src/pkcm/
  data/      dex 로더 — 종족/기술/특성/도구/타입표/레귤레이션
  engine/    순수 함수 배틀 코어 — stats.py(SP 체계) 외 나머지 예정
  envs/      PettingZoo ParallelEnv 어댑터 (예정)
scripts/     데이터 수집·가공
data/raw/         Showdown 클라이언트 데이터 (gitignored)
data/champions/   레귤레이션 합법 목록 + 룰 상수 (커밋)
```

## 데이터 출처

- 종족값·기술·특성·도구·타입표: [Pokémon Showdown](https://play.pokemonshowdown.com/data/) (MIT)
- 레귤레이션 합법 목록: [Bulbapedia — Regulation Set M-B](https://bulbapedia.bulbagarden.net/wiki/Regulation_Set_M-B)
