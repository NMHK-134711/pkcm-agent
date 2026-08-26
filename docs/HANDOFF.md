# 인계 문서 — 2026-08-26

새 세션이 여기부터 이어받으면 된다. 설계 원칙은 [DESIGN.md](DESIGN.md), 현재 상태 요약은
[README](../README.md). 이 문서는 **다음에 뭘 할지**와 **밟기 쉬운 함정**을 적어둔 것이다.

---

## 지금까지 무엇이 끝났나

**싱글 6→3 배틀 시뮬레이터가 완성됐다.** 챔피언스에 존재하는 기술 500종 전부, 출전 로스터
특성 201종 중 198종, 도구 147종 전부가 실행된다. 테스트 355개, 코드 13.5k줄, 커밋 15개.

| | |
|---|---|
| 종족 | 235 + 메가 76 |
| 기술 | **500 / 500** |
| 특성 | 198 / 201 (남은 3종은 도구에 작용 — 되새김질·숙성·점착) |
| 도구 | **147 / 147** (배틀 도구 72 + 메가스톤 75) |
| 처리량 | 3,300 turns/s (단일 코어) |

메가진화(배틀당 1회), 팀 프리뷰, TOD 4단계 판정, 한국어 표기까지 들어있다.

---

## 다음 갈림길

### A. 더블배틀 6→4
공식 랭크의 주력 포맷이고, "완벽한 레플리카"라면 여기가 남았다. 필요한 것:

- 슬롯이 2개가 되는 상태 변경 (`SideState.active`가 단일 int → 쌍)
- 타겟팅 (`move.target`이 실제로 의미를 갖게 됨: `allAdjacentFoes`, `adjacentAlly`, ...)
- 광범위 기술 데미지 0.75배 (`spreadHit`)
- 유인 계열 — 지금 `ALLY_ONLY`로 실패 처리한 12종(도우미·날따름·성원 등)이 여기서 살아난다
- 동료 대상 특성 (플라워기프트, 프렌드가드, 플러스/마이너스 등 `SINGLES_INERT` 목록)

`moveeffects.ALLY_ONLY`와 `abilities.SINGLES_INERT`가 그 작업 목록 역할을 한다.

### B. PettingZoo 어댑터 + 학습 루프
원래 목표(파티 세팅까지 학습)로 가는 길. 엔진은 준비돼 있다.

- `pkcm/envs/`는 **비어 있다.** DESIGN §1f대로 ParallelEnv 어댑터를 쓰면 된다
- 행동 마스크는 `state.legal_actions(state, player)`에서 그대로 나온다 — 엔진과 환경이
  같은 소스를 보므로 어긋날 수 없다
- 관측 인코딩은 아직 없다. **DESIGN §1c(진실/정보집합 분리)가 여기서 처음 쓰인다** —
  `Observation(state, player)`와 `determinize`가 아직 미구현이고, 하이브리드 탐색을
  붙이려면 이게 먼저다

### C. 파티 구성 학습 (DESIGN §6)
B가 선행돼야 한다. 배틀 정책이 없으면 팀의 좋고 나쁨을 평가할 수가 없다.

---

## 밟기 쉬운 함정

**id는 절대 번역하지 않는다.** Showdown 데이터·챔피언스 override·op.gg 도구 목록을 잇는
키다. 이벤트도 id를 싣고, 사람이 읽는 이름은 `pkcm/render/names.py`가 만든다.

**메커니즘을 기억으로 재구성하지 말 것.** `data/reference/*.ts`를 읽어서 옮긴다. 기억으로
쓴 특성 초안은 "등록만 하고 아무것도 안 하는 핸들러"를 여럿 포함하고 있었고 통째로 버렸다.
`scripts/ability_report.py --show <name>`이 소스를 바로 보여준다.

**"합법인가"와 "구현했는가"는 다른 질문이다** (DESIGN §1g). `legality.py`는 닌텐도가,
`scope.py`는 우리가 답한다. 섞으면 마일스톤이 올라갈 때 팀 합법성이 조용히 바뀐다.

**volatile을 붙이기만 하고 읽는 쪽을 안 만들면 그 기술은 조용히 아무것도 안 한다.**
이 프로젝트에서 가장 자주 나온 버그다. `move_support`가 값까지 검사하는 이유이고,
`test_abilities.py`에 핸들러 없는 등록은 전부 이유를 대야 통과하는 테스트가 있다.

**오버라이드에는 수명이 있다.** 대부분 교체 시 되돌아가고(변환자재·변신·트레이스·배틀스위치),
메가진화와 깨진 탈만 `__permanent__`로 남는다.

**상태기술은 타입 면역을 기본적으로 무시한다.** 전기자석파처럼 `ignoreImmunity: false`를
명시한 것만 타입표를 받는다. 반대로 구현했다가 저주가 노말에게 안 닿는 버그가 있었다.

**턴은 재개 가능한 큐다.** 유턴이 턴 중간에 `MID_TURN_SWITCH`로 멈춘다. 턴 루프를 고칠 때
`_run_queue`가 중간에 return할 수 있다는 걸 잊지 말 것.

---

## 데이터 재생성

`data/raw/`와 `data/reference/`는 gitignore돼 있다. 새 환경에서는:

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -e ".[rl,dev]" json5
```

```bash
.venv/Scripts/python.exe scripts/fetch_showdown_data.py && .venv/Scripts/python.exe scripts/fetch_showdown_source.py && .venv/Scripts/python.exe scripts/fetch_regulation.py m_b
```

```bash
.venv/Scripts/python.exe scripts/build_champions_overrides.py && .venv/Scripts/python.exe scripts/build_champions_data.py && .venv/Scripts/python.exe scripts/build_champions_items.py && .venv/Scripts/python.exe scripts/build_names.py
```

`data/champions/`의 산출물은 커밋돼 있으므로, 재생성은 upstream이 바뀌었을 때만 필요하다.

---

## 사용자의 관찰을 기록하는 법

게임에서 뭔가를 보면 `tests/scenarios/`에 JSON 한 장으로 적는다. 형식은 그 폴더의 README.
미구현 메커니즘을 요구하는 시나리오는 **실패가 아니라 이유와 함께 skip**되므로, 관찰이
구현을 기다리다 유실되지 않는다. 지금은 전부 실행 중이다 (메타몽 괴짜/변신 포함).

---

## 남은 정확도 의문

- **PP 상한**: 챔피언스는 `(pp/5+1)*4`에 기본 PP 20 상한. 소스에서 확인했지만 게임에서
  한 번 대조해볼 가치가 있다.
- **op.gg 스크랩의 마지막 항목**(플카열매)이 잘려 있어 설명·가격이 비어 있다. id는 정상.
- **특성 2종**(`eelevate`, `firemane`)은 공개된 한국어 명칭이 없어 영어로 표시된다.
- **학습기술은 전 세대 합집합**이다. 챔피언스 자체 학습기술표가 공개되면
  `legality.learnable_moves`를 좁혀야 한다 — 지금은 관대한 쪽으로 틀린다.
