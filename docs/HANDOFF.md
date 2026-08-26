# 인계 문서 — 2026-08-26

새 세션이 여기부터 이어받으면 된다. 설계 원칙은 [DESIGN.md](DESIGN.md), 현재 상태 요약은
[README](../README.md). 이 문서는 **다음에 뭘 할지**와 **밟기 쉬운 함정**을 적어둔 것이다.

---

## 지금까지 무엇이 끝났나

**싱글 6→3과 더블 6→4 시뮬레이터가 둘 다 완성됐다.** 챔피언스에 존재하는 기술 500종 전부,
출전 로스터 특성 201종 중 198종, 도구 147종 전부가 실행된다. 테스트 450개, 코드 14.9k줄.

| | |
|---|---|
| 포맷 | 싱글 6→3, 더블 6→4 |
| 종족 | 235 + 메가 76 |
| 기술 | **500 / 500** |
| 특성 | 198 / 201 (남은 3종은 도구에 작용 — 되새김질·숙성·점착) |
| 도구 | **147 / 147** (배틀 도구 72 + 메가스톤 75) |
| 처리량 | 싱글 2,500 / 더블 680 turns/s (단일 코어) |

메가진화(배틀당 1회), 팀 프리뷰, TOD 4단계 판정, 한국어 표기까지 들어있다.
싱글·더블 각각 무작위 120전이 `unimplemented` 이벤트 0으로 끝난다.

---

## 다음 갈림길

더블이 끝났으므로 **레플리카 쪽 큰 덩어리는 남지 않았다.** 남은 건 원래 목표다.

### A. PettingZoo 어댑터 + 관측 인코딩 ← 여기가 먼저다
파티 세팅까지 학습한다는 원래 목표로 가는 길. 엔진은 준비돼 있다.

- `pkcm/envs/`는 **비어 있다.** DESIGN §1f대로 ParallelEnv 어댑터를 쓰면 된다
- 행동 마스크는 `state.legal_actions(state, player, position)`에서 그대로 나온다 —
  엔진과 환경이 같은 소스를 보므로 어긋날 수 없다
- **더블에서 에이전트를 어떻게 쪼갤지가 첫 설계 결정이다.** 위치마다 에이전트인지
  (PettingZoo다움), 한 플레이어가 튜플을 내는지 (엔진 API 그대로). 후자가 자연스럽다 —
  "같은 포켓몬을 두 위치에 못 낸다"는 제약이 위치를 가로지르기 때문에, 위치별 에이전트로
  쪼개면 그 제약을 표현할 데가 없다
- 관측 인코딩은 아직 없다. **DESIGN §1c(진실/정보집합 분리)가 여기서 처음 쓰인다** —
  `Observation(state, player)`와 `determinize`가 미구현이고, 하이브리드 탐색을 붙이려면
  이게 먼저다

### B. 파티 구성 학습 (DESIGN §6)
A가 선행돼야 한다. 배틀 정책이 없으면 팀의 좋고 나쁨을 평가할 수가 없다.

### C. 정확도 다듬기
아래 "남은 정확도 의문"이 그 목록이다. 급하지 않다.

---

## 밟기 쉬운 함정

**id는 절대 번역하지 않는다.** Showdown 데이터·챔피언스 override·op.gg 도구 목록을 잇는
키다. 이벤트도 id를 싣고, 사람이 읽는 이름은 `pkcm/render/names.py`가 만든다.

**메커니즘을 기억으로 재구성하지 말 것.** `data/reference/*.ts`를 읽어서 옮긴다. 기억으로
쓴 특성 초안은 "등록만 하고 아무것도 안 하는 핸들러"를 여럿 포함하고 있었고 통째로 버렸다.
`scripts/ability_report.py --show <name>`이 소스를 바로 보여준다.

**등록만 하고 아무도 안 읽는 효과가 이 프로젝트의 최다 빈출 버그다.** 지금까지 이렇게
잡힌 것: 봉인, 시끄러운소리, 와이드가드, 매직룸, 원더룸, 중력, 조가비갑옷, 전투무장.
전부 "등록돼 있고, 이름이 있고, 로그에도 나오고, 아무도 안 읽는" 상태였다. 실패하지
않았기 때문에 안 보였다 — **아무것도 안 하는 기술은 그냥 약한 기술처럼 보인다.**

**같은 실패의 두 번째 형태: 데미지는 들어가는데 조건부 효과만 빠진 기술.** 그래스슬라이더
(우선도 +1), 익스팬션포스, 라이징볼트, 테라버스트, 미스트버스트, 스틸롤러 — 터레인을 읽는
6종이 전부 이랬다. `move_support`는 **효과 없는 변화기만** 잡기 때문에 데미지 기술은
통과시킨다. 데미지 기술에 조건부 효과를 넣을 땐 그 검사가 안 봐준다는 걸 기억할 것.

지금은 세 개의 가드가 지킨다. 새 메커니즘을 넣을 때 이 셋을 통과시켜야 한다:

- `test_registered_effects_with_no_handlers_are_deliberate` — 핸들러 없는 volatile /
  사이드컨디션 / 룸은 **읽는 쪽이 어디 있는지 이름을 대야** 한다
- `test_no_orphaned_handler_functions` — 모듈 레벨 `_헬퍼` 중 아무도 참조하지 않는 게
  있으면 실패. 봉인·시끄러운소리가 정확히 이랬다 (핸들러를 다 쓰고 `register()`에 안 넘김)
- `abilities.INERT` 루프 — 이제 **이미 등록된 특성 위에 덮어쓰면 예외를 던진다.**
  예전엔 조용히 덮어써서 조가비갑옷·전투무장을 무력화했다

**"합법인가"와 "구현했는가"는 다른 질문이다** (DESIGN §1g). `legality.py`는 닌텐도가,
`scope.py`는 우리가 답한다. 섞으면 마일스톤이 올라갈 때 팀 합법성이 조용히 바뀐다.

**"데이터에 없다"와 "룰이 금지한다"도 다른 질문이고, 이건 실제로 틀렸던 적이 있다.**
개굴닌자의 유대변화를 dex에서 지웠다가 되돌렸다 — 데이터엔 있고 **룰이 금지하는** 것이라
`legality.BANNED_ABILITIES`가 맞는 자리다. 데이터 층에 규칙을 넣으면 엔진이 "왜 못 쓰는지"를
말할 수 없게 되고, 룰이 바뀌면 고칠 곳이 데이터 생성 스크립트가 된다.

**오버라이드에는 수명이 있다.** 대부분 교체 시 되돌아가고(변환자재·변신·트레이스·배틀스위치),
메가진화와 깨진 탈만 `__permanent__`로 남는다.

**상태기술은 타입 면역을 기본적으로 무시한다.** 전기자석파처럼 `ignoreImmunity: false`를
명시한 것만 타입표를 받는다.

**턴은 재개 가능한 큐다.** 유턴이 턴 중간에 `MID_TURN_SWITCH`로 멈춘다. 큐는 이제
`(플레이어, 필드위치)` 쌍을 담고, 애교부리기/뒤로돌기가 그 큐를 직접 재배열한다.

### 더블에서 새로 생긴 함정

**`Ref`는 필드 위치가 아니라 파티 슬롯이다.** `(side, slot)`. 위치는 `side.active[position]`
으로 슬롯을 가리키는 배열이고, 자리바꾸기는 그 배열만 바꾼다. 이 구분 덕에 HP·랭크·volatile이
그대로 쓰였다 — 헷갈려서 위치로 색인하기 시작하면 그 이점이 통째로 날아간다.

**기절한 포켓몬은 교체될 때까지 자기 위치에 남아 있다.** `must_switch[position]`이 어느
자리를 빚졌는지 알아야 하기 때문. "지금 필드에 누가 있나"를 묻는 쪽은 전부
`side.active_slots()` / `state.active_refs()`를 써야 한다 (기절한 쪽이 걸러진다).

**한 쪽이 한 마리만 남으면 자리 하나를 비우고 3인 필드로 계속한다.** 못 채우는 교체를
요구하면 턴이 데드락된다. `_end_of_turn`이 벤치 수만큼만 `must_switch`를 준다.

**교체는 leave / arrive / greet 세 단계다.** 한 마리씩 세 단계를 다 돌리면 먼저 도착한
쪽의 위협이 아직 안 나온 파트너를 때린다. 여러 마리가 동시에 들어올 땐 반드시 전원
arrive 후 greet.

**싱글에서 재현 불가능한 버그가 더블에서 나온다.** 매직미러 무한 반사가 그랬다 —
보유자 둘이 마주 볼 수가 없어서 가드가 없는 채로 남아 있었다.

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
더블 시나리오는 `position`(필드 위치)을 쓸 수 있고, 생략하면 0 — 기존 싱글 시나리오는
전부 그대로 돈다. 미구현 메커니즘을 요구하는 시나리오는 **실패가 아니라 이유와 함께 skip**
되므로, 관찰이 구현을 기다리다 유실되지 않는다.

**자가대전 로그를 눈으로 보는 것이 여전히 가장 잘 듣는 방법이다.** 그래서 렌더러는 이제
어떤 이벤트도 `repr()`로 떨어뜨리지 않는다 — 18종이 디버그 텍스트로 나오고 있었다.

```bash
.venv/Scripts/python.exe scripts/selfplay_demo.py --format doubles --seed 3
```

---

## hk가 게임에서 확인해준 것 (2026-08-26)

**다시 묻지 말 것.** 전부 구현과 일치했고, 각각 테스트로 박혀 있다.

| 확인한 것 | 답 | 어디에 |
|---|---|---|
| 매직미러가 반사한 변화기 | **특성 효과로 취급 → 재반사 불가.** 쓴 쪽만 피해 | `test_a_bounced_move_counts_as_an_ability_effect` |
| 더블 우선도·스피드 완전 동점 | **무작위.** 자리 순서 같은 결정적 규칙 없음 | `test_a_speed_tie_is_settled_at_random` |
| `normal` 기술로 자기 파트너 조준 | **UI에 실제로 나온다.** 행동 공간에 남겨둘 것 | `test_you_may_aim_at_your_own_partner` |
| 포챔스의 마비 너프 | **확률만 1/4→1/8.** 스피드 감소는 통상대로 절반 | `test_paralysis_halves_speed` |
| 급소 | **시리즈와 동일** (기본 1/24, 1.5배) | `test_crit_rate_is_one_in_twentyfour` |
| 더블에서 메타몽 괴짜 | **대각선** 복사 (정면 아님) | `test_imposter_copies_diagonally` |
| 더블에서 위협 | **두 마리 다** | `test_intimidate_drops_both_foes` |
| PP 상한 공식 | **`(기본PP//5+1)*4`, 기본 PP 20 상한** 맞음 | `test_pp_formula_is_not_the_series_one` |
| 더블 TOD | **싱글과 동일한 4단계** | `test_time_over_uses_the_same_four_tiers_in_doubles` |
| 해피너스·럭키 | **포챔스 미출전.** 알낳기가 그쪽 목록에 없던 이유 | `test_clefable_lost_its_gen_one_tm_moves` |
| 개굴닌자 유대변화 | **데이터엔 있고 룰이 금지.** dex가 아니라 클로즈 | `test_battle_bond_is_banned_rather_than_absent` |
| 포커스렌즈 | **실재함.** op.gg 스크랩이 빠뜨린 것 | `test_zoom_lens_only_helps_when_moving_second` |
| 암컷 냐오닉스 메가진화 | **가능.** 포케챔스 도감 쪽이 불완전 | `compare_pokechams.SETTLED` |

---

## 남은 정확도 의문

- **op.gg 스크랩의 마지막 항목**(플카열매)이 잘려 있어 설명·가격이 비어 있다. id는 정상.
- **비비용 id 불일치.** 우리 레귤레이션은 `vivillonfancy`, 포케챔스는 `vivillon`.
  종족값·타입 동일하고 학습기술은 base species로 상속되므로 동작엔 영향 없다.
- **특성 2종**(`eelevate`, `firemane`)은 공개된 한국어 명칭이 없어 영어로 표시된다.
- ~~학습기술이 전 세대 합집합~~ **해결됨.** 포케챔스 도감(hk가 찾음)에서 게임의 실제
  종족별 습득 기술표를 받아온다. `scripts/fetch_pokechams.py` →
  `scripts/build_champions_learnsets.py` → `data/champions/learnsets.json`.
  **클로즈는 그 위에 별도로 적용된다** — 표는 "게임이 가르치는가", 클로즈는 "포맷이
  허용하는가"로 다른 질문이다 (최면술·노래는 표에 있고 팀엔 못 넣는다).
