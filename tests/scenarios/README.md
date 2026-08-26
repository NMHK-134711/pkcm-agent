# 시나리오 — 실제 게임에서 본 것을 영구 테스트로

포켓몬 챔피언스에 대한 유일한 정답지는 **챔피언스 본편**이다. Showdown은 스칼렛/바이올렛을
구현한 것이고, 챔피언스가 다른 지점(SP 능력치, 랭크전 테라스탈 부재, 메가 부활과 신규 메가,
`Mega Sol`·`Dragonize` 같은 신규 특성)이 하필 우리가 가장 정확해야 하는 지점이다.

그래서 오라클은 **게임을 직접 하는 사람**이다. 이 디렉터리는 거기서 본 것이 저장소로 들어와
남는 방법이다.

## 핵심 성질: 구현 전에 먼저 적어도 된다

각 시나리오는 필요한 메커니즘을 `requires`에 선언한다. 엔진이 아직 구현하지 않은 걸 요구하면
**실패가 아니라 이유와 함께 skip**된다.

```
SKIPPED  needs abilities, mega-evolution, stat-stages, transform
         (recorded from: Pokemon Champions, reported by hk)
```

관찰한 즉시 적어두면 되고, 마일스톤이 올라가는 순간 스스로 켜져서 검증을 시작한다.
메커니즘을 구현하는 커밋에서 `pkcm/testing/scenario.py`의 `IMPLEMENTED`에 이름을 추가하면 된다.

## 형식

```jsonc
{
  "name":     "고유 슬러그",
  "source":   "어디서 관찰했는지 — 날짜, 상황",
  "requires": ["abilities", "stat-stages"],   // 없으면 빈 배열
  "notes":    "무엇을 왜 확인하는지. 나중의 자신을 위해 길게 써도 좋다.",
  "format":   "singles",

  "teams": [ [ 6마리 ], [ 6마리 ] ],          // {"species","ability","moves","nature","sp","item"}
  "select": [[0,1,2], [0,1,2]],              // 팀 프리뷰 선출 (앞이 선봉)

  "setup":  [ {"do": "set_hp", "side": 1, "slot": 0, "value": 1} ],   // 선택
  "turns":  [ ["move:0", "move:0"], ["switch:1", "move:1"] ],

  "expect": [ {"check": "winner", "value": 0, "note": "왜 이걸 기대하는지"} ]
}
```

`teams`는 6마리를 채워야 한다 (팀 프리뷰가 6→3이므로). 시나리오 팀은 **합법성 검사를 거치지
않는다** — 엔진 픽스처지 등록할 팀이 아니다.

### 행동

`"move:0"` `"switch:2"` `"struggle"` `"pass"`

### setup 연산

| `do` | 인자 | 비고 |
|---|---|---|
| `set_hp` | `side`, `slot`, `value` | |
| `set_pp` | `side`, `slot`, `move`, `value` | |
| `boost` | `side`, `slot`, `stat`, `stages` | **미구현** (M1) |
| `mega` | `side`, `slot` | **미구현** (M4) |

미구현 연산은 그걸 쓰는 시나리오가 어차피 skip되므로 지금 적어둬도 안전하다. 해당 마일스톤이
오면 큰 소리로 실패하면서 무엇을 만들어야 하는지 알려준다.

### expect 체크

| `check` | 인자 | 뜻 |
|---|---|---|
| `species` | `side`, (`slot`) | 현재 폼의 종족 id |
| `ability` | `side`, (`slot`) | 특성 id |
| `hp` | `side`, (`slot`) | 남은 HP |
| `stat` | `side`, `stat`, (`slot`) | 실능력치 (랭크 미반영) |
| `boost` | `side`, `stat`, (`slot`) | 랭크 단계 — **미구현** (M1) |
| `fainted` | `side`, (`slot`) | |
| `active` | `side` | 현재 나와 있는 슬롯 번호 |
| `phase` | | `battle` / `forced_switch` / `finished` |
| `winner` | | `0` / `1` / `null`(무승부) |
| `first_mover` | | 그 턴에 먼저 움직인 쪽 |
| `event` | `kind` 및 임의의 이벤트 필드 | 로그에 그런 이벤트가 있었는가 |

`slot`을 생략하면 그 시점의 활성 포켓몬을 본다.

## 좋은 시나리오의 조건

- **하나만 확인한다.** 한 파일에 여러 메커니즘을 섞으면 깨졌을 때 원인을 못 찾는다.
- **`note`에 숫자의 근거를 남긴다.** `"(102 + 20 + 32) * 1.1"` 처럼.
- **`source`를 정확히 쓴다.** 나중에 "이게 진짜 게임에서 본 건지, 내가 추측한 건지"를
  구분할 수 있어야 한다. 추측이면 추측이라고 쓴다.
