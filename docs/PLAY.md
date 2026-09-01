# 에이전트와 직접 대전하기

브라우저에서 학습된 에이전트와 싱글 6→3 배틀을 합니다. 한국어 UI이고,
설치 후 명령 하나면 뜹니다.

```bash
python scripts/play_web.py --checkpoint runs/pilot43/best.pt --agent-party 43
```

브라우저가 `http://127.0.0.1:8760` 으로 열립니다. 팀 프리뷰에서 셋을 고르고,
매 턴 기술이나 교체를 누르면 됩니다.

## 상대가 누구인가

> **2026-09-02: 망은 지금 엔진에서 우세가 없습니다.** 파티 43 미러, 398게임,
> 손으로 짠 탐색 상대 **47.7% [42.9%, 52.6%] — 분리되지 않습니다.** 아래
> 54.5%는 그 이후 열일곱 개 조건부 위력, `selfBoost`, 스핀의 필드 정리,
> 탁쳐서떨구기, 도둑질 계열, 솔라빔의 쾌청 즉발, 그림자밟기 구속 해제가
> 들어가기 *전* 엔진에서 잰 값입니다. 망은 그게 전부 없는 세계에서 학습했고,
> 탐색은 새 규칙으로 돕니다.
>
> 떨어진 폭 자체는 1.6σ라 분리되지 않습니다 — "예전보다 나빠졌다"는 말은 아직
> 못 합니다. 확실한 건 **지금은 망을 얹을 이유가 없다**는 것뿐입니다.
> `--checkpoint` 를 빼고 손으로 짠 탐색으로 두시고, 고쳐진 엔진에서 재학습한
> 뒤에 다시 재는 것이 순서입니다.

**`runs/pilot43/best.pt` + 파티 43** 이 가장 센 조합이었습니다. 파티 43(메가
메타그로스·하마돈 모래·메가마폭시)은 46개 랭커 파티 라운드로빈의 1위
(66.0% [61.0%, 70.7%])로, 필드 위로 분리되는 유일한 팀입니다. 이 조합은 이전
배포본(curriculum4 + 파티 7)과의 맞대결 200게임에서 **73.5% [67.0%, 79.1%]**
로 이겼습니다 — 둘 다 이전 엔진에서의 측정입니다.

경기 방식은 AlphaZero 와 같습니다. 신경망이 각 수의 사전확률과 국면의 가치를
내고, 결정화 MCTS 가 그것을 씨앗 삼아 탐색합니다. 상대(사람)의 숨은 세트는
실제 랭커 파티 풀(46파티, 276세트)에서 표본을 뽑아 추정하며, 관찰한 기술로
후보를 좁혀 나갑니다.

이전 배포본도 남아 있습니다 — `runs/curriculum4/best.pt` 는 파티 7을 모는
실력만은 여전히 최고입니다(같은 팀 미러에서 74.0%). 골라 쓰면 됩니다:

```bash
python scripts/play_web.py --checkpoint runs/curriculum4/best.pt --agent-party 7
```

## 어느 파티를 주는가

파티 인덱스는 `runs/tournament_46.json` 에 있습니다 (0-19 는 pkmnchamps 임포트,
20-45 는 pokedb 전사본). 에이전트별 최적 파티:

| 체크포인트 | 파티 | 근거 |
|---|---|---|
| `pilot43/best.pt` | **43** | 그 팀으로 10 이터레이션 특화. 미러 54.5% |
| `curriculum4/best.pt` | **7** | 미러 74.0%. 단, 파티 43 같은 신규 종족 팀을 주면 33.7%까지 떨어짐 |

학습하지 않은 파티를 주면 어느 쪽이든 약해집니다. 특화의 대가입니다.

```bash
# 현재 최강 조합 그대로
python scripts/play_web.py --checkpoint runs/pilot43/best.pt     --agent-party 43 --your-party 14
```

`--your-party` 를 빼면 내 팀은 랭커 풀에서 무작위로 조립됩니다.

## 강도 조절

```bash
--search-iterations 400    # 빠름, 약함 (수당 1초 미만)
--search-iterations 800    # 학습에 쓰는 예산
--search-iterations 3200   # 기본값. 가장 강하고 수당 몇 초
```

탐색 예산은 배가할 때마다 강해지는 것이 측정됐습니다(+9.7 / +7.6 / +6.5pp, 3200에서
평평해짐). 낮추면 확실히 약해집니다.

신경망 없이 손으로 짠 탐색만 상대하려면 `--checkpoint` 를 빼면 됩니다.

```bash
python scripts/play_web.py --agent-party 7
```

## 그 밖의 옵션

| 옵션 | 뜻 |
|---|---|
| `--seed 42` | 팀을 고정. 같은 판을 다시 하고 싶을 때 |
| `--format doubles` | 더블 6→4 |
| `--teams parties:7,17,10,14` | 양쪽 다 이 파티들 중에서 |
| `--port 9000` | 다른 포트 |
| `--no-browser` | 브라우저 자동 실행 안 함 |

터미널에서 하고 싶으면 `scripts/play.py` 가 같은 대전을 텍스트로 진행합니다.

## 설치

README 의 [셋업](../README.md#셋업) 을 따르면 됩니다. 요약하면:

```bash
git clone https://github.com/NMHK-134711/pkcm-agent
```

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install -e ".[rl,train,dev]" json5
```

그 다음 `README.md` 의 데이터 받기 단계(`fetch_showdown_data.py`,
`fetch_showdown_source.py`, `fetch_regulation.py m_b`, `build_champions_data.py`)를
한 번 돌리면 엔진이 읽는 표가 만들어집니다. `data/raw/` 는 gitignore 라 저장소에
없습니다.

신경망을 굴리려면 `torch` 가 필요하고, 위 `[train]` 에 들어 있습니다. CPU 만으로도
돌아갑니다 — GPU 가 있으면 추론이 그쪽으로 갑니다.
