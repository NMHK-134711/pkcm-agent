# 에이전트와 직접 대전하기

브라우저에서 `curriculum4` 에이전트와 싱글 6→3 배틀을 합니다. 한국어 UI이고,
설치 후 명령 하나면 뜹니다.

```bash
python scripts/play_web.py --checkpoint runs/curriculum4/best.pt --agent-party 7
```

브라우저가 `http://127.0.0.1:8760` 으로 열립니다. 팀 프리뷰에서 셋을 고르고,
매 턴 기술이나 교체를 누르면 됩니다.

## 상대가 누구인가

`runs/curriculum4/best.pt` 는 이 저장소에서 **손으로 짠 탐색보다 분리 가능하게
강한 것이 측정된 첫 신경망**입니다. 자기가 학습한 네 파티에서 397 결정 게임 동안
59.4% [54.5%, 64.2%] 를 기록했습니다. 그 전 여덟 번의 학습 실행은 전부 신뢰구간이
50% 를 덮었습니다 — 즉 탐색과 구분되지 않았습니다.

경기 방식은 AlphaZero 와 같습니다. 신경망이 각 수의 사전확률과 국면의 가치를 내고,
결정화 MCTS 가 그것을 씨앗 삼아 탐색합니다. 상대(사람)의 숨은 세트는 실제 랭커
파티 풀에서 표본을 뽑아 추정하며, 관찰한 기술로 후보를 좁혀 나갑니다.

## 어느 파티를 주는가

파티 인덱스는 `runs/tournament_800_fixed.json` 과 `runs/profile_curriculum4.json`
에 있습니다. 이 에이전트가 가장 잘 쓰는 셋은 프로파일에서 잰 순서대로입니다.

| 파티 | 에이전트 승률 | 팀 |
|---|---|---|
| **7** | **74.0%** | ミルク Garchomp & Mega Dragonite Offense |
| 17 | 67.3% | Ayaka_p_p Delphox and Friends Offense |
| 10 | 58.0% | 엑자몽 중심 10일전 랭크 13위 달성 파티 |
| 14 | 52.0% | 가람 Mega Gengar and Kangaskhan Offense |

이 넷이 학습에 쓰인 파티라 가장 셉니다. 다른 파티를 주면 약해집니다 — 파티 5나 13을
주면 10~14% 까지 떨어지고, 왜 그런지는 아직 모릅니다.

```bash
# 에이전트에게 최강 파티를, 나에게 다른 랭커 파티를
python scripts/play_web.py --checkpoint runs/curriculum4/best.pt \
    --agent-party 7 --your-party 17
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
