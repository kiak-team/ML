# open-world 실험 리포트 (B 파트)

이 문서의 모든 수치는 아래 0절의 leakage 처리를 거친 데이터에서 나온 것이다. 처리를 하지 않은 조건의 수치는 비교용(E4, E6 민감도)으로만 쓰며, 표에 따로 표시한다.

## 0. 데이터 누수(leakage) 처리 — 먼저 읽을 것

> **요약**: 본 데이터(pkl)와 extra credit(TrafficSliver)의 **모든 기록에서 앞 14셀을 버리고** 피처를 계산한다.
> 버린다는 것은 피처를 계산할 때 기록의 1~14번째 셀을 건너뛰고 15번째 셀부터 센다는 뜻이다(`t[14:]`).
> 원본 파일과 A팀 산출물은 바꾸지 않는다.
> pkl은 A팀이 이미 처리한 피처를 그대로 쓰고, TrafficSliver는 이번에 처리하지 않으면 누수가 남는다는 것을 확인해 같은 방식으로 처리했다.

### 0.1 누수 패턴

조교 공지: monitored trace의 99.98%는 처음 14셀(인덱스 0~13)의 방향 패턴이 같고, unmonitored에서는 10,000개 중 1개뿐이다.
"인덱스 7번 셀이 outgoing이면 monitored"라는 규칙 하나로 open-world 이진 분류가 99.99%가 된다.
웹사이트 내용이 아니라 수집 과정에서 생긴 흔적이므로, 이 신호를 쓴 점수는 실제 공격 성능이 아니다.

### 0.2 본 데이터(pkl): A팀 처리 승계

| 항목 | 내용 | 근거 |
|---|---|---|
| 처리 방식 | 앞 14셀을 **잘라낸 뒤** 피처 376개를 전부 다시 계산 (특정 피처만 빼는 방식이 아니다) | `code/features_ow.py` (`t[K:]`, K=14), 커밋 `ac64bc2` (2026-10-16) |
| 적용 시점 | 피처 추출 단계. mon·unmon 모두 동일하게 적용 | 같은 파일 |
| 시간 축 | 버린 뒤에도 시각은 원래 값 그대로 (0으로 다시 맞추지 않음) | 같은 파일 docstring |
| K=14 근거 | mon은 1~14번째 셀이 사실상 전부 같고(사이트 정보 없음), 버려도 closed-world 정확도 손실 없음 (95.56% → 95.67%) | `docs/OPENWORLD.md` 2.2–2.4 |
| 산출물 | **leak 포함**: `features/features_*.csv` (55+150+171) / **leak 제거**: `features/ow_skip14_*.csv` (376, 이름·순서 동일) | 둘 다 커밋됨 → E4 가능 |

커밋된 CSV로 다시 잰 값 (`code/ow_leak_check.py`, `logs/ow_leak_check.log`):

| 피처 | mon 앞 14셀 최빈 패턴 비율 | "idx7 outgoing → mon" acc / TPR / FPR |
|---|---|---|
| leak 포함 (`features_extra2_*`) | **99.98%** (`--+-+-++------`) | **0.9999** / 0.9999 / 0.0002 |
| leak 제거 (`ow_skip14_*`) | 3.65% | 0.579 / 0.435 / 0.147 (규칙이 더는 통하지 않음) |

leak 포함 수치가 조교 공지(99.98%, unmon 1만 개 중 1개)와 정확히 일치한다.

### 0.3 extra credit(TrafficSliver): 이번에 확인하고 같은 방식으로 처리

A팀은 split만 검사해 "자를 필요 없다"고 판단했다(`docs/EXTRA_CREDIT.md` 4.5, 단일 피처 AUC 최대 0.555).
원본 `mon.zip`·`unmon.zip`의 파일 228,000개 전부에서 앞 14셀 방향을 직접 읽어 다시 확인했다.

| 데이터 | mon 최빈 패턴 | unmon 최빈 패턴 | "idx7 outgoing → mon" acc / TPR / FPR |
|---|---|---|---|
| **join (방어 없음)** | `--+-+-++------` 67.0% | `--+-+-+-+-++--` 23.3% | **0.9940** / 0.9977 / 0.0097 |
| split 0~4 전체 (방어 있음) | `--------------` 68.5% | `--------------` 67.5% | 0.515 / 0.281 / 0.245 |
| split_0 ~ split_4 각각 | (전체와 같은 모양) | | 0.512 ~ 0.520 |
| └ split 중 앞 7셀이 `--+-+-+`로 시작하는 것 (mon 3.4%, unmon 4.5%) | `--+-+-++------` 87.2% | `--+-+-+-+-++--` 23.8% | **0.9952** / 1.000 / 0.0086 |

- **join에는 pkl과 같은 누수가 있다.** 따라서 A팀의 extra credit "방어 없음" open-world 수치(binary ROC-AUC 0.9977, TPR@FPR1% 97.1%)는 누수로 부풀려진 값이다. 이 문서는 그 값을 기준선으로 쓰지 않는다.
- **split은 전체로는 누수가 없지만 일부 조각에 남아 있다.** 방문마다 최대 1개의 split이 join의 시작 구간을 그대로 받으며, 그 부분집합에서는 규칙이 99.5% 맞는다.
- **처리 (결정: 2026-11-26)**: join과 split **모두** 앞 14셀을 일괄로 버린다. pkl과 같은 방식이다.
  누수가 있는 split은 흔적이 사라지고, 나머지 split은 셀 약 2.6%(중앙값 530셀 중 14셀)를 잃는다.
  E6에서는 버리지 않은 조건도 민감도 행으로 함께 보고한다.
- closed-world(mon끼리 비교)는 이 누수의 영향을 거의 받지 않는다. mon/unmon을 가르는 흔적이기 때문이다. A팀의 extra credit closed-world 수치는 그대로 유효하다.

### 0.4 이 리포트가 세운 가정

| 항목 | 가정 | 근거 |
|---|---|---|
| monitored 클래스 수 | **95** (사이트 0~94). open-world multi는 95 + unmon(−1) = **96클래스** | pkl: `features_mon.csv`의 `site` 고유값 95, URL 950, 사이트당 200·URL당 20. TrafficSliver: `mon.zip` 파일명 고유 사이트 95, 방문 0~199. 코드는 하드코딩하지 않고 데이터에서 센다 |
| "100개 사이트, 101 클래스" | 이 데이터와 맞지 않으므로 쓰지 않는다 | 위와 같음 |
| pkl 라벨 | `site = url // 10`, subpage = `url` (사이트당 10개), 반복 관측 = `inst` (0~19) | `code/load_data.py` |
| TrafficSliver 파일명 | mon `{사이트}-{방문}_split_{0-4}.cell` / `_join.cell`, unmon `{번호}_split_{k}.cell` / `_join.cell`. 한 줄 = `시각\t방향(±1)\t크기(±512)` | zip 파일명 전수 파싱(실패 0개), 실제 파일 열어 확인 |
| TrafficSliver subpage | 파일명에 subpage 정보가 없다. **방문(group) 단위 분할만** 한다 | 파일명 구조 |
| TrafficSliver split 시각 | **split 파일마다 시각이 0부터 다시 시작한다**(100%). join 시계 기준 offset이 원본에 없다. E6(b) merge 기준은 6단계에서 정해 명시한다 | `logs/ow_leak_check.log` |
| 빈 split | 뺀다 (mon 7,962개, unmon 9,953개) | A팀과 같음 |

재현: `python -I -X utf8 code/ow_leak_check.py "<extra credit 폴더>/mon.zip" "<extra credit 폴더>/unmon.zip" > logs/ow_leak_check.log` (약 20초, zip은 풀지 않는다)
