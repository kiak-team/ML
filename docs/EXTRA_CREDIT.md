# Extra credit: TrafficSliver 방어 깨기 — 데이터 전처리·피처 정리

2026-11-10 기준. join 파일 규칙과 첫 결과는 [EXTRA_JOIN.md](EXTRA_JOIN.md)에 따로 정리했다.

## 요약

- 조교가 준 데이터는 BigEnough 데이터셋에 TrafficSliver BWR(경로 5개)을 적용한 것이다. 방문 1회가 split 파일 5개 + join 파일 1개다.
- 공격자는 입구 노드 하나만 본다고 가정하므로 **split 파일 하나가 instance 하나**다.
- 95-class closed-world, RF(200) 기준:

| 피처 | 정확도 |
|---|---|
| 방어 없음 (join→join, 기존 피처 375개) | 89.3% |
| 방어 있음 (split→split, 기존 피처 375개) | 7.5% |
| 방어 있음, `@timing` 18개만 | 10.4% |
| **방어 있음, `@timing` + 새 시간 피처 124개 (142개)** | **15.5%** |

  찍으면 1.05%다. 방어가 강하지만, 시간 피처로 처음 기준의 2배까지 올렸다.
- open-world(mon vs unmon)에서는 수집 환경 차이가 보이지 않는다. 피처 하나로 낸 AUC는 최대 0.555다.
  pkl 데이터처럼 앞 몇 셀을 잘라낼 필요가 없다.

## 1. 데이터

| | mon | unmon |
|---|---|---|
| 원본 zip | `mon.zip` (사이트 0–94 × 방문 0–199) | `unmon.zip` (0–18999) |
| 방문 수 | 19,000 | 19,000 |
| 파일 이름 | `{사이트}-{방문}_split_{0-4}.cell`, `_join.cell` | `{번호}_split_{k}.cell`, `_join.cell` |
| 빈 split (뺐다) | 7,962 (8.4%) | 9,953 (10.5%) |
| 쓰는 split | 87,038 | 85,047 |

- 한 줄은 `시간\t방향(±1)\t크기(±512)`이고, 크기는 항상 512다. 기존 pkl 규칙(방향 × 512)과 같다.
- split의 시간은 join의 시계를 그대로 가지고 있다. 파일마다 첫 셀이 0이 되도록 맞췄다.
- split 길이는 중앙값 약 400셀, 약 9초다(95%가 48초 이내). 50셀 미만인 split이 약 16%다.
- 줄인 zip(`mon_25`, `unmon_5000` 등)은 쓸 필요가 없었다. 전체 데이터를 numpy로 바꾸면 RAM에 들어간다.

## 2. 파이프라인

| 단계 | 명령 | 시간 | 출력 |
|---|---|---|---|
| zip → npz | `python -I -X utf8 code/load_data.py <Downloads>/mon.zip <Downloads>/sliver_mon.npz` (unmon도) | 각 3–4분 | `sliver_mon.npz` 798MB, `sliver_unmon.npz` 895MB |
| 피처 | `python -I -W ignore -X utf8 code/features_sliver.py <npz> features/sliver_mon.csv` (unmon도) | 각 약 6분 | 피처 500개 CSV, 각 약 220MB (gitignore) |
| 피처 중요도 (5-fold) | `code/feature_importance_sliver.py features/sliver_mon.csv > logs/feature_importance_sliver.log` | 약 2시간 | 기준선 A/B/C, 그룹 permutation |
| 세트 비교 + OW 확인 | `code/screen_sliver.py > logs/screen_sliver.log` | 약 30분 | 세트별 정확도, mon vs unmon AUC |

- npz는 zip을 풀지 않고 바로 읽어서 만든다. 저장소 밖(Downloads)에 둔다.
- **로그에는 stdout만 보낸다.** `2>&1`을 붙이면 joblib 경고가 섞여 로그가 71MB까지 커진 적이 있다.
- mon과 unmon은 따로 실행한다. 둘을 한꺼번에 메모리에 올리면 약 1.7GB가 든다.

코드 변경

- `load_data.py`
  - `build_sliver`: zip → npz, join 길이 = split 합 assert
  - `iter_sliver`: 기본은 split만, `parts=[5]`면 join만
  - `sliver_is_test`: 최종 test로 뗄 방문 10%
  - `as_seq`: pkl 기록과 `(times, sizes)`를 같은 형식으로 맞춤
- `features.py`, `features_extra.py`, `features_extra2.py`: `(times, sizes)`도 받는다. pkl로 뽑은 결과는 기존 CSV와 같다(확인함).
- `features_time.py` (새 파일): 시간 피처 124개. 파일을 직접 실행하면 assert 자가 확인이 돈다.
- `features_sliver.py`, `feature_importance_sliver.py`, `screen_sliver.py` (새 파일)

## 3. 분할 규칙 (누수 방지)

- 같은 방문의 split 5개와 join은 `group` 값이 같다. **train/test와 CV fold는 반드시 group 단위로 나눈다.**
  같은 방문의 조각이 양쪽에 들어가면 점수가 부풀려진다.
- 최종 test는 `ld.sliver_is_test(group)`, 방문의 10%다. 사이트마다 18–23개가 들어가고, 정수 해시라 언제 돌려도 같다. 튜닝 중에는 보지 않는다.
- 빠른 비교(`screen_sliver.py`)는 test를 뺀 방문 중 20%를 검증용으로 쓰는 holdout 1번이다.
  기존 피처 375개로 7.58%가 나와 5-fold 값(7.46%)과 맞는다.

## 4. 결과

### 4.1 기준선 (5-fold, 기존 피처 375개)

| | 정확도 |
|---|---|
| A join→join (방어 없음) | 89.3% ± 0.6 |
| B split→split (방어 있음) | 7.5% ± 0.1 |
| C join→split | 1.4% ± 0.1 |

C가 찍는 수준이다. 방어 전 데이터로 학습한 모델은 split에 통하지 않으므로 학습도 split으로 한다.

### 4.2 split에서 쓸모 있는 피처

- 그룹을 섞었을 때 정확도 하락은 `@timing` −5.0%p, `@cumul100` −2.6%p이고, 나머지 그룹은 0에 가깝다.
- 방어가 없을 때 핵심이던 방향·순서 피처(`@early`, `@bseq`, `@dir`)는 쓸모없다.
  BWR이 셀 묶음을 무작위 경로로 보내서, 한 경로 안의 셀 순서가 사이트와 무관해지기 때문으로 보인다.

### 4.3 새 시간 피처 (`features_time.py`, 124개)

| 그룹 | 개수 | 내용 |
|---|---|---|
| `tcum` | 38 | 고정 시각(0.25–60초)까지 받은/보낸 셀 수 |
| `tbin` | 40 | 처음 10초를 0.5초 칸으로 나눈 칸별 수신/송신 셀 수 |
| `tnorm` | 20 | 기록 길이를 20등분한 시각까지 받은 셀 비율 (시간 축 CUMUL) |
| `gap` | 26 | 셀 사이 빈 시간: 임계값별 개수, 가장 긴 10개, 0.2초 넘는 빈 시간이 처음 10번 시작한 시각 |

### 4.4 피처 세트 비교 (holdout, RF 200, 95-class)

| 세트 | 개수 | 정확도 |
|---|---|---|
| 기존 전체 | 375 | 7.6% |
| `@timing` | 18 | 10.4% |
| `@timing @cumul100` | 118 | 7.9% |
| 새 시간 피처만 | 124 | 14.6% |
| **`@timing` + 새 시간** | **142** | **15.5%** |
| `@timing @cumul100` + 새 시간 | 242 | 14.3% |
| `@timing @cumul100 @count @burst` + 새 시간 | 257 | 14.6% |
| 전체 | 499 | 13.8% |
| `@timing` + `tcum` | 56 | 11.3% |
| `@timing` + `tbin` | 58 | 11.9% |
| `@timing` + `tnorm` | 38 | 14.0% |
| `@timing` + `gap` | 44 | 12.5% |

- **피처를 줄일수록 좋아진다.** 쓸모없는 피처가 많으면 RF가 분기마다 무작위로 고르는 후보에 좋은 피처가 덜 들어간다.
- `@cumul100`은 5-fold에서 그룹을 섞었을 때는 중요하게 나왔지만, 넣으면 오히려 떨어진다. 피처 100개가 시간 피처를 묻어 버리는 것으로 보인다.
- 새 시간 피처 중 `tnorm`이 피처 20개로 +3.6%p를 올려 효과가 가장 크다. 기록 길이와 무관한 "로딩 곡선 모양"에 정보가 남아 있다.
- 피처 500개 전체 기준 중요도 상위: `iat_max`, `gap_top_01~10`, `iat_zero_frac`, `duration`.

### 4.5 open-world 수집 환경 차이 확인 (split만, mon vs unmon 이진)

| 세트 | AUC |
|---|---|
| 전체 (499) | 0.805 |
| `@timing` + 새 시간 (142) | 0.816 |

- 피처 하나로 낸 AUC는 최대 0.555(`tnorm_in_*`)다. 한두 피처로 mon/unmon이 갈리는 수집 차이 신호가 없다.
  pkl 데이터에서는 앞 14셀만으로 AUC 1.0이 나왔다.
- 이진 분류에 쓰인 상위 피처는 `tcum_in_1.5~3`(처음 1.5–3초에 받은 셀 수)과 `gap_at_02~04`다.

## 5. 남은 일

- [ ] 후보 세트(`@timing` + 새 시간, 142개)를 5-fold로 확인하고, 다른 모델(ET, k-NN, SVM, MLP 등 slide 16)과 비교
- [ ] 시간 피처 더 늘리기: `tnorm`이 효과가 커서 송신 쪽 `tnorm_out`, 더 촘촘한 등분 후보
- [ ] 짧은 split(50셀 미만, 약 16%) 처리 결정: 길이별 정확도 확인 후 그대로 둘지, 기준을 둘지
- [ ] open-world 본 실험: 임계값, TPR/FPR, PR 곡선 (본 프로젝트와 같은 방식)
- [ ] (NN을 쓸 경우) split용 시퀀스: 길이 L은 조각 길이 분포에 맞추고(중앙값 400셀), 시간 정보도 함께 넣기
- [ ] 마지막에 최종 test(`sliver_is_test`)로 한 번만 평가
