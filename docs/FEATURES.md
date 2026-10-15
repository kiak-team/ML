# 피처 55개 설명

`code/features.py`의 `feats()`가 기록 하나에서 추출하는 값이다. 결과는 `features/features_mon.csv`(19,000행)와 `features/features_unmon.csv`(10,000행)에 저장된다. 방향은 [DATA.md](DATA.md)의 팀 공통 규칙(`c > 0`)을 따른다.

용어:
- **셀**: 기록 안의 숫자 하나 (512바이트 패킷 하나)
- **IAT**: 직전 셀과의 시간 차이 (inter-arrival time)
- **버스트**: 같은 방향 셀이 연속된 묶음. 예: 수신 수신 송신 수신 → 수신 버스트(2), 송신 버스트(1), 수신 버스트(1)

## 그룹별 목록

### 1. 개수 (4개)
| 이름 | 뜻 |
|---|---|
| `n_total` | 전체 셀 수 |
| `n_in` / `n_out` | 수신 / 송신 셀 수 |
| `frac_out` | 송신 비율 = `n_out / n_total` |

### 2. 시간 (18개)
| 이름 | 뜻 |
|---|---|
| `duration` | 마지막 셀의 시각 (초) |
| `cells_per_sec` | `n_total / duration` |
| `t_25pct`, `t_50pct`, `t_75pct` | 전체 셀의 25·50·75%가 지난 시각 |
| `iat_mean`, `iat_std`, `iat_max`, `iat_p90` | IAT의 평균·표준편차·최댓값·90% 분위수 |
| `iat_zero_frac` | IAT가 정확히 0인 비율 |
| `iat_in_mean`, `iat_in_std` | 수신 셀끼리의 IAT 평균·표준편차 |
| `iat_out_mean`, `iat_out_std` | 송신 셀끼리의 IAT 평균·표준편차 |
| `pps_mean`, `pps_std`, `pps_max`, `pps_median` | 1초 구간별 셀 수의 평균·표준편차·최댓값·중앙값 |

### 3. 순서·집중도 (10개, k-FP 논문 참고)
| 이름 | 뜻 |
|---|---|
| `out_order_mean`, `out_order_std` | 송신 셀의 기록 내 위치(인덱스) 평균·표준편차 |
| `in_order_mean`, `in_order_std` | 수신 셀 위치의 평균·표준편차 |
| `first30_out`, `last30_out` | 처음 / 마지막 30셀 중 송신 셀 수 |
| `first100_out` | 처음 100셀 중 송신 셀 수 |
| `conc_std`, `conc_max`, `conc_median` | 20셀 단위 구간별 송신 셀 수의 표준편차·최댓값·중앙값 |

### 4. 버스트 (11개)
| 이름 | 뜻 |
|---|---|
| `n_bursts` | 버스트 개수 |
| `in_burst_mean`, `in_burst_std`, `in_burst_max` | 수신 버스트 길이의 평균·표준편차·최댓값 |
| `out_burst_mean`, `out_burst_std`, `out_burst_max` | 송신 버스트 길이의 평균·표준편차·최댓값 |
| `in_burst_ge10` | 길이 10 이상인 수신 버스트 수 |
| `in_burst_1`, `in_burst_2`, `in_burst_3` | 첫 번째·두 번째·세 번째 수신 버스트 길이 |

### 5. CUMUL (12개, CUMUL 논문 참고)
방향을 +1/-1로 두고 앞에서부터 누적한 합이다. 슬라이드의 "누적 패킷 크기"를 512로 나눈 값과 같다.

| 이름 | 뜻 |
|---|---|
| `cumul_10pct` ~ `cumul_100pct` | 기록의 10%, 20%, …, 100% 지점의 누적합 (10개) |
| `cumul_min` | 누적합의 최솟값 |
| `cumul_at_500` | 500번째 셀의 누적합 |

## 피처별 구분력 (η², 탐색용)

η²(에타 제곱)는 "피처 하나만으로 사이트가 얼마나 잘 갈리는가"를 0~1로 나타낸다. 1에 가까울수록 사이트별 값 차이가 뚜렷하다. 우연 수준은 약 0.005다. 전체 표는 `logs/features_run.log`에 있다.

- **높음 (0.52~0.56)**: `n_out`, `out_order_mean`, `n_bursts`, `cumul_*`, `n_total`, `n_in`
  → 단, 이 피처들끼리 상관계수가 0.9를 넘는다. 사실상 "기록 길이"라는 같은 정보다.
- **중간 (0.3~0.47)**: `frac_out`, `first30_out`, 버스트 평균, 시간 피처 대부분
- **거의 0**: `in_burst_1`, `in_burst_2`, `in_burst_3`
  → 모든 기록이 같은 패턴으로 시작해 값이 거의 상수다. 제거 후보다.

**주의**: 이 η² 표는 mon 19,000개 전체(테스트 데이터 포함)로 계산했다. 이 표로 피처를 고른 뒤 같은 테스트 데이터로 성능을 재면 **데이터 누수**가 생긴다. 피처 선택 시에는 train 데이터로만 계산한다(`code/eval_nearest_centroid.py`의 [B] 방식).

## 모델 기준 중요도 (랜덤 포레스트)

η²는 피처를 하나씩 따로 본다. 피처끼리 정보가 겹치는 것까지 반영한 랜덤 포레스트 중요도는 [PREPROCESSING.md](PREPROCESSING.md)의 "랜덤 포레스트 피처 중요도" 절에 있다. 요약:
- 기록 길이 계열 17개는 묶음으로는 가장 중요하지만 서로 대신한다 → 대표 1~2개면 충분하다.
- 단일 피처로는 `first30_out`, `first100_out`, `cumul_at_500` (기록 앞부분의 방향 패턴)이 가장 중요하다.
- 시간 피처 18개는 빼도 사이트 분류 정확도가 그대로다.
