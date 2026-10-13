# 피처 55개 설명

`code/features.py`의 `feats()`가 기록 하나에서 뽑는 값이에요. 결과는 `features/features_mon.csv`(19,000행)와 `features/features_unmon.csv`(10,000행)에 저장돼 있어요. 방향은 [DATA.md](DATA.md)의 팀 공통 규칙(`c > 0`)을 따라요.

아래 설명에서 쓰는 말:
- **셀**: 기록 안의 숫자 하나 (512바이트 패킷 하나)
- **IAT**: 바로 앞 셀과의 시간 차이 (inter-arrival time)
- **버스트**: 같은 방향 셀이 연달아 이어지는 묶음. 예: 수신 수신 송신 수신 → 수신 버스트(2), 송신 버스트(1), 수신 버스트(1)

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
| `t_25pct`, `t_50pct`, `t_75pct` | 전체 셀의 25·50·75%가 지나간 시각 |
| `iat_mean`, `iat_std`, `iat_max`, `iat_p90` | IAT의 평균·표준편차·최댓값·90% 분위수 |
| `iat_zero_frac` | IAT가 정확히 0인 비율 |
| `iat_in_mean`, `iat_in_std` | 수신 셀끼리의 IAT 평균·표준편차 |
| `iat_out_mean`, `iat_out_std` | 송신 셀끼리의 IAT 평균·표준편차 |
| `pps_mean`, `pps_std`, `pps_max`, `pps_median` | 1초 구간마다 센 셀 수의 평균·표준편차·최댓값·중앙값 |

### 3. 순서·집중도 (10개, k-FP 논문 참고)
| 이름 | 뜻 |
|---|---|
| `out_order_mean`, `out_order_std` | 송신 셀이 기록 안에서 몇 번째 위치에 있는지의 평균·표준편차 |
| `in_order_mean`, `in_order_std` | 수신 셀 위치의 평균·표준편차 |
| `first30_out`, `last30_out` | 처음 / 마지막 30셀 중 송신 셀 수 |
| `first100_out` | 처음 100셀 중 송신 셀 수 |
| `conc_std`, `conc_max`, `conc_median` | 20셀씩 끊었을 때 구간마다 송신 셀 수의 표준편차·최댓값·중앙값 |

### 4. 버스트 (11개)
| 이름 | 뜻 |
|---|---|
| `n_bursts` | 버스트 개수 |
| `in_burst_mean`, `in_burst_std`, `in_burst_max` | 수신 버스트 길이의 평균·표준편차·최댓값 |
| `out_burst_mean`, `out_burst_std`, `out_burst_max` | 송신 버스트 길이의 평균·표준편차·최댓값 |
| `in_burst_ge10` | 길이 10 이상인 수신 버스트 수 |
| `in_burst_1`, `in_burst_2`, `in_burst_3` | 첫 번째·두 번째·세 번째 수신 버스트의 길이 |

### 5. CUMUL (12개, CUMUL 논문 참고)
방향을 +1/-1로 두고 앞에서부터 차례로 더한 누적합이에요. 슬라이드의 "누적 패킷 크기"를 512로 나눈 값과 같아요.

| 이름 | 뜻 |
|---|---|
| `cumul_10pct` ~ `cumul_100pct` | 기록의 10%, 20%, …, 100% 지점에서의 누적합 (10개) |
| `cumul_min` | 누적합의 최솟값 |
| `cumul_at_500` | 500번째 셀에서의 누적합 |

## 피처마다 쓸모가 있어 보이는지 (η², 탐색용)

η²(에타 제곱)는 "이 피처 값 하나만 보고도 사이트가 얼마나 잘 갈리는가"를 0~1로 나타낸 값이에요. 1에 가까울수록 사이트마다 값이 확연히 달라요. 우연히 나오는 수준은 약 0.005예요. 전체 표는 `logs/features_run.log`에 있어요.

- **높음 (0.52~0.56)**: `n_out`, `out_order_mean`, `n_bursts`, `cumul_*`, `n_total`, `n_in`
  → 다만 이 피처들끼리 상관계수가 0.9가 넘어요. 사실상 "기록이 얼마나 긴가"라는 같은 정보예요.
- **중간 (0.3~0.47)**: `frac_out`, `first30_out`, 버스트 평균, 시간 피처 대부분
- **거의 0**: `in_burst_1`, `in_burst_2`, `in_burst_3`
  → 모든 기록이 똑같은 패턴으로 시작해서 값이 거의 상수예요. 빼도 되는 후보예요.

**주의**: 이 η² 표는 mon 19,000개 전체(테스트 데이터 포함)로 계산했어요. 이 표를 보고 피처를 고른 다음 같은 테스트 데이터로 성능을 재면 **데이터 누수**가 생겨요. 피처를 고를 때는 train 데이터로만 계산하세요(`code/eval_nearest_centroid.py`의 [B] 방식).
