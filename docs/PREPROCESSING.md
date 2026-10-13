# 데이터 전처리 작업 정리

Tor 웹사이트 핑거프린팅(WF) 프로젝트의 **데이터 전처리** 작업 베이스다. 데이터 읽기, 데이터 점검, 피처 추출, 피처 유효성 간단 확인까지 포함한다.

- 데이터 구조와 통계 → [DATA.md](DATA.md)
- 피처 55개의 정의 → [FEATURES.md](FEATURES.md)

## 폴더 구조와 파일별 역할

```
wf_project/
├─ README.md                  프로젝트 전체 소개 (작성 예정)
├─ docs/
│  ├─ PREPROCESSING.md        ← 이 파일 (전처리 작업 정리)
│  ├─ DATA.md                 데이터 설명 (구조, 통계, 품질 점검)
│  └─ FEATURES.md             피처 55개 설명
├─ code/
│  ├─ load_data.py            ① 공용 데이터 로더
│  ├─ check_data.py           ② 데이터 품질 점검
│  ├─ explore.py              ③ 기본 통계 (길이·시간·송신 비율)
│  ├─ features.py             ④ 피처 추출 + 피처 분석
│  └─ eval_nearest_centroid.py ⑤ 피처 유효성 간단 평가
├─ features/
│  ├─ features_mon.csv        ④가 저장한 mon 피처 (19,000행 × site,url,inst + 55열)
│  └─ features_unmon.csv      ④가 저장한 unmon 피처 (10,000행 × 55열)
└─ logs/                      각 코드의 실행 출력
   ├─ check_data.log          ②의 출력
   ├─ explore.log             ③의 출력
   ├─ features_run.log        ④의 출력
   └─ eval_nearest_centroid.log ⑤의 출력
```

### ① `code/load_data.py`: 공용 데이터 로더 (**팀 전원이 이 파일로 데이터를 읽는다**)
교수님이 제공한 `load_pickle_code.ipynb`와 **같은 규칙**으로 pkl을 읽는다.
- 라벨 `i // 10` (사이트 95개), 방향 `c > 0`이면 송신(+1) 아니면 수신(-1), 시간 `abs(c)`, 크기는 방향 × 512
- `load_mon(경로)` → 노트북과 같은 `X1`(시간), `X2`(방향×512), `y`(라벨)
- `load_unmon(경로)` → `X1`, `X2`
- `iter_mon(경로)` → 기록을 하나씩 반환해 메모리를 덜 쓴다. URL 번호와 방문 순번(inst)도 함께 반환한다.

```python
import sys; sys.path.insert(0, 'code')
import load_data as ld
X1, X2, y = ld.load_mon('C:/Users/user/Downloads')
```

### ② `code/check_data.py`: 데이터 품질 점검
빈 기록, NaN, 시간 역행, 길이 상한, 0.01초 단위 여부 등을 확인한다. 또한 **`load_data.py`가 교수님 노트북 코드와 같은 결과를 내는지** 29,000개 기록 전부를 비교한다(현재 불일치 0개).

### ③ `code/explore.py`: 기본 통계
길이 분위수, 지속 시간, 송신 비율을 mon과 unmon 각각 출력한다. [DATA.md](DATA.md) 3절의 표가 이 결과다.

### ④ `code/features.py`: 피처 추출 + 분석
1. 기록마다 피처 55개를 추출해 CSV 2개로 저장한다. **모델 학습에 쓰는 결과물이 이 CSV다.**
2. 피처별로 사이트 구분력(η²)과 mon/unmon 구분력(AUC)을 계산한다.
3. 거의 같은 정보를 담은 피처 쌍(상관계수 0.9 초과)을 찾는다.
4. 간단한 분류기(⑤와 동일)로 정확도를 출력한다.

### ⑤ `code/eval_nearest_centroid.py`: 피처 유효성 평가
④가 저장한 CSV를 읽어 아주 단순한 분류기로 정확도를 잰다. **최종 모델이 아니라** "이 피처들로 사이트를 구분할 수 있는가"를 빠르게 확인하는 용도다.
- **분류기 (nearest centroid)**: train 데이터로 사이트별 피처 평균점을 구하고, test 기록을 가장 가까운 평균점의 사이트로 예측한다.
- **분할**: URL마다 방문 20회 중 0~17번째는 train(17,100개), 18~19번째는 test(1,900개). 무작위 요소가 없다.
- **전처리**: `sign(x)·log(1+|x|)`로 값의 크기를 줄인 뒤 train의 평균과 표준편차로 표준화한다.
- ④와의 차이: 피처 선택용 η²를 **train으로만** 계산한다. ④는 test까지 포함해 계산하므로 데이터 누수가 있다.

## 실행 방법

Python 3 표준 라이브러리만 사용한다(numpy·sklearn 불필요). `<pkl 폴더>`는 pkl 3개를 둔 폴더다. 아래 명령은 **프로젝트 최상위 폴더(`wf_project/`)에서** 실행한다.

```
python -I code/check_data.py <pkl 폴더>                          # 수 분 소요
python -I code/explore.py    <pkl 폴더>
python -I -X utf8 code/features.py <pkl 폴더> features            # 약 2~3분, CSV 2개 생성
python -I -X utf8 code/eval_nearest_centroid.py features/features_mon.csv
```

## 현재까지 결과 (nearest centroid, 사이트 95개, test 1,900개)

우연히 맞힐 확률은 1/95 = 1.05%다.

| 피처 조합 | 정확도 |
|---|---|
| 전체 55개 | 25.7% |
| 순서·집중도 (10개) | 25.5% |
| 버스트 (11개) | 17.5% |
| 시간 (18개) | 11.4% |
| 개수 (4개) | 10.2% |
| CUMUL (12개) | 8.2% |
| η² 상위 10개 (train 기준) | 10.7% |
| η² 상위 20개 (train 기준) | 9.7% |
| η² 하위 20개 (train 기준) | 15.3% |

- 아주 단순한 분류기로도 우연(1%)보다 훨씬 높으므로, 피처에 사이트 정보가 담겨 있다.
- η² 상위 20개가 하위 20개보다 정확도가 **낮다**. 상위 피처끼리 같은 정보("기록 길이")를 반복해서 담고 있기 때문이다. 피처 선택 시 상관관계도 함께 봐야 한다.
- 강의에서 소개된 기법(k-FP, CUMUL, DF)은 90% 이상이다. 실제 모델(랜덤 포레스트, SVM, k-NN 등)은 모델링 단계에서 이 CSV로 학습한다.

## 변경 기록

- **2026-10-12**: 최초 작성. 피처 추출과 평가.
- **2026-10-13**: 교수님 노트북과 규칙 통일.
  - 공용 로더 `load_data.py` 추가.
  - 첫 셀 `0.0`의 방향을 **송신 → 수신**으로 변경. 이전 코드는 `+0.0`을 송신으로 처리해 노트북과 달랐다.
  - 그 결과 모든 기록에서 `n_out`이 1 감소, `n_in`이 1 증가했다. 방향을 쓰는 피처는 조금씩 바뀌었고 시간만 쓰는 피처는 그대로다. 정확도 변화는 ±0.3%p 이내다.
  - 피처 CSV와 로그를 모두 다시 생성했다.

## 다음에 할 일 (전처리 담당)

- [ ] 그래프: mon/unmon 길이 분포, 지속 시간 분포, 사이트별 길이 분포
- [ ] 시퀀스 모델(DF 등)용 입력: 고정 길이(예: 5,000)로 자르기/0 채우기. 잘리는 기록 비율도 확인
- [ ] open-world용 데이터: unmon 라벨(예: -1) 결정과 train/test 분할
- [ ] 거의 상수인 피처(`in_burst_1~3`) 제거, 서로 겹치는 피처 정리
