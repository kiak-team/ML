# Kiak!

Tor 웹사이트 핑거프린팅(WF) 프로젝트

> 프로젝트 전체 소개는 추후 작성 예정이다.

## 문서

- [docs/WORK_SUMMARY.md](docs/WORK_SUMMARY.md): **데이터 전처리 담당 작업 전체 정리** (여기부터 읽기)
- [docs/PREPROCESSING.md](docs/PREPROCESSING.md): 데이터 전처리 작업 정리 (폴더 구조, 파일별 역할, 실행 방법, 결과, 변경 기록, 다음에 할 일)
- [docs/DATA.md](docs/DATA.md): 데이터 설명
- [docs/FEATURES.md](docs/FEATURES.md): 피처 55개 설명
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): 피처 선택 실험 기록 (단계별 가설·결과·결정)

## 현재 후보 피처셋

`@all -@const @early @bseq @dir50` = **118개** (closed-world, RF 기준 CV 95.4%).
모델별로 다르게 쓰는 것을 기본으로 한다 — 근거는 EXPERIMENTS.md 4단계·4b단계.

> **6단계 제안 (팀 결정 대기)**: 여기에 `@cumulabs` 13개를 더한 **131개**로 바꾸자.
> 5개 모델 전부에서 118개보다 높다(RF 95.74, ET 95.83). → EXPERIMENTS.md 6단계

open-world는 피처셋이 다르다 (`@all @early @bseq @dir50 @qconst`, 앞 14셀 버림, 134개) — docs/OPENWORLD.md.
