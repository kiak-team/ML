"""추가 피처 추출 (3단계 실험용). 기존 features.py와 CSV는 건드리지 않는다.

사용법: python -I -X utf8 code/features_extra.py <pkl 폴더> features      (약 1~2분)
출력: features/features_extra_mon.csv (site, url, inst + 추가 피처)
      features/features_extra_unmon.csv (추가 피처)
행 순서는 features_mon.csv / features_unmon.csv와 같다. experiment.py가 자동으로 붙여 쓴다.

검증할 가설 (근거는 logs/feature_importance.log)
- @early    (7개)  기록 앞부분의 송신 수가 가장 중요했다(first30_out, first100_out).
                    → 더 여러 길이에서 센다: 처음 10, 20, 50, 200, 300, 500, 1000셀 중 송신 수
- @cumulabs (13개) 고정 위치의 누적합(cumul_at_500)이 비율 위치의 누적합(cumul_10pct 등)보다 유용했다.
                    → 100~5000번째 셀 위치의 누적합. 기록이 그보다 짧으면 마지막 누적합
- @bseq     (30개) 버스트 통계(평균 등)가 아니라 앞쪽 버스트 길이의 "순서"에 사이트 정보가 있다.
                    → 1~30번째 버스트의 길이. 홀수 번째는 수신, 짝수 번째는 송신 버스트다
                      (모든 기록이 수신으로 시작). 버스트가 모자라면 0
- @cumul100 (100개) CUMUL 논문 방식: 누적합 곡선을 셀 개수 축으로 100등분한 지점에서 선형 보간.
                    기존 cumul_10pct~100pct(10개)를 더 촘촘하게 본다
"""
import csv, os, sys
from itertools import groupby
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld

EARLY = [10, 20, 50, 200, 300, 500, 1000]
CUM_POS = [100, 200, 300, 400, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000]
N_BSEQ = 30
N_CUMUL = 100


def feats_extra(t):
    n = len(t)
    d = [ld.direction(c) for c in t]  # 방향 규칙은 load_data.py(교수님 노트북)와 동일
    f = {}
    # --- @early ---
    for k in EARLY:
        f[f'first{k}_out'] = sum(1 for x in d[:k] if x == 1)
    # --- 누적합 ---
    c = 0; cum = []
    for x in d:
        c += x; cum.append(c)
    # --- @cumulabs ---
    for k in CUM_POS:
        f[f'cumul_at_{k}'] = cum[min(k, n) - 1]
    # --- @bseq ---
    runs = [len(list(g)) for _, g in groupby(d)]
    for j in range(N_BSEQ):
        f[f'bseq_{j + 1}'] = runs[j] if j < len(runs) else 0
    # --- @cumul100: x = 셀 번호(1..n), y = 누적합. x를 100등분한 지점에서 선형 보간 ---
    for j in range(1, N_CUMUL + 1):
        x = 1 + (n - 1) * j / N_CUMUL
        i = min(int(x), n - 1)          # cum[i-1] (x=i) 와 cum[i] (x=i+1) 사이
        w = x - i
        f[f'cumul100_{j:03d}'] = cum[i - 1] + w * (cum[i] - cum[i - 1])
    return f


if __name__ == '__main__':
    D = sys.argv[1]; OUT = sys.argv[2]
    mon = ld.load_raw_mon(D)
    traces, keys = [], []
    for i in range(ld.TOTAL_URLS):
        for j, s in enumerate(mon[i]):
            traces.append(s); keys.append((ld.site_label(i), i, j))
    del mon
    unmon = ld.load_raw_unmon(D)

    with Pool(os.cpu_count()) as p:
        F = p.map(feats_extra, traces, chunksize=200)
        U = p.map(feats_extra, unmon, chunksize=200)
    del traces, unmon
    names = list(F[0].keys())

    with open(os.path.join(OUT, 'features_extra_mon.csv'), 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['site', 'url', 'inst'] + names)
        for r, k in zip(F, keys):
            w.writerow(list(k) + [round(r[x], 4) for x in names])
    with open(os.path.join(OUT, 'features_extra_unmon.csv'), 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(names)
        for r in U:
            w.writerow([round(r[x], 4) for x in names])

    print(f'mon {len(F)} rows, unmon {len(U)} rows, {len(names)} extra features')
    const = [x for x in names if len({r[x] for r in F}) == 1]
    print('constant in mon:', const or 'none')
