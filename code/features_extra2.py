"""추가 피처 2차 추출 (4단계 실험용). 기존 features.py / features_extra.py와 CSV는 건드리지 않는다.

사용법: python -I -X utf8 code/features_extra2.py <pkl 폴더> features      (약 1~2분)
출력: features/features_extra2_mon.csv (site, url, inst + 추가 피처)
      features/features_extra2_unmon.csv (추가 피처)
행 순서는 features_mon.csv / features_unmon.csv와 같다. experiment.py가 자동으로 붙여 쓴다.

검증할 가설 (근거는 docs/EXPERIMENTS.md 3단계: @early, @bseq가 모든 모델에서 효과가 컸다)
- @bseqlong (70개)  4~30번째 버스트 길이(@bseq)가 가장 효과가 컸다.
                     → 31~100번째 버스트 길이까지 늘린다. 버스트가 모자라면 0
- @dir      (100개) 처음 N셀 중 송신 "수"(@early)가 중요했다. 개수가 아니라 위치에도 정보가 있을 것이다.
                     → 처음 100셀의 방향값 그대로 (+1 송신, -1 수신). 기록이 짧으면 0
                     (@dir50 = 처음 50셀)
"""
import csv, os, sys
from itertools import groupby
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld

BSEQ_FROM, BSEQ_TO = 31, 100
N_DIR = 100


def feats_extra2(t):
    d = [ld.direction(c) for c in t]  # 방향 규칙은 load_data.py(교수님 노트북)와 동일
    f = {}
    # --- @bseqlong ---
    runs = [len(list(g)) for _, g in groupby(d)]
    for j in range(BSEQ_FROM, BSEQ_TO + 1):
        f[f'bseq_{j}'] = runs[j - 1] if j <= len(runs) else 0
    f['n_bursts_total'] = len(runs)  # 버스트가 몇 번째까지 있는지 확인용 (실험 그룹에는 넣지 않는다)
    # --- @dir ---
    for j in range(N_DIR):
        f[f'dir_{j + 1:03d}'] = d[j] if j < len(d) else 0
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
        F = p.map(feats_extra2, traces, chunksize=200)
        U = p.map(feats_extra2, unmon, chunksize=200)
    del traces, unmon
    names = list(F[0].keys())

    with open(os.path.join(OUT, 'features_extra2_mon.csv'), 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['site', 'url', 'inst'] + names)
        for r, k in zip(F, keys):
            w.writerow(list(k) + [r[x] for x in names])
    with open(os.path.join(OUT, 'features_extra2_unmon.csv'), 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(names)
        for r in U:
            w.writerow([r[x] for x in names])

    print(f'mon {len(F)} rows, unmon {len(U)} rows, {len(names)} extra features')
    nb = sorted(r['n_bursts_total'] for r in F)
    print('bursts per trace (mon): min', nb[0], 'p10', nb[len(nb) // 10], 'median', nb[len(nb) // 2])
    print('traces with >= 100 bursts:', sum(x >= 100 for x in nb), '/', len(nb))
    const = [x for x in names if len({r[x] for r in F}) == 1]
    print('constant in mon:', const or 'none')
