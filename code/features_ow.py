"""open-world용 피처 추출: 기록의 앞 K셀을 버리고 기존 피처(features.py + features_extra.py + features_extra2.py)를 다시 계산한다.

사용법: python -I -X utf8 code/features_ow.py <pkl 폴더> features 14 20 30      (K 여러 개 가능, K당 약 2분)
출력: features/ow_skip<K>_mon.csv (site, url, inst + 피처), features/ow_skip<K>_unmon.csv (피처)
      피처 이름은 기존 CSV와 같다. 단 위치를 세는 피처(first30_out, dir_001, bseq_4 등)는 K+1번째 셀부터 센다.

왜 버리나 (docs/OPENWORLD.md):
  mon은 1~14번째 셀 방향이 19,000개 기록 전부 같다(--+-+-++------). 사이트와 무관한 연결 준비 구간이다.
  unmon은 8번째 셀부터 mon과 다르다(8번째 셀: mon 100% 송신, unmon 99.98% 수신). 수집 환경 차이로 보인다.
  그대로 두면 open-world 모델이 사이트가 아니라 이 차이로 mon/unmon을 가른다(ROC-AUC 1.0000).
  closed-world(mon끼리 비교)에서는 이 구간이 모두 같아서 영향이 없으므로 기존 CSV는 그대로 쓴다.

시간: 버린 뒤에도 시각은 원래 기록 기준(절댓값) 그대로 둔다. 0으로 다시 맞추면 첫 셀이 0.0이 되어
      방향 규칙(0.0 → 수신)에 따라 방향이 바뀔 수 있기 때문이다.
"""
import csv, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 찾도록
import load_data as ld
from features import feats
from features_extra import feats_extra
from features_extra2 import feats_extra2

K = 0


def init(k):
    global K
    K = k


def all_feats(t):
    t = t[K:]
    return {**feats(t), **feats_extra(t), **feats_extra2(t)}


def fmt(v):
    return round(v, 4) if isinstance(v, float) else v


if __name__ == '__main__':
    D, OUT, ks = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]]
    mon = ld.load_raw_mon(D)
    traces, keys = [], []
    for i in range(ld.TOTAL_URLS):
        for j, s in enumerate(mon[i]):
            traces.append(s); keys.append((ld.site_label(i), i, j))
    del mon
    unmon = ld.load_raw_unmon(D)
    print('shortest trace:', min(map(len, traces)), min(map(len, unmon)))

    for k in ks:
        with Pool(os.cpu_count(), initializer=init, initargs=(k,)) as p:
            F = p.map(all_feats, traces, chunksize=200)
            U = p.map(all_feats, unmon, chunksize=200)
        names = list(F[0].keys())
        with open(os.path.join(OUT, f'ow_skip{k}_mon.csv'), 'w', newline='') as fo:
            w = csv.writer(fo); w.writerow(['site', 'url', 'inst'] + names)
            for r, key in zip(F, keys):
                w.writerow(list(key) + [fmt(r[x]) for x in names])
        with open(os.path.join(OUT, f'ow_skip{k}_unmon.csv'), 'w', newline='') as fo:
            w = csv.writer(fo); w.writerow(names)
            for r in U:
                w.writerow([fmt(r[x]) for x in names])
        print(f'skip {k}: mon {len(F)} rows, unmon {len(U)} rows, {len(names)} features', flush=True)
