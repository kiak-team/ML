"""join 파일의 앞 K셀을 버리고 피처를 다시 뽑는다 — join 쪽 mon/unmon 수집 차이 확인용.

사용법: python -I -W ignore -X utf8 code/features_sliver_join_skip.py <npz> <out.csv> <K>

왜 (A팀 지적, 2026-11-26)
  join은 방어 전 전체 트래픽이므로 pkl과 같은 수집 차이를 그대로 갖고 있다. 실제로 재 보면
  join에서 dir_008 하나로 mon/unmon이 AUC 0.9940으로 갈린다(docs/OPENWORLD.md의 "8번째 셀"과 같다).
  split에서는 같은 피처가 0.5200이어서 차이가 없다 — BWR이 셀을 경로 5개로 흩어 그 고정 구간이 사라진다.
  그래서 EXTRA_CREDIT.md 4.12의 **join 열(방어 없음)**이 수집 차이에 올라탄 값일 수 있다.
  split 열(우리 결과)은 영향이 없지만, 거기서 나온 "방어가 탐지를 N배 깎는다"의 배수는 영향을 받는다.

pkl용 features_ow.py를 그대로 쓸 수 없다: 그쪽은 기록이 (시각, 크기) 쌍의 리스트여서 t[K:]로 자르는데,
sliver 기록은 (시각 배열, 크기 배열) 꼴이다. 여기서는 두 배열을 같이 자른다.
시각은 features_time.py가 ts - ts[0]으로 다시 맞추므로 절댓값을 그대로 둬도 된다.
방향은 크기의 부호에서 오므로(features.py) 첫 셀 시각이 0인지와 무관하다.

join만 뽑는다(19,000행). split은 4.5에서 차이가 없음을 확인했고, 행이 5배라 쓸 이유가 없다.
"""
import csv, os, sys
from itertools import islice
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 찾도록
import load_data as ld
from features import feats
from features_extra import feats_extra
from features_extra2 import feats_extra2
from features_ow import fmt
from features_time import feats_time

CHUNK = 5000
K = 0


def init(k):
    global K
    K = k


def all_feats(t):
    ts, sizes = t
    ts, sizes = ts[K:], sizes[K:]
    assert len(ts) >= 2, f'앞 {K}셀을 버리니 셀이 {len(ts)}개뿐이다'
    t = (ts, sizes)
    return {**feats(t), **feats_extra(t), **feats_extra2(t), **feats_time(t)}


if __name__ == '__main__':
    npz, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
    it = ld.iter_sliver(npz, parts=[ld.JOIN_PART])
    n_rows, names = 0, None
    with Pool(os.cpu_count(), initializer=init, initargs=(k,)) as p, open(out, 'w', newline='') as fo:
        w = csv.writer(fo)
        while True:
            batch = [((t.tolist(), s.tolist()), (lab, g, part)) for t, s, lab, g, part in islice(it, CHUNK)]
            if not batch:
                break
            for r, key in zip(p.map(all_feats, [b[0] for b in batch], chunksize=100), (b[1] for b in batch)):
                if names is None:
                    names = list(r); w.writerow(['label', 'group', 'part'] + names)
                w.writerow(list(key) + [fmt(r[x]) for x in names])
            n_rows += len(batch); print(n_rows, flush=True)
    print(f'{npz}: join {n_rows} rows, {len(names)} features, 앞 {k}셀 버림 → {out}')
