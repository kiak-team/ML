"""Extra credit 피처 추출: TrafficSliver npz(load_data.build_sliver 출력) → 기존과 같은 피처 CSV.

사용법: python -I -X utf8 code/features_sliver.py <npz> <out.csv>
        (mon, unmon을 따로 한 번씩 실행한다. 둘을 동시에 메모리에 올리지 않기 위해서다)
출력: label, group, part + features.py / features_extra.py / features_extra2.py 피처 (이름도 같다) + features_time.py 시간 피처
      part 0~4 = split, 5 = join. train/test는 반드시 group 단위로 나눈다 (docs/EXTRA_JOIN.md).

open-world처럼 앞 14셀을 버리지 않는다: split은 셀이 경로 5개로 흩어져서 그 고정 구간이 없다.
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


def all_feats(t):
    return {**feats(t), **feats_extra(t), **feats_extra2(t), **feats_time(t)}


CHUNK = 5000  # 이만큼씩 계산해서 바로 쓴다 (전부 모으면 10만 행 × 330 피처 dict가 GB 단위)


if __name__ == '__main__':
    npz, out = sys.argv[1], sys.argv[2]
    it = ld.iter_sliver(npz, parts=range(ld.JOIN_PART + 1))
    n_rows, names = 0, None
    with Pool(os.cpu_count()) as p, open(out, 'w', newline='') as fo:
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
    print(f'{npz}: {n_rows} rows, {len(names)} features → {out}')
