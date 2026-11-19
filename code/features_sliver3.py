"""Extra credit 3차 시간 피처 추출 (11/15 세트 재현용): npz → features_time2.py 피처만 담은 CSV.

사용법: python -I -W ignore -X utf8 code/features_sliver2.py <npz> <out.csv>

features_sliver.py를 다시 돌리지 않는 이유: 기존 500개는 그대로 쓰고 새 100개만 붙이면 되고,
방금 검증한 142개 기준선(logs/verify_sliver_sets.log)을 바꾸지 않으려면 기존 피처가 그대로여야 한다.
verify_sliver_sets2.py가 (group, part)로 두 CSV를 합친다.
"""
import csv, os, sys
from itertools import islice
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 찾도록
import load_data as ld
from features_ow import fmt
from features_time3 import feats_time3

CHUNK = 5000

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
            for r, key in zip(p.map(feats_time3, [b[0] for b in batch], chunksize=100), (b[1] for b in batch)):
                if names is None:
                    names = list(r); w.writerow(['label', 'group', 'part'] + names)
                w.writerow(list(key) + [fmt(r[x]) for x in names])
            n_rows += len(batch); print(n_rows, flush=True)
    print(f'{npz}: {n_rows} rows, {len(names)} features → {out}')
