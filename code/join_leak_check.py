"""join 쪽 mon/unmon 수집 차이가 4.12의 join 열(방어 없음)을 부풀렸는지 판정한다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/join_leak_check.py > logs/join_leak_check.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

배경 (다른 팀 확인 요청, 2026-11-26)
  join은 방어 전 전체 트래픽이므로 pkl과 같은 수집 차이를 그대로 갖고 있다. 실제로 재 보면
  join에서 dir_008 하나로 mon/unmon이 AUC 0.9940으로 갈린다. split에서는 0.5200이다
  (BWR이 셀을 경로 5개로 흩어 그 고정 구간이 사라진다. 4.5의 "차이 없음"은 split 한정이다).

판정해야 하는 것
  최종 202개 세트에는 dir_*·bseq_*·first*가 없다. 그래서 dir_008은 모델에 들어가지 않는다.
  다만 tcum_in_3이 join에서 단일 AUC 0.7668이다. 이것이 수집 차이인지, mon(특정 95사이트)과
  unmon(무작위 19,000사이트)의 진짜 분포 차이인지는 피처만 봐서는 가를 수 없다.
  그래서 앞 K셀을 버리고 join open-world를 다시 낸다. AUC가 거의 안 움직이면 4.12의 join 열은
  그대로 유효하고(시간 피처가 그 구간에 올라타지 않았다), 크게 떨어지면 고쳐야 한다.

분할·지표는 4.12와 같다(group 단위 test 10%, openworld.py의 scores/at/report).
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld
from openworld import at, report, scores

N_TREES = 200
OFFSET = 10 ** 7
V2_TIME = ('tcum_', 'tbin_', 'gap_', 'tnorm_')
SKIPS = (0, 14, 28)
EARLY = [f'dir_{j:03d}' for j in range(1, 15)] + [f'bseq_{j}' for j in range(1, 7)]
MID = [f'dir_{j:03d}' for j in range(15, 29)]


def load(skip):
    f = os.path.join(ex.ROOT, 'features')
    out = []
    for tag in ('mon', 'unmon'):
        p = (f'sliver_{tag}_v2.csv' if skip == 0 else f'sliver_{tag}_join_skip{skip}.csv')
        df = pd.read_csv(os.path.join(f, p))
        df = df[df.part == ld.JOIN_PART].reset_index(drop=True)  # join만
        if tag == 'unmon':
            df['label'] = -1
            df['group'] = df.group.values + OFFSET
        out.append(df)
    return pd.concat(out, ignore_index=True)


def main():
    print(f'join 전용. 앞 {SKIPS}셀을 버린 세 경우를 같은 분할·같은 지표로 비교한다.\n', flush=True)
    t0 = time.time()
    for skip in SKIPS:
        df = load(skip)
        names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
        fs = ex.groups(names, pkl_qconst=False)['timing'] + [c for c in names if c.startswith(V2_TIME)]
        assert len(fs) == 202, len(fs)
        y, g = df.label.values, df.group.values
        is_test = ld.sliver_is_test(g)
        ybin = (y != -1).astype(int)
        X = df[fs].values.astype(np.float32)
        print(f'===== 앞 {skip}셀 버림 (mon {(y != -1).sum()} / unmon {(y == -1).sum()}행) =====')

        # 단일 피처 AUC: 누수가 남아 있는지
        for gname, cols in (('앞 14셀 방향·버스트', EARLY), ('15-28번째 셀', MID), ('최종 202개', fs)):
            cols = [c for c in cols if c in df.columns]
            best = (0.0, None)
            for c in cols:
                v = df[c].values.astype(float)
                if np.all(v == v[0]):
                    continue
                a = roc_auc_score(ybin, v)
                best = max(best, (max(a, 1 - a), c))
            print(f'  단일 AUC {gname:18s} ({len(cols):3d}개) 최대 {best[0]:.4f}  ({best[1]})')

        # open-world (4.12와 같은 방식)
        for mode, yy in (('binary', ybin), ('multi', y)):
            m = RandomForestClassifier(n_estimators=N_TREES if mode == 'binary' else 103,
                                       n_jobs=-1, random_state=0)
            m.fit(X[~is_test], yy[~is_test])
            s, pred = scores(m.predict_proba(X[is_test]), m.classes_, mode)
            r = report(s, y[is_test], pred)
            row = dict(r['rows'])
            site = lambda k: (f'  TPR_site@{k} {row[f"FPR≤{k}"]["TPR_site"]:.4f}'
                              if 'TPR_site' in row[f'FPR≤{k}'] else '')
            print(f'  {mode:6s} ROC-AUC {r["ROC-AUC"]:.4f}  PR-AUC {r["PR-AUC"]:.4f}'
                  f'  TPR@1% {row["FPR≤1%"]["TPR"]:.4f}{site("1%")}'
                  f'  TPR@5% {row["FPR≤5%"]["TPR"]:.4f}{site("5%")}'
                  f'  ({time.time() - t0:.0f}s)', flush=True)
        print(flush=True)
    print('판정: 앞 14셀을 버려도 binary ROC-AUC가 거의 그대로면 4.12의 join 열은 유효하다.')
    print('      크게 떨어지면 그만큼 수집 차이에 올라탄 값이므로 4.12의 배수를 고쳐야 한다.')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
