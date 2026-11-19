"""Extra credit ③: 짧은 split(50셀 미만, 약 16%)을 어떻게 다룰지 정한다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/short_splits.py > logs/short_splits.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

세트는 verify_sliver_sets2.py가 확정한 202개(@timing + 새 시간 124 + tnout/tnqin/tnqout)다.
fold도 같은 GroupKFold(5) 고정이라 logs/verify_sliver_sets2.log와 바로 비교할 수 있다.

재는 것
  (a) 길이별 정확도 — 짧은 split이 정말 못 맞히는지, 전체 정확도를 얼마나 끌어내리는지
  (b) 짧은 것을 학습에서 빼면 긴 split 정확도가 오르는지 (짧은 게 잡음이라면 올라야 한다)
  (c) 짧은 것만으로 학습한 모델이 짧은 split에서 더 나은지 (길이별 모델이 필요한지)

n_total은 길이를 나누는 데만 쓰고 피처로는 넣지 않는다.
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
SEEDS = (0, 1)
SHORT = 50  # 셀. EXTRA_CREDIT.md 1절의 "50셀 미만 약 16%"
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')
FINAL_EXTRA = ('tnout_', 'tnqin_', 'tnqout_')
EDGES = [0, 10, 25, 50, 100, 200, 400, 800, 1600, 10**9]


def load():
    a = pd.read_csv(os.path.join(ex.ROOT, 'features', 'sliver_mon.csv'))
    b = pd.read_csv(os.path.join(ex.ROOT, 'features', 'sliver_mon_time2.csv')).drop(columns=['label'])
    df = a.merge(b, on=['group', 'part'], how='inner', validate='one_to_one')
    assert len(df) == len(a), f'(group, part) 합치기 실패: {len(df)} != {len(a)}'
    df = df[(df.part < ld.JOIN_PART) & ~ld.sliver_is_test(df.group.values)].reset_index(drop=True)
    names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    df[names] = df[names].astype(np.float32)
    return df, names


def main():
    df, names = load()
    G = ex.groups(names)
    fs = G['timing'] + [f for f in names if f.startswith(TIME_NEW + FINAL_EXTRA)]
    n = df.n_total.values.astype(int)
    short = n < SHORT
    print(f'train {len(df)}행, 최종 세트 {len(fs)}개')
    print(f'{SHORT}셀 미만 {short.sum()}행 ({short.mean() * 100:.1f}%), '
          f'길이 중앙값 {np.median(n):.0f}셀\n', flush=True)

    X, y, g = df[fs].values, df.label.values, df.group.values
    ug = np.unique(g)
    folds = [(np.isin(g, ug[tr]), np.isin(g, ug[va])) for tr, va in GroupKFold(5).split(ug, groups=ug)]
    t0 = time.time()

    def fit_predict(tr_mask, seed):
        """fold별로 tr_mask 안에서만 학습하고, 검증 행 전체에 대한 맞춘 여부를 돌려준다."""
        ok = np.zeros(len(df), dtype=bool)
        for mtr, mva in folds:
            m = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
            m.fit(X[mtr & tr_mask], y[mtr & tr_mask])
            ok[mva] = m.predict(X[mva]) == y[mva]
        print(f'    ({time.time() - t0:.0f}s)', flush=True)
        return ok

    allm = np.ones(len(df), dtype=bool)
    res = {}
    for seed in SEEDS:
        print(f'===== 시드 {seed} =====', flush=True)
        print('  전체 학습', flush=True);        res[('all', seed)] = fit_predict(allm, seed)
        print('  긴 것만 학습', flush=True);     res[('long', seed)] = fit_predict(~short, seed)
        print('  짧은 것만 학습', flush=True);   res[('short', seed)] = fit_predict(short, seed)

    def acc(key, rows):
        v = [res[(key, s)][rows].mean() for s in SEEDS]
        return np.mean(v), max(v) - min(v)

    print('\n===== (a) 길이별 정확도 (전체 학습, 시드 평균) =====')
    print(f'{"길이(셀)":>14s} {"행 수":>7s} {"비율":>6s} {"정확도":>8s} {"시드변동폭":>9s}')
    for lo, hi in zip(EDGES, EDGES[1:]):
        rows = (n >= lo) & (n < hi)
        if not rows.any():
            continue
        a, sp = acc('all', rows)
        tag = f'{lo}~{hi}' if hi < 10**8 else f'{lo}+'
        print(f'{tag:>14s} {rows.sum():7d} {rows.mean() * 100:5.1f}% {a * 100:7.2f}% {sp * 100:8.2f}%')
    a_all, sp_all = acc('all', allm)
    print(f'{"전체":>14s} {len(df):7d} 100.0% {a_all * 100:7.2f}% {sp_all * 100:8.2f}%')

    print('\n===== (b) 짧은 것을 학습에서 빼면 긴 split이 오르는가 =====')
    base_long, sp1 = acc('all', ~short)
    only_long, sp2 = acc('long', ~short)
    print(f'긴 split({SHORT}셀 이상 {(~short).sum()}행) 정확도')
    print(f'  전체 학습   {base_long * 100:6.2f}% (시드 변동폭 {sp1 * 100:.2f}%p)')
    print(f'  긴 것만 학습 {only_long * 100:6.2f}% (시드 변동폭 {sp2 * 100:.2f}%p)')
    d = only_long - base_long
    print(f'  차이 {d * 100:+.2f}%p → '
          + ('짧은 split은 학습에 해가 된다. 빼는 쪽을 쓴다.' if d > max(sp1, sp2) else
             '변동폭 안이다. 짧은 split이 학습을 망치지는 않으므로 그대로 둔다.'))

    print('\n===== (c) 짧은 split에 길이별 모델이 필요한가 =====')
    base_short, sp3 = acc('all', short)
    only_short, sp4 = acc('short', short)
    print(f'짧은 split({SHORT}셀 미만 {short.sum()}행) 정확도')
    print(f'  전체 학습     {base_short * 100:6.2f}% (시드 변동폭 {sp3 * 100:.2f}%p)')
    print(f'  짧은 것만 학습 {only_short * 100:6.2f}% (시드 변동폭 {sp4 * 100:.2f}%p)')
    d2 = only_short - base_short
    print(f'  차이 {d2 * 100:+.2f}%p → '
          + ('짧은 split 전용 모델이 낫다.' if d2 > max(sp3, sp4) else
             '전용 모델이 나을 근거가 없다. 모델 하나로 간다.'))
    print(f'\n찍으면 {100 / 95:.2f}%다. 짧은 split이 찍는 수준이면 그 조각은 포기하고,'
          f' 전체 정확도를 길이로 나눠 보고하는 쪽이 정직하다.', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
