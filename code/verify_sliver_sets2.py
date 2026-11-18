"""Extra credit ②: @tnorm 확장 후보가 142개 기준선을 넘는지 group 5-fold로 잰다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/verify_sliver_sets2.py > logs/verify_sliver_sets2.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

기준선: @timing + 새 시간 124개 = 142개, 5-fold 15.36% ± 0.16, 시드 변동폭 0.06%p (logs/verify_sliver_sets.log)
방식은 verify_sliver_sets.py와 같다. 같은 fold(GroupKFold, 고정)를 쓰므로 그 로그와 바로 비교할 수 있다.

단계
  1) 시드 0으로 후보 전부 → 어느 그룹이 올리는지 본다
  2) 기준선보다 높은 후보만 시드 1·2 추가 → 재현되는지 본다 (시드를 더 돌릴 가치가 있는 것만 돌린다)
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')
NEW2 = ('tnout_', 'tnqin_', 'tnqout_', 'tnfine_')


def load():
    a = pd.read_csv(os.path.join(ex.ROOT, 'features', 'sliver_mon.csv'))
    b = pd.read_csv(os.path.join(ex.ROOT, 'features', 'sliver_mon_time2.csv'))
    assert len(a) == len(b), (len(a), len(b))
    b = b.drop(columns=['label'])
    df = a.merge(b, on=['group', 'part'], how='inner', validate='one_to_one')
    assert len(df) == len(a), f'(group, part) 합치기 실패: {len(df)} != {len(a)}'
    df = df[(df.part < ld.JOIN_PART) & ~ld.sliver_is_test(df.group.values)].reset_index(drop=True)
    names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    df[names] = df[names].astype(np.float32)
    return df, names


def main():
    df, names = load()
    G = ex.groups(names)
    grp = {p[:-1]: [f for f in names if f.startswith(p)] for p in TIME_NEW + NEW2}
    timenew = sum((grp[p[:-1]] for p in TIME_NEW), [])
    base = G['timing'] + timenew  # 142개 기준선
    tn_in = grp['tnorm']          # 기존 tnorm_in 20개
    print(f'train {len(df)}행, 피처 {len(names)}개')
    print('그룹 크기: ' + ', '.join(f'@{k} {len(v)}' for k, v in grp.items()))
    print(f'기준선 @timing + 새 시간 = {len(base)}개\n', flush=True)

    sets = {
        '기준선 142 (재확인)': base,
        '+ @tnout': base + grp['tnout'],
        '+ @tnqin': base + grp['tnqin'],
        '+ @tnqout': base + grp['tnqout'],
        '+ @tnout @tnqin': base + grp['tnout'] + grp['tnqin'],
        '+ @tnout @tnqin @tnqout': base + grp['tnout'] + grp['tnqin'] + grp['tnqout'],
        'tnorm_in 20 → @tnfine 40': [f for f in base if f not in tn_in] + grp['tnfine'],
        '+ 새 100개 전부': base + grp['tnout'] + grp['tnqin'] + grp['tnqout'] + grp['tnfine'],
    }

    X, y, g = df[names].values, df.label.values, df.group.values
    ug = np.unique(g)
    folds = [(np.isin(g, ug[tr]), np.isin(g, ug[va])) for tr, va in GroupKFold(5).split(ug, groups=ug)]
    col = {f: i for i, f in enumerate(names)}
    t0, acc = time.time(), {}

    def run(k, fs, seed):
        cols = [col[f] for f in fs]
        v = []
        for mtr, mva in folds:
            m = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
            m.fit(X[mtr][:, cols], y[mtr])
            v.append(m.score(X[mva][:, cols], y[mva]))
        acc[(k, seed)] = v
        print(f'  {k:28s} ({len(fs):3d}개)  {np.mean(v):.4f} ± {np.std(v):.4f}'
              f'  {np.round(v, 4)}  ({time.time() - t0:.0f}s)', flush=True)
        return np.mean(v)

    print('===== 1단계: 시드 0 =====', flush=True)
    m0 = {k: run(k, fs, 0) for k, fs in sets.items()}

    ref = m0['기준선 142 (재확인)']
    cand = [k for k, v in m0.items() if v > ref and k != '기준선 142 (재확인)']
    print(f'\n기준선 {ref:.4f}보다 높은 후보: {cand if cand else "없음"}', flush=True)

    print('\n===== 2단계: 후보 + 기준선에 시드 1·2 =====', flush=True)
    for seed in (1, 2):
        print(f'--- 시드 {seed} ---', flush=True)
        for k in ['기준선 142 (재확인)'] + cand:
            run(k, sets[k], seed)

    print('\n===== 정리 =====')
    print(f'{"세트":28s} {"개수":>4s} {"평균":>7s} {"fold표준편차":>11s} {"시드변동폭":>9s} {"시드":>5s}')
    rows = []
    for k, fs in sets.items():
        done = [acc[(k, s)] for s in (0, 1, 2) if (k, s) in acc]
        means = [np.mean(v) for v in done]
        rows.append((np.mean(means), k, len(fs), np.mean([np.std(v) for v in done]),
                     max(means) - min(means), len(done)))
    for mean, k, n, fstd, span, ns in sorted(rows, reverse=True):
        print(f'{k:28s} {n:4d} {mean:7.4f} {fstd:11.4f} {span:9.4f} {ns:5d}')

    print('\n== 판단 ==')
    r = {x[1]: x for x in rows}['기준선 142 (재확인)']
    best = max((x for x in rows if x[1] != '기준선 142 (재확인)'), key=lambda x: x[0])
    gap = best[0] - r[0]
    print(f'기준선 142개 {r[0]:.4f} (fold 표준편차 {r[3]:.4f}, 시드 변동폭 {r[4]:.4f})')
    print(f'최고 후보 {best[1]} ({best[2]}개) {best[0]:.4f}')
    print(f'차이 {gap:+.4f}')
    print('→ 차이가 기준선의 fold 표준편차·시드 변동폭을 모두 넘는다. 후보를 채택한다.'
          if gap > max(r[3], r[4]) else
          '→ 차이가 변동폭 안이다. 피처를 100개 늘릴 근거가 없으므로 기준선 142개를 유지한다.')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
