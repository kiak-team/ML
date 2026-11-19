"""Extra credit ③-후속: 남은 세트 후보 2개를 group 5-fold로 잰다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/verify_sliver_sets3.py > logs/verify_sliver_sets3.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

기준선: 최종 202개 = @timing + 새 시간 124 + tnout/tnqin/tnqout, 5-fold 15.98% (logs/verify_sliver_sets2.log)
fold는 verify_sliver_sets2.py와 같은 GroupKFold(5) 고정이라 그 로그와 바로 비교할 수 있다.

후보
  206 = 202 + @count(n_total n_in n_out frac_out)
        근거: logs/short_splits.log. 정확도가 길이에 대해 U자다(25~50셀 21.7%, 1600+ 21.5%,
        가운데 200~400셀 13.7%). 길이 극단이 사이트를 좁힌다는 뜻인데 202개에는 셀 개수가 없고
        duration·cells_per_sec·pps_*로 간접 추정만 한다. 4.4에서 @count가 졌던 건 257개짜리라
        @cumul100 100개에 묻힌 결과였고, 4개만 더한 건 재지 않았다.
  182 = 142 + tnout + tnqout (tnqin 제외)
        근거: logs/verify_sliver_sets2.log. tnqin은 단독으로 +0.00%p다(tnorm_in의 역함수라 겹친다).
        성능이 같다면 피처 20개 적은 쪽을 쓴다.
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
SEEDS = (0, 1, 2)
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')


def main():
    from short_splits import load  # 같은 합치기·행 걸러내기를 쓴다
    df, names = load()
    G = ex.groups(names)
    timenew = [f for f in names if f.startswith(TIME_NEW)]
    grp = {p[:-1]: [f for f in names if f.startswith(p)] for p in ('tnout_', 'tnqin_', 'tnqout_')}
    base142 = G['timing'] + timenew
    final202 = base142 + grp['tnout'] + grp['tnqin'] + grp['tnqout']
    assert len(base142) == 142 and len(final202) == 202, (len(base142), len(final202))

    sets = {
        '기준선 202 (재확인)': final202,
        '+ @count (206)': final202 + G['count'],
        '142 + tnout tnqout (182)': base142 + grp['tnout'] + grp['tnqout'],
    }
    print(f'train {len(df)}행, 피처 {len(names)}개')
    print('@count: ' + ', '.join(G['count']))
    for k, v in sets.items():
        print(f'  {k}: {len(v)}개')
    print(flush=True)

    X, y, g = df[names].values, df.label.values, df.group.values
    ug = np.unique(g)
    folds = [(np.isin(g, ug[tr]), np.isin(g, ug[va])) for tr, va in GroupKFold(5).split(ug, groups=ug)]
    col = {f: i for i, f in enumerate(names)}
    t0, acc = time.time(), {}

    for seed in SEEDS:
        print(f'===== 시드 {seed} =====', flush=True)
        for k, fs in sets.items():
            cols = [col[f] for f in fs]
            v = []
            for mtr, mva in folds:
                m = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
                m.fit(X[mtr][:, cols], y[mtr])
                v.append(m.score(X[mva][:, cols], y[mva]))
            acc[(k, seed)] = v
            print(f'  {k:26s} ({len(fs):3d}개)  {np.mean(v):.4f} ± {np.std(v):.4f}'
                  f'  {np.round(v, 4)}  ({time.time() - t0:.0f}s)', flush=True)

    print('\n===== 정리 =====')
    print(f'{"세트":26s} {"개수":>4s} {"평균":>7s} {"fold표준편차":>11s} {"시드변동폭":>9s}')
    rows = []
    for k, fs in sets.items():
        done = [acc[(k, s)] for s in SEEDS]
        means = [np.mean(v) for v in done]
        rows.append((np.mean(means), k, len(fs), np.mean([np.std(v) for v in done]),
                     max(means) - min(means)))
    for mean, k, n, fstd, span in sorted(rows, reverse=True):
        print(f'{k:26s} {n:4d} {mean:7.4f} {fstd:11.4f} {span:9.4f}')

    print('\n== 판단 ==')
    r = {x[1]: x for x in rows}['기준선 202 (재확인)']
    lim = max(r[3], r[4])
    print(f'기준선 202개 {r[0]:.4f} (fold 표준편차 {r[3]:.4f}, 시드 변동폭 {r[4]:.4f}) → 유의 기준 {lim:.4f}')
    for mean, k, n, fstd, span in rows:
        if k == '기준선 202 (재확인)':
            continue
        d = mean - r[0]
        if d > lim:
            v = '올랐다. 채택한다.'
        elif d < -lim:
            v = '떨어졌다. 버린다.'
        elif n < r[2]:
            v = '변동폭 안이고 피처가 적다. 이쪽으로 바꾼다.'
        else:
            v = '변동폭 안이고 피처가 많다. 기준선을 유지한다.'
        print(f'{k:26s} ({n:3d}개) {mean:.4f} 차이 {d:+.4f} → {v}')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
