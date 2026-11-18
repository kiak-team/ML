"""Extra credit 후보 세트 검증: screen_sliver.py의 holdout 1회 결과를 group 5-fold × 시드 여러 개로 다시 잰다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/verify_sliver_sets.py features/sliver_mon.csv > logs/verify_sliver_sets.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

왜 다시 재나
  screen_sliver.py는 세트 12개를 holdout 1번 + 시드 1개로 비교했다. 검증 15,771행에서 15.48%의 표준오차는
  0.29%p라, 1·2·3등의 차이(15.48 / 14.58 / 14.56%)가 평가 노이즈의 3배 수준이다. 학습 쪽 변동(시드)은 아예 없다.
  closed-world 6단계에서 @cumulabs를 넣을 때 쓴 기준과 같게, fold 간 표준편차와 시드 변동폭을 함께 본다.

방식
  - test(ld.sliver_is_test, 방문 10%)는 쓰지 않는다. 나머지를 group 단위 5-fold로 나눈다 (같은 방문의 split이 fold를 넘지 않는다).
  - GroupKFold는 시드가 없으므로 fold 구성은 고정이고, 시드는 RF만 바꾼다. 그래서 시드 간 차이가 곧 모델 변동폭이다.
  - 시드를 바깥 루프로 돌려서 시드 0이 끝나면 전체 세트 순위가, 시드가 더 돌면 재현성이 보인다 (중간에 끊어도 쓸 수 있다).

세트
  screen_sliver.py 상위 4개 + 비교용 @timing. 여기에 @cumul100을 10개로 줄인 세트를 더했다.
  @cumul100은 그룹 permutation에서 2순위(-2.6%p)인데 세트에 넣으면 떨어진다(@timing 10.4% → +@cumul100 7.9%).
  피처 100개가 많아서인지 정보가 겹쳐서인지 가르기 위해, 10% 간격 10개만 남긴 세트를 같이 잰다.
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
SEEDS = [0, 1, 2]
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')


def main(path):
    df = pd.read_csv(path)
    df = df[(df.part < ld.JOIN_PART) & ~ld.sliver_is_test(df.group.values)].reset_index(drop=True)
    names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    df[names] = df[names].astype(np.float32)

    G = ex.groups(names)
    new = {p[:-1]: [f for f in names if f.startswith(p)] for p in TIME_NEW}
    timenew = sum(new.values(), [])
    # @cumul100 100개 중 10% 간격 10개만 (cumul100_010, 020, ... 100)
    cumul10 = [f for f in G['cumul100'] if int(f.rsplit('_', 1)[1]) % 10 == 0]
    sets = {
        '@timing + 새 시간': G['timing'] + timenew,
        '새 시간만': timenew,
        '@timing @cumul100 @count @burst + 새 시간': G['timing'] + G['cumul100'] + G['count'] + G['burst'] + timenew,
        '@timing + tnorm만': G['timing'] + new['tnorm'],
        '@timing + cumul100 10개 + 새 시간': G['timing'] + cumul10 + timenew,
        '@timing (비교용)': G['timing'],
    }

    X, y, g = df[names].values, df.label.values, df.group.values
    ug = np.unique(g)
    folds = [(np.isin(g, ug[tr]), np.isin(g, ug[va])) for tr, va in GroupKFold(5).split(ug, groups=ug)]
    col = {f: names.index(f) for f in names}
    print(f'train {len(df)}행, 피처 {len(names)}개, 사이트 {len(set(y))}개, 방문 {len(ug)}개')
    print(f'fold별 검증 행 수: {[int(va.sum()) for _, va in folds]}\n', flush=True)

    acc = {}  # (세트, 시드) → fold 5개 정확도
    t0 = time.time()
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
            print(f'  {k:42s} ({len(fs):3d}개)  {np.mean(v):.4f} ± {np.std(v):.4f}'
                  f'  {np.round(v, 4)}  ({time.time() - t0:.0f}s)', flush=True)

    print('\n===== 정리: 세트별 시드 평균과 변동폭 =====')
    print(f'{"세트":42s} {"개수":>4s} {"평균":>7s} {"fold표준편차":>11s} {"시드변동폭":>9s}')
    rows = []
    for k, fs in sets.items():
        done = [acc[(k, s)] for s in SEEDS if (k, s) in acc]
        if not done:
            continue
        means = [np.mean(v) for v in done]
        rows.append((np.mean(means), k, len(fs), np.mean([np.std(v) for v in done]), max(means) - min(means), len(done)))
    for mean, k, n, fstd, span, n_seed in sorted(rows, reverse=True):
        print(f'{k:42s} {n:4d} {mean:7.4f} {fstd:11.4f} {span:9.4f}  (시드 {n_seed}개)')

    print('\n== 판단 ==')
    rows.sort(reverse=True)
    if len(rows) >= 2:
        top, second = rows[0], rows[1]
        gap = top[0] - second[0]
        print(f'1등 {top[1]} ({top[2]}개) {top[0]:.4f}')
        print(f'2등 {second[1]} ({second[2]}개) {second[0]:.4f}')
        print(f'차이 {gap:.4f} / 1등의 fold 표준편차 {top[3]:.4f} / 시드 변동폭 {top[4]:.4f}')
        print('→ 차이가 fold 표준편차와 시드 변동폭보다 크면 1등을 후보로 쓴다. 아니면 피처가 적은 쪽을 쓴다.'
              if gap > max(top[3], top[4]) else
              '→ 차이가 fold 표준편차/시드 변동폭 안이다. 두 세트를 같은 수준으로 보고 피처가 적은 쪽을 쓴다.')
    print('DONE', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
