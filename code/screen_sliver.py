"""Extra credit 빠른 비교: 피처 세트별 split→split 정확도 + open-world 수집 환경 차이 확인 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/screen_sliver.py > logs/screen_sliver.log      (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

- 최종 test(ld.sliver_is_test)는 쓰지 않는다. 나머지 방문 중 20%를 group 단위 검증용으로 뗀 holdout 1번이다.
  5-fold(feature_importance_sliver.py)는 375개에 2시간이 걸려서, 세트 비교는 이걸로 빠르게 하고 최종 후보만 CV로 확인한다.
- 1. closed-world 95-class (split만): 세트별 RF 정확도
- 2. open-world mon vs unmon (split만): AUC와 상위 피처. pkl 데이터에서는 앞 14셀 수집 차이로 AUC 1.0이 나왔다.
     여기서도 한두 피처로 AUC가 1에 가까우면 사이트가 아니라 수집 차이를 맞힌 것이다.
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

ROOT = ex.ROOT
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')


def is_val(group):
    """test가 아닌 방문 중 약 20%. sliver_is_test와 같은 해시의 다른 자리를 쓴다."""
    h = np.asarray(group, dtype=np.uint64) * np.uint64(2654435761) % np.uint64(2**32)
    return (h // 10) % 5 == 0


def load(name):
    df = pd.read_csv(os.path.join(ROOT, 'features', f'sliver_{name}.csv'))
    df = df[(df.part < ld.JOIN_PART) & ~ld.sliver_is_test(df.group.values)].reset_index(drop=True)
    names = [c for c in df.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    df[names] = df[names].astype(np.float32)
    return df, names


def rf(**kw):
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0, **kw)


def main():
    mon, names = load('mon')
    G = ex.groups(names)
    new = {p[:-1]: [f for f in names if f.startswith(p)] for p in TIME_NEW}
    timenew = sum(new.values(), [])
    old = [f for f in names if f not in timenew]
    sets = {
        '기존 전체 (375)': old,
        '@timing': G['timing'],
        '@timing @cumul100': G['timing'] + G['cumul100'],
        '새 시간 피처 (124)': timenew,
        '@timing + 새 시간': G['timing'] + timenew,
        '@timing @cumul100 + 새 시간': G['timing'] + G['cumul100'] + timenew,
        '@timing @cumul100 @count @burst + 새 시간': G['timing'] + G['cumul100'] + G['count'] + G['burst'] + timenew,
        '전체 (500)': names,
    }
    for k, v in new.items():
        sets[f'@timing + 새 {k}만'] = G['timing'] + v

    va = is_val(mon.group.values)
    y = mon.label.values
    print(f'closed-world: train {(~va).sum()}행 / 검증 {va.sum()}행 (split만, 95-class)', flush=True)
    print('\n== 1. 피처 세트별 RF(200) 정확도 ==')
    t0 = time.time()
    for k, fs in sets.items():
        m = rf().fit(mon.loc[~va, fs].values, y[~va])
        print(f'  {k:42s} ({len(fs):3d}개)  {m.score(mon.loc[va, fs].values, y[va]):.4f}  ({time.time() - t0:.0f}s)', flush=True)
        if k == '전체 (500)':
            imp = pd.Series(m.feature_importances_, fs).sort_values(ascending=False)
            print('     상위 15: ' + ', '.join(f'{a}={b:.4f}' for a, b in imp.head(15).items()))

    un, _ = load('unmon')
    X = pd.concat([mon[names], un[names]], ignore_index=True)
    yb = np.r_[np.ones(len(mon)), np.zeros(len(un))]
    vb = np.r_[va, is_val(un.group.values)]
    print(f'\n== 2. open-world mon vs unmon (split만): train {(~vb).sum()} / 검증 {vb.sum()} ==')
    for k in ('전체 (500)', '@timing + 새 시간'):
        fs = sets[k]
        m = rf(class_weight='balanced').fit(X.loc[~vb, fs].values, yb[~vb])
        auc = roc_auc_score(yb[vb], m.predict_proba(X.loc[vb, fs].values)[:, 1])
        imp = pd.Series(m.feature_importances_, fs).sort_values(ascending=False)
        print(f'  {k}: AUC {auc:.4f}')
        print('     상위 10: ' + ', '.join(f'{a}={b:.3f}' for a, b in imp.head(10).items()), flush=True)
    # 피처 하나로 mon/unmon이 갈리는지 (수집 차이 신호): 피처별 단독 AUC 상위
    single = {f: roc_auc_score(yb[~vb], X.loc[~vb, f].values) for f in names}
    single = pd.Series({f: max(a, 1 - a) for f, a in single.items()}).sort_values(ascending=False)
    print('  피처 하나만으로 AUC 상위 10: ' + ', '.join(f'{a}={b:.3f}' for a, b in single.head(10).items()))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
