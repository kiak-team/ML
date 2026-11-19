"""Extra credit ④: 최종 202개 세트에서 모델을 바꿔 본다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/models_sliver.py > logs/models_sliver.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

여기까지 RF(200) 하나만 썼다. 피처 조합 탐색은 수확이 줄어드는 구간이다
(logs/verify_sliver_sets3.log: @count 4개를 더해도, tnqin 20개를 빼도 0.04%p 안이다).
남은 지렛대는 모델 쪽이므로 수업에서 다룬 모델(slide 16)을 같은 fold에서 비교한다.

세트는 verify_sliver_sets2.py가 확정한 202개, fold도 같은 GroupKFold(5) 고정이다.
기준선: RF(200) 5-fold 15.98% (logs/verify_sliver_sets2.log)

- 트리가 아닌 모델은 fold의 train으로만 StandardScaler를 맞춘다(검증 fold를 보면 누수다).
- SVC는 뺐다. 커널 SVM은 행 수에 제곱으로 들어서 78,331행 × 95클래스는 끝나지 않는다.
- 싼 모델부터 돌려서 느린 모델이 남아도 앞쪽 결과는 쓸 수 있게 한다.
- 1단계는 시드 0, 2단계는 기준선보다 높은 상위 2개에만 시드 1·2를 더한다.
"""
import os, sys, time

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex

TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')
FINAL_EXTRA = ('tnout_', 'tnqin_', 'tnqout_')

# 이름: (만드는 함수(시드), 트리인가)  — 트리가 아니면 StandardScaler를 끼운다
MODELS = {
    'nb':    (lambda s: GaussianNB(), False),
    'dt':    (lambda s: DecisionTreeClassifier(random_state=s), True),
    'et200': (lambda s: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=s), True),
    'rf200': (lambda s: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=s), True),
    'knn5':  (lambda s: KNeighborsClassifier(n_neighbors=5, n_jobs=-1), False),
    'et600': (lambda s: ExtraTreesClassifier(n_estimators=600, n_jobs=-1, random_state=s), True),
    'lr':    (lambda s: LogisticRegression(max_iter=2000, n_jobs=-1), False),
    'mlp':   (lambda s: MLPClassifier(hidden_layer_sizes=(256,), max_iter=500,
                                      early_stopping=True, random_state=s), False),
}
BASE = 'rf200'


def main():
    from short_splits import load  # 같은 합치기·행 걸러내기를 쓴다
    df, names = load()
    G = ex.groups(names)
    fs = G['timing'] + [f for f in names if f.startswith(TIME_NEW + FINAL_EXTRA)]
    assert len(fs) == 202, len(fs)
    X, y, g = df[fs].values, df.label.values, df.group.values
    ug = np.unique(g)
    folds = [(np.isin(g, ug[tr]), np.isin(g, ug[va])) for tr, va in GroupKFold(5).split(ug, groups=ug)]
    print(f'train {len(df)}행, 최종 세트 {len(fs)}개, 사이트 95개')
    print(f'모델 {len(MODELS)}개: ' + ' '.join(MODELS) + f'  (기준선 {BASE})\n', flush=True)

    t0, acc = time.time(), {}

    def run(k, seed):
        make, is_tree = MODELS[k]
        v = []
        for mtr, mva in folds:
            xtr, xva = X[mtr], X[mva]
            if not is_tree:  # 스케일러는 train fold로만 맞춘다
                sc = StandardScaler().fit(xtr)
                xtr, xva = sc.transform(xtr), sc.transform(xva)
            m = make(seed)
            m.fit(xtr, y[mtr])
            v.append(m.score(xva, y[mva]))
        acc[(k, seed)] = v
        print(f'  {k:7s}  {np.mean(v):.4f} ± {np.std(v):.4f}  {np.round(v, 4)}'
              f'  ({time.time() - t0:.0f}s)', flush=True)
        return np.mean(v)

    print('===== 1단계: 시드 0 =====', flush=True)
    m0 = {}
    for k in MODELS:
        try:
            m0[k] = run(k, 0)
        except Exception as e:  # 한 모델이 터져도 나머지는 쓴다
            print(f'  {k:7s}  실패: {type(e).__name__}: {e}', flush=True)

    ref = m0.get(BASE, 0.0)
    cand = sorted((v, k) for k, v in m0.items() if k != BASE and v > ref)[-2:]
    cand = [k for _, k in reversed(cand)]
    print(f'\n기준선 {BASE} {ref:.4f}보다 높은 상위 2개: {cand if cand else "없음"}', flush=True)

    print('\n===== 2단계: 기준선 + 후보에 시드 1·2 =====', flush=True)
    for seed in (1, 2):
        print(f'--- 시드 {seed} ---', flush=True)
        for k in [BASE] + cand:
            run(k, seed)

    print('\n===== 정리 =====')
    print(f'{"모델":8s} {"평균":>7s} {"fold표준편차":>11s} {"시드변동폭":>9s} {"시드":>5s}')
    rows = []
    for k in MODELS:
        done = [acc[(k, s)] for s in (0, 1, 2) if (k, s) in acc]
        if not done:
            continue
        means = [np.mean(v) for v in done]
        rows.append((np.mean(means), k, np.mean([np.std(v) for v in done]),
                     max(means) - min(means), len(done)))
    for mean, k, fstd, span, ns in sorted(rows, reverse=True):
        print(f'{k:8s} {mean:7.4f} {fstd:11.4f} {span:9.4f} {ns:5d}')

    print('\n== 판단 ==')
    r = {x[1]: x for x in rows}.get(BASE)
    if r is None:
        print('기준선이 돌지 않았다.'); return
    lim = max(r[2], r[3])
    best = max((x for x in rows if x[1] != BASE), key=lambda x: x[0])
    gap = best[0] - r[0]
    print(f'기준선 {BASE} {r[0]:.4f} (fold 표준편차 {r[2]:.4f}, 시드 변동폭 {r[3]:.4f}) → 유의 기준 {lim:.4f}')
    print(f'최고 {best[1]} {best[0]:.4f}, 차이 {gap:+.4f}')
    print(f'→ {best[1]}로 바꾼다.' if gap > lim else f'→ 변동폭 안이다. {BASE}를 유지한다.')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
