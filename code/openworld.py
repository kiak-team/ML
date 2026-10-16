"""open-world 실험 (sklearn 필요). experiment.py의 피처 지정·모델을 그대로 쓴다.

open-world: 감시 대상(mon) 95개 사이트 + 감시 대상이 아닌 사이트(unmon). 과제 기준 mon = positive, unmon = negative.
평가 지표(과제 18쪽): TPR, FPR, precision, PR curve, ROC.

명령줄:
  python -I -X utf8 code/openworld.py @all -@const @early @bseq @dir50                  # 후보 131개, rf, 두 방식
  python -I -X utf8 code/openworld.py @all -@const @early @bseq @dir50 -@timing --mode multi
  python -I -X utf8 code/openworld.py @all -@const @early @bseq @dir50 --skip 14       # 앞 14셀을 버린 피처
  python -I -X utf8 code/openworld.py @all -@const @early @bseq @dir50 --test           # 최종 1회 평가

--skip K: features_ow.py가 만든 features/ow_skip<K>_*.csv를 쓴다(기록의 앞 K셀을 버리고 다시 계산한 피처).
  기본(0)은 기존 CSV다. mon과 unmon은 앞 14셀에서 수집 환경 차이가 드러나므로 open-world에는 --skip을 쓴다
  (docs/OPENWORLD.md).

방식 (--mode):
  binary  mon(1) vs unmon(0) 이진 분류. 점수 = mon일 확률
  multi   96클래스(사이트 0~94 + unmon -1) 분류. 점수 = 사이트 95개 중 가장 높은 확률, 예측 사이트 = 그 사이트
  점수가 임계값 이상이면 mon으로 판정한다. 임계값을 바꾸면 TPR과 FPR이 함께 바뀐다(ROC, PR 곡선).

라벨·분할:
  mon   라벨 = 사이트 번호 0~94. train = inst 0~17 (17,100개), test = inst 18~19 (1,900개). experiment.py와 같다
  unmon 라벨 = -1. 기록마다 다른 사이트라 무작위로 나눈다(시드 0): test 1,000개, train 9,000개
        → test의 unmon은 학습 때 보지 못한 사이트다
  CV fold: mon = inst % 5, unmon = train 안 순번 % 5. 기본은 train 내부 CV의 예측(out-of-fold)으로 평가한다

지표:
  TPR       mon 중 mon으로 판정한 비율
  FPR       unmon 중 mon으로 잘못 판정한 비율
  precision mon으로 판정한 것 중 실제 mon 비율
  (multi만) TPR_site, precision_site: 사이트까지 맞혀야 정답으로 센다. 다른 감시 사이트로 틀리게 맞힌 것은 오답
  ROC-AUC, PR-AUC(average precision): 임계값 전체에 대한 요약
  주의: mon:unmon = 17,100:9,000으로 mon이 더 많다. 현실(감시 대상은 극소수)과 반대라 precision이 실제보다 높게 나온다
"""
import argparse, os, sys, time, warnings

import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex

warnings.filterwarnings('ignore', message='`sklearn.utils.parallel.delayed`')  # sklearn 1.9 + Python 3.14 경고. 결과와 무관

UNMON_CSVS = ['features_unmon.csv', 'features_extra_unmon.csv', 'features_extra2_unmon.csv']
N_UNMON_TEST = 1000
SEED = 0

_cache = {}


def load_unmon():
    """unmon 피처 CSV를 mon과 같은 열 순서로 붙여 읽는다."""
    if 'unmon' not in _cache:
        parts = [pd.read_csv(os.path.join(ex.ROOT, 'features', f)) for f in UNMON_CSVS
                 if os.path.exists(os.path.join(ex.ROOT, 'features', f))]
        u = pd.concat(parts, axis=1)
        _, names = ex.load()
        assert list(u.columns) == names, 'mon/unmon 피처 열이 다르다'
        _cache['unmon'] = u
    return _cache['unmon']


def load(skip=0):
    """(mon DataFrame, 피처 이름, unmon DataFrame). skip > 0이면 앞 skip셀을 버린 피처."""
    if skip == 0:
        df, names = ex.load()
        return df, names, load_unmon()
    if skip not in _cache:
        f = os.path.join(ex.ROOT, 'features', f'ow_skip{skip}_%s.csv')
        if not os.path.exists(f % 'mon'):
            raise SystemExit(f'{f % "mon"} 없음. features_ow.py <pkl 폴더> features {skip} 를 먼저 실행')
        df, u = pd.read_csv(f % 'mon'), pd.read_csv(f % 'unmon')
        names = [c for c in df.columns if c not in ('site', 'url', 'inst')]
        assert list(u.columns) == names == ex.load()[1], '피처 열이 기존 CSV와 다르다'
        _cache[skip] = df, names, u
    return _cache[skip]


def split(skip=0):
    """(train DataFrame, test DataFrame). label 열 = 사이트 번호 또는 -1, fold 열 = CV fold."""
    df, names, u = load(skip)
    u = u.copy()
    perm = np.random.default_rng(SEED).permutation(len(u))
    u_te, u_tr = u.iloc[perm[:N_UNMON_TEST]].copy(), u.iloc[perm[N_UNMON_TEST:]].copy()
    u_tr['fold'] = np.arange(len(u_tr)) % 5
    for x in (u_te, u_tr):
        x['label'] = -1
    m = df.copy(); m['label'] = m.site; m['fold'] = m.inst % 5
    tr = pd.concat([m[m.inst <= 17], u_tr], ignore_index=True)
    te = pd.concat([m[m.inst >= 18], u_te], ignore_index=True)
    return tr, te


def scores(proba, classes, mode):
    """predict_proba 결과 → (점수, 예측 사이트). binary면 예측 사이트는 None."""
    if mode == 'binary':
        return (proba if proba.ndim == 1 else proba[:, list(classes).index(1)]), None  # 1차원 = SVM 결정 함수
    mon_cols = [i for i, c in enumerate(classes) if c != -1]
    p = proba[:, mon_cols]
    return p.max(axis=1), np.asarray(classes)[mon_cols][p.argmax(axis=1)]


def at(t, s, y, pred):
    """임계값 t에서의 지표."""
    is_mon, flag = y != -1, s >= t
    r = {'t': t, 'TPR': (flag & is_mon).sum() / is_mon.sum(), 'FPR': (flag & ~is_mon).sum() / (~is_mon).sum(),
         'precision': (flag & is_mon).sum() / max(flag.sum(), 1)}
    if pred is not None:
        hit = flag & is_mon & (pred == y)
        r['TPR_site'] = hit.sum() / is_mon.sum()
        r['precision_site'] = hit.sum() / max(flag.sum(), 1)
    return r


def report(s, y, pred):
    is_mon = y != -1
    out = {'ROC-AUC': roc_auc_score(is_mon, s), 'PR-AUC': average_precision_score(is_mon, s), 'rows': []}
    u = np.sort(s[~is_mon])
    out['rows'].append(('t=0.5', at(0.5, s, y, pred)))
    for fpr in (0.05, 0.01):
        # unmon 점수의 (1-fpr) 분위수보다 큰 점수만 mon으로 판정하면 FPR ≤ fpr
        t = np.nextafter(u[int(np.ceil(len(u) * (1 - fpr))) - 1], np.inf)
        out['rows'].append((f'FPR≤{fpr:.0%}', at(t, s, y, pred)))
    return out


def run(features, model='rf', mode='binary', test=False, skip=0, **params):
    _, names, _ = load(skip)
    feats = ex.select(features, names)
    pipe, log, scale = ex.build(model, params)
    # SVM은 확률(probability=True)을 켜면 10배 이상 느려서 결정 함수 값을 점수로 쓴다.
    # ROC·PR과 FPR 기준 임계값은 점수의 순서만 쓰므로 그대로 비교할 수 있다. 단 t=0.5 행은 의미가 없다
    method = 'decision_function' if model == 'svm' else 'predict_proba'
    tr, te = split(skip)
    ytr = tr.label.values if mode == 'multi' else (tr.label.values != -1).astype(int)

    t0 = time.time()
    if test:
        fitted = clone(pipe).fit(tr[feats].values, ytr)
        proba, classes, y = getattr(fitted, method)(te[feats].values), fitted.classes_, te.label.values
    else:
        proba = cross_val_predict(pipe, tr[feats].values, ytr, cv=PredefinedSplit(tr.fold.values),
                                  method=method, n_jobs=1)
        classes, y = np.unique(ytr), tr.label.values
    s, pred = scores(proba, classes, mode)
    r = report(s, y, pred)
    if not test:  # fold별 ROC-AUC의 표준편차 (판단 기준용)
        r['auc_std'] = np.std([roc_auc_score(y[f] != -1, s[f]) for f in (tr.fold.values == k for k in range(5))])

    where = f'TEST (mon {(te.label != -1).sum()}, unmon {(te.label == -1).sum()})' if test else \
        f'CV 5-fold (train: mon {(tr.label != -1).sum()}, unmon {(tr.label == -1).sum()})'
    print(f'mode={mode} skip={skip} model={model} {params or ""} log={log} scale={scale} features={len(feats)}  [{where}]')
    print(f'  ROC-AUC {r["ROC-AUC"]:.4f}' + (f' (fold std {r["auc_std"]:.4f})' if 'auc_std' in r else '') +
          f'   PR-AUC {r["PR-AUC"]:.4f}')
    for name, m in r['rows']:
        line = f'  {name:<8} TPR {m["TPR"]:.4f}  FPR {m["FPR"]:.4f}  precision {m["precision"]:.4f}'
        if 'TPR_site' in m:
            line += f'  | TPR_site {m["TPR_site"]:.4f}  precision_site {m["precision_site"]:.4f}'
        print(line)
    print(f'  {time.time() - t0:.1f}s', flush=True)
    r.update(features=feats, score=s, pred=pred, y=y)
    return r


def main():
    ap = argparse.ArgumentParser(description='open-world 실험. 자세한 사용법은 파일 맨 위 설명 참고.')
    ap.add_argument('-m', '--model', default='rf', choices=ex.MODELS)
    ap.add_argument('-p', '--param', action='append', default=[], help='하이퍼파라미터 키=값')
    ap.add_argument('--mode', default='both', choices=['binary', 'multi', 'both'])
    ap.add_argument('--test', action='store_true', help='test로 평가 (최종 1회만)')
    ap.add_argument('--skip', type=int, default=0, help='앞 K셀을 버린 피처 사용 (features_ow.py 출력)')
    # '-@timing' 같은 제외 표기를 옵션으로 오인하지 않도록 분리한다 (experiment.py와 같은 방식)
    opts = {o for act in ap._actions for o in act.option_strings}
    takes_value = {'-m', '--model', '-p', '--param', '--mode', '--skip'}
    argv, spec, it = [], [], iter(sys.argv[1:])
    for tok in it:
        if tok in opts or tok.split('=', 1)[0] in opts:
            argv.append(tok)
            if tok in takes_value:
                argv.append(next(it, ''))
        else:
            spec.append(tok)
    a = ap.parse_args(argv)
    params = {k: ex.parse_value(v) for k, v in (kv.split('=', 1) for kv in a.param)}
    for mode in (['binary', 'multi'] if a.mode == 'both' else [a.mode]):
        run(spec or ['@all'], a.model, mode, a.test, a.skip, **params)


if __name__ == '__main__':
    main()
