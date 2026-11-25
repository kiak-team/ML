"""open-world 실험 러너 (B 파트). 결과는 results/openworld_results.csv에 한 줄씩 누적한다.

사용법 (레포 루트에서, stdout만 로그로):
  python -I -W ignore -X utf8 code/ow_run.py e1 --subset   > logs/ow_e1_subset.log   # 파이프라인 검증 (mon 25 / unmon 5000)
  python -I -W ignore -X utf8 code/ow_run.py e1            > logs/ow_e1.log

데이터 (pkl)
  피처 = features/ow_skip14_*.csv (앞 14셀 버림, 376개. reports/openworld.md 0절)
  split = splits/pkl_*.csv (reports/openworld.md 1절). 평가는 train 내부 5-fold CV의 out-of-fold 예측. test는 쓰지 않는다
  --subset: mon site < 25, unmon row < 5000. TrafficSliver 축소판(mon_25 / unmon_5000)과 같은 규모의 pkl 부분집합
  피처 행렬은 cache/pkl_skip14_*.npy로 캐싱한다 (git 제외)

출력
  results/openworld_results.csv  고정 컬럼 COLUMNS (C팀과 공유. 바꾸지 말 것)
  results/confusion/<exp_id>.csv confusion matrix (행 = 실제, 열 = 예측, 첫 행/열 = -1)
  cache/oof/<exp_id>.npz         out-of-fold predict_proba (E2 임계값·E3 sweep에서 재학습 없이 쓴다)
"""
import argparse, json, os, sys, time

import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ow_metrics import UNMON, apply_threshold, ow_metrics, unmon_confusion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results', 'openworld_results.csv')
CACHE = os.path.join(ROOT, 'cache')
COLUMNS = ['exp_id', 'dataset', 'split_mode', 'feature_set', 'model', 'imbalance', 'threshold', 'n_classes',
           'TPR', 'FPR', 'precision', 'recall', 'macro_f1', 'notes']
RANDOM_STATE = 42
SPLIT_MODES = ('visit', 'subpage')


# ---------------- 데이터 ----------------
def _cached_matrix(stem):
    """features/<stem>.csv → (X float32, 피처 이름). cache/<stem>.npy에 캐싱."""
    npy, js = os.path.join(CACHE, f'{stem}.npy'), os.path.join(CACHE, f'{stem}.names.json')
    csv = os.path.join(ROOT, 'features', f'{stem}.csv')
    if os.path.exists(npy) and os.path.getmtime(npy) >= os.path.getmtime(csv):
        return np.load(npy), json.load(open(js))
    df = pd.read_csv(csv)
    names = [c for c in df.columns if c not in ('site', 'url', 'inst')]
    X = df[names].to_numpy(np.float32)
    os.makedirs(CACHE, exist_ok=True)
    np.save(npy, X); json.dump(names, open(js, 'w'))
    if 'site' in df:
        np.save(os.path.join(CACHE, f'{stem}.ids.npy'), df[['site', 'url', 'inst']].to_numpy())
    return X, names


def load_pkl(skip=14, subset=False):
    """mon + unmon을 이어 붙인 dict. 행 순서 = [mon 19,000 (CSV 순서), unmon 10,000 (CSV 순서)]."""
    Xm, names = _cached_matrix(f'ow_skip{skip}_mon')
    Xu, names_u = _cached_matrix(f'ow_skip{skip}_unmon')
    assert names == names_u, 'mon/unmon 피처 열이 다르다'
    ids = np.load(os.path.join(CACHE, f'ow_skip{skip}_mon.ids.npy'))
    sm = pd.read_csv(os.path.join(ROOT, 'splits', 'pkl_mon.csv'))
    su = pd.read_csv(os.path.join(ROOT, 'splits', 'pkl_unmon.csv'))
    assert (sm[['site', 'url', 'inst']].to_numpy() == ids).all(), 'split 파일과 피처 CSV의 행이 어긋난다'
    assert len(su) == len(Xu) and (su.row.values == np.arange(len(Xu))).all()
    d = {
        'X': np.vstack([Xm, Xu]), 'names': names,
        'y': np.concatenate([sm.site.values, np.full(len(Xu), UNMON)]),
        'is_test': np.concatenate([sm.is_test.values, su.is_test.values]),
        'fold_visit': np.concatenate([sm.fold_visit.values, su.fold.values]),
        'fold_subpage': np.concatenate([sm.fold_subpage.values, su.fold.values]),
        'row': np.concatenate([sm.row.values, su.row.values]),
    }
    if subset:
        keep = np.concatenate([sm.site.values < 25, su.row.values < 5000])
        d = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    d['dataset'] = 'pkl_m25_u5000' if subset else 'pkl'
    return d


# ---------------- 실험 ----------------
def rf():
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)


def cv_oof(X, y, fold, make_model):
    """fold >= 0인 행(train)에서 5-fold out-of-fold predict_proba. 반환: (proba, classes, fold별 시간)."""
    tr = fold >= 0
    Xt, yt, ft = X[tr], y[tr], fold[tr]
    classes = np.unique(yt)
    proba = np.zeros((len(yt), len(classes)), np.float32)
    secs = []
    for k in range(5):
        t0 = time.time()
        m = clone(make_model()).fit(Xt[ft != k], yt[ft != k])
        assert (m.classes_ == classes).all(), f'fold {k}: train에 빠진 클래스가 있다'
        proba[ft == k] = m.predict_proba(Xt[ft == k])
        secs.append(time.time() - t0)
        print(f'    fold {k}: train {int((ft != k).sum())} / val {int((ft == k).sum())}  {secs[-1]:.0f}s', flush=True)
    return proba, classes, secs


def fold_spread(y, pred, fold):
    """fold별 TPR·FPR의 표준편차 (판단 기준용)."""
    per = [ow_metrics(y[fold == k], pred[fold == k]) for k in range(5)]
    return np.std([p['TPR'] for p in per]), np.std([p['FPR'] for p in per])


def record(row):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    row = {c: row.get(c, '') for c in COLUMNS}
    for c in ('TPR', 'FPR', 'precision', 'recall', 'macro_f1'):
        row[c] = round(float(row[c]), 6)
    new = not os.path.exists(RESULTS)
    if not new:
        assert list(pd.read_csv(RESULTS, nrows=0).columns) == COLUMNS, '결과 CSV 컬럼이 고정 컬럼과 다르다'
    pd.DataFrame([row], columns=COLUMNS).to_csv(RESULTS, mode='a', header=new, index=False)


def evaluate(exp_id, d, split_mode, proba, classes, threshold, feature_set, model, imbalance, extra_notes=''):
    fold = d[f'fold_{split_mode}']
    tr = fold >= 0
    y, f = d['y'][tr], fold[tr]
    pred = apply_threshold(proba, classes, 0.0 if threshold == '' else threshold)
    r = ow_metrics(y, pred, labels=classes)
    sd_tpr, sd_fpr = fold_spread(y, pred, f)
    cm, labels, cs = unmon_confusion(y, pred)
    os.makedirs(os.path.join(ROOT, 'results', 'confusion'), exist_ok=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(ROOT, 'results', 'confusion', f'{exp_id}.csv'))
    notes = (f'CV5 OOF train only (mon {r["n_mon"]}, unmon {r["n_unmon"]}); '
             f'decision={"argmax" if threshold == "" else "max proba < t -> -1"}; '
             f'TP {r["TP"]}, wrong_site {r["wrong_site"]} ({r["wrong_site_rate"]:.4f}), FN {r["FN"]}, FP {r["FP"]}; '
             f'det_TPR {r["det_TPR"]:.4f}; fold std TPR {sd_tpr:.4f} FPR {sd_fpr:.4f}' + extra_notes)
    record({'exp_id': exp_id, 'dataset': d['dataset'], 'split_mode': split_mode, 'feature_set': feature_set,
            'model': model, 'imbalance': imbalance, 'threshold': threshold, 'n_classes': len(classes),
            'TPR': r['TPR'], 'FPR': r['FPR'], 'precision': r['precision'], 'recall': r['recall'],
            'macro_f1': r['macro_f1'], 'notes': notes})
    print(f'  [{exp_id}] TPR {r["TPR"]:.4f}  FPR {r["FPR"]:.4f}  precision {r["precision"]:.4f}  '
          f'macro_f1 {r["macro_f1"]:.4f}  | det_TPR {r["det_TPR"]:.4f}  wrong_site {r["wrong_site"]} '
          f'({r["wrong_site_rate"]:.2%})  FN {r["FN"]}  FP {r["FP"]}  | fold std TPR {sd_tpr:.4f} FPR {sd_fpr:.4f}')
    print(f'      -1 행(실제 unmon {r["n_unmon"]}): -1로 {cs["unmon_as_unmon"]}, mon으로 {cs["unmon_as_mon"]} '
          f'(FP가 많은 사이트 {cs["top_FP_sites"]})')
    print(f'      -1 열(-1로 예측된 mon): {cs["mon_as_unmon"]} = FN', flush=True)
    return r


def save_oof(exp_id, d, split_mode, proba, classes):
    os.makedirs(os.path.join(CACHE, 'oof'), exist_ok=True)
    fold = d[f'fold_{split_mode}']
    tr = fold >= 0
    np.savez_compressed(os.path.join(CACHE, 'oof', f'{exp_id}.npz'), proba=proba, classes=classes,
                        y=d['y'][tr], fold=fold[tr], row=d['row'][tr])


def e1(subset):
    """E1 베이스라인: (N+1) 클래스, 전체 피처, RF, 불균형 처리 없음, argmax 판정."""
    d = load_pkl(14, subset)
    n_mon = len(np.unique(d['y'][d['y'] != UNMON]))
    print(f'E1 dataset={d["dataset"]}: 피처 {len(d["names"])}개, monitored 사이트 {n_mon}개(데이터에서 셈) + unmon -1 '
          f'= {n_mon + 1}클래스, 행 {len(d["y"])} (test {int(d["is_test"].sum())}개는 쓰지 않음)', flush=True)
    tag = '-subset' if subset else ''
    for mode in SPLIT_MODES:
        exp_id = f'E1{tag}-{mode}'
        print(f'\n## {exp_id}', flush=True)
        proba, classes, secs = cv_oof(d['X'], d['y'], d[f'fold_{mode}'], rf)
        assert len(classes) == n_mon + 1
        save_oof(exp_id, d, mode, proba, classes)
        evaluate(exp_id, d, mode, proba, classes, '', f'all{len(d["names"])}_skip14',
                 'RF(n=200,rs=42)', 'none', f'; fit+predict {sum(secs):.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['e1'])
    ap.add_argument('--subset', action='store_true')
    a = ap.parse_args()
    {'e1': e1}[a.exp](a.subset)
