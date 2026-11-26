"""피처와 모델을 골라 직접 실험한다 (sklearn 필요).

기본은 train 내부 5-fold 교차검증(CV)이다. test는 --test를 줄 때만 쓴다.
피처를 고르는 동안에는 CV 점수만 보고, 최종 조합이 정해지면 test를 한 번만 본다.

명령줄:
  python -I -X utf8 code/experiment.py --list                         # 피처·그룹·모델 목록
  python -I -X utf8 code/experiment.py @all                           # 55개 전체, 기본 모델 rf
  python -I -X utf8 code/experiment.py @order @burst -in_burst_1 -m knn
  python -I -X utf8 code/experiment.py n_total frac_out first30_out -m svm -p C=10
  python -I -X utf8 code/experiment.py @all -m rf --test              # 최종 1회 평가

노트북:
  import sys; sys.path.insert(0, 'code')
  from experiment import run
  run(['@order', 'n_total'], model='knn', n_neighbors=3)

피처 지정:
  이름       그 피처를 추가 (예: first30_out)
  @그룹      그룹 전체를 추가 (@all @count @timing @order @burst @cumul @length @const)
             features_extra.py를 실행했으면 @early @cumulabs @bseq @cumul100 @extra 도 쓸 수 있다
             features_extra2.py를 실행했으면 @bseqlong @dir @dir50 @qconst 도 쓸 수 있다
             (@all은 기존 55개만 뜻한다)
  -이름/-@그룹  앞에서 고른 것에서 제외 (예: @all -@timing)

  @qconst (13개)는 mon에서 준상수라 @early/@bseq/@dir50에서 빼 두었다. groups()의 설명 참고.
  이전 로그(stage3/3b/4)의 피처 수를 재현하려면 맨 뒤에 @qconst를 붙인다.

분할:
  test = URL마다 방문 2개 (load_data.mon_is_test, 1,900개), train = 나머지 17,100개. 11/15에 바꿨다.
  이전(10/12~11/14)에는 test = inst 18~19였다. 마지막 방문(inst 19)이 짧고 느려서 test가 CV보다 8.6%p 낮았고,
  CV로 고른 설정이 test에 그대로 옮겨 간다고 볼 수 없어서 바꿨다(docs/TEST_SPLIT.md). 이전 로그의 CV 점수는 옛 분할 기준이다.
  CV fold = inst % 5. URL마다 test 방문 2개는 inst % 5가 같아서, fold 5개의 크기는 모두 3,420개로 같다.
  --test는 test 중 inst 19(95개)의 정확도를 따로 출력한다. 수집 상태가 다른 방문에서의 성능으로 보고서에 함께 쓴다.

전처리 (모델이 자동으로 정한다. --log/--no-log, --scale/--no-scale로 바꿀 수 있다):
  트리 모델(dt, rf, et): 그대로 쓴다. 트리는 값의 크기에 영향을 받지 않는다.
  그 외(lr, nb, svm, knn, mlp): sign(x)·log(1+|x|) 후 표준화. 표준화 기준은 학습 데이터로만 계산한다.
  --gw: 표준화 뒤 그룹마다 1/√(그룹 크기)를 곱한다. 후보 131개에서 k-NN +1.4%p, SVM +0.4%p (EXPERIMENTS.md 6c단계).
        LR에는 쓰지 않는다(−1.9%p. 값이 작아져 L2 정규화가 사실상 세진다).
        예) python -I -X utf8 code/experiment.py @all -@const @early @bseq @cumulabs @dir50 -m knn --gw

결과는 logs/experiments.csv에 한 줄씩 쌓인다.
"""
import argparse, csv, os, sys, time
from datetime import datetime

import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import PredefinedSplit, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, 'features', 'features_mon.csv')
EXTRA_PATH = os.path.join(ROOT, 'features', 'features_extra_mon.csv')  # features_extra.py 출력 (있으면 붙인다)
EXTRA2_PATH = os.path.join(ROOT, 'features', 'features_extra2_mon.csv')  # features_extra2.py 출력 (있으면 붙인다)
LOG_PATH = os.path.join(ROOT, 'logs', 'experiments.csv')

# 수업에서 다룬 모델 (slide 16). 기본 하이퍼파라미터는 -p 키=값으로 바꾼다.
MODELS = {
    'lr':  (lambda: LogisticRegression(max_iter=2000), False),
    'nb':  (lambda: GaussianNB(), False),
    'svm': (lambda: SVC(C=10, gamma='scale'), False),
    'dt':  (lambda: DecisionTreeClassifier(random_state=0), True),
    'rf':  (lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0), True),
    'et':  (lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=0), True),
    'knn': (lambda: KNeighborsClassifier(n_neighbors=5), False),
    'mlp': (lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, early_stopping=True, random_state=0), False),
}

_data = None


def load():
    global _data
    if _data is None:
        df = pd.read_csv(CSV_PATH)
        names = [c for c in df.columns if c not in ('site', 'url', 'inst')]
        for path in (EXTRA_PATH, EXTRA2_PATH):
            if not os.path.exists(path):
                continue
            ex = pd.read_csv(path)
            assert (ex[['site', 'url', 'inst']].values == df[['site', 'url', 'inst']].values).all(), '행 순서 불일치'
            extra = [c for c in ex.columns if c not in ('site', 'url', 'inst')]
            df = pd.concat([df, ex[extra]], axis=1)
            names += extra
        _data = df, names
    return _data


def groups(names, pkl_qconst=True):
    """피처 이름 → 그룹 딕셔너리.

    pkl_qconst=True(기본)는 pkl 데이터에서 구한 준상수 제외를 적용한다. 그 목록은
    "모든 기록이 같은 핸드셰이크 14셀로 시작한다"는 pkl의 성질에서 나온 것이다.

    TrafficSliver 데이터에는 그 성질이 없다. BWR이 셀 묶음을 경로 5개에 뿌려서 조각마다
    시작 지점이 달라, 같은 피처들이 전혀 상수가 아니다(train split에서 mode_frac을 재면
    dir_001~014는 0.70~0.76, bseq_1은 0.049로 고유값이 3,929개다. 준상수 기준은 0.9998).
    그래서 sliver 쪽 스크립트는 pkl_qconst=False로 불러, 실재하는 피처 24개가 조용히
    빠지는 것을 막는다(@bseq 30, @dir 100, @dir50 50, @early 7, @qconst 없음).
    """
    base = names[:55]  # features_mon.csv의 55개. 추가 피처는 @extra 등으로 따로 고른다
    g = {
        'all': base,
        'count': ['n_total', 'n_in', 'n_out', 'frac_out'],
        'timing': [f for f in base if f.startswith(('duration', 'cells_per', 't_', 'iat', 'pps'))],
        'order': [f for f in base if f.startswith(('out_order', 'in_order', 'first', 'last30', 'conc'))],
        'burst': [f for f in base if 'burst' in f],
        'cumul': [f for f in base if f.startswith('cumul')],
    }
    # feature_importance.log에서 서로 상관이 높아 한 묶음으로 나온 "기록 길이" 계열 17개
    g['length'] = ['n_total', 'n_in', 'out_order_std', 'in_order_mean', 'in_order_std', 'in_burst_ge10',
                   'cumul_min'] + [f'cumul_{p}0pct' for p in range(1, 11)]
    # 모든 기록에서 (거의) 상수 — pkl 기준. sliver에서는 변하므로 비운다
    g['const'] = ['in_burst_1', 'in_burst_2', 'in_burst_3'] if pkl_qconst else []
    # features_extra.py의 추가 피처 (가설은 그 파일 맨 위 설명 참고)
    extra = names[55:]
    if extra:
        lo = 7 if pkl_qconst else 1  # pkl에서는 bseq_1~6, dir_001~014가 준상수다
        g['early'] = [f for f in extra if f.startswith('first')
                      and not (pkl_qconst and f == 'first10_out')]  # first10_out은 pkl에서 준상수
        g['cumulabs'] = [f for f in extra if f.startswith('cumul_at_')]
        g['bseq'] = [f'bseq_{j}' for j in range(lo, 31) if f'bseq_{j}' in extra]
        g['cumul100'] = [f for f in extra if f.startswith('cumul100_')]
        g['extra'] = g['early'] + g['cumulabs'] + g['bseq'] + g['cumul100']
        # features_extra2.py의 추가 피처
        if 'bseq_31' in extra:
            g['bseqlong'] = [f'bseq_{j}' for j in range(31, 101)]
            d0 = 15 if pkl_qconst else 1  # dir_001~014는 pkl에서 준상수
            g['dir'] = [f'dir_{j:03d}' for j in range(d0, 101)]
            g['dir50'] = [f'dir_{j:03d}' for j in range(d0, 51)]
        # mon에서 한 값이 99.98% 이상인 준상수 13개. 위 그룹에서 빼 두고 이름만 남긴다.
        # closed-world(95-class)에서는 정보가 없다: train-only η² = 0.0054(우연 수준 0.005),
        # 빼도 k-NN은 소수점 4자리까지 같고 RF는 -0.03%p다. 대신 RF가 1.6배 빨라진다.
        # 주의: unmon에서는 값이 변하고(일부는 다른 상수: bseq_6 mon 2 / unmon 1),
        #       이 13개만으로 mon/unmon이 AUC 0.9999로 갈린다. 수집 환경 차이로 보이지만
        #       2026-11-26 수업 공지로 원인이 확정됐다: unmon을 Colab 메모리 때문에 줄이는 과정에서
        #       이 패턴이 우연히 생겼다. 우연이므로 이 패턴에 기댄 피처·규칙은 일반화되지 않는다.
        #       open-world에서는 쓰지 않는 쪽이 맞다.
        # 이전 로그(stage3/3b/4) 재현: @all -@const @early @bseq @dir50 @qconst → 131개
        g['qconst'] = [f for f in ['first10_out'] + [f'bseq_{j}' for j in range(4, 7)]
                       + [f'dir_{j:03d}' for j in range(6, 15)] if f in extra] if pkl_qconst else []
    return g


def select(spec, names):
    """['@order', 'n_total', '-in_burst_1'] → 피처 이름 리스트 (순서 유지, 중복 제거)."""
    g = groups(names)
    chosen = []

    def expand(tok):
        if tok.startswith('@'):
            if tok[1:] not in g:
                raise SystemExit(f'알 수 없는 그룹: {tok} (가능: {", ".join("@" + k for k in g)})')
            return g[tok[1:]]
        if tok not in names:
            raise SystemExit(f'알 수 없는 피처: {tok} (--list로 확인)')
        return [tok]

    for tok in spec:
        if tok.startswith('-'):
            drop = set(expand(tok[1:]))
            chosen = [f for f in chosen if f not in drop]
        else:
            chosen += [f for f in expand(tok) if f not in chosen]
    if not chosen:
        raise SystemExit('선택된 피처가 없다.')
    return chosen


def signed_log(X):
    return np.sign(X) * np.log1p(np.abs(X))


GW_ORDER = ['count', 'timing', 'order', 'burst', 'cumul', 'early', 'bseq', 'cumulabs', 'cumul100',
            'bseqlong', 'dir', 'qconst', 'const']


def group_weights(feats, names):
    """피처마다 1/√(그 피처가 속한 그룹의 선택된 피처 수). 그룹이 없는 피처는 혼자 한 그룹이다.

    k-NN·SVM·LR은 모든 피처를 똑같이 쓰므로 @dir50(36개)처럼 큰 그룹이 거리를 독차지한다.
    그룹마다 비중을 맞추면 k-NN +1.4%p, SVM +0.4%p (6c단계, 5개 fold 전부 상승). LR은 −1.9%p라 쓰지 않는다.
    """
    owner = {}
    for k in GW_ORDER:
        for f in groups(names).get(k, []):
            owner.setdefault(f, k)
    key = [owner.get(f, f) for f in feats]
    return np.array([1 / np.sqrt(key.count(k)) for k in key])


class GroupWeight(TransformerMixin, BaseEstimator):
    """표준화 뒤에 열마다 고정 가중치를 곱한다. 데이터로 학습하는 것은 없다."""
    def __init__(self, w=None):
        self.w = w

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X * self.w


def build(model, params, log=None, scale=None, weights=None):
    """weights: 열마다 곱할 가중치 (표준화 뒤). group_weights()의 결과를 넣는다. 트리 모델에는 의미가 없다."""
    make, is_tree = MODELS[model]
    est = make().set_params(**params)
    log = (not is_tree) if log is None else log
    scale = (not is_tree) if scale is None else scale
    steps = ([FunctionTransformer(signed_log)] if log else []) + ([StandardScaler()] if scale else []) \
        + ([GroupWeight(weights)] if weights is not None else []) + [est]
    return make_pipeline(*steps), log, scale


def run(features=('@all',), model='rf', test=False, log=None, scale=None, note='', save=True, gw=False, **params):
    """피처 지정과 모델로 실험하고 결과 dict를 반환한다. 노트북에서 쓰기 위한 함수.

    gw=True: 그룹 가중치(group_weights)를 표준화 뒤에 곱한다. k-NN·SVM용 (LR은 손해).
    """
    df, names = load()
    if isinstance(features, str):
        features = [features]
    feats = select(features, names)
    pipe, log, scale = build(model, params, log, scale, group_weights(feats, names) if gw else None)
    if gw:
        params = {**params, 'gw': True}  # 출력과 experiments.csv에 남기기 위해

    is_te = ld.mon_is_test(df.url.values, df.inst.values)
    tr, te = df[~is_te], df[is_te]
    Xtr, ytr = tr[feats].values, tr.site.values

    t0 = time.time()
    cv = cross_val_score(pipe, Xtr, ytr, cv=PredefinedSplit(tr.inst.values % 5), n_jobs=1)
    res = {'cv_mean': cv.mean(), 'cv_std': cv.std(), 'test_acc': None}
    if test:
        fitted = clone(pipe).fit(Xtr, ytr)
        hit = fitted.predict(te[feats].values) == te.site.values
        res['test_acc'], res['test_acc_inst19'] = hit.mean(), hit[te.inst.values == 19].mean()
    res['sec'] = time.time() - t0

    print(f'model={model} {params or ""} log={log} scale={scale}  features={len(feats)}')
    print(f'  CV 5-fold (train 내부): {res["cv_mean"]:.4f} ± {res["cv_std"]:.4f}   '
          f'[{" ".join(f"{a:.3f}" for a in cv)}]')
    if test:
        print(f'  TEST (URL마다 2개, 1,900개): {res["test_acc"]:.4f}   (그중 inst 19, 95개: {res["test_acc_inst19"]:.4f})'
              '   ← 최종 평가용. 이걸 보고 피처를 다시 고르지 말 것')
    print(f'  {res["sec"]:.1f}s')

    if save:
        new = not os.path.exists(LOG_PATH)
        with open(LOG_PATH, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['time', 'model', 'params', 'log', 'scale', 'n_features', 'spec', 'features',
                            'cv_mean', 'cv_std', 'test_acc', 'sec', 'note'])
            w.writerow([datetime.now().strftime('%Y-%m-%d %H:%M'), model,
                        ' '.join(f'{k}={v}' for k, v in params.items()), log, scale, len(feats),
                        ' '.join(features), ' '.join(feats), f'{res["cv_mean"]:.4f}', f'{res["cv_std"]:.4f}',
                        '' if res['test_acc'] is None else f'{res["test_acc"]:.4f}', f'{res["sec"]:.1f}', note])
    res['features'] = feats
    return res


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.startswith('(') and v.endswith(')'):  # 예: hidden_layer_sizes=(128,64)
        return tuple(int(x) for x in v[1:-1].split(',') if x.strip())
    return {'True': True, 'False': False, 'None': None}.get(v, v)


def main():
    ap = argparse.ArgumentParser(description='피처·모델을 골라 CV로 실험한다. 자세한 사용법은 파일 맨 위 설명 참고.')
    ap.add_argument('features', nargs='*', help='피처 이름, @그룹, -제외 (예: @all -@timing)')
    ap.add_argument('-m', '--model', default='rf', choices=MODELS)
    ap.add_argument('-p', '--param', action='append', default=[], help='하이퍼파라미터 키=값 (여러 번 가능)')
    ap.add_argument('--test', action='store_true', help='test 1,900개로도 평가 (최종 1회만)')
    ap.add_argument('--log', dest='log', action='store_true', default=None)
    ap.add_argument('--no-log', dest='log', action='store_false')
    ap.add_argument('--scale', dest='scale', action='store_true', default=None)
    ap.add_argument('--no-scale', dest='scale', action='store_false')
    ap.add_argument('--gw', action='store_true', help='그룹 가중치: 큰 피처 그룹이 거리를 독차지하지 않게 한다 (k-NN·SVM용, LR은 손해)')
    ap.add_argument('--note', default='', help='experiments.csv에 남길 메모')
    ap.add_argument('--list', action='store_true', help='피처·그룹·모델 목록 출력')
    # '-in_burst_1', '-@timing' 같은 제외 표기를 argparse가 옵션으로 오인하지 않도록 미리 분리한다
    opts = {o for act in ap._actions for o in act.option_strings}
    takes_value = {'-m', '--model', '-p', '--param', '--note'}
    argv, spec, it = [], [], iter(sys.argv[1:])
    for tok in it:
        if tok in opts or tok.split('=', 1)[0] in opts:
            argv.append(tok)
            if tok in takes_value:
                argv.append(next(it, ''))
        else:
            spec.append(tok)
    a = ap.parse_args(argv)
    a.features = spec or ['@all']

    if a.list:
        _, names = load()
        for k, v in groups(names).items():
            print(f'@{k} ({len(v)}): {" ".join(v)}\n')
        print('models:', ' '.join(MODELS))
        return
    params = dict(kv.split('=', 1) for kv in a.param)
    params = {k: parse_value(v) for k, v in params.items()}
    run(a.features, a.model, a.test, a.log, a.scale, a.note, gw=a.gw, **params)


if __name__ == '__main__':
    main()
