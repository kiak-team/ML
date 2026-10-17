"""open-world 점수의 출처 분해: ROC-AUC 0.993 중 얼마가 사이트 지문인가.

사용법: python -I -X utf8 code/check_openworld_gap.py > logs/check_openworld_gap.log   (약 10분)
pkl 없이 features/ow_skip14_*.csv만으로 돌아간다. 결과 해석은 docs/OPENWORLD.md 3.10에 있다.

왜 보나: open-world는 "감시 대상인가"를 맞히는 과제다. 그런데 mon과 unmon은 서로 다른 파일이고
수집 조건이 같다는 보장이 없다. 앞 14셀에서 실제로 수집 환경 흔적이 발견됐다(2절). 그렇다면
모델이 사이트 지문이 아니라 "이 기록이 어느 파일에서 왔나"를 맞히고 있을 수 있다.

점검 4개
  [1] 단일 피처 분해: 피처 하나만으로 mon/unmon이 갈리는가 (뚜렷한 범인이 있나)
  [2] 사이트 식별 정보가 없는 피처(개수·시간)만으로 얼마나 갈리는가
  [3] 사이트 홀드아웃: 사이트를 통째로 빼고 학습해도 갈리는가
      -> 유지되면 사이트 지문과 무관한 모집단 차이, 급락하면 95개 사이트를 외운 효과
      평가 사이트가 19개뿐이라 조합을 5번 바꿔 흔들림을 함께 본다
  [4] 절차 점검: unmon을 무작위로 반 갈라 구분시킨다. 0.5가 나와야 파이프라인이 정상이다

주의: [3]의 수치는 성능 추정치가 아니다. 실제 공격자는 감시 대상을 알고 학습하므로
      기존 평가 방식(같은 사이트, 다른 방문)이 맞는 시나리오다. [3]은 "분리 가능성 중
      얼마가 사이트 지문과 무관한가"를 재는 진단값이다.
"""
import sys, os, warnings

warnings.filterwarnings('ignore')
# sklearn 1.9 + Python 3.14에서 joblib 경고가 호출마다 나온다. 결과와 무관하고 로그만 길어진다
warnings.filterwarnings('ignore', message='`sklearn.utils.parallel.delayed`')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import experiment as ex
import openworld as ow

SKIP = 14
N_HOLDOUT_SITES = 19   # 95개 중 평가용으로 뺄 사이트 수 (약 20%)
N_UNMON_EVAL = 1000    # openworld.py의 test용 unmon 수와 같게 맞춘다
N_PARTITIONS = 5
RF = dict(n_estimators=200, n_jobs=-1, random_state=0)


def auc_binary(tr_pos, tr_neg, te_pos, te_neg, feats):
    """mon(1) vs unmon(0) 이진 분류의 평가셋 ROC-AUC."""
    Xtr = np.vstack([tr_pos[feats].values, tr_neg[feats].values]).astype(float)
    ytr = np.r_[np.ones(len(tr_pos)), np.zeros(len(tr_neg))]
    Xte = np.vstack([te_pos[feats].values, te_neg[feats].values]).astype(float)
    yte = np.r_[np.ones(len(te_pos)), np.zeros(len(te_neg))]
    m = RandomForestClassifier(**RF).fit(Xtr, ytr)
    return roc_auc_score(yte, m.predict_proba(Xte)[:, 1])


def main():
    df, names, u = ow.load(SKIP)
    g = ex.groups(names)
    mon = df[df.inst <= 17]
    F_NOSITE = g['count'] + g['timing']                                       # 22개
    F_CAND = ex.select(['@all', '@early', '@bseq', '@dir50', '@qconst'], names)  # 134개 (3.7 후보)
    print(f'mon train {len(mon)}개 (사이트 95개), unmon {len(u)}개 (사이트 {len(u)}개)')
    print(f'피처: 앞 {SKIP}셀을 버리고 다시 계산한 것 (features/ow_skip{SKIP}_*.csv)\n')

    print('=' * 78)
    print('[1] 단일 피처 하나만으로 mon/unmon 가르기 (0.5 = 무정보)')
    print('=' * 78)
    rows = []
    for f in F_NOSITE:
        a, b = mon[f].values.astype(float), u[f].values.astype(float)
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        auc = roc_auc_score(y, np.r_[a, b])
        rows.append((max(auc, 1 - auc), f, np.median(a), np.median(b)))
    rows.sort(reverse=True)
    print(f'{"피처":16s} {"AUC":>7s} {"mon 중앙값":>12s} {"unmon 중앙값":>13s}')
    for auc, f, ma, mb in rows:
        print(f'{f:16s} {auc:7.4f} {ma:12.2f} {mb:13.2f}')
    print(f'\n최대 {rows[0][0]:.4f} ({rows[0][1]}). 뚜렷한 단일 범인은 없다 -> 약한 피처의 조합이다')

    print('\n' + '=' * 78)
    print('[2] 사이트 식별 정보가 없는 피처만으로 (이진, train 내부 CV)')
    print('=' * 78)
    for spec, nm in (([ '@count', '@timing'], '개수 4 + 시간 18 = 22개'),
                     (['@count'], '개수만 4개'),
                     (['@timing'], '시간만 18개')):
        r = ow.run(spec, model='rf', mode='binary', skip=SKIP)
        print(f'>>> {nm}: ROC-AUC {r["ROC-AUC"]:.4f}\n')
    print('기록의 전체 요약(길이·시간)만 알려줘도 상당히 갈린다.')
    print('참고: 시간 피처를 "빼도" 점수가 같다는 것(3.2)은 시간 피처에 정보가 없다는 뜻이 아니다.')
    print('      길이 계열이 같은 정보를 대신 담고 있어서다. 단독으로 보면 위와 같이 드러난다.')

    print('\n' + '=' * 78)
    print(f'[3] 사이트 홀드아웃: 사이트 {N_HOLDOUT_SITES}개를 통째로 빼고 학습')
    print('=' * 78)
    for nm, feats in (('개수+시간 22개', F_NOSITE), ('후보 134개', F_CAND)):
        # 기준: 기존 평가 방식과 같은 구조 (같은 사이트, 다른 방문)
        base = auc_binary(mon[mon.inst <= 13], u.iloc[N_UNMON_EVAL:],
                          mon[mon.inst >= 14], u.iloc[:N_UNMON_EVAL], feats)
        aucs = []
        for seed in range(N_PARTITIONS):
            rng = np.random.default_rng(seed)
            pu = rng.permutation(len(u))
            s_te = set(rng.permutation(95)[:N_HOLDOUT_SITES])
            aucs.append(auc_binary(mon[~mon.site.isin(s_te)], u.iloc[pu[N_UNMON_EVAL:]],
                                   mon[mon.site.isin(s_te)], u.iloc[pu[:N_UNMON_EVAL]], feats))
        print(f'{nm}')
        print(f'  ① 같은 사이트, 다른 방문        AUC {base:.4f}')
        print(f'  ② 본 적 없는 사이트 (조합 {N_PARTITIONS}개)  ' +
              ' '.join(f'{a:.3f}' for a in aucs))
        print(f'     평균 {np.mean(aucs):.4f}, 최소 {min(aucs):.3f}, 최대 {max(aucs):.3f}, '
              f'폭 {max(aucs) - min(aucs):.3f}')
        print(f'  → ①과 ②의 차이 {base - np.mean(aucs):+.4f}'
              f'  (작으면 사이트 지문과 무관한 모집단 차이가 대부분이라는 뜻)\n')

    print('=' * 78)
    print('[4] 절차 점검: unmon을 무작위로 반 갈라 서로 구분 (0.5여야 정상)')
    print('=' * 78)
    rng = np.random.default_rng(0)
    p = rng.permutation(len(u))
    A, B = u.iloc[p[:4500]], u.iloc[p[4500:9000]]
    half = 3500
    print(f'  AUC {auc_binary(A.iloc[:half], B.iloc[:half], A.iloc[half:], B.iloc[half:], F_NOSITE):.4f}')
    print('  0.5 근처면 파이프라인에 가짜 신호가 없다.')


if __name__ == '__main__':
    main()
