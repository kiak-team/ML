"""제안서·보고서용 그래프 (matplotlib 필요). 결과는 figures/*.png.

사용법: python -I -X utf8 code/plots.py data ow stages importance   (원하는 것만 골라도 된다)
  data        데이터 그래프 4개: mon/unmon 길이 분포, 지속 시간 분포, 사이트별 길이, 방문 순서(inst)별 길이
  ow          open-world ROC·PR 곡선 (후보 134개, 앞 14셀 버림, RF 기본값, train 내부 CV 예측)
              + 임계값 표, mon:unmon 비율을 바꿨을 때 precision 표 → logs/ow_curves.log
              처음 실행할 때 RF를 2번 학습한다(약 10분). 점수는 logs/ow_cv_scores.npz에 저장해 다시 쓴다
  stages      피처 실험 결과: 피처셋 × 모델 CV 정확도 (EXPERIMENTS.md 6단계, logs/stage6a.log)
  importance  후보 131개의 그룹별 중요도 (feature_importance_cv.py 출력이 있어야 한다)

모든 점수는 train 내부 CV다. test는 쓰지 않는다.
색: 1 파랑 #2a78d6, 2 주황 #eb6834, 3 청록 #1baf7a, 4 노랑 #eda100, 5 분홍 #e87ba4 (색각 이상 검증 통과).
    주황·청록·노랑·분홍은 배경 대비가 낮아서 범례와 직접 라벨을 함께 둔다.
"""
import logging, os, sys, warnings

import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  # 없는 한글 글꼴 후보 경고
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex

ROOT = ex.ROOT
FIG = os.path.join(ROOT, 'figures')
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
OW_SPEC = ['@all', '@early', '@bseq', '@dir50', '@qconst']  # open-world 후보 134개 (--skip 14)

plt.rcParams.update({
    'font.family': ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10, 'axes.titlesize': 12,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'legend.frameon': False, 'lines.linewidth': 2,
    'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print('저장:', os.path.join('figures', name), flush=True)


def data():
    mon = pd.read_csv(os.path.join(ROOT, 'features', 'features_mon.csv'))
    unmon = pd.read_csv(os.path.join(ROOT, 'features', 'features_unmon.csv'))

    # 1) 길이 분포
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 10001, 250)
    for d, name, c in ((mon, 'mon (감시 대상 95개 사이트, 19,000개)', C[0]), (unmon, 'unmon (그 외 사이트, 10,000개)', C[1])):
        ax.hist(d.n_total, bins=bins, histtype='step', color=c, linewidth=2, density=True, label=name)
    ax.set_title('기록 길이(셀 수) 분포')
    ax.set_xlabel('셀 수 (최대 약 10,000셀에서 잘려 있어 오른쪽 끝에 몰린다)'); ax.set_ylabel('비율 (밀도)')
    ax.set_yticklabels([])
    ax.legend(loc='upper right')
    save(fig, 'data_length_mon_unmon.png')

    # 2) 지속 시간 분포 (log 축)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    lo = max(min(mon.duration[mon.duration > 0].min(), unmon.duration[unmon.duration > 0].min()), 0.1)
    bins = np.logspace(np.log10(lo), np.log10(max(mon.duration.max(), unmon.duration.max())), 50)
    for d, name, c in ((mon, 'mon', C[0]), (unmon, 'unmon', C[1])):
        ax.hist(d.duration.clip(lower=lo), bins=bins, histtype='step', color=c, linewidth=2, density=True,
                label=f'{name} (중앙값 {d.duration.median():.1f}초)')
    ax.set_xscale('log')
    ax.set_title('페이지 로딩 지속 시간 분포')
    ax.set_xlabel('지속 시간 (초, 로그 축)'); ax.set_ylabel('비율 (밀도)'); ax.set_yticklabels([])
    ax.legend(loc='upper left')
    save(fig, 'data_duration_mon_unmon.png')

    # 3) 사이트별 길이: 중앙값과 사분위 범위, 중앙값 순 정렬
    g = mon.groupby('site').n_total
    s = pd.DataFrame({'q1': g.quantile(.25), 'med': g.median(), 'q3': g.quantile(.75)}).sort_values('med')
    x = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.vlines(x, s.q1, s.q3, color=C[0], alpha=0.45, linewidth=2)
    ax.scatter(x, s.med, s=12, color=C[0], zorder=3)
    ax.set_title('사이트별 기록 길이 (95개 사이트, 중앙값 순)')
    ax.set_xlabel('사이트 (중앙값이 짧은 순)'); ax.set_ylabel('셀 수')
    ax.set_xticks([]); ax.set_xlim(-1, len(s))
    ax.text(0.01, 0.95, '점 = 중앙값, 선 = 25~75% 범위 (사이트당 200개 기록)', transform=ax.transAxes,
            color=INK2, va='top', fontsize=9)
    save(fig, 'data_length_by_site.png')

    # 4) 방문 순서(inst)별 길이 — inst 19 문제 (PREPROCESSING.md "마지막 방문")
    g = mon.groupby('inst')
    med, dur = g.n_total.median(), g.duration.median()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, v, title, unit in ((axs[0], med, '방문 순서별 길이 중앙값', '셀 수'),
                               (axs[1], dur, '방문 순서별 지속 시간 중앙값', '초')):
        col = [C[1] if i >= 18 else C[0] for i in v.index]
        ax.bar(v.index, v.values, color=col, width=0.75)
        ax.set_title(title); ax.set_xlabel('방문 순서 (inst)'); ax.set_ylabel(unit)
        ax.set_xticks([0, 5, 10, 15, 19])
        ax.annotate(f'inst 19: {v[19]:,.0f}' if unit == '셀 수' else f'inst 19: {v[19]:.1f}초',
                    (19, v[19]), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8, color=INK)
    fig.tight_layout()
    fig.text(0.01, -0.06, '주황 = 지금의 test 분할(inst 18~19). 마지막 방문(inst 19)이 짧고 느리다', color=INK2, fontsize=9)
    save(fig, 'data_by_inst.png')


def ow_scores():
    path = os.path.join(ROOT, 'logs', 'ow_cv_scores.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import openworld as ow
    out = {}
    for mode in ('binary', 'multi'):
        r = ow.run(OW_SPEC, model='rf', mode=mode, skip=14)
        out[f'{mode}_score'] = r['score']; out['y'] = r['y']
        if r['pred'] is not None:
            out['multi_pred'] = r['pred']
    np.savez(path, **out)
    return out


def ow():
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    z = ow_scores()
    y = z['y']; is_mon = y != -1
    lines = []

    def log(s=''):
        print(s, flush=True); lines.append(s)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (mode, name) in enumerate((('multi', '96클래스 (사이트 95개 + unmon)'), ('binary', '이진 (mon vs unmon)'))):
        s = z[f'{mode}_score']
        fpr, tpr, _ = roc_curve(is_mon, s)
        prec, rec, _ = precision_recall_curve(is_mon, s)
        a1.plot(np.maximum(fpr, 1e-4), tpr, color=C[i], label=f'{name}  AUC {auc(fpr, tpr):.4f}')
        a2.plot(rec, prec, color=C[i], label=name)
        # FPR 1% 지점 표시 (openworld.py와 같은 방식으로 임계값을 정한다)
        u = np.sort(s[~is_mon]); t = np.nextafter(u[int(np.ceil(len(u) * 0.99)) - 1], np.inf)
        flag = s >= t
        tp, fp = (flag & is_mon).sum(), (flag & ~is_mon).sum()
        a1.scatter([max(fp / (~is_mon).sum(), 1e-4)], [tp / is_mon.sum()], s=40, color=C[i], edgecolor=SURF,
                   linewidth=2, zorder=4)
        a2.scatter([tp / is_mon.sum()], [tp / (tp + fp)], s=40, color=C[i], edgecolor=SURF, linewidth=2, zorder=4)
    a1.axvline(0.01, color=INK2, linewidth=1, linestyle=':')
    a1.text(0.0108, 0.985, 'FPR 1%', color=INK2, fontsize=8, va='top')
    a1.set_xscale('log'); a1.set_xlim(1e-4, 1); a1.set_ylim(0.5, 1.005)
    a1.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1]); a1.set_xticklabels(['0.01%', '0.1%', '1%', '10%', '100%'])
    a1.minorticks_off()
    a1.set_title('ROC 곡선'); a1.set_xlabel('FPR: unmon을 mon으로 잘못 판정한 비율 (로그 축)')
    a1.set_ylabel('TPR: mon을 mon으로 잡은 비율'); a1.legend(loc='lower right', fontsize=8.5)
    a2.set_xlim(0.5, 1.005); a2.set_ylim(0.9, 1.002)
    a2.set_title('PR 곡선'); a2.set_xlabel('recall (= TPR)'); a2.set_ylabel('precision: mon 판정 중 실제 mon 비율')
    a2.legend(loc='lower left', fontsize=8.5)
    fig.text(0.01, -0.06, 'open-world, 후보 피처 134개(앞 14셀 버림), RF 기본값, train 내부 5-fold CV 예측 '
             '(mon 17,100 / unmon 9,000). 점 = FPR 1% 임계값', color=INK2, fontsize=8.5)
    save(fig, 'ow_roc_pr.png')

    # 임계값 표 (팀 결정용)
    log('open-world 임계값 표 (96클래스, RF 기본값, 후보 134개 --skip 14, train 내부 CV 예측)')
    log('  FPR 목표   임계값   TPR      TPR_site  FPR      precision  precision_site')
    s, pred = z['multi_score'], z['multi_pred']
    u = np.sort(s[~is_mon])
    rows = []
    for f in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10):
        t = np.nextafter(u[int(np.ceil(len(u) * (1 - f))) - 1], np.inf)
        flag = s >= t
        tp, fp = (flag & is_mon).sum(), (flag & ~is_mon).sum()
        hit = (flag & is_mon & (pred == y)).sum()
        r = dict(f=f, t=t, tpr=tp / is_mon.sum(), fpr=fp / (~is_mon).sum(), tsite=hit / is_mon.sum())
        rows.append(r)
        log(f'  {f:>6.1%}   {t:.3f}   {r["tpr"]:.4f}   {r["tsite"]:.4f}    {r["fpr"]:.4f}   '
            f'{tp / max(tp + fp, 1):.4f}     {hit / max(tp + fp, 1):.4f}')

    # mon:unmon 비율을 바꿨을 때 precision (TPR·FPR은 그대로, 개수만 바꿔 계산)
    log('\nmon:unmon 비율에 따른 precision (위 TPR·FPR로 계산. 실제 데이터는 mon 17,100 : unmon 9,000 ≈ 1.9:1)')
    ratios = [(17100 / 9000, '실험 비율 1.9:1'), (1, '1:1'), (0.1, '1:10'), (0.01, '1:100')]
    log('  FPR 목표   ' + '   '.join(f'{n:>14}' for _, n in ratios))
    for r in rows:
        ps = [r['tpr'] * k / (r['tpr'] * k + r['fpr']) if r['tpr'] * k + r['fpr'] > 0 else 0 for k, _ in ratios]
        log(f'  {r["f"]:>6.1%}   ' + '   '.join(f'{p:>14.4f}' for p in ps))
    log('  → unmon이 mon보다 100배 많은 현실적인 상황에서는 FPR 1%로도 precision이 크게 떨어진다. 보고서 한계에 쓴다')
    with open(os.path.join(ROOT, 'logs', 'ow_curves.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def stages():
    rows = [l.split('\t') for l in open(os.path.join(ROOT, 'logs', 'stage6a.log'), encoding='utf-8-sig')
            if l.startswith('RESULT')]
    d = pd.DataFrame([(r[1].replace('S6a ', ''), r[2], int(r[4]), float(r[5])) for r in rows],
                     columns=['set', 'model', 'n', 'acc'])
    order = ['B', 'C', 'C+cumulabs', 'C+dir50', 'C+cumulabs+dir50', 'C+cumulabs+dir']
    label = {'B': '기존 52', 'C': '+ 앞부분 방향 패턴 (82)', 'C+cumulabs': '+ 누적합 위치 (95)',
             'C+dir50': '+ 처음 50셀 방향 (118)', 'C+cumulabs+dir50': '후보: 둘 다 (131)',
             'C+cumulabs+dir': '처음 100셀까지 (181)'}
    models = [('rf', 'RF'), ('et', 'ET'), ('svm', 'SVM'), ('knn', 'k-NN'), ('lr', 'LR')]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ypos = np.arange(len(order))[::-1]
    for i, (m, name) in enumerate(models):
        v = d[d.model == m].set_index('set').acc.reindex(order) * 100
        off = (i - 2) * 0.12
        ax.plot(v.values, ypos + off, color=C[i], linewidth=1, alpha=0.35)
        ax.scatter(v.values, ypos + off, s=30, color=C[i], edgecolor=SURF, linewidth=1.5, zorder=3,
                   label=f'{name} (후보 {v.values[-2]:.1f})')
    ax.axhspan(ypos[-2] - 0.4, ypos[-2] + 0.4, color=C[0], alpha=0.06, zorder=0)
    ax.set_yticks(ypos); ax.set_yticklabels([label[s] for s in order])
    ax.set_xlim(58, 100); ax.set_xlabel('CV 정확도 (%), train 내부 5-fold, 기본 하이퍼파라미터')
    ax.set_title('피처셋에 따른 closed-world 정확도 (95개 사이트)')
    ax.legend(ncol=5, loc='upper left', bbox_to_anchor=(-0.02, -0.13), fontsize=8.5, columnspacing=1)
    fig.text(0.01, -0.1, '모델은 모두 기본값이라 모델 순위가 아니다. 위아래 비교(피처셋 효과)만 읽는다', color=INK2, fontsize=8.5)
    save(fig, 'stages_featureset_model.png')


def importance():
    path = os.path.join(ROOT, 'logs', 'feature_importance_cv_cand.log')
    if not os.path.exists(path):
        raise SystemExit('logs/feature_importance_cv_cand.log 없음. feature_importance_cv.py cand를 먼저 실행')
    lines = open(path, encoding='utf-8').read().split('== 그룹 permutation')[1].split('==\n', 1)[1].split('\n\n')[0]
    rows = []
    for l in lines.strip().splitlines():
        name, rest = l.split('(', 1)
        n = int(rest.split('개')[0]); mean, std = rest.split(')')[1].split('±')
        rows.append((name.strip().lstrip('@'), n, float(mean) * 100, float(std) * 100))
    d = pd.DataFrame(rows, columns=['group', 'n', 'drop', 'std']).sort_values('drop')
    desc = {'count': '셀 개수', 'timing': '시간', 'order': '순서·위치', 'burst': '버스트', 'cumul': '누적합 분위',
            'early': '앞부분 송신 수', 'bseq': '앞쪽 버스트 길이 순서', 'cumulabs': '누적합 고정 위치',
            'dir50': '처음 50셀 방향'}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(d))
    ax.barh(y, d['drop'], xerr=d['std'], color=C[0], height=0.6, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for yi, v, sd in zip(y, d['drop'], d['std']):
        ax.text(v + sd + 0.12, yi, f'{v:.1f}%p', va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([f'{desc.get(g, g)} @{g} ({n})' for g, n in zip(d.group, d.n)])
    ax.set_xlabel('그룹을 섞었을 때 정확도 하락 (%p), train 내부 5-fold 평균')
    ax.set_title('피처 그룹별 중요도 (후보 131개, RF)')
    ax.set_xlim(0, d['drop'].max() * 1.25)
    save(fig, 'importance_groups.png')


if __name__ == '__main__':
    for job in sys.argv[1:] or ['data', 'stages', 'ow']:
        {'data': data, 'ow': ow, 'stages': stages, 'importance': importance}[job]()
