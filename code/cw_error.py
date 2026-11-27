"""closed-world 오류 분석: 남은 오류가 어디서 나오는지 (train OOF만 사용, test는 쓰지 않는다).

사용법:
  python -I -W ignore -X utf8 code/cw_error.py                       # cw_ensemble.json의 멤버로 stacking
  python -I -W ignore -X utf8 code/cw_error.py --members lgbm_tuned et_base rf_base df_base knn_tuned lr_base --method stack

출력:
  화면: ① 앙상블로 더 올릴 여지(오라클) ② 방문 번호별 정확도 ③ 기록 길이·시간별 정확도
        ④ 가장 틀리는 사이트 ⑤ 가장 많이 헷갈리는 사이트 쌍 ⑥ 모델 종류별로 누가 무엇을 맞히나
  figures/cw_errors.png, results/cw_errors_site.csv, results/cw_errors_pairs.csv
"""
import argparse, json, os, sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cw_tuning as cw
import experiment as ex
import load_data as ld


def nested_pred(P, y, fold, method, stack_c):
    pred = np.zeros_like(y)
    for k in range(5):
        tr, va = fold != k, fold == k
        Ptr, Pva = [p[tr] for p in P], [p[va] for p in P]
        w = cw.fit_weights(Ptr, y[tr], seed=k) if method == 'wvote' else None
        meta = cw.fit_stack(Ptr, y[tr], stack_c) if method == 'stack' else None
        pred[va] = cw.combine(method, Pva, w, meta).argmax(1)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', nargs='+')
    ap.add_argument('--method', default='stack', choices=['mean', 'geo', 'wvote', 'stack'])
    ap.add_argument('--stack-c', type=float, default=1.0)
    a = ap.parse_args()

    o = cw.load_oofs()
    members = a.members or json.load(open(os.path.join(cw.RES, 'cw_ensemble.json'), encoding='utf-8'))['members']
    any_name = members[0]
    z = np.load(os.path.join(cw.OOF_DIR, f'{any_name}.npz'))
    order = np.argsort(z['url'].astype(np.int64) * 20 + z['inst'])
    url, inst = z['url'][order].astype(int), z['inst'][order].astype(int)
    fold = inst % 5
    y = o[any_name][0]
    P = [o[n][2] for n in members]
    ens = nested_pred(P, y, fold, a.method, a.stack_c)
    best_single = max(members, key=lambda n: (o[n][1] == y).mean())
    single = o[best_single][1]
    print(f'앙상블({a.method}, {len(members)}개): {(ens == y).mean():.4f}   단일 최고 {best_single}: {(single == y).mean():.4f}')
    print(f'오류 {int((ens != y).sum())}개 / {len(y)}개\n')

    # ① 오라클: 멤버 중 하나라도 맞힌 비율 = 섞어서 얻을 수 있는 이론상 최대
    preds = np.array([o[n][1] for n in members])
    any_ok = (preds == y).any(0)
    top3 = np.argsort(-np.array([p for p in P]).mean(0), axis=1)[:, :3]
    in_top3 = (top3 == y[:, None]).any(1)
    print('① 더 올릴 여지')
    print(f'  멤버 중 하나라도 맞힘(오라클)          {any_ok.mean():.4f}   ← 섞는 방법을 아무리 잘 만들어도 이 위로는 못 간다')
    print(f'  평균 확률 상위 3개 안에 정답            {in_top3.mean():.4f}')
    print(f'  모든 멤버가 틀림                        {(~any_ok).mean():.4f}  ({int((~any_ok).sum())}개) ← 새 정보(입력·모델)가 필요한 부분\n')

    # ② 방문 번호별
    df_feat, _ = ex.load()
    is_te = ld.mon_is_test(df_feat.url.values, df_feat.inst.values)
    tr = df_feat[~is_te].sort_values(['url', 'inst']).reset_index(drop=True)
    assert (tr.url.values == url).all() and (tr.inst.values == inst).all()
    by_inst = pd.DataFrame({'inst': inst, 'ens': ens == y, 'single': single == y}).groupby('inst').mean()
    print('② 방문 번호(inst)별 정확도 (앙상블 / 단일)')
    print('  ' + '  '.join(f'{i}:{r.ens * 100:.1f}' for i, r in by_inst.iterrows()))
    rest = by_inst.drop(index=19).ens.mean()
    print(f'  inst 19: 앙상블 {by_inst.ens[19] * 100:.1f}% / 단일 {by_inst.single[19] * 100:.1f}%   (나머지 평균 {rest * 100:.1f}%)')
    share19 = ((ens != y) & (inst == 19)).sum() / max((ens != y).sum(), 1)
    print(f'  전체 오류 중 inst 19 비율: {share19 * 100:.1f}%  (inst 19는 train의 {np.mean(inst == 19) * 100:.1f}%)\n')

    # ③ 길이·시간
    q = pd.qcut(tr.n_total, 5, labels=['아주 짧음', '짧음', '보통', '김', '아주 김'])
    d = pd.qcut(tr.duration, 5, labels=['아주 빠름', '빠름', '보통', '느림', '아주 느림'])
    print('③ 기록 길이(셀 수) 5분위별 앙상블 정확도: ' +
          '  '.join(f'{k} {v * 100:.1f}' for k, v in pd.Series(ens == y).groupby(q.values, observed=True).mean().items()))
    print('   로딩 시간 5분위별 앙상블 정확도:      ' +
          '  '.join(f'{k} {v * 100:.1f}' for k, v in pd.Series(ens == y).groupby(d.values, observed=True).mean().items()))
    trunc = tr.n_total.values >= 9900
    print(f'   약 10,000셀에서 잘린 기록({trunc.mean() * 100:.1f}%): {np.mean((ens == y)[trunc]) * 100:.1f}% / '
          f'나머지 {np.mean((ens == y)[~trunc]) * 100:.1f}%\n')

    # ④ 사이트별
    site = pd.DataFrame({'site': y, 'ok': ens == y}).groupby('site').ok.agg(['mean', 'size'])
    site['wrong'] = ((1 - site['mean']) * site['size']).round().astype(int)
    site = site.sort_values('mean')
    print('④ 정확도가 가장 낮은 사이트 10개 (사이트 번호: 정확도, 틀린 수/180)')
    print('  ' + '  '.join(f'{s}: {r["mean"] * 100:.0f}% ({int(r.wrong)})' for s, r in site.head(10).iterrows()))
    print(f'  정확도 100%인 사이트: {(site["mean"] == 1).sum()}개 / 95개.  하위 10개 사이트가 전체 오류의 '
          f'{site.head(10).wrong.sum() / max(site.wrong.sum(), 1) * 100:.0f}%\n')

    # ⑤ 헷갈리는 쌍
    pairs = Counter((int(t), int(p)) for t, p in zip(y, ens) if t != p)
    sym = Counter()
    for (t, p), c in pairs.items():
        sym[tuple(sorted((t, p)))] += c
    print('⑤ 가장 많이 헷갈리는 사이트 쌍 (양방향 합)')
    print('  ' + '  '.join(f'{a_}↔{b_}: {c}' for (a_, b_), c in sym.most_common(10)))
    top_share = sum(c for _, c in sym.most_common(10)) / max(sum(sym.values()), 1)
    print(f'  상위 10쌍이 전체 오류의 {top_share * 100:.0f}%\n')

    # ⑥ 모델별로 혼자만 맞히는 기록
    print('⑥ 다른 멤버가 모두 틀렸는데 이 모델만 맞힌 기록 수 (클수록 대체 불가)')
    for i, n in enumerate(members):
        others = np.delete(preds, i, axis=0)
        only = ((preds[i] == y) & ~(others == y).any(0)).sum()
        print(f'  {n:14s} {only:4d}')

    # 저장
    site.to_csv(os.path.join(cw.RES, 'cw_errors_site.csv'), encoding='utf-8-sig')
    pd.DataFrame([dict(site_a=a_, site_b=b_, count=c) for (a_, b_), c in sym.most_common(50)]) \
        .to_csv(os.path.join(cw.RES, 'cw_errors_pairs.csv'), index=False, encoding='utf-8-sig')

    import plots as Pl
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    a1.plot(by_inst.index, by_inst.single * 100, marker='o', color=Pl.C[0], alpha=0.7, label=f'단일 최고 ({best_single})')
    a1.plot(by_inst.index, by_inst.ens * 100, marker='s', color=Pl.INK, lw=2.4, label=f'앙상블 ({a.method})')
    a1.axvspan(18.5, 19.5, color='#f3e1d6', zorder=0)
    a1.set_xticks(range(20)); a1.set_xlabel('방문 번호 (inst)'); a1.set_ylabel('정확도 (%)')
    a1.set_title('방문 번호별 정확도 (train OOF)'); a1.legend(fontsize=8.5, loc='lower left')
    a2.bar(range(10), site.head(10)['mean'] * 100, color=Pl.C[1])
    a2.set_xticks(range(10), [str(s) for s in site.head(10).index], fontsize=8.5)
    a2.set_xlabel('사이트 번호'); a2.set_ylabel('정확도 (%)'); a2.set_title('가장 어려운 사이트 10개 (앙상블)')
    a2.set_ylim(min(site.head(10)['mean'] * 100) - 5, 100)
    Pl.save(fig, 'cw_errors.png')


if __name__ == '__main__':
    main()
