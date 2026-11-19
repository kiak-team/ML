"""Extra credit ⑦: TrafficSliver 데이터로 open-world 본 실험 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/openworld_sliver.py [v2] > logs/openworld_sliver.log

  인자 없음: 옛 피처 CSV 두 개를 합쳐 쓴다(4.7 세트). v2: 11/15 정의로 다시 뽑은 CSV 하나(최종 세트).
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

과제 18쪽 지표를 본 프로젝트와 같은 정의로 낸다. 지표 함수(scores/at/report)를 openworld.py에서
그대로 가져와 쓰므로 두 결과를 나란히 놓을 수 있다.
  mon = positive(감시 대상 95개 사이트), unmon = negative
  binary  mon(1) vs unmon(0). 점수 = mon일 확률
  multi   96클래스(사이트 0~94 + unmon -1). 점수 = 사이트 95개 중 최고 확률, 예측 사이트 = 그 사이트
          TPR_site: 사이트까지 맞혀야 정답. 다른 감시 사이트로 틀리면 오답
  임계값: t=0.5, 그리고 unmon 점수 분위수로 잡은 FPR≤5%, FPR≤1%

누수 방지
  mon과 unmon은 group 번호가 겹치므로 unmon에 OFFSET을 더해 분리한다. 더하지 않으면
  서로 다른 기록이 같은 방문으로 취급돼 분할이 깨진다.
  test는 양쪽 모두 `ld.sliver_is_test(group)`로 뗀다. unmon은 기록마다 다른 사이트라
  test의 unmon은 학습 때 보지 못한 사이트가 된다.

pkl과 다른 점
  - 앞 셀을 잘라내지 않는다. 4.5에서 수집 환경 차이가 없음을 확인했다(피처 하나 AUC 최대 0.555).
    pkl은 앞 14셀만으로 AUC 1.0이 나와서 잘라내야 했다.
  - mon:unmon이 약 1:1이다. 현실(감시 대상은 극소수)과 달라 precision이 실제보다 높게 나온다.
    ROC-AUC와 FPR 기준 TPR은 비율에 둔감하지만 PR-AUC·precision은 그렇지 않다.
  - multi는 클래스가 96개여서 메모리가 크다(4.9: 노드 수 × 클래스 × 8바이트). 나무를 몇 그루까지
    올릴 수 있는지 20그루로 재 보고 예산 안에서 정한다.
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld
from openworld import at, report, scores  # 지표 정의를 본 프로젝트와 공유한다

N_TREES = 200
SEEDS_BIN = (0, 1, 2)
MEM_BUDGET_GB = 8.0  # 이 머신 여유가 12GB 안팎이다 (4.9)
OFFSET = 10 ** 7     # unmon group 번호를 mon과 분리
V2 = 'v2' in sys.argv[1:]
V2_TIME = ('tcum_', 'tbin_', 'gap_', 'tnorm_')  # 11/15 정의: tnorm이 40등분 수신·송신 80개
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')
FINAL_EXTRA = ('tnout_', 'tnqin_', 'tnqout_')


def load_side(tag):
    f = os.path.join(ex.ROOT, 'features')
    if V2:
        df = pd.read_csv(os.path.join(f, f'sliver_{tag}_v2.csv'))
    else:
        a = pd.read_csv(os.path.join(f, f'sliver_{tag}.csv'))
        b = pd.read_csv(os.path.join(f, f'sliver_{tag}_time2.csv')).drop(columns=['label'])
        df = a.merge(b, on=['group', 'part'], how='inner', validate='one_to_one')
        assert len(df) == len(a), f'{tag} (group, part) 합치기 실패: {len(df)} != {len(a)}'
    return df.reset_index(drop=True)  # split과 join 모두 (join은 방어 없음 비교용)


def main():
    m, u = load_side('mon'), load_side('unmon')
    names = [c for c in m.columns if c not in ('label', 'group', 'part', 'n_bursts_total')]
    assert [c for c in u.columns if c not in ('label', 'group', 'part', 'n_bursts_total')] == names, \
        'mon/unmon 피처 열이 다르다'
    fs = ex.groups(names, pkl_qconst=False)['timing'] + [
        f for f in names if f.startswith(V2_TIME if V2 else TIME_NEW + FINAL_EXTRA)]
    assert len(fs) == 202, len(fs)
    print(f'세트: {"11/15 정의 (tnorm 40등분 수신·송신)" if V2 else "11/19 정의"}', flush=True)

    u = u.copy()
    u['label'] = -1
    u['group'] = u.group.values + OFFSET
    cols = ['label', 'group', 'part'] + fs
    df = pd.concat([m[cols], u[cols]], ignore_index=True)
    df[fs] = df[fs].astype(np.float32)

    y, g = df.label.values, df.group.values
    is_split = df.part.values != ld.JOIN_PART
    is_test = ld.sliver_is_test(g)
    X = df[fs].values
    gtr, gte = set(g[~is_test]), set(g[is_test])
    assert not (gtr & gte), f'누수: 방문 {len(gtr & gte)}개가 train과 test 양쪽에 있다'
    print(f'split: mon {((y != -1) & is_split).sum()}행 / unmon {((y == -1) & is_split).sum()}행')
    print(f'join : mon {((y != -1) & ~is_split).sum()}행 / unmon {((y == -1) & ~is_split).sum()}행')
    print(f'피처 {len(fs)}개, 방문 train {len(gtr)}개 / test {len(gte)}개 — 겹치는 방문 0개')
    print(f'mon 비율 {(y[is_split] != -1).mean():.3f} (현실은 훨씬 낮다. precision 해석에 주의)\n',
          flush=True)

    t0 = time.time()

    def show(tag, s, yy, pred):
        r = report(s, yy, pred)
        print(f'  {tag}: ROC-AUC {r["ROC-AUC"]:.4f}  PR-AUC {r["PR-AUC"]:.4f}')
        for name, d in r['rows']:
            extra = (f'  TPR_site {d["TPR_site"]:.4f}  precision_site {d["precision_site"]:.4f}'
                     if 'TPR_site' in d else '')
            print(f'    {name:9s} TPR {d["TPR"]:.4f}  FPR {d["FPR"]:.4f}'
                  f'  precision {d["precision"]:.4f}{extra}')
        print(f'    ({time.time() - t0:.0f}s)', flush=True)
        return r

    ybin = (y != -1).astype(int)

    def scenario(name, rows, seeds):
        """rows = split(방어 있음) 또는 join(방어 없음). 같은 코드로 둘을 재서 방어 효과만 남긴다."""
        tr, te = rows & ~is_test, rows & is_test
        print(f'########## {name} (train {tr.sum()}행 / test {te.sum()}행) ##########', flush=True)

        print('--- binary (mon vs unmon) ---', flush=True)
        aucs = []
        for seed in seeds:
            f = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
            f.fit(X[tr], ybin[tr])
            s, _ = scores(f.predict_proba(X[te]), f.classes_, 'binary')
            aucs.append(show(f'시드 {seed}', s, y[te], None)['ROC-AUC'])
        if len(aucs) > 1:
            print(f'  ROC-AUC 시드 평균 {np.mean(aucs):.4f}, 변동폭 {max(aucs) - min(aucs):.4f}',
                  flush=True)

        print('--- multi (96클래스) ---', flush=True)
        probe = RandomForestClassifier(n_estimators=20, n_jobs=4, random_state=0).fit(X[tr], y[tr])
        nodes = np.mean([t.tree_.node_count for t in probe.estimators_])
        per_gb = nodes * len(probe.classes_) * 8 / 1e9
        n_tree = min(N_TREES, max(20, int(MEM_BUDGET_GB / per_gb)))
        print(f'  나무당 노드 {nodes:,.0f}개 × 클래스 {len(probe.classes_)}개 → {per_gb * 1000:.0f}MB/그루')
        print(f'  예산 {MEM_BUDGET_GB}GB → 나무 {n_tree}그루'
              f'{" (200그루는 %.0fGB로 넘친다)" % (per_gb * N_TREES) if n_tree < N_TREES else ""}',
              flush=True)
        f = RandomForestClassifier(n_estimators=n_tree, n_jobs=-1, random_state=0)
        f.fit(X[tr], y[tr])
        s, pred = scores(f.predict_proba(X[te]), f.classes_, 'multi')
        show(f'시드 0 ({n_tree}그루)', s, y[te], pred)
        print('  주의: 96클래스에서는 최고 확률이 0.5를 넘는 일이 거의 없어 t=0.5 행은 의미가 없다.'
              ' FPR 기준 행만 쓴다.\n', flush=True)

    scenario('split → split (방어 있음) — 우리 결과', is_split, SEEDS_BIN)
    scenario('join → join (방어 없음) — 같은 데이터에서 방어만 끈 비교', ~is_split, (0,))

    print('두 시나리오는 같은 데이터·같은 피처·같은 분할이므로 차이는 방어 때문이다.')
    print('본 프로젝트(pkl, 방어 없음)는 binary ROC-AUC 0.9932, FPR≤1%에서 TPR 0.786이었다'
          ' (logs/ow_modes.log). 데이터셋이 달라 직접 비교가 아니라 참고용이다.', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
