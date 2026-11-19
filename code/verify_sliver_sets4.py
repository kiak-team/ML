"""11/15 세트와 11/19 세트를 같은 fold에서 맞대결한다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/verify_sliver_sets4.py [시드...] > logs/verify_sliver_sets4.log
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

이어서 돌리기
  시드를 인자로 줄 수 있다. 예: `... verify_sliver_sets4.py 1 2 >> logs/verify_sliver_sets4.log`
  fold(GroupKFold를 정렬된 고유 group에 적용)도 RF(random_state)도 고정이고 행 필터·float32 변환·
  merge 순서까지 결정적이라, 나눠 돌려도 한 번에 돌린 것과 값이 같다. 머신 속도는 걸리는 시간만 바꾼다.
  주의: 피처 열 순서가 바뀌면 RF가 분기마다 뽑는 후보 순서가 바뀌어 값이 조금 움직인다
  (같은 세트가 logs/verify_sliver_sets2.log에서 0.1599, 여기서 0.1606이다. 변동폭 안이지만
  로그 사이의 숫자를 섞지 말고 한 실행 안에서 비교해야 한다).

왜 (docs/EXTRA_CREDIT.md 6절)
  202개짜리 세트가 두 개다. 개수가 우연히 같고 정확도도 비슷한데, 측정이 달라서 직접 비교할 수 없다.
    11/15 (5.1)  @tnorm을 40등분 수신·송신 80개로 교체        holdout 16.46% / 16.57%
    11/19 (4.7)  @tnorm 20등분 수신 + 송신 쪽 60개 추가       group 5-fold 15.98%, 최종 test 16.51%
  4.6에서 holdout이 5-fold보다 0.1%p 가량 후하고 0.5%p 안쪽 차이는 못 가린다는 것을 확인했다.
  그래서 같은 GroupKFold(5) 고정 fold에서 시드 0·1·2로 둘을 나란히 잰다.

세트 (공통 = @timing 18 + tcum 38 + tbin 40 + gap 26 = 122개)
  기준선 142 = 공통 + tnorm_in 20등분 20
  11/15  202 = 공통 + tfin 40등분 수신 40 + tfout 40등분 송신 40
  11/19  202 = 기준선 142 + tnout 20 + tnqin 20 + tnqout 20
  합집합 282 = 11/15 202 + tnout + tnqin + tnqout  (둘을 합치면 더 오르는지)

tfin/tfout은 features_time3.py가 팀원 공식 그대로 뽑은 것이다(값 일치를 assert로 확인했다).
"""
import os, sys, time

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
SEEDS = (0, 1, 2)
COMMON = ('tcum_', 'tbin_', 'gap_')


def load():
    f = os.path.join(ex.ROOT, 'features')
    a = pd.read_csv(os.path.join(f, 'sliver_mon.csv'))
    b = pd.read_csv(os.path.join(f, 'sliver_mon_time2.csv')).drop(columns=['label'])
    c = pd.read_csv(os.path.join(f, 'sliver_mon_time3.csv')).drop(columns=['label'])
    df = a.merge(b, on=['group', 'part'], how='inner', validate='one_to_one')
    df = df.merge(c, on=['group', 'part'], how='inner', validate='one_to_one')
    assert len(df) == len(a), f'합치기 실패: {len(df)} != {len(a)}'
    df = df[(df.part < ld.JOIN_PART) & ~ld.sliver_is_test(df.group.values)].reset_index(drop=True)
    names = [x for x in df.columns if x not in ('label', 'group', 'part', 'n_bursts_total')]
    df[names] = df[names].astype(np.float32)
    return df, names


def main(seeds=SEEDS):
    df, names = load()
    pre = lambda *ps: [x for x in names if x.startswith(ps)]
    common = ex.groups(names, pkl_qconst=False)['timing'] + pre(*COMMON)
    tn20, tfin, tfout = pre('tnorm_'), pre('tfin_'), pre('tfout_')
    new60 = pre('tnout_') + pre('tnqin_') + pre('tnqout_')
    sets = {
        '기준선 142': common + tn20,
        '11/15 세트 (40등분 수신·송신)': common + tfin + tfout,
        '11/19 세트 (20등분 + 송신 60)': common + tn20 + new60,
        '합집합 (11/15 + 송신 60)': common + tfin + tfout + new60,
    }
    print(f'train {len(df)}행, 피처 {len(names)}개')
    print(f'공통 {len(common)}개 (@timing {len(common) - len(pre(*COMMON))} + tcum/tbin/gap {len(pre(*COMMON))})')
    print(f'tnorm 20등분 {len(tn20)} / tfin {len(tfin)} / tfout {len(tfout)} / 송신 60 {len(new60)}')
    for k, v in sets.items():
        assert len(v) == len(set(v)), f'{k}에 중복'
        print(f'  {k}: {len(v)}개')
    print(flush=True)

    X, y, g = df[names].values, df.label.values, df.group.values
    ug = np.unique(g)
    folds = [(np.isin(g, ug[tr]), np.isin(g, ug[va])) for tr, va in GroupKFold(5).split(ug, groups=ug)]
    col = {f: i for i, f in enumerate(names)}
    t0, acc = time.time(), {}

    for seed in seeds:
        print(f'===== 시드 {seed} =====', flush=True)
        for k, fs in sets.items():
            cols = [col[f] for f in fs]
            v = []
            for mtr, mva in folds:
                m = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
                m.fit(X[mtr][:, cols], y[mtr])
                v.append(m.score(X[mva][:, cols], y[mva]))
            acc[(k, seed)] = v
            print(f'  {k:30s} ({len(fs):3d}개)  {np.mean(v):.4f} ± {np.std(v):.4f}'
                  f'  {np.round(v, 4)}  ({time.time() - t0:.0f}s)', flush=True)

    print('\n===== 정리 =====')
    print(f'{"세트":30s} {"개수":>4s} {"평균":>7s} {"fold표준편차":>11s} {"시드변동폭":>9s}')
    rows = []
    for k, fs in sets.items():
        done = [acc[(k, s)] for s in seeds]
        means = [np.mean(v) for v in done]
        rows.append((np.mean(means), k, len(fs), np.mean([np.std(v) for v in done]),
                     max(means) - min(means)))
    for mean, k, n, fstd, span in sorted(rows, reverse=True):
        print(f'{k:30s} {n:4d} {mean:7.4f} {fstd:11.4f} {span:9.4f}')

    print('\n== 판단 ==')
    d = {x[1]: x for x in rows}
    a, b = d['11/15 세트 (40등분 수신·송신)'], d['11/19 세트 (20등분 + 송신 60)']
    lim = max(a[3], a[4], b[3], b[4])
    gap = a[0] - b[0]
    print(f'11/15 {a[0]:.4f} (fold 표준편차 {a[3]:.4f}, 시드 변동폭 {a[4]:.4f})')
    print(f'11/19 {b[0]:.4f} (fold 표준편차 {b[3]:.4f}, 시드 변동폭 {b[4]:.4f})')
    print(f'차이 {gap:+.4f} / 유의 기준 {lim:.4f}')
    print('→ ' + ('11/15 세트를 쓴다.' if gap > lim else
                  '11/19 세트를 쓴다.' if -gap > lim else
                  '차이가 변동폭 안이다. 둘 중 어느 쪽을 써도 된다 — 구현이 단순한 11/15(그룹 하나)를 쓰고,'
                  ' 11/19는 어느 피처가 왜 효과를 냈는지 보여주는 근거로 문서에만 남긴다.'))
    u = d['합집합 (11/15 + 송신 60)']
    best = max(a[0], b[0])
    print(f'합집합 {u[0]:.4f} ({u[2]}개), 더 나은 202개와 차이 {u[0] - best:+.4f}')
    print('→ ' + ('합집합이 더 낫다. 피처 80개를 더 쓸 값이 있다.' if u[0] - best > lim else
                  '합집합은 변동폭 안이다. 202개로 충분하다.'))
    print('DONE', flush=True)


if __name__ == '__main__':
    main(tuple(int(x) for x in sys.argv[1:]) or SEEDS)
