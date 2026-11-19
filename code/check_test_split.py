"""closed-world test 분할 실험: (a) inst 18~19 유지 vs (b) URL마다 방문 2개를 골고루 뽑기 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/check_test_split.py > logs/check_test_split.log   (stdout만 로그로)

피처는 후보 131개(@all -@const @early @bseq @cumulabs @dir50), 모델은 experiment.py의 RF(200) 기본값.
이 실험은 분할을 정하기 위한 것이다. 여기 나온 test 점수로 피처나 모델을 고르면 안 된다.

- T1. 방문 2개씩 돌아가며 test로 빼기 (0~1, 2~3, ..., 18~19): inst 18~19만 낮은지, 새 피처에서도 그런지
- T2. 느린 방문(inst 19)을 학습에 조금 넣으면 맞히는지: URL 홀짝으로 반씩 나눠 서로 바꿔 평가
      대조군은 같은 수만큼 inst 18을 넣은 것 (데이터가 늘어난 효과와 구분)
- T3. 분할 (b)의 모습: URL마다 방문 2개를 test로(load_data.mon_is_test, 이 실험 뒤 기본 분할로 채택). 방문 번호별 개수, CV와 test가 맞는지를 (a)와 비교
"""
import os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld
from sklearn.model_selection import PredefinedSplit, cross_val_score

SPEC = ['@all', '-@const', '@early', '@bseq', '@cumulabs', '@dir50']


def rf():
    return ex.MODELS['rf'][0]()


def acc(tr, te):
    m = rf().fit(X[tr], y[tr])
    return (m.predict(X[te]) == y[te]).mean()


if __name__ == '__main__':
    df, names = ex.load()
    feats = ex.select(SPEC, names)
    X, y, inst, url = df[feats].values, df.site.values, df.inst.values, df.url.values
    print(f'피처 {len(feats)}개, 기록 {len(df)}개, RF(200) 기본값', flush=True)
    t0 = time.time()

    print('\n== T1. 방문 2개씩 test로 빼기 (나머지 18개로 학습) ==')
    for k in range(0, 20, 2):
        te = (inst == k) | (inst == k + 1)
        m = rf().fit(X[~te], y[~te])
        p = m.predict(X[te]) == y[te]
        a, b = p[inst[te] == k].mean(), p[inst[te] == k + 1].mean()
        print(f'  test inst {k:2d}~{k + 1:2d}: {p.mean():.4f}   (inst {k}: {a:.4f}, inst {k + 1}: {b:.4f})  ({time.time() - t0:.0f}s)', flush=True)

    print('\n== T2. inst 19를 학습에 반만 넣으면 나머지 반의 inst 19를 맞히는가 (URL 홀짝 교차, 평균) ==')
    base = inst <= 17
    res = {}
    for par in (0, 1):
        mine, other = url % 2 == par, url % 2 != par  # mine = 학습에 더할 쪽, other = 평가할 쪽
        ev19, ev18 = (inst == 19) & other, (inst == 18) & other
        for name, add in (('inst 0~17만', np.zeros_like(base)),
                          ('+ 다른 URL의 inst 18 (대조군)', (inst == 18) & mine),
                          ('+ 다른 URL의 inst 19', (inst == 19) & mine)):
            m = rf().fit(X[base | add], y[base | add])
            r = res.setdefault(name, [[], []])
            r[0].append((m.predict(X[ev19]) == y[ev19]).mean())
            r[1].append((m.predict(X[ev18]) == y[ev18]).mean())
    for name, (a19, a18) in res.items():
        print(f'  {name:28s}  평가 inst 19: {np.mean(a19):.4f}   평가 inst 18: {np.mean(a18):.4f}')
    print(f'  ({time.time() - t0:.0f}s)', flush=True)

    print('\n== T3. 분할 (a) vs (b): CV와 test가 맞는가 ==')
    for name, te in (('(a) inst 18~19', inst >= 18), ('(b) URL마다 골고루 2개', ld.mon_is_test(url, inst))):
        tr = ~te
        # fold = URL 안에서 train 방문의 순번 % 5 (a에서는 inst % 5와 같다)
        order = np.zeros(len(df), int)
        for u in np.unique(url):
            idx = np.flatnonzero(tr & (url == u))
            order[idx] = np.arange(len(idx)) % 5
        cv = cross_val_score(rf(), X[tr], y[tr], cv=PredefinedSplit(order[tr]), n_jobs=1)
        m = rf().fit(X[tr], y[tr])
        p = m.predict(X[te]) == y[te]
        cnt = np.bincount(inst[te], minlength=20)
        print(f'  {name}: test {te.sum()}개, CV {cv.mean():.4f} ± {cv.std():.4f}, test {p.mean():.4f}, 차이 {p.mean() - cv.mean():+.4f}')
        print(f'     test 방문 번호별 개수: {" ".join(map(str, cnt))}')
        print(f'     test 중 inst 19 정확도: {p[inst[te] == 19].mean():.4f} ({(inst[te] == 19).sum()}개), '
              f'나머지: {p[inst[te] != 19].mean():.4f}  ({time.time() - t0:.0f}s)', flush=True)

    # 자가 확인: (b)는 URL마다 정확히 2개, 방문 번호마다 95개
    sb = ld.mon_is_test(url, inst)
    assert (np.bincount(url[sb]) == 2).all() and (np.bincount(inst[sb]) == 95).all()
    print('DONE', flush=True)
