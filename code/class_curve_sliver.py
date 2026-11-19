"""Extra credit ⑥: 사이트 수(클래스 수)를 바꿔 가며 최종 test 정확도를 잰다 (sklearn 필요).

사용법: python -I -W ignore -X utf8 code/class_curve_sliver.py [v2] > logs/class_curve_sliver.log

  v2: 11/15 정의로 다시 뽑은 CSV와 최종 세트 202개를 쓴다 (4.13). 인자가 없으면 옛 세트다.
        (stdout만 로그로. 2>&1 금지: joblib 경고가 로그를 GB로 불린다)

왜 재나
  조교가 준 데이터에는 mon_25 / mon_50 / mon_75 / mon(95)이 따로 있다. 확인해 보니 사이트를
  줄인 것이다(mon_25 = 사이트 0~24 × 방문 200). 그 파일들이 가리키는 실험이 이것이다.
  4.10의 16.5%는 95-class 값 하나여서, 그것만으로는 "방어가 강한 것"과 "사이트가 많아서 어려운 것"을
  가를 수 없다. 25-class에서 40%가 나오면 전자가 아니고, 25-class에서도 20%면 전자다.
  TrafficSliver 논문들은 클래스 수가 제각각이라, 비교를 하려면 이 곡선이 있어야 한다.

줄인 zip은 받지 않는다. 전체의 부분집합이므로 이미 뽑은 피처에서 `label < N`으로 걸러내면 같다.
세트(202개), 모델(RF 200), 분할(`sliver_is_test`)은 4.10과 똑같이 두어 95-class 값이 4.10과
맞는지로 자기 검증이 된다.

사이트를 줄이면 총 train 행은 줄지만 **사이트 하나당 학습 예시는 그대로다**(모든 사이트가 방문
200개씩이므로 25-class의 train은 95-class의 25/95다. 20,615 = 78,331 × 25/95). 그래서 이 곡선에는
"데이터가 적어서 어려워진 것"이 섞이지 않는다. 섞이지 않은 클래스 수 효과만 본다.
찍기 대비 몇 배인지를 함께 내서 난이도가 클래스 수에 어떻게 붙는지 본다.
"""
import os, sys, time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment as ex
import load_data as ld

N_TREES = 200
SEEDS = (0, 1, 2)
N_SITES = (25, 50, 75, 95)
V2 = 'v2' in sys.argv[1:]
V2_TIME = ('tcum_', 'tbin_', 'gap_', 'tnorm_')
TIME_NEW = ('tcum_', 'tbin_', 'tnorm_', 'gap_')
FINAL_EXTRA = ('tnout_', 'tnqin_', 'tnqout_')


def main():
    import final_test_sliver as ft
    ft.V2 = V2  # 같은 로더를 쓰되 CSV 선택만 넘긴다
    load = ft.load
    df, names = load()
    G = ex.groups(names, pkl_qconst=False)
    fs = G['timing'] + [f for f in names
                        if f.startswith(V2_TIME if V2 else TIME_NEW + FINAL_EXTRA)]
    assert len(fs) == 202, len(fs)

    is_test = ld.sliver_is_test(df.group.values)
    is_split = df.part.values != ld.JOIN_PART
    X, y = df[fs].values, df.label.values
    print(f'세트 {len(fs)}개, RF({N_TREES}), 시드 {SEEDS}, split만 (join 제외)')
    print(f'사이트 수 {N_SITES} — label < N으로 걸러낸다 (조교의 mon_25/50/75/95와 같은 부분집합)\n',
          flush=True)

    t0, res = time.time(), {}
    print('===== 최종 test =====', flush=True)
    for n in N_SITES:
        sel = is_split & (y < n)
        tr, te = sel & ~is_test, sel & is_test
        v = []
        for seed in SEEDS:
            m = RandomForestClassifier(n_estimators=N_TREES, n_jobs=-1, random_state=seed)
            m.fit(X[tr], y[tr])
            v.append(m.score(X[te], y[te]))
            print(f'  {n:2d}-class 시드 {seed}  train {tr.sum():6d} / test {te.sum():5d}'
                  f'  {v[-1]:.4f}  ({time.time() - t0:.0f}s)', flush=True)
        res[n] = (np.mean(v), max(v) - min(v), tr.sum(), te.sum())

    print('\n===== 정리 =====')
    print(f'{"사이트":>5s} {"train":>7s} {"test":>6s} {"정확도":>8s} {"시드변동폭":>9s}'
          f' {"찍기":>7s} {"찍기대비":>8s}')
    for n in N_SITES:
        m, sp, ntr, nte = res[n]
        print(f'{n:5d} {ntr:7d} {nte:6d} {m * 100:7.2f}% {sp * 100:8.2f}%'
              f' {100 / n:6.2f}% {m * n:7.1f}배')

    print('\n== 판단 ==')
    m25, m95 = res[25][0], res[95][0]
    print(f'25-class {m25 * 100:.1f}% / 95-class {m95 * 100:.1f}%')
    print(f'찍기 대비는 {m25 * 25:.1f}배 → {m95 * 95:.1f}배')
    if m25 < 0.30:
        print('→ 사이트를 25개로 줄여도 30%를 넘지 못한다. 95개라서 어려운 것이 아니라 방어가 강하다.')
    else:
        print('→ 사이트를 줄이면 크게 오른다. 95-class 16.5%는 클래스 수 탓이 크다.')
    print('95-class 값이 4.10(final_test_sliver.py)의 B split→split와 맞는지로 자기 검증이 된다.'
          ' 같은 세트·같은 분할·같은 모델이므로 같은 값이 나와야 한다.', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
