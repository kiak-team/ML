"""데이터 품질 점검 + load_data.py가 교수님 노트북과 같은 결과를 내는지 확인.

사용법: python -I code/check_data.py <pkl 폴더>
결과는 logs/check_data.log 에 저장해 둠. docs/DATA.md 의 수치가 여기서 나온다.
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld

D = sys.argv[1]


def pct(a, q):
    a = sorted(a); return a[min(len(a) - 1, int(q / 100 * len(a)))]


def notebook_convert(sample):
    # load_pickle_code.ipynb 반복문을 글자 그대로 옮긴 것
    size_seq = []
    time_seq = []
    for c in sample:
        dr = 1 if c > 0 else -1
        time_seq.append(abs(c))
        size_seq.append(dr * 512)
    return time_seq, size_seq


def check(name, traces):
    L = [len(t) for t in traces]
    print(f"[{name}] n={len(traces)}")
    print("  len>=10000:", sum(l >= 10000 for l in L), " 9900-9999:", sum(9900 <= l <= 9999 for l in L),
          " 50-59:", sum(50 <= l <= 59 for l in L), " <50:", sum(l < 50 for l in L))
    vals = offgrid = nan = 0
    zg_tot = g_tot = 0; per = []
    first_pos0 = second_in = back = mismatch = 0
    for t in traces:
        vals += len(t)
        for c in t:
            if math.isnan(c) or math.isinf(c): nan += 1
            elif abs(round(c * 100) - c * 100) > 1e-6: offgrid += 1
        z = sum(1 for j in range(len(t) - 1) if abs(t[j + 1]) - abs(t[j]) == 0)
        zg_tot += z; g_tot += len(t) - 1; per.append(z / (len(t) - 1))
        if t[0] == 0 and math.copysign(1, t[0]) > 0: first_pos0 += 1
        if len(t) > 1 and t[1] < 0: second_in += 1
        if any(abs(t[j + 1]) < abs(t[j]) for j in range(len(t) - 1)): back += 1
        if ld.convert(t) != notebook_convert(t): mismatch += 1
    print(f"  values={vals} offgrid(0.01)={offgrid} nan/inf={nan} time-backward traces={back}")
    print(f"  IAT==0 pooled={zg_tot / g_tot * 100:.2f}% per-trace median={pct(per, 50) * 100:.2f}% mean={sum(per) / len(per) * 100:.2f}%")
    print(f"  first==+0.0: {first_pos0}  second cell incoming: {second_in}")
    print(f"  load_data.convert != notebook: {mismatch}")
    return zg_tot, g_tot


mon = ld.load_raw_mon(D)
print('mon URLs:', len(mon), ' samples per URL: min', min(len(mon[i]) for i in range(ld.TOTAL_URLS)),
      'max', max(len(mon[i]) for i in range(ld.TOTAL_URLS)))
labels = [ld.site_label(i) for i in range(ld.TOTAL_URLS) for _ in mon[i]]
cnt = {}
for l in labels: cnt[l] = cnt.get(l, 0) + 1
print('sites:', len(cnt), ' samples per site: min', min(cnt.values()), 'max', max(cnt.values()))
print('labels == notebook (i // 10):', labels == [i // 10 for i in range(950) for _ in mon[i]])
allm = [s for i in range(ld.TOTAL_URLS) for s in mon[i]]
del mon
a = check('mon', allm); del allm
b = check('unmon10', ld.load_raw_unmon(D))
print(f"mon+unmon pooled IAT0 = {(a[0] + b[0]) / (a[1] + b[1]) * 100:.2f}%")
