import sys, math, os, csv
from itertools import groupby
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I 에서도 load_data를 찾도록
import load_data as ld


def mstd(a):
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    m = sum(a) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in a) / n)


def q(sorted_a, p):
    if not sorted_a:
        return 0.0
    return sorted_a[min(len(sorted_a) - 1, int(p * len(sorted_a)))]


def feats(t):
    n = len(t)
    ts, sizes = ld.convert(t)  # 방향·시간 규칙은 load_data.py(교수님 노트북)와 동일
    d = [1 if s > 0 else -1 for s in sizes]
    out_idx = [i for i, x in enumerate(d) if x == 1]
    in_idx = [i for i, x in enumerate(d) if x == -1]
    n_out, n_in = len(out_idx), len(in_idx)
    f = {}
    # --- 개수 ---
    f['n_total'] = n
    f['n_in'] = n_in
    f['n_out'] = n_out
    f['frac_out'] = n_out / n
    # --- 시간 ---
    dur = ts[-1]
    f['duration'] = dur
    f['cells_per_sec'] = n / dur if dur > 0 else 0
    f['t_25pct'] = ts[int(0.25 * (n - 1))]
    f['t_50pct'] = ts[int(0.50 * (n - 1))]
    f['t_75pct'] = ts[int(0.75 * (n - 1))]
    iat = [b - a for a, b in zip(ts, ts[1:])]
    m, s = mstd(iat); f['iat_mean'], f['iat_std'] = m, s
    si = sorted(iat)
    f['iat_max'] = si[-1] if si else 0
    f['iat_p90'] = q(si, 0.9)
    f['iat_zero_frac'] = sum(1 for x in iat if x == 0) / max(len(iat), 1)
    ti = [ts[i] for i in in_idx]; to = [ts[i] for i in out_idx]
    iin = [b - a for a, b in zip(ti, ti[1:])]; iout = [b - a for a, b in zip(to, to[1:])]
    f['iat_in_mean'], f['iat_in_std'] = mstd(iin)
    f['iat_out_mean'], f['iat_out_std'] = mstd(iout)
    # --- 순서 (k-FP ordering) ---
    f['out_order_mean'], f['out_order_std'] = mstd(out_idx)
    f['in_order_mean'], f['in_order_std'] = mstd(in_idx)
    # --- 앞/뒤 30셀 ---
    f['first30_out'] = sum(1 for x in d[:30] if x == 1)
    f['last30_out'] = sum(1 for x in d[-30:] if x == 1)
    f['first100_out'] = sum(1 for x in d[:100] if x == 1)
    # --- concentration (20셀 단위 outgoing 수) ---
    conc = [sum(1 for x in d[i:i + 20] if x == 1) for i in range(0, n, 20)]
    m, s = mstd(conc); sc = sorted(conc)
    f['conc_std'] = s; f['conc_max'] = sc[-1]; f['conc_median'] = q(sc, 0.5)
    # --- 초당 셀 수 ---
    bins = [0] * (int(dur) + 1)
    for x in ts:
        bins[int(x)] += 1
    m, s = mstd(bins); sb = sorted(bins)
    f['pps_mean'], f['pps_std'] = m, s
    f['pps_max'] = sb[-1]; f['pps_median'] = q(sb, 0.5)
    # --- burst ---
    runs = [(k, len(list(g))) for k, g in groupby(d)]
    ib = [l for k, l in runs if k == -1]; ob = [l for k, l in runs if k == 1]
    f['n_bursts'] = len(runs)
    f['in_burst_mean'], f['in_burst_std'] = mstd(ib)
    f['in_burst_max'] = max(ib) if ib else 0
    f['out_burst_mean'], f['out_burst_std'] = mstd(ob)
    f['out_burst_max'] = max(ob) if ob else 0
    f['in_burst_ge10'] = sum(1 for l in ib if l >= 10)
    for j in range(3):
        f[f'in_burst_{j + 1}'] = ib[j] if j < len(ib) else 0
    # --- CUMUL (누적 방향합, 10개 지점) ---
    c = 0; cum = []
    for x in d:
        c += x; cum.append(c)
    f['cumul_min'] = min(cum)
    for k in range(1, 11):
        f[f'cumul_{k * 10}pct'] = cum[int(k / 10 * (n - 1))]
    f['cumul_at_500'] = cum[min(499, n - 1)]
    return f


def ranks(a):
    idx = sorted(range(len(a)), key=a.__getitem__)
    r = [0.0] * len(a); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and a[idx[j + 1]] == a[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def eta2(vals, labels):
    r = ranks(vals); n = len(r); gm = sum(r) / n
    sst = sum((x - gm) ** 2 for x in r)
    if sst == 0:
        return 0.0
    sums, cnts = {}, {}
    for x, l in zip(r, labels):
        sums[l] = sums.get(l, 0) + x; cnts[l] = cnts.get(l, 0) + 1
    ssb = sum(cnts[l] * (sums[l] / cnts[l] - gm) ** 2 for l in sums)
    return ssb / sst


def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra); vb = sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else 0


def slog(x):
    return math.copysign(math.log1p(abs(x)), x)


def nearest_centroid(F, names, site, is_train):
    X = [[slog(r[k]) for k in names] for r in F]
    tr = [i for i in range(len(X)) if is_train[i]]
    mu = [sum(X[i][j] for i in tr) / len(tr) for j in range(len(names))]
    sd = [math.sqrt(sum((X[i][j] - mu[j]) ** 2 for i in tr) / len(tr)) or 1 for j in range(len(names))]
    Z = [[(x[j] - mu[j]) / sd[j] for j in range(len(names))] for x in X]
    cen = {}
    for i in tr:
        cen.setdefault(site[i], []).append(Z[i])
    cen = {k: [sum(c) / len(v) for c in zip(*v)] for k, v in cen.items()}
    te = [i for i in range(len(X)) if not is_train[i]]
    ok = 0
    for i in te:
        best = min(cen, key=lambda k: sum((a - b) ** 2 for a, b in zip(Z[i], cen[k])))
        ok += best == site[i]
    return ok / len(te)


if __name__ == '__main__':
    D = sys.argv[1]; OUT = sys.argv[2]
    mon = ld.load_raw_mon(D)
    traces, site, url, inst = [], [], [], []
    for i in range(ld.TOTAL_URLS):
        for j, s in enumerate(mon[i]):
            traces.append(s); site.append(ld.site_label(i)); url.append(i); inst.append(j)
    del mon
    unmon = ld.load_raw_unmon(D)

    # 데이터 자체에 대한 추가 사실
    decs = set()
    for t in traces[:2000]:
        for c in t[:200]:
            decs.add(len(repr(abs(c)).split('.')[-1]) if '.' in repr(c) else 0)
    print('timestamp decimal places seen:', sorted(decs))

    with Pool(os.cpu_count()) as p:
        F = p.map(feats, traces, chunksize=200)
        U = p.map(feats, unmon, chunksize=200)
    del traces, unmon
    names = list(F[0].keys())

    with open(os.path.join(OUT, 'features_mon.csv'), 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['site', 'url', 'inst'] + names)
        for r, a, b, c in zip(F, site, url, inst):
            w.writerow([a, b, c] + [r[k] for k in names])
    with open(os.path.join(OUT, 'features_unmon.csv'), 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(names)
        for r in U:
            w.writerow([r[k] for k in names])

    res = []
    for k in names:
        v = [r[k] for r in F]; vu = [r[k] for r in U]
        e_site = eta2(v, site); e_url = eta2(v, url)
        allv = v + vu; rr = ranks(allv); n1 = len(v); n0 = len(vu)
        auc = (sum(rr[:n1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
        sv = sorted(v); su = sorted(vu)
        res.append((k, e_site, e_url, auc, q(sv, .5), q(su, .5)))
    res.sort(key=lambda x: -x[1])
    print('\n%-18s %8s %8s %7s %10s %10s' % ('feature', 'eta2_site', 'eta2_url', 'AUC', 'mon_med', 'unmon_med'))
    for k, a, b, c, m1, m0 in res:
        print('%-18s %8.3f %8.3f %7.3f %10.3f %10.3f' % (k, a, b, c, m1, m0))
    print('random baseline eta2_site ~ %.4f, eta2_url ~ %.4f' % (94 / 18999, 949 / 18999))

    top = [x[0] for x in res[:15]]
    print('\nSpearman corr among top-15 (|r|>0.9 pairs):')
    cols = {k: [r[k] for r in F] for k in top}
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            r = spearman(cols[top[i]], cols[top[j]])
            if abs(r) > 0.9:
                print('  %s ~ %s : %.3f' % (top[i], top[j], r))

    is_train = [j < 18 for j in inst]
    print('\nNearest-centroid accuracy (95 sites, 10% held out, chance=1.05%):')
    print('  all %d features : %.3f' % (len(names), nearest_centroid(F, names, site, is_train)))
    print('  top-10 by eta2  : %.3f' % nearest_centroid(F, [x[0] for x in res[:10]], site, is_train))
    print('  top-20 by eta2  : %.3f' % nearest_centroid(F, [x[0] for x in res[:20]], site, is_train))
    print('  bottom-20       : %.3f' % nearest_centroid(F, [x[0] for x in res[-20:]], site, is_train))
    groups = {
        'count only': ['n_total', 'n_in', 'n_out', 'frac_out'],
        'timing only': [k for k in names if k.startswith(('iat', 't_', 'duration', 'pps', 'cells_per'))],
        'burst only': [k for k in names if 'burst' in k],
        'cumul only': [k for k in names if k.startswith('cumul')],
        'order/conc': [k for k in names if 'order' in k or k.startswith('conc') or k.startswith(('first', 'last'))],
    }
    for g, ks in groups.items():
        print('  %-15s (%2d): %.3f' % (g, len(ks), nearest_centroid(F, ks, site, is_train)))
