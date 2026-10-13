"""Nearest-centroid 재평가: η² 기반 피처 선택을 train에서만 하도록 수정.

사용법: python -I eval_nearest_centroid.py <features_mon.csv 경로>

- 피처 값은 features.py가 저장한 features_mon.csv에서 읽는다.
- eta2 / nearest_centroid 함수는 features.py의 것을 그대로 import 해서 쓴다
  (계산 로직이 기존 실행과 동일함을 보장하기 위함).
- [A] 기존 방식: η²를 mon 19,000개 전체(test 포함)로 계산해 피처 선택 → 누수 있음
- [B] 수정 방식: η²를 train 17,100개로만 계산해 피처 선택
"""
import csv, os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('features', os.path.join(HERE, 'features.py'))
features = importlib.util.module_from_spec(spec); spec.loader.exec_module(features)
eta2, nearest_centroid = features.eta2, features.nearest_centroid

path = sys.argv[1]
with open(path, newline='') as f:
    rd = csv.reader(f); header = next(rd)
    rows = [r for r in rd]
names = header[3:]
site = [int(r[0]) for r in rows]
inst = [int(r[2]) for r in rows]
F = [dict(zip(names, map(float, r[3:]))) for r in rows]

# 분할: URL마다 20회 중 inst 0~17 → train, 18~19 → test (무작위 없음)
is_train = [j < 18 for j in inst]
tr = [i for i in range(len(F)) if is_train[i]]
print(f'samples={len(F)} train={len(tr)} test={len(F) - len(tr)} features={len(names)} sites={len(set(site))}')

# 경계 조건 점검 (features.py의 빈 배열 / duration=0 분기가 실제로 쓰였는지)
print('edge cases: duration==0: %d, n_total<2: %d, n_out<2: %d, n_in<2: %d' % (
    sum(r['duration'] == 0 for r in F), sum(r['n_total'] < 2 for r in F),
    sum(r['n_out'] < 2 for r in F), sum(r['n_in'] < 2 for r in F)))


def rank_by_eta2(idx):
    lab = [site[i] for i in idx]
    sc = [(k, eta2([F[i][k] for i in idx], lab)) for k in names]
    sc.sort(key=lambda x: -x[1])  # 안정 정렬: 동점이면 피처 반환 순서 유지 (features.py와 동일)
    return sc


A = rank_by_eta2(range(len(F)))
B = rank_by_eta2(tr)

print('\n== 선택 없는 실험 (η² 선택과 무관, 누수 없음) ==')
print('  all %d features : %.4f' % (len(names), nearest_centroid(F, names, site, is_train)))
groups = {
    'count only': ['n_total', 'n_in', 'n_out', 'frac_out'],
    'timing only': [k for k in names if k.startswith(('iat', 't_', 'duration', 'pps', 'cells_per'))],
    'burst only': [k for k in names if 'burst' in k],
    'cumul only': [k for k in names if k.startswith('cumul')],
    'order/conc': [k for k in names if 'order' in k or k.startswith('conc') or k.startswith(('first', 'last'))],
}
for g, ks in groups.items():
    print('  %-15s (%2d): %.4f' % (g, len(ks), nearest_centroid(F, ks, site, is_train)))

print('\n== η² 순위 선택 실험 ==')
print('%-12s %-28s %-28s' % ('selection', '[A] 기존: η² on all 19,000', '[B] 수정: η² on train 17,100'))
for label, sl in (('top-10', slice(0, 10)), ('top-20', slice(0, 20)), ('bottom-20', slice(-20, None))):
    ka = [k for k, _ in A[sl]]; kb = [k for k, _ in B[sl]]
    acc_a = nearest_centroid(F, ka, site, is_train); acc_b = nearest_centroid(F, kb, site, is_train)
    print('%-12s %-28.4f %-28.4f  same feature set: %s' % (label, acc_a, acc_b, set(ka) == set(kb)))
    if set(ka) != set(kb):
        print('    only in A:', sorted(set(ka) - set(kb)))
        print('    only in B:', sorted(set(kb) - set(ka)))

print('\n== η² 순위 (A vs B) ==')
pos_b = {k: i for i, (k, _) in enumerate(B)}
eb = dict(B)
print('%4s %-18s %9s %9s %6s' % ('A#', 'feature', 'eta2_A', 'eta2_B', 'B#'))
for i, (k, e) in enumerate(A):
    print('%4d %-18s %9.4f %9.4f %6d' % (i + 1, k, e, eb[k], pos_b[k] + 1))
