"""open-world 공유 split 인덱스 생성 → splits/*.csv. 팀 전원이 이 파일을 읽어 같은 split을 쓴다.

사용법:
  python -I -X utf8 code/ow_splits.py                                   # pkl만 (피처 CSV로 충분)
  python -I -X utf8 code/ow_splits.py <mon.zip> <unmon.zip>             # + TrafficSliver
  > logs/ow_splits.log  (약 10초. zip은 파일 목록만 읽는다)

결정 (reports/openworld.md 1절, 2026-11-26):
  test holdout과 visit 모드 CV는 A팀 규칙을 그대로 옮긴다(기존 A팀 수치와 바로 비교하기 위해).
  subpage 모드만 새로 만들고, 거기에 random_state=42를 쓴다.

pkl (features/*.csv와 같은 행 순서. row = CSV의 0부터 센 행 번호)
  splits/pkl_mon.csv   row, site, url, inst, is_test, fold_visit, fold_subpage
  splits/pkl_unmon.csv row, is_test, fold
  - is_test (mon)  : ld.mon_is_test(url, inst) — URL마다 방문 2개, 1,900개 (A팀, docs/TEST_SPLIT.md)
  - is_test (unmon): np.random.default_rng(0).permutation(10000)의 앞 1,000개 (A팀 openworld.split)
  - fold_visit     : inst % 5 (A팀). 같은 URL의 다른 방문이 train/검증 양쪽에 들어간다 = "랜덤 split"에 해당
  - fold_subpage   : URL(subpage) 단위 StratifiedGroupKFold(5, shuffle, random_state=42).
                     한 URL의 train 방문 18개가 모두 같은 fold에 들어가고, 사이트마다 fold당 URL 2개다
  - fold (unmon)   : 섞은 순서에서 train i번째 → i % 5 (A팀). unmon은 기록마다 다른 사이트라 두 모드에 같은 값을 쓴다
                     → visit/subpage 비교에서 바뀌는 변수는 mon의 묶는 방식 하나뿐이다
  - test 행의 fold는 -1
  - 주의: test holdout은 URL마다 방문 2개라 subpage 모드에서도 test의 URL은 train에 있다.
          subpage 모드는 CV(train 내부)로만 비교하고, 최종 test 1회 평가는 visit 모드 기준이다.

TrafficSliver (방문 단위. 행 순서와 무관하게 group으로 매칭한다)
  splits/sliver_mon.csv   group, site, inst, n_nonempty_split, is_test, fold
  splits/sliver_unmon.csv group, id, n_nonempty_split, is_test, fold
  - group          : load_data._sliver_name과 같다 (mon = site*1000 + inst, unmon = 10**6 + id).
                     같은 방문의 split 5개와 join이 같은 group → 반드시 group 단위로 나눈다
  - is_test (mon)  : ld.sliver_is_test(group) (A팀, 방문 10%)
  - is_test (unmon): ld.sliver_is_test(group + 10**7) — A팀 openworld_sliver.py가 unmon group에 OFFSET을 더한 뒤 뗐다
  - fold (mon)     : A팀 verify_sliver_sets.py와 같다. 비어 있지 않은 split이 있는 train 방문 → GroupKFold(5)
  - fold (unmon)   : 같은 방식을 unmon train 방문에 따로 적용 (A팀은 sliver open-world CV를 정의하지 않았다)
  - subpage 정보가 파일명에 없으므로 subpage 모드는 없다
  - 축소판(mon_25 등)은 같은 파일의 부분집합이다(site < N, id < N). 이 표를 거르면 된다
"""
import os, re, sys, zipfile

import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_data as ld

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'splits')
SEED_A_UNMON = 0      # A팀 openworld.SEED
N_UNMON_TEST = 1000   # A팀 openworld.N_UNMON_TEST
RANDOM_STATE = 42     # 새로 만드는 분할(subpage)에 쓰는 시드
SLIVER_UNMON_OFFSET = 10 ** 7  # A팀 openworld_sliver.OFFSET


def pkl_splits():
    m = pd.read_csv(os.path.join(ROOT, 'features', 'features_mon.csv'), usecols=['site', 'url', 'inst'])
    n_unmon = sum(1 for _ in open(os.path.join(ROOT, 'features', 'features_unmon.csv'))) - 1
    m.insert(0, 'row', np.arange(len(m)))
    m['is_test'] = ld.mon_is_test(m.url.values, m.inst.values)
    tr = ~m.is_test.values
    m['fold_visit'] = np.where(tr, m.inst % 5, -1)
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_STATE)
    fs = np.full(len(m), -1)
    idx = np.flatnonzero(tr)
    for k, (_, va) in enumerate(sg.split(idx, m.site.values[idx], groups=m.url.values[idx])):
        fs[idx[va]] = k
    m['fold_subpage'] = fs

    perm = np.random.default_rng(SEED_A_UNMON).permutation(n_unmon)
    u = pd.DataFrame({'row': np.arange(n_unmon), 'is_test': False, 'fold': -1})
    u.loc[perm[:N_UNMON_TEST], 'is_test'] = True
    u.loc[perm[N_UNMON_TEST:], 'fold'] = np.arange(n_unmon - N_UNMON_TEST) % 5

    # ---- 검증 ----
    n_site = m.site.nunique()
    assert m.is_test.sum() == 2 * m.url.nunique()
    trm = m[tr]
    per_url = trm.groupby('url').fold_subpage.nunique()
    assert (per_url == 1).all(), 'subpage fold: 한 URL이 여러 fold에 걸쳐 있다'
    site_fold = trm.drop_duplicates('url').groupby(['site', 'fold_subpage']).size().unstack(fill_value=0)
    vis = trm.groupby('url').fold_visit.nunique()
    assert (vis == 5).all(), 'visit fold: 각 URL이 5개 fold 전부에 나와야 한다'
    # A팀 openworld.split()과 같은 분할인지 직접 대조
    import openworld as ow
    a_tr, a_te = ow.split(skip=14)
    am, au = a_tr[a_tr.label != -1], a_tr[a_tr.label == -1]
    assert (am[['url', 'inst']].to_numpy() == trm[['url', 'inst']].to_numpy()).all()
    assert (am.fold.values == trm.fold_visit.values).all()
    assert sorted(a_te[a_te.label != -1].url * 100 + a_te[a_te.label != -1].inst) == \
        sorted(m[m.is_test].url * 100 + m[m.is_test].inst)
    # unmon: A팀 split은 ignore_index=True로 이어 붙여 원래 행 번호가 없으므로 순서로 대조한다
    ref_tr = perm[N_UNMON_TEST:]
    assert (au.fold.values == np.arange(len(ref_tr)) % 5).all() and len(au) == len(ref_tr)
    assert (u.fold.values[ref_tr] == au.fold.values).all()

    print(f'[pkl] mon {len(m)} (test {m.is_test.sum()}, train {tr.sum()}), 사이트 {n_site}, URL {m.url.nunique()}')
    print(f'      unmon {n_unmon} (test {u.is_test.sum()}, train {(~u.is_test).sum()})')
    print(f'      fold별 행 수 visit  : mon {np.bincount(trm.fold_visit).tolist()}, unmon {np.bincount(u.fold[u.fold >= 0]).tolist()}')
    print(f'      fold별 행 수 subpage: mon {np.bincount(trm.fold_subpage).tolist()} (unmon은 visit과 같다)')
    print(f'      subpage fold의 사이트별 URL 수: 최소 {site_fold.values.min()}, 최대 {site_fold.values.max()} '
          f'(사이트마다 fold당 {"정확히 2개" if (site_fold.values == 2).all() else "2개가 아님"})')
    print(f'      A팀 openworld.split(skip=14)과 train/test/visit fold 모두 일치 (assert 통과)')
    print(f'      train에서 unmon:mon 1개 사이트 = {(~u.is_test).sum()}:{len(trm) // n_site} '
          f'= {(~u.is_test).sum() / (len(trm) / n_site):.0f}배')
    m.to_csv(os.path.join(OUT, 'pkl_mon.csv'), index=False)
    u.to_csv(os.path.join(OUT, 'pkl_unmon.csv'), index=False)


def sliver_visits(zip_path):
    """zip 파일 목록만 읽어 방문별 비어 있지 않은 split 수를 센다 (압축을 풀지 않는다)."""
    pm = re.compile(r'(\d+)-(\d+)_(?:split_([0-4])|join)$')
    pu = re.compile(r'(\d+)_(?:split_([0-4])|join)$')
    rows = {}
    for info in zipfile.ZipFile(zip_path).infolist():
        n = info.filename
        if not n.endswith('.cell'):
            continue
        stem = n.rsplit('/', 1)[-1][:-len('.cell')]
        label, group, part = ld._sliver_name(n)
        mm = pm.match(stem)
        key = (int(mm[1]), int(mm[2])) if mm else (int(pu.match(stem)[1]),)
        r = rows.setdefault(group, [label, *key, 0])
        if part != ld.JOIN_PART and info.file_size > 0:
            r[-1] += 1
    return rows


def sliver_splits(mon_zip, unmon_zip):
    for side, zp in [('mon', mon_zip), ('unmon', unmon_zip)]:
        rows = sliver_visits(zp)
        g = np.array(sorted(rows))
        if side == 'mon':
            d = pd.DataFrame([[k, *rows[k][1:]] for k in g], columns=['group', 'site', 'inst', 'n_nonempty_split'])
            d['is_test'] = ld.sliver_is_test(d.group.values)
        else:
            d = pd.DataFrame([[k, *rows[k][1:]] for k in g], columns=['group', 'id', 'n_nonempty_split'])
            d['is_test'] = ld.sliver_is_test(d.group.values + SLIVER_UNMON_OFFSET)
        d['fold'] = -1
        cv = (~d.is_test) & (d.n_nonempty_split > 0)
        ug = d.group.values[cv]  # A팀: np.unique(df.group) — 정렬된 방문 번호
        for k, (_, va) in enumerate(GroupKFold(5).split(ug, groups=ug)):
            d.loc[d.group.isin(ug[va]), 'fold'] = k
        if side == 'mon':
            print(f'[sliver] mon 방문 {len(d)}, 사이트 {d.site.nunique()} [{d.site.min()}..{d.site.max()}], '
                  f'test {d.is_test.sum()} ({d.is_test.mean():.1%}), 사이트별 test {d[d.is_test].groupby("site").size().min()}–'
                  f'{d[d.is_test].groupby("site").size().max()}개')
        else:
            print(f'[sliver] unmon 방문 {len(d)}, test {d.is_test.sum()} ({d.is_test.mean():.1%})')
        print(f'         비어 있지 않은 split이 없는 방문 {(d.n_nonempty_split == 0).sum()}개 (fold -1), '
              f'fold별 방문 수 {np.bincount(d.fold[d.fold >= 0]).tolist()}')
        d.to_csv(os.path.join(OUT, f'sliver_{side}.csv'), index=False)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    pkl_splits()
    if len(sys.argv) == 3:
        sliver_splits(sys.argv[1], sys.argv[2])
