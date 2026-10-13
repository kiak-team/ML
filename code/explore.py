import pickle, sys, statistics as st
def pct(a,q):
    a=sorted(a); return a[min(len(a)-1,int(q/100*len(a)))]
D = sys.argv[1]
def stats(name, traces):
    L = [len(t) for t in traces]
    dur = [abs(t[-1]) for t in traces if len(t)]
    out = [sum(1 for c in t if c > 0)/max(len(t),1) for t in traces]
    print(f"[{name}] n={len(traces)} len: min={min(L)} p25={pct(L,25)} med={pct(L,50)} p75={pct(L,75)} p95={pct(L,95)} max={max(L)}")
    print(f"   duration(s): med={pct(dur,50):.1f} p95={pct(dur,95):.1f} max={max(dur):.1f}; outgoing ratio med={pct(out,50):.3f} p5={pct(out,5):.3f} p95={pct(out,95):.3f}")
    print(f"   len<50: {sum(l<50 for l in L)}, len==0: {sum(l==0 for l in L)}")

with open(D+'/mon_standard.pkl','rb') as f: mon = pickle.load(f)
print(type(mon), len(mon))
k0 = list(mon.keys())[:3] if isinstance(mon, dict) else None
print('keys sample', k0)
first = mon[0]; print('mon[0] type', type(first), 'len', len(first)); print('sample[0][:15]', first[0][:15])
cnts = [len(mon[i]) for i in range(len(mon))]
print('instances per URL: min', min(cnts), 'max', max(cnts), 'total', sum(cnts))
allm = [s for i in range(len(mon)) for s in mon[i]]
stats('mon', allm)
# zero-time / sign check
zeros = sum(1 for t in allm for c in t if c == 0)
print('exact-zero entries (direction ambiguous):', zeros)
mono = sum(1 for t in allm if all(abs(t[j]) <= abs(t[j+1]) for j in range(len(t)-1)))
print('monotonic timestamps:', mono, '/', len(allm))
del mon, allm
for fn in ['unmon_standard10.pkl','unmon_standard10_3000.pkl']:
    with open(D+'/'+fn,'rb') as f: u = pickle.load(f)
    print(fn, type(u), len(u)); 
    tr = [u[i] for i in range(len(u))]
    print('  [0][:10]', tr[0][:10])
    stats(fn, tr)
    del u, tr
