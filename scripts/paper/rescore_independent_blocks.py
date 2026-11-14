"""Re-scores every saved Chandigarh run against the validation block, an independent 10-day block
and the whole test block (never used for budget selection).  Usage: python -I scripts/paper/rescore_independent_blocks.py"""
import os as _os, sys as _sys
_REPO=_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_OUT=_os.path.join(_REPO,"results","chandigarh","revision")
_DEFAULT_OUT=_os.path.join(_OUT,"rescore_independent_blocks.csv")
_os.makedirs(_OUT,exist_ok=True)
import sys, os, glob, json
sys.path.insert(0, (sys.argv[1] if len(sys.argv)>1 else _REPO))
import numpy as np, pandas as pd
import config as C
from src import preprocess as P, windows as W, evaluate as E
r = P.prepare_dataset('chandigarh', save=False); df = r[0] if isinstance(r, tuple) else r
cfg = C.DATASETS['chandigarh']; shared = list(cfg['base_cols']); seg = df['segment'].to_numpy()
v = df[shared].to_numpy(float); nt, nv = cfg['train_rows'], cfg['val_rows']
def win(a, b): return W.make_windows(v[a:b], seg[a:b], C.SEQ_LEN, C.SEQ_LEN)
blocks = {'val': win(nt, nt+nv), 'indep': win(nt+nv, nt+2*nv), 'testall': win(nt+nv, len(v))}
train_ref = win(0, nt); train_win = W.make_windows(v[:nt], seg[:nt], C.SEQ_LEN, C.TRAIN_STRIDE)
info = {k: len(w) for k, w in blocks.items()}
idx = df.index
info['indep_dates'] = [str(idx[nt+nv]), str(idx[nt+2*nv-1])]; info['val_dates'] = [str(idx[nt]), str(idx[nt+nv-1])]
info['train_dates'] = [str(idx[0]), str(idx[nt-1])]
# real-train vs block floor
info['floor_emd'] = {k: E.fidelity_block(w, train_ref, shared, shared)['shared_cols']['emd_norm'] for k, w in blocks.items()}
print(json.dumps(info, indent=1))
rows = []
for f in sorted(glob.glob(os.path.join((sys.argv[1] if len(sys.argv)>1 else _REPO), 'results/chandigarh/synthetic/*.npz'))):
    d = np.load(f); cols = list(d['cols']); synth = d['synth'].astype(float)[:, :, [cols.index(c) for c in shared]]
    name = os.path.basename(f)[:-4]; arm, seed = name.rsplit('_seed', 1); seed = int(seed)
    row = {'arm': arm, 'seed': seed}
    for k, w in blocks.items():
        fb = E.fidelity_block(w, synth, shared, shared)['shared_cols']
        row[f'emd_{k}'] = fb['emd_norm']; row[f'acf_{k}'] = fb['delta_acf']; row[f'frob_{k}'] = fb['frobenius']; row[f'js_{k}'] = fb['js_distance']
        if k != 'testall':
            m = E.nn_memorisation(train_win, w, synth, shared, shared, np.random.default_rng(seed)); row[f'nn_{k}'] = m['ratio']
    js = json.load(open(os.path.join((sys.argv[1] if len(sys.argv)>1 else _REPO), f'results/chandigarh/runs/{name}.json')))
    row['emd_val_orig'] = js['fidelity_vs_heldout']['shared_cols']['emd_norm']; row['nn_orig'] = js['memorisation']['ratio']
    rows.append(row)
t = pd.DataFrame(rows); t.to_csv((sys.argv[2] if len(sys.argv)>2 else _DEFAULT_OUT), index=False)
pd.set_option('display.width', 250)
print(t[['arm','seed','emd_val','emd_val_orig','nn_val','nn_orig']].round(4).to_string())
g = t.drop(columns='seed').groupby('arm'); print((g.mean().round(4).astype(str) + ' ± ' + g.std().round(4).astype(str)).T.to_string())
p = t.pivot(index='seed', columns='arm')
for a, b in [('a2_time_long','a1_base_long'),('a3_base_short','a1_base_long'),('a4_time_short','a2_time_long'),('a4_time_short','a3_base_short'),('a4_time_short','a1_base_long')]:
    for k in ['emd_indep','emd_testall']:
        dd = p[k][a]-p[k][b]; print(a,'-',b,k, round(dd.mean(),4),'±',round(dd.std(),4), 'signs', list(np.sign(dd).astype(int)))
