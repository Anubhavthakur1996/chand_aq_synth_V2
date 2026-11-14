"""Recomputes every number added in the revision (CO source scale, interpolation shares, ACF check,
generated-hour diagnostics) from the raw data and saved results.  Usage: python -I paper_checks.py <repo_dir>"""
import os as _os, sys as _sys
_REPO=_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_OUT=_os.path.join(_REPO,"results","chandigarh","revision")
_DEFAULT_OUT=_os.path.join(_OUT,"paper_checks.json")
_os.makedirs(_OUT,exist_ok=True)

import sys, os, glob, json
sys.path.insert(0, (sys.argv[1] if len(sys.argv)>1 else _REPO)); root = (sys.argv[1] if len(sys.argv)>1 else _REPO)
import numpy as np, pandas as pd
import config as C
from src import preprocess as P, windows as W
cfg = C.DATASETS['chandigarh']; cols = list(cfg['base_cols']); cols = [c for c in cols if c != 'nox']
raw, _, _ = P._load_chandigarh(cfg)
raw.index = raw.index.floor('h'); raw = raw.groupby(level=0).mean().sort_index()
raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='h'))[cols]
isna = raw.isna(); f = raw.interpolate(method='linear', limit=C.INTERP_LIMIT, limit_area='inside')
keep = ~f.isna().any(axis=1); interp = (isna & ~f.isna())[keep]; df = f[keep]
seg = (df.index.to_series().diff() > pd.Timedelta(hours=1)).cumsum().to_numpy()
out = {}
out['usable_hours'] = int(len(df))
out['interp_share_all_pct'] = round(interp.values.mean() * 100, 2)
out['interp_hours_any_pct'] = round(interp.any(axis=1).mean() * 100, 2)
out['interp_share_per_var_pct'] = (interp.mean() * 100).round(2).to_dict()
out['interp_train_share_pct'] = round(interp.iloc[:336].values.mean() * 100, 2)
out['interp_train_hours_any_pct'] = round(interp.iloc[:336].any(axis=1).mean() * 100, 2)
# ACF over lags 1..12 inside segments: all pairs vs pairs of originally observed hours only
def acf(x, ok, k):
    a, b, oka, okb, sa, sb = x[:-k], x[k:], ok[:-k], ok[k:], seg[:-k], seg[k:]
    m = (sa == sb) & oka & okb
    return np.corrcoef(a[m], b[m])[0, 1]
acf_rows = {}
for c in cols:
    x = df[c].to_numpy(); allok = np.ones(len(x), bool); obs = ~interp[c].to_numpy()
    d = np.array([acf(x, obs, k) - acf(x, allok, k) for k in range(1, 13)])
    acf_rows[c] = {'mean_abs_change': round(float(np.mean(np.abs(d))), 4), 'max_abs_change': round(float(np.max(np.abs(d))), 4)}
out['acf_observed_only_vs_all'] = acf_rows
co_src = df['co'] / (P.MOLAR_MASS['co'] / P.MOLAR_VOLUME)
out['co_source_scale'] = {k: round(float(v), 2) for k, v in [('mean', co_src.mean()), ('std', co_src.std()), ('min', co_src.min()), ('max', co_src.max())]}
out['co_factor'] = round(P.MOLAR_MASS['co'] / P.MOLAR_VOLUME, 3)
clk = {}
for ds, arm in [('chandigarh', 'a2_time_long'), ('chandigarh', 'a4_time_short'), ('bilbao', 'a4_time_short')]:
    rs = [json.load(open(p))['clock_diagnostics_before_projection'] for p in sorted(glob.glob(os.path.join(root, f'results/{ds}/runs/{arm}_seed*.json')))]
    g = lambda k, q: np.array([r[k][q] for r in rs])
    clk[f'{ds}/{arm}'] = {'hour_share_steps_0.5_1.5_pct': [round(g('hour', 'share_steps_0.5_to_1.5h').mean() * 100, 1), round(g('hour', 'share_steps_0.5_to_1.5h').std(ddof=1) * 100, 1)],
                          'hour_radius_sq_mean': round(g('hour', 'radius_sq_mean').mean(), 2), 'dow_radius_sq_mean': round(g('dow', 'radius_sq_mean').mean(), 2)}
out['clock_diagnostics'] = clk
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open((sys.argv[2] if len(sys.argv)>2 else _DEFAULT_OUT), 'w'), indent=1, ensure_ascii=False)
