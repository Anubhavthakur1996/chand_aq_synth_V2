"""Rebuilds Fig. 1 (data blocks and interpolation share) and Fig. 3 (fidelity and TSTR utility).
Usage: python -I make_fig1_fig3.py <repo_dir> <out_dir>"""
import os as _os, sys as _sys
_REPO=_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_OUT=_os.path.join(_REPO,"results","chandigarh","revision")
_DEFAULT_OUT=_OUT
_os.makedirs(_OUT,exist_ok=True)

import sys, os
sys.path.insert(0, (sys.argv[1] if len(sys.argv)>1 else _REPO)); root, outd = (sys.argv[1] if len(sys.argv)>1 else _REPO), (sys.argv[2] if len(sys.argv)>2 else _DEFAULT_OUT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
import config as C
from src import preprocess as P
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'dejavuserif'})
cfg = C.DATASETS['chandigarh']; cols = [c for c in cfg['base_cols'] if c != 'nox']
raw, _, _ = P._load_chandigarh(cfg)
raw.index = raw.index.floor('h'); raw = raw.groupby(level=0).mean().sort_index()
raw = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq='h'))[cols]
isna = raw.isna(); f = raw.interpolate(method='linear', limit=C.INTERP_LIMIT, limit_area='inside')
keep = ~f.isna().any(axis=1); interp = (isna & ~f.isna())[keep]; idx = f.index[keep]
nt, nv = cfg['train_rows'], cfg['val_rows']
# ---------------- Fig. 1
fig, (a, b) = plt.subplots(2, 1, figsize=(6.4, 6.0), dpi=200, gridspec_kw={'height_ratios': [1, 1.25], 'hspace': 0.55})
daily = pd.Series(keep.astype(float).to_numpy(), index=f.index).resample('D').mean()
blocks = [(idx[0], idx[nt - 1], '#2f6fa8', 'train (13 days)'), (idx[nt], idx[nt + nv - 1], '#d9862b', 'validation (14 days)'),
          (idx[nt + nv], idx[-1], '#cfe3d4', f'test ({len(idx) - nt - nv:,} usable hours)')]
blocks[2] = (blocks[2][0], blocks[2][1], '0.55', blocks[2][3])
day_idx = daily.index
colors = np.full(len(day_idx), '0.55', dtype=object)
for st, en, col, _ in blocks[:2]:
    colors[(day_idx >= st.normalize()) & (day_idx <= en.normalize())] = col
a.bar(day_idx, daily.values, width=1.0, color=list(colors), align='edge', lw=0)
a.set_ylim(0, 1.0); a.set_yticks([0, 0.5, 1]); a.set_ylabel('Usable share\nper day')
a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12])); a.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
a.set_xlim(idx[0] - pd.Timedelta(days=3), idx[-1] + pd.Timedelta(days=1))
a.set_title('(a) Usable data and the three blocks', fontsize=13, loc='left')
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=c, label=l) for _, _, c, l in blocks], ncol=3, frameon=False, fontsize=9.5, loc='lower center', bbox_to_anchor=(0.5, 1.12), handlelength=1.2, columnspacing=1.0)
names = {'pm25': 'PM2.5', 'pm10': 'PM10', 'o3': 'O$_3$', 'co': 'CO', 'so2': 'SO$_2$', 'temp': 'Temp.', 'rh': 'RH', 'ws': 'Wind sp.', 'wd': 'Wind dir.', 'no': 'NO', 'no2': 'NO$_2$'}
share = (interp.mean() * 100).sort_values(); tot = interp.values.mean() * 100
b.barh([names[c] for c in share.index], share.values, color='0.45', height=0.7)
for i, v in enumerate(share.values): b.text((v if abs(v - tot) > 0.25 else max(v, tot)) + 0.08, i, f'{v:.1f}', va='center', fontsize=10.5)
b.axvline(tot, ls='--', color='black', lw=1.2); b.text(tot + 0.1, -0.9, f'all variables {tot:.1f}%', fontsize=10, va='bottom')
b.set_xlim(0, 8.3); b.set_ylim(-1.0, len(share) - 0.4); b.set_xlabel('Interpolated values among usable hours (%)')
b.set_title('(b) Share of linearly interpolated values', fontsize=13, loc='left')
fig.savefig(os.path.join(outd, 'fig1_data_blocks.png'), bbox_inches='tight', dpi=200); plt.close(fig)
# ---------------- Fig. 3
s = pd.read_csv(os.path.join(root, 'results/chandigarh/all_runs.csv'))
arms = ['a1_base_long', 'a2_time_long', 'a3_base_short', 'a4_time_short']; lab = ['Base\n100k', 'Time\n100k', 'Base\n7.5k', 'Time\n7.5k']
g = s.groupby('arm')
trtr = float(s['trtr_rf_mae'].mean())
fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 6.0), dpi=200)
m = g['emd_norm_heldout'].mean()[arms]; sd = g['emd_norm_heldout'].std()[arms]
a.errorbar(range(4), m, yerr=sd, fmt='o', color='black', ms=9, capsize=5, mew=1.5, lw=1.5)
a.axhline(0.326, ls='--', color='0.45', lw=1.3); a.text(3.3, 0.3247, 'real train vs\nheld-out (0.326)', ha='right', va='top', color='0.35', fontsize=10.5)
a.set_xticks(range(4)); a.set_xticklabels(lab); a.set_xlim(-0.4, 3.4); a.set_ylim(0.315, 0.372)
a.set_ylabel('Held-out EMD (lower = better)'); a.set_title('(a) Fidelity', fontsize=14)
for k, (col, mk, fc, dx, name) in enumerate([('tstr_rf_mae', 'o', 'black', -0.12, 'Random forest'), ('tstr_gbm_mae', 's', 'white', 0.12, 'Grad. boosting')]):
    m = g[col].mean()[arms]; sd = g[col].std()[arms]
    b.errorbar(np.arange(4) + dx, m, yerr=sd, fmt=mk, mfc=fc, mec='black', color='black', ms=9, capsize=5, mew=1.5, lw=1.5, label=name)
b.axhline(trtr, ls='--', color='0.45', lw=1.3); b.text(-0.35, trtr + 0.12, f'train on real only ({trtr:.2f})', color='0.35', fontsize=10.5)
b.axhline(4.68, ls=':', color='0.45', lw=1.6); b.text(-0.35, 4.68 + 0.12, 'persistence (4.68)', color='0.35', fontsize=10.5)
b.set_xticks(range(4)); b.set_xticklabels(lab); b.set_xlim(-0.45, 3.45); b.set_ylim(4, 9.3)
b.set_ylabel('Next-hour PM2.5 MAE (µg/m³)'); b.set_title('(b) TSTR utility', fontsize=14)
b.legend(frameon=False, loc='center left', bbox_to_anchor=(0.0, 0.42), fontsize=11)
fig.tight_layout(w_pad=2.5); fig.savefig(os.path.join(outd, 'fig3_fidelity_utility.png'), dpi=200); plt.close(fig)
print('ok', trtr)
