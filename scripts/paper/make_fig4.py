"""Fig. 4: (a) held-out EMD vs training budget (tuning runs), (b) generated hour steps.  Usage: python -I scripts/paper/make_fig4.py"""
import os as _os, sys as _sys
_REPO=_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_OUT=_os.path.join(_REPO,"results","chandigarh","revision")
_DEFAULT_OUT=_os.path.join(_OUT,"fig4_budget_and_hour_steps.png")
_os.makedirs(_OUT,exist_ok=True)
import sys, glob, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root, out = (sys.argv[1] if len(sys.argv)>1 else _REPO), (sys.argv[2] if len(sys.argv)>2 else _DEFAULT_OUT)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'dejavuserif'})
t = pd.read_csv(os.path.join(root,'results/chandigarh/tuning/tuning_runs.csv'))
fig,(a,b)=plt.subplots(1,2,figsize=(9.6,6.6),dpi=200)
for arm,lab,mk,fc,dx in [('a3_base_short','Base','o','black',0.97),('a4_time_short','Time','s','white',1.03)]:
    g=t[t.arm==arm].groupby('epochs').emd_norm.agg(['mean','std'])
    a.errorbar(g.index*dx,g['mean'],yerr=g['std'],marker=mk,mfc=fc,mec='black',color='black',ms=8,capsize=4,lw=1.2,mew=1.5,label=lab)
a.set_xscale('log'); ticks=[1000,2000,3000,5000,7500,10000]
a.set_xticks(ticks); a.set_xticklabels(['1k','2k','3k','5k','7.5k','10k']); a.minorticks_off()
a.axvline(7500,ls='--',color='0.45',lw=1.2); a.text(7500*0.94,0.66,'selected',ha='right',color='0.35',fontsize=10)
a.axhline(0.326,ls=':',color='0.45',lw=1.5); a.text(1000,0.335,'real train vs\nheld-out (0.326)',color='0.35',fontsize=10)
a.set_xlabel('Training epochs'); a.set_ylabel('Held-out EMD (lower = better)'); a.set_title('(a) Budget selection',fontsize=14)
a.legend(frameon=False,loc='upper right'); a.set_ylim(0.3,0.75)
def steps(arm):
    out=[]
    for f in glob.glob(os.path.join(root,f'results/chandigarh/synthetic/{arm}_seed*.npz')):
        d=np.load(f); cols=list(d['cols']); s=d['synth'][:,:,cols.index('hour_sin')]; c=d['synth'][:,:,cols.index('hour_cos')]
        ang=(np.arctan2(s,c)%(2*np.pi))*24/(2*np.pi); st=np.diff(ang,axis=1); out.append(((st+12)%24-12).ravel())
    return np.concatenate(out)
bins=np.arange(-1,4.001,0.2)
s1,s2=steps('a2_time_long'),steps('a4_time_short')
for s,lab,kw in [(s1,'Time-100k',dict(color='0.55',alpha=0.9,histtype='stepfilled')),(s2,'Time-7.5k',dict(color='black',histtype='step',lw=1.6))]:
    w=np.ones_like(s)/len(s)*100
    b.hist(np.clip(s,bins[0],bins[-1]-1e-9),bins=bins,weights=w,label=f'{lab}: {np.mean((s>0.5)&(s<1.5))*100:.1f}%\nof steps in 0.5–1.5 h',**kw)
b.axvspan(0.5,1.5,color='0.85',zorder=0,lw=0)
b.set_xlabel('Step between consecutive\ngenerated hours (h)'); b.set_ylabel('Share of steps (%)'); b.set_title('(b) Generated hour steps',fontsize=14)
b.legend(frameon=False,loc='upper right',fontsize=11); b.set_xticks(range(-1,5)); b.set_ylim(0,48); b.text(1.0,44.5,'valid step\n(0.5–1.5 h)',ha='center',fontsize=8.5,color='0.35') if False else None

fig.tight_layout(w_pad=2.5); fig.savefig(out,dpi=200); print(np.mean((s1>0.5)&(s1<1.5)),np.mean((s2>0.5)&(s2<1.5)))
