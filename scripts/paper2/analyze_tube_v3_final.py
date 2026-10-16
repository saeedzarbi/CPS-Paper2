import pandas as pd,glob,numpy as np,os,json,sys
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R='final_test'; FIG=sys.argv[1]; os.makedirs(FIG,exist_ok=True)
SETS=['5g_markov','5g_harsh','bb','5g_irish']
NICE={'tube_v2_step_static_a0.15':'Tube α=0.15 (primary)','tube_v2_step_static_a0.1':'Tube α=0.10','tube_v2_maxscore_a0.1':'Tube max-score α=0.10','tube_v2_step_aci_a0.15@g0.002':'Tube+ACI α=0.15','tube_v2_bonf_static':'Tube Bonferroni','robust_mpc_logu':'RobustMPC','mpc_fixed_quantile_native':'Fixed-quantile MPC','mpc_fixed_rho_0.5':'MPC ρ=0.5','mpc_fixed_rho_0.6':'MPC ρ=0.6','mpc_fixed_rho_0.7':'MPC ρ=0.7','raw':'BBA (raw)'}
rng0=lambda:np.random.default_rng(0)
def clus(d,c,n=4000):
    g=d.groupby('trace_file')[c].apply(np.array).tolist();r=rng0();k=len(g)
    bs=[np.concatenate([g[i] for i in r.integers(0,k,k)]).mean() for _ in range(n)];return d[c].mean(),*np.percentile(bs,[2.5,97.5])
def pair(m,c,n=4000):
    dd=(m[c]-m[c+'_r']).values;r=rng0();bs=[dd[r.integers(0,len(dd),len(dd))].mean() for _ in range(n)];return dd.mean(),*np.percentile(bs,[2.5,97.5])
def pclus(m,c,n=4000):
    m=m.assign(dd=m[c]-m[c+'_r']);g=m.groupby('trace_file').dd.apply(np.array).tolist();r=rng0();k=len(g)
    bs=[np.concatenate([g[i] for i in r.integers(0,k,k)]).mean() for _ in range(n)];return m.dd.mean(),*np.percentile(bs,[2.5,97.5])
F=lambda t,p=2:f"{t[0]:+.{p}f} [{t[1]:+.{p}f},{t[2]:+.{p}f}]"
G=lambda t,p=2:f"{t[0]:.{p}f} [{t[1]:.{p}f},{t[2]:.{p}f}]"
L=[];D={}
for ts in SETS:
    A={os.path.basename(p)[:-4].replace('_at_','@'):pd.read_csv(p,comment='#') for p in sorted(glob.glob(f'{R}/{ts}/*.csv'))}
    D[ts]=A; d0=next(iter(A.values()))
    L.append(f"\n## {ts} (test) — n_traces={d0.trace_file.nunique()}, episodes/arm={len(d0)}\n\nMeans with **trace-cluster bootstrap** 95% CI. ★ = QoE–rebuffer Pareto-efficient.\n\n| arm | QoE | rebuf (s) | VMAF | cov h1..h5 | joint |\n|---|---|---|---|---|---|")
    pts={k:(d.episodic_qoe.mean(),d.rebuffer_total.mean()) for k,d in A.items()}
    for k,d in sorted(A.items(),key=lambda kv:-kv[1].episodic_qoe.mean()):
        q,r=pts[k]; par=not any((q2>=q and r2<=r) and (q2>q or r2<r) for q2,r2 in pts.values())
        cov='/'.join(f"{d[f'cov_h{h}'].mean():.3f}" for h in range(1,6)) if 'cov_h1' in d else '—'
        j=f"{d.joint_coverage_ep.mean():.3f}" if 'joint_coverage_ep' in d else '—'
        L.append(f"| `{k}`{' ★' if par else ''} | {G(clus(d,'episodic_qoe'))} | {G(clus(d,'rebuffer_total'))} | {G(clus(d,'mean_vmaf'))} | {cov} | {j} |")
    for ref in ['robust_mpc_logu','mpc_fixed_quantile_native']:
        L.append(f"\n**Δ vs `{ref}`** (arm − ref). Cluster CI (primary) / paired-episode CI; W/T/L on QoE.\n\n| arm | ΔQoE cluster | ΔQoE paired | Δrebuf cluster | ΔVMAF cluster | W/T/L |\n|---|---|---|---|---|---|")
        for k in A:
            if not k.startswith('tube'):continue
            m=A[k].merge(A[ref],on=['trace_file','episode'],suffixes=('','_r'))
            dq=(m.episodic_qoe-m.episodic_qoe_r).values;w=(dq>1e-9).sum();l=(dq<-1e-9).sum()
            L.append(f"| `{k}` | {F(pclus(m,'episodic_qoe'))} | {F(pair(m,'episodic_qoe'))} | {F(pclus(m,'rebuffer_total'))} | {F(pclus(m,'mean_vmaf'))} | {w}/{len(m)-w-l}/{l} |")
open('tables.md','w').write("\n".join(L))
# Fig1 Pareto
fig,ax=plt.subplots(1,4,figsize=(17,4.2))
for a,ts in zip(ax,SETS):
    for k,d in D[ts].items():
        if k=='raw':continue
        q,r=d.episodic_qoe.mean(),d.rebuffer_total.mean()
        tube=k.startswith('tube');prim=k=='tube_v2_step_static_a0.15'
        a.scatter(r,q,s=120 if prim else 55,marker='*' if prim else ('o' if tube else 's'),c='tab:red' if prim else ('tab:blue' if tube else 'tab:gray'),zorder=3)
        a.annotate(NICE.get(k,k),(r,q),fontsize=6.5,xytext=(3,2),textcoords='offset points')
    a.set_title(ts);a.set_xlabel('mean rebuffering per episode (s)');a.grid(alpha=.3)
    if ts in ('bb','5g_harsh','5g_markov'):
        raw=D[ts].get('raw')
        if raw is not None and raw.rebuffer_total.mean()>5*max(d.rebuffer_total.mean() for k,d in D[ts].items() if k!='raw'):
            others=[(d.rebuffer_total.mean(),d.episodic_qoe.mean()) for k,d in D[ts].items() if k!='raw']
            xs,ys=zip(*others);a.set_xlim(min(xs)-.5,max(xs)*1.3+.5);a.set_ylim(min(ys)-8,max(ys)+8);a.text(.98,.02,'raw BBA off-axis',transform=a.transAxes,ha='right',fontsize=7)
ax[0].set_ylabel('mean episodic QoE')
fig.suptitle('Test split: QoE vs rebuffering (red star = primary Tube α=0.15; blue = Tube variants; gray = baselines; raw BBA omitted, see tables)',fontsize=10)
fig.tight_layout();fig.savefig(f'{FIG}/pareto_qoe_rebuffer.png',dpi=200);plt.close(fig)
# Fig2 coverage vs horizon
fig,ax=plt.subplots(1,4,figsize=(17,3.8),sharey=True)
for a,ts in zip(ax,SETS):
    for k,d in D[ts].items():
        if 'cov_h1' not in d:continue
        c=[d[f'cov_h{h}'].mean() for h in range(1,6)]
        a.plot(range(1,6),c,marker='o',label=f"{NICE[k]} (joint {d.joint_coverage_ep.mean():.2f})")
    for t in (0.85,0.9):a.axhline(t,ls='--',c='k',lw=.7)
    a.set_title(ts);a.set_xlabel('horizon step h');a.set_xticks(range(1,6));a.grid(alpha=.3);a.legend(fontsize=6)
ax[0].set_ylabel('empirical per-step coverage')
fig.suptitle('Test split: per-step coverage vs horizon (dashed: nominal 0.85 / 0.90)',fontsize=10)
fig.tight_layout();fig.savefig(f'{FIG}/coverage_vs_horizon.png',dpi=200);plt.close(fig)
# Fig3 shift
if os.path.exists('shift_coverage.json'):
    S=json.load(open('shift_coverage.json'));fig,a=plt.subplots(figsize=(7,3.5))
    for k,v in S.items():a.plot(v['rolling_h1'],label=k.replace('step_','Tube ').replace('aci','+ACI (γ=0.002)'))
    a.axvline(300,c='k',ls=':');a.axhline(0.85,c='k',ls='--',lw=.7);a.set_xlabel('chunk (markov → harsh at 300)');a.set_ylabel('rolling h=1 coverage (21-chunk)');a.legend(fontsize=8);a.grid(alpha=.3)
    a.set_title('Distribution shift (VAL traces, offline tube): α=0.15',fontsize=10);fig.tight_layout();fig.savefig(f'{FIG}/shift_rolling_coverage.png',dpi=200);plt.close(fig)
print('ok')
