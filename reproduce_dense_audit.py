"""Reconstruct dense timing and measurement-grid audits from anonymous inputs.

Each NPZ supplies pooled sketches or exact squared-secant measurements, plus
the scalar dispersion computed from its groups. Inputs choose beta before losses enter.
No network, raw training, or external dataset is required.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def aggregate(x):
    x=np.asarray(x,float)
    return dict(mean=float(x.mean()),maximum=float(x.max()),
                cvar25=float(np.sort(x)[-math.ceil(len(x)/4):].mean()))


def choose(z,t,lags,grid):
    pos,full,V,compact=z
    k=pos[t]
    ns=np.asarray(lags,float)
    if compact is None:
        ys=np.array([np.sum((full[k]-full[pos[t-int(n)]])**2)/n**2 for n in ns])
    elif t == 200:
        ys=np.array([compact['mesh'][int(n)-20] for n in ns])
    else:
        ys=np.array([compact['secants'][t-100,compact['lags'][int(n)]] for n in ns])
    I,C=np.linalg.lstsq(np.column_stack([np.ones(len(ns)),1/ns**2]),ys,rcond=None)[0]
    D=max(0.,float(I))+max(0.,float(C))/10000
    raw_H=(V[k]/(4*D))**(1/3) if D>0 else 1/(1-max(grid))
    H=min(1/(1-max(grid)),max(1/(1-min(grid)),raw_H))
    beta=min(grid,key=lambda b:abs(b-(1-1/H)))
    return dict(V=float(V[k]),D=D,H=float(H),continuous_beta=float(1-1/H),beta=beta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--supplement',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args();sup=args.supplement.resolve();out=args.output.resolve()
    inp=json.loads((sup/'inputs.json').read_text(encoding='utf-8'))
    cases=inp['cases'];grid=inp['grid'];names=[c['workload'] for c in cases]
    zs={}
    for name in names:
        with np.load(sup/'robustness_probes'/(name+'_mb16.npz')) as z:
            steps=z['dense_steps'].astype(int)
            compact=None
            if 'dense_proj_r_full' in z:
                full=z['dense_proj_r_full'].astype(float)
            else:
                assert np.array_equal(z['secant_steps'],np.arange(100,501))
                assert np.array_equal(z['secant_lags'],np.array([20,50,100]))
                assert np.array_equal(z['mesh_lags'],np.arange(20,101))
                full=None
                compact=dict(secants=z['secants'].astype(float),
                    lags={int(n):i for i,n in enumerate(z['secant_lags'])},
                    mesh=z['mesh_secants'].astype(float))
                assert np.array_equal(compact['secants'][100],compact['mesh'][[0,30,80]])
            zs[name]=({int(t):i for i,t in enumerate(steps)},full,z['dense_V'].astype(float),compact)
            raw_positions={int(t):i for i,t in enumerate(z['steps'])}
            group_steps=z['group_steps'] if 'group_steps' in z else z['steps']
            for j,t in enumerate(group_steps):
                i=raw_positions[int(t)];k=zs[name][0][int(t)]
                if full is not None:
                    assert np.array_equal(z['proj_r_full'][i],full[k])
                direct=16/(z['proj_r_group'].shape[1]-1)*np.mean(np.sum(
                    (z['proj_r_group'][j]-z['proj_r_full'][i])**2,axis=1))
                assert abs(direct-zs[name][2][k])<1e-10*max(1,abs(direct))
                if compact is not None and int(t)>=100:
                    for n in [20,50,100]:
                        if int(t)-n in raw_positions:
                            y=np.sum((z['proj_r_full'][i]-z['proj_r_full'][raw_positions[int(t)-n]])**2)/float(n)**2
                            assert y==compact['secants'][int(t)-100,compact['lags'][n]]
    timing=[dict(step=t,lags=[20,50,100],decisions=[choose(zs[n],t,(20,50,100),grid) for n in names])
            for t in range(100,501)]
    mesh=[dict(spacing=d,lags=list(range(20,101,d)),
        checkpoints=1+len(range(20,101,d)),
        decisions=[choose(zs[n],200,tuple(range(20,101,d)),grid) for n in names])
        for d in [1,2,5,10,20,40,80]]
    base=next(r for r in timing if r['step']==200)
    assert [d['beta'] for d in base['decisions']]==[c['expected_beta'] for c in cases]
    # All choices are fixed before validation outcomes are used below.
    loss=np.array([[c['seed1_grid_losses'][str(b)] for b in grid] for c in cases])
    gaps=100*(loss/loss.min(axis=1)[:,None]-1)
    fixed={str(b):aggregate(gaps[:,j]) for j,b in enumerate(grid)}
    for rows in [timing,mesh]:
        for row in rows:
            idx=[grid.index(d['beta']) for d in row['decisions']]
            rg=gaps[np.arange(len(names)),idx]
            row['metrics']=aggregate(rg)
            row['gap_pct_by_workload']=dict(zip(names,map(float,rg)))
            row['agreement_with_update200']=sum(d['beta']==b['beta'] for d,b in zip(row['decisions'],base['decisions']))
    rows=[r for r in timing if r['step']>=150]
    counts={str(b):{k:sum(r['metrics'][k]<fixed[str(b)][k] for r in rows)
            for k in ['mean','maximum','cvar25']} for b in [.94377,.96838]}
    assert counts[str(.94377)]==dict(mean=339,maximum=339,cvar25=351)
    assert counts[str(.96838)]==dict(mean=288,maximum=288,cvar25=288)
    result=dict(scope='Within-pilot matched-seed sensitivity; same saved training outcomes.',
        workload_order=names,canonical_step=200,timing=timing,regular_mesh=mesh,
        fixed=fixed,dense_window=[150,500],dense_count=len(rows),beats_fixed_counts=counts)
    out.mkdir(parents=True,exist_ok=True)
    (out/'dense_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}rrrr@{}}',
           r'\toprule Decision update & Mean gap & Maximum gap & CVaR$_{25}$\\\midrule']
    for t in [100,150,200,250]:
        r=next(r for r in timing if r['step']==t)
        lines.append(str(t)+' & '+' & '.join(f"{r['metrics'][k]:.4f}" for k in ['mean','maximum','cvar25'])+r'\\')
    lines.append(r'\bottomrule\end{tabular*}')
    (out/'timing_table.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(times=len(timing),dense_times=len(rows),counts=counts,
        mesh=[dict(spacing=r['spacing'],checkpoints=r['checkpoints'],agreement=r['agreement_with_update200'],
                   metrics=r['metrics']) for r in mesh]),indent=2))


if __name__=='__main__':main()
