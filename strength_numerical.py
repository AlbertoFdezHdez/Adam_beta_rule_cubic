"""Offline descriptive robustness; never changes the canonical policy or trains.

Inputs: published supplement's sketches and all seed-1 candidate losses.
Every main/timing beta is reconstructed before accessing its candidate loss.
Bootstrap resamples workloads, not independent pilots or training replicates.
"""
import hashlib
import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--supplement',type=Path,default=HERE if (HERE/'inputs.json').exists() else HERE.parents[1]/'output/aistats2027_supplement')
parser.add_argument('--output',type=Path,default=HERE)
args=parser.parse_args()
SUP=args.supplement.resolve()
OUTPUT=args.output.resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
spec = importlib.util.spec_from_file_location('offline_robustness', SUP / 'reproduce_robustness.py')
import sys
sys.path.insert(0, str(SUP))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
select_beta = module.select_beta


def aggregate(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), maximum=float(x.max()),
                cvar25=float(np.sort(x)[-math.ceil(len(x) / 4):].mean()))


def contrast(rule, fixed):
    benefit = fixed - rule
    a, b = aggregate(rule), aggregate(fixed)
    return dict(rule=a, fixed=b, reductions_pct={k: 100 * (1 - a[k] / b[k]) if b[k] else None for k in a},
                wins=int(np.sum(benefit > 1e-10)), ties=int(np.sum(np.abs(benefit) <= 1e-10)),
                losses=int(np.sum(benefit < -1e-10)), summed_gap_benefit_pp=float(benefit.sum()),
                gross_gain_pp=float(np.maximum(benefit, 0).sum()),
                gross_cost_pp=float(np.maximum(-benefit, 0).sum()))


def family(name):
    if name.startswith(('llama', 'nanogpt')): return 'causal_lm'
    if name.startswith('resnet'): return 'resnet'
    if name.startswith(('vit', 'swin')): return 'vision_transformer'
    return 'other'


def main():
    inputs = json.loads((SUP / 'inputs.json').read_text())
    timing_archive = json.loads((SUP / 'timing_audit.json').read_text())
    budget_archive = json.loads((SUP / 'probe_budget_audit.json').read_text())
    cases, grid = inputs['cases'], inputs['grid']
    n = len(cases)
    names = [c['workload'] for c in cases]
    decisions = [select_beta(SUP / 'robustness_probes' / (name + '_mb16.npz'), 200, 16, grid) for name in names]
    assert [d['beta'] for d in decisions] == [c['expected_beta'] for c in cases]
    # All canonical decisions have now been made; only now construct loss arrays.
    losses = np.array([[c['seed1_grid_losses'][str(b)] for b in grid] for c in cases], float)
    oracles = losses.min(axis=1)
    gaps = 100 * (losses / oracles[:, None] - 1)
    choice_idx = [grid.index(d['beta']) for d in decisions]
    selected_losses = losses[np.arange(n), choice_idx]
    rule = gaps[np.arange(n), choice_idx]
    baseline_idx = grid.index(inputs['baseline'])
    best_idx = int(gaps.mean(axis=0).argmin())
    comparators = {'baseline': baseline_idx, 'best_global': best_idx}
    result = dict(scope='Retrospective seed-1 descriptive suite audit; no independent confirmation; fixed policy 16 probes, factor 4, decision 200.',
                  grid=grid, workload_order=names, canonical_selections=[d['beta'] for d in decisions],
                  all_fixed_comparisons={str(b): contrast(rule, gaps[:, j]) for j, b in enumerate(grid)})
    result['per_case'] = []
    for i, name in enumerate(names):
        row = dict(workload=name, family=family(name), selected_beta=decisions[i]['beta'],
                   oracle_beta=grid[int(losses[i].argmin())], rule_loss=float(selected_losses[i]),
                   rule_gap_pct=float(rule[i]))
        for label, j in comparators.items():
            row[label] = dict(fixed_beta=grid[j], fixed_loss=float(losses[i, j]), fixed_gap_pct=float(gaps[i, j]),
                              gap_benefit_pp=float(gaps[i, j] - rule[i]),
                              direct_loss_improvement_pct=float(100 * (losses[i, j] - selected_losses[i]) / losses[i, j]))
        result['per_case'].append(row)
    result['attribution'] = {}
    for label, j in comparators.items():
        delta = gaps[:, j] - rule
        direct = 100 * (losses[:, j] - selected_losses) / losses[:, j]
        result['attribution'][label] = dict(summary=contrast(rule, gaps[:, j]),
            families={f: dict(n=int(sum(family(name) == f for name in names)),
                             gap_benefit_sum_pp=float(sum(delta[i] for i, name in enumerate(names) if family(name) == f)))
                      for f in ['causal_lm', 'resnet', 'vision_transformer', 'other']},
            mean_direct_loss_improvement_pct=float(np.mean(100 * (losses[:, j] - selected_losses) / losses[:, j])),
            average_positive_direct_loss_improvement_pct=float(direct[direct > 1e-10].mean()),
            average_adverse_direct_loss_increase_pct=float(-direct[direct < -1e-10].mean()),
            maximum_direct_loss_increase_pct=float(-direct.min()),
            average_positive_gap_benefit_pp=float(delta[delta > 1e-10].mean()),
            average_adverse_gap_cost_pp=float(-delta[delta < -1e-10].mean()),
            resnet_fraction_net_benefit=float(sum(delta[i] for i, name in enumerate(names) if family(name) == 'resnet') / delta.sum()),
            resnet_fraction_gross_gain=float(sum(max(delta[i], 0) for i, name in enumerate(names) if family(name) == 'resnet') / np.maximum(delta, 0).sum()))
    result['family_compromise'] = {f: dict(n=len(idx), best_fixed_beta=grid[int(gaps[idx].mean(axis=0).argmin())],
        rule=aggregate(rule[idx]), baseline=aggregate(gaps[idx, baseline_idx]), best_global=aggregate(gaps[idx, best_idx]))
        for f in ['causal_lm', 'resnet', 'vision_transformer', 'other']
        if (idx := [i for i, name in enumerate(names) if family(name) == f])}
    result['leave_one_workload_out'] = []
    for i, name in enumerate(names):
        keep = np.arange(n) != i
        j = int(gaps[keep].mean(axis=0).argmin())
        result['leave_one_workload_out'].append(dict(excluded=name, best_fixed_beta_other_ten=grid[j],
            remaining_suite=contrast(rule[keep], gaps[keep, j]),
            against_each_fixed={label:contrast(rule[keep],gaps[keep,k]) for label,k in comparators.items()},
            heldout_rule_gap=float(rule[i]), heldout_fixed_gap=float(gaps[i, j]),
            heldout_gap_benefit_pp=float(gaps[i, j] - rule[i])))
    result['leave_family_out'] = []
    for f in ['causal_lm', 'resnet', 'vision_transformer', 'other']:
        keep = np.array([family(name) != f for name in names])
        j = int(gaps[keep].mean(axis=0).argmin())
        result['leave_family_out'].append(dict(excluded=f, best_fixed_beta=grid[j], remaining_suite=contrast(rule[keep], gaps[keep, j])))
    timing_rows, beta_histories = [], [[] for _ in names]
    for step in range(150, 501, 25):
        ds = [select_beta(SUP / 'robustness_probes' / (name + '_mb16.npz'), step, 16, grid) for name in names]
        chosen = [grid.index(d['beta']) for d in ds]
        tg = gaps[np.arange(n), chosen]
        expected = next(r for r in timing_archive['summary'] if r['budget'] == 16 and r['step'] == step)
        for key,archived in [('mean','mean_gap_pct'),('maximum','max_gap_pct'),('cvar25','cvar25_gap_pct')]:
            assert abs(aggregate(tg)[key] - expected[archived]) < 1e-10
        for i, d in enumerate(ds): beta_histories[i].append(d['beta'])
        timing_rows.append(dict(step=step, metrics=aggregate(tg),
            comparisons={label: contrast(tg, gaps[:, j]) for label, j in comparators.items()},
            selections=[d['beta'] for d in ds], per_case_gap=tg.tolist()))
    result['timing'] = dict(rows=timing_rows,
        metric_ranges={k:dict(min=min(r['metrics'][k] for r in timing_rows), max=max(r['metrics'][k] for r in timing_rows),
            best_times=[r['step'] for r in timing_rows if r['metrics'][k] == min(s['metrics'][k] for s in timing_rows)],
            worst_times=[r['step'] for r in timing_rows if r['metrics'][k] == max(s['metrics'][k] for s in timing_rows)]) for k in ['mean','maximum','cvar25']},
        beats_fixed_counts={label:{k:sum(r['metrics'][k] < aggregate(gaps[:, j])[k] for r in timing_rows) for k in ['mean','maximum','cvar25']} for label,j in comparators.items()},
        per_workload=[dict(workload=name, beta_frequencies=dict(Counter(beta_histories[i])),
            canonical_agreement=sum(b == decisions[i]['beta'] for b in beta_histories[i]),
            gap_min=min(r['per_case_gap'][i] for r in timing_rows),gap_max=max(r['per_case_gap'][i] for r in timing_rows)) for i,name in enumerate(names)])
    result['budget'] = []
    for budget in [2,4,8,16]:
        ds=[select_beta(SUP/'robustness_probes'/(name+'_mb'+str(budget)+'.npz'),200,budget,grid) for name in names]
        bg=gaps[np.arange(n),[grid.index(d['beta']) for d in ds]]
        expected=next(r for r in budget_archive['summary'] if r['budget']==budget and r['factor']==4)
        for key,archived in [('mean','mean_gap_pct'),('maximum','max_gap_pct'),('cvar25','cvar25_gap_pct')]:
            assert abs(aggregate(bg)[key]-expected[archived])<1e-10
        result['budget'].append(dict(budget=budget,metrics=aggregate(bg),selections=[d['beta'] for d in ds],
            comparisons={label:contrast(bg,gaps[:,j]) for label,j in comparators.items()}))
    # Both fixed comparators are evaluated, and best fixed is reoptimized per resample.
    rng = np.random.default_rng(20261006)
    sample_idx = rng.integers(0, n, size=(100000, n))
    boot_rule = rule[sample_idx].mean(axis=1)
    boot_fixed = gaps[sample_idx].mean(axis=1)
    result['workload_bootstrap_descriptive'] = dict(draws=len(sample_idx), seed=20261006,
        interpretation='Empirical workload reweighting only; eleven nonrandom benchmark tasks, correlated models, one pilot seed; neither confidence guarantee nor independent validation.',
        comparisons={label:dict(quantiles_mean_gap_benefit_pp=np.quantile(boot_fixed[:,j]-boot_rule,[.025,.5,.975]).tolist(),
            fraction_positive=float(np.mean(boot_fixed[:,j]>boot_rule))) for label,j in comparators.items()},
        reoptimized_fixed=dict(quantiles_mean_gap_benefit_pp=np.quantile(boot_fixed.min(axis=1)-boot_rule,[.025,.5,.975]).tolist(),
            fraction_positive=float(np.mean(boot_fixed.min(axis=1)>boot_rule)),
            beta_frequencies=dict(Counter(grid[int(j)] for j in boot_fixed.argmin(axis=1)))))
    # Marginal stability: ratio threshold and exact adjacent loss costs, all tasks.
    result['rounding_loss_stability'] = []
    for i,(name,d) in enumerate(zip(names,decisions)):
        j = choice_idx[i]
        neighbors=[]
        for adj in [j-1,j+1]:
            if 0 <= adj < len(grid):
                midpoint=(grid[j]+grid[adj])/2
                multiplier=(1/((1-midpoint)*d['memory']))**3
                neighbors.append(dict(adjacent_beta=grid[adj], ratio_multiplier_at_boundary=multiplier,
                    ratio_change_pct=100*(multiplier-1), log_ratio_distance=abs(math.log(multiplier)),
                    adjacent_gap_pct=float(gaps[i,adj]), gap_change_pp=float(gaps[i,adj]-rule[i]),
                    relative_loss_change_pct=float(100*(losses[i,adj]/selected_losses[i]-1))))
        result['rounding_loss_stability'].append(dict(workload=name, selected_beta=grid[j], neighbors=neighbors))
    all_boundaries=[v for row in result['rounding_loss_stability'] for v in row['neighbors']]
    result['all_decisions_invariant_ratio_interval'] = dict(
        lower=max(v['ratio_multiplier_at_boundary'] for v in all_boundaries if v['ratio_multiplier_at_boundary']<1),
        upper=min(v['ratio_multiplier_at_boundary'] for v in all_boundaries if v['ratio_multiplier_at_boundary']>1),
        meaning='For each workload, any independent multiplicative perturbation of its V/D inside this open interval leaves all eleven grid decisions unchanged. Deterministic arithmetic property, not a sampling-accuracy guarantee.')
    interval=result['all_decisions_invariant_ratio_interval']
    result['coefficient_perturbation_bound']=dict(
        coefficient_relative_error_bound=.07,
        resulting_ratio_interval=[.93/1.07,1.07/.93],
        maximum_open_symmetric_relative_bound=min((1-interval['lower'])/(1+interval['lower']),
                                                 (interval['upper']-1)/(interval['upper']+1)),
        both_coefficients_10pct_not_certified=True,
        meaning='If each measured coefficient V and D is independently multiplied by any factor in [.93,1.07], their ratio remains in the deterministic safe interval. With both coefficients +/-10%, the largest ratio multiplier is 1.1/.9, outside that interval.')
    assert interval['lower'] < .93/1.07 <= 1.07/.93 < interval['upper']
    factors=[.5,.75,.9,.95,.975,1,1.025,1.05,1.1,1.25,1.5,2]
    result['common_ratio_perturbation'] = []
    for factor in factors:
        betas=[min(grid,key=lambda b:abs(b-(1-1/(d['memory']*factor**(1/3))))) for d in decisions]
        pg=gaps[np.arange(n),[grid.index(b) for b in betas]]
        result['common_ratio_perturbation'].append(dict(ratio_multiplier=factor,selections=betas,
            changed_workloads=[name for name,b,d in zip(names,betas,decisions) if b!=d['beta']], metrics=aggregate(pg)))
    paired=[]
    for c in cases:
        fixed=float(np.mean([s['fixed'] for s in c['common_seed_losses']]))
        selected=float(np.mean([s['selected'] for s in c['common_seed_losses']]))
        paired.append(dict(workload=c['workload'],seeds=[s['seed'] for s in c['common_seed_losses']],
            fixed_loss=fixed,selected_loss=selected,improvement_pct=100*(fixed-selected)/fixed))
    result['paired_training'] = dict(mean_direct_loss_improvement_pct=float(np.mean([r['improvement_pct'] for r in paired])),per_case=paired)
    result['provenance_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [SUP/'inputs.json',SUP/'timing_audit.json',SUP/'probe_budget_audit.json']}
    (OUTPUT/'strength_numerical.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    report = []
    report.append('Policy: 16 probes, factor 4, update 200 unchanged. CPU-only saved-data analysis.')
    report.append('DECISIVE DIAGNOSTIC 1: The two strongest fixed memories express an architectural compromise. Fixed .94377 minimizes the four causal-LM losses and the vision-transformer family mean; .96838 minimizes the ResNet family mean and the whole-suite fixed mean. Against .94377, ResNets supply 99.8505% of gross gains and 133.3624% of net gains. Against .96838 they contribute exactly zero: Swin and ViT-CIFAR supply all positive gains, partly offset by four causal LM losses and ViT-Tiny. This explains why the same measured rule can outperform both suite-wide compromises without dominating individual tasks.')
    report.append('DECISIVE DIAGNOSTIC 2: Arithmetic stability and loss consequence differ. All eleven beta decisions are invariant under independently perturbing each V/D ratio inside (0.8242860753, 1.1617178628). Swin is the first upward boundary (+16.1718% ratio), and crossing it increases its saved loss by 2.18625%. Seven workloads keep the same beta at all 15 times; 152/165 decisions match update 200. The four times failing against the best fixed value are 175, 225, 300 (Swin changes to .96838) and 325 (ViT-CIFAR changes to .9). These are descriptive facts from the same measured pilot; they do not certify coefficient sampling error or independent seed stability.')
    report.append('COEFFICIENT PRECISION: +/-10% of V/D preserves every decision. This does not mean independently perturbing both V and D by +/-10%, since 1.1/.9=1.22222 exceeds the upper margin. Independent +/-7% coefficient perturbations are sufficient: their ratio lies in [.93/1.07,1.07/.93]=[.86916,1.15054], strictly inside the safe interval. This is an arithmetic statement conditional on the measured coefficients, not a confidence bound on estimation error.')
    report.append('ARTICLE-SAFE WORDING: The rule addresses a compromise among task-specific memory requirements. On the seed-matched archive it retains the ResNet-compatible memory while avoiding the larger shared-memory penalty on Swin and ViT-CIFAR. This gains 2.5899 summed gap points relative to the strongest global fixed memory, offset by 0.6465 points of adverse changes in five tasks. Its mean direct loss improvement is 0.1726%, distinguishing the absolute loss scale from the 32.3% relative gap reduction. Timing sensitivity preserves mean and tail improvement against the strongest fixed memory in eleven of fifteen audited windows; all fifteen improve mean and CVaR against .94377. Removing Swin reverses the comparison with the strongest fixed memory, so the suite-level benefit is task-composition dependent.')
    report.append('All fixed-grid comparisons: '+json.dumps({k:v['reductions_pct'] for k,v in result['all_fixed_comparisons'].items()}))
    report.append('Attribution: '+json.dumps(result['attribution']))
    report.append('Family compromise: '+json.dumps(result['family_compromise']))
    report.append('Timing: '+json.dumps({k:v for k,v in result['timing'].items() if k!='rows'}))
    report.append('Leave-one-out: '+json.dumps(result['leave_one_workload_out']))
    report.append('Leave-family-out: '+json.dumps(result['leave_family_out']))
    report.append('Descriptive bootstrap: '+json.dumps(result['workload_bootstrap_descriptive']))
    report.append('Rounding and saved loss consequences: '+json.dumps(result['rounding_loss_stability']))
    report.append('Common ratio perturbations: '+json.dumps(result['common_ratio_perturbation']))
    report.append('Can claim: superior mean and tail gap versus every fixed grid value in matched seed-1 suite; task-dependent compromise, quantified per-case and timing sensitivity. Cannot claim: independent robustness, seed-general benefit, accurate population drift estimation, prospective protocol validation. Low probe budgets and common-seed adverse aggregate remain material.')
    (OUTPUT/'strength_numerical_notes.txt').write_text('\n\n'.join(report),encoding='utf-8')
    display = dict(efficientnetb0_stanfordcars='EfficientNet / Cars', llama60m_c4='Llama / C4',
        llama60m_slimpajama6b='Llama / SlimPajama',nanogpt_openwebtext='GPT / OpenWebText',
        nanogpt_wikitext103='GPT / WikiText',resnet50_food101='ResNet / Food',
        resnet50_imagenet100='ResNet / ImageNet100',swin_t_caltech256='Swin / Caltech',
        t5small_bookcorpus='T5 / BookCorpus',vitb16_cifar100='ViT / CIFAR',vitb16_tinyimagenet='ViT / TinyImageNet')
    def table(filename, header, columns, rows):
        body=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}'+columns+'@{}}',r'\toprule '+header+r'\\\midrule']
        body+=[' & '.join(str(x) for x in row)+r'\\' for row in rows]
        body+=[r'\bottomrule\end{tabular*}']
        (OUTPUT/filename).write_text('\n'.join(body),encoding='utf-8')
    table('strength_summary_table.tex',r'Method & Mean gap & Maximum & CVaR$_{25}$','lrrr',
        [[label]+[f'{aggregate(values)[key]:.4f}' for key in ['mean','maximum','cvar25']]
         for label,values in [('Fixed 0.94377',gaps[:,baseline_idx]),('Best global fixed 0.96838',gaps[:,best_idx]),('Measured rule',rule)]])
    table('strength_case_attribution.tex',r'& & & \multicolumn{2}{c}{Gap benefit (pp)} & \multicolumn{2}{c}{Loss improvement (\%)}\\ \cmidrule(lr){4-5}\cmidrule(lr){6-7} Workload & $\widehat\beta$ & Rule gap & .94377 & .96838 & .94377 & .96838',
        'lrrrrrr',[[display[r['workload']],f"{r['selected_beta']:.5f}",f"{r['rule_gap_pct']:.4f}",
        f"{r['baseline']['gap_benefit_pp']:+.4f}",f"{r['best_global']['gap_benefit_pp']:+.4f}",
        f"{r['baseline']['direct_loss_improvement_pct']:+.4f}",f"{r['best_global']['direct_loss_improvement_pct']:+.4f}"] for r in result['per_case']])
    table('strength_leave_one_out.tex',r'& & \multicolumn{2}{c}{Mean-gap reduction (\%)} & \multicolumn{2}{c}{Best fixed on ten}\\ \cmidrule(lr){3-4}\cmidrule(lr){5-6} Excluded & Rule mean & vs .94377 & vs .96838 & Beta & Reduction (\%)',
        'lrrrrr',[[display[r['excluded']],f"{r['remaining_suite']['rule']['mean']:.4f}",
        f"{r['against_each_fixed']['baseline']['reductions_pct']['mean']:+.2f}",
        f"{r['against_each_fixed']['best_global']['reductions_pct']['mean']:+.2f}",
        f"{r['best_fixed_beta_other_ten']:.5f}",f"{r['remaining_suite']['reductions_pct']['mean']:+.2f}"] for r in result['leave_one_workload_out']])
    table('strength_all_timing.tex',r'& & & & \multicolumn{2}{c}{Mean-gap reduction (\%)}\\ \cmidrule(lr){5-6} Update & Mean gap & Maximum & CVaR$_{25}$ & vs .94377 & vs .96838',
        'rrrrrr',[[r['step']]+[f"{r['metrics'][k]:.4f}" for k in ['mean','maximum','cvar25']]+
        [f"{r['comparisons'][label]['reductions_pct']['mean']:+.2f}" for label in ['baseline','best_global']] for r in timing_rows])
    table('strength_timing_frequencies.tex',r'Workload & Selected beta: count & Matches & Min. gap & Max. gap',
        'llrrr',[[display[r['workload']],', '.join(f'{b:.5f}: {count}' for b,count in sorted(r['beta_frequencies'].items())),
        f"{r['canonical_agreement']}/15",f"{r['gap_min']:.4f}",f"{r['gap_max']:.4f}"] for r in result['timing']['per_workload']])
    table('strength_marginal_stability.tex',r'Workload & $\widehat\beta$ & Adjacent beta & $\Delta$ ratio (\%) & $\Delta$ gap (pp) & $\Delta$ loss (\%)',
        'lrrrrr',[[display[r['workload']],f"{r['selected_beta']:.5f}",f"{v['adjacent_beta']:.5f}",
        f"{v['ratio_change_pct']:+.2f}",f"{v['gap_change_pp']:+.4f}",f"{v['relative_loss_change_pct']:+.4f}"]
        for r in result['rounding_loss_stability'] if (v:=min(r['neighbors'],key=lambda v:v['log_ratio_distance']))])
    print(json.dumps(dict(attribution=result['attribution'],family_compromise=result['family_compromise'],
        timing_ranges=result['timing']['metric_ranges'],timing_counts=result['timing']['beats_fixed_counts'],
        leave_out=[dict(excluded=r['excluded'],beta=r['best_fixed_beta_other_ten'],reduction=r['remaining_suite']['reductions_pct']['mean']) for r in result['leave_one_workload_out']],
        leave_family_out=result['leave_family_out'],bootstrap=result['workload_bootstrap_descriptive']),indent=2))


if __name__=='__main__': main()
