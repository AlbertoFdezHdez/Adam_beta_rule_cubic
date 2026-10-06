"""Reconstruct all three trajectory summaries from anonymous validation curves.

No training. Default exports trusted local archived pickles; --inputs uses only
anonymous JSON, suitable for the supplement. Fixed selections never change.
"""
import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs',type=Path)
parser.add_argument('--output',type=Path,default=HERE)
args=parser.parse_args()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=True)


def summarize(values,metric):
    return min(values) if metric=='minimum' else values[-1] if metric=='last' else statistics.mean(values[-3:])


def aggregate(values):
    return dict(mean=statistics.mean(values),maximum=max(values),cvar25=statistics.mean(sorted(values,reverse=True)[:math.ceil(len(values)/4)]))


def export():
    root=HERE.parents[1]
    policy=json.loads((HERE/'policy_audit.json').read_text())
    prefixes={r['exp_id']:r['result_prefix'] for r in csv.DictReader((root/'adam_tracking_v3_evaluation/outputs/tables/beta_sweep_curves.tsv').open(),delimiter='\t')}
    cases=[]
    for row in policy['rows']:
        curves={}
        for beta in policy['observed_grid']:
            seeds=sorted(set([1]+(row['paired']['seeds'] if beta in [.94377,row['beta']] else [])))
            for seed in seeds:
                path=root/'descargas/results'/f"{prefixes[row['exp']]}_beta{beta:g}_seed{seed}.pkl"
                if not path.exists():
                    curves[f'{beta}:{seed}']=dict(missing=True)
                    continue
                payload=pickle.load(path.open('rb'))
                history=payload['history']
                values=list(map(float,history['val_loss']))
                coords=next((list(history[k]) for k in ['step','iter','epoch'] if k in history and len(history[k])==len(values)),None)
                sha=hashlib.sha256(path.read_bytes()).hexdigest()
                archived=policy['provenance'].get(str(path.relative_to(root)))
                if archived: assert sha==archived
                if seed==1: assert abs(min(values)-row['seed1_grid_losses'][str(beta)])<1e-12
                curves[f'{beta}:{seed}']=dict(validation_loss=values,coordinates=coords if coords is not None else list(range(1,len(values)+1)),
                    coordinate_source='explicit' if coords is not None else 'implicit_epoch_order',
                    source_file=path.name,source_sha256=sha,validated_primary_minimum=seed==1,
                    matched_policy_source_hash=bool(archived))
        cases.append(dict(workload=row['exp'],selected_beta=row['beta'],common_seeds=row['paired']['seeds'],curves=curves))
    result=dict(scope='Anonymous validation loss trajectories from the same completed archive; no new training or beta decisions.',
        grid=policy['observed_grid'],baseline=.94377,cases=cases)
    (OUT/'validation_curves.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    data=json.loads(args.inputs.read_text()) if args.inputs else export()
    grid=data['grid']
    audits=[]
    limitations=[]
    eligible=[]
    for c in data['cases']:
        cand=[c['curves'][f'{beta}:1'] for beta in grid]
        complete=all(not r.get('missing') for r in cand)
        matched=complete and all(r['coordinates']==cand[0]['coordinates'] and len(r['validation_loss'])==len(cand[0]['validation_loss']) for r in cand)
        paired_checks=[]
        for seed in c['common_seeds']:
            f=c['curves'][f"{data['baseline']}:{seed}"]
            s=c['curves'][f"{c['selected_beta']}:{seed}"]
            okay=not f.get('missing') and not s.get('missing') and f['coordinates']==s['coordinates']==cand[0]['coordinates'] and len(f['validation_loss'])==len(s['validation_loss'])
            paired_checks.append(dict(seed=seed,matched=okay))
        audit=dict(workload=c['workload'],grid_complete=complete,grid_counts_and_schedule_match=matched,
            candidate_count=len(cand),validation_count=len(cand[0].get('validation_loss',[])),
            schedule_coordinate_source=cand[0].get('coordinate_source'),common_seed_checks=paired_checks)
        audits.append(audit)
        if matched and all(x['matched'] for x in paired_checks): eligible.append(c)
        else: limitations.append(audit)
    result=dict(scope='Retrospective sensitivity to summaries of the same saved trajectories, not independent replication; all three summaries reported.',
        schedule_audits=audits,limitations=limitations,
        schedule_limit='Vision files store epoch-ordered validation arrays without explicit coordinates. Equal counts and archive per-epoch protocol support matched epoch order; explicit iteration coordinates are verified for language runs.',
        summaries={})
    result['anonymous_curve_count']=sum(not v.get('missing') for c in data['cases'] for v in c['curves'].values())
    for metric in ['minimum','last','last3_mean']:
        rows=[]
        grid_gaps={str(beta):[] for beta in grid}
        rule_gaps=[]
        for c in eligible:
            pairs=[]
            for seed in c['common_seeds']:
                fixed=summarize(c['curves'][f"{data['baseline']}:{seed}"]['validation_loss'],metric)
                rule=summarize(c['curves'][f"{c['selected_beta']}:{seed}"]['validation_loss'],metric)
                pairs.append(dict(seed=seed,fixed_loss=fixed,rule_loss=rule,relative_improvement_pct=100*(fixed-rule)/fixed))
            fm=statistics.mean(r['fixed_loss'] for r in pairs)
            rm=statistics.mean(r['rule_loss'] for r in pairs)
            candidates={beta:summarize(c['curves'][f'{beta}:1']['validation_loss'],metric) for beta in grid}
            oracle=min(candidates.values())
            rg=100*(candidates[c['selected_beta']]/oracle-1)
            rule_gaps.append(rg)
            for beta in grid: grid_gaps[str(beta)].append(100*(candidates[beta]/oracle-1))
            rows.append(dict(workload=c['workload'],selected_beta=c['selected_beta'],common_seeds=c['common_seeds'],
                per_seed=pairs,paired_ratio_of_mean_losses_improvement_pct=100*(fm-rm)/fm,
                paired_mean_of_seed_relative_improvements_pct=statistics.mean(r['relative_improvement_pct'] for r in pairs),
                seed1_improvement_pct=next(r['relative_improvement_pct'] for r in pairs if r['seed']==1),
                seed1_rule_gap_pct=rg,seed1_oracle_beta=min(candidates,key=candidates.get)))
        fixed_summary={beta:aggregate(v) for beta,v in grid_gaps.items()}
        rule_summary=aggregate(rule_gaps)
        best=min(fixed_summary,key=lambda beta:fixed_summary[beta]['mean'])
        metric_result=dict(per_case=rows,eligible_workloads=len(rows),rule_gap_summary=rule_summary,
            all_fixed_grid_gap_summaries=fixed_summary,best_global_fixed_beta=float(best),
            mean_paired_ratio_of_mean_losses_improvement_pct=statistics.mean(r['paired_ratio_of_mean_losses_improvement_pct'] for r in rows),
            mean_paired_mean_of_seed_relative_improvements_pct=statistics.mean(r['paired_mean_of_seed_relative_improvements_pct'] for r in rows),
            mean_seed1_improvement_pct=statistics.mean(r['seed1_improvement_pct'] for r in rows),
            gap_reduction_pct={label:{key:100*(1-rule_summary[key]/fixed_summary[str(beta)][key]) for key in rule_summary}
                for label,beta in [('baseline',data['baseline']),('best_global',float(best))]},
            beats_all_fixed={key:all(rule_summary[key]<v[key] for v in fixed_summary.values()) for key in rule_summary})
        result['summaries'][metric]=metric_result
    (OUT/'strength_trajectory.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=[result['scope'],result['schedule_limit'],f'Excluded or incomplete: {len(limitations)}']
    for name,s in result['summaries'].items():
        lines.append(f"{name}: seed1 mean direct improvement {s['mean_seed1_improvement_pct']:.6f}%; paired ratio-of-means improvement {s['mean_paired_ratio_of_mean_losses_improvement_pct']:.6f}%; mean per-seed relative improvement {s['mean_paired_mean_of_seed_relative_improvements_pct']:.6f}%. Rule gaps {s['rule_gap_summary']}; best fixed beta {s['best_global_fixed_beta']}; beats all fixed {s['beats_all_fixed']}.")
        lines.extend(f"  {r['workload']}: seed1 {r['seed1_improvement_pct']:+.6f}%; paired ratio-of-means {r['paired_ratio_of_mean_losses_improvement_pct']:+.6f}%; mean-of-seed-percentages {r['paired_mean_of_seed_relative_improvements_pct']:+.6f}%; seeds {r['common_seeds']}" for r in s['per_case'])
    (OUT/'strength_trajectory_notes.txt').write_text('\n'.join(lines),encoding='utf-8')
    table=[r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',r'\toprule & \multicolumn{3}{c}{Mean relative gap (\%)} & & \multicolumn{2}{c}{Direct improvement (\%)}\\ \cmidrule(lr){2-4}\cmidrule(lr){6-7} Summary & Rule & Fixed .94377 & Best fixed & Fixed beta & Seed 1 & Common seeds\\\midrule']
    labels=dict(minimum='Minimum recorded',last='Last recorded',last3_mean='Mean of last three')
    for metric,s in result['summaries'].items():
        table.append(' & '.join([labels[metric],r'\textbf{'+f"{s['rule_gap_summary']['mean']:.4f}"+'}',
            f"{s['all_fixed_grid_gap_summaries']['0.94377']['mean']:.4f}",
            f"{s['all_fixed_grid_gap_summaries'][str(s['best_global_fixed_beta'])]['mean']:.4f}",
            f"{s['best_global_fixed_beta']:.5f}",f"{s['mean_seed1_improvement_pct']:+.4f}",
            f"{s['mean_paired_ratio_of_mean_losses_improvement_pct']:+.4f}"])+r'\\')
    table.append(r'\bottomrule\end{tabular*}')
    (OUT/'strength_trajectory_table.tex').write_text('\n'.join(table),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__': main()
