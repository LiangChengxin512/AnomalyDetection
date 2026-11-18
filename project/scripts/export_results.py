"""Schema-versioned long tables, explicit missing values and reproducible plot inputs."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.common.io_utils import read_json,write_json,sha256_file
from src.data.prepare_factory import path_for
from src.baselines.registry import DETECTORS
from src.bench.statistics import paired_interval


def export_results(root,manifest,tasks,detection,business,rca,fitted,cfg,site):
    tables=root/'tables';plots=root/'plot_data'
    tables.mkdir(exist_ok=True);plots.mkdir(exist_ok=True)
    env=manifest['environment']
    common=dict(schema_version='2.0',run_id=manifest['run_id'],dataset_id=site.get('dataset_id',site['site_id']),
        dataset_fingerprint=manifest['dataset_fingerprint'],scope=manifest['scope'],config_hash=manifest['config_hash'],
        code_hash=env['code_hash'],git_commit=env['git_commit'],git_dirty=env['git_dirty'],device=env['selected_device'],
        os=env['os'],python_version=env['python_version'],torch_version=env['torch_version'],created_at=manifest['created_at'],
        graph_version=sha256_file(path_for(site['process']['graph'])) if site['process'].get('graph') else None,
        catalog_version=sha256_file(path_for(site['decision']['action_catalog'])),split_id='test')
    rows={name:[] for name in ['metrics_detection','metrics_rca','metrics_policy','metrics_ablation','metrics_negative_controls','metrics_efficiency','metrics_ci']}
    def metric(table,experiment,seed,name,value,**extra):
        detector_id=experiment if experiment.startswith('D') else 'D07'
        token=f'{detector_id}_seed_{seed if seed is not None else "deterministic"}'
        threshold_path=root/'scores'/token/'threshold.json'
        binding=read_json(threshold_path) if threshold_path.exists() else {}
        row=dict(common,experiment_id=experiment,variant_id=experiment if experiment.startswith('B') else '',
            model_id=DETECTORS.get(experiment,'TranAD_C_G_E'),seed=seed,metric=name,value=value,
            checkpoint_sha256=binding.get('weight_sha256') if detector_id in ['D06','D07'] else None,
            estimator_sha256=binding.get('weight_sha256') if detector_id not in ['D06','D07'] else None,
            threshold_id=binding.get('threshold_id'),
            status='SUCCEEDED' if value is not None and not pd.isna(value) else 'NOT_APPLICABLE',reason='',**extra)
        if row['status']=='NOT_APPLICABLE':row['reason']=extra.get('reason') or ('Author VUS implementation not installed/validated' if name=='VUS_PR' else
            'Action response is not qualified; surrogate effects unavailable' if name=='surrogate_only_risk' else 'Undefined denominator, no eligible observations, or optional metric dependency unavailable')
        rows[table].append(row)
    event_detection=[];event_rca=[];event_policy=[];policy_decisions=[];candidates=[];constraints=[];latency=[]
    training=[];summaries=[];ci_rows=[]
    for (model,seed),path in fitted.items():
        for filename,collector in [('training_history.csv',training),('training_summary.csv',summaries)]:
            if (path/filename).exists():collector.append(pd.read_csv(path/filename))
    for (model,seed),path in detection.items():
        if not (path/'metrics.json').exists():continue
        metrics=read_json(path/'metrics.json')
        token=f'{model}_seed_{seed if seed is not None else "deterministic"}'
        scoring=root/'scores'/token
        if (scoring/'efficiency.json').exists():
            eff=read_json(scoring/'efficiency.json')
            for name in ['p50_ms','p95_ms','throughput_windows_per_second','memory_bytes']:
                metric('metrics_efficiency',model,seed,name,eff[name],batch=1,windows=eff['windows'],
                       warmup_count=eff['warmup_count'],memory_kind=eff['memory_kind'],inference_source='measured_batch_1')
        for name,value in metrics.items():
            if name in ['reason','status']:continue
            metric('metrics_detection',model,seed,name,value,n_samples=None,n_events=metrics.get('n_events'),normal_hours=metrics.get('normal_hours'),
                   protocol='unadjusted points; segment-safe one-to-one intervals; separate any-overlap coverage')
        if (path/'event_detection.csv').exists():
            f=pd.read_csv(path/'event_detection.csv');f['experiment_id']=model;f['seed']=seed;event_detection.append(f)
        if (path/'ci.json').exists():ci_rows.append(dict(common,experiment_id=model,seed=seed,**read_json(path/'ci.json')))
        for name in ['pr_curve','roc_curve','score_timeline']:
            file=path/(name+'.csv')
            if file.exists():
                f=pd.read_csv(file);f['experiment_id']=model;f['seed']=seed
                f.to_csv(plots/f'{name}_{model}_seed_{seed}.csv',index=False)
    truth=pd.concat(event_detection,ignore_index=True) if event_detection else pd.DataFrame()
    labels=pd.read_parquet(root/'evaluation/evaluation_labels.parquet') if (root/'evaluation/evaluation_labels.parquet').exists() else pd.DataFrame()
    for (experiment,seed),path in business.items():
        if not (path/'workflow_metrics.json').exists():continue
        metrics=read_json(path/'workflow_metrics.json')
        advice=pd.read_csv(path/'policy_decisions.csv')
        score=pd.read_parquet(path/'scores.parquet')
        if len(labels):
            data=score.merge(labels[['sample_id','label']],on='sample_id',validate='one_to_one')
            normal=data.loc[~data.warmup & data.label.eq(0)]
            hours=len(normal)*site['input']['sample_interval_seconds']/3600
            joined=advice.merge(data[['sample_id','label']],on='sample_id',validate='many_to_one')
            metrics['normal_adjustment_advice_per_hour']=float((joined.label.eq(0)&joined.physical_adjustment.eq(True)).sum()/hours) if hours else None
            events=truth.loc[truth.experiment_id.eq('D07') & truth.seed.eq(seed) & truth.record_type.eq('truth')] if len(truth) else pd.DataFrame()
            for e in events.to_dict('records'):
                sub=data.loc[data.source_file.eq(e['source_file']) & pd.to_datetime(data.timestamp).between(pd.Timestamp(e['start']),pd.Timestamp(e['end']))]
                decisions=advice.loc[advice.sample_id.isin(sub.sample_id)]
                event_policy.append(dict(gt_event_id=e['gt_event_id'],source_file=e['source_file'],experiment_id=experiment,seed=seed,
                    detected=bool(sub.alarm_id.notna().any()),alarms=int(sub.alarm_opened.sum()),advice_requests=len(decisions),
                    n_offered=int(decisions.action_id.notna().sum()),action_cost=float(decisions.cost.sum()),
                    unique_cases=sub.alarm_id.nunique(),duplicates=int(decisions.duplicate.sum())))
        for name,value in metrics.items():
            metric('metrics_ablation' if experiment.startswith('B') else 'metrics_policy',experiment,seed,name,value,
                n_requests=metrics.get('n_requests'),n_offered=metrics.get('n_offered'),n_abstained=metrics.get('n_abstained'),
                surrogate_only=name=='surrogate_only_risk',reference_id='B00' if experiment.startswith('B') else 'P00')
        for filename,collector in [('policy_decisions.csv',policy_decisions),('rootcause_candidates.csv',candidates),('policy_constraints.csv',constraints)]:
            f=pd.read_csv(path/filename);f['experiment_id']=experiment;f['seed']=seed;collector.append(f)
        if (path/'latency_samples.csv').exists():
            f=pd.read_csv(path/'latency_samples.csv')
            if len(f):
                for column in ['explanation_ms','end_to_end_ms']:
                    metric('metrics_efficiency',experiment,seed,column+'_p50',f[column].median(),batch=1,inference_source='precomputed')
                    metric('metrics_efficiency',experiment,seed,column+'_p95',f[column].quantile(.95),batch=1,inference_source='precomputed')
    for (method,seed),path in rca.items():
        if not (path/'event_rca.csv').exists():continue
        frame=pd.read_csv(path/'event_rca.csv');frame['seed']=seed;frame['experiment_id']=method;event_rca.append(frame)
        for scope in ['strict_observable_target','associated_loop_tags']:
            for offset in [0,5,10]:
                subset=frame.loc[frame.target_scope.eq(scope)&frame.snapshot_seconds.eq(offset)]
                for conditioning in ['end_to_end','detected']:
                    eligible=subset.loc[subset.eligible.eq(True)]
                    if conditioning=='detected':eligible=eligible.loc[eligible.detected.eq(True)]
                    for name in ['top1','top3','top5','MRR']:
                        metric('metrics_rca',method,seed,name,float(eligible[name].mean()) if len(eligible) else None,
                            target_scope=scope,snapshot_seconds=offset,conditioning=conditioning,n_eligible=len(eligible),
                            n_evaluated=int(eligible.explained.sum()),unobservable_events=int((~subset.eligible).sum()) if len(subset) else 0)
    # Preserve required tasks that failed in the summary instead of dropping them.
    for task in tasks.rows:
        if task['status'] not in ['FAILED','SKIPPED_DEPENDENCY']:continue
        experiment=task['task_id'].split('_')[0]
        table='metrics_detection' if task['task_id'].startswith(('fit_','score_','detect_')) else 'metrics_rca' if experiment.startswith('R') else 'metrics_ablation' if experiment.startswith('B') else 'metrics_policy'
        rows[table].append(dict(common,experiment_id=experiment,seed=None,metric='task_status',value=None,status=task['status'],reason=task['reason']))
    for task in tasks.rows:
        experiment=task['task_id'].split('_')[0]
        if not experiment.startswith('N'):continue
        path=root/task['artifacts'];seed=int(task['task_id'].split('_')[-1])
        metric('metrics_negative_controls',experiment,seed,'execution_passed',1 if task['status'] in ['SUCCEEDED','REUSED'] else None,control_seed=seed)
        if (path/'event_rca.csv').exists():
            f=pd.read_csv(path/'event_rca.csv')
            for scope in ['strict_observable_target','associated_loop_tags']:
                subset=f.loc[f.eligible.eq(True)&f.target_scope.eq(scope)&f.snapshot_seconds.eq(0)]
                metric('metrics_negative_controls',experiment,seed,'top3',float(subset.top3.mean()) if len(subset) else None,target_scope=scope)
        if (path/'workflow_metrics.json').exists():
            for name,value in read_json(path/'workflow_metrics.json').items():metric('metrics_negative_controls',experiment,seed,name,value)
    if event_rca:
        all_rca=pd.concat(event_rca,ignore_index=True)
        for seed in sorted(all_rca.seed.unique()):
            for scope in all_rca.target_scope.unique():
                subset=all_rca.loc[all_rca.seed.eq(seed)&all_rca.target_scope.eq(scope)&all_rca.snapshot_seconds.eq(0)&all_rca.eligible.eq(True)]
                reference=subset.loc[subset.experiment_id.eq('R05')]
                for method in ['R01','R02','R03','R04']:
                    variant=subset.loc[subset.experiment_id.eq(method)]
                    ci=paired_interval(reference,variant,['gt_event_id'],'top3',cfg['eval']['bootstrap_samples'],cfg['eval']['bootstrap_seed'])
                    ci_rows.append(dict(common,experiment_id=method,seed=seed,metric='top3_delta_vs_R05',target_scope=scope,**ci,
                        resampling_unit='paired attack event, within seed',B=cfg['eval']['bootstrap_samples'],bootstrap_seed=cfg['eval']['bootstrap_seed']))
    if event_policy:
        work=pd.DataFrame(event_policy)
        for seed in work.seed.unique():
            reference=work.loc[work.seed.eq(seed)&work.experiment_id.eq('B00')]
            for experiment in work.experiment_id.unique():
                if not experiment.startswith('B') or experiment=='B00':continue
                variant=work.loc[work.seed.eq(seed)&work.experiment_id.eq(experiment)]
                for value in ['advice_requests','action_cost','detected']:
                    ref=reference.copy();var=variant.copy();ref[value]=ref[value].astype(float);var[value]=var[value].astype(float)
                    ci=paired_interval(ref,var,['gt_event_id'],value,cfg['eval']['bootstrap_samples'],cfg['eval']['bootstrap_seed'])
                    ci_rows.append(dict(common,experiment_id=experiment,seed=seed,metric=value+'_delta_vs_B00',**ci,resampling_unit='paired attack event, within seed',B=cfg['eval']['bootstrap_samples']))
                    metric('metrics_ablation',experiment,seed,value+'_per_attack',float(var[value].mean()) if len(var) else None,
                           reference_id='B00',paired_delta=ci['estimate'],n_pairs=ci['n_pairs'],removed_capability=experiment)
    def save(name,frames,columns):
        f=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)
        if 'schema_version' not in f:f.insert(0,'schema_version','2.0')
        f.to_csv(tables/(name+'.csv'),index=False)
    for name,data in rows.items():
        if name=='metrics_ci':data=ci_rows
        frame=pd.DataFrame(data,columns=None if data else ['schema_version','experiment_id','seed','metric','value','status','reason'])
        frame.to_csv(tables/(name+'.csv'),index=False);frame.to_csv(plots/(name+'.csv'),index=False)
        if len(frame) and 'value' in frame:
            good=frame.loc[frame.status.eq('SUCCEEDED')].copy();good['value']=pd.to_numeric(good.value,errors='coerce')
            grouping=[k for k in ['experiment_id','metric','target_scope','conditioning','snapshot_seconds'] if k in good]
            if len(good):
                summary=good.groupby(grouping,dropna=False).value.agg(['mean','std','count']).reset_index()
                summary.to_csv(tables/('summary_'+name.removeprefix('metrics_')+'.csv'),index=False)
    for name,frames,cols in [('training_history',training,['model_id','seed','epoch','train_loss','val_recon_mse']),
        ('training_summary',summaries,['model_id','seed','best_epoch','stop_reason']),('event_detection',event_detection,['gt_event_id','alarm_id','matched','delay','missed']),
        ('event_rca',event_rca,['gt_event_id','method','target_scope','top3','MRR']),('policy_decisions',policy_decisions,['advice_id','alarm_id','mode','reason']),
        ('rootcause_candidates',candidates,['alarm_id','tag','rank','score']),('policy_constraints',constraints,['action_id','constraint_name','passed'])]:save(name,frames,cols)
    save('event_policy',[pd.DataFrame(event_policy)] if event_policy else [],['gt_event_id','experiment_id','seed','advice_requests','action_cost'])
    save('alarm_workload',[pd.DataFrame(event_policy)] if event_policy else [],['gt_event_id','experiment_id','seed','alarms','unique_cases','duplicates'])
    for name,cols in [('policy_rollout',['schema_version','status','reason','surrogate_only']),('coverage_risk',['schema_version','status','reason','coverage','risk'])]:
        pd.DataFrame([dict(schema_version='2.0',status='NOT_APPLICABLE',reason='Action response not qualified',surrogate_only=True)]).reindex(columns=cols).to_csv(tables/(name+'.csv'),index=False)
    if (root/'surrogate/surrogate_validation.csv').exists():
        import shutil
        shutil.copyfile(root/'surrogate/surrogate_validation.csv',tables/'surrogate_validation.csv')
    pd.read_csv(tables/'training_history.csv').to_csv(plots/'training_history.csv',index=False)
    from src.bench.analysis_exports import export_sensitivity,export_coverage
    if detection and tasks.status.get('surrogate') in ['SUCCEEDED','REUSED']:
        export_sensitivity(root,cfg,site)
        export_coverage(root,cfg,site,business)
    write_json(root/'statistics_manifest.json',dict(bootstrap_seed=cfg['eval']['bootstrap_seed'],B=cfg['eval']['bootstrap_samples'],
        normal_block_seconds=max(10*cfg['window']['length']*site['input']['sample_interval_seconds'],1800),
        seed_uncertainty='mean/std across seeds; deterministic methods counted once',event_uncertainty='paired event bootstrap within each seed',
        missing_values='empty numeric cells with status/reason, never zero imputation'))
