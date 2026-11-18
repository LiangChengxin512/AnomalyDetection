#!/usr/bin/env python3
"""Single cross-platform entry point for training, experiments and deployable bundles."""
from __future__ import annotations
import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='config/experiment.yaml');p.add_argument('--site',default='config/sites/hai2305.yaml')
    p.add_argument('--suite',choices=['full','smoke','business_ablation','onboarding'],default='full')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda']);p.add_argument('--seeds')
    p.add_argument('--max-epochs',type=int);p.add_argument('--epochs',type=int);p.add_argument('--patience',type=int)
    p.add_argument('--min-delta-abs',type=float);p.add_argument('--physical-batch-size',type=int)
    p.add_argument('--replay-physical-batch-size',type=int,help='Override downstream attribution batches without changing the training contract')
    p.add_argument('--models',help='Comma-separated detector IDs or model names');p.add_argument('--experiments',help='Comma-separated D/R/P/B/N experiment IDs')
    p.add_argument('--run-id');p.add_argument('--resume',action='store_true');p.add_argument('--train-only',action='store_true')
    p.add_argument('--skip-train',action='store_true');p.add_argument('--checkpoint-manifest')
    p.add_argument('--export-csv',action='store_true',help='CSV is always exported; retained for explicit scripts')
    p.add_argument('--export-bundle',action='store_true');p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--mock',action='store_true');p.add_argument('--smoke-rows',type=int)
    p.add_argument('--smoke-replay-rows',type=int,default=0,help='Separate, explicit test-prefix budget for smoke only')
    p.add_argument('--max-windows',type=int,help='Deprecated smoke-only alias for --smoke-rows')
    p.add_argument('--dry-run',action='store_true',help='Resolve protocol and registry without training')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.replay_physical_batch_size is not None and args.replay_physical_batch_size<1:raise ValueError('Replay physical batch size must be positive')
    if args.mock:raise ValueError('The continuation pipeline requires real data/models. Historical MOCK artifacts are not accepted.')
    if args.epochs is not None:
        if args.max_epochs is not None and args.max_epochs!=args.epochs:raise ValueError('--epochs and --max-epochs disagree')
        print('Deprecated: --epochs means total --max-epochs, including resumed epochs.',file=sys.stderr);args.max_epochs=args.epochs
    if args.skip_train and not args.checkpoint_manifest:raise ValueError('--skip-train requires --checkpoint-manifest')
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from src.common.config import load_config,load_yaml
    from src.common.device import get_best_device
    from src.common.io_utils import read_json,write_json,sha256_file
    from src.common.provenance import digest,source_hash,environment,run_lock,inventory,now
    from src.common.tasks import Tasks
    from src.data.prepare_factory import load_site,prepare,input_fingerprint,path_for,load_frames
    from src.baselines.registry import DETECTORS,DETERMINISTIC
    from src.models.train import train_model
    from src.detect.evaluate_detector import fit_estimator,calibrate_and_replay
    from src.bench.evaluation_data import load_labels
    from src.bench.protocol_metrics import evaluate_detection
    from src.bench.continuation import business_replay,evaluate_rca,rewire_graph
    from src.sim.qualified import fit_surrogates
    from scripts.preflight import check
    cfg=load_config(args.config);cfg.pop('_config_path',None)
    suite=load_yaml(f'config/suites/{args.suite}.yaml');site=load_site(args.site)
    cfg['window']['length']=site['model']['window_samples'];cfg['model']['n_window']=cfg['window']['length']
    cfg['training']['max_epochs']=args.max_epochs if args.max_epochs is not None else (2 if args.suite=='smoke' else cfg['training']['max_epochs'])
    if args.patience is not None:cfg['training']['early_stopping']['patience']=args.patience
    if args.min_delta_abs is not None:cfg['training']['early_stopping']['min_delta_abs']=args.min_delta_abs
    if args.physical_batch_size is not None:cfg['training']['physical_batch_size']=args.physical_batch_size
    if cfg['training']['max_epochs']<1 or cfg['training']['early_stopping']['patience']<1:raise ValueError('Epoch budget and patience must be positive')
    seeds=[int(s) for s in args.seeds.split(',')] if args.seeds else suite['seeds']
    if len(set(seeds))!=len(seeds) or not seeds:raise ValueError('Seeds must be unique and nonempty')
    inverse={v:k for k,v in DETECTORS.items()}
    models=[inverse.get(m,m) for m in args.models.split(',')] if args.models else suite['models']
    if args.train_only and not args.models:models=['D06','D07']
    if any(m not in DETECTORS for m in models):raise ValueError('Unknown detector ID/name')
    experiments=set(args.experiments.split(',')) if args.experiments else None
    known=set(DETECTORS)|set(suite['required_rca'])|set(suite['required_policies'])|set(suite['required_ablations'])|set(suite['negative_controls'])
    if experiments and not experiments<=known:raise ValueError('Unknown experiment ID')
    smoke_rows=args.smoke_rows if args.smoke_rows is not None else args.max_windows if args.max_windows is not None else suite['smoke_rows']
    if smoke_rows<0 or (smoke_rows and args.suite!='smoke'):raise ValueError('Data caps are allowed only in the separate smoke suite')
    if args.smoke_replay_rows<0 or (args.smoke_replay_rows and args.suite!='smoke'):raise ValueError('Replay caps are allowed only for smoke')
    if smoke_rows and smoke_rows<cfg['window']['length']+2:raise ValueError('Smoke rows must exceed the history window')
    device=get_best_device(args.device,configured=cfg.get('device','auto'));cfg['device']=str(device)
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    run_id=args.run_id or time.strftime(args.suite+'_%Y%m%d_%H%M%S')
    if Path(run_id).name!=run_id or run_id in ['.','..']:raise ValueError('Run ID must be one directory name')
    root=PROJECT/'runs'/run_id
    resolved=dict(config=cfg,site=site,suite=suite,seeds=seeds,models=models,experiments=sorted(experiments) if experiments else None,
                  smoke_rows=smoke_rows,smoke_replay_rows=args.smoke_replay_rows,train_only=args.train_only,export_bundle=args.export_bundle)
    if args.replay_physical_batch_size is not None:resolved['replay_physical_batch_size']=args.replay_physical_batch_size
    if args.dry_run:
        print(yaml.safe_dump(resolved,sort_keys=False));return 0
    if root.exists() and not args.resume:raise FileExistsError(f'{root} exists; use --resume or a new --run-id')
    with run_lock(root/'.lock'):
        env=check(args.site,args.device,cfg.get('device','auto'))
        fingerprint,input_hashes=input_fingerprint(site)
        frozen=digest(resolved)
        old=read_json(root/'run_manifest.json') if (root/'run_manifest.json').exists() else None
        if old and old['dataset_fingerprint']!=fingerprint:raise ValueError('Input data changed; create a new run ID')
        # Task keys decide which stages change. Core training changes cannot reuse old weights.
        manifest=dict(schema_version='2.0',run_id=run_id,status='RUNNING',scope=args.suite,
            full_acceptance=False,dataset_fingerprint=fingerprint,config_hash=frozen,created_at=now(),environment=env,
            input_hashes=input_hashes,seeds=seeds,models=models,required_experiments=sorted(known),
            resolved_config='resolved_config.yaml',is_mock=False)
        write_json(root/'run_manifest.json',manifest)
        (root/'resolved_config.yaml').write_text(yaml.safe_dump(resolved,sort_keys=False),encoding='utf-8')
        write_json(root/'preflight.json',env)
        tasks=Tasks(root,args.resume,args.fail_fast)
        context_hashes=dict(graph=sha256_file(path_for(site['process']['graph'])) if site['process'].get('graph') else None,
                            catalog=sha256_file(path_for(site['decision']['action_catalog'])))
        def src(*names):return source_hash([PROJECT/n for n in names])
        data_key=dict(data=fingerprint,window=cfg['window'],cap=smoke_rows,replay_cap=args.smoke_replay_rows,code=src('src/data','src/common'))
        prepared=tasks.run('prepare',[],Path('shared')/fingerprint[:16],data_key,lambda out:prepare(site,cfg,out,smoke_rows,args.smoke_replay_rows))
        def evaluation(out):
            idx=read_json(prepared/'index.json');frames=load_frames(prepared,idx,'test')
            if frames:load_labels(site,pd.concat(frames,ignore_index=True),out)
            else:pd.DataFrame(columns=['sample_id','label']).to_parquet(out/'evaluation_labels.parquet',index=False)
        labels_dir=tasks.run('evaluation_labels',['prepare'],Path('evaluation'),dict(evaluation=site.get('evaluation'),code=src('src/bench')),evaluation)
        detection_outputs={};fit_outputs={};score_outputs={};business_outputs={};rca_outputs={}
        for model_id in models:
            model_seeds=[None] if model_id in DETERMINISTIC else seeds
            for seed in model_seeds:
                token=f'{model_id}_seed_{seed if seed is not None else "deterministic"}'
                fit_id='fit_'+token;score_id='score_'+token;detect_id='detect_'+token
                train_key=dict(model=model_id,seed=seed,training=cfg['training'],model_config=cfg['model'],code=src('src/models','src/baselines','src/common'))
                def fit(out,m=model_id,s=seed):
                    if m in ['D06','D07']:
                        if args.skip_train:
                            mapping=read_json(path_for(args.checkpoint_manifest));entry=mapping[f'{m}/seed_{s}']
                            source=path_for(entry['checkpoint'])
                            if sha256_file(source)!=entry['sha256']:raise ValueError('External checkpoint hash mismatch')
                            from src.models.load_checkpoint import load_checkpoint
                            _,payload=load_checkpoint(source,device_requested='cpu',expected={'feature_order':read_json(prepared/'scaler.json')['tags'],'seed':s,'data_fingerprint':fingerprint})
                            shutil.copyfile(source,out/'best.ckpt')
                            shutil.copyfile(path_for(entry['training_summary']),out/'training_summary.json')
                        else:
                            # Resume only an identical interrupted attempt; never load stale optimizer state.
                            marker=out/'training_contract.json'
                            if marker.exists() and read_json(marker)!=train_key:
                                for name in ['latest.ckpt','best.ckpt','training_history.csv','training_summary.csv','training_summary.json']:
                                    (out/name).unlink(missing_ok=True)
                            write_json(marker,train_key)
                            train_model(cfg,prepared,out,DETECTORS[m],s,device,run_id,resume=args.resume)
                    else:fit_estimator(m,prepared,out,s or 1)
                fitted=tasks.run(fit_id,['prepare'],Path('models')/model_id/f'seed_{seed if seed is not None else "deterministic"}',train_key,fit)
                fit_outputs[(model_id,seed)]=fitted
                if args.train_only:continue
                scored=tasks.run(score_id,[fit_id],Path('scores')/token,
                    dict(threshold=cfg['threshold'],alarm=cfg['alarm'],score=cfg['model'],scaler=cfg['scaler'],onset=cfg['rootcause']['onset_quantile'],code=src('src/detect')),
                    lambda out,m=model_id,f=fitted:calibrate_and_replay(m,f,prepared,out,cfg,site,device))
                score_outputs[(model_id,seed)]=scored
                def detect(out,scores=scored):
                    labels=pd.read_parquet(labels_dir/'evaluation_labels.parquet')
                    if not len(labels):
                        write_json(out/'metrics.json',{'status':'NOT_APPLICABLE','reason':'Evaluation labels were not supplied'});return
                    metrics,events,curves,data=evaluate_detection(pd.read_parquet(scores/'test_scores.parquet'),labels,site['input']['sample_interval_seconds'],cfg)
                    write_json(out/'metrics.json',metrics);events.to_csv(out/'event_detection.csv',index=False)
                    for name,frame in curves.items():frame.to_csv(out/(name+'.csv'),index=False)
                    timeline=data[['source_file','segment_id','timestamp','score','threshold','label','alarm_state']].copy()
                    timeline['purpose']='evaluation_plot';timeline.to_csv(out/'score_timeline.csv',index=False)
                    from src.bench.statistics import detection_intervals
                    units,ci=detection_intervals(data,site['input']['sample_interval_seconds'],cfg['window']['length'],cfg['eval']['bootstrap_samples'],cfg['eval']['bootstrap_seed'])
                    ci['estimate']=metrics['point_f1'];write_json(out/'ci.json',ci);units.to_csv(out/'bootstrap_units.csv',index=False)
                det=tasks.run(detect_id,[score_id,'evaluation_labels'],Path('experiments')/model_id/f'seed_{seed}',dict(eval=cfg['eval'],code=src('src/bench')),detect)
                detection_outputs[(model_id,seed)]=det
        surrogate=tasks.run('surrogate',['prepare'],Path('surrogate'),dict(config=cfg['surrogate'],code=src('src/sim')),
                            lambda out:fit_surrogates(prepared,out,site,cfg)) if not args.train_only else None
        # Keep completed fit/score task keys unchanged. This override applies only
        # after training, scoring and surrogate fitting have been dispatched.
        if args.replay_physical_batch_size is not None:
            cfg=copy.deepcopy(cfg)
            cfg['training']['physical_batch_size']=args.replay_physical_batch_size
            cfg['attribution']['replay_physical_batch_size']=args.replay_physical_batch_size
            write_json(root/'replay_execution.json',dict(physical_batch_size=args.replay_physical_batch_size,
                training_physical_batch_size=resolved['config']['training']['physical_batch_size']))
        if not args.train_only and 'D07' in models and site['data'].get('replay_files'):
            for seed in seeds:
                fitted=fit_outputs[('D07',seed)];scored=score_outputs[('D07',seed)]
                for variant in suite['required_ablations']:
                    if experiments and variant not in experiments and variant!='B00':continue
                    tid=f'{variant}_seed_{seed}'
                    business_outputs[(variant,seed)]=tasks.run(tid,[f'score_D07_seed_{seed}','surrogate'],Path('experiments')/variant/f'seed_{seed}',
                        dict(variant=variant,policy=cfg['policy'],rootcause=cfg['rootcause'],attribution=cfg['attribution'],site=site,context=context_hashes,code=src('src/runtime','src/explain','src/decision')),
                        lambda out,v=variant:business_replay(prepared,fitted,scored,surrogate,out,cfg,site,device,v,'P03',v=='B00'))
                for policy in suite['required_policies']:
                    if experiments and policy not in experiments:continue
                    business_outputs[(policy,seed)]=tasks.run(f'{policy}_seed_{seed}',[f'score_D07_seed_{seed}','surrogate'],Path('experiments')/policy/f'seed_{seed}',
                        dict(policy=policy,config=cfg['policy'],rootcause=cfg['rootcause'],attribution=cfg['attribution'],context=context_hashes,code=src('src/runtime','src/explain','src/decision')),
                        lambda out,p=policy:business_replay(prepared,fitted,scored,surrogate,out,cfg,site,device,'B00',p))
                baseline=business_outputs[('B00',seed)]
                for method in suite['required_rca']:
                    if experiments and method not in experiments:continue
                    def rca(out,m=method):
                        if not (detection_outputs[('D07',seed)]/'event_detection.csv').exists():
                            pd.DataFrame(columns=['gt_event_id','source_file','method','target_scope','snapshot_seconds','target_count','eligible','explained','detected','MRR','top1','top3','top5']).to_csv(out/'event_rca.csv',index=False)
                            write_json(out/'status.json',dict(status='NOT_APPLICABLE',reason='Evaluation labels unavailable'))
                            return
                        events=pd.read_csv(detection_outputs[('D07',seed)]/'event_detection.csv')
                        rankings=pd.read_csv(baseline/'rootcause_candidates.csv')
                        table=evaluate_rca(site,read_json(prepared/'scaler.json')['tags'],events,rankings,m)
                        table.to_csv(out/'event_rca.csv',index=False)
                    rca_outputs[(method,seed)]=tasks.run(f'{method}_seed_{seed}',[f'B00_seed_{seed}',f'detect_D07_seed_{seed}'],Path('experiments')/method/f'seed_{seed}',dict(method=method,code=src('src/bench')),rca)
                for control in suite['negative_controls']:
                    if experiments and control not in experiments:continue
                    def negative(out,n=control):
                        if n in ['N02','N04'] and not (detection_outputs[('D07',seed)]/'event_detection.csv').exists():
                            write_json(out/'status.json',dict(status='NOT_APPLICABLE',reason='Evaluation labels unavailable'));return
                        if n in ['N01','N03']:
                            graph=read_json(path_for(site['process']['graph'])) if site['process'].get('graph') else {'edges':[],'nodes':[]}
                            perm=np.random.default_rng(seed).permutation(len(site['features'])) if n=='N03' else None
                            business_replay(prepared,fitted,scored,surrogate,out,cfg,site,device,graph=rewire_graph(graph,seed) if n=='N01' else graph,onset_permutation=perm)
                            if (detection_outputs[('D07',seed)]/'event_detection.csv').exists():
                                negative_rca=evaluate_rca(site,read_json(prepared/'scaler.json')['tags'],pd.read_csv(detection_outputs[('D07',seed)]/'event_detection.csv'),pd.read_csv(out/'rootcause_candidates.csv'),'R05')
                                negative_rca.to_csv(out/'event_rca.csv',index=False)
                        elif n=='N02':
                            table=evaluate_rca(site,read_json(prepared/'scaler.json')['tags'],pd.read_csv(detection_outputs[('D07',seed)]/'event_detection.csv'),
                                pd.read_csv(baseline/'rootcause_candidates.csv'),'R05',permutation_seed=seed)
                            table.to_csv(out/'event_rca.csv',index=False)
                        elif n=='N04':
                            m=read_json(detection_outputs[('D07',seed)]/'metrics.json');write_json(out/'metrics.json',{k:v for k,v in m.items() if k in ['point_f1','PA_F1_appendix_segment_weighted']})
                        else:
                            from src.bench.leakage_audit import audit
                            write_json(out/'audit.json',audit(prepared,fitted,scored,surrogate,cfg,site,device))
                    tasks.run(f'{control}_seed_{seed}',[f'B00_seed_{seed}',f'detect_D07_seed_{seed}'],Path('experiments')/control/f'seed_{seed}',dict(control=control,seed=seed,code=src('src/bench','src/runtime')),negative)
        from scripts.export_results import export_results
        export_results(root,manifest,tasks,detection_outputs,business_outputs,rca_outputs,fit_outputs,cfg,site)
        if args.export_bundle:
            if ('D07',1) not in fit_outputs or args.train_only:
                raise ValueError('Bundle export requires D07 seed=1 and completed calibration/runtime stages')
            def bundle(out):
                from src.runtime.bundle import export_bundle,load_bundle
                dest=PROJECT/'bundles'/f'{run_id}_primary'
                if dest.exists():
                    if not args.resume:raise FileExistsError(str(dest))
                    existing=read_json(dest/'bundle_manifest.json')
                    if existing['checkpoint_sha256']!=sha256_file(fit_outputs[('D07',1)]/'best.ckpt') or existing['code_hash']!=source_hash():
                        dest=PROJECT/'bundles'/f'{run_id}_primary_{digest([sha256_file(fit_outputs[("D07",1)]/"best.ckpt"),source_hash()])[:8]}'
                    if not dest.exists():export_bundle(run_id,prepared,fit_outputs[('D07',1)],score_outputs[('D07',1)],surrogate,site,cfg,dest,args.suite)
                else:export_bundle(run_id,prepared,fit_outputs[('D07',1)],score_outputs[('D07',1)],surrogate,site,cfg,dest,args.suite)
                load_bundle(dest,str(device),site)
                write_json(out/'bundle_reference.json',dict(bundle_path=str(dest.relative_to(PROJECT)),bundle_manifest_sha256=sha256_file(dest/'bundle_manifest.json')))
                print(f'Application bundle: {dest}',flush=True)
            tasks.run('bundle',['score_D07_seed_1','surrogate'],Path('deployment'),dict(config=cfg,site=site,context=context_hashes,code=src('src/runtime')),bundle)
        failed=any(r['status'] in ['FAILED','SKIPPED_DEPENDENCY'] for r in tasks.rows)
        complete_matrix=(args.suite=='full' and models==suite['models'] and seeds==[1,7,21] and not experiments and not args.train_only)
        manifest.update(status='INCOMPLETE' if failed else 'SUCCEEDED',full_acceptance=bool(complete_matrix and not failed),
                        finished_at=now(),task_counts=pd.Series([r['status'] for r in tasks.rows]).value_counts().to_dict(),
                        result_files=inventory(root/'tables'),bundle_path=read_json(root/'deployment/bundle_reference.json')['bundle_path'] if args.export_bundle and tasks.status.get('bundle') in ['SUCCEEDED','REUSED'] else None)
        write_json(root/'run_manifest.json',manifest)
        from scripts.audit_run import audit
        audit_result=audit(root,require_full=bool(complete_matrix))
        if not audit_result['passed']:
            failed=True
            manifest.update(status='INCOMPLETE',full_acceptance=False,integrity_errors=audit_result['errors'])
            write_json(root/'run_manifest.json',manifest)
        from scripts.make_figures import render
        render(root,path_for('config/plots.yaml'))
        (root/'reports').mkdir(exist_ok=True)
        (root/'reports/results.md').write_text(f'# Run {run_id}\n\nStatus: {manifest["status"]}. Scope: {args.suite}. Full matrix acceptance: {manifest["full_acceptance"]}.\n\nNo surrogate intervention claims are supported. See tables and task_registry.csv for explicit failures and unavailable metrics.\n\nDevice: {device}. Seeds: {seeds}. Smoke row cap per source/role: {smoke_rows}.\n',encoding='utf-8')
        print(f'Run: {root}\nStatus: {manifest["status"]}; full acceptance: {manifest["full_acceptance"]}',flush=True)
        return 1 if failed else 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        print(f'Pipeline error: {exc}',file=sys.stderr);raise SystemExit(2)
