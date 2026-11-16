"""Post-evaluation sensitivity tables; never choose or overwrite deployment thresholds."""
from __future__ import annotations
import copy
import json
import numpy as np
import pandas as pd
from src.common.io_utils import read_json
from src.common.config import load_yaml
from src.data.prepare_factory import load_frames
from src.detect.evaluate_detector import alarm_table
from src.decision.safe_policy import AdvicePolicy


def export_sensitivity(root,cfg,site):
    registry=pd.read_csv(root/'task_registry.csv')
    interval=site['input']['sample_interval_seconds']
    labels=pd.read_parquet(root/'evaluation/evaluation_labels.parquet')
    rows=[]
    manifest=read_json(root/'run_manifest.json')
    quantiles=cfg.get('analysis',{}).get('threshold_quantiles',[.99,.995,.999,.9995,1.])
    for task in registry.loc[registry.task_id.str.startswith('score_')&registry.status.isin(['SUCCEEDED','REUSED'])].to_dict('records'):
        directory=root/task['artifacts'];val=pd.read_parquet(directory/'val_threshold_scores.parquet')
        frozen=read_json(directory/'threshold.json')
        normal=~val.warmup & val.quality_flags.eq('') & np.isfinite(val.score)
        if not normal.any():continue
        hours=normal.sum()*interval/3600
        test=pd.read_parquet(directory/'test_scores.parquet') if (directory/'test_scores.parquet').exists() else pd.DataFrame()
        if len(test) and len(labels):
            joined=test.merge(labels[['sample_id','label']],on='sample_id',validate='one_to_one')
            if len(joined)!=len(test):raise ValueError('Sensitivity evaluation labels must cover every score')
            test=joined.loc[~joined.warmup & joined.quality_flags.eq('') & np.isfinite(joined.score)]
        else:test=pd.DataFrame()
        # Stored scores already contain the locked-threshold alarm annotations.
        # Candidate thresholds require a fresh alarm state machine, not a merge
        # with those annotations (which would create misleading _x/_y columns).
        alarm_input=val[['sample_id','source_file','segment_id','timestamp','score','warmup','quality_flags']]
        for quantile in quantiles:
            if not 0<=quantile<=1:raise ValueError('Sensitivity quantiles must be in [0, 1]')
            threshold=float(np.quantile(val.loc[normal,'score'],quantile))
            alarm=alarm_table(alarm_input,threshold,cfg['alarm'],interval)
            seed_token=task['task_id'].rsplit('_',1)[-1]
            result=dict(schema_version='2.0',run_id=manifest['run_id'],scope=manifest['scope'],
                task_id=task['task_id'],model_id=frozen['model_id'],seed=None if seed_token=='deterministic' else int(seed_token),quantile=quantile,
                candidate_threshold=threshold,threshold_source='val_threshold',locked_threshold=frozen['value'],
                locked_threshold_id=frozen['threshold_id'],purpose='analysis_only_no_threshold_selection',
                normal_hours=hours,normal_event_far_per_hour=float(alarm.alarm_opened.sum()/hours),
                point_exceedance_rate=float((val.loc[normal,'score']>threshold).mean()),
                test_precision=None,test_recall=None,test_f1=None,status='SUCCEEDED')
            if len(test):
                y=test.label.to_numpy(bool);p=test.score.to_numpy()>threshold
                tp,fp,fn=int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum())
                result.update(test_precision=tp/(tp+fp) if tp+fp else None,
                              test_recall=tp/(tp+fn) if tp+fn else None,
                              test_f1=2*tp/(2*tp+fp+fn) if y.any() and 2*tp+fp+fn else None)
            rows.append(result)
    table=pd.DataFrame(rows,columns=None if rows else ['schema_version','task_id','quantile','candidate_threshold','normal_event_far_per_hour','status'])
    table.to_csv(root/'tables/threshold_sensitivity.csv',index=False)
    table.to_csv(root/'plot_data/threshold_sensitivity.csv',index=False)


def export_coverage(root,cfg,site,business):
    manifest=read_json(root/'run_manifest.json');prepared=root/'shared'/manifest['dataset_fingerprint'][:16]
    index=read_json(prepared/'index.json');frames=load_frames(prepared,index,'test')
    rows=[]
    if not frames:
        pd.DataFrame(columns=['schema_version','seed','margin_threshold','coverage','risk','status']).to_csv(root/'tables/coverage_risk.csv',index=False);return
    telemetry=pd.concat(frames,ignore_index=True).set_index('sample_id',verify_integrity=True)
    labels=pd.read_parquet(root/'evaluation/evaluation_labels.parquet')
    label_map=dict(zip(labels.sample_id,labels.label)) if len(labels) else {}
    catalog=load_yaml(site['decision']['action_catalog']);reference=read_json(prepared/'normal_reference.json')
    specific_actions={a['id'] for a in catalog['actions'] if '*' not in a.get('applies_to',[])}
    card=read_json(root/'surrogate/card.json');tags=reference['tags']
    margins=cfg.get('analysis',{}).get('evidence_margins',[0.,.01,.05,.1])
    for (experiment,seed),directory in business.items():
        if experiment!='B00' or not (directory/'rootcause.jsonl').exists():continue
        policies=[];counts=[]
        for margin in margins:
            if margin<0:raise ValueError('Evidence margins must be nonnegative')
            configuration=copy.deepcopy(cfg);configuration['rootcause']['abstain_margin']=margin
            policies.append(AdvicePolicy(site,catalog,reference,card,configuration))
            counts.append(dict(requests=0,offered=0,specific_offered=0,abstained=0,suppressed=0,normal_adjustments=0))
        with (directory/'rootcause.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                record=json.loads(line)
                if record['method']!='R05':continue
                frame=telemetry.loc[record['sample_id']];raw={t:float(frame[t]) for t in tags}
                for policy,count in zip(policies,counts):
                    advice,_=policy.decide(record['alarm_id'],record['sample_id'],record['as_of_time'],pd.Timestamp(record['as_of_time']).timestamp(),raw,record['candidates'],record['margin'])
                    count['requests']+=1;count['offered']+=advice['action_id'] is not None
                    count['specific_offered']+=advice['action_id'] in specific_actions
                    count['abstained']+=advice['specific_advice_abstained'] or advice['mode']=='abstain'
                    count['suppressed']+=advice['mode']=='suppressed'
                    count['normal_adjustments']+=advice['physical_adjustment'] and label_map.get(record['sample_id'])==0
        for margin,count in zip(margins,counts):
            rows.append(dict(schema_version='2.0',run_id=manifest['run_id'],scope=manifest['scope'],seed=seed,margin_threshold=margin,**count,
                coverage=count['offered']/count['requests'] if count['requests'] else None,
                specific_coverage=count['specific_offered']/count['requests'] if count['requests'] else None,
                risk=None,risk_status='NOT_APPLICABLE',reason='Action response is unqualified; no surrogate risk is invented',
                status='SUCCEEDED' if count['requests'] else 'NOT_APPLICABLE',
                purpose='Reexecute policy gates on identical frozen causal ranking snapshots'))
    frame=pd.DataFrame(rows,columns=None if rows else ['schema_version','seed','margin_threshold','coverage','risk','status','reason'])
    frame.to_csv(root/'tables/coverage_risk.csv',index=False);frame.to_csv(root/'plot_data/coverage_risk.csv',index=False)
