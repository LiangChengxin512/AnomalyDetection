"""Explicit inputs for causal business replay and evaluation; no directory guessing."""
from __future__ import annotations
import copy
import json
import numpy as np
import pandas as pd
from src.common.io_utils import read_json,write_json,write_jsonl
from src.common.config import load_yaml
from src.data.prepare_factory import load_frames,path_for
from src.detect.evaluate_detector import Detector
from src.runtime.engine import RuntimeEngine
from src.bench.evaluation_data import target_mapping
from src.common.events import binary_to_ranges

RANK_COLUMNS=['sample_id','alarm_id','as_of_time','method','tag','rank','score','residual','attribution','onset','graph','stability','margin','abstain','checkpoint_sha256']
ADVICE_COLUMNS=['advice_id','alarm_id','sample_id','as_of_time','policy_id','variant_id','action_id','mode','reason','legal','cost','physical_adjustment','duplicate']


def make_engine(prepared,fitted,scored,surrogate,cfg,site,device,variant='B00',policy_id='P03',all_methods=False,cache=None,graph=None,onset_permutation=None):
    scaler=read_json(prepared/'scaler.json')
    detector=Detector('D07',fitted,scaler,cfg,device)
    detector.scale=np.array(read_json(scored/'residual_scale.json')['scale'])
    graph=graph if graph is not None else (read_json(path_for(site['process']['graph'])) if site['process'].get('graph') else {'nodes':[],'edges':[]})
    catalog=load_yaml(site['decision']['action_catalog'])
    card=read_json(surrogate/'card.json')
    return RuntimeEngine(detector,site,cfg,read_json(scored/'threshold.json'),read_json(scored/'channel_thresholds.json')['values'],
        graph,catalog,read_json(prepared/'normal_reference.json'),card,variant,policy_id,all_methods,cache,onset_permutation)


def business_replay(prepared,fitted,scored,surrogate,output,cfg,site,device,variant='B00',policy_id='P03',all_methods=False,graph=None,onset_permutation=None):
    output.mkdir(parents=True,exist_ok=True)
    engine=make_engine(prepared,fitted,scored,surrogate,cfg,site,device,variant,policy_id,all_methods,graph=graph,onset_permutation=onset_permutation)
    idx=read_json(prepared/'index.json')
    frames=load_frames(prepared,idx,'test')
    if not frames:raise ValueError('No replay files')
    scores=pd.read_parquet(scored/'test_scores.parquet')
    residual=np.load(scored/'test_residual.npy',mmap_mode='r');evidence=np.load(scored/'test_evidence.npy',mmap_mode='r')
    telemetry=pd.concat(frames,ignore_index=True)
    if not telemetry.sample_id.equals(scores.sample_id):raise ValueError('Precomputed scores must match source/segment/sample order exactly')
    rank_rows,rank_audit,advice_rows,constraint_rows,alarm_rows,efficiency=[],[],[],[],[],[]
    for i,values in enumerate(telemetry.itertuples(index=False,name=None)):
        frame=dict(zip(telemetry.columns,values))
        # Cache only the current request, keeping memory independent of replay duration.
        engine.cache.clear()
        update=engine.process_frame(frame,(float(scores.score.iloc[i]),residual[i],evidence[i]))
        alarm_rows.append(dict(update['detection'],alarm_state=update['alarm']['state'],alarm_id=update['alarm']['alarm_id'],
                               alarm_opened=update['alarm']['opened'],alarm_closed=update['alarm']['closed']))
        for ranking in update['rankings']:
            rank_audit.append(ranking)
            base={k:v for k,v in ranking.items() if k!='candidates'}
            rank_rows.extend(dict(base,**candidate) for candidate in ranking['candidates'])
        if update['advice']:
            advice_rows.append(update['advice'])
            constraint_rows.extend(dict(sample_id=frame['sample_id'],alarm_id=update['alarm']['alarm_id'],**r) for r in update['constraints'])
        if update['efficiency'].get('explanation_ms') is not None:efficiency.append(update['efficiency'])
    pd.DataFrame(alarm_rows).to_parquet(output/'scores.parquet',index=False)
    pd.DataFrame(rank_rows,columns=None if rank_rows else RANK_COLUMNS).to_csv(output/'rootcause_candidates.csv',index=False)
    write_jsonl(output/'rootcause.jsonl',rank_audit)
    advice=pd.DataFrame(advice_rows,columns=None if advice_rows else ADVICE_COLUMNS)
    advice.to_csv(output/'policy_decisions.csv',index=False)
    pd.DataFrame(constraint_rows,columns=None if constraint_rows else ['sample_id','alarm_id','action_id','constraint_name','passed']).to_csv(output/'policy_constraints.csv',index=False)
    pd.DataFrame(efficiency,columns=['inference_ms','explanation_ms','end_to_end_ms','inference_source']).to_csv(output/'latency_samples.csv',index=False)
    write_json(output/'resolved_business.json',dict(variant=variant,policy=policy_id,all_methods=all_methods,
        frozen_threshold=engine.threshold,source_checkpoint_sha256=engine.detector.weight_hash,
        removed_capability=load_yaml('config/ablations/business.yaml').get(variant,variant),
        surrogate_card=engine.card))
    offered=advice.action_id.notna() if len(advice) else np.zeros(0,dtype=bool)
    metrics=dict(advice_coverage=float(offered.mean()) if len(advice) else None,n_requests=len(advice),n_offered=int(sum(offered)),
        n_abstained=int(advice['mode'].eq('abstain').sum()),n_suppressed=int(advice['mode'].eq('suppressed').sum()),
        legal_offered_rate=float(advice.loc[offered,'legal'].mean()) if sum(offered) else None,
        action_cost=float(advice.loc[offered,'cost'].sum()) if sum(offered) else 0.,
        alarms=sum(r['alarm_opened'] for r in alarm_rows),advice_duplicates=int(advice.duplicate.sum()) if len(advice) else 0,
        surrogate_only_risk=None)
    metrics['n_specific_advice_abstained'] = int(advice.specific_advice_abstained.sum()) if len(advice) else 0
    metrics['duplicates_prevented'] = int(advice.suppressed_duplicate.sum()) if len(advice) else 0
    valid_hours=sum(not r['warmup'] for r in alarm_rows)*site['input']['sample_interval_seconds']/3600
    metrics['alarm_requests_per_hour']=metrics['alarms']/valid_hours if valid_hours else None
    write_json(output/'workflow_metrics.json',metrics)
    return metrics


def evaluate_rca(site,tags,event_table,ranks,method,permutation_seed=None):
    rows=[]
    truth=event_table.loc[event_table.record_type.eq('truth')]
    eligible_map={}
    for event in truth.to_dict('records'):
        strict,weak,source=target_mapping(site,event['source_file'],pd.Timestamp(event['start']),pd.Timestamp(event['end']),tags)
        eligible_map[event['gt_event_id']]=(strict,weak,source)
    if permutation_seed is not None:
        # Permute within observable target pool; target count and eligibility are unchanged.
        pool=sorted({tag for strict,_,_ in eligible_map.values() for tag in strict})
        rng=np.random.default_rng(permutation_seed)
        perm=dict(zip(pool,rng.permutation(pool)))
        eligible_map={k:([perm[t] for t in strict],weak,source) for k,(strict,weak,source) in eligible_map.items()}
    for event in truth.to_dict('records'):
        strict,weak,source=eligible_map[event['gt_event_id']]
        start,end=pd.Timestamp(event['start']),pd.Timestamp(event['end'])
        available=ranks.loc[ranks.method.eq(method)].copy()
        if len(available):
            times=pd.to_datetime(available.as_of_time)
            available=available.loc[times<=end]
            # Alarm IDs are based on full source keys, never on local row alone.
            if event.get('alarm_id') and pd.notna(event['alarm_id']):available=available.loc[available.alarm_id.eq(event['alarm_id'])]
            else:available=available.iloc[:0]
        first=pd.to_datetime(event.get('first_trigger')) if pd.notna(event.get('first_trigger')) else pd.NaT
        if pd.isna(first) and len(available):
            first=pd.to_datetime(available.as_of_time).min()
        for scope,targets in [('strict_observable_target',strict),('associated_loop_tags',weak)]:
            for offset in [0,5,10]:
                deadline=first+pd.Timedelta(seconds=offset) if pd.notna(first) else pd.NaT
                snapshot=available.loc[pd.to_datetime(available.as_of_time)<=deadline] if pd.notna(deadline) else available.iloc[:0]
                if len(snapshot):
                    latest=pd.to_datetime(snapshot.as_of_time).max()
                    snapshot=snapshot.loc[pd.to_datetime(snapshot.as_of_time).eq(latest)].sort_values('rank')
                ranked=snapshot.tag.tolist()
                positions=[ranked.index(t)+1 for t in targets if t in ranked]
                row=dict(gt_event_id=event['gt_event_id'],source_file=event['source_file'],method=method,target_scope=scope,
                    snapshot_seconds=offset,target_count=len(targets),eligible=bool(targets),explained=bool(ranked),
                    detected=not bool(event['missed']),decision_time=str(deadline) if pd.notna(deadline) else None,
                    preexisting=bool(event.get('preexisting',False)),
                    mapping_source=source,abstain=bool(snapshot.abstain.iloc[0]) if len(snapshot) else True,
                    MRR=1/min(positions) if positions else 0.,delay=(first-start).total_seconds() if pd.notna(first) else None)
                for k in [1,3,5]:
                    row[f'top{k}']=float(any(p<=k for p in positions))
                    row[f'target_recall{k}']=sum(p<=k for p in positions)/len(targets) if targets else None
                rows.append(row)
    return pd.DataFrame(rows,columns=None if rows else ['gt_event_id','source_file','method','target_scope','snapshot_seconds','target_count','eligible','explained','detected','MRR','top1','top3','top5'])


def rewire_graph(graph,seed):
    graph=copy.deepcopy(graph);edges=graph.get('edges',[]);rng=np.random.default_rng(seed)
    pairs={(e['source'],e['target']) for e in edges}
    for _ in range(len(edges)*10):
        if len(edges)<2:break
        i,j=rng.choice(len(edges),2,replace=False)
        a,b=edges[i],edges[j];p=(a['source'],b['target']);q=(b['source'],a['target'])
        if p[0]==p[1] or q[0]==q[1] or p in pairs or q in pairs:continue
        pairs.remove((a['source'],a['target']));pairs.remove((b['source'],b['target']))
        a['target'],b['target']=b['target'],a['target'];pairs.update([p,q])
    graph['graph_version']=f'degree_preserving_negative_{seed}'
    return graph
