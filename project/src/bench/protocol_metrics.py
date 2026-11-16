"""Point and segment-wise, one-to-one event metrics; no point adjustment in primary metrics."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve,roc_auc_score,precision_recall_fscore_support,auc
from src.common.events import binary_to_ranges
from src.bench.metrics import etapr_metrics,pa_f1


def match_events(truth,prediction):
    if not truth or not prediction:
        return []
    weight=np.zeros((len(truth),len(prediction)))
    base=min(len(truth),len(prediction))+1
    for i,(s,e) in enumerate(truth):
        for j,(ps,pe) in enumerate(prediction):
            intersection=max(0,min(e,pe)-max(s,ps)+1)
            if intersection:
                weight[i,j]=base+intersection/(max(e,pe)-min(s,ps)+1)+1e-10/(1+i+j)
    a,b=linear_sum_assignment(-weight)
    return [(int(i),int(j)) for i,j in zip(a,b) if weight[i,j]>0]


def evaluate_detection(table,labels,interval,cfg):
    if table.sample_id.duplicated().any() or labels.sample_id.duplicated().any():
        raise ValueError('Duplicate sample keys in scores or evaluation labels')
    data=table.merge(labels[['sample_id','label']],on='sample_id',validate='one_to_one')
    if len(data)!=len(table):
        raise ValueError('Evaluation labels must cover every score; partial joins are prohibited')
    valid=data.loc[~data.warmup & data.quality_flags.eq('')].copy()
    y=valid.label.to_numpy(int); pred=valid.raw_pred.to_numpy(bool); score=valid.score.to_numpy(float)
    if not len(y):
        raise ValueError('No common effective scoring samples')
    precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
    metrics=dict(point_precision=precision,point_recall=recall,point_f1=f1,point_AP=None,point_pr_auc=None,point_roc_auc=None)
    curves={}
    if len(np.unique(y))==2:
        p,r,t=precision_recall_curve(y,score)
        fpr,tpr,rt=roc_curve(y,score)
        metrics.update(point_AP=average_precision_score(y,score),point_pr_auc=auc(r,p),point_roc_auc=roc_auc_score(y,score))
        curves['pr_curve']=pd.DataFrame(dict(precision=p,recall=r,score_threshold=np.r_[t,np.nan]))
        curves['roc_curve']=pd.DataFrame(dict(fpr=fpr,tpr=tpr,score_threshold=rt))
    events=[]; matched=truth_count=pred_count=false_count=coverage=0
    delays=[]; pa=[]; etapr=[]
    for (source,segment), group in valid.groupby(['source_file','segment_id'],sort=False):
        group=group.reset_index(drop=True)
        gt=binary_to_ranges(group.label.to_numpy(int))
        alarm_ids=group.alarm_id.fillna('')
        prs=[]; ids=[]
        # Identity, not a concatenated binary array, defines alarm boundaries.
        for aid, rows in group.loc[alarm_ids.ne('')].groupby('alarm_id',sort=False):
            prs.append((int(rows.index.min()),int(rows.index.max())));ids.append(aid)
        pairs=match_events(gt,prs)
        mapping=dict(pairs)
        matched+=len(pairs);truth_count+=len(gt);pred_count+=len(prs)
        for j,(ps,pe) in enumerate(prs):
            overlaps=[i for i,(s,e) in enumerate(gt) if ps<=e and pe>=s]
            false_count+=not overlaps
            events.append(dict(record_type='prediction',source_file=source,segment_id=segment,alarm_id=ids[j],
                gt_event_id=None,matched=j in mapping.values(),duplicate=bool(overlaps) and j not in mapping.values(),
                start=str(group.timestamp.iloc[ps]),end=str(group.timestamp.iloc[pe]),missed=None,delay=None))
        for i,(s,e) in enumerate(gt):
            positive=np.flatnonzero(group.raw_pred.iloc[s:e+1].to_numpy())
            overlaps=[j for j,(ps,pe) in enumerate(prs) if ps<=e and pe>=s]
            hit=bool(overlaps)
            coverage+=hit
            delay=float(positive[0]*interval) if hit and len(positive) else None
            if delay is not None:delays.append(delay)
            eid=f'{source}:{segment}:attack:{int(group.row_in_source.iloc[s])}'
            events.append(dict(record_type='truth',source_file=source,segment_id=segment,gt_event_id=eid,
                alarm_id=ids[mapping[i]] if i in mapping else None,matched=i in mapping,duplicate=False,
                start=str(group.timestamp.iloc[s]),end=str(group.timestamp.iloc[e]),row_start=int(group.row_in_source.iloc[s]),
                row_end=int(group.row_in_source.iloc[e]),missed=not hit,delay=delay,
                first_trigger=str(group.timestamp.iloc[prs[mapping[i]][0]]) if i in mapping else None,
                preexisting=any(prs[j][0]<s for j in overlaps),attack_event_covered=hit))
        pa.append((len(group),pa_f1(group.raw_pred.to_numpy(int),group.label.to_numpy(int))))
        etapr.append((len(group),etapr_metrics(group.raw_pred.to_numpy(int),group.label.to_numpy(int),
                      cfg['eval']['etapr_theta_p'],cfg['eval']['etapr_theta_r'],cfg['eval']['etapr_delta'])))
    normal_hours=float((y==0).sum()*interval/3600)
    ep=matched/pred_count if pred_count else 0.;er=matched/truth_count if truth_count else None
    metrics.update(event_precision=ep,event_recall=er,event_f1=(2*ep*er/(ep+er) if ep+er else 0.) if er is not None else None,
        attack_event_coverage=coverage/truth_count if truth_count else None,false_alert_events_per_hour=false_count/normal_hours if normal_hours else None,
        mean_detection_delay_s=float(np.mean(delays)) if delays else None,missed_attacks=truth_count-coverage,
        scoring_hours=len(y)*interval/3600,normal_hours=normal_hours,warmup_excluded=int(data.warmup.sum()),
        n_events=truth_count,n_pred_events=pred_count,point_exceedance_rate=float(pred.mean()),
        PA_F1_appendix_segment_weighted=sum(n*v for n,v in pa)/len(y),VUS_PR=None)
    for deadline in [10,30,60]:
        metrics[f'detected_within_{deadline}s']=sum(d<=deadline for d in delays)/truth_count if truth_count else None
    for name in ['eTaPR_f1','eTaP','eTaR']:
        metrics[name+'_segment_weighted']=sum(n*m[name] for n,m in etapr)/len(y) if all(m.get(name) is not None for n,m in etapr) else None
    return metrics,pd.DataFrame(events, columns=None if events else ['record_type','source_file','segment_id','gt_event_id','alarm_id','matched','duplicate','start','end','missed','delay','preexisting']),curves,valid
