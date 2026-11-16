"""Reproducible event/normal-block and paired-event intervals."""
import numpy as np
import pandas as pd
from src.common.events import binary_to_ranges


def paired_interval(reference,variant,keys,value,B=2000,seed=2026):
    pair=reference[keys+[value]].merge(variant[keys+[value]],on=keys,suffixes=('_reference','_variant'),validate='one_to_one')
    delta=(pair[value+'_variant']-pair[value+'_reference']).dropna().to_numpy()
    if not len(delta):return dict(estimate=None,ci_low=None,ci_high=None,n_pairs=0,status='NOT_APPLICABLE',reason='No eligible paired events')
    rng=np.random.default_rng(seed)
    draws=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(B)])
    return dict(estimate=float(delta.mean()),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),n_pairs=len(delta),status='SUCCEEDED',reason='')


def detection_intervals(data,interval,window,B=2000,seed=2026):
    units=[];block=max(10*window,int(1800/interval))
    for (source,segment),group in data.groupby(['source_file','segment_id'],sort=False):
        group=group.reset_index(drop=True)
        labels=group.label.to_numpy(bool)
        for kind,flags in [('attack',labels),('normal',~labels)]:
            for start,end in binary_to_ranges(flags.astype(int)):
                for lo in range(start,end+1,block if kind=='normal' else end-start+1):
                    hi=min(lo+block,end+1) if kind=='normal' else end+1
                    part=group.iloc[lo:hi];y=part.label.to_numpy(bool);p=part.raw_pred.to_numpy(bool)
                    units.append(dict(source_file=source,segment_id=segment,unit_kind=kind,
                        tp=int((p&y).sum()),fp=int((p&~y).sum()),fn=int((~p&y).sum()),
                        normal_hours=float((~y).sum()*interval/3600),rows=len(part)))
    df=pd.DataFrame(units);rng=np.random.default_rng(seed);draws=[]
    if len(df):
        strata=[g for _,g in df.groupby(['source_file','unit_kind'],sort=False)]
        for _ in range(B):
            sampled=pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for g in strata])
            tp,fp,fn=sampled[['tp','fp','fn']].sum()
            draws.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return df,dict(metric='point_f1',estimate=None,ci_low=float(np.quantile(draws,.025)) if draws else None,
        ci_high=float(np.quantile(draws,.975)) if draws else None,confidence_level=.95,
        resampling_unit='source-stratified attack clusters and nonoverlapping normal blocks; no remerging',B=B,bootstrap_seed=seed)
