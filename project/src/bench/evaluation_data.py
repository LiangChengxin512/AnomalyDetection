"""Evaluation-only label loading, including the hash-locked HAI minute-label adapter."""
import numpy as np
import pandas as pd
from src.common.io_utils import read_json, write_json, sha256_file
from src.data.prepare_factory import path_for
from src.common.config import PROJECT_ROOT
from src.common.events import binary_to_ranges
from src.data.prepare_hai import _verify_against_summary


def load_labels(site, frames, output):
    label_rows=[]
    mapping=site.get('evaluation',{}).get('label_files',{})
    for source, group in frames.groupby('source_file',sort=False):
        if source not in mapping:
            continue
        path=path_for(mapping[source])
        labels=pd.read_csv(path)
        label_col=site.get('evaluation',{}).get('label_column','label')
        if label_col not in labels:
            label_col=labels.columns[-1]
        values=pd.to_numeric(labels[label_col],errors='raise').to_numpy()
        if not np.isin(values,[0,1]).all():
            raise ValueError(f'{source}: labels must be binary')
        label_times=pd.to_datetime(labels.iloc[:,0],errors='raise')
        positions=group.row_in_source.to_numpy(int)
        if positions.max()>=len(labels):
            raise ValueError(f'{source}: label rows do not cover telemetry')
        times=pd.to_datetime(group.timestamp).reset_index(drop=True)
        selected=label_times.iloc[positions].reset_index(drop=True)
        if not (times==selected).all():
            if site.get('dataset_id')!='HAI-23.05' or source!='hai-test2.csv' or path.name!='label-test2.csv':
                raise ValueError(f'{source}: mismatched label timestamps; row alignment is prohibited')
            manifest=read_json(PROJECT_ROOT/'data/raw_manifest.json')
            expected={__import__('pathlib').Path(r['file']).name:r['sha256'] for r in manifest['files']}
            raw=next(path_for(p) for p in site['data']['replay_files'] if path_for(p).name==source)
            if sha256_file(path)!=expected[path.name] or sha256_file(raw)!=expected[source]:
                raise ValueError('HAI row-alignment exception requires the registered original hashes')
            # HAI label-test2 repeats minute starts, a bounded [0,59] second offset.
            delta=(times-selected).dt.total_seconds()
            if not delta.between(0,59).all():
                raise ValueError('HAI minute-resolution timestamp exception failed')
            raw_df=pd.read_csv(raw,usecols=['timestamp'])
            raw_df['timestamp']=pd.to_datetime(raw_df.timestamp,errors='raise')
            if len(raw_df)!=len(labels) or not _verify_against_summary(path_for(site['evaluation']['summary_files'][source]),raw_df,values):
                raise ValueError('HAI row count/summary alignment failed')
        out=group[['sample_id','source_file','segment_id','row_in_source','timestamp']].copy()
        out['label']=values[positions].astype(int)
        label_rows.append(out)
    result=pd.concat(label_rows,ignore_index=True) if label_rows else pd.DataFrame(columns=['sample_id','label'])
    result.to_parquet(output/'evaluation_labels.parquet',index=False)
    return result


def target_mapping(site, source, start, end, tags):
    if not site.get('evaluation',{}).get('attack_targets'):
        return [],[],None
    from src.common.config import load_yaml
    mappings=load_yaml(site['evaluation']['attack_targets'])
    matches=[e for e in mappings['events'] if e['file']==source and pd.Timestamp(e['start'])<=end and pd.Timestamp(e['end'])>=start]
    strict=sorted(set(t for e in matches for t in e.get('target_points',[]) if t in tags))
    weak=sorted(set(t for e in matches for t in e.get('observable_tags',[]) if t in tags))
    return strict,weak,mappings.get('source')
