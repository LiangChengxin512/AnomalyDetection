#!/usr/bin/env python3
"""Export an explicit score partition or channel evidence selection to CSV."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import pandas as pd,numpy as np
    from src.data.prepare_factory import path_for
    from src.common.io_utils import read_json
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--model',default='D07');p.add_argument('--seed',default='1')
    p.add_argument('--source');p.add_argument('--start');p.add_argument('--end');p.add_argument('--tag');p.add_argument('--output',required=True);a=p.parse_args()
    run=path_for(a.run_dir);source=run/'scores'/f'{a.model}_seed_{a.seed}'
    table=pd.read_parquet(source/'test_scores.parquet');mask=pd.Series(True,index=table.index)
    if a.source:mask &=table.source_file.eq(a.source)
    if a.start:mask &=pd.to_datetime(table.timestamp)>=pd.Timestamp(a.start)
    if a.end:mask &=pd.to_datetime(table.timestamp)<=pd.Timestamp(a.end)
    out=table.loc[mask].copy()
    if a.tag:
        manifest=read_json(run/'run_manifest.json');tags=read_json(run/'shared'/manifest['dataset_fingerprint'][:16]/'scaler.json')['tags']
        if a.model in ['D04','D05']:raise ValueError('Scalar estimators do not provide channel evidence')
        evidence=np.load(source/'test_evidence.npy',mmap_mode='r');out['tag']=a.tag;out['normalized_evidence']=evidence[mask.to_numpy(),tags.index(a.tag)]
    out.to_csv(path_for(a.output),index=False)
if __name__=='__main__':main()
