#!/usr/bin/env python3
"""Validate explicit run artifacts; full acceptance requires uncapped complete data."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def audit(root,require_full=False):
    import numpy as np,pandas as pd,yaml
    from src.common.io_utils import read_json,sha256_file,write_json
    from src.common.provenance import inventory
    root=Path(root);manifest=read_json(root/'run_manifest.json')
    resolved=yaml.safe_load((root/'resolved_config.yaml').read_text(encoding='utf-8'))
    registry=pd.read_csv(root/'task_registry.csv');errors=[];notes=[]
    bad=registry.loc[~registry.status.isin(['SUCCEEDED','REUSED','NOT_APPLICABLE'])]
    if len(bad):errors.append(f'{len(bad)} failed/incomplete tasks')
    for task in registry.loc[registry.status.isin(['SUCCEEDED','REUSED'])].to_dict('records'):
        receipt=read_json(root/'receipts'/f'{task["task_id"]}.json')
        if inventory(root/task['artifacts'])!=receipt['files']:errors.append('Output receipt mismatch: '+task['task_id'])
    index=read_json(root/'shared'/manifest['dataset_fingerprint'][:16]/'index.json')
    input_ids=[]
    for record in index['splits'].get('test',[]):
        input_ids.extend(pd.read_parquet(root/'shared'/manifest['dataset_fingerprint'][:16]/record['parquet'],columns=['sample_id']).sample_id)
    for task in registry.loc[registry.task_id.str.startswith('score_') & registry.status.isin(['SUCCEEDED','REUSED'])].to_dict('records'):
        directory=root/task['artifacts'];threshold=read_json(directory/'threshold.json')
        if threshold['calibration_split']!='val_threshold':errors.append('Invalid calibration source: '+task['task_id'])
        scorefile=directory/'test_scores.parquet'
        if not scorefile.exists():continue
        scores=pd.read_parquet(scorefile)
        if scores.sample_id.duplicated().any() or list(scores.sample_id)!=input_ids:errors.append('Input/score keys mismatch: '+task['task_id'])
        if scores.threshold_id.nunique()!=1 or scores.threshold_id.iloc[0]!=threshold['threshold_id']:errors.append('Threshold binding mismatch')
        if not np.isfinite(scores.score).all():errors.append('Nonfinite scores')
    for filename in ['metrics_detection','metrics_rca','metrics_policy','metrics_ablation','metrics_negative_controls','metrics_ci','training_history','training_summary','event_detection','event_rca','rootcause_candidates','policy_decisions','policy_constraints','surrogate_validation']:
        if not (root/'tables'/f'{filename}.csv').exists() and not resolved['train_only']:errors.append('Missing export: '+filename)
    if require_full:
        if manifest['scope']!='full' or resolved['smoke_rows'] or resolved.get('smoke_replay_rows') or resolved['train_only'] or resolved['experiments']:
            errors.append('Not an uncapped full experiment')
        if resolved['seeds']!=[1,7,21] or set(resolved['models'])!={f'D{i:02}' for i in range(1,8)}:errors.append('Required model/seed matrix missing')
        required={f'{prefix}{i:02}_seed_{seed}' for prefix,ids in [('R',range(1,6)),('P',range(4)),('B',range(8)),('N',range(1,6))] for i in ids for seed in [1,7,21]}
        if not required<=set(registry.task_id):errors.append('Required downstream experiment tasks missing')
        labels=pd.read_parquet(root/'evaluation/evaluation_labels.parquet')
        if not len(labels) or labels.label.nunique()<2:errors.append('Full evaluation requires normal and attack labels')
    notes.append('VUS-PR unavailable; no approximation substituted')
    notes.append('Physical action response unqualified; effect rows unavailable')
    report=dict(schema_version='2.0',passed=not errors,requested_full=require_full,errors=errors,notes=notes)
    write_json(root/'integrity_audit.json',report)
    return report


def main():
    import argparse,json
    from src.data.prepare_factory import path_for
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--require-full',action='store_true');a=p.parse_args()
    result=audit(path_for(a.run_dir),a.require_full);print(json.dumps(result,indent=2));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
