#!/usr/bin/env python3
"""Recompute CPU-only analysis tables from frozen scores and causal rankings."""
from pathlib import Path
import argparse
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))


def export(run):
    import pandas as pd
    from src.common.config import load_yaml
    from src.common.io_utils import read_json,write_json,sha256_file
    from src.common.provenance import source_hash,inventory,run_lock,now
    from src.bench.analysis_exports import export_sensitivity,export_coverage
    with run_lock(run/'.lock'):
        manifest=read_json(run/'run_manifest.json')
        resolved=load_yaml(run/'resolved_config.yaml')
        registry=pd.read_csv(run/'task_registry.csv')
        successful=registry.loc[registry.status.isin(['SUCCEEDED','REUSED'])]
        if 'surrogate' not in set(successful.task_id):
            raise ValueError('A successful surrogate qualification stage is required')
        business={}
        for row in successful.to_dict('records'):
            if row['task_id'].startswith(('B','P')):
                experiment,_,seed=row['task_id'].partition('_seed_')
                business[(experiment,int(seed))]=run/row['artifacts']
        # These are read-only inputs, including weights, score arrays, frozen
        # thresholds and estimator state. Save both inventories for an audit.
        before={name:inventory(run/name) for name in ['models','scores','surrogate']}
        export_sensitivity(run,resolved['config'],resolved['site'])
        export_coverage(run,resolved['config'],resolved['site'],business)
        after={name:inventory(run/name) for name in before}
        if before!=after:raise RuntimeError('Frozen input artifacts changed during analysis')
        outputs={name:sha256_file(run/'tables'/name) for name in ['threshold_sensitivity.csv','coverage_risk.csv']}
        report=dict(schema_version='2.0',run_id=manifest['run_id'],status='PASSED',device='cpu',created_at=now(),
            code_hash=source_hash(),purpose='Analysis only; no training or threshold selection',
            sensitivity_rows=len(pd.read_csv(run/'tables/threshold_sensitivity.csv')),
            coverage_rows=len(pd.read_csv(run/'tables/coverage_risk.csv')),
            model_and_threshold_artifacts_unchanged=True,protected_artifact_hashes=before,output_hashes=outputs)
        report_path=run/'reports/analysis_exports.json'
        write_json(report_path,report)
        manifest['result_files']=inventory(run/'tables')
        manifest['supplemental_analysis']=dict(report=report_path.relative_to(run).as_posix(),code_hash=report['code_hash'],device='cpu')
        write_json(run/'run_manifest.json',manifest)
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    from src.data.prepare_factory import path_for
    result=export(path_for(args.run_dir))
    print(f"Analysis passed: {result['sensitivity_rows']} threshold rows, {result['coverage_rows']} coverage rows; frozen inputs unchanged")


if __name__=='__main__':main()
