#!/usr/bin/env python3
"""Export the preselected D07 seed=1 artifacts of an explicit run."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import yaml
    from src.common.io_utils import read_json
    from src.data.prepare_factory import path_for
    from src.runtime.bundle import export_bundle
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    run=path_for(args.run_dir);manifest=read_json(run/'run_manifest.json')
    resolved=yaml.safe_load((run/'resolved_config.yaml').read_text(encoding='utf-8'))
    prepared=run/'shared'/manifest['dataset_fingerprint'][:16]
    export_bundle(manifest['run_id'],prepared,run/'models/D07/seed_1',run/'scores/D07_seed_1',run/'surrogate',
                  resolved['site'],resolved['config'],path_for(args.output),manifest['scope'])
    print(path_for(args.output))
if __name__=='__main__':main()
