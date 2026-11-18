#!/usr/bin/env python3
"""Hash-inventoried recursive packaging of an explicit run or deployment bundle."""
import argparse,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common.provenance import inventory
from src.data.prepare_factory import path_for

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root,out=path_for(a.source).resolve(),path_for(a.output).resolve()
    if out.is_relative_to(root):raise ValueError('Write archive outside its source tree')
    if not ((root/'run_manifest.json').exists() or (root/'bundle_manifest.json').exists()):raise ValueError('Explicit run or bundle root required')
    files={k:v for k,v in inventory(root).items() if not k.startswith(('shared/','.matplotlib/')) and k!='.lock'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(root/name,name)
        archive.writestr('package_inventory.json',json.dumps(files,indent=2))
    print(out)
if __name__=='__main__':main()
