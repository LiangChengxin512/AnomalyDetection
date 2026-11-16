#!/usr/bin/env python3
"""Diagnose the exact interpreter, dependencies, data and requested device."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def check(site_path,requested=None,configured='auto',output=None):
    import importlib.util
    required=['numpy','pandas','pyarrow','sklearn','yaml','torch','matplotlib']
    missing=[p for p in required if importlib.util.find_spec(p) is None]
    if missing:
        raise RuntimeError(f'{sys.executable}: missing {missing}. Install requirements/base.txt in this interpreter.')
    from src.common.device import get_best_device
    from src.common.provenance import environment
    from src.common.io_utils import write_json
    from src.data.prepare_factory import load_site,path_for
    import torch
    site=load_site(site_path)
    for group in ['train_files','normal_validation_files','replay_files']:
        for p in site['data'].get(group,[]):
            if not path_for(p).is_file():raise FileNotFoundError(f'Site {group}: {path_for(p)}')
    device=get_best_device(requested,configured=configured)
    # Real attention, convolution, top-k and input-gradient check on requested device.
    from src.models.tranad_adapter import build_model,predict_batch
    model=build_model('TranAD_C_G_E',4,8,device,torch.float32,seed=1)
    model.set_continuous_mask([True,False,True,False])
    x=torch.ones(2,8,4,device=device,requires_grad=True)
    _,y=predict_batch(model,x)
    y.topk(2).values.sum().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():raise RuntimeError('Device gradient smoke failed')
    env=environment();env.update(selected_device=str(device),device_tensor_check='PASSED',site_id=site['site_id'])
    if output:write_json(output,env)
    return env


def main():
    import argparse,json
    ap=argparse.ArgumentParser()
    ap.add_argument('--site',default='config/sites/hai2305.yaml');ap.add_argument('--device',choices=['auto','cpu','mps','cuda'])
    ap.add_argument('--output');args=ap.parse_args()
    try:
        print(json.dumps(check(args.site,args.device,output=args.output),indent=2));return 0
    except Exception as exc:
        print(f'Preflight failed ({sys.executable}): {exc}',file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
