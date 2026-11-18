#!/usr/bin/env python3
"""Check a real bundle's CPU/device score agreement at a declared tolerance."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import numpy as np
    import torch
    from src.common.io_utils import write_json
    from src.runtime.bundle import load_bundle
    from src.data.prepare_factory import load_site,path_for,read_telemetry
    from src.data.fit_scaler import apply_scaler
    from src.data.window import build_index
    from src.explain.causal import model_attribution
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--site',default='config/sites/hai2305.yaml')
    p.add_argument('--input',default='data/raw/hai-23.05/hai-test1.csv');p.add_argument('--device',required=True)
    p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=32);a=p.parse_args()
    torch.set_num_threads(2);site=load_site(a.site);manifest,cpu=load_bundle(path_for(a.bundle),'cpu',site);_,other=load_bundle(path_for(a.bundle),a.device,site)
    data=read_telemetry(path_for(a.input),site).iloc[:cpu.detector.window+a.rows]
    X=apply_scaler(data[cpu.tags].to_numpy(np.float32),cpu.detector.scaler);index=build_index([X],cpu.detector.window,pad_mode='skip')
    scores=[[],[]];evidences=[[],[]]
    for i in range(min(a.rows,len(index))):
        w=index.gather(np.array([i]))[0]
        for k,engine in enumerate([cpu,other]):
            score,_,evidence=engine.detector.window_score(w);scores[k].append(score);evidences[k].append(evidence)
    atol,rtol=1e-4,1e-3
    sc=np.array(scores);ev=np.array(evidences)
    grad=[model_attribution(e.detector,w,e.reference['scaled_median'])[0] for e in [cpu,other]]
    passed=bool(np.allclose(sc[0],sc[1],atol=atol,rtol=rtol) and np.allclose(ev[0],ev[1],atol=atol,rtol=rtol) and np.allclose(grad[0],grad[1],atol=atol,rtol=rtol))
    threshold=cpu.threshold['value'];near=np.abs(sc[0]-threshold)<=atol+rtol*abs(threshold)
    report=dict(bundle_id=manifest['bundle_id'],reference_device='cpu',tested_device=a.device,rows=sc.shape[1],atol=atol,rtol=rtol,
        max_absolute_score_difference=float(np.max(abs(sc[0]-sc[1]))),max_absolute_evidence_difference=float(np.max(abs(ev[0]-ev[1]))),
        max_absolute_gradient_difference=float(np.max(abs(grad[0]-grad[1]))),threshold_band_samples=int(near.sum()),
        prediction_disagreements=int(((sc[0]>threshold)!=(sc[1]>threshold)).sum()),passed=passed)
    write_json(path_for(a.output),report);print(report);return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
