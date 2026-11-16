#!/usr/bin/env python3
"""Read-only compatibility audit; legacy weights are never certified by file existence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import torch
    from src.common.config import PROJECT_ROOT
    from src.common.io_utils import read_json,write_json,sha256_file
    from src.models.load_checkpoint import load_checkpoint
    from src.models.tranad_adapter import predict_batch
    checkpoint=PROJECT_ROOT/'checkpoints/TranAD_C_G_E_HAI-23.05/best.ckpt'
    if not checkpoint.exists():return 0
    model,payload=load_checkpoint(checkpoint,device_requested='cpu',allow_legacy=True);model.eval()
    scaler=read_json(PROJECT_ROOT/'data/processed/scaler.json')
    x=torch.full((2,model.n_window,model.n_feats),.5)
    with torch.no_grad():a=predict_batch(model,x)[1]
    model.set_continuous_mask(scaler['continuous_mask'])
    with torch.no_grad():b=predict_batch(model,x)[1]
    report=dict(path=str(checkpoint.relative_to(PROJECT_ROOT)),sha256=sha256_file(checkpoint),model=payload['model_name'],
        n_feats=model.n_feats,n_window=model.n_window,saved_epoch=payload['epoch'],saved_metrics=payload['metrics'],
        batch_shape=list(b.shape),finite_forward=bool(torch.isfinite(b).all()),
        mask_sensitive_max_difference=float((a-b).abs().max()),
        certification='NOT_REUSABLE_UNDER_V2_PROTOCOL',reason='Missing authenticated feature order/type mask, validation role and complete training/resume provenance',
        audit_only_mask_source='historical processed scaler; not evidence of original checkpoint binding')
    write_json(PROJECT_ROOT/'reports/continuation_validation/legacy_checkpoint.json',report);print(report)
if __name__=='__main__':main()
