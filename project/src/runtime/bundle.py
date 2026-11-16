"""Content-verified portable deployment packages containing frozen inference inputs."""
from __future__ import annotations
import copy
import shutil
from pathlib import Path
import numpy as np
import torch
import yaml
from src.common.config import load_yaml
from src.common.io_utils import read_json,write_json,sha256_file
from src.common.provenance import digest,inventory,relative_file,source_hash
from src.data.prepare_factory import path_for,site_contract
from src.models.load_checkpoint import atomic_save,load_checkpoint
from src.detect.evaluate_detector import Detector
from src.runtime.engine import RuntimeEngine


def export_bundle(run_id,prepared,fitted,scored,surrogate,site,cfg,output,validation_scope='smoke'):
    output=Path(output)
    if output.exists():raise FileExistsError(f'Bundle already exists: {output}; choose a new bundle ID')
    temp=output.with_name(output.name+'.building')
    if temp.exists():shutil.rmtree(temp)
    temp.mkdir(parents=True)
    files={'model/best.ckpt':fitted/'best.ckpt','preprocessing/scaler.json':prepared/'scaler.json',
           'preprocessing/normal_reference.json':prepared/'normal_reference.json',
           'preprocessing/residual_scale.json':scored/'residual_scale.json','preprocessing/channel_thresholds.json':scored/'channel_thresholds.json',
           'detection/threshold.json':scored/'threshold.json','surrogate/card.json':surrogate/'card.json','surrogate/models.json':surrogate/'models.json',
           'decision/action_catalog.yaml':path_for(site['decision']['action_catalog'])}
    if site['process'].get('graph'):files['process/process_graph.json']=path_for(site['process']['graph'])
    for name,source in files.items():
        target=temp/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name=='model/best.ckpt':
            payload=torch.load(source,map_location='cpu',weights_only=False)
            # Preserve hash binding of the original best checkpoint; best has no optimizer.
            shutil.copyfile(source,target)
        else:shutil.copyfile(source,target)
    if not site['process'].get('graph'):write_json(temp/'process/process_graph.json',dict(nodes=[],edges=[]))
    # No original data paths travel inside the site profile or runtime configuration.
    profile=copy.deepcopy(site_contract(site))
    profile['process']['graph']='process/process_graph.json'
    profile['decision']['action_catalog']='decision/action_catalog.yaml'
    profile['dataset_id']=site.get('dataset_id',site['site_id'])
    runtime_cfg={k:copy.deepcopy(cfg[k]) for k in ['schema_version','protocol_id','window','training','model','alarm','attribution','rootcause','policy']}
    for name,value in [('runtime/site_profile.yaml',profile),('runtime/config.yaml',runtime_cfg),('detection/alarm_policy.yaml',cfg['alarm'])]:
        dest=temp/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    scaler=read_json(prepared/'scaler.json');summary=read_json(fitted/'training_summary.json')
    write_json(temp/'runtime/feature_order.json',dict(tags=scaler['tags'],continuous_mask=scaler['continuous_mask']))
    (temp/'README_EN.md').write_text(f'# {output.name}\n\nFrozen {validation_scope} bundle from run {run_id}.\nModel: TranAD_C_G_E, seed {summary["seed"]}, best epoch {summary["best_epoch"]}.\nInspection advice only. No equipment commands are sent. Do not interpret smoke validation as benchmark performance.\n',encoding='utf-8')
    manifest=dict(schema_version='2.0',bundle_id=output.name,source_run=run_id,model_id='D07',seed=summary['seed'],
                  best_epoch=summary['best_epoch'],checkpoint_sha256=sha256_file(fitted/'best.ckpt'),
                  feature_order=scaler['tags'],continuous_mask=scaler['continuous_mask'],window=cfg['window']['length'],
                  sample_interval_seconds=site['input']['sample_interval_seconds'],code_hash=source_hash(),
                  original_site_contract_hash=digest(site_contract(site)),capabilities=['detect','rank','inspect_advice','case_workflow'],
                  disabled_reason=read_json(surrogate/'card.json')['reason'],validation_scope=validation_scope,
                  files=inventory(temp),dependency_spec='Python 3.10/3.11; torch 2.2.2; numpy 1.26.4')
    write_json(temp/'bundle_manifest.json',manifest)
    temp.rename(output)
    return manifest


def load_bundle(path,device='auto',site=None):
    path=Path(path).resolve();manifest=read_json(path/'bundle_manifest.json')
    if manifest.get('schema_version')!='2.0':raise ValueError('Unsupported bundle schema')
    required={'model/best.ckpt','preprocessing/scaler.json','preprocessing/normal_reference.json','preprocessing/residual_scale.json',
              'preprocessing/channel_thresholds.json','detection/threshold.json','runtime/site_profile.yaml','runtime/config.yaml',
              'runtime/feature_order.json','process/process_graph.json','decision/action_catalog.yaml','surrogate/card.json'}
    if not required<=set(manifest['files']):raise ValueError('Bundle is missing required artifacts')
    for name,expected in manifest['files'].items():
        file=relative_file(path,name)
        if not file.is_file() or sha256_file(file)!=expected:raise ValueError(f'Bundle integrity failure: {name}')
    profile=yaml.safe_load((path/'runtime/site_profile.yaml').read_text(encoding='utf-8'))
    cfg=yaml.safe_load((path/'runtime/config.yaml').read_text(encoding='utf-8'))
    scaler=read_json(path/'preprocessing/scaler.json');threshold=read_json(path/'detection/threshold.json')
    scale=read_json(path/'preprocessing/residual_scale.json');channel=read_json(path/'preprocessing/channel_thresholds.json')
    if site is not None and digest(site_contract(site))!=manifest['original_site_contract_hash']:
        raise ValueError('Site feature/unit/transform/runtime contract differs from the deployed bundle')
    order=read_json(path/'runtime/feature_order.json')
    if order['tags']!=manifest['feature_order'] or order['tags']!=scaler['tags'] or order['continuous_mask']!=scaler['continuous_mask']:
        raise ValueError('Feature order/mask mismatch')
    D=len(order['tags'])
    if any(len(scaler[k])!=D for k in ['min','span','continuous_mask']) or len(scale['scale'])!=D or len(channel['values'])!=D:
        raise ValueError('Preprocessing dimensions disagree')
    if threshold['weight_sha256']!=manifest['checkpoint_sha256'] or sha256_file(path/'model/best.ckpt')!=manifest['checkpoint_sha256']:
        raise ValueError('Threshold/checkpoint binding mismatch')
    if threshold['scaler_hash']!=digest(scaler) or scale['scaler_hash']!=digest(scaler) or threshold['residual_scale_hash']!=digest(scale['scale']):
        raise ValueError('Scaler/residual/threshold binding mismatch')
    if channel['threshold_id']!=threshold['threshold_id'] or cfg['window']['length']!=manifest['window']:
        raise ValueError('Window/channel threshold binding mismatch')
    graph=read_json(path/'process/process_graph.json')
    nodes={n['id'] if isinstance(n,dict) else n for n in graph.get('nodes',[])}
    if not nodes<=set(order['tags']):raise ValueError('Graph contains unregistered features')
    catalog=yaml.safe_load((path/'decision/action_catalog.yaml').read_text(encoding='utf-8'))
    for action in catalog['actions']:
        if not set(action.get('required_tags',[]))<=set(order['tags']):raise ValueError('Action contains unregistered required tags')
    from src.common.device import get_best_device
    detector=Detector('D07',path/'model',scaler,cfg,get_best_device(device))
    detector.scale=np.array(scale['scale'])
    if detector.model.n_window!=manifest['window']:raise ValueError('Model window mismatch')
    engine=RuntimeEngine(detector,profile,cfg,threshold,channel['values'],graph,catalog,
                         read_json(path/'preprocessing/normal_reference.json'),read_json(path/'surrogate/card.json'))
    return manifest,engine
