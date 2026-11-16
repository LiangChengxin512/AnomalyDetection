"""High-risk temporal, model, orchestration, metric and workflow contracts."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.common.device import get_best_device
from src.common.io_utils import write_json
from src.common.provenance import run_lock
from src.common.tasks import Tasks
from src.data.segments import identify,transition_pairs
from src.data.split_protocol import validation_ranges
from src.data.window import build_index
from src.models.early_stopping import EarlyStopping
from src.models.load_checkpoint import save_checkpoint,load_checkpoint
from src.models.tranad_adapter import build_model,predict_batch,model_state_hash
from src.detect.alarm_state import AlarmMachine
from src.bench.protocol_metrics import match_events,evaluate_detection
from src.decision.safe_policy import check_action
from src.decision.action_transform import transform
from src.runtime.case_store import CaseStore
from src.sim.rollout import rollout

class Contracts(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_early_stopping(self):
        s=EarlyStopping(patience=3,min_epochs=3)
        for i,v in enumerate([.5,.4,.39,.391,.392,.393],1):s.update(v,i)
        self.assertTrue(s.stopped);self.assertEqual(s.best_epoch,3);self.assertEqual(s.bad_epochs,3)
        tiny=EarlyStopping(min_delta_abs=.01,min_delta_rel=0);tiny.update(.5,1)
        self.assertEqual(tiny.update(.499,2),(True,False))
        with self.assertRaises(FloatingPointError):s.update(float('nan'),7)
    def test_early_stop_continues_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            tasks=Tasks(Path(td));called=[]
            def train(out):
                s=EarlyStopping(patience=3,min_epochs=3)
                for i,v in enumerate([.5,.4,.39,.391,.392,.393],1):s.update(v,i)
                write_json(out/'summary.json',{'stop_reason':'early_stopping','best_epoch':s.best_epoch})
            tasks.run('train',[],Path('model'),{},train)
            tasks.run('calibrate',['train'],Path('calibration'),{},lambda out:(called.append(1),write_json(out/'threshold.json',{'value':1})))
            self.assertEqual(called,[1]);self.assertEqual(tasks.status['calibrate'],'SUCCEEDED')
    def test_device_precedence(self):
        with patch.dict(os.environ,{'PROJECT_DEVICE':'invalid'}):
            self.assertEqual(get_best_device('cpu').type,'cpu')
            with self.assertRaises(ValueError):get_best_device(None)
    def test_model_shape_mask_hash(self):
        with tempfile.TemporaryDirectory() as td:
            model=build_model('TranAD_C_G_E',4,8,torch.device('cpu'),torch.float32,seed=1)
            model.set_continuous_mask([True,False,True,False]);model.eval();x=torch.rand(3,8,4)
            expected=predict_batch(model,x)[1].detach();self.assertEqual(expected.shape,(3,4));self.assertEqual(predict_batch(model,x[:1])[1].shape,(1,4))
            file=Path(td)/'best.ckpt';save_checkpoint(file,model,None,None,1,{},{},dict(feature_order=list('abcd'),continuous_mask=[True,False,True,False]))
            loaded,_=load_checkpoint(file,device_requested='cpu');loaded.eval();torch.testing.assert_close(expected,predict_batch(loaded,x)[1])
            self.assertEqual(loaded.feature_gate.continuous_mask.tolist(),[True,False,True,False]);old=model_state_hash(loaded)
            with torch.no_grad():next(loaded.parameters()).add_(1)
            self.assertNotEqual(old,model_state_hash(loaded))
            with self.assertRaises(ValueError):predict_batch(model,torch.rand(1,9,4))
    def test_legacy_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            m=build_model('TranAD',2,4,torch.device('cpu'),torch.float32);p=Path(td)/'old.ckpt';save_checkpoint(p,m,None,None,0,{}, {})
            with self.assertRaisesRegex(ValueError,'Legacy'):load_checkpoint(p,device_requested='cpu')
    def test_time_and_keys(self):
        f=pd.DataFrame({'timestamp':['2020-01-01 00:00:00','2020-01-01 00:00:01','2020-01-01 00:00:03','2020-01-01 00:00:02']})
        a=identify(f,'data','a.csv');b=identify(f,'data','b.csv')
        self.assertEqual(a.segment_id.tolist(),[0,0,1,2]);self.assertFalse(set(a.sample_id)&set(b.sample_id))
        f.loc[2,'timestamp']='bad'
        with self.assertRaisesRegex(ValueError,'row 4'):identify(f,'data','a.csv')
    def test_windows_transitions(self):
        frames=[np.array([[1],[2],[3]],np.float32),np.array([[100],[101]],np.float32)];idx=build_index(frames,3)
        self.assertEqual(idx.gather(np.array([3])).ravel().tolist(),[100,100,100]);x,y=transition_pairs(frames)
        self.assertTrue(np.all(y-x==1));self.assertNotIn([3,100],np.c_[x,y].tolist())
    def test_validation_embargo(self):
        r=validation_ranges(2000,128);self.assertEqual(r['val_uncertainty'][0]-r['val_model'][1],127)
        with self.assertRaises(ValueError):validation_ranges(100,128)
    def test_alarm_hold_and_ablation(self):
        a=AlarmMachine(10,on_count=2,recovery_count=2,merge_gap_s=1)
        u=[a.update(s,str(i),str(i),i) for i,s in enumerate([0,11,11,9,9,9])]
        self.assertEqual(sum(r['opened'] for r in u),1);self.assertTrue(u[-1]['closed'])
        self.assertEqual(a.update(0,'x','x',9,quality=True)['state'],'DATA_QUALITY_HOLD')
        b=AlarmMachine(10,consolidate=False);self.assertTrue(b.update(11,'1','1',1)['opened']);self.assertTrue(b.update(11,'2','2',2)['opened'])
    def test_one_to_one(self):
        self.assertEqual(len(match_events([(1,2),(5,6),(9,10)],[(0,12)])),1)
        self.assertEqual(len(match_events([(1,4),(5,8)],[(1,8),(1,2)])),2)
    def test_missing_delays_and_single_class(self):
        f=pd.DataFrame(dict(sample_id=list('abcdefgh'),source_file='f',segment_id=0,row_in_source=range(8),timestamp=pd.date_range('2020',periods=8,freq='s'),warmup=False,quality_flags='',score=0.,raw_pred=False,alarm_id=None))
        labels=pd.DataFrame({'sample_id':f.sample_id,'label':[0,0,1,1,0,0,0,0]});cfg={'eval':{'etapr_theta_p':.7,'etapr_theta_r':.1,'etapr_delta':0}}
        m,e,_,_=evaluate_detection(f,labels,1,cfg);self.assertIsNone(m['mean_detection_delay_s']);self.assertEqual(m['missed_attacks'],1);self.assertTrue(e.delay.isna().all())
        labels['label']=0;m,_,_,_=evaluate_detection(f,labels,1,cfg);self.assertIsNone(m['point_AP']);self.assertIsNone(m['point_roc_auc'])
    def test_missing_constraints(self):
        a=dict(id='a',kind='inspect',required_tags=[],review_status='approved',evidence_source='test',requires={'missing':{'min':0,'max':1}})
        self.assertFalse(all(c['passed'] for c in check_action(a,{},dict(features=[],decision={}))))
    def test_no_fake_inspection_effect(self):
        self.assertEqual(transform({'kind':'inspect'},{'u':5},10),{'u':5})
        a=dict(kind='adjust',control_tag='u',absolute_bounds={'min':0,'max':10},unit='pct',max_delta=2,rate_limit=1)
        self.assertEqual(transform(a,{'u':5},1)['u'],6)
        with self.assertRaises(ValueError):transform(a,{'u':5},3)
        with self.assertRaisesRegex(ValueError,'not_qualified'):rollout({}, {},[],[],a,1,5)
    def test_persistent_case_workflow(self):
        with tempfile.TemporaryDirectory() as td:
            s=CaseStore(td);s.bind({'bundle':'b'});u={'detection':{'sample_id':'s1'},'alarm':{'alarm_id':'c1','closed':False},'rankings':[],'advice':None}
            self.assertTrue(s.apply_update(u));self.assertFalse(s.apply_update(u))
            with self.assertRaises(ValueError):s.command('c1','close-case','op','early')
            for cmd in ['acknowledge','reject-advice','escalate','add-note','close-case']:s.command('c1',cmd,'operator','review reason')
            s.close();s=CaseStore(td);self.assertTrue(s.contains('s1'))
            with self.assertRaises(ValueError):s.bind({'bundle':'changed'})
            s.close();self.assertTrue((Path(td)/'case_actions.csv').exists())
    def test_verified_task_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);calls=[]
            def fn(out):calls.append(1);write_json(out/'a.json',{'n':1})
            Tasks(root).run('a',[],Path('a'),{'x':1},fn);Tasks(root,resume=True).run('a',[],Path('a'),{'x':1},fn);self.assertEqual(len(calls),1)
            write_json(root/'a/a.json',{'n':2});Tasks(root,resume=True).run('a',[],Path('a'),{'x':1},fn);self.assertEqual(len(calls),2)
    def test_process_context_does_not_change_telemetry_fingerprint(self):
        from src.data.prepare_factory import input_fingerprint
        site=dict(site_id='fixture',input={'sample_interval_seconds':1},features=[{'name':'a'}],data={},process={'graph':'g1'},decision={'catalog':'c1'})
        first,_=input_fingerprint(site)
        site['process']['graph']='g2';site['decision']['catalog']='c2'
        self.assertEqual(first,input_fingerprint(site)[0])
    def test_invalidated_stage_removes_stale_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            def fn(out):write_json(out/'new.json',{'ok':True})
            Tasks(root).run('score',[],Path('score'),{'version':1},fn)
            write_json(root/'score/stale.json',{'stale':True})
            Tasks(root,resume=True).run('score',[],Path('score'),{'version':2},fn)
            self.assertFalse((root/'score/stale.json').exists())
    def test_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as td:
            with run_lock(Path(td)/'lock'):
                with self.assertRaises(RuntimeError):
                    with run_lock(Path(td)/'lock'):pass
if __name__=='__main__':unittest.main()
