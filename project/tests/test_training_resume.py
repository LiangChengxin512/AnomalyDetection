"""Exercise actual optimizer/checkpoint resume with a deterministic validation monitor."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.common.config import load_config
from src.common.io_utils import write_json,read_json
from src.models.train import train_model

class TrainingResume(unittest.TestCase):
    def fixture(self,root):
        prepared=root/'data';prepared.mkdir()
        scaler=dict(tags=['a','b'],min=[0,0],span=[1,1],max=[1,1],continuous_mask=[True,False])
        write_json(prepared/'scaler.json',scaler)
        index=dict(dataset_fingerprint='fixture',splits={})
        for split in ['train','val_model']:
            path=split+'.parquet';pd.DataFrame({'a':np.linspace(0,1,16),'b':np.arange(16)%2}).to_parquet(prepared/path,index=False)
            index['splits'][split]=[{'parquet':path}]
        write_json(prepared/'index.json',index)
        cfg=load_config();cfg['window']['length']=4;cfg['model']['n_window']=4
        cfg['training'].update(max_epochs=10,batch_size=8,physical_batch_size=3)
        cfg['training']['early_stopping'].update(patience=3,min_epochs=3)
        return prepared,cfg
    def test_actual_early_stop_and_completed_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data,cfg=self.fixture(root);out=root/'model'
            with patch('src.models.train.validation_loss',side_effect=[.5,.4,.39,.391,.392,.393]):
                summary=train_model(cfg,data,out,'TranAD_C_G_E',1,torch.device('cpu'),'fixture')
            self.assertEqual(summary['completed_epochs'],6);self.assertEqual(summary['best_epoch'],3);self.assertEqual(summary['stop_reason'],'early_stopping')
            with patch('src.models.train.train_epoch',side_effect=AssertionError('completed early stop must not restart')):
                resumed=train_model(cfg,data,out,'TranAD_C_G_E',1,torch.device('cpu'),'fixture',resume=True)
            self.assertEqual(resumed['completed_epochs'],6);self.assertEqual(resumed['resume_count'],1)
            self.assertEqual(len(pd.read_csv(out/'training_history.csv')),6)
    def test_interruption_resume_preserves_total_budget_and_rng(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data,cfg=self.fixture(root);cfg['training']['max_epochs']=4
            with patch('src.models.train.validation_loss',side_effect=[.5,.4,KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):train_model(cfg,data,root/'interrupted','TranAD',7,torch.device('cpu'),'fixture')
            with patch('src.models.train.validation_loss',side_effect=[.3,.2]):
                summary=train_model(cfg,data,root/'interrupted','TranAD',7,torch.device('cpu'),'fixture',resume=True)
            with patch('src.models.train.validation_loss',side_effect=[.5,.4,.3,.2]):
                train_model(cfg,data,root/'continuous','TranAD',7,torch.device('cpu'),'fixture')
            self.assertEqual(summary['completed_epochs'],4);self.assertEqual(summary['stop_reason'],'max_epochs')
            a=torch.load(root/'interrupted/best.ckpt',map_location='cpu',weights_only=False)['model_state_dict']
            b=torch.load(root/'continuous/best.ckpt',map_location='cpu',weights_only=False)['model_state_dict']
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    def test_nonfinite_monitor_is_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);data,cfg=self.fixture(root)
            with patch('src.models.train.validation_loss',return_value=float('nan')):
                with self.assertRaises(FloatingPointError):train_model(cfg,data,root/'model','TranAD',1,torch.device('cpu'),'fixture')
            self.assertFalse((root/'model/training_summary.json').exists())
if __name__=='__main__':unittest.main()
