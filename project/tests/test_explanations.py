import unittest
import numpy as np
import torch
from src.models.tranad_adapter import build_model
from src.explain.attribution import occlusion_scores,input_gradient_scores
from src.bench.continuation import evaluate_rca
import pandas as pd
from unittest.mock import patch

class ExplanationTests(unittest.TestCase):
    def test_occlusion_batching_preserves_channel_intervention(self):
        torch.set_num_threads(1)
        model=build_model('TranAD_C_G_E',4,8,torch.device('cpu'),torch.float32,seed=3);model.eval()
        model.set_continuous_mask([True,False,True,True]);w=np.random.default_rng(1).random((2,8,4),dtype=np.float32)
        args=(model,w,np.ones(4),'topk',3,np.array([.2,1,.3,.4]))
        serial=occlusion_scores(*args,batch_limit=2)
        batched=occlusion_scores(*args,batch_limit=8)
        np.testing.assert_allclose(serial,batched,atol=1e-6,rtol=1e-5)
        gradient=input_gradient_scores(model,w,np.ones(4),'topk',3)
        self.assertEqual(gradient.shape,(2,4));self.assertTrue(np.isfinite(gradient).all())
    def test_preexisting_alarm_snapshot_cannot_use_later_ranking(self):
        event=pd.DataFrame([dict(record_type='truth',gt_event_id='event',source_file='f',start='2020-01-01 00:00:10',end='2020-01-01 00:00:30',first_trigger='2020-01-01 00:00:02',alarm_id='alarm',missed=False,preexisting=True)])
        ranks=pd.DataFrame([dict(method='R05',alarm_id='alarm',as_of_time='2020-01-01 00:00:02',tag='a',rank=1,abstain=False),dict(method='R05',alarm_id='alarm',as_of_time='2020-01-01 00:00:02',tag='b',rank=2,abstain=False),dict(method='R05',alarm_id='alarm',as_of_time='2020-01-01 00:00:12',tag='b',rank=1,abstain=False)])
        with patch('src.bench.continuation.target_mapping',return_value=(['b'],['b'],'fixture')):
            result=evaluate_rca({},['a','b'],event,ranks,'R05')
        initial=result.loc[result.snapshot_seconds.eq(0)];later=result.loc[result.snapshot_seconds.eq(10)]
        self.assertTrue(initial.top1.eq(0).all());self.assertTrue(later.top1.eq(1).all());self.assertTrue(initial.preexisting.all())
if __name__=='__main__':unittest.main()
