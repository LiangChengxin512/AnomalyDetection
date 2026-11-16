"""Integration checks against the explicitly named real smoke bundle, when available."""
import shutil,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.common.config import PROJECT_ROOT
from src.common.io_utils import read_json
from src.data.prepare_factory import load_site,read_telemetry,path_for
from src.runtime.bundle import load_bundle
from src.bench.metrics import etapr_metrics

class BundleIntegration(unittest.TestCase):
    def test_author_etapr_golden_examples(self):
        labels=np.array([0,1,1,0,0,1,1,0]);pred=labels.copy()
        result=etapr_metrics(pred,labels);self.assertEqual(result['eTaPR_f1'],1.)
        self.assertEqual(etapr_metrics(np.zeros_like(labels),labels)['eTaR'],0.)
        self.assertIsNone(etapr_metrics(np.zeros_like(labels),np.zeros_like(labels))['eTaR'])
    def test_real_bundle_cpu_relocation_corruption_and_causality(self):
        bundle=PROJECT_ROOT/'bundles/continuation_smoke_primary'
        if not bundle.exists():self.skipTest('Run the explicitly named real smoke first')
        site=load_site('config/sites/hai2305.yaml')
        with tempfile.TemporaryDirectory(prefix='factory space ! ') as td:
            moved=Path(td)/'moved';shutil.copytree(bundle,moved)
            manifest,engine=load_bundle(moved,'cpu',site)
            frame=read_telemetry(path_for(site['data']['replay_files'][0]),site).iloc[:135]
            _,again=load_bundle(moved,'cpu',site)
            for values in frame.itertuples(index=False,name=None):
                row=dict(zip(frame.columns,values));first=engine.process_frame(row)
                row['label']=1;row['attack_id']='hidden';second=again.process_frame(row)
                self.assertAlmostEqual(first['detection']['score'],second['detection']['score'],places=5)
            with (moved/'preprocessing/scaler.json').open('a') as handle:handle.write(' ')
            with self.assertRaisesRegex(ValueError,'integrity'):load_bundle(moved,'cpu',site)
            site['features'][0]['unit']='invented'
            with self.assertRaises(ValueError):load_bundle(bundle,'cpu',site)
if __name__=='__main__':unittest.main()
