"""Recalculation must not reuse alarms from the deployment threshold."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.bench.analysis_exports import export_sensitivity
from src.common.io_utils import write_json,sha256_file


class AnalysisExports(unittest.TestCase):
    def fixture(self,root):
        directory=root/'scores/D01_seed_deterministic'
        for p in [directory,root/'tables',root/'plot_data',root/'evaluation']:p.mkdir(parents=True)
        write_json(root/'run_manifest.json',dict(run_id='fixture',scope='smoke'))
        pd.DataFrame([dict(task_id='score_D01_seed_deterministic',status='SUCCEEDED',artifacts='scores/D01_seed_deterministic')]).to_csv(root/'task_registry.csv',index=False)
        frame=pd.DataFrame(dict(sample_id=[f's{i}' for i in range(8)],source_file='sensor.csv',segment_id=0,
            timestamp=pd.date_range('2020',periods=8,freq='s'),score=[0.,0.,0.,0.,10.,10.,10.,10.],
            warmup=False,quality_flags='',alarm_state='NORMAL',alarm_id=None,alarm_opened=False,alarm_closed=False))
        for name in ['val_threshold','test']:frame.to_parquet(directory/f'{name}_scores.parquet',index=False)
        pd.DataFrame(dict(sample_id=frame.sample_id,label=[0,0,0,0,1,1,1,1])).to_parquet(root/'evaluation/evaluation_labels.parquet',index=False)
        write_json(directory/'threshold.json',dict(value=100.,threshold_id='frozen',model_id='D01'))
        return directory,dict(alarm=dict(on_count=2,recovery_count=5,merge_gap_s=3,off_ratio=.9),analysis=dict(threshold_quantiles=[.5])),dict(input=dict(sample_interval_seconds=1))

    def test_candidate_alarm_recomputed_and_frozen_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);directory,cfg,site=self.fixture(root)
            before={p.name:sha256_file(p) for p in directory.iterdir()}
            export_sensitivity(root,cfg,site)
            row=pd.read_csv(root/'tables/threshold_sensitivity.csv').iloc[0]
            self.assertEqual(row.candidate_threshold,5.)
            self.assertEqual(row.locked_threshold,100.)
            self.assertAlmostEqual(row.normal_event_far_per_hour,450.)
            self.assertEqual(row.test_f1,1.)
            self.assertEqual(before,{p.name:sha256_file(p) for p in directory.iterdir()})

    def test_partial_evaluation_join_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);_,cfg,site=self.fixture(root)
            path=root/'evaluation/evaluation_labels.parquet'
            pd.read_parquet(path).iloc[:-1].to_parquet(path,index=False)
            with self.assertRaisesRegex(ValueError,'cover every score'):export_sensitivity(root,cfg,site)


if __name__=='__main__':unittest.main()
