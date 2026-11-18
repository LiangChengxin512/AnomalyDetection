import tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.prepare_factory import read_telemetry

class FactoryInput(unittest.TestCase):
    def site(self):
        return dict(site_id='factory_a',input=dict(timestamp_column='time',sample_interval_seconds=1,source_timezone='UTC'),features=[
            dict(name='PRESSURE',source_column='Pressure_kPa',kind='continuous',transform={'scale':.01,'offset':0}),
            dict(name='RUNNING',source_column='Pump',kind='discrete',allowed_values=[0,1],transform={'scale':1,'offset':0})])
    def test_units_types_timezone_and_raw_quality(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'factory.csv';pd.DataFrame({'time':['2020-01-01 00:00:00','2020-01-01 00:00:01'],'Pressure_kPa':[250,300],'Pump':[1,2],'label':[1,1]}).to_csv(p,index=False)
            f=read_telemetry(p,self.site());self.assertEqual(f.PRESSURE.tolist(),[2.5,3.]);self.assertEqual(str(f.timestamp.dt.tz),'UTC')
            self.assertTrue(np.isnan(f.RUNNING.iloc[1]));self.assertNotEqual(f.quality_flags.iloc[1],'');self.assertNotIn('label',f)
    def test_missing_sensor_does_not_get_zero_filled(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'factory.csv';pd.DataFrame({'time':['2020-01-01'],'Pressure_kPa':[250]}).to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'required'):read_telemetry(p,self.site())
if __name__=='__main__':unittest.main()
