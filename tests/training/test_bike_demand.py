"""Independent count/clock examples and complete small train/export/replay checks."""
import copy
import csv
from datetime import datetime, timedelta
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import bike_demand as b
import train_bike_demand as run


def fixture(n=240, missing=()):
    out=io.StringIO(newline=''); writer=csv.writer(out);writer.writerow(b.HEADER)
    for i in range(n):
        if i in missing:continue
        d=datetime(2011,1,1)+timedelta(hours=i)
        count=20+2*d.hour+(i//24)%7
        writer.writerow([i+1,
            d.date().isoformat(),1,0,1,d.hour,0,(d.weekday()+1)%7,int(d.weekday()<5),1,
            .5,.5,.5,.1,5,count-5,count])
    # instant is a row id, not an elapsed-hour index.
    rows=list(csv.reader(io.StringIO(out.getvalue())))
    for i,row in enumerate(rows[1:],1):row[0]=str(i)
    out=io.StringIO(newline='');csv.writer(out).writerows(rows)
    return out.getvalue().encode()


def frame(n=240,missing=()):return b.parse_csv(fixture(n,missing))


def setUpModule():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)


class BikeData(unittest.TestCase):
    def test_calendar_and_counts_hand_example(self):
        f=frame()
        self.assertEqual(f.hours[:3].tolist(),[0,1,2])
        self.assertEqual(f.values[0,0],20)
        self.assertAlmostEqual(f.values[0,1],np.sin(2*np.pi/24))
        # Saturday origin hour 23 has Sunday's known calendar, Monday=0 convention.
        self.assertAlmostEqual(f.values[23,3],np.sin(2*np.pi*6/7))
        self.assertAlmostEqual(f.values[23,2],1)
        self.assertFalse(f.values.flags.writeable)
        self.assertFalse(f.hours.flags.writeable)

    def test_source_contract_and_corruption(self):
        raw=fixture()
        for changed in (raw.replace(b'cnt',b'target',1),raw.replace(b'20\r\n',b'21\r\n',1),
                        raw.replace(b',0.5,',b',nan,',1),raw+raw.splitlines(keepends=True)[-1]):
            with self.assertRaises(ValueError):b.parse_csv(changed)
        with self.assertRaises(ValueError):b.parse_csv(raw,expected_sha='0'*64)
        with self.assertRaises(ValueError):b.parse_csv(raw[:50])

    def test_gaps_exclude_whole_history_and_target(self):
        f=frame(160,missing=(60,))
        s=b.split(f,48,48,160)
        self.assertEqual(s.target_hours.tolist(),list(range(48,60))+list(range(109,160)))
        self.assertEqual(s.excluded_hours,49)
        np.testing.assert_array_equal(s.x[0,:,0],f.values[:48,0])
        self.assertEqual(float(s.y[0]),float(f.values[48,0]))
        self.assertEqual(s.input_end_hours[0],48)

    def test_split_and_yesterday_offset(self):
        f=frame();a=b.split(f,48,0,120);v=b.split(f,48,120,168)
        self.assertEqual(a.target_hours[-1],119);self.assertEqual(v.target_hours[0],120)
        self.assertEqual(v.x[0,-1,0],f.values[119,0])
        self.assertEqual(v.x[0,48-24,0],f.values[96,0])
        self.assertEqual(v.actual[0],f.values[120,0])
        for args in ((0,0,120),(48,120,0),(48,0,10),(True,0,120)):
            with self.assertRaises(ValueError):b.split(f,*args)

    def test_future_count_cannot_change_inputs_or_scaler(self):
        f=frame();g=f.values.copy();g[120:,0]+=999
        changed=b.Frame(f.hours,g);a=b.split(f,48,0,120);z=b.split(changed,48,0,120)
        np.testing.assert_array_equal(a.x,z.x);np.testing.assert_array_equal(a.y,z.y)
        m,s=b.fit_scaler(f,a);mm,ss=b.fit_scaler(changed,z)
        np.testing.assert_array_equal(m,mm);np.testing.assert_array_equal(s,ss)
        # Unique training input rows 0..118, each once; row119 is only a label.
        np.testing.assert_allclose(m,f.values[:119].mean(axis=0).astype(np.float32),rtol=0,atol=0)
        v=b.split(f,48,120,168);w=b.split(changed,48,120,168)
        np.testing.assert_array_equal(v.x[0],w.x[0]);self.assertNotEqual(v.y[0],w.y[0])

    def test_scaler_omits_rows_that_cannot_enter_any_valid_window(self):
        f=frame(180,missing=(50,));s=b.split(f,48,0,180)
        changed=f.values.copy();changed[49,0]=1e6 # hour49 can be a label, never a valid input
        a=b.fit_scaler(f,s);z=b.fit_scaler(b.Frame(f.hours,changed),s)
        for x,y in zip(a,z):np.testing.assert_array_equal(x,y)

    def test_day_cluster_interval_independent_constant_difference(self):
        actual=np.ones(48);p=np.ones(48);base=np.zeros(48)
        settings=dict(replicates=50,seed=3,confidence=.95)
        result=b.loss_interval(p,base,actual,np.arange(48),settings)
        self.assertEqual(result['mean_difference'],-1)
        self.assertEqual(result['lower'],-1);self.assertEqual(result['upper'],-1)
        self.assertEqual(result['days'],2)
        self.assertEqual(result,b.loss_interval(p,base,actual,np.arange(48),settings))


class Study(unittest.TestCase):
    def tiny(self):
        p=copy.deepcopy(run.read_recipe());p['epochs']=1;p['seeds']=[7,19]
        p['model'].update(width=4,heads=2,hidden=6)
        p['split_ends']={'train':'2011-01-06T00:00:00','validation':'2011-01-08T00:00:00','test':'2011-01-11T00:00:00'}
        p['batch_size']=8;p['evaluation_batch_size']=8;p['full_trace_parity_windows']=2
        p['uncertainty']['replicates']=50
        return p

    def test_frozen_design(self):
        p=run.read_recipe()
        self.assertEqual((p['seeds'],p['epochs']),([7,19,43],8))
        self.assertEqual(p['model']['context'],48)
        self.assertEqual(p['model']['output'],'correction')

    def test_test_counts_cannot_change_selection(self):
        f=frame();values=f.values.copy();values[168:,0]+=100
        changed=b.Frame(f.hours,values);p=self.tiny()
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';z=Path(tmp)/'z'
            run.run_study(a,f,p,{});run.run_study(z,changed,p,{})
            for name in ['scaler.json','baselines.json','selection.json','seed-7/model.stlf','seed-19/model.stlf']:
                self.assertEqual((a/name).read_bytes(),(z/name).read_bytes(),name)
            self.assertNotEqual((a/'predictions.jsonl').read_bytes(),(z/'predictions.jsonl').read_bytes())

    def test_complete_initial_persistence_native_replay_and_tamper(self):
        p=self.tiny();f=frame()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run';result=run.run_study(root,f,p,{})
            self.assertEqual(result['test_samples'],72)
            self.assertEqual(set(result['baselines']),{'last_hour','same_hour_yesterday','training_weekday_hour_mean','ridge'})
            val=b.split(f,48,120,168)
            for seed in ('7','19'):
                history=result['models'][seed]['validation_history']
                self.assertAlmostEqual(history[0]['validation']['mae'],np.abs(val.x[:,-1,0]-val.actual).mean(),places=5)
            self.assertEqual(run.replay(root,f),(root/'predictions.jsonl').read_bytes())
            self.assertEqual(run.verify_run(root)['status'],'complete')
            with self.assertRaises(FileExistsError):run.run_study(root,f,p,{})
            with (root/'seed-7/model.stlf').open('ab') as out:out.write(b'x')
            with self.assertRaises(ValueError):run.verify_run(root)

    def test_failure_keeps_checkpoint_without_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            with patch.object(run,'train_epoch',side_effect=ValueError('injected failure')):
                with self.assertRaises(ValueError):run.run_study(root,frame(),self.tiny(),{})
            self.assertTrue((root/'failure.json').exists())
            self.assertTrue((root/'seed-7/epoch-000/manifest.json').exists())
            self.assertFalse((root/'selection.json').exists());self.assertFalse((root/'complete.json').exists())


if __name__=='__main__':unittest.main()
