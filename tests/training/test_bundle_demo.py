"""Small real-workflow tests; no network and no public-data dependency."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import prepare_public_benchmark as data
from torch_forecaster import Config, Forecaster
from model_bundle import create_bundle, load_bundle
from bundle_demo import read_recipe, demo_specs, replay_decisions, run_demo, verify_demo


def setUpModule():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class Demo(unittest.TestCase):
    def test_frozen_recipe_and_reserved_tail(self):
        p,task,c=read_recipe()
        self.assertEqual((p['seeds'],p['epochs'],p['split_ends']),([7],2,dict(train=2048,validation=2560,test=3072)))
        self.assertEqual((c.context,c.horizon,c.width,c.heads,c.hidden,c.output),(96,4,64,4,256,'correction'))
        self.assertEqual(task['input_columns'],list(data.COLUMNS[1:]))
        self.assertLess(p['split_ends']['test'],data.SOURCE['split_ends']['train'])

    def test_replay_alignment_and_future_suffix(self):
        values=np.arange(80,dtype=np.float32).reshape(40,2)
        c=Config(2,4,2,6,3,2,1,output='correction')
        specs=demo_specs(['load','temperature'],10)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle'
            ident=create_bundle(path,Forecaster(c,[0,0],[1,1],seed=7).export_bytes(),specs,
                                interval_ns=10,grid_origin_ns=1,clock_domain='synthetic')
            bundle=load_bundle(path,expected_id=ident)
            a=replay_decisions(bundle,values,5,11)
            changed=values.copy(); changed[11:]=999
            self.assertEqual(a,replay_decisions(bundle,changed,5,11))
            self.assertEqual([r.value for r in a],[None,None,15.,17.,19.,21.])
            self.assertEqual(a[2].decision_at_ns,71)
            self.assertEqual(a[2].target_at_ns,91)
            for start,end in ((-1,11),(5,41),(6,6)):
                with self.assertRaises(ValueError): replay_decisions(bundle,values,start,end)

    def test_complete_tiny_train_bundle_replay_and_seal(self):
        p,task,_=read_recipe()
        p=copy.deepcopy(p)
        p.update(split_ends=dict(train=24,validation=34,test=44),batch_size=4,
                 evaluation_batch_size=4,native_parity_windows_per_split=2)
        task={**task,'input_columns':['a','b'],'context':8,'horizon_steps':2,'target_column':'b','interval_seconds':1}
        c=Config(2,4,2,6,8,2,1,output='correction')
        frame=data.Frame(tuple(str(i) for i in range(44)),tuple((i/10,1+i/20) for i in range(44)))
        binding={'protocol_sha256':'a'*64,'task_sha256':'b'*64,'data_sha256':'c'*64,'code_commit':'d'*40}
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            result=run_demo(output,frame,p,task,c,binding)
            sealed=verify_demo(output)
            self.assertEqual(result['scores']['native']['count'],9)
            self.assertEqual(result['replay']['warming_up'],7)
            self.assertEqual(sealed['bundle_id'],result['bundle_id'])
            with self.assertRaises(FileExistsError): run_demo(output,frame,p,task,c,binding)
            changed=(output/'bundle'/'model.stlf').read_bytes()
            (output/'bundle'/'model.stlf').write_bytes(changed[:-1])
            with self.assertRaises(ValueError): verify_demo(output)


if __name__=='__main__': unittest.main()
