"""Clock alignment, baseline answers, manifests and a complete miniature replay."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from torch_forecaster import Config
from train_ettm1 import split_tensors, baseline_predictions, run_seed, verify_run, evaluate_runs, read_recipe


class Experiment(unittest.TestCase):
    def test_target_clocks_and_baselines(self):
        rows=torch.arange(80,dtype=torch.float32).reshape(40,2)
        c=Config(2,4,2,5,8,2,1); ends=dict(train=20,validation=30,test=40)
        x,y=split_tensors(rows,ends,'validation',c)
        self.assertEqual(tuple(x.shape),(9,8,2))
        self.assertEqual(x[0,0,1],25)
        self.assertEqual(x[0,-1,1],39)
        self.assertEqual(y[0],43)
        self.assertEqual(y[-1],59)
        last,seasonal=baseline_predictions(x,c,period=8)
        self.assertEqual(last[0],39)
        self.assertEqual(seasonal[0],27)
        changed=rows.clone(); changed[20:]=999
        train_before=split_tensors(rows,ends,'train',c)
        train_after=split_tensors(changed,ends,'train',c)
        for a,b in zip(train_before,train_after): self.assertTrue(torch.equal(a,b))

    def test_invalid_window_and_metric_protocol(self):
        rows=torch.zeros(40,2); c=Config(2,4,2,5,8,2,1)
        with self.assertRaises(ValueError): split_tensors(rows,dict(train=30,validation=20,test=40),'train',c)
        with self.assertRaises(ValueError): baseline_predictions(rows[None],c,period=1)
        protocol,task,c=read_recipe()
        self.assertEqual(protocol['seeds'],[7,19,43]); self.assertEqual(c.context,96)
        self.assertEqual(task['target_column'],'OT')

    def test_complete_run_resume_replay_and_sealed_evaluation(self):
        protocol,task,_=read_recipe(); protocol=copy.deepcopy(protocol)
        protocol.update(epochs=2,seeds=[7,19],batch_size=4,accumulate=2,evaluation_batch_size=8,native_parity_windows_per_split=2)
        c=Config(2,4,2,5,8,2,1)
        task={**task,'input_columns':['a','b'],'context':8,'horizon_steps':2,'target_column':'b'}
        rows=torch.arange(100,dtype=torch.float32).reshape(50,2)/100
        ends=dict(train=26,validation=38,test=50)
        binding={'protocol_sha256':'a'*64,'task_sha256':'b'*64,'data_sha256':'c'*64,'scaler_sha256':'d'*64,'code_commit':'e'*40}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=run_seed(root/'seed7',rows,ends,c,[0,0],[1,1],protocol,task,binding,7)
            replay=run_seed(root/'replay',rows,ends,c,[0,0],[1,1],protocol,task,binding,7,resume=root/'seed7'/'epoch-001')
            self.assertEqual(a['history'],replay['history'])
            self.assertEqual((root/'seed7'/'model.stlf').read_bytes(),(root/'replay'/'model.stlf').read_bytes())
            with self.assertRaises(ValueError): evaluate_runs(root/'bad',[root/'seed7'],rows,ends,c,protocol,task,binding,period=8)
            self.assertFalse((root/'bad').exists())
            run_seed(root/'seed19',rows,ends,c,[0,0],[1,1],protocol,task,binding,19)
            result=evaluate_runs(root/'scores',[root/'seed7',root/'seed19'],rows,ends,c,protocol,task,binding,period=8)
            self.assertEqual(set(result['models']),{'7','19'})
            self.assertEqual(result['baselines']['last_observation']['count'],11)
            self.assertTrue((root/'scores'/'complete.json').exists())
            with self.assertRaises(FileExistsError): evaluate_runs(root/'scores',[root/'seed7',root/'seed19'],rows,ends,c,protocol,task,binding,period=8)
            blob=(root/'seed7'/'model.stlf').read_bytes()
            (root/'seed7'/'model.stlf').write_bytes(blob[:-1])
            with self.assertRaises(ValueError): verify_run(root/'seed7',protocol,binding,7)

if __name__=='__main__': unittest.main()
