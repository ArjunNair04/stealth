"""Malformed NumPy declarations must fail before any array allocation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import numpy as np
import torch
from torch_forecaster import Config,Forecaster
from training_state import save_checkpoint,load_checkpoint,make_optimizer

def setUpModule():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class ArchiveBounds(unittest.TestCase):
    def test_false_array_size_rejected_before_numpy_load(self):
        m=Forecaster(Config(1,2,1,3,2,1,0),[0],[1])
        o=make_optimizer(m,dict(lr=.001,weight_decay=.01,betas=[.9,.999],eps=1e-8))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state'; save_checkpoint(path,m,o,torch.Generator(),{'epoch':0},{})
            with zipfile.ZipFile(path/'arrays.npz') as z: files={n:z.read(n) for n in z.namelist()}
            header=io.BytesIO()
            np.lib.format.write_array_header_1_0(header,dict(descr='<f4',fortran_order=False,shape=(2**40,)))
            files['flat.npy']=header.getvalue()
            with zipfile.ZipFile(path/'arrays.npz','w') as z:
                for name,value in files.items(): z.writestr(name,value)
            meta=json.loads((path/'manifest.json').read_text())
            meta['arrays_sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
            (path/'manifest.json').write_text(json.dumps(meta))
            with patch('training_state.np.load',side_effect=AssertionError('Array loader reached before header bounds')):
                with self.assertRaises(ValueError): load_checkpoint(path,{})
