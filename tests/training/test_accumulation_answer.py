"""Independent scalar-gradient answer for a shorter final accumulation group."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from training_state import train_epoch

def setUpModule():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class AccumulationAnswer(unittest.TestCase):
    def test_partial_group_against_hand_gradient(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight=torch.nn.Parameter(torch.tensor(1.))
                self.mean=torch.zeros(1); self.scale=torch.ones(1)
                self.config=SimpleNamespace(target=0)
            def forward(self,x,standardized=False): return self.weight*x[:,:,0]
        model=Linear(); optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        x=torch.arange(1,8,dtype=torch.float32).reshape(7,1,1)/10
        y=torch.linspace(-.5,.5,7)
        order=torch.randperm(7,generator=torch.Generator().manual_seed(19)).numpy()
        expected=1.
        for group in (order[:4],order[4:]):
            xs=x.numpy()[group,0,0].astype(np.float64); ys=y.numpy()[group].astype(np.float64)
            expected-=.1*2*np.mean((expected*xs-ys)*xs)
        train_epoch(model,optimizer,x,y,torch.Generator().manual_seed(19),batch_size=2,accumulate=2,clip_norm=100.)
        self.assertAlmostEqual(float(model.weight),expected,delta=1e-6)
