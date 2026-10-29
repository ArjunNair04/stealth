from pathlib import Path
import copy
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from train_correction_ettm1 import read_recipe,validate_recipe,experiment_rows
from train_ettm1 import read_recipe as original_recipe

class CorrectionProtocol(unittest.TestCase):
    def test_frozen_budget_and_one_architecture_change(self):
        p,t,c=read_recipe();old,_,base=original_recipe()
        for k in ('seeds','epochs','batch_size','accumulate','clip_norm','optimizer','selection','evaluation_batch_size'):
            self.assertEqual(p[k],old[k])
        self.assertEqual(c.output,'correction');self.assertEqual(c.width,base.width)
        self.assertEqual(c.context,96);self.assertEqual(c.horizon,4)
        self.assertEqual(p['maximum_observation_end_exclusive'],57600)
        for key,value in (('epochs',7),('seeds',[7]),('maximum_observation_end_exclusive',69680)):
            changed=copy.deepcopy(p);changed[key]=value
            with self.assertRaises(ValueError):validate_recipe(changed,t)
        changed=copy.deepcopy(t);changed['model']['blocks']=2
        with self.assertRaises(ValueError):validate_recipe(p,changed)

    def test_reserved_tail_excluded_before_tensor_construction(self):
        values=np.zeros((57610,7),dtype=np.float64);values[57600:]=np.nan
        rows=experiment_rows(values)
        self.assertEqual(tuple(rows.shape),(57600,7))
        self.assertTrue(bool(rows.isfinite().all()))
        with self.assertRaises(ValueError):experiment_rows(values[:50000])
        values[0,0]=np.nan
        with self.assertRaises(ValueError):experiment_rows(values)

if __name__=='__main__':unittest.main()
