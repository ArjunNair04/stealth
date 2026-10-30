"""Immutable input capture, all-window parity and preserved admission failures."""
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from torch_forecaster import Config,Forecaster
from runtime_common import input_bytes
from admit_runtime import run_cell,AdmissionFailure,supervise


@unittest.skipUnless(os.name=='posix','bounded compiler supervisor currently targets POSIX hosts')
class Admission(unittest.TestCase):
    def fixtures(self,path):
        m=Forecaster(Config(2,4,2,6,3,1,1,output='correction'),[0,0],[1,1],seed=7)
        (path/'model').write_bytes(m.export_bytes())
        (path/'input').write_bytes(input_bytes(list(range(24)),4,3,2))
        # Entire traces, independently known from the zero correction head.
        (path/'expected').write_bytes(struct.pack('<12f',*range(1,24,2)))
        return dict(model=path/'model',inputs=path/'input',expected=path/'expected')

    def test_complete_all_window_parity_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); paths=self.fixtures(p)
            result=run_cell(p/'admission',engine='torch_eager',batch=2,timeout=30,**paths)
            self.assertEqual((result['windows_checked'],result['max_abs_difference']),(4,0.))
            self.assertEqual(result['status'],'admitted')
            self.assertNotIn('times_ns',result)
            for name in ('started.json','result.json','complete.json','stdout.log','stderr.log','model.stlf','input.bin','expected.bin'):
                self.assertTrue((p/'admission'/name).is_file(),name)
            self.assertEqual((p/'admission'/'model.stlf').read_bytes(),paths['model'].read_bytes())
            with self.assertRaises(FileExistsError):run_cell(p/'admission',engine='torch_eager',batch=2,**paths)

    def test_bad_last_window_and_failure_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); paths=self.fixtures(p)
            good=paths['expected'].read_bytes()
            # Earlier batches pass; the final window must still be checked.
            paths['expected'].write_bytes(good[:-4]+struct.pack('<f',999.))
            with self.assertRaises(AdmissionFailure):run_cell(p/'bad',engine='torch_eager',batch=2,timeout=30,**paths)
            self.assertTrue((p/'bad'/'failed.json').is_file())
            self.assertFalse((p/'bad'/'complete.json').exists())
            self.assertIn('Runtime parity failed',(p/'bad'/'stderr.log').read_text())
            paths['expected'].write_bytes(good[:-4])
            with self.assertRaises(AdmissionFailure):run_cell(p/'short',engine='torch_eager',batch=2,timeout=30,**paths)

    def test_timeout_preserves_logs_and_no_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); paths=self.fixtures(p)
            with self.assertRaises(AdmissionFailure):run_cell(p/'slow',engine='torch_eager',batch=2,timeout=.001,**paths)
            self.assertIn('timeout',(p/'slow'/'failed.json').read_text())
            self.assertFalse((p/'slow'/'complete.json').exists())

    def test_process_group_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            marker=p/'orphan-finished'
            grandchild=f"import time; from pathlib import Path; time.sleep(.5); Path({str(marker)!r}).write_text('alive')"
            child=f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{grandchild!r}]); print('started',flush=True); time.sleep(5)"
            result=supervise([sys.executable,'-c',child],
                              p,timeout=.2,env=os.environ.copy())
            self.assertTrue(result['timed_out'])
            self.assertNotEqual(result['returncode'],0)
            self.assertIn('started',(p/'stdout.log').read_text())
            time.sleep(.5)
            self.assertFalse(marker.exists(),'compiler descendant survived its process-group timeout')

    def test_invalid_arguments_do_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);paths=self.fixtures(p)
            for extra in (dict(engine='unknown',batch=1,timeout=1),dict(engine='torch_eager',batch=True,timeout=1),
                          dict(engine='torch_eager',batch=1,timeout=0),dict(engine='torch_eager',batch=1,timeout=181)):
                with self.assertRaises(ValueError):run_cell(p/'bad',**extra,**paths)
                self.assertFalse((p/'bad').exists())


if __name__=='__main__':unittest.main()
