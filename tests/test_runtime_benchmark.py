"""Timing math and native runner contracts, without training/data dependencies."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import runtime_common as common

class Timing(unittest.TestCase):
    def test_nearest_rank_and_mean(self):
        r=common.summarize([100,1,2,3])
        self.assertEqual(r,dict(count=4,min_ns=1,max_ns=100,p50_ns=2,p95_ns=100,p99_ns=100,mean_ns=26.5))
        self.assertEqual(common.summarize(range(1,101))['p99_ns'],99)
    def test_bad_timings(self):
        for a in ([],[math.nan],[math.inf],[-1],[True],['1']):
            with self.assertRaises(ValueError): common.summarize(a)
    def test_parity_checks_shape_and_each_value(self):
        self.assertEqual(common.parity([0,1],[0,1],5e-5,2e-4),0)
        self.assertAlmostEqual(common.parity([0],[4e-5],5e-5,2e-4),4e-5)
        for a,b in (([0],[.001]),([0],[0,1]),([math.nan],[0]),([0],[math.inf]),([],[])):
            with self.assertRaises(ValueError): common.parity(a,b,5e-5,2e-4)
    def test_input_format_bounds_and_truncation(self):
        blob=common.input_bytes([float(i) for i in range(24)],4,3,2)
        self.assertEqual(common.read_input(blob),(4,3,2,[float(i) for i in range(24)]))
        for bad in (blob[:-1],blob+b'x',b'x'+blob[1:],b'STLBN01\0'+struct.pack('<III',2**32-1,3,2)):
            with self.assertRaises(ValueError): common.read_input(bad)
    def test_protocol_no_reserved_tail_or_outcome_selection(self):
        p=json.loads((ROOT/'benchmarks/ettm1/runtime_v1.json').read_text())
        self.assertEqual(p['seeds'],[7,19,43]);self.assertEqual(len(p['window_ends']),32)
        self.assertEqual(p['window_ends'][0],34560)
        self.assertEqual(p['window_ends'][-1]+4,46080)
        self.assertTrue(all(e<=46080 for e in p['window_ends']))

class NativeRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary=Path(os.environ.get('STEALTH_RUNTIME_BENCH',ROOT/'build/release'/('stealth_runtime_bench.exe' if os.name=='nt' else 'stealth_runtime_bench')))
        if not cls.binary.is_file(): raise unittest.SkipTest('Build STEALTH_BUILD_BENCHMARKS to test the native runner')
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        # Independent zero-head correction: every output MUST equal its own raw target.
        f,w,h=2,4,5
        shapes=[(2,False),(2,True),(w*f,False),(w,False),(w,True),(w,False),(3*w*w,False),(3*w,False),(w*w,False),(w,False),(w,True),(w,False),(h*w,False),(h,False),(w*h,False),(w,False),(w,True),(w,False),(w,False),(1,False)]
        values=[float(one) for n,one in shapes for _ in range(n)]
        head=b'STLFC02\0'+struct.pack('<10IfIII',2,0x01020304,64,f,w,2,h,3,1,1,1e-5,len(values),1,0)
        body=head+struct.pack('<'+'f'*len(values),*values)
        (self.root/'model').write_bytes(body+hashlib.sha256(body).digest())
        self.x=[float(i)/10 for i in range(24)]
        (self.root/'input').write_bytes(common.input_bytes(self.x,4,3,2))
        (self.root/'expected').write_bytes(struct.pack('<12f',*self.x[1::2]))
    def run_native(self,batch='1',samples='3',warmup='2',output=None):
        extra=[] if output is None else ['--output',output]
        return subprocess.run([str(self.binary),'--model',str(self.root/'model'),'--input',str(self.root/'input'),'--expected',str(self.root/'expected'),'--batch',batch,'--samples',samples,'--warmup',warmup]+extra,capture_output=True,text=True)
    def test_valid_batch_and_consumed_outputs(self):
        for batch in ('1','2','4'):
            p=self.run_native(batch);self.assertEqual(p.returncode,0,p.stderr)
            r=json.loads(p.stdout);self.assertEqual(len(r['times_ns']),3)
            self.assertEqual(r['max_abs_difference'],0)
            self.assertGreater(r['checksum'],0);self.assertGreater(r['workspace_bytes'],0)
    def test_inspection_reports_serialized_version_and_output_mode(self):
        cli=self.binary.parent/('stealth.exe' if os.name=='nt' else 'stealth')
        for version,mode in ((2,'correction'),(1,'level')):
            blob=bytearray((self.root/'model').read_bytes())
            blob[6]=ord(str(version));struct.pack_into('<I',blob,8,version)
            struct.pack_into('<I',blob,56,version-1)
            blob[-32:]=hashlib.sha256(blob[:-32]).digest()
            (self.root/'model').write_bytes(blob)
            result=subprocess.run([str(cli),'--inspect-model',str(self.root/'model')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(f'format_version: {version}\n',result.stdout)
            self.assertIn(f'output_mode: {mode}\n',result.stdout)

    def test_latest_mode_selects_each_last_origin(self):
        for batch in (1,2,4):
            proc=self.run_native(str(batch),output='last');self.assertEqual(proc.returncode,0,proc.stderr)
            result=json.loads(proc.stdout);self.assertEqual(result['output_mode'],'last')
            values=struct.unpack('<24f',struct.pack('<24f',*self.x))
            expected=sum(values[(((i%(4//batch))*batch+i%batch)*3+2)*2+1] for i in range(3))
            self.assertEqual(result['checksum'],expected)
        self.assertNotEqual(self.run_native(output='unknown').returncode,0)
    def test_latest_parity_uses_only_requested_origins(self):
        expected=self.x[1::2];expected[0]=99
        (self.root/'expected').write_bytes(struct.pack('<12f',*expected))
        self.assertEqual(self.run_native(output='last').returncode,0)
        self.assertNotEqual(self.run_native().returncode,0)
        expected[2]=99
        (self.root/'expected').write_bytes(struct.pack('<12f',*expected))
        proc=self.run_native(output='last');self.assertNotEqual(proc.returncode,0);self.assertEqual(proc.stdout,'')

    def test_parity_must_fail_before_measurement(self):
        (self.root/'expected').write_bytes(struct.pack('<12f',*[99]*12))
        p=self.run_native();self.assertNotEqual(p.returncode,0);self.assertEqual(p.stdout,'')
        self.assertIn('parity',p.stderr)
    def test_malformed_inputs_and_expected(self):
        original=(self.root/'input').read_bytes()
        for bad in (original[:-1],original+b'x',b'STLBN01\0'+struct.pack('<III',2**32-1,3,2),original[:20]+struct.pack('<f',math.nan)+original[24:]):
            (self.root/'input').write_bytes(bad)
            self.assertNotEqual(self.run_native().returncode,0)
        (self.root/'input').write_bytes(original)
        for bad in (b'',struct.pack('<12f',*[math.nan]*12),b'x'*49):
            (self.root/'expected').write_bytes(bad);self.assertNotEqual(self.run_native().returncode,0)
    def test_cli_bounds_and_corrupt_model(self):
        for batch,samples,warmup in (('3','3','2'),('0','3','2'),('1','0','2'),('1','1000001','2'),('1','3','-1')):
            self.assertNotEqual(self.run_native(batch,samples,warmup).returncode,0)
        (self.root/'model').write_bytes(b'bad');self.assertNotEqual(self.run_native().returncode,0)

if __name__=='__main__': unittest.main()
