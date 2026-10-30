import itertools
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from compare_native_latency import aggregate

class LocalLatency(unittest.TestCase):
    def records(self):
        return [dict(seed=s,trial=t,batch=b,mode=m,implementation=i,checksum=12.,
                     latency=dict(p50_ns=100 if i=='baseline' else 90,p99_ns=200))
                for s,t,b,m,i in itertools.product((7,19,43),(1,2,3),(1,8),('last','all'),
                                                   ('baseline','candidate'))]
    def test_complete_grid_and_unchanged_output_required(self):
        r=self.records()
        self.assertTrue(aggregate(r)['gate']['accepted'])
        for bad in (r[:-1],r+[r[0]]):
            with self.assertRaises(ValueError):aggregate(bad)
        r[0]['checksum']=1
        with self.assertRaises(ValueError):aggregate(r)
    def test_tail_and_secondary_workload_regressions_reject(self):
        for mode,batch,metric,value in (('last',1,'p99_ns',230),('all',1,'p50_ns',110),
                                       ('all',8,'p50_ns',110),('last',8,'p50_ns',110)):
            r=self.records()
            for cell in r:
                if cell['implementation']=='candidate' and cell['mode']==mode and cell['batch']==batch:
                    cell['latency'][metric]=value
            self.assertFalse(aggregate(r)['gate']['accepted'])
    def test_invalid_secondary_timings_fail_closed(self):
        for metric,value in (('p50_ns',float('nan')),('p99_ns',float('inf')),
                             ('p50_ns',0),('p99_ns',1)):
            r=self.records()
            record=next(x for x in r if x['mode']=='all' and x['batch']==8)
            record['latency'][metric]=value
            with self.assertRaises(ValueError):aggregate(r)
