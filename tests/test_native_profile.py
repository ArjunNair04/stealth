import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from native_profile import summarize_profile,accept_comparison

class Profile(unittest.TestCase):
    def test_inclusive_counts_do_not_double_count_softmax(self):
        text="""private metadata
Call graph:
    100 Thread_1
      100 main  (in app)
      + 40 stealth::numeric::causal_attention(x)  (in app)
      +   10 stealth::numeric::softmax(x)  (in app)
      + 20 stealth::numeric::Dense::forward(x)  (in app)
      + 15 stealth::numeric::gelu(x)  (in app)
      + 5 stealth::numeric::layer_norm(x)  (in app)
Total number in stack
"""
        r=summarize_profile(text)
        self.assertEqual(r['samples'],100);self.assertEqual(r['counts']['attention_including_softmax'],40)
        self.assertEqual(r['counts']['other'],20);self.assertEqual(r['counts']['softmax_subset'],10)
        self.assertNotIn('private',str(r))
    def test_row_kernel_and_wrapper_are_counted_once(self):
        text = "Call graph:\n    100 Thread_1\n      40 Dense::forward(x)  (in app)\n        30 Dense::forward_rows(x)  (in app)\n      20 Dense::forward_rows(x)  (in app)\nTotal number in stack"
        result = summarize_profile(text)
        self.assertEqual(result['counts']['dense'], 60)
        self.assertEqual(result['counts']['other'], 40)

    def test_unrecognised_or_inconsistent_profile_fails(self):
        for text in ('', 'Call graph:\n  10 Thread_1\n  20 stealth::numeric::gelu(x)  (in app)\nTotal number in stack'):
            with self.assertRaises(ValueError): summarize_profile(text)
    def test_gate_requires_every_seed_and_preserves_tail(self):
        base=[dict(seed=s,trial=t,batch=b,implementation=k,latency=dict(p50_ns=(100 if k=='baseline' else 90),p99_ns=200)) for s in (7,19,43) for t in (1,2,3) for b in (1,8) for k in ('baseline','candidate')]
        self.assertTrue(accept_comparison(base)['accepted'])
        with self.assertRaises(ValueError): accept_comparison(base[:-1])
        with self.assertRaises(ValueError): accept_comparison(base+[base[0]])
        for r in base:
            if r['implementation']=='candidate' and r['seed']==7: r['latency']['p50_ns']=110
        self.assertFalse(accept_comparison(base)['accepted'])
        for r in base:
            if r['implementation']=='candidate': r['latency'].update(p50_ns=90,p99_ns=230)
        self.assertFalse(accept_comparison(base)['accepted'])
