"""Installed extension tests; expected answers use independent equations."""
import concurrent.futures
import gc
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
import stealth_cpu as s
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import generate_forecast_fixture as fixture


def data():
    p, x, y, _, artifact = fixture.fixture()
    c = s.ForecastConfig(7, 4, 2, 7, 5, 4, 6)
    return c, np.array(p, dtype=np.float32), np.array(x, dtype=np.float32).reshape(5, 7), np.array(y), artifact


class Components(unittest.TestCase):
    def test_dense_hand_answer_and_parameter_ownership(self):
        w = np.array([[1, 2], [-1, 3]], dtype=np.float32)
        b = np.array([1, -2], dtype=np.float32)
        d = s.Dense(w, b)
        w[:] = 99; b[:] = 99
        np.testing.assert_array_equal(d.forward(np.array([2, 4], dtype=np.float32)), [11, 8])

    def test_gelu(self):
        x = np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        expected = [.5 * float(v) * math.erfc(-float(v) / math.sqrt(2)) for v in x]
        np.testing.assert_allclose(s.gelu(x), expected, atol=2e-6, rtol=2e-5)

    def test_softmax(self):
        x = np.array([1000, 1001, 999], dtype=np.float32)
        e = np.exp(x.astype(np.float64)-1001); e /= e.sum()
        np.testing.assert_allclose(s.softmax(x), e, atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(s.softmax(x, active=1), [1, 0, 0])
        with self.assertRaises(ValueError): s.softmax(x, active=0)

    def test_norm_and_residual(self):
        x = np.array([1, 3], dtype=np.float32)
        scale = np.array([2, 3], dtype=np.float32); shift = np.array([4, 5], dtype=np.float32)
        expected = (x.astype(np.float64)-2)/math.sqrt(1+1e-5)*scale+shift
        np.testing.assert_allclose(s.layer_norm(x, scale, shift), expected, atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(s.residual_add(x, scale), [3, 6])

    def test_attention_hand_answer_two_heads(self):
        q = np.zeros((3, 4), dtype=np.float32)
        v = np.array([[1,2,3,4], [3,4,5,6], [5,6,7,8]], dtype=np.float32)
        expected = np.array([[1,2,3,4], [2,3,4,5], [3,4,5,6]], dtype=np.float32)
        np.testing.assert_allclose(s.causal_attention(q, q, v, heads=2), expected, atol=2e-6, rtol=2e-5)
        with self.assertRaises(ValueError): s.causal_attention(q, q, v, heads=3)

    def test_feed_forward_independent(self):
        x = np.array([1, 3], dtype=np.float32)
        one = np.ones(2, dtype=np.float32); zero = np.zeros(2, dtype=np.float32)
        eye = np.eye(2, dtype=np.float32)
        ff = s.FeedForward(one, zero, eye, zero, eye, zero)
        n = (x.astype(np.float64)-2)/math.sqrt(1+1e-5)
        expected = x + np.array([.5*v*math.erfc(-v/math.sqrt(2)) for v in n])
        np.testing.assert_allclose(ff.forward(x), expected, atol=2e-6, rtol=2e-5)

    def test_strict_buffers(self):
        bad = [[1.0], np.ones(2, dtype=np.float64), np.ones(2, dtype=np.int32),
               np.ones(4, dtype=np.float32)[::2], np.ones(2, dtype='>f4'),
               np.ndarray((2,), dtype=np.float32, buffer=bytearray(9), offset=1)]
        for x in bad:
            with self.subTest(value=repr(x)), self.assertRaises((ValueError, TypeError)): s.gelu(x)
        with self.assertRaises(ValueError): s.gelu(np.ones((2,2), dtype=np.float32))
        with self.assertRaises(ValueError): s.gelu(np.array([np.nan], dtype=np.float32))
        with self.assertRaises(ValueError): s.softmax(np.array([], dtype=np.float32))

    def test_component_failures(self):
        x = np.ones(2,dtype=np.float32); one = np.ones(1,dtype=np.float32)
        with self.assertRaises(ValueError): s.Dense(np.ones((2,2),dtype=np.float32),one)
        with self.assertRaises(ValueError): s.Dense(np.eye(2,dtype=np.float32),x).forward(one)
        with self.assertRaises(ValueError): s.layer_norm(x,one,x)
        with self.assertRaises(ValueError): s.layer_norm(x,x,x,epsilon=0)
        with self.assertRaises(ValueError): s.residual_add(x,one)
        with self.assertRaises(ValueError): s.causal_attention(x.reshape(1,2),x.reshape(2,1),x.reshape(1,2),1)
        with self.assertRaises(ValueError): s.FeedForward(x,x,np.eye(2,dtype=np.float32),x,np.ones((1,2),dtype=np.float32),one)
        huge = np.full(2,np.finfo(np.float32).max,dtype=np.float32)
        with self.assertRaises(OverflowError): s.residual_add(huge,huge)

    def test_readonly_input_and_owned_output(self):
        x = np.array([1,2,3], dtype=np.float32); x.flags.writeable = False
        y = s.gelu(x); before = y.copy(); del x; gc.collect()
        self.assertTrue(y.flags.owndata); np.testing.assert_array_equal(y, before)


class Models(unittest.TestCase):
    def setUp(self):
        self.c, self.p, self.x, self.y, self.artifact = data()
        self.m = s.ForecastModel(self.c, self.p)

    def test_fixture_and_kernels(self):
        for kernel in ['reference', 'packed']:
            model = s.ForecastModel(self.c, self.p, kernel=kernel)
            np.testing.assert_allclose(model.session().predict(self.x), self.y, atol=5e-5, rtol=2e-4)
        self.assertEqual(s.parameter_count(self.c), self.p.size)
        with self.assertRaises(ValueError): s.ForecastModel(self.c, self.p, kernel='unknown')

    def test_full_task_fixture(self):
        sizes = (7,7,64*7,64,64,64,3*64*64,3*64,64*64,64,64,64,256*64,256,64*256,64,64,64,64,1)
        p = []
        for index, size in enumerate(sizes):
            if index == 0: values = [float(i)-3 for i in range(size)]
            elif index == 1: values = [float(i)+1 for i in range(size)]
            else:
                values = fixture.a.pattern(size,index*7+3,128)
                if index in (4,10,16): values = [1+v for v in values]
            p.extend(values)
        x = fixture.a.pattern(96*7,17,8)
        y = fixture.full_case()
        model = s.ForecastModel(s.ForecastConfig(7,64,4,256,96,4,6), np.array(p,dtype=np.float32))
        np.testing.assert_allclose(model.session().predict(np.array(x,dtype=np.float32).reshape(96,7)), y, atol=5e-5,rtol=2e-4)

    def test_readonly_configuration(self):
        with self.assertRaises(AttributeError): self.c.width = 16
        for args in [(0,4,2,7,5,4,6), (7,4,3,7,5,4,6), (7,4,2,7,1025,4,6)]:
            with self.assertRaises((ValueError,RuntimeError)): s.ForecastConfig(*args)

    def test_causal_prefix_and_batch(self):
        session = self.m.session()
        for n in range(1,6):
            np.testing.assert_allclose(session.predict(self.x[:n]), self.y[:n], atol=5e-5,rtol=2e-4)
        batch = np.stack([self.x, self.x*2])
        result = session.predict_batch(batch)
        np.testing.assert_array_equal(result[0], session.predict(batch[0]))
        np.testing.assert_array_equal(result[1], session.predict(batch[1]))
        self.assertTrue(result.flags.owndata)

    def test_failure_recovery_and_input_preservation(self):
        session = self.m.session(); original = self.x.copy()
        for bad in [self.x.astype(np.float64), self.x[:,::-1], self.x.T.copy(), self.x[:0], np.zeros((6,7),dtype=np.float32)]:
            with self.assertRaises((ValueError,TypeError)): session.predict(bad)
        bad = self.x.copy(); bad[-1,-1] = np.inf
        with self.assertRaises(ValueError): session.predict(bad)
        batch = np.stack([self.x,bad])
        with self.assertRaises(ValueError): session.predict_batch(batch)
        with self.assertRaises(ValueError): session.predict_batch(np.zeros((0,5,7), dtype=np.float32))
        np.testing.assert_allclose(session.predict(self.x),self.y,atol=5e-5,rtol=2e-4)
        np.testing.assert_array_equal(self.x,original)

    def test_readonly_prediction_input(self):
        self.x.flags.writeable = False
        np.testing.assert_allclose(self.m.session().predict(self.x),self.y,atol=5e-5,rtol=2e-4)

    def test_separate_streams_concurrent(self):
        def work(_):
            stream = self.m.window_stream()
            for episode in range(20):
                stream.reset()
                for row in self.x: result = stream.push(row)
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(work,range(8)): self.assertAlmostEqual(result,self.y[-1],places=4)

    def test_parameter_and_model_lifetime(self):
        session = self.m.session(); self.p[:] = np.nan; del self.m; gc.collect()
        first = session.predict(self.x); session.predict(self.x*2)
        np.testing.assert_allclose(first,self.y,atol=5e-5,rtol=2e-4)
        self.assertTrue(first.flags.owndata)

    def test_load_bytes_and_file(self):
        for m in [s.ForecastModel.from_bytes(self.artifact), self.m]:
            np.testing.assert_allclose(m.session().predict(self.x),self.y,atol=5e-5,rtol=2e-4)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.stlf'; path.write_bytes(self.artifact)
            np.testing.assert_allclose(s.ForecastModel.load(path).session().predict(self.x),self.y,atol=5e-5,rtol=2e-4)
            with self.assertRaises(RuntimeError): s.ForecastModel.load(Path(d)/'missing')
        for bad in [b'',self.artifact[:-1],bytes([self.artifact[0]^1])+self.artifact[1:]]:
            with self.assertRaises(ValueError): s.ForecastModel.from_bytes(bad)
        with self.assertRaises((ValueError,RuntimeError)): s.ForecastModel.from_bytes(b'x'*(16*1024*1024+1))

    def test_stream_rolling_reset_and_failure(self):
        stream = self.m.window_stream(); session = self.m.session(); rows = []
        for i in range(30):
            row = self.x[i%5]; rows.append(row); rows = rows[-5:]
            count = stream.size
            with self.assertRaises(ValueError): stream.push(np.full(7,np.nan,dtype=np.float32))
            self.assertEqual(stream.size,count)
            actual = stream.push(row)
            if i<4: self.assertIsNone(actual); self.assertFalse(stream.ready)
            else:
                self.assertTrue(stream.ready)
                self.assertEqual(actual,float(session.predict(np.array(rows,dtype=np.float32))[-1]))
        stream.reset(); stream.reset(); self.assertEqual(stream.size,0); self.assertFalse(stream.ready)
        for row in self.x: result = stream.push(row)
        self.assertAlmostEqual(result,self.y[-1],places=4)
        del self.m; gc.collect(); self.assertIsInstance(stream.push(self.x[0]),float)

    def test_separate_sessions_concurrent(self):
        def work(i):
            session = self.m.session()
            for _ in range(30): result = session.predict(self.x)
            return result
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(work,range(8)):
                np.testing.assert_allclose(result,self.y,atol=5e-5,rtol=2e-4)

    def test_shared_session_rejects_overlap(self):
        session = self.m.session()
        batch = np.repeat(self.x[None,:,:],32768,axis=0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(session.predict_batch,batch)
            deadline = time.monotonic()+10
            while not session.busy and not future.done() and time.monotonic()<deadline: time.sleep(.001)
            self.assertTrue(session.busy,'long native call must expose overlap for this test')
            with self.assertRaises(RuntimeError): session.predict(self.x)
            result = future.result(timeout=60)
        self.assertFalse(session.busy)
        np.testing.assert_allclose(result[-1],self.y,atol=5e-5,rtol=2e-4)

    def test_batch_resource_limit(self):
        # Resource checks happen before copying even for a valid dense layout.
        for count in [430000, 480000]:
            batch = np.empty((count,5,7),dtype=np.float32)
            with self.assertRaises(ValueError): self.m.session().predict_batch(batch)
            del batch
        # Dense broadcast views are rejected rather than silently copied into large arrays.
        with self.assertRaises(ValueError): self.m.session().predict_batch(np.broadcast_to(self.x,(1000000,5,7)))


if __name__ == '__main__': unittest.main()
