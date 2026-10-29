"""Independent answers for ridge fits, window clocks and retrospective diagnostics."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from forecast_diagnostics import (RidgeSystem, LinearModel, window_view, design_matrix,
                                  change_thresholds, describe, select_candidate, fit_linear_families, basic_metrics)
from diagnose_ettm1 import run_analysis


class Ridge(unittest.TestCase):
    def test_scalar_hand_answer_and_owned_parameters(self):
        x=np.array([[0.],[1.],[2.]])
        y=np.array([1.,3.,5.])
        model=RidgeSystem(x).fit(y,1.)
        # cov(x)=2/3, cov(x,y)=4/3 -> slope .8; intercept 3-.8*1=2.2
        np.testing.assert_allclose(model.coefficients,[.8],atol=1e-12,rtol=0)
        self.assertAlmostEqual(model.intercept,2.2,places=12)
        np.testing.assert_allclose(model.predict(x),[2.2,3.,3.8],atol=1e-12,rtol=0)
        coefficients=np.array([2.]); owned=LinearModel(coefficients,1.)
        coefficients[:]=99
        np.testing.assert_array_equal(owned.predict(x),y)

    def test_matches_independent_augmented_least_squares(self):
        x=np.array([[1.,2.],[2.,-1.],[-1.,0.],[3.,4.],[2.,2.]])
        y=np.array([2.,1.,-2.,4.,3.]); alpha=.3
        # Independent solve on raw samples plus penalty rows; intercept unpenalized.
        a=np.column_stack([np.ones(len(x)),x])
        augmented=np.vstack([a,np.column_stack([np.zeros(2),np.eye(2)*np.sqrt(len(x)*alpha)])])
        expected=np.linalg.lstsq(augmented,np.r_[y,0.,0.],rcond=None)[0]
        model=RidgeSystem(x).fit(y,alpha)
        np.testing.assert_allclose(np.r_[model.intercept,model.coefficients],expected,atol=1e-11,rtol=1e-11)

    def test_level_change_equivalence_without_shrinkage(self):
        x=np.array([[-2.,1.],[-1.,0.],[0.,-1.],[1.,0.],[2.,2.]])
        y=np.array([3.,1.,0.,2.,4.]); last=x[:,0]*2+7
        system=RidgeSystem(x)
        a=system.fit(y,0).predict(x)
        b=system.fit(y-last,0).predict(x)+last
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)

    def test_constant_feature_and_large_penalty(self):
        x=np.ones((4,2)); y=np.arange(4,dtype=float)
        m=RidgeSystem(x).fit(y,1.)
        np.testing.assert_allclose(m.predict(x),[1.5]*4,atol=1e-12)
        with self.assertRaises(ValueError): RidgeSystem(x).fit(y,0)
        x=np.arange(4,dtype=float)[:,None]
        np.testing.assert_allclose(RidgeSystem(x).fit(y,1e12).predict(x),[1.5]*4,atol=1e-10)

    def test_invalid_shapes_values_and_penalty(self):
        for x in (np.zeros((0,2)),np.ones(4),np.full((3,2),np.nan),np.zeros((2,1025))):
            with self.assertRaises(ValueError): RidgeSystem(x)
        system=RidgeSystem(np.arange(4,dtype=float)[:,None])
        for y,a in ((np.ones(3),1),(np.ones(4),-1),(np.ones(4),float('nan')),(np.full(4,np.inf),1)):
            with self.assertRaises(ValueError): system.fit(y,a)
        m=system.fit(np.ones(4),1)
        with self.assertRaises(ValueError): m.predict(np.zeros((4,2)))
        with self.assertRaises(ValueError): m.predict(np.full((2,1),np.inf))


class WindowsAndDiagnostics(unittest.TestCase):
    def test_clock_order_and_reserved_tail_exclusion(self):
        rows=np.arange(80,dtype=np.float32).reshape(40,2)
        ends=dict(train=20,validation=30,test=36)
        x,y,indices=window_view(rows,ends,'validation',context=4,horizon=2,target=1)
        np.testing.assert_array_equal(x[0,:,1],[33,35,37,39])
        self.assertEqual(y[0],43); self.assertEqual(indices[0],21)
        z=design_matrix(x,[1,0],[0,0],[1,1])
        np.testing.assert_array_equal(z[0],[33,32,35,34,37,36,39,38])
        changed=rows.copy(); changed[36:]=np.nan
        before=window_view(rows,ends,'test',context=4,horizon=2,target=1)
        after=window_view(changed,ends,'test',context=4,horizon=2,target=1)
        for a,b in zip(before,after): np.testing.assert_array_equal(a,b)
        changed=rows.copy(); changed[20:]=99
        for a,b in zip(window_view(rows,ends,'train',4,2,1),window_view(changed,ends,'train',4,2,1)):
            np.testing.assert_array_equal(a,b)

    def test_known_errors_buckets_and_undefined_correlation(self):
        y=np.array([0.,1.,2.,4.]); last=np.zeros(4); pred=np.ones(4)
        thresholds=change_thresholds(y,last,[.5,.9])
        np.testing.assert_allclose(thresholds,[1.5,3.4],atol=1e-12)
        result=describe(pred,y,last,thresholds,['a','a','b','b'])
        self.assertEqual(result['overall']['mae'],1.25)
        self.assertEqual(result['overall']['bias'],-.75)
        self.assertEqual(result['overall']['fraction_worse_than_persistence'],.25)
        self.assertEqual(result['overall']['mean_absolute_predicted_change'],1.)
        self.assertIsNone(result['overall']['change_correlation'])
        self.assertEqual([result['movement'][n]['count'] for n in ('low','middle','high')],[2,1,1])
        self.assertEqual(result['months']['b']['mae'],2.)
        allzero=describe(last,last,last,[0.,0.],['a']*4)
        self.assertIsNone(allzero['movement']['high'])
        self.assertEqual(allzero['movement']['low']['count'],4)

    def test_selection_uses_validation_only_and_stable_ties(self):
        candidates=[{'penalty':1.,'validation':{'mae':2.},'test':{'mae':0.}},
                    {'penalty':.1,'validation':{'mae':1.},'test':{'mae':99.}},
                    {'penalty':.01,'validation':{'mae':1.},'test':{'mae':999.}}]
        self.assertEqual(select_candidate(candidates)['penalty'],.01)

    def test_family_fitting_never_sees_evaluation_data(self):
        # The fitting API accepts only training and validation sets.
        train_x=np.array([[[0.]],[[1.]],[[2.]],[[3.]]],dtype=np.float32)
        val_x=np.array([[[4.]],[[5.]]],dtype=np.float32)
        records,models=fit_linear_families(train_x,np.arange(4.)*2+1,val_x,np.array([9.,11.]),
                                         [0.],[1.],{'one':[0]},['level','change'],[.01,1.],0)
        self.assertEqual(len(records),4); self.assertEqual(set(models),{'one_level','one_change'})
        for name,model in models.items(): self.assertEqual(model.coefficients.shape,(1,))

    def test_metric_overflow_fails_closed(self):
        with np.errstate(over='ignore',invalid='ignore'):
            with self.assertRaises(ValueError): basic_metrics([1e308],[-1e308])
            with self.assertRaises(ValueError): basic_metrics([1e200],[0.])

    def test_invalid_diagnostic_inputs(self):
        with self.assertRaises(ValueError): describe([1],[1,2],[0],[0,1],['a'])
        with self.assertRaises(ValueError): change_thresholds([0],[0],[.9,.5])
        with self.assertRaises(ValueError): design_matrix(np.ones((2,3,2)),[0],[0,0],[0,1])
        with self.assertRaises(ValueError): window_view(np.zeros((10,2)),dict(train=5,validation=8,test=10),'test',3,1,1)

class EndToEnd(unittest.TestCase):
    def test_frozen_selection_replay_and_unused_tail(self):
        import json
        from datetime import datetime,timedelta
        protocol=dict(experiment_id='diagnostic_test',context=4,horizon_steps=2,target_index=1,
                      split_ends=dict(train=20,validation=30,test=40),maximum_observation_end_exclusive=40,
                      feature_sets={'temperature':[1],'all':[0,1]},target_modes=['level','change'],
                      ridge_penalties=[.01,1.],movement_quantiles=[.5,.9],seasonal_period_steps=4,
                      original_score_replay_absolute_tolerance=1e-6)
        rows=np.arange(96,dtype=np.float32).reshape(48,2)/20
        clocks=[str(datetime(2020,1,1)+timedelta(minutes=15*i)) for i in range(48)]
        scaler=dict(mean=[0,0],scale=[1,1])
        def fixed(x):
            self.assertLess(float(x.max()),4.)
            return x[:,-1,1]+.1
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=run_analysis(root/'a',rows,clocks,scaler,protocol,{'frozen':fixed},{'test':'binding'})
            changed=rows.copy();changed[40:]=np.nan
            b=run_analysis(root/'b',changed,clocks,scaler,protocol,{'frozen':fixed},{'test':'binding'})
            self.assertEqual(a,b)
            self.assertEqual((root/'a/complete.json').read_bytes(),(root/'b/complete.json').read_bytes())
            shifted=rows.copy();shifted[30:40]+=1
            run_analysis(root/'c',shifted,clocks,scaler,protocol,{}, {'test':'binding'})
            self.assertEqual((root/'a/selection.json').read_bytes(),(root/'c/selection.json').read_bytes())
            original={'frozen':a['splits']['test']['frozen']['overall']}
            run_analysis(root/'verified',rows,clocks,scaler,protocol,{'frozen':fixed},{'test':'binding'},original)
            wrong={'frozen':dict(original['frozen'],mae=99.)}
            with self.assertRaisesRegex(ValueError,'Original score replay mismatch'):
                run_analysis(root/'mismatch',rows,clocks,scaler,protocol,{'frozen':fixed},{'test':'binding'},wrong)
            self.assertTrue((root/'mismatch/failure.json').exists())
            self.assertFalse((root/'mismatch/complete.json').exists())
            self.assertTrue((root/'a/complete.json').exists())
            with self.assertRaises(FileExistsError): run_analysis(root/'a',rows,clocks,scaler,protocol,{}, {})

    def test_failure_keeps_incomplete_run(self):
        from datetime import datetime,timedelta
        protocol=dict(experiment_id='diagnostic_test',context=4,horizon_steps=2,target_index=1,
                      split_ends=dict(train=20,validation=30,test=40),maximum_observation_end_exclusive=40,
                      feature_sets={'temperature':[1]},target_modes=['level'],ridge_penalties=[.01],
                      movement_quantiles=[.5,.9],seasonal_period_steps=4)
        rows=np.arange(80,dtype=np.float32).reshape(40,2)
        clocks=[str(datetime(2020,1,1)+timedelta(minutes=15*i)) for i in range(40)]
        def broken(x): raise RuntimeError('test failure')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed'
            with self.assertRaises(RuntimeError): run_analysis(path,rows,clocks,dict(mean=[0,0],scale=[1,1]),protocol,{'broken':broken},{})
            self.assertTrue((path/'failure.json').exists());self.assertFalse((path/'complete.json').exists())


if __name__=='__main__': unittest.main()
