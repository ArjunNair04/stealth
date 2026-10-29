"""Small NumPy-only linear baselines and descriptive forecast-error checks."""
from dataclasses import dataclass
import math
import numpy as np
import prepare_public_benchmark as data


def matrix(value):
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=2 or not 1<=a.shape[0]<=100000 or not 1<=a.shape[1]<=1024 or not np.isfinite(a).all():
        raise ValueError('Expected bounded finite matrix')
    return a


def vector(value,length):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(length,) or not np.isfinite(a).all(): raise ValueError('Invalid finite vector')
    return a


@dataclass(frozen=True)
class LinearModel:
    coefficients: np.ndarray
    intercept: float

    def __post_init__(self):
        c=np.array(self.coefficients,dtype=np.float64,copy=True)
        if c.ndim!=1 or not 1<=len(c)<=1024 or not np.isfinite(c).all() or not math.isfinite(self.intercept):
            raise ValueError('Invalid linear parameters')
        c.flags.writeable=False
        object.__setattr__(self,'coefficients',c)
        object.__setattr__(self,'intercept',float(self.intercept))

    def predict(self,x):
        x=matrix(x)
        if x.shape[1]!=len(self.coefficients): raise ValueError('Linear input width mismatch')
        y=x@self.coefficients+self.intercept
        if not np.isfinite(y).all(): raise ValueError('Nonfinite linear prediction')
        return y


class RidgeSystem:
    """Fit MSE + alpha*||weights||², centering training data for a free intercept."""
    def __init__(self,x):
        x=matrix(x)
        self.mean=x.mean(axis=0)
        self.centered=x-self.mean
        covariance=self.centered.T@self.centered/len(x)
        if not np.isfinite(covariance).all(): raise ValueError('Nonfinite covariance')
        values,vectors=np.linalg.eigh(covariance)
        if not np.isfinite(values).all() or values[0]<-1e-10*max(1.,float(values[-1])):
            raise ValueError('Invalid covariance eigendecomposition')
        self.values=np.maximum(values,0.)
        self.vectors=vectors

    def fit(self,y,alpha):
        y=vector(y,len(self.centered))
        if not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or not 0<=alpha<=1e12:
            raise ValueError('Invalid ridge penalty')
        if alpha==0 and self.values[0]<=1e-12*max(1.,float(self.values[-1])):
            raise ValueError('Unregularized design is singular or ill-conditioned')
        mean=float(y.mean())
        cross=self.centered.T@(y-mean)/len(y)
        coefficients=self.vectors@((self.vectors.T@cross)/(self.values+alpha))
        return LinearModel(coefficients,mean-float(self.mean@coefficients))


def window_view(rows,ends,split,context,horizon,target):
    if (not isinstance(rows,np.ndarray) or rows.ndim!=2 or rows.dtype!=np.float32
            or not 1<=rows.shape[1]<=64 or type(target) is not int or not 0<=target<rows.shape[1]):
        raise ValueError('Expected FP32 observations and valid target')
    windows=list(data.windows(len(rows),ends,split,context,horizon,'point'))
    if not windows: raise ValueError('Empty window split')
    first,last=windows[0],windows[-1]
    inputs=rows[first.input_start:last.input_end]
    x=np.lib.stride_tricks.sliding_window_view(inputs,context,axis=0).transpose(0,2,1)
    indices=np.arange(first.target_start,last.target_end,dtype=np.int64)
    y=rows[indices,target].astype(np.float64)
    if len(x)!=len(y) or not np.isfinite(inputs).all() or not np.isfinite(y).all():
        raise ValueError('Invalid selected observations')
    return x,y,indices


def design_matrix(x,columns,mean,scale):
    x=np.asarray(x)
    if (x.ndim!=3 or not 1<=x.shape[0]<=100000 or not 1<=x.shape[1]<=1024 or not 1<=x.shape[2]<=64
            or not columns or len(set(columns))!=len(columns)
            or any(type(c) is not int or not 0<=c<x.shape[2] for c in columns)
            or x.shape[1]*len(columns)>1024):
        raise ValueError('Invalid lagged feature layout')
    mean=vector(mean,x.shape[2]); scale=vector(scale,x.shape[2])
    if not (scale>0).all() or not np.isfinite(x).all(): raise ValueError('Invalid input scale or observations')
    values=(x[:,:,columns].astype(np.float64)-mean[columns])/scale[columns]
    return matrix(values.reshape(len(x),-1))


def basic_metrics(prediction,target):
    y=np.asarray(target,dtype=np.float64)
    if y.ndim!=1 or not len(y): raise ValueError('Empty target')
    y=vector(y,len(y)); p=vector(prediction,len(y))
    with np.errstate(over='ignore',invalid='ignore'):
        error=p-y
        mae=float(np.mean(np.abs(error)))
        rmse=float(np.sqrt(np.mean(error**2)))
    if not math.isfinite(mae) or not math.isfinite(rmse): raise ValueError('Nonfinite error metric')
    return dict(count=len(y),mae=mae,rmse=rmse)


def select_candidate(candidates):
    if not candidates or any(not math.isfinite(r['validation']['mae']) for r in candidates):
        raise ValueError('Invalid selection candidates')
    return min(candidates,key=lambda r:(r['validation']['mae'],r['penalty']))


def fit_linear_families(train_x,train_y,val_x,val_y,mean,scale,feature_sets,modes,penalties,target):
    if (not feature_sets or not modes or any(m not in ('level','change') for m in modes)
            or not penalties or len(set(penalties))!=len(penalties)):
        raise ValueError('Invalid linear families')
    train_y=vector(train_y,len(train_x)); val_y=vector(val_y,len(val_x))
    records=[]; selected={}
    for name,columns in feature_sets.items():
        x=design_matrix(train_x,columns,mean,scale); v=design_matrix(val_x,columns,mean,scale)
        system=RidgeSystem(x)
        for mode in modes:
            family=name+'_'+mode
            offset=train_x[:,-1,target].astype(np.float64) if mode=='change' else np.zeros(len(x))
            val_offset=val_x[:,-1,target].astype(np.float64) if mode=='change' else np.zeros(len(v))
            candidates=[]; models={}
            for penalty in penalties:
                model=system.fit(train_y-offset,penalty)
                record=dict(family=family,features=list(columns),target_mode=mode,penalty=penalty,
                            parameters=len(model.coefficients)+1,
                            training=basic_metrics(model.predict(x)+offset,train_y),
                            validation=basic_metrics(model.predict(v)+val_offset,val_y))
                records.append(record); candidates.append(record); models[penalty]=model
            selected[family]=models[select_candidate(candidates)['penalty']]
    return records,selected


def change_thresholds(target,last,quantiles):
    y=np.asarray(target,dtype=np.float64)
    if y.ndim!=1 or not len(y) or len(quantiles)!=2 or not 0<quantiles[0]<quantiles[1]<1:
        raise ValueError('Invalid movement quantiles')
    y=vector(y,len(y)); last=vector(last,len(y))
    return np.quantile(np.abs(y-last),quantiles,method='linear').tolist()


def describe(prediction,target,last,thresholds,months):
    y=np.asarray(target,dtype=np.float64)
    if y.ndim!=1 or not len(y): raise ValueError('Invalid target')
    y=vector(y,len(y)); p=vector(prediction,len(y)); last=vector(last,len(y))
    if len(thresholds)!=2 or not 0<=thresholds[0]<=thresholds[1]<float('inf') or len(months)!=len(y):
        raise ValueError('Invalid diagnostic grouping')
    actual=y-last; proposed=p-last
    def score(mask):
        if not mask.any(): return None
        result=basic_metrics(p[mask],y[mask]); a,b=actual[mask],proposed[mask]
        centered_a=a-a.mean(); centered_b=b-b.mean()
        denominator=float(np.linalg.norm(centered_a)*np.linalg.norm(centered_b))
        correlation=None if denominator==0 else float(np.clip(centered_a@centered_b/denominator,-1.,1.))
        result.update(bias=float(np.mean(p[mask]-y[mask])),
                      mean_absolute_actual_change=float(np.mean(np.abs(a))),
                      mean_absolute_predicted_change=float(np.mean(np.abs(b))),
                      change_correlation=correlation,
                      fraction_worse_than_persistence=float(np.mean(np.abs(p[mask]-y[mask])>np.abs(a))))
        return result
    movement=np.abs(actual); labels=np.asarray(months)
    return dict(overall=score(np.ones(len(y),dtype=bool)),
                movement={'low':score(movement<=thresholds[0]),
                          'middle':score((movement>thresholds[0])&(movement<=thresholds[1])),
                          'high':score(movement>thresholds[1])},
                months={label:score(labels==label) for label in sorted(set(months))})
