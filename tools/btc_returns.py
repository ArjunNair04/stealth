"""Causal BTC hourly return features, chronological samples and simple baselines."""
from dataclasses import dataclass
import math
import numpy as np
from hourly_bars import HOUR_NS, MAX_ROWS

FEATURES = ('return_bps','log_range_bps','log1p_quote_volume','hour_sin','hour_cos')


@dataclass(frozen=True)
class FeatureFrame:
    end_ns: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        clocks=np.array(self.end_ns,copy=True); values=np.array(self.values,dtype=np.float64,copy=True)
        if (clocks.dtype!=np.int64 or clocks.ndim!=1 or not 2<=len(clocks)<=MAX_ROWS
                or values.shape!=(len(clocks),len(FEATURES)) or not np.isfinite(values).all()
                or (np.abs(values)>np.finfo(np.float32).max).any()
                or (clocks<0).any() or (clocks%HOUR_NS).any()
                or not (np.diff(clocks)==HOUR_NS).all()):
            raise ValueError('Invalid contiguous hourly feature frame')
        clocks.flags.writeable=False;values.flags.writeable=False
        object.__setattr__(self,'end_ns',clocks);object.__setattr__(self,'values',values)


def features(bars):
    if not 3<=len(bars)<=MAX_ROWS:raise ValueError('Need bounded hourly history')
    prior=None;values=[];clocks=[]
    for bar in bars:
        if (bar.symbol!='BTCUSDT' or bar.end_ns-bar.start_ns!=HOUR_NS
                or (prior is not None and bar.start_ns!=prior.end_ns)
                or not all(math.isfinite(x) for x in (bar.open,bar.high,bar.low,bar.close,bar.quote_volume))
                or not (0<bar.low<=bar.open<=bar.high and bar.low<=bar.close<=bar.high)
                or bar.quote_volume<0):
            raise ValueError('Invalid BTC hourly source series')
        if prior is not None:
            hour=(bar.end_ns//HOUR_NS)%24
            values.append((10000*(bar.close/prior.close-1),
                           10000*(math.log(bar.high)-math.log(bar.low)),
                           math.log1p(bar.quote_volume),math.sin(2*math.pi*hour/24),math.cos(2*math.pi*hour/24)))
            clocks.append(bar.end_ns)
        prior=bar
    return FeatureFrame(np.asarray(clocks,dtype=np.int64),np.asarray(values,dtype=np.float64))


@dataclass(frozen=True)
class Samples:
    x: np.ndarray
    y: np.ndarray  # FP32 training labels; scoring uses original FP64 actual below.
    actual: np.ndarray
    origins: np.ndarray
    input_end_ns: np.ndarray
    label_end_ns: np.ndarray


def split(frame,context,start_ns,end_ns):
    if (type(context) is not int or not 1<=context<=1024
            or type(start_ns) is not int or type(end_ns) is not int
            or not 0<=start_ns<end_ns or start_ns%HOUR_NS or end_ns%HOUR_NS):
        raise ValueError('Invalid split or context')
    origin=np.arange(context-1,len(frame.values)-1,dtype=np.int64)
    targets=frame.end_ns[origin+1]
    origin=origin[(targets>start_ns)&(targets<=end_ns)]
    if not len(origin):raise ValueError('Empty chronological split')
    # One contiguous FP32 feature matrix; window views avoid a full N*context copy.
    values=np.asarray(frame.values,dtype=np.float32)
    windows=np.lib.stride_tricks.sliding_window_view(values,context,axis=0).transpose(0,2,1)
    x=windows[origin[0]-context+1:origin[-1]-context+2]
    actual=frame.values[origin+1,0].copy()
    return Samples(x,actual.astype(np.float32),actual,origin,frame.end_ns[origin],frame.end_ns[origin+1])


def fit_scaler(frame,train):
    rows=frame.values[:int(train.origins[-1])+1]
    mean=rows.mean(axis=0);scale=rows.std(axis=0,ddof=0)
    scale[np.ptp(rows,axis=0)==0]=1.
    mean=mean.astype(np.float32);scale=scale.astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Invalid training-only scaler')
    return mean,scale


def _flat(x):
    x=np.asarray(x)
    if (x.ndim!=3 or not 1<=len(x)<=MAX_ROWS or not 1<=x.shape[1]<=1024
            or not 1<=x.shape[2]<=64 or x.shape[1]*x.shape[2]>1024
            or not np.isfinite(x).all()):raise ValueError('Invalid ridge matrix')
    return x.reshape(len(x),-1).astype(np.float64)


def fit_ridge(x,y,alpha):
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Invalid ridge penalty')
    x=_flat(x);y=np.asarray(y,dtype=np.float64)
    if y.shape!=(len(x),) or not np.isfinite(y).all():raise ValueError('Invalid ridge labels')
    mean=x.mean(axis=0);scale=x.std(axis=0);scale[np.ptp(x,axis=0)==0]=1.
    z=(x-mean)/scale;intercept=float(y.mean())
    gram=z.T@z/len(z);gram.flat[::len(gram)+1]+=alpha
    weight=np.linalg.solve(gram,z.T@(y-intercept)/len(z))
    if not np.isfinite(weight).all():raise ValueError('Invalid ridge solution')
    return dict(mean=mean.tolist(),scale=scale.tolist(),weight=weight.tolist(),intercept=intercept,alpha=alpha)


def ridge_predict(model,x):
    x=_flat(x);mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);weight=np.asarray(model['weight'])
    if (mean.shape!=(x.shape[1],) or scale.shape!=mean.shape or weight.shape!=mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(weight).all() or not (scale>0).all()
            or not math.isfinite(model['intercept'])):raise ValueError('Invalid ridge artifact')
    result=((x-mean)/scale)@weight+model['intercept']
    if not np.isfinite(result).all():raise ValueError('Nonfinite ridge prediction')
    return result


def scores(prediction,actual):
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(actual,dtype=np.float64)
    if p.ndim!=1 or p.shape!=y.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid score vectors')
    error=p-y;mse=float(np.mean(error**2))
    if not math.isfinite(mse):raise ValueError('Score overflow')
    correlation=None if np.ptp(p)==0 or np.ptp(y)==0 else float(np.corrcoef(p,y)[0,1])
    return dict(count=len(p),mse=mse,rmse=math.sqrt(mse),mae=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)),prediction_std=float(np.std(p)),actual_std=float(np.std(y)),correlation=correlation)


def loss_interval(prediction,baseline,actual,settings):
    # The primary comparison is fixed before evaluation; secondary seeds are descriptive.
    scores(prediction,actual);scores(baseline,actual)
    d=(np.asarray(prediction,dtype=np.float64)-actual)**2-(np.asarray(baseline,dtype=np.float64)-actual)**2
    block=settings['block_hours'];reps=settings['replicates'];confidence=settings['confidence']
    if (type(block) is not int or not 1<=block<=len(d) or type(reps) is not int
            or not 20<=reps<=10000 or not 0<confidence<1):raise ValueError('Invalid bootstrap settings')
    rng=np.random.default_rng(settings['seed']);means=[];offset=np.arange(block)
    for _ in range(reps):
        starts=rng.integers(0,len(d),size=(len(d)+block-1)//block)
        indices=((starts[:,None]+offset)%len(d)).ravel()[:len(d)]
        means.append(float(d[indices].mean()))
    lower,upper=np.quantile(means,[(1-confidence)/2,1-(1-confidence)/2])
    return dict(mean_difference=float(d.mean()),lower=float(lower),upper=float(upper),
                confidence=confidence,block_hours=block,replicates=reps,
                interpretation='negative favors model; approximate exploratory interval')
