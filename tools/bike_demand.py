"""Pinned UCI hourly counts, causal calendar features and gap-aware windows."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import csv
import hashlib
import io
import math
import numpy as np
# Reuse tested numerical helpers without changing the frozen BTC experiment.
from btc_returns import scores, fit_ridge, ridge_predict

HEADER=('instant','dteday','season','yr','mnth','hr','holiday','weekday','workingday',
        'weathersit','temp','atemp','hum','windspeed','casual','registered','cnt')
FEATURES=('rentals','next_hour_sin','next_hour_cos','next_weekday_sin','next_weekday_cos')
EPOCH=datetime(2011,1,1)
MAX_BYTES=4*1024*1024
MAX_HOURS=24*731


def hour(value):
    try:d=datetime.strptime(value,'%Y-%m-%dT%H:%M:%S')
    except (ValueError,TypeError) as exc:raise ValueError('Expected naive source-calendar hour') from exc
    if d.strftime('%Y-%m-%dT%H:%M:%S')!=value or d.minute or d.second:
        raise ValueError('Expected exact source-calendar hour')
    h=int((d-EPOCH).total_seconds())//3600
    if not 0<=h<=MAX_HOURS:raise ValueError('Hour outside dataset calendar')
    return h


def label(h):return (EPOCH+timedelta(hours=int(h))).isoformat()


def weekday(h):return ((np.asarray(h)//24)+EPOCH.weekday())%7


@dataclass(frozen=True)
class Frame:
    hours: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        h=np.array(self.hours,copy=True);v=np.array(self.values,dtype=np.float64,copy=True)
        if (h.dtype!=np.int64 or h.ndim!=1 or not 2<=len(h)<=MAX_HOURS
                or v.shape!=(len(h),5) or not np.isfinite(v).all()
                or (h<0).any() or (h>=MAX_HOURS).any() or not (np.diff(h)>0).all()
                or (v[:,0]<0).any() or (v[:,0]>1e6).any() or (np.abs(v[:,1:])>1.00000001).any()):
            raise ValueError('Invalid bike hourly feature frame')
        h.flags.writeable=False;v.flags.writeable=False
        object.__setattr__(self,'hours',h);object.__setattr__(self,'values',v)


def parse_csv(raw,expected_sha=None):
    if not isinstance(raw,bytes) or not 1<=len(raw)<=MAX_BYTES:raise ValueError('Invalid source size')
    if expected_sha is not None and hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Source SHA256 mismatch')
    try:
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8'),newline=''),strict=True)
        if tuple(reader.fieldnames or ())!=HEADER:raise ValueError('Unexpected UCI hourly schema')
        hours=[];values=[]
        integer_fields=HEADER[:10]+HEADER[14:]
        integer_fields=tuple(k for k in integer_fields if k!='dteday')
        for index,row in enumerate(reader,1):
            if index>MAX_HOURS or set(row)!=set(HEADER) or any(v is None for v in row.values()):
                raise ValueError('Invalid UCI row shape')
            integers={k:int(row[k]) for k in integer_fields}
            if any(str(v)!=row[k] for k,v in integers.items()):raise ValueError('Noncanonical integer')
            r=integers;h=hour(row['dteday']+'T00:00:00')+r['hr'];d=EPOCH+timedelta(hours=h)
            weather=[float(row[k]) for k in ('temp','atemp','hum','windspeed')]
            if (r['instant']!=index or not 0<=r['hr']<24 or r['yr']!=d.year-2011
                    or r['mnth']!=d.month or r['weekday']!=(d.weekday()+1)%7
                    or r['holiday'] not in (0,1) or r['workingday']!=int(d.weekday()<5 and not r['holiday'])
                    or not 1<=r['season']<=4 or not 1<=r['weathersit']<=4
                    or not all(math.isfinite(v) and 0<=v<=1 for v in weather)
                    or not all(0<=r[k]<=100000 for k in ('casual','registered','cnt'))
                    or r['cnt']!=r['casual']+r['registered']):
                raise ValueError('Invalid UCI calendar, count or weather row')
            # Future calendar is known from the clock; no future observed row is read.
            nh=h+1;phase=2*math.pi*(nh%24)/24;week=2*math.pi*int(weekday(nh))/7
            hours.append(h);values.append((r['cnt'],math.sin(phase),math.cos(phase),math.sin(week),math.cos(week)))
        return Frame(np.asarray(hours,dtype=np.int64),np.asarray(values))
    except (UnicodeError,csv.Error,TypeError,KeyError,OverflowError) as exc:
        raise ValueError('Malformed UCI hourly source') from exc


@dataclass(frozen=True)
class Samples:
    x: np.ndarray
    y: np.ndarray
    actual: np.ndarray
    origins: np.ndarray
    input_end_hours: np.ndarray
    target_hours: np.ndarray
    excluded_hours: int


def split(frame,context,start,end):
    if (type(context) is not int or not 1<=context<=1024 or type(start) is not int
            or type(end) is not int or not 0<=start<end<=MAX_HOURS):
        raise ValueError('Invalid split or context')
    origins=np.arange(context-1,len(frame.hours)-1,dtype=np.int64)
    target=frame.hours[origins+1]
    # Strictly increasing integer hours: this span equality implies every step is one hour.
    contiguous=target-frame.hours[origins-context+1]==context
    origins=origins[contiguous&(target>=start)&(target<end)]
    if not len(origins):raise ValueError('No complete windows in split')
    windows=np.lib.stride_tricks.sliding_window_view(frame.values.astype(np.float32),context,axis=0).transpose(0,2,1)
    x=windows[origins-context+1];actual=frame.values[origins+1,0].copy();y=actual.astype(np.float32)
    for a in (x,y,actual,origins):a.flags.writeable=False
    return Samples(x,y,actual,origins,frame.hours[origins]+1,frame.hours[origins+1],end-start-len(origins))


def fit_scaler(frame,train):
    mask=np.zeros(len(frame.hours),dtype=bool);context=train.x.shape[1]
    for origin in train.origins:mask[origin-context+1:origin+1]=True
    rows=frame.values[mask];mean=rows.mean(axis=0);scale=rows.std(axis=0)
    scale[np.ptp(rows,axis=0)==0]=1
    mean=mean.astype(np.float32);scale=scale.astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale>0).all():
        raise ValueError('Invalid training-only scaler')
    return mean,scale


def calendar_index(hours):return (weekday(hours)*24+np.asarray(hours)%24).astype(np.int64)


def loss_interval(prediction,baseline,actual,hours,settings):
    scores(prediction,actual);scores(baseline,actual)
    d=np.abs(np.asarray(prediction,dtype=np.float64)-actual)-np.abs(np.asarray(baseline,dtype=np.float64)-actual)
    hours=np.asarray(hours)
    if hours.shape!=d.shape or hours.dtype!=np.int64 or not (np.diff(hours)>0).all():
        raise ValueError('Invalid bootstrap hours')
    reps=settings['replicates'];confidence=settings['confidence']
    if type(reps) is not int or not 20<=reps<=10000 or not 0<confidence<1:raise ValueError('Invalid bootstrap settings')
    _,group=np.unique(hours//24,return_inverse=True)
    totals=np.bincount(group,weights=d);counts=np.bincount(group)
    if len(counts)<2:raise ValueError('Bootstrap requires at least two days')
    rng=np.random.default_rng(settings['seed']);means=[]
    for _ in range(reps):
        chosen=rng.integers(0,len(counts),size=len(counts))
        means.append(float(totals[chosen].sum()/counts[chosen].sum()))
    lo,hi=np.quantile(means,[(1-confidence)/2,1-(1-confidence)/2])
    return dict(mean_difference=float(d.mean()),lower=float(lo),upper=float(hi),confidence=confidence,
                days=len(counts),replicates=reps,interpretation='Negative favors model; exploratory day-cluster interval')
