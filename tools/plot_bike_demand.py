#!/usr/bin/env python3
"""Render the preregistered bike-demand example from sealed native predictions.

Optional presentation dependency: matplotlib. Never used by training or native code.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as dates
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT/'local'):raise ValueError('Use ignored local/ for figures')
    seal=json.loads((args.run/'complete.json').read_text())
    if seal.get('status')!='complete':raise ValueError('Study is incomplete')
    def read(name):
        raw=(args.run/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seal['files'][name]:raise ValueError('Artifact digest mismatch')
        return raw
    p=json.loads(read('protocol.json'));s=json.loads(read('scores.json'))
    rows=[json.loads(line) for line in read('predictions.jsonl').splitlines()]
    start,end=map(datetime.fromisoformat,p['plot_interval']);seed=str(p['uncertainty']['primary_seed'])
    rows_by_hour={datetime.fromisoformat(r['target_start_source_local']):r for r in rows}
    clocks=[start+timedelta(hours=i) for i in range(int((end-start).total_seconds()/3600))]
    def series(fn):return [fn(rows_by_hour[t]) if t in rows_by_hour else float('nan') for t in clocks]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'stealth-bike-demand-v1'})
    fig,(ax,bars)=plt.subplots(2,1,figsize=(12,8.6),gridspec_kw={'height_ratios':[1.4,1]})
    fig.subplots_adjust(left=.21,right=.96,top=.87,bottom=.14,hspace=.48)
    fig.text(.06,.965,'Can the model learn the daily rental rhythm?',fontsize=21,weight='bold',va='top')
    fig.text(.06,.921,'Past 48 hours + calendar → next-hour rentals  |  PyTorch training → Stealth C++ predictions',fontsize=11,color='#475569')
    ax.plot(clocks,series(lambda r:r['actual_rentals']),color='#0f172a',lw=1.8,label='Actual rentals')
    ax.plot(clocks,series(lambda r:r['native_rentals'][seed]),color='#2563eb',lw=1.6,label='Stealth (seed 7)')
    ax.plot(clocks,series(lambda r:r['baseline_rentals']['last_hour']),color='#d97706',lw=1,ls='--',alpha=.65,label='Repeat last hour')
    ax.set_title('First evaluation week: 1–7 July 2012 (chosen before scoring)',loc='left',pad=14,fontsize=12)
    ax.set_ylabel('Rentals per hour');ax.set_xlim(start,end)
    ax.xaxis.set_major_locator(dates.DayLocator());ax.xaxis.set_major_formatter(dates.DateFormatter('%a\n%d Jul'))
    ax.grid(axis='y',alpha=.18);ax.legend(loc='upper right',frameon=False,ncol=3,fontsize=9)
    names=['Repeat last hour','Same hour yesterday','Training calendar average','Linear model (ridge)']
    values=[s['baselines'][k]['mae'] for k in ('last_hour','same_hour_yesterday','training_weekday_hour_mean','ridge')]
    colors=['#d97706','#94a3b8','#94a3b8','#64748b']
    for key in p['seeds']:
        names.append(f'Stealth · seed {key}');values.append(s['models'][str(key)]['test']['mae'])
        colors.append('#2563eb' if str(key)==seed else '#93b4f5')
    bars.barh(names,values,color=colors,height=.65);bars.invert_yaxis()
    bars.set_title(f'All {s["test_samples"]:,} eligible evaluation hours, July–December 2012',loc='left',pad=12,fontsize=12)
    bars.set_xlabel('Average absolute error in rentals (lower is better)');bars.set_xlim(0,max(values)*1.18)
    for i,value in enumerate(values):bars.text(value+max(values)*.018,i,f'{value:.1f}',va='center',fontsize=10)
    bars.grid(axis='x',alpha=.18);bars.set_axisbelow(True)
    reduction=100*(1-s['models'][seed]['test']['mae']/s['baselines']['last_hour']['mae'])
    fig.text(.06,.072,f'Primary model: {reduction:.1f}% lower error than repeating the latest count. Negative forecasts are kept.',fontsize=10)
    fig.text(.06,.043,'Source: Hadi Fanaee-T (2013), UCI Bike Sharing, DOI 10.24432/C5W894 · CC BY 4.0. Figure: Stealth.',fontsize=9,color='#475569')
    fig.text(.06,.023,'Source-local calendar; gaps excluded, never filled. A learning demonstration; no live-service or state-of-the-art claim.',fontsize=9,color='#475569')
    args.output.mkdir(parents=True,exist_ok=False)
    fig.savefig(args.output/'forecast.png',dpi=170,facecolor='white')
    fig.savefig(args.output/'forecast.svg',facecolor='white',metadata={'Date':None,'Creator':'Stealth'})
    fig.savefig(args.output/'forecast.pdf',facecolor='white',metadata={'CreationDate':None,'ModDate':None,'Creator':'Stealth'})
    plt.close(fig)
    print(json.dumps({'output':str(args.output),'week_hours':len(clocks),'week_forecasts':sum(t in rows_by_hour for t in clocks)}))


if __name__=='__main__':main()
