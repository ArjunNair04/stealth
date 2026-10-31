"""Summarize optional macOS samples and audit a fixed native A/B experiment.

Sampling is attribution, never latency evidence. Count the outermost matching
primitive frame; softmax remains a subset of attention, not an extra bucket.
"""
import itertools
import math
import re
import statistics


def summarize_profile(text):
    if 'Call graph:\n' not in text or '\nTotal number in stack' not in text:
        raise ValueError('Unrecognised sample report')
    graph=text.split('Call graph:\n',1)[1].split('\nTotal number in stack',1)[0]
    threads=re.findall(r'^\s*(\d+) Thread_',graph,re.M)
    if len(threads)!=1 or int(threads[0])<=0:
        raise ValueError('Expected one sampled inference thread')
    total=int(threads[0])
    needles=dict(dense=('Dense::forward(', 'Dense::forward_rows('),
                 attention_including_softmax=('::causal_attention(',),
                 softmax_subset=('::softmax(',),gelu=('::gelu(',),
                 layer_norm=('::layer_norm(', '::layer_norm_rows('),residual_add=('::residual_add(',))
    counts={k:0 for k in needles};ancestors=[]
    for line in graph.splitlines():
        match=re.match(r'^([ +!:|]*)(\d+) (.+?)  \(in ',line)
        if not match: continue
        depth=len(match[1])
        while ancestors and ancestors[-1][0]>=depth: ancestors.pop()
        matched={key for key,names in needles.items() if any(name in match[3] for name in names)}
        for key in matched:
            if not any(key in keys for _,keys in ancestors): counts[key]+=int(match[2])
        ancestors.append((depth,matched))
    primary=sum(v for k,v in counts.items() if k!='softmax_subset')
    if primary>total or primary==0 or counts['softmax_subset']>counts['attention_including_softmax']:
        raise ValueError('Inconsistent or recursive primitive attribution')
    counts['other']=total-primary
    return dict(samples=total,counts=counts,percent={k:100*v/total for k,v in counts.items()})


def accept_comparison(records):
    expected=set(itertools.product((7,19,43),(1,2,3),(1,8),('baseline','candidate')))
    keys=[(r['seed'],r['trial'],r['batch'],r['implementation']) for r in records]
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Incomplete or duplicated comparison cells')
    by_key=dict(zip(keys,records))
    reductions=[];tails=[];per_seed={}
    for seed in (7,19,43):
        changes=[]
        for trial in (1,2,3):
            baseline=by_key[seed,trial,1,'baseline']['latency']
            candidate=by_key[seed,trial,1,'candidate']['latency']
            for stats in (baseline,candidate):
                if any(not math.isfinite(stats[k]) or stats[k]<=0 for k in ('p50_ns','p99_ns')):
                    raise ValueError('Invalid comparison timings')
            changes.append(1-candidate['p50_ns']/baseline['p50_ns'])
            tails.append(candidate['p99_ns']/baseline['p99_ns']-1)
        reductions.extend(changes);per_seed[str(seed)]=statistics.median(changes)
    reduction=statistics.median(reductions);tail=statistics.median(tails)
    return dict(accepted=reduction>=.05 and all(v>0 for v in per_seed.values()) and tail<=.1,
                median_p50_reduction=reduction,per_seed_median_reduction=per_seed,
                median_p99_increase=tail)
