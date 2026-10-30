"""Plot the published sizing evidence without training dependencies."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='bike-size-v1'

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
training=json.loads((args.evidence/'training.json').read_bytes())
latency=json.loads((args.evidence/'latency.json').read_bytes())
fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
colors=['#236B8E','#B76B22','#A73535']
for width,color in zip(('64','128','256'),colors):
 row=training['widths'][width]; timing=latency['widths'][width]
 maes=[r['test']['mae'] for r in row['models'].values()]
 x=timing['p50_us']; y=statistics.median(maes)
 ax.errorbar(x,y,xerr=[[x-timing['p50_range_us'][0]],[timing['p50_range_us'][1]-x]],
             yerr=[[y-min(maes)],[max(maes)-y]],fmt='o' if row['accepted'] else 'X',capsize=5,color=color,markersize=8,
             label=f"{row['parameters']['trainable']:,} parameters" + (" — check failed" if not row['accepted'] else ""))
 ax.annotate(f'{y:.2f} rentals\n{x:.1f} µs',(x,y),xytext=(8,10),textcoords='offset points',fontsize=10,color=color)
ax.set_xlabel('Native latest forecast: median of worker p50 values (µs)')
ax.set_ylabel('Evaluation MAE: median across three seeds (rentals)')
ax.set_title('Does more model capacity earn its inference cost?',loc='left',fontsize=14,pad=16)
ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
ax.margins(y=.25)
ax.set_xlim(left=0,right=630)
ax.legend(loc='best',frameon=False)
fig.text(.01,-.16,'Lower and further left is better. Bars show observed ranges, not confidence intervals.\n'
         'Bike rentals · 48-hour history · one block · reused evaluation period · local Intel CPU.\n'
         'Red cross: diagnostic only; the original numerical check remains failed.',fontsize=9,color='#555555')
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ('.svg','.png'):
 fig.savefig(args.output.with_suffix(extension),dpi=160,bbox_inches='tight',metadata={'Creator':'Stealth bike sizing study','Date':None} if extension=='.svg' else None)

# Matplotlib leaves harmless trailing spaces in SVG paths; normalize for Git checks.
svg=args.output.with_suffix('.svg')
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
