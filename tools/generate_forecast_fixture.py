"""Independent full-forecaster equations and portable artifact, diagnostic weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import generate_attention_fixture as a

F, W, H, D, N, HORIZON, TARGET = 7, 4, 2, 7, 5, 4, 6
ORDER = ('mean','scale','input_weight','input_bias','attn_scale','attn_shift',
         'qkv_weights','qkv_bias','projection_weights','projection_bias','ff_scale','ff_shift',
         'up_weights','up_bias','down_weights','down_bias','final_scale','final_shift','head_weight','head_bias')


def evaluate(p, raw, width, heads, target):
    positions = [[a.fp32((math.sin if c%2==0 else math.cos)(t/10000**(2*(c//2)/width)))
                  for c in range(width)] for t in range(len(raw))]
    normalized = [[(v-m)/s for v,m,s in zip(row,p['mean'],p['scale'])] for row in raw]
    embedded = [[v+pos for v,pos in zip(a.dense(row,p['input_weight'],p['input_bias']),pe)]
                for row,pe in zip(normalized,positions)]
    qkv = [a.dense(a.norm(row,p['attn_scale'],p['attn_shift']),p['qkv_weights'],p['qkv_bias']) for row in embedded]
    q,k,v = ([row[i*width:(i+1)*width] for row in qkv] for i in range(3))
    attention = a.attention(q,k,v,heads)
    residual = [[x+y for x,y in zip(row,a.dense(ctx,p['projection_weights'],p['projection_bias']))]
                for row,ctx in zip(embedded,attention)]
    block = []
    for row in residual:
        hidden = a.dense(a.norm(row,p['ff_scale'],p['ff_shift']),p['up_weights'],p['up_bias'])
        activated = [.5*x*math.erfc(-x/math.sqrt(2)) for x in hidden]
        block.append([x+y for x,y in zip(row,a.dense(activated,p['down_weights'],p['down_bias']))])
    expected = [a.dense(a.norm(row,p['final_scale'],p['final_shift']),p['head_weight'],p['head_bias'])[0]
                *p['scale'][target]+p['mean'][target] for row in block]
    return expected,a.flatten(positions)


def fixture():
    p = {'mean': [float(j-3) for j in range(F)], 'scale': [float(j+1) for j in range(F)],
         'input_weight': a.pattern(W*F,3), 'input_bias': a.pattern(W,5,64),
         'attn_scale':[1.,.75,1.25,-.5], 'attn_shift':a.pattern(W,1,64),
         'qkv_weights':a.pattern(3*W*W,7), 'qkv_bias':a.pattern(3*W,3,64),
         'projection_weights':a.pattern(W*W,11), 'projection_bias':a.pattern(W,13,64),
         'ff_scale':[.5,1.,-.75,1.25], 'ff_shift':a.pattern(W,5,64),
         'up_weights':a.pattern(D*W,17), 'up_bias':a.pattern(D,19,64),
         'down_weights':a.pattern(W*D,23), 'down_bias':a.pattern(W,2,64),
         'final_scale':[1.,.5,.75,1.25], 'final_shift':a.pattern(W,9,64),
         'head_weight':a.pattern(W,4), 'head_bias':[.125]}
    raw = a.rows(a.pattern(N*F,17,8),F)
    expected,positions = evaluate(p,raw,W,H,TARGET)
    parameters = [v for name in ORDER for v in p[name]]
    header = b'STLFC01\0' + struct.pack('<10IfIII',1,0x01020304,64,F,W,H,D,N,HORIZON,TARGET,a.EPS,len(parameters),0,0)
    assert len(header)==64
    payload = header + struct.pack('<'+str(len(parameters))+'f',*parameters)
    artifact = payload + hashlib.sha256(payload).digest()
    return parameters,a.flatten(raw),expected,positions,artifact


def full_case():
    features,width,hidden,tokens,heads,target=7,64,256,96,4,6
    sizes=(features,features,width*features,width,width,width,3*width*width,3*width,
           width*width,width,width,width,hidden*width,hidden,width*hidden,width,width,width,width,1)
    p={}
    for index,(name,size) in enumerate(zip(ORDER,sizes)):
        if index==0: values=[float(i)-3 for i in range(size)]
        elif index==1: values=[float(i)+1 for i in range(size)]
        else:
            values=a.pattern(size,index*7+3,128)
            if index in (4,10,16): values=[1+x for x in values]
        p[name]=values
    raw=a.rows(a.pattern(tokens*features,17,8),features)
    return evaluate(p,raw,width,heads,target)[0]


def generate():
    parameters,inputs,expected,positions,artifact = fixture()
    lines = ['// Independent double equations, untrained diagnostic fixture.', '#pragma once',
             '#include <array>', '#include <cstddef>', 'namespace forecast_fixture {',
             f'inline constexpr std::size_t features={F}, width={W}, heads={H}, hidden={D}, tokens={N}, horizon={HORIZON}, target={TARGET};']
    for name,values in (('parameters',parameters),('input',inputs),('expected',expected),('positions',positions)):
        lines.append(f'inline constexpr std::array<float,{len(values)}> {name}'+'{'+','.join(format(v,'.10e')+'F' for v in values)+'};')
    lines.append(f'inline constexpr std::array<unsigned char,{len(artifact)}> artifact'+'{'+','.join(str(b) for b in artifact)+'};')
    # Exercise SHA padding boundaries and multi-block processing against hashlib.
    lengths = (0,1,3,55,56,63,64,65,119,120,127,128,129,1024,65535,65536,1048576)
    lines.append('inline constexpr std::array<std::size_t,17> hash_lengths{'+','.join(map(str,lengths))+'};')
    lines.append('inline constexpr std::array<std::array<unsigned char,32>,17> hashes{{'+','.join(
        '{'+','.join(str(v) for v in hashlib.sha256(bytes((i*37+11)%256 for i in range(n))).digest())+'}' for n in lengths)+'}};')
    full=full_case()
    lines.append('inline constexpr std::array<float,96> full_expected{'+','.join(format(v,'.10e')+'F' for v in full)+'};')
    lines.append('} // namespace forecast_fixture')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--artifact',type=Path,help='Optional new local diagnostic model file')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]; path=root/'tests/fixtures/forecast.hpp'
    content=generate()
    metadata={'schema_version':1,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'equations_dependency_sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
              'fixture_sha256':hashlib.sha256(content.encode()).hexdigest(),
              'artifact_sha256':hashlib.sha256(fixture()[-1]).hexdigest(),
              'reference':'Independent Python double equations plus hashlib; not PyTorch; no native outputs',
              'purpose':'Untrained diagnostic weights only; compact and full 96x7 width-64 task shapes', 'tolerance':{'absolute':5e-5,'relative':2e-4}}
    if args.check:
        if path.read_text()!=content or json.loads(path.with_suffix('.json').read_text())!=metadata:
            raise SystemExit('Forecast fixture/provenance mismatch: investigate before changing expected answers')
        print('Forecast fixture and provenance reproduced exactly')
    else:
        path.write_text(content); path.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.artifact:
        with args.artifact.open('xb') as stream: stream.write(fixture()[-1])


if __name__=='__main__': main()
