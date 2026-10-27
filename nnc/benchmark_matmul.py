#!/usr/bin/env python3
"""BF16 dynamic-RHS Matmul: DMA hardware slices tiles into an 8-slot ring."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from nnc import compiler,run_smoke
from nnc.codegen import tensor_tile_bits
from nnc.benchmark_quant import revision
from nnc.test.smoke_matmul import nnedge_matmul
from nnc.test.smoke_matmul_transpose import nnedge_matmul_transpose


class Model(torch.nn.Module):
    def __init__(self,transpose,heads):
        super().__init__(); self.transpose=transpose; self.heads=heads
    def forward(self,x,rhs):
        return (nnedge_matmul_transpose if self.transpose else nnedge_matmul)(x,rhs,self.heads)


def expected(model,x,rhs):
    with torch.no_grad(): y=model(x,rhs).to(torch.bfloat16)
    bits=tuple(int(b)&65535 for b in y.contiguous().view(torch.int16).flatten().tolist())
    return tensor_tile_bits(bits,tuple(y.shape))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--sim-exe',type=Path,required=True)
    p.add_argument('--edge32-include',type=Path,required=True)
    p.add_argument('--e3-include',type=Path,required=True)
    p.add_argument('--capacity',type=int,action='append',choices=(32,64,128))
    p.add_argument('--tokens',type=int,default=128)
    p.add_argument('--heads',type=int,default=1)
    p.add_argument('--k',type=int,default=64)
    p.add_argument('--n',type=int,default=64)
    a=p.parse_args();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(nnc=revision(ROOT),edge32=revision(a.edge32_include.resolve().parent),
                    e3=revision(a.e3_include.resolve().parent),
                    asic=revision(a.e3_include.resolve().parent.parent/'edge-asic'),command=sys.argv,
                    compiler=subprocess.check_output([run_smoke.tool('CLANGXX','clang++'),'--version'],text=True),
                    sim_sha256=hashlib.sha256(a.sim_exe.read_bytes()).hexdigest(),
                    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in [*sorted((ROOT/'cpp/libnn').glob('*.hpp')),
                                       *sorted((ROOT/'nnc').glob('*.py')),
                                       ROOT/'nnc/compiler.py',ROOT/'nnc/codegen.py',ROOT/'nnc/lowering.py',ROOT/'nnc/layout.py',
                                       Path(__file__).resolve(),ROOT/'nnc/test/matmul_benchmark_main.cpp',
                                       ROOT/'nnc/test/matmul_mutation_main.cpp']},
                    flags='RV32IM/Zba ilp32 medany C++17 -O2',ring_slots=8,tile_bytes=128,
                    rhs_layout='original contiguous BF16 runtime matrix; never packed/quantized',
                    warmup='none; mutation checks use a separate two-invocation run')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    records=[]
    rows,k,n,heads=a.tokens,a.k,a.n,a.heads
    for transpose in (False,True):
        name='matmul_transpose' if transpose else 'matmul'
        model=Model(transpose,heads)
        shape=(rows,heads*k) if transpose else (heads,rows,k)
        x=((torch.arange(torch.tensor(shape).prod().item()).reshape(shape)%13-6).float()/8).to(torch.bfloat16)
        rhs_shape=(n,heads*k) if transpose else (k,heads*n)
        rhs=((torch.arange(rhs_shape[0]*rhs_shape[1]).reshape(rhs_shape)%11-5).float()/16).to(torch.bfloat16)
        for capacity in a.capacity or (32,64,128):
            case=out/f'{name}-{capacity}';generated=case/'generated'
            compiler.compile(model,(x,rhs),out_dir=generated,save_reference=True,
                             matmul_token_capacity=capacity)
            # No static weights; RHS must remain byte-for-byte contiguous
            # in input.bin at the compiler's aligned second-input offset.
            payload=(generated/'input.bin').read_bytes()
            rhs_bytes=rhs.contiguous().view(torch.uint16).numpy().tobytes()
            rhs_offset=(x.numel()*2+63)//64*64
            if payload[rhs_offset:rhs_offset+len(rhs_bytes)] != rhs_bytes or (generated/'weight.bin').read_bytes():
                raise RuntimeError('matmul RHS was unexpectedly packed or treated as static weights')
            cfg=argparse.Namespace(profile=False,phase_profile=False,
                main_file=ROOT/'nnc/test/matmul_benchmark_main.cpp',
                edge32_include=a.edge32_include.resolve(),e3_include=a.e3_include.resolve(),
                output_base=0x100000,sim_exe=a.sim_exe.resolve(),sim_build_script=None,
                skip_verilator_build=True,max_cycles=5000000,max_abs_error=0,
                sim_arg=['+profile_mac','+trace_cmd'])
            run_smoke.build(cfg,generated,case);run_smoke.run(cfg,case,expected(model,x,rhs))
            log=(case/'software-console.log').read_text()
            profile,=re.findall(r'^NNC_MATMUL_PROFILE (.+)$',log,re.M)
            mac,=re.findall(r'^MAC PROFILE (.+)$',log,re.M)
            counts={key:int(v) for key,v in re.findall(r'(\w+)=(\d+)',profile)}
            counters={key:int(v) for key,v in re.findall(r'(\w+)=(\d+)',mac)}
            chunks=heads*((rows+capacity-1)//capacity)
            ideal=heads*rows*k*n//64
            if counts['capacity']!=min(rows,capacity) or counts['chunks']!=chunks or counts['ring_slots']!=8:
                raise RuntimeError(f'unexpected token/ring partition {counts}')
            if counters['issue']!=ideal or counters['starts']!=chunks*(k//8)*(n//8):
                raise RuntimeError(f'MAC work/COPY accounting failure {counters}')
            # Confirm each hardware circular command exposes eight slots;
            # ordinary/strided I/O commands are separate from RHS slicing.
            rings=[];copies=0;config={}
            for word,value in re.findall(r'lite accel inst=([0-9a-f]+) capture=([0-9a-f]+)',log):
                w=int(word,16);op=w>>25;mode=((w>>20)&31)*8+((w>>12)&7)
                v=int(value,16)
                if op==3:config['fragment_bytes']=v
                if op==8:config['entry_bytes']=v
                if op==10:config[['x_stride','x_count','y_stride','y_count'][mode]]=v
                if op==1 and mode==3:
                    rings.append(v)
                    want=dict(fragment_bytes=16,entry_bytes=128,x_stride=rhs.shape[1]*2,
                              x_count=n if transpose else k,y_stride=16,
                              y_count=k//8 if transpose else n//8)
                    if config != want:raise RuntimeError(f'DMA hardware XY config {config} != {want}')
                if op==0x15 and mode&32:copies+=1
            if rings != [8]*chunks or copies:
                raise RuntimeError(f'RHS is not hardware ring8 slicing: {rings}, copies={copies}')
            records.append(dict(op=name,requested_capacity=capacity,rows=rows,k=k,n=n,heads=heads,
                                ideal_cycles=ideal,stream_util_percent=ideal*100/counts['stream_cycles'],
                                op_util_percent=ideal*100/counts['op_cycles'],**counts))
            (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
            if capacity==128:
                mutation=case/'rhs-mutation'
                cfg.main_file=ROOT/'nnc/test/matmul_mutation_main.cpp'
                cfg.sim_arg=['+profile_mac']
                run_smoke.build(cfg,generated,mutation)
                changed=rhs.clone();changed.view(torch.int16).flatten()[0]^=-32768
                run_smoke.run(cfg,mutation,expected(model,x,changed))
    with (out/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# BF16 runtime-RHS Matmul, hardware ring8','',
           'RHS stays contiguous in DRAM. DMA XY parameters slice 8x8 tiles; no CPU payload slicing.',
           'NNC plans tiled SRAM IO by default; ordinary boundary DMA lies outside operator timing.','',
           '| Op | Capacity | Stream cycles | Stream utilization | Operator cycles | Operator utilization |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append(f'| {r["op"]} | {r["capacity"]} | {r["stream_cycles"]} | '
                     f'{r["stream_util_percent"]:.2f}% | {r["op_cycles"]} | {r["op_util_percent"]:.2f}% |')
    (out/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()
