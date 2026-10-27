#!/usr/bin/env python3
"""Run the sole Matmul paths and all named quant Linear APIs, then combine evidence."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
SPECS=('nvfp4','mxfp4','q8_0','fp4','fp8_e4m3fn','fp8_e5m2')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('out-dir','sim-exe','edge32-include','e3-include'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    common=['--sim-exe',str(a.sim_exe.resolve()),'--edge32-include',str(a.edge32_include.resolve()),
            '--e3-include',str(a.e3_include.resolve())]
    jobs=[]
    for kind in ('matmul','quant'):
        command=[sys.executable,str(ROOT/f'nnc/benchmark_{kind}.py'),
                 '--out-dir',str(out/kind),*common]
        if kind=='quant':
            for spec in SPECS: command.extend(('--spec',spec))
        log=(out/f'{kind}.log').open('w')
        jobs.append((kind,log,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)))
    failures=[]
    for kind,log,job in jobs:
        code=job.wait();log.close()
        if code: failures.append(kind)
    if failures: raise SystemExit(f"benchmark failed: {failures}; inspect {out}/*.log")
    write_report(out)

def write_report(out):
    records=[]
    for r in json.loads((out/'matmul/results.json').read_text()):
        records.append(dict(operator=r['op'],capacity=r['capacity'],chunks=r['chunks'],
                            stream_cycles=r['stream_cycles'],stream_util_percent=r['stream_util_percent'],
                            operator_cycles=r['op_cycles'],operator_util_percent=r['op_util_percent']))
    for r in json.loads((out/'quant/results.json').read_text()):
        records.append(dict(operator='linear_'+r['spec'],capacity=r['capacity'],chunks=r['chunks'],
                            stream_cycles=r['stream_cycles'],stream_util_percent=r['stream_util_percent'],
                            operator_cycles=r['op_cycles'],operator_util_percent=r['op_util_percent']))
    assert len(records)==24
    (out/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    with (out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    header=['| Operator | Capacity | Chunks | Stream cycles | Stream MAC utilization | Operator cycles | Operator MAC utilization |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def table(rows):
        return header+[f"| {r['operator']} | {r['capacity']} | {r['chunks']} | {r['stream_cycles']:,} | {r['stream_util_percent']:.2f}% | {r['operator_cycles']:,} | {r['operator_util_percent']:.2f}% |" for r in rows]
    lines=['# Tensor operators: 64×64 weights, 128 tokens', '',
           'All 24 capacity measurements pass exact BF16 output comparison. Ideal: 524,288 MACs / 64 MACs per cycle = 8,192 cycles. No warmup.', '',
           'Input/output reside in tensor SRAM order. Stream cycles include weight DMA submission through final Tensor/DMA sync. Operator cycles include setup and validation, excluding ordinary boundary input/output DMA.', '',
           'Matmul has one implementation and one storage contract: full row-major BF16 RHS in DRAM; hardware strided circular DMA slices 8×8 tiles into ring8 at runtime. Matmul-transpose uses the corresponding DMA order and WLD orientation. NNC generates the SRAM operand storage by default, with no layout-mode switch. Quant Linear uses NNC-quantized, compile-time tiled weights.', '',
           '## Capacity 128', '']+table([r for r in records if r['capacity']==128])
    lines+=['', '## Token capacity sweep (same operators)', '']+table(records)
    lines+=version_lines(out)
    lines+=['', '## Evidence', '',
            '- [Matmul](matmul/results.md), [source/tool/simulator provenance](matmul/provenance.json).',
            '- [Quant Linear](quant/results.md), [source/tool/simulator provenance](quant/provenance.json).',
            '- Raw logs, generated payloads/headers, ELF and exact comparisons are retained in each case directory.',
            '- Matmul verifies exact RHS DMA descriptors, no Tensor COPY and runtime RHS mutation. Quant capacity-128 cases also verify generated forward, SRAM guards and allocator-pressure fallback.',
            '- Different formats use their own arithmetic goldens. Quant formats share the same source linear weights/input; Matmul uses BF16 runtime matrices.', '']
    (out/'results.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:lines.index('## Token capacity sweep (same operators)')]))
    print(out/'results.md')


def version_lines(out):
    provenance={kind:json.loads((out/kind/'provenance.json').read_text())
                for kind in ('matmul','quant')}
    q=provenance['quant']
    asic=q.get('asic')
    if asic is None:
        # Older measurements recorded the full workspace commit. Recover its
        # exact submodule pin instead of labelling today's HEAD as measured.
        workspace=q['simulator_workspace']['head']
        head=subprocess.check_output(['git','-C',str(ROOT.parent.parent),
                                      'rev-parse',workspace+':src/edge-asic'],text=True).strip()
        asic={'head':head,'status':None}
    lines=['', '## Measured revisions', '',
           '| Repository | Commit | Measurement state |', '| --- | --- | --- |']
    for label,key in [('NNC','nnc'),('edge-e3','e3'),('edge-asic','asic'),('Edge-32','edge32')]:
        version=asic if key=='asic' else q[key]
        state='workspace submodule pin' if version.get('status') is None else ('dirty' if version['status'] else 'clean')
        lines.append(f"| {label} | `{version['head']}` | {state} |")
    note = ('NNC was measured with uncommitted changes on the listed base commit; per-file source SHA-256 hashes and dirty state are retained in provenance.'
            if q['nnc']['status'] else 'Revisions and per-file source SHA-256 hashes were recorded before the build.')
    lines+=['', note, '',
            'LLVM 22.1.8, RV32IM/Zba, `-O2`, E3 wide H4, ring8, 128 KiB SRAM.', '',
            'Simulator SHA-256: `'+q['sim_sha256']+'`.', '']
    return lines

if __name__=='__main__':main()
