#!/usr/bin/env python3
"""Benchmark public quant Linear: BF16 SRAM I/O, packed weights in DRAM.

The stream window matches the tiled circular Matmul benchmark: first weight
DMA submission through tensor.sync + dma.sync. NNC preplans tensor IO storage;
ordinary whole-tensor DRAM/SRAM DMA transfers are measured separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nnc import compiler, run_smoke


def revision(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    return {"head": git("rev-parse", "HEAD"), "status": git("status", "--short")}


def fixture(spec):
    import torch
    import nnc
    # Same nontrivial source weights and inputs for all formats; the golden
    # output uses each format's native quantization and hardware arithmetic.
    linear = torch.nn.Linear(64, 64, bias=False)
    with torch.no_grad():
        values = ((torch.arange(64*64).reshape(64,64)*7)%41-20).float()/16
        values *= torch.arange(1,65).reshape(64,1)/32
        values[0,:32] = 0
        linear.weight.copy_(values)
    cls = {"mxfp4": nnc.LinearMxFp4, "nvfp4": nnc.LinearNvFp4,
           "q8_0": nnc.LinearQ8_0}[spec]
    x = ((torch.arange(128*64).reshape(128,64)%13-6).float()/32).to(torch.bfloat16)
    return cls(linear).eval(), (x,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sim-exe", type=Path, required=True)
    parser.add_argument("--edge32-include", type=Path, required=True)
    parser.add_argument("--e3-include", type=Path, required=True)
    parser.add_argument("--spec", action="append", choices=("nvfp4", "mxfp4", "q8_0"))
    parser.add_argument("--capacity", action="append", type=int, choices=(32,64,128))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sim = args.sim_exe.resolve()
    provenance = {"nnc": revision(ROOT),
                  "edge32": revision(args.edge32_include.resolve().parent),
                  "e3": revision(args.e3_include.resolve().parent),
                  "sim_exe": str(sim), "sim_sha256": hashlib.sha256(sim.read_bytes()).hexdigest(),
                  "python": sys.version, "command": sys.argv,
                  "simulator_workspace": revision(sim.parent),
                  "compile_flags": "riscv32-unknown-elf rv32im_zba ilp32 medany C++17 -O2 freestanding",
                  "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [*sorted((ROOT/"cpp/libnn").glob("*.hpp")),
                                              Path(__file__).resolve(), ROOT/"nnc/run_smoke.py",
                                              ROOT/"nnc/compiler.py", ROOT/"nnc/codegen.py",
                                              ROOT/"nnc/test/quant_benchmark_main.cpp",
                                              ROOT/"nnc/test/quant_capacity_main.cpp",
                                              ROOT/"nnc/test/quant_forward_main.cpp"]},
                  "clang": subprocess.check_output([run_smoke.tool("CLANGXX", "clang++"), "--version"], text=True),
                  "shape": {"tokens": 128, "k": 64, "n": 64},
                  "memory": "NNC-planned tensor-tile BF16 IO in SRAM/DRAM; packed DRAM weights",
                  "warmup": "none; first invocation, ordinary DMA load before kernel timing",
                  "stream_window": "weight DMA submission through tensor.sync + dma.sync; per-chunk windows summed"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    records = []
    for spec in args.spec or ("nvfp4", "mxfp4", "q8_0"):
        model, inputs = fixture(spec)
        generated = out / spec / "generated"
        compiler.compile(model, inputs, out_dir=generated, save_reference=True,
                         tensor_tile_io=True,quant_token_capacity=128)
        provenance.setdefault("artifacts",{})[spec] = {
            name: hashlib.sha256((generated/name).read_bytes()).hexdigest()
            for name in ("input.bin","weight.bin","expected.bin","io_layout.json")}
        (out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
        program, _ = compiler.compile_model(model, inputs)
        node, = program.nodes
        packed = node.attrs["packed"]
        global_arg = ", " + node.attrs["global_bits"] if spec == "nvfp4" else ""
        expected_raw = (generated / "expected.bin").read_bytes()
        import struct
        expected = struct.unpack(f"<{len(expected_raw)//2}H", expected_raw)
        for capacity in args.capacity or (32,64,128):
            build = out / spec / str(capacity)
            build.mkdir(parents=True, exist_ok=True)
            case_generated = build / "generated"
            shutil.copytree(generated, case_generated, dirs_exist_ok=True)
            # The selected entry point is the real linear_quant.hpp API.
            (case_generated / "bench_config.hpp").write_text(
                '#pragma once\n#include "init.hpp"\n'
                f'using BenchmarkPack = nnedge::{dict(mxfp4="MXFP4WeightPack", nvfp4="NVFP4WeightPack", q8_0="Q8_0WeightPack")[spec]};\n'
                'template <typename View>\ninline bool benchmark_linear(View y, View x) {\n'
                f'    return nnedge::op::linear_{spec}<{capacity}>(y, x, '
                f'model::mio::ctx.{packed}{global_arg});\n}}\n')
            build_args = argparse.Namespace(
                profile=False, phase_profile=False,
                main_file=ROOT / "nnc/test/quant_benchmark_main.cpp",
                edge32_include=args.edge32_include.resolve(), e3_include=args.e3_include.resolve(),
                sim_build_script=None, skip_verilator_build=True, sim_exe=sim,
                output_base=0x100000, max_cycles=5000000, max_abs_error=0,
                sim_arg=["+profile_mac"])
            run_smoke.build(build_args, case_generated, build)
            run_smoke.run(build_args, build, expected)
            console = (build / "software-console.log").read_text()
            profiles = re.findall(r"^NNC_QUANT_PROFILE (.+)$", console, re.M)
            mac_profiles = re.findall(r"^MAC PROFILE (.+)$", console, re.M)
            if len(profiles) != 1 or len(mac_profiles) != 1 or "PASS: Edge-32 Tensor" not in console:
                raise RuntimeError(f"{spec}/{capacity}: missing unique runtime/MAC profile or harness pass")
            counts = {k: int(v) for k,v in re.findall(r"(\w+)=(\d+)", profiles[0])}
            mac = {k: int(v) for k,v in re.findall(r"(\w+)=(\d+)", mac_profiles[0])}
            if counts["capacity"] != capacity or counts["chunks"] != 128//capacity:
                raise RuntimeError(f"{spec}/{capacity}: unexpected token partition {counts}")
            # NNC-prepacked IO needs ordinary DMA only; no SRAM COPY.
            copy_starts = 0
            compute_starts = 64*128//capacity
            if mac["macs"] != 128*64*64 or mac["issue"] != 8192 or mac["starts"] != compute_starts+copy_starts:
                raise RuntimeError(f"{spec}/{capacity}: incorrect work accounting {mac}")
            transfer, = re.findall(r"^NNC_QUANT_TRANSFER (.+)$",console,re.M)
            transfers = {k: int(v) for k,v in re.findall(r"(\w+)=(\d+)",transfer)}
            total_cycles = counts["op_cycles"]+sum(transfers.values())
            records.append(dict(spec=spec, requested_capacity=capacity, ideal_cycles=8192,
                                compute_starts=compute_starts, copy_starts=copy_starts,
                                stream_util_percent=819200/counts["stream_cycles"],
                                op_util_percent=819200/counts["op_cycles"],
                                transfer_inclusive_cycles=total_cycles,
                                transfer_inclusive_util_percent=819200/total_cycles,
                                **transfers,
                                **counts, **{f"mac_{k}": v for k,v in mac.items()}))
            (out / "results.json").write_text(json.dumps(records, indent=2)+"\n")
            if capacity == 128:
                # Exercise actual fallback under allocator pressure and with
                # manually placed SRAM input; verify both outputs and guards.
                regression = out / spec / "capacity-regression"
                build_args.main_file = ROOT / "nnc/test/quant_capacity_main.cpp"
                run_smoke.build(build_args, case_generated, regression)
                run_smoke.run(build_args, regression, expected)
                log = (regression / "software-console.log").read_text()
                capacities = [int(v) for v in re.findall(r"NNC_QUANT_PROFILE capacity=(\d+)", log)]
                # Physical ring bytes: eight dense FP4 slots, four Q8 slots.
                ring_bytes = {"mxfp4": 2176, "nvfp4": 1152, "q8_0": 1088}[spec]
                want = [128,(4096-ring_bytes)//32]
                if capacities != want:
                    raise RuntimeError(f"{spec}: capacity fallback {capacities} != {want}")
                accounting, = re.findall(r"^MAC PROFILE (.+)$",log,re.M)
                counters = {k: int(v) for k,v in re.findall(r"(\w+)=(\d+)",accounting)}
                expected_starts = 64*sum((128+c-1)//c for c in want)
                if counters["starts"] != expected_starts or counters["macs"] != 2*128*64*64:
                    raise RuntimeError(f"{spec}: unexpected COPY or missing work in capacity test: {counters}")
                forward_build = out / spec / "generated-forward"
                build_args.main_file = ROOT / "nnc/test/quant_forward_main.cpp"
                run_smoke.build(build_args,case_generated,forward_build)
                run_smoke.run(build_args,forward_build,expected)
                forward_log = (forward_build/"software-console.log").read_text()
                if not re.search(r"MAC PROFILE .*macs=524288 starts=64 ",forward_log):
                    raise RuntimeError(f"{spec}: generated forward did not use native 128-token SRAM path")
    with (out / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# Quant Linear 64×64, 128 tokens", "",
             "All 8192 BF16 output values match the format-specific NNC golden exactly.", "",
             "Ideal cycles = 128×64×64 / 64 MACs per cycle = 8192.", "",
             "Working SRAM uses tensor order; kernel time contains no SRAM-to-SRAM COPY.",
             "Stream time includes weight DMA and both final syncs; operator time also includes setup.",
             "NNC plans IO storage at compile time. Boundary load/store use ordinary whole-tensor DMA.",
             "The transfer-inclusive total adds ordinary DRAM load/store; printf and allocation are excluded.", "",
             "| Quant | Max tokens | Chunks | Stream cycles | Stream MAC utilization | Operator cycles | Operator MAC utilization | With boundary DMA | With DMA utilization |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in records:
        lines.append(f'| {r["spec"]} | {r["requested_capacity"]} | {r["chunks"]} | '
                     f'{r["stream_cycles"]} | {r["stream_util_percent"]:.2f}% | '
                     f'{r["op_cycles"]} | {r["op_util_percent"]:.2f}% | '
                     f'{r["transfer_inclusive_cycles"]} | {r["transfer_inclusive_util_percent"]:.2f}% |')
    (out / "results.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
