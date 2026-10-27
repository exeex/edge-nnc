#!/usr/bin/env python3
"""Compile a PyTorch model to NNC headers and ELF payloads.

The public compiler entry point keeps graph reading, lowering, and C++
generation separate. Tensor allocation remains in the runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Allow both `python -m nnc.compiler` and `python nnc/compiler.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep existing `nnc.compiler` imports working while implementation lives in
# smaller modules.
from nnc.codegen import ForwardRenderer, InitRenderer, tensor_tile_bits
from nnc.graph import (
    AbiValue, ExportGraphReader, ForwardABI, Graph, GraphNode, LoweredNode,
    LoweredProgram, Shape, TensorSpec, export_graph, load_model_example,
)
from nnc.liveness import Liveness
from nnc.lowering import (
    OP_LOWERERS, AddLowerer, AttentionLowerer, BinarySameShapeLowerer,
    KvCacheLowerer, LinearLowerer, Lowerer, MatmulLowerer, MulLowerer,
    OpLowerer, RmsNormLowerer, RopeLowerer, SoftmaxLowerer, TensorTable,
    UnarySameShapeLowerer,
)
from nnc.weights import WeightRecord, WeightStore


def compile_model(model: Any, example_args: tuple[Any, ...]) -> tuple[LoweredProgram, WeightStore]:
    constants = dict(model.named_parameters()) | dict(model.named_buffers())
    lowerer = Lowerer(export_graph(model, example_args), constants)
    return lowerer.lower(), lowerer.weights


def write_artifacts(abi: ForwardABI, program: LoweredProgram,
                    weights: WeightStore, out_dir: Path, *, tensor_tile_io=False,
                    quant_token_capacity=None, matmul_token_capacity=None) -> None:
    """Write every artifact consumed by the bare-metal build."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forward.hpp").write_text(
        ForwardRenderer(program, abi, weights, tensor_tile_io=tensor_tile_io,
                        quant_token_capacity=quant_token_capacity,
                        matmul_token_capacity=matmul_token_capacity).render(), encoding="utf-8"
    )
    tile_inputs = {program.nodes[0].args[0]} if tensor_tile_io else set()
    init = InitRenderer(abi, weights, tensor_tile_io=tensor_tile_io,tile_input_names=tile_inputs)
    (out_dir / "init.hpp").write_text(init.render(), encoding="utf-8")
    (out_dir / "input.bin").write_bytes(init.input_data)
    (out_dir / "weight.bin").write_bytes(weights.data)
    if tensor_tile_io:
        import json
        (out_dir / "io_layout.json").write_text(json.dumps({
            "layout": "channel_block_token_lane", "lanes": 8,
            "inputs": {v.name: list(v.shape) for v in abi.inputs if v.kind == "tensor"},
            "input_layouts": {v.name: "tensor_tiles" if v.name in tile_inputs else "contiguous"
                              for v in abi.inputs if v.kind == "tensor"},
            "outputs": {v.name: list(v.shape) for v in abi.outputs if v.kind == "tensor"},
            "payloads": "input.bin follows input_layouts; output storage and expected.bin use tensor-tile order",
        },indent=2)+"\n")
    else:
        # Do not leave a stale packed-ABI marker when regenerating in place.
        (out_dir / "io_layout.json").unlink(missing_ok=True)


def compile(model: Any, example_args: tuple[Any, ...], *,
            out_dir: Path | str = "build/generated", save_reference: bool = False,
            tensor_tile_io: bool = False, quant_token_capacity: int | None = None,
            matmul_token_capacity: int | None = None) -> Path:
    """Export C++/payloads; optionally snapshot the BF16 reference for replay."""
    destination = Path(out_dir)
    abi = ForwardABI.from_example(model, example_args)
    program, weights = compile_model(model, example_args)
    write_artifacts(abi, program, weights, destination, tensor_tile_io=tensor_tile_io,
                    quant_token_capacity=quant_token_capacity,matmul_token_capacity=matmul_token_capacity)
    if save_reference:
        import struct
        import torch
        with torch.no_grad():
            output = model(*example_args)
        if isinstance(output, tuple) and len(output) == 1:
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise ValueError("saved reference requires a single tensor output")
        bits = output.detach().to(torch.bfloat16).cpu().contiguous().view(torch.int16).flatten().tolist()
        if tensor_tile_io:
            bits = tensor_tile_bits(bits,tuple(output.shape))
        (destination / "expected.bin").write_bytes(struct.pack(f"<{len(bits)}H", *(b & 0xffff for b in bits)))
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("example/llama/build/generated"))
    parser.add_argument("--tensor-tile-io", action="store_true",
                        help="planned LHS/output SRAM ABI for one quant Linear or BF16 matmul; matmul RHS stays contiguous")
    parser.add_argument("--quant-token-capacity", type=int)
    parser.add_argument("--matmul-token-capacity", type=int)
    parser.add_argument(
        "--model-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "example" / "llama" / "model.py",
    )
    args = parser.parse_args()

    model, example_args = load_model_example(args.model_file)
    compile(model, example_args, out_dir=args.out_dir, tensor_tile_io=args.tensor_tile_io,
            quant_token_capacity=args.quant_token_capacity,matmul_token_capacity=args.matmul_token_capacity)


if __name__ == "__main__":
    main()
