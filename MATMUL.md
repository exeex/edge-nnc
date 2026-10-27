# BF16 Matmul: runtime DMA slicing

Matmul and Linear have different RHS contracts:

| Operation | RHS preparation | Runtime DMA |
| --- | --- | --- |
| Quant Linear | NNC quantizes static weights and serializes tilewise scale/weight slots | Reads consecutive packed slots, allowing long AXI bursts |
| BF16 Matmul / Matmul-transpose | RHS remains an ordinary runtime BF16 matrix | DMA XY hardware slices the matrix into 8×8 tiles |

Matmul has no quant mode. It never runs a CPU payload-copy, transpose, or
tile-packing loop on RHS. The CPU submits DMA parameters and Tensor commands.
The hardware producer and consumer recycle a fixed eight-slot circular ring.

## DMA parameters

One BF16 tile is 8×8×2 = 128 bytes, composed of eight 16-byte fragments.
Eight slots consume **1024 bytes** of SRAM, independently of full RHS size.
For per-head K/N dimensions and the original RHS row stride in bytes:

| Parameter | Matmul | Matmul-transpose |
| --- | --- | --- |
| Fragment bytes | 16 | 16 |
| X stride | RHS row stride | RHS row stride |
| X count | K | N |
| Y stride | 16 | 16 |
| Y count | N/8 | K/8 |
| Entry bytes | 128 | 128 |
| Circular slots | 8 | 8 |
| Consumer | WLD-T | WLD |
| Command loop order | output block → K block | K block → output block |

One XY circular command covers a head/token chunk. Every eight fragments
publish a complete tile to the ring. WLD consumes a slot and the producer
continues filling available slots. AXI request formation and any legal
coalescing belong to DMA hardware; software does not issue eight individual
payload reads per tile. The RHS source base advances by the head's K offset
for transposed RHS or N offset for ordinary RHS.

## SRAM and numerical behavior

`cpp/libnn/matmul.hpp` supports BF16 input, RHS and final output. Tile modes
use FP32 containers for the existing BF22 partial sums, rounding only the
final tile to BF16, as in the original tensor/matmul.hpp tests. No weight
quantization or quant scale tokens are involved.

The default token ceiling is 128, configurable through the C++ template or
`nnc.compile(..., matmul_token_capacity=32/64/128)`. Runtime bounds the actual
capacity by available SRAM. It processes one head/chunk at a time and
allocates a fixed RHS ring, not a full SRAM copy of RHS. Scratch avoids the
allocator high-water mark and explicitly supplied working SRAM tensors.
All engines finish before a ring or scratch region is reused.

Two IO contracts are explicit:

- Default contiguous tensors use DMA input staging/output restoration, like
  the original runtime Matmul tests. They do not use SRAM-to-SRAM COPY.
- `TensorTileView` LHS/output use preplanned working SRAM directly, with no
  internal IO staging. The RHS remains a normal contiguous DRAM tensor.

The planned compiler mode supports a single BF16 Matmul operator, including
explicit heads and transpose, with complete LHS/output fitting the SRAM
tensor arena:

```python
nnc.compile(model, (lhs, rhs), out_dir="build/matmul",
            save_reference=True, tensor_tile_io=True,
            matmul_token_capacity=128)
```

NNC plans **only LHS/output** storage. It leaves the runtime RHS byte-for-byte
contiguous in input.bin, with no static weight payload. `io_layout.json`
describes each input separately. Generated code copies planned LHS/output
with ordinary DMA and calls the working-SRAM overload. Dynamic producers
must follow the chosen physical IO ABI; whole-graph layout propagation is
not implemented by this single-op mode. Default contiguous export is unchanged.

## Verification and measurements

```sh
python nnc/benchmark_matmul.py \
  --out-dir build/matmul-dma-owned \
  --sim-exe /path/to/Vedge32_tensor_integration_tb \
  --edge32-include /path/to/edge-32/include \
  --e3-include /path/to/edge-e3/include
```

This compares capacities 32/64/128 and contiguous/planned IO for both BF16
operations with 64×64 RHS and 128 tokens. It checks exact output values,
MAC and descriptor counts, zero Tensor COPY starts, and each circular
command's fragment/entry/X/Y/slot configuration. It verifies that exported
RHS bytes are untouched and weight.bin is empty. Separate two-invocation
tests modify RHS in DRAM after the first invocation and compare the second
result, proving that the kernel uses current runtime data.

`--tokens 129 --heads 2 --n 16 --capacity 128` additionally exercises
multiple heads, rectangular RHS, and a one-token tail. Both IO contracts
and transpose directions pass these checks. Existing multihead/tail CTest
cases and the quant Linear regressions also pass.

Measured 2026-10-28, E3 `851bc5b6`, Edge-32 `55a089b3`, LLVM 22.1.8,
RV32IM/Zba `-O2`, wide H4, ring8, 64×64 RHS × 128 tokens:

| Planned working-SRAM IO | Stream cycles | Stream MAC utilization | Complete SRAM kernel cycles | Kernel MAC utilization |
| --- | ---: | ---: | ---: | ---: |
| Matmul | 8,787 | 93.23% | 9,653 | 84.86% |
| Matmul-transpose | 8,839 | 92.68% | 9,677 | 84.65% |

Ideal cycles are 128×64×64/64 = 8192. Stream timing starts before RHS circular
DMA submission and ends after both Tensor/DMA syncs. Kernel time includes
validation and capacity/setup. ASIC/DMA operators perform no D-cache
maintenance under the [tensor ownership contract](MEMORY_OWNERSHIP.md). Planned IO's ordinary
boundary DMA is outside these measurements; contiguous operator measurements
include their DMA staging/restoration and are labeled separately in the report.
Profiling is opt-in with `NNEDGE_MATMUL_PROFILE=1`; printing follows timing.

Source/compiler/simulator hashes, raw command traces, JSON/CSV/Markdown
results and generated artifacts are in the consuming workspace's
`build/matmul-dma-owned/` and `build/matmul-dma-owned-head-tail/`. A forced-inline
parameter helper avoids scalar function-call overhead between tile starts;
it performs no software payload slicing.

Removing redundant operator cache maintenance reduced planned SRAM kernel
cycles from 10,977 to 9,653 (Matmul) and 10,987 to 9,677 (transpose).
The weight stream, DMA tile slicing, and hardware are unchanged.
