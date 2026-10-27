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

## One Matmul contract

`matmul(y, lhs, rhs, heads)` and `matmul_transpose(y, lhs, rhs, heads)` each
have one public Tensor API and one physical storage contract:

- RHS is the complete row-major BF16 matrix in DRAM. NNC does not quantize,
  transpose or pre-tile it. Hardware strided circular DMA slices it at runtime.
- LHS and output use the tensor engine's SRAM order: `[channel/8, token, 8]`,
  with an optional outer head axis. Tensor commands address these SRAM vectors.
- For the measured 64×64×128 workload, input/output remain in SRAM throughout
  the operator. There is no internal input/output DMA or SRAM-to-SRAM COPY.
- If an operand exceeds SRAM, the same storage contract is streamed in bounded
  token chunks. Complete chunks use ordinary DMA; partial input slices use
  source-strided DMA. Partial DRAM output writes use contiguous bursts per
  channel block because the DMA interface has no destination-stride field.

There is no `TensorTileView` Matmul overload, contiguous Matmul fallback,
compiler layout-mode flag, or contiguous/planned benchmark pair. C++ residency
specialization only eliminates scratch/staging branches for SRAM operands;
it does not introduce a different layout or algorithm.

```python
nnc.compile(model, (lhs, rhs), out_dir="build/matmul",
            save_reference=True, matmul_token_capacity=128)
```

NNC's `nnc/layout.py` assigns storage on graph edges. Compatible pointwise
producers/consumers preserve tensor order directly. RHS producers write
row-major DRAM storage. Attention expands into Matmul-transpose, softmax, and
Matmul so their layout and memory boundaries remain visible to the compiler.
A boundary to an operator with a different required layout uses explicit DMA
XY descriptors; it never uses scalar payload loops or SRAM-to-SRAM Tensor
COPY. These boundary costs are included in generated per-node profiling.

`io_layout.json` describes every external tensor's physical storage. Exported
input bytes, output storage and `expected.bin` follow this metadata; logical
PyTorch shapes remain unchanged. Consumers of generated payloads must use
this compiler-owned ABI. Matmul RHS is separately verified byte-for-byte as
row-major, and its `weight.bin` is empty.

## Verification and measurements

```sh
python nnc/benchmark_tensor.py \
  --out-dir build/tensor-operator-unified \
  --sim-exe /path/to/Vedge32_tensor_integration_tb \
  --edge32-include /path/to/edge-32/include \
  --e3-include /path/to/edge-e3/include
```

The combined report has one row per operator at capacity 128, followed by a
32/64/128 capacity sweep using the same implementations. Matmul validates
exact BF16 output, MAC/START counts, zero Tensor COPY, exact ring8 DMA XY
parameters, and RHS mutation between two invocations.

`benchmark_matmul.py --tokens 129 --heads 2 --n 16 --capacity 128` additionally
checks multiple heads, rectangular RHS and the final one-token chunk.
`smoke_matmul_pipeline.py`, `smoke_matmul_large.py`, attention and transformer
regressions cover graph layout propagation, runtime-produced RHS, and tensors
larger than SRAM.

The stream window begins before RHS DMA submission and ends after Tensor/DMA
sync. Operator time also includes validation and setup. Ordinary harness
input/output boundary DMA is outside this SRAM-operand timing. Ideal cycles
for 64×64×128 are 8192 at 64 MACs/cycle. ASIC operators perform no D-cache
maintenance under [MEMORY_OWNERSHIP.md](MEMORY_OWNERSHIP.md).

Measured results, compiler/source/simulator hashes, revisions/dirty state,
generated artifacts and raw command traces are preserved in the consuming
workspace's `build/tensor-operator-unified/`. Older reports retain their
historical IO variants and must not be treated as the current API.

The checked-in measurement table and version manifest are in
[benchmarks/tensor-64x64-128](benchmarks/tensor-64x64-128/README.md).

## Scalar control

Matmul retains P/partial-output pointers for a complete K reduction;
Matmul-transpose retains its input pointer for all output blocks at one K
block. First/middle/last tile modes are emitted from separate loop boundaries,
without specializing runtime tensor dimensions. The hardware START path copies
configuration forward between its CSR banks, so unchanged pointers need no
repeated setter commands. Tensor metadata passes by const reference.

The [benchmark report](benchmarks/tensor-64x64-128/README.md#software-control-optimization)
records the A/B cycles, pointer command counts, code size and source versions.
