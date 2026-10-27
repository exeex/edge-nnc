# Tensor operator benchmark: 64×64 weights, 128 tokens

Measured **2026-10-28**. Rebuilt from the clean implementation commit listed below.

All 24 capacity measurements pass exact BF16 output comparison. Ideal: 524,288 MACs / 64 MACs per cycle = 8,192 cycles. No warmup.

Input/output reside in tensor SRAM order. Stream cycles include weight DMA submission through final Tensor/DMA sync. Operator cycles include setup and validation, excluding ordinary boundary input/output DMA.

Matmul has one implementation and one storage contract: full row-major BF16 RHS in DRAM; hardware strided circular DMA slices 8×8 tiles into ring8 at runtime. Matmul-transpose uses the corresponding DMA order and WLD orientation. NNC generates the SRAM operand storage by default, with no layout-mode switch. Quant Linear uses NNC-quantized, compile-time tiled weights.

## Capacity 128

| Operator | Capacity | Chunks | Stream cycles | Stream MAC utilization | Operator cycles | Operator MAC utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul | 128 | 1 | 8,725 | 93.89% | 9,329 | 87.81% |
| matmul_transpose | 128 | 1 | 8,725 | 93.89% | 9,243 | 88.63% |
| linear_nvfp4 | 128 | 1 | 8,833 | 92.74% | 9,423 | 86.94% |
| linear_mxfp4 | 128 | 1 | 8,861 | 92.45% | 9,403 | 87.12% |
| linear_q8_0 | 128 | 1 | 8,765 | 93.46% | 9,210 | 88.95% |
| linear_fp4 | 128 | 1 | 8,747 | 93.65% | 9,186 | 89.18% |
| linear_fp8_e4m3fn | 128 | 1 | 8,765 | 93.46% | 9,210 | 88.95% |
| linear_fp8_e5m2 | 128 | 1 | 8,765 | 93.46% | 9,210 | 88.95% |

## Token capacity sweep (same operators)

| Operator | Capacity | Chunks | Stream cycles | Stream MAC utilization | Operator cycles | Operator MAC utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul | 32 | 4 | 11,684 | 70.11% | 12,399 | 66.07% |
| matmul | 64 | 2 | 9,238 | 88.68% | 9,879 | 82.92% |
| matmul | 128 | 1 | 8,725 | 93.89% | 9,329 | 87.81% |
| matmul_transpose | 32 | 4 | 13,676 | 59.90% | 14,293 | 57.31% |
| matmul_transpose | 64 | 2 | 9,566 | 85.64% | 10,117 | 80.97% |
| matmul_transpose | 128 | 1 | 8,725 | 93.89% | 9,243 | 88.63% |
| linear_nvfp4 | 32 | 4 | 13,304 | 61.58% | 14,299 | 57.29% |
| linear_nvfp4 | 64 | 2 | 9,358 | 87.54% | 10,083 | 81.25% |
| linear_nvfp4 | 128 | 1 | 8,833 | 92.74% | 9,423 | 86.94% |
| linear_mxfp4 | 32 | 4 | 11,356 | 72.14% | 12,267 | 66.78% |
| linear_mxfp4 | 64 | 2 | 9,366 | 87.47% | 10,031 | 81.67% |
| linear_mxfp4 | 128 | 1 | 8,861 | 92.45% | 9,403 | 87.12% |
| linear_q8_0 | 32 | 4 | 10,289 | 79.62% | 10,938 | 74.89% |
| linear_q8_0 | 64 | 2 | 9,257 | 88.50% | 9,770 | 83.85% |
| linear_q8_0 | 128 | 1 | 8,765 | 93.46% | 9,210 | 88.95% |
| linear_fp4 | 32 | 4 | 10,227 | 80.10% | 10,870 | 75.36% |
| linear_fp4 | 64 | 2 | 9,231 | 88.74% | 9,738 | 84.12% |
| linear_fp4 | 128 | 1 | 8,747 | 93.65% | 9,186 | 89.18% |
| linear_fp8_e4m3fn | 32 | 4 | 10,289 | 79.62% | 10,938 | 74.89% |
| linear_fp8_e4m3fn | 64 | 2 | 9,257 | 88.50% | 9,770 | 83.85% |
| linear_fp8_e4m3fn | 128 | 1 | 8,765 | 93.46% | 9,210 | 88.95% |
| linear_fp8_e5m2 | 32 | 4 | 10,289 | 79.62% | 10,938 | 74.89% |
| linear_fp8_e5m2 | 64 | 2 | 9,257 | 88.50% | 9,770 | 83.85% |
| linear_fp8_e5m2 | 128 | 1 | 8,765 | 93.46% | 9,210 | 88.95% |

## Measured revisions

| Repository | Commit | Measurement state |
| --- | --- | --- |
| NNC | `79091dc70e8b8fdb5d46a8595160675fc40549b0` | clean |
| edge-e3 | `851bc5b60152e2c11bf42c9f9d9360a31a04a954` | clean |
| edge-asic | `3390ec73ac28041c4039572674ce42ee11ddbaed` | clean |
| Edge-32 | `55a089b372e2a18f8fa99185db7db264ad72602d` | clean |

Revisions and per-file source SHA-256 hashes were recorded before the build.

LLVM 22.1.8, RV32IM/Zba, `-O2`, E3 wide H4, ring8, 128 KiB SRAM.

Simulator SHA-256: `bf0e01047b00027e920c4dddaf2065614fe4c5a0e4bd9495d3e90e2693bab970`.


## Software-control optimization

Rows, K, N and head count remain runtime values. No static-dimension template
specialization was added. The existing token-capacity options and DMA layout
contract are unchanged.

- Keep Matmul's PSUM pointer and intermediate output pointer across K tiles.
- Keep Matmul-transpose's input pointer across output blocks at a fixed K tile.
- Peel single/first/middle/last K boundaries so tile modes are constant within their loops.
- Pass Tensor metadata by const reference through the Matmul call chain.
- START snapshots configuration and copies it forward to the next CSR bank;
  later descriptors safely inherit unchanged pointers. DMA order, WLD/START
  counts, arithmetic, syncs, and timing windows are unchanged.

Fresh baseline versus optimized complete operator measurements:

| Operator | Capacity | Before cycles | After cycles | Before MAC utilization | After MAC utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| matmul | 32 | 16,367 | 12,399 | 50.05% | 66.07% |
| matmul | 64 | 10,187 | 9,879 | 80.42% | 82.92% |
| matmul | 128 | 9,443 | 9,329 | 86.75% | 87.81% |
| matmul_transpose | 32 | 18,358 | 14,293 | 44.62% | 57.31% |
| matmul_transpose | 64 | 10,426 | 10,117 | 78.57% | 80.97% |
| matmul_transpose | 128 | 9,432 | 9,243 | 86.85% | 88.63% |

At capacity 128 (64 tile STARTs), logical pointer setters change as follows:

| Operator | SETIN | SETOUT | SETPSUM |
| --- | ---: | ---: | ---: |
| Baseline, either operator | 64 | 64 | 64 |
| Matmul | 64 | 16 | 8 |
| Matmul-transpose | 8 | 64 | 56 |

Each address setter emits both RV32 address halves. The measured Matmul
capacity-128 ELF `.text` decreases from 11,508 to 11,312 bytes. This is a
specific test-image size comparison, not an instruction-cache miss estimate.

## Reproduction and evidence

```sh
python nnc/benchmark_tensor.py \
  --out-dir build/tensor-operator-control \
  --sim-exe /path/to/Vedge32_tensor_integration_tb \
  --edge32-include /path/to/edge-32/include \
  --e3-include /path/to/edge-e3/include
```

- [CSV](results.csv), [JSON](results.json), [version/source manifest](provenance.json).
- [Baseline and control comparison](control-comparison.json).
- Local raw logs/generated headers/payloads/ELF: consuming workspace `build/tensor-operator-control/`.
- All 24 operator/capacity measurements pass exact BF16 comparison, including generated quant forward and allocator-pressure guards.
- All 28 NNC CTests pass. Additional Matmul tests cover single/two K tiles, multiple heads, rectangular matrices, 129-token tails and runtime RHS mutation.
- Model entry/exit transfers and KV-cache traffic are outside this operator benchmark; no transfer-inclusive result is listed.
- [Matmul contract](../../MATMUL.md), [quant Linear contract](../../QUANT_LINEAR.md).
