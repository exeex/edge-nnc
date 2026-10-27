# Tensor operator benchmark: 64×64 weights, 128 tokens

Measured **2026-10-28**, E3 wide H4, 128 KiB SRAM. All 24 operator/capacity measurements pass exact BF16 output comparison.

## Measurement scope

- BF16 input/output are already in tensor SRAM order; weights/RHS reside in DRAM.
- Stream cycles run from weight DMA submission through final Tensor/DMA sync.
- Operator cycles additionally include validation and setup. Weight DMA is included in both windows.
- Model entry/exit input/output transfers and KV-cache traffic are outside this operator benchmark. No transfer-inclusive result is calculated or listed.
- Ideal work is 524,288 MACs / 64 MACs per cycle = 8,192 cycles. No warmup.
- Matmul and Matmul-transpose each have one implementation: a complete row-major BF16 RHS in DRAM, sliced into 8×8 tiles on the fly by hardware strided circular DMA into ring8.
- Quant Linear weights are quantized and tiled by NNC at compile time. All quant formats share source weights/input and use their own arithmetic golden.

## Versions

| Repository | Commit |
| --- | --- |
| nnc | `4879aafcf887363067b1d5d04347bdc520febf39` |
| edge-e3 | `851bc5b60152e2c11bf42c9f9d9360a31a04a954` |
| edge-asic | `3390ec73ac28041c4039572674ce42ee11ddbaed` |
| edge-32 | `55a089b372e2a18f8fa99185db7db264ad72602d` |

The implementation was committed before this report was added. These are the previously completed measurements, with no rerun. Runtime/compiler source SHA-256 hashes match the listed NNC implementation commit; only report writers changed afterward. Original run metadata, including the pre-commit working-tree state and source hashes, is preserved in [provenance.json](provenance.json).

LLVM 22.1.8, RV32IM/Zba, ILP32, C++17, `-O2`. Simulator SHA-256: `bf0e01047b00027e920c4dddaf2065614fe4c5a0e4bd9495d3e90e2693bab970`.

## Capacity 128

| Operator | Stream cycles | Stream MAC utilization | Operator cycles | Operator MAC utilization |
| --- | ---: | ---: | ---: | ---: |
| matmul | 8,776 | 93.35% | 9,443 | 86.75% |
| matmul_transpose | 8,814 | 92.94% | 9,432 | 86.85% |
| linear_nvfp4 | 8,833 | 92.74% | 9,423 | 86.94% |
| linear_mxfp4 | 8,861 | 92.45% | 9,403 | 87.12% |
| linear_q8_0 | 8,765 | 93.46% | 9,210 | 88.95% |
| linear_fp4 | 8,747 | 93.65% | 9,186 | 89.18% |
| linear_fp8_e4m3fn | 8,765 | 93.46% | 9,210 | 88.95% |
| linear_fp8_e5m2 | 8,765 | 93.46% | 9,210 | 88.95% |

## Token capacity sweep

Capacity changes the number of token chunks within the same implementation.

| Operator | Capacity | Chunks | Stream cycles | Stream MAC utilization | Operator cycles | Operator MAC utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| matmul | 32 | 4 | 15,502 | 52.84% | 16,367 | 50.05% |
| matmul | 64 | 2 | 9,454 | 86.65% | 10,187 | 80.42% |
| matmul | 128 | 1 | 8,776 | 93.35% | 9,443 | 86.75% |
| matmul_transpose | 32 | 4 | 17,560 | 46.65% | 18,358 | 44.62% |
| matmul_transpose | 64 | 2 | 9,748 | 84.04% | 10,426 | 78.57% |
| matmul_transpose | 128 | 1 | 8,814 | 92.94% | 9,432 | 86.85% |
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

## Reproduction and evidence

From the NNC repository root, using a consumer-built matching E3/Edge-32 simulator:

```sh
python nnc/benchmark_tensor.py \
  --out-dir build/tensor-operator-unified \
  --sim-exe /path/to/Vedge32_tensor_integration_tb \
  --edge32-include /path/to/edge-32/include \
  --e3-include /path/to/edge-e3/include
```

- [CSV](results.csv), [JSON](results.json), [version/source manifest](provenance.json).
- Measured local artifacts and raw logs: consuming workspace `build/tensor-operator-unified/`.
- Matmul checks exact DMA XY/slot parameters, MAC/START counts, zero Tensor COPY and runtime RHS mutation.
- Quant capacity-128 cases check generated forward, SRAM guards and capacity fallback.
- Implementation validation: 49 host tests, 28 NNC CTests, full Transformer, producer/consumer pipelines, SRAM-oversized tensors, and multihead/129-token tails passed before report publication.
- [Matmul contract](../../MATMUL.md), [quant Linear contract](../../QUANT_LINEAR.md).
