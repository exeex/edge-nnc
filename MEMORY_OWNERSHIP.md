# Tensor payload ownership

NNC keeps shapes, descriptors, and other scalar metadata in cacheable CPU
memory. Tensor input, output, and weight payloads belong to DMA/ASIC engines.
Metadata and payloads must not share cache lines. Allocate DRAM payload arenas
on cache-line boundaries, including padding at the end of each arena.

Matmul, Linear, quant Linear, activation, and other ASIC operators synchronize
their DMA/compute engines before exposing results. They do not clean or
invalidate D-cache. Ordinary DMA tensor copy follows the same rule. Engine
completion and CPU cache ownership are separate responsibilities.

When an operator accesses a DRAM tensor through scalar loads/stores:

1. Call `scalar_tensor_acquire(tensor)` before CPU access to discard a stale
   CPU view of a completed DMA result. The previous scalar producer must
   already have published its writes.
2. After scalar writes, call `scalar_tensor_publish(tensor)` before returning
   or submitting DMA that consumes those writes. This cleans the written
   payload to DRAM.

Both helpers skip DTCM tensors. Scalar `zero`, DRAM-to-DRAM scalar `copy`, and
the scalar multiplication fallback own these boundaries internally. External
CPU producers, CPU result checkers, and future scalar operators must follow
the same contract. In-place scalar operators acquire all operands before
writing and publish their output afterward.

Generated `init.hpp` initializes tensor metadata and pointers. Preloaded input
and weight bytes are already in DRAM, so init performs no payload cache clean.
If a caller writes input bytes using the CPU instead, that caller publishes
them before `forward()`; a CPU reader acquires outputs after `forward()`.

`nnc/test/cache_ownership_main.cpp` exercises scalar publication, DMA output
replacement after CPU reads, scalar acquisition, zero, and scalar copy on the
Edge-32 RTL harness. Matmul mutation and weight-pack tests also publish their
CPU-produced payloads explicitly.
