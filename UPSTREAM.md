# Extraction provenance

Source: https://github.com/exeex/edge-cores
Commit: 00b6fb743d3c5d845da8564b6935b1debca6397f

Compiler, matched C++ runtime, intrinsic wrappers, bare-metal support, compiler
tests, Llama example and ELF conversion tool are extracted together. Hardware
RTL and edge-32 are supplied by the consumer, not copied here.

## Integration changes

- Simulator and Edge32 include path are explicit consumer inputs; removed
  the implicit encrypted simulator build dependency.
- Harness cycle parsing accepts both upstream and local E3 PASS output.
- BF16 runtime operations select dtype=0, matching current E3 BF16 PSUM
  storage. The imported typed BF16 configuration selected dtype=1, which
  reads FP32 PSUM on current E3 and failed the linear output comparison.
- Package metadata/lock name is edge-nnc. Hardware repositories are excluded.
- No changes were made to the source edge-cores workspace.

- Default cycle budget is 5,000,000; current E3 tiny Llama takes 2,030,770
  cycles, exceeding the imported 2,000,000-cycle single-model default.
