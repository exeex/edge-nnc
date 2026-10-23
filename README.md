# Edge NNC

PyTorch graph lowering, generated C++ ABI, RV32 runtime, model fixtures and
correctness runners. Compiler and runtime are one versioned unit.

The consuming workspace supplies `edge32_intrinsic.hpp` and a Verilator
simulator. This package contains no RTL or encrypted simulator dependency.
See `UPSTREAM.md` for the imported source revision and integration changes.

Install with Python 3.12 and `uv sync --extra cpu`. Compiler tests:
`uv run --extra cpu python -m unittest nnc.test_compiler`.

Build a model with `nnc/run_smoke.py --build-only --edge32-include PATH`.
For RTL correctness, also provide `--sim-exe PATH` and omit `--build-only`.
Use `--model-file nnc/test/smoke_linear.py` for the minimal linear fixture.
The simulator must accept `+mem128`, `+mem128_words`, `+max_cycles`,
`+dump_base`, `+dump_len`, and `+dump_file`; dump one hexadecimal byte per line,
return nonzero on faults/timeouts/nonzero program status, and support the
linker memory map in `cpp/baremetal/linker.ld` (at least 6 MiB external RAM).
`nnc/run_harness.py` runs all smoke fixtures and the small Llama fixture.

The runtime supports BF16 Linear and a single-token FP4 + BF16-scale Linear
through WSLD; see [FP4_LINEAR.md](FP4_LINEAR.md). FP8 lowering is not implemented. Runtime PSUM tensors use BF16 storage and select E3
`dtype=0`. E3 `dtype=1` reads FP32 PSUM and cannot be used with those tensors.

Canonical repository: https://github.com/exeex/edge-nnc.
Consuming workspaces can pin this compiler/runtime pair as a Git submodule.
