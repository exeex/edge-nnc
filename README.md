# Edge NNC

PyTorch graph lowering, generated C++ ABI, RV32 runtime, model fixtures and
correctness runners. Compiler and runtime are one versioned unit.

The consuming workspace supplies the E3 SDK (`include/edge_e3/`), the
matching `edge32_intrinsic.hpp` backend and a Verilator simulator. This package contains no RTL or encrypted simulator dependency.
See `UPSTREAM.md` for the imported source revision and integration changes.

Install with Python 3.12 and `uv sync --extra cpu`. Compiler tests:
`uv run --extra cpu python -m unittest nnc.test_compiler`.

Build a model with `nnc/run_smoke.py --build-only --edge32-include PATH --e3-include PATH`.
The first path is Edge-32 `include/`; the second is E3 `include/`. The
`EDGE32_INCLUDE` and `EDGE_E3_INCLUDE` environment variables also select them.
`cpp/intrinsic/` contains forwarding headers only; E3 owns the implementation.
The shared RV32 runner support remains under `cpp/baremetal/`.
For RTL correctness, also provide `--sim-exe PATH` and omit `--build-only`.
Use `--model-file nnc/test/smoke_linear.py` for the minimal linear fixture.
The simulator must accept `+mem128`, `+mem128_words`, `+max_cycles`,
`+dump_base`, `+dump_len`, and `+dump_file`; dump one hexadecimal byte per line,
return nonzero on faults/timeouts/nonzero program status, and support the
linker memory map in `cpp/baremetal/linker.ld` (at least 6 MiB external RAM).
`nnc/run_harness.py` runs all smoke fixtures and the small Llama fixture.

The runtime supports BF16 Linear and a token-batched FP4 + BF16-scale Linear
through WSLD; see [FP4_LINEAR.md](FP4_LINEAR.md). FP8/Q8 graph lowering is not implemented. `cpp/libnn/linear_quant.hpp` exposes the named `linear_fp4`, `linear_q8_0`, `linear_fp8_e4m3fn`, and `linear_fp8_e5m2` K32/eight-BF16-scale presets as thin forwards to the shared kernel; Q8_0 requires converted E3 packing. The shared `linear_weight_slots(y, x, Tensor<WeightPack<WType, SType, Tiles, Scales>>)` runtime also accepts BF16, FP8 E4M3FN/E5M2 and INT8 packs. Each pack stores BF16 scales (`SType=0`), then `Tiles` output-major 8x8 tiles, then padding to 16 bytes. `Scales` is 1 (broadcast) or 8 (per lane); scale-free BF16 defaults to `Tiles=1, Scales=0`. Tensor shapes count packs, ordered by output block then K group; the group spans `8*Tiles` K elements. The byte-oriented `linear_weight_slots<WType>` overload remains available for generated code and existing benchmark adapters. Ordinary BF16 operators use BF16 PSUM with `dtype=0`. FP4 Linear uses the
E3 BF22-in-FP32 tile-mode PSUM path and rounds only the final output to BF16.

Canonical repository: https://github.com/exeex/edge-nnc.
Consuming workspaces can pin this compiler/runtime pair as a Git submodule.

For operator cycle counts, pass `--profile` to `nnc/run_smoke.py`. Runtime
phase diagnostics are opt-in with `--phase-profile`; their console cost is
included in operator timings, so use separate runs for phase diagnosis.
Pass simulator options with repeatable `--sim-arg` (for example
`--sim-arg +profile_mac` on the composed E3 harness).

BF16 Linear currently accepts static weights and rank-two inputs with K/N
multiples of eight; M may have a tail. Matmul implements the head-aware
normal/transposed layouts used by NNC, with per-head K/N multiples of eight
and a 32 KiB staging limit. Unsupported shapes are rejected during lowering.
These contracts do not cover general PyTorch broadcasting or arbitrary K/N tails.
