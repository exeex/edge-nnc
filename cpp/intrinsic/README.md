# Platform intrinsic adapters

This directory contains compatibility include names only. Implementations are
supplied by the consuming platform's
[E3 SDK](https://github.com/exeex/edge-e3/tree/main/include).

Supply its `include/` directory with `--e3-include` (or `EDGE_E3_INCLUDE`) and
the matching Edge-32 backend with `--edge32-include` (or `EDGE32_INCLUDE`).
Compiler lowering and libnn live in this repository; product intrinsic
implementations do not.
