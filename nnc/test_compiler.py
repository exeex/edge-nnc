#!/usr/bin/env python3
"""Focused tests for NNC lowering metadata and forward allocation."""

from __future__ import annotations

import struct
import unittest

import torch

from nnc import compiler


class RmsNormWeightTest(unittest.TestCase):
    def test_rms_norm_uses_dedicated_mean_and_sum_weights(self) -> None:
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("x", (32, 64), "input"))
        tensors.add(
            compiler.TensorSpec("weight", (64,), "weight", "norm.weight")
        )
        weights = compiler.WeightStore(
            {"norm.weight": torch.ones(64, dtype=torch.float32)}
        )
        node = compiler.GraphNode(
            ("out",), "rms_norm", ("x", "weight"), {"eps": "1e-05f"}
        )

        lowered = compiler.RmsNormLowerer().lower(node, tensors, weights)

        self.assertEqual(lowered.attrs["square_weight"], "rms_square_mean_64")
        self.assertEqual(lowered.attrs["reduce_weight"], "rms_reduce_sum8")
        self.assertEqual(lowered.attrs["weight"], "weight")
        self.assertEqual(lowered.attrs["eye"], "tensor_eye8")
        self.assertNotIn("setup", lowered.attrs)
        self.assertNotIn("out_rms_setup16", weights.records)

        square = weights.records["rms_square_mean_64"]
        square_bits = struct.unpack(
            "<64H", weights.data[square.offset:square.offset + square.nbytes]
        )
        self.assertEqual(
            square_bits,
            tuple(
                0x3C80 if row == col else 0
                for row in range(8)
                for col in range(8)
            ),
        )

        reduce_sum = weights.records["rms_reduce_sum8"]
        reduce_bits = struct.unpack(
            "<64H",
            weights.data[
                reduce_sum.offset:reduce_sum.offset + reduce_sum.nbytes
            ],
        )
        self.assertEqual(reduce_bits, (0x3F80,) * 64)


class DramPreferenceTest(unittest.TestCase):
    def test_softmax_prefers_input_and_output_in_dram(self) -> None:
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("score", (32, 32), "ssa"))
        node = compiler.GraphNode(("prob",), "softmax", ("score",))

        lowered = compiler.SoftmaxLowerer().lower(
            node, tensors, compiler.WeightStore({})
        )

        self.assertEqual(lowered.dram_preferred, ("score", "prob"))

    def test_elementwise_lowerers_do_not_force_rhs_to_dram(self) -> None:
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("lhs", (32, 64), "ssa"))
        tensors.add(compiler.TensorSpec("rhs", (32, 64), "ssa"))
        weights = compiler.WeightStore({})

        add = compiler.AddLowerer().lower(
            compiler.GraphNode(("add_out",), "add", ("lhs", "rhs")),
            tensors,
            weights,
        )
        mul = compiler.MulLowerer().lower(
            compiler.GraphNode(("mul_out",), "mul", ("lhs", "rhs")),
            tensors,
            weights,
        )

        self.assertEqual(add.dram_preferred, ())
        self.assertEqual(mul.dram_preferred, ())

    def test_matmul_lowerer_marks_rhs(self) -> None:
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("lhs", (32, 64), "input"))
        tensors.add(compiler.TensorSpec("rhs", (64, 64), "ssa"))
        node = compiler.GraphNode(("out",), "matmul", ("lhs", "rhs"))

        lowered = compiler.MatmulLowerer(False).lower(
            node, tensors, compiler.WeightStore({})
        )

        self.assertEqual(lowered.dram_preferred, ("rhs",))

    def test_attention_lowerer_keeps_rhs_and_unused_params_in_dram(self) -> None:
        tensors = compiler.TensorTable()
        for name in ("xq", "keys", "values", "params"):
            tensors.add(compiler.TensorSpec(name, (32, 64), "ssa"))
        node = compiler.GraphNode(
            ("out",), "attention", ("xq", "keys", "values", "params"),
            {"head_count": "2"},
        )

        lowered = compiler.AttentionLowerer().lower(
            node, tensors, compiler.WeightStore({})
        )

        self.assertEqual(
            lowered.dram_preferred,
            ("xq", "keys", "values", "params", "out"),
        )
        self.assertEqual(lowered.attrs["head_count"], "2")

    def test_attention_renderer_passes_head_count(self) -> None:
        out = compiler.TensorSpec("out", (32, 64), "ssa")
        program = compiler.LoweredProgram(
            (
                compiler.LoweredNode(
                    (out,),
                    "attention",
                    ("xq", "keys", "values", "params"),
                    {"eye": "tensor_eye8", "head_count": "2"},
                ),
            ),
            "out",
        )
        abi = compiler.ForwardABI(
            tuple(
                compiler.AbiValue(name, "tensor", (32, 64))
                for name in ("xq", "keys", "values", "params")
            ),
            (compiler.AbiValue("y", "tensor", (32, 64)),),
        )
        weights = compiler.WeightStore({})
        weights.use_eye8("tensor_eye8")

        rendered = compiler.ForwardRenderer(program, abi, weights).render()

        self.assertIn("nnedge::op::matmul_transpose(out_scores, xq_dtcm, keys, 2u)", rendered)
        self.assertIn("nnedge::op::matmul(out, layout_2_0, values, 2u)", rendered)
        self.assertNotIn("nnedge::op::attention(", rendered)
        self.assertIn("nnedge::op::softmax(out_prob, layout_1_0, ctx->tensor_eye8, 2u)", rendered)

    def test_kv_cache_lowerer_keeps_cache_outputs_in_dram(self) -> None:
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("xk", (32, 64), "ssa"))
        tensors.add(compiler.TensorSpec("xv", (32, 64), "ssa"))
        tensors.add(compiler.TensorSpec("start_pos", (), "input"))
        node = compiler.GraphNode(
            ("keys", "values"),
            "kv_cache_update",
            ("xk", "xv", "start_pos"),
        )

        lowered = compiler.KvCacheLowerer().lower(
            node, tensors, compiler.WeightStore({})
        )

        self.assertEqual(lowered.dram_preferred, ("keys", "values"))

    def test_renderer_allocates_preferred_ssa_value_in_dram(self) -> None:
        rhs = compiler.TensorSpec("rhs", (64, 64), "ssa")
        out = compiler.TensorSpec("out", (32, 64), "ssa")
        program = compiler.LoweredProgram(
            (
                compiler.LoweredNode((rhs,), "silu", ("x",), {}),
                compiler.LoweredNode(
                    (out,),
                    "matmul",
                    ("x", "rhs"),
                    {},
                    dram_preferred=("rhs",),
                ),
            ),
            "out",
        )
        abi = compiler.ForwardABI(
            (compiler.AbiValue("x", "tensor", (32, 64)),),
            (compiler.AbiValue("y", "tensor", (32, 64)),),
        )

        rendered = compiler.ForwardRenderer(
            program, abi, compiler.WeightStore({})
        ).render()

        self.assertIn("nnedge::malloc_tensor_dram(rhs);", rendered)

    def test_renderer_keeps_preferred_input_in_dram(self) -> None:
        out = compiler.TensorSpec("out", (32, 64), "ssa")
        program = compiler.LoweredProgram(
            (
                compiler.LoweredNode(
                    (out,),
                    "matmul",
                    ("lhs", "rhs"),
                    {},
                    dram_preferred=("rhs",),
                ),
            ),
            "out",
        )
        abi = compiler.ForwardABI(
            (
                compiler.AbiValue("lhs", "tensor", (32, 64)),
                compiler.AbiValue("rhs", "tensor", (64, 64)),
            ),
            (compiler.AbiValue("y", "tensor", (32, 64)),),
        )

        rendered = compiler.ForwardRenderer(
            program, abi, compiler.WeightStore({})
        ).render()

        self.assertIn("Tensor<dtype> lhs_dtcm", rendered)
        self.assertNotIn("Tensor<dtype> rhs_dtcm", rendered)
        self.assertIn("nnedge::op::matmul(out, lhs_dtcm, rhs, 1u)", rendered)


class InitRendererTest(unittest.TestCase):
    def test_io_payloads_use_aligned_section_offsets(self) -> None:
        abi = compiler.ForwardABI(
            (
                compiler.AbiValue("x", "tensor", (3,), (0x3f80, 0xbf80, 0)),
                compiler.AbiValue("x2", "tensor", (2,), (0x4000, 0x4040)),
            ),
            (
                compiler.AbiValue("y", "tensor", (3,)),
                compiler.AbiValue("y2", "tensor", (2,)),
            ),
        )
        init = compiler.InitRenderer(abi, compiler.WeightStore({}))
        rendered = init.render()

        self.assertEqual(init.input_offsets, {"x": 0, "x2": 64})
        self.assertEqual(init.output_offsets, {"y": 0, "y2": 64})
        self.assertEqual(struct.unpack_from("<3H", init.input_data), (0x3f80, 0xbf80, 0))
        self.assertEqual(struct.unpack_from("<2H", init.input_data, 64), (0x4000, 0x4040))
        self.assertIn("input_begin) + 64", rendered)
        self.assertIn("output_begin + 64", rendered)
        self.assertIn('section(".nnedge_outputs")', rendered)
        self.assertNotIn("dtype::from_bits", rendered)



class LinearMatmulShapeContractTest(unittest.TestCase):
    def linear(self, x, w, source="weight"):
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("x", x, "input"))
        tensors.add(compiler.TensorSpec("w", w, source, "w" if source == "weight" else ""))
        return compiler.LinearLowerer().lower(
            compiler.GraphNode(("y",), "linear", ("x", "w")), tensors,
            compiler.WeightStore({"w": torch.ones(w)}))

    def matmul(self, lhs, rhs, heads=1, transpose=False):
        tensors = compiler.TensorTable()
        tensors.add(compiler.TensorSpec("a", lhs, "input"))
        tensors.add(compiler.TensorSpec("b", rhs, "input"))
        return compiler.MatmulLowerer(transpose).lower(
            compiler.GraphNode(("y",), "matmul", ("a", "b"),
                               {"head_count": str(heads)}),
            tensors, compiler.WeightStore({}))

    def test_linear_rejects_runtime_silent_return_shapes(self):
        for x, w in [((2, 8, 8), (8, 8)), ((4, 9), (8, 9)),
                     ((4, 8), (9, 8)), ((4, 8), (8, 16)), ((0, 8), (8, 8))]:
            with self.subTest(x=x, w=w), self.assertRaises(SystemExit):
                self.linear(x, w)

    def test_linear_rejects_dynamic_weight(self):
        with self.assertRaisesRegex(SystemExit, "static"):
            self.linear((4, 8), (8, 8), "input")

    def test_odd_token_count_is_supported(self):
        self.assertEqual(self.linear((9, 16), (8, 16)).outputs[0].shape, (9, 8))
        self.assertEqual(self.matmul((9, 16), (16, 8)).outputs[0].shape, (9, 8))

    def test_matmul_rejects_mismatched_or_unaligned_axes(self):
        for lhs, rhs, heads, transpose in [
            ((8,), (8, 8), 1, False), ((8, 8), (16, 8), 1, False),
            ((8, 9), (9, 8), 1, False), ((8, 8), (8, 9), 1, False),
            ((3, 8, 8), (8, 16), 2, False),
            ((8, 16), (8, 24), 2, True), ((0, 8), (8, 8), 1, False),
        ]:
            with self.subTest(lhs=lhs, rhs=rhs), self.assertRaises(SystemExit):
                self.matmul(lhs, rhs, heads, transpose)

    def test_matmul_scratch_boundary(self):
        self.matmul((96, 64), (64, 64))
        self.matmul((128, 64), (64, 64))  # Runtime chunking, eight RHS slots.
        self.matmul((1024, 64), (64, 64))
        with self.assertRaisesRegex(SystemExit, "scratch"):
            self.matmul((128, 65536), (65536, 8))


if __name__ == "__main__":
    unittest.main()
