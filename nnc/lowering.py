"""Lower graph nodes to kernel calls and record their tensor shapes."""

from __future__ import annotations

import torch

from nnc.graph import Graph, GraphNode, LoweredNode, LoweredProgram, Shape, TensorSpec
from nnc.weights import WeightStore


class TensorTable:
    def __init__(self) -> None:
        self.specs: dict[str, TensorSpec] = {}

    def add(self, spec: TensorSpec) -> None:
        self.specs[spec.name] = spec

    def require(self, name: str) -> TensorSpec:
        try:
            return self.specs[name]
        except KeyError as exc:
            raise SystemExit(f"unknown tensor: {name}") from exc


class OpLowerer:
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        raise NotImplementedError

    @staticmethod
    def static_or_dynamic(tensors: TensorTable, name: str) -> TensorSpec:
        return tensors.require(name)

    @staticmethod
    def output(name: str, shape: Shape) -> TensorSpec:
        return TensorSpec(name, shape, "ssa")


class UnarySameShapeLowerer(OpLowerer):
    def __init__(self, op: str) -> None:
        self.op = op

    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        del weights
        src = self.static_or_dynamic(tensors, node.args[0])
        return LoweredNode((self.output(node.outputs[0], src.shape),), self.op, node.args, node.attrs)


class BinarySameShapeLowerer(OpLowerer):
    def __init__(self, op: str) -> None:
        self.op = op

    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        del weights
        lhs = self.static_or_dynamic(tensors, node.args[0])
        return LoweredNode((self.output(node.outputs[0], lhs.shape),), self.op, node.args, node.attrs)


class AddLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        lhs = self.static_or_dynamic(tensors, node.args[0])
        rhs = self.static_or_dynamic(tensors, node.args[1])
        if rhs.shape != lhs.shape and not (len(lhs.shape) == 2 and rhs.shape == (lhs.shape[1],)):
            raise SystemExit("add requires equal shapes or a rank-1 per-row bias")
        if rhs.target in weights.parameters:
            weights.use(rhs)
        eye = weights.use_eye8("tensor_eye8")
        return LoweredNode(
            (self.output(node.outputs[0], lhs.shape),),
            "add",
            node.args,
            {"eye": eye.name, **node.attrs},
        )


class MulLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        lhs = self.static_or_dynamic(tensors, node.args[0])
        eye = weights.use_eye8("tensor_eye8")
        return LoweredNode(
            (self.output(node.outputs[0], lhs.shape),),
            "mul",
            node.args,
            {"eye": eye.name, **node.attrs},
        )


class SoftmaxLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        src = self.static_or_dynamic(tensors, node.args[0])
        if len(src.shape) != 2:
            raise SystemExit(f"Tensor softmax currently requires a rank-2 tensor: {src.shape}")
        cols = src.shape[1]
        if cols <= 0 or cols > 64 or cols % 8 != 0:
            raise SystemExit(
                f"Tensor softmax width must be a positive multiple of 8 up to 64: {cols}"
            )
        eye = weights.use_eye8("tensor_eye8")
        return LoweredNode(
            (self.output(node.outputs[0], src.shape),),
            "softmax",
            node.args,
            {"eye": eye.name, **node.attrs},
            dram_preferred=(node.args[0], node.outputs[0]),
        )


class RmsNormLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        src = self.static_or_dynamic(tensors, node.args[0])
        weight = self.static_or_dynamic(tensors, node.args[1])
        if not src.shape:
            raise SystemExit("Tensor RMSNorm input must have at least one dimension")
        cols = src.shape[-1]
        if weight.shape != (cols,):
            raise SystemExit(f"RMSNorm weight shape {weight.shape} does not match width {cols}")
        weights.use(weight)
        eye = weights.use_eye8("tensor_eye8")
        square_weight = weights.use_rms_square_mean(
            f"rms_square_mean_{cols}", cols
        )
        reduce_weight = weights.use_rms_reduce_sum8("rms_reduce_sum8")
        return LoweredNode(
            (self.output(node.outputs[0], src.shape),),
            "rms_norm",
            (node.args[0],),
            {
                "weight": weight.name,
                "eye": eye.name,
                "reduce_weight": reduce_weight.name,
                "square_weight": square_weight.name,
                **node.attrs,
            },
        )


class LinearLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        x = self.static_or_dynamic(tensors, node.args[0])
        weight = self.static_or_dynamic(tensors, node.args[1])
        if len(x.shape) != 2 or len(weight.shape) != 2:
            raise SystemExit("BF16 Linear requires rank-2 input and weight")
        if min(*x.shape, *weight.shape) <= 0 or x.shape[1] != weight.shape[1]:
            raise SystemExit("BF16 Linear requires positive matching K dimensions")
        if x.shape[1] % 8 or weight.shape[0] % 8:
            raise SystemExit("BF16 Linear requires K%8=0 and N%8=0")
        if weight.target not in weights.parameters:
            raise SystemExit("BF16 Linear weight must be a static parameter")
        weights.use(weight)
        out_shape = (*x.shape[:-1], weight.shape[0])
        attrs = {"quant": "bf16", "tile": "auto", **node.attrs}
        return LoweredNode((self.output(node.outputs[0], out_shape),), "linear", node.args, attrs)


class LinearQuantLowerer(OpLowerer):
    def __init__(self, spec):
        self.spec = spec

    def lower(self, node, tensors, weights):
        if len(node.args) not in (2, 3):
            raise SystemExit("quant Linear requires (x, weight) or (x, q, scale)")
        x, q = (tensors.require(name) for name in node.args[:2])
        scale = tensors.require(node.args[2]) if len(node.args) == 3 else None
        if (len(q.shape) != 2 or len(x.shape) != 2 or
                x.shape[0] <= 0 or x.shape[1] != q.shape[1]):
            raise SystemExit("quant Linear requires positive matching [M,K] and [N,K]")
        n, k = q.shape
        if n <= 0 or k <= 0 or n % 8 or k % 32 or (scale is not None and scale.shape != (n, k // 32)):
            raise SystemExit("quant Linear requires N%8=0, K%32=0, scales[N,K/32]")
        slot_bytes = 144 if self.spec == "fp4" else 272
        if ((2*k+63)//64*64 + (2*n+63)//64*64 + 128 + 4*slot_bytes) > 32768:
            raise SystemExit("quant Linear operands exceed DTCM scratch capacity")
        if q.target not in weights.parameters or (scale is not None and scale.target not in weights.parameters):
            raise SystemExit("quant weight and scale must be static parameters")
        if self.spec == "fp4":
            from nnc.fp4 import pack_slots, quantize
        else:
            from nnc.quant import pack_slots, quantize
        if scale is None:
            original = weights.parameters[q.target]
            values, scales = quantize(original) if self.spec == "fp4" else quantize(original, self.spec)
            packed_name = q.name + "_" + self.spec + "_slots"
        else:
            values, scales = weights.parameters[q.target], weights.parameters[scale.target]
            packed_name = q.name + "_" + scale.name + "_" + self.spec + "_slots"
        payload = pack_slots(values, scales) if self.spec == "fp4" else pack_slots(values, scales, self.spec)
        packed = TensorSpec(packed_name, (len(payload),), "synthetic")
        # A previous BF16 constant need not end on a DMA beat boundary.
        if packed.name not in weights.records:
            padding = (-weights.cursor) % 16
            weights.data.extend(bytes(padding))
            weights.cursor += padding
            weights.use_payload(packed, payload, "uint8_t")
        return LoweredNode((self.output(node.outputs[0], (x.shape[0], n)),),
                           "linear_" + self.spec, (x.name,), {"packed": packed.name, "n": str(n)},
                           dram_preferred=(x.name, node.outputs[0]) if x.shape[0] > 1 else ())


class LinearMicroscaleLowerer(OpLowerer):
    def __init__(self, spec):
        self.spec = spec

    def lower(self, node, tensors, weights):
        from nnc.microscale import group_k, quantize, pack_slots, global_bits
        size = group_k(self.spec)
        explicit = 3 if self.spec == 'mxfp4' else 4
        if len(node.args) not in (2, explicit):
            raise SystemExit('microscale Linear requires static weight or explicit codes/scales/global')
        x, w = (tensors.require(name) for name in node.args[:2])
        if len(x.shape) != 2 or len(w.shape) != 2:
            raise SystemExit('microscale Linear requires rank-two operands')
        n,k = w.shape
        slot_bytes = 272 if self.spec == 'mxfp4' else 144
        output_bytes = 2
        if min(n,k,x.shape[0]) <= 0 or n%8 or k%size or x.shape[1] != k:
            raise SystemExit('microscale Linear shape/group mismatch')
        if ((2*k+63)//64*64 + (output_bytes*n+63)//64*64 + 128 + 8*slot_bytes) > 32768:
            raise SystemExit('microscale Linear operands exceed DTCM scratch capacity')
        params = [tensors.require(name) for name in node.args[1:]]
        if any(p.target not in weights.parameters for p in params):
            raise SystemExit('microscale weights/scales must be static')
        if len(node.args) == 2:
            q,scale,global_scale = quantize(weights.parameters[w.target],self.spec)
        else:
            q,scale = (weights.parameters[p.target] for p in params[:2])
            global_scale = (torch.tensor(1.,dtype=torch.float32) if self.spec == 'mxfp4' else
                            weights.parameters[params[2].target])
        payload = pack_slots(q,scale,self.spec)
        packed = TensorSpec('_'.join(node.args[1:])+'_'+self.spec+'_slots',(len(payload),),'synthetic')
        if packed.name not in weights.records:
            padding = (-weights.cursor)%16
            weights.data.extend(bytes(padding)); weights.cursor += padding
            weights.use_payload(packed,payload,'uint8_t')
        return LoweredNode((self.output(node.outputs[0],(x.shape[0],n)),),
            'linear_'+self.spec,(x.name,),{'packed':packed.name,'n':str(n),
            'global_bits':f'0x{global_bits(global_scale):08x}u'},
            dram_preferred=(x.name,node.outputs[0]) if x.shape[0] > 1 else ())


class LinearFP4Lowerer(LinearQuantLowerer):
    def __init__(self):
        super().__init__("fp4")


class RopeLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        xq = self.static_or_dynamic(tensors, node.args[0])
        xk = self.static_or_dynamic(tensors, node.args[1])
        start_pos = self.static_or_dynamic(tensors, node.args[2])
        del start_pos
        rope_table = weights.use_rope_table("rope_table", max(64, xq.shape[0]), xq.shape[-1])
        return LoweredNode(
            (self.output(node.outputs[0], xq.shape), self.output(node.outputs[1], xk.shape)),
            "rope",
            node.args,
            {"rope_table": rope_table.name, **node.attrs},
        )


class KvCacheLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        del weights
        xk = self.static_or_dynamic(tensors, node.args[0])
        xv = self.static_or_dynamic(tensors, node.args[1])
        return LoweredNode(
            (self.output(node.outputs[0], xk.shape), self.output(node.outputs[1], xv.shape)),
            "kv_cache_update",
            node.args,
            node.attrs,
            dram_preferred=node.outputs,
        )


class AttentionLowerer(OpLowerer):
    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        xq = self.static_or_dynamic(tensors, node.args[0])
        head_count = int(node.attrs.get("head_count", "1"))
        if head_count <= 0 or xq.shape[-1] % head_count != 0:
            raise SystemExit(
                f"attention width {xq.shape[-1]} is not divisible by {head_count} heads"
            )
        if (xq.shape[-1] // head_count) % 8 != 0:
            raise SystemExit("attention head dimension must be a multiple of 8")
        eye = weights.use_eye8("tensor_eye8")
        return LoweredNode(
            (self.output(node.outputs[0], xq.shape),),
            "attention",
            node.args,
            {"eye": eye.name, **node.attrs},
            dram_preferred=node.args[:4] + node.outputs,
        )


class MatmulLowerer(OpLowerer):
    def __init__(self, rhs_transposed: bool) -> None:
        self.rhs_transposed = rhs_transposed

    def lower(self, node: GraphNode, tensors: TensorTable, weights: WeightStore) -> LoweredNode:
        del weights
        lhs = self.static_or_dynamic(tensors, node.args[0])
        rhs = self.static_or_dynamic(tensors, node.args[1])
        head_count = int(node.attrs.get("head_count", "1"))
        if head_count <= 0:
            raise SystemExit("matmul head_count must be positive")
        if len(rhs.shape) != 2 or len(lhs.shape) not in ((2,) if self.rhs_transposed else (2, 3)):
            raise SystemExit("matmul requires rank-2 RHS and rank-2 or explicit-head rank-3 LHS")
        if min(*lhs.shape, *rhs.shape) <= 0:
            raise SystemExit("matmul dimensions must be positive")
        if self.rhs_transposed:
            if lhs.shape[1] % head_count or rhs.shape[1] != lhs.shape[1]:
                raise SystemExit("matmul_transpose requires matching widths divisible by head_count")
            rows, k, n = lhs.shape[0], lhs.shape[1] // head_count, rhs.shape[0]
            out_shape = (head_count, rows, n)
        else:
            if rhs.shape[1] % head_count:
                raise SystemExit("matmul output width must be divisible by head_count")
            if len(lhs.shape) == 3 and lhs.shape[0] != head_count:
                raise SystemExit("matmul LHS head axis must match head_count")
            rows, k, n = lhs.shape[-2], lhs.shape[-1], rhs.shape[1] // head_count
            if rhs.shape[0] != k:
                raise SystemExit("matmul K dimensions must match")
            out_shape = (rows, rhs.shape[1])
        if k % 8 or n % 8:
            raise SystemExit("matmul requires K%8=0 and per-head N%8=0")
        if 2 * head_count * (rows * n + rows * k + k * n) > 32768:
            raise SystemExit("matmul operands exceed 32 KiB DTCM scratch capacity")
        return LoweredNode(
            (self.output(node.outputs[0], out_shape),),
            "matmul_transpose" if self.rhs_transposed else "matmul",
            node.args,
            node.attrs,
            dram_preferred=(
                node.args[:2] + node.outputs
            ),
        )


OP_LOWERERS: dict[str, OpLowerer] = {
    "rms_norm": RmsNormLowerer(),
    "linear": LinearLowerer(),
    "linear_fp4": LinearFP4Lowerer(),
    "linear_mxfp4": LinearMicroscaleLowerer('mxfp4'),
    "linear_nvfp4": LinearMicroscaleLowerer('nvfp4'),
    "linear_q8_0": LinearQuantLowerer("q8_0"),
    "linear_fp8_e4m3fn": LinearQuantLowerer("fp8_e4m3fn"),
    "linear_fp8_e5m2": LinearQuantLowerer("fp8_e5m2"),
    "rope": RopeLowerer(),
    "kv_cache_update": KvCacheLowerer(),
    "attention": AttentionLowerer(),
    "matmul": MatmulLowerer(False),
    "matmul_transpose": MatmulLowerer(True),
    "softmax": SoftmaxLowerer(),
    "add": AddLowerer(),
    "mul": MulLowerer(),
    "sigmoid": UnarySameShapeLowerer("sigmoid"),
    "silu": UnarySameShapeLowerer("silu"),
    "tanh": UnarySameShapeLowerer("tanh"),
}


class Lowerer:
    def __init__(self, graph: Graph, parameters: dict[str, torch.Tensor]) -> None:
        self.graph = graph
        self.tensors = TensorTable()
        self.weights = WeightStore(parameters)
        self.nodes: list[LoweredNode] = []

    def lower(self) -> LoweredProgram:
        for spec in self.graph.static_tensors:
            self.tensors.add(spec)

        for node in self.graph.nodes:
            lowered = OP_LOWERERS[node.op].lower(node, self.tensors, self.weights)
            for output in lowered.outputs:
                self.tensors.add(output)
            self.nodes.append(lowered)

        return LoweredProgram(tuple(self.nodes), self.graph.output)
