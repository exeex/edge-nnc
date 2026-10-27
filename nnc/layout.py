"""Compiler-owned tensor storage and explicit layout boundaries.

Matmul always consumes/produces channel-block/token/lane storage. Runtime RHS
stays row-major in DRAM. Pointwise nodes preserve compatible storage; other
operators get explicit DMA layout edges, never a second Matmul implementation.
"""
from nnc.graph import LoweredNode, LoweredProgram, TensorSpec

TILED = 'tensor_tiles'
CONTIGUOUS = 'contiguous'
QUANT_OPS = {'linear_fp4', 'linear_nvfp4', 'linear_mxfp4', 'linear_q8_0',
             'linear_fp8_e4m3fn', 'linear_fp8_e5m2'}
MATMUL_OPS = {'matmul', 'matmul_transpose'}


def expand_attention(program, abi):
    shapes = {v.name: v.shape for v in abi.inputs if v.kind == 'tensor'}
    nodes = []
    for node in program.nodes:
        if node.op == 'attention':
            q, k, v = node.args[:3]
            heads = int(node.attrs['head_count'])
            shape = (heads, shapes[q][0], shapes[k][0])
            base = node.outputs[0].name
            score = TensorSpec(base+'_scores', shape, 'ssa')
            prob = TensorSpec(base+'_prob', shape, 'ssa')
            nodes.extend((
                LoweredNode((score,), 'matmul_transpose', (q,k), {'head_count': str(heads)}, (k,)),
                LoweredNode((prob,), 'softmax', (score.name,), {'eye': node.attrs['eye'], 'head_count': str(heads)}, (score.name,prob.name)),
                LoweredNode(node.outputs, 'matmul', (prob.name,v), {'head_count': str(heads)}, (v,)),
            ))
        else:
            nodes.append(node)
        shapes.update({v.name: v.shape for v in node.outputs})
    return LoweredProgram(tuple(nodes), program.output)


class LayoutPlan:
    def __init__(self, program, abi):
        self.program = expand_attention(program, abi)
        self.shapes = {v.name: v.shape for v in abi.inputs if v.kind == 'tensor'}
        self.shapes.update({v.name: v.shape for n in self.program.nodes for v in n.outputs})
        self.layouts = {name: CONTIGUOUS for name in self.shapes}
        self.native_quant = set()
        # Conservative arena bound includes every SSA allocation and potential
        # input conversion. This guarantees SRAM residency for native quant
        # kernels even before a future lifetime-aware memory planner is used.
        arena_bound = sum(self.bytes(shape) for shape in self.shapes.values())
        arena_bound += sum(self.bytes(self.shapes[name]) for n in self.program.nodes
                           for name in n.args if name in self.shapes)

        self.transparent = []
        # Seed hardware contracts. A large quant op keeps its chunked DRAM ABI;
        # Matmul's sole tiled-storage kernel supports both SRAM and DRAM chunks.
        for n in self.program.nodes:
            if n.op in MATMUL_OPS:
                for v in n.outputs: self.layouts[v.name] = TILED
            if n.op in QUANT_OPS:
                if arena_bound <= 96*1024:
                    self.native_quant.add(n.outputs[0].name)
                    for v in n.outputs: self.layouts[v.name] = TILED
            if n.op in {'sigmoid','silu','tanh','add','mul'}:
                names = [*n.args, *(v.name for v in n.outputs)]
                if (all(name in self.shapes for name in names) and
                    len({self.shapes[name] for name in names}) == 1 and
                    self.tileable(self.shapes[names[0]])):
                    self.transparent.append(names)
        producers = {v.name:n for n in self.program.nodes for v in n.outputs}
        # An input/pointwise producer can directly adopt a consumer's layout.
        seeds = [n.args[0] for n in self.program.nodes
                 if n.op in MATMUL_OPS or n.outputs[0].name in self.native_quant]
        mutable = {v.name for v in abi.inputs if v.kind == 'tensor'}
        mutable.update(name for group in self.transparent for name in group
                       if name not in producers or producers[name].op in {'sigmoid','silu','tanh','add','mul'})
        for name in seeds:
            if name in mutable: self.layouts[name] = TILED
        for _ in range(len(self.transparent)+1):
            for group in self.transparent:
                if any(self.layouts[name] == TILED for name in group):
                    for name in group:
                        if name in mutable: self.layouts[name] = TILED
        self.dram_names = {name for n in self.program.nodes for name in n.dram_preferred}
        # Matmul LHS/output no longer carry the old forced-DRAM preference.
        for n in self.program.nodes:
            if n.op in MATMUL_OPS or n.outputs[0].name in self.native_quant:
                self.dram_names.difference_update((n.args[0], *(v.name for v in n.outputs)))
        self.dram_names.difference_update(name for name,layout in self.layouts.items() if layout == TILED)
        self.dram_names.update(n.args[1] for n in self.program.nodes
                               if n.op in MATMUL_OPS and self.layouts[n.args[1]] == CONTIGUOUS)

    @staticmethod
    def bytes(shape):
        n = 2
        for d in shape: n *= d
        return (n+63)//64*64

    @staticmethod
    def tileable(shape):
        return len(shape) in (2,3) and min(shape)>0 and shape[-1]%8 == 0

    def wanted(self, node, index):
        if node.op in MATMUL_OPS: return TILED if index == 0 else CONTIGUOUS
        if node.outputs[0].name in self.native_quant: return TILED
        names = [*node.args, *(v.name for v in node.outputs)]
        if names in self.transparent: return self.layouts[node.outputs[0].name]
        return CONTIGUOUS
