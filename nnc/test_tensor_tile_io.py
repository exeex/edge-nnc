"""Explicit preplanned IO ABI; default contiguous exports stay unchanged."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
import nnc
from nnc import compiler
from nnc.codegen import tensor_tile_bits


class TensorTileIOTests(unittest.TestCase):
    def test_exact_order_and_shape_rejection(self):
        self.assertEqual(tensor_tile_bits(tuple(range(32)),(2,16)),
                         tuple(range(8))+tuple(range(16,24))+tuple(range(8,16))+tuple(range(24,32)))
        for shape in ((2,7),(0,8),(2,8,1)):
            with self.assertRaises(ValueError):
                tensor_tile_bits((),shape)
        with self.assertRaises(ValueError):
            tensor_tile_bits((1,),(2,8))

    def test_export_payloads_forward_and_default_abi(self):
        model = nnc.LinearQ8_0(torch.nn.Linear(32,16,bias=False)).eval()
        x = torch.arange(9*32).reshape(9,32).to(torch.bfloat16)/32
        def raw(t):
            return t.contiguous().view(torch.uint16).numpy().tobytes()
        def packed(t):
            return raw(t.reshape(t.shape[0],-1,8).permute(1,0,2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            compiler.compile(model,(x,),out_dir=path,save_reference=True,
                             tensor_tile_io=True,quant_token_capacity=128)
            with torch.no_grad():
                expected = model(x)
            self.assertEqual((path/"input.bin").read_bytes(),packed(x))
            self.assertEqual((path/"expected.bin").read_bytes(),packed(expected))
            forward = (path/"forward.hpp").read_text()
            self.assertIn("linear_q8_0<128, true>(",forward)
            self.assertNotIn("malloc_tensor_dram",forward)
            self.assertNotIn("start_strided",forward)
            self.assertEqual(json.loads((path/"io_layout.json").read_text())["lanes"],8)
            compiler.compile(model,(x,),out_dir=path,save_reference=True)
            self.assertFalse((path/"io_layout.json").exists())
            self.assertEqual((path/"input.bin").read_bytes(),raw(x))
            self.assertEqual((path/"expected.bin").read_bytes(),raw(expected))
            self.assertNotIn("TensorTileView",(path/"forward.hpp").read_text())

    def test_incompatible_graph_and_large_sram_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for model,x in (
                (nnc.LinearQ8_0(torch.nn.Linear(32,16,bias=True)),torch.ones(2,32,dtype=torch.bfloat16)),
                (nnc.LinearQ8_0(torch.nn.Linear(32,16,bias=False)),torch.ones(2048,32,dtype=torch.bfloat16))):
                with self.assertRaises(ValueError):
                    compiler.compile(model,(x,),out_dir=directory,tensor_tile_io=True)

    def test_matmul_rhs_is_runtime_contiguous_bf16(self):
        from nnc.test.smoke_matmul import make_model_and_args as normal
        from nnc.test.smoke_matmul_transpose import make_model_and_args as transpose
        for fixture in (normal,transpose):
            model,args=fixture()
            x,rhs=(t.to(torch.bfloat16) for t in args)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)
                compiler.compile(model,(x,rhs),out_dir=path,save_reference=True,
                                 tensor_tile_io=True,matmul_token_capacity=128)
                payload=(path/'input.bin').read_bytes()
                offset=(x.numel()*2+63)//64*64
                self.assertEqual(payload[offset:offset+rhs.numel()*2],
                                 rhs.contiguous().view(torch.uint16).numpy().tobytes())
                self.assertEqual((path/'weight.bin').read_bytes(),b'')
                layout=json.loads((path/'io_layout.json').read_text())
                self.assertEqual(layout['input_layouts'],{'x':'tensor_tiles','rhs':'contiguous'})
                code=(path/'forward.hpp').read_text()
                self.assertIn('<128>(nnedge::TensorTileView(',code)
                self.assertNotIn('TensorTileView(rhs)',code)
                self.assertNotIn('rhs_dtcm',code)

    def test_headed_tile_order(self):
        x=torch.arange(2*9*16).reshape(2,9,16)
        want=tuple(x.reshape(2,9,2,8).permute(0,2,1,3).flatten().tolist())
        self.assertEqual(tensor_tile_bits(tuple(x.flatten().tolist()),tuple(x.shape)),want)


if __name__ == "__main__":
    unittest.main()
