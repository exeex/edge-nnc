"""Named quant spec packing, independent decoding, and compiler integration."""
import math
import tempfile
import nnc
from pathlib import Path
import struct
import unittest

import torch
from nnc import compiler
from nnc.quant import pack_slots, quantize, quantize_source, linear_q8_0

SPECS = ('q8_0', 'fp8_e4m3fn', 'fp8_e5m2')


def decode_byte(byte, spec):
    if spec == 'q8_0':
        return byte if byte < 128 else byte - 256
    fraction_bits, bias = (3, 7) if spec == 'fp8_e4m3fn' else (2, 15)
    fraction = byte & ((1 << fraction_bits) - 1)
    exponent = (byte & 127) >> fraction_bits
    magnitude = (fraction / (1 << fraction_bits) * 2.**(1-bias) if exponent == 0
                 else (1 + fraction / (1 << fraction_bits)) * 2.**(exponent-bias))
    return -magnitude if byte & 128 else magnitude


class QuantTest(unittest.TestCase):
    def test_q8_reference_codes_precede_scale_adaptation(self):
        w=torch.zeros(8,64)
        below_half=torch.nextafter(torch.tensor(.5),torch.tensor(0.)).item()
        w[0,:7]=torch.tensor([127.,.5,-.5,1.5,-1.5,below_half,-below_half])
        # The second group has a scale that FP16 -> BF16 changes. Quant codes
        # must still come from the FP32 normalization, not the adapted scale.
        w[1]=torch.linspace(-1.2345,1.2345,64)
        q,source=quantize_source(w,'q8_0')
        actual,scale=quantize(w,'q8_0')
        self.assertEqual(q[0,:7].tolist(),[127.,1.,-1.,2.,-2.,0.,0.])
        self.assertEqual(source.dtype,torch.float16)
        self.assertEqual(float(source[2,0]),0.)
        self.assertTrue(torch.equal(actual,q))
        self.assertTrue(torch.equal(scale,source.to(torch.bfloat16)))
        self.assertGreater(float((source.float()-scale.float()).abs().max()),0.)
        # Independently check the source policy for non-tie values too.
        for row in range(8):
            for group in range(2):
                values=w[row,group*32:(group+1)*32].tolist()
                delta=max(abs(v) for v in values)/127
                expected=[math.copysign(math.floor(abs(v*(1/delta))+.5),v)
                          if delta else 0 for v in values]
                self.assertEqual(q[row,group*32:(group+1)*32].tolist(),expected)

    def test_all_finite_codes_and_slot_order(self):
        n, k = 16, 64
        scale = (torch.arange(n*2).reshape(n,2)+1).to(torch.bfloat16)/8
        for spec in SPECS:
            with self.subTest(spec=spec):
                valid = [b for b in range(256) if
                         spec == 'q8_0' or
                         (spec == 'fp8_e4m3fn' and b & 127 != 127) or
                         (spec == 'fp8_e5m2' and b & 124 != 124)]
                codes = torch.tensor([valid[i % len(valid)] for i in range(n*k)],
                                     dtype=torch.uint8).reshape(n,k)
                q = torch.tensor([[decode_byte(int(b),spec) for b in row] for row in codes],
                                 dtype=torch.bfloat16)
                payload = pack_slots(q,scale,spec)
                self.assertEqual(len(payload), 4*272)
                for ob in range(n//8):
                    for group in range(k//32):
                        slot = payload[(ob*2+group)*272:(ob*2+group+1)*272]
                        decoded_scale = torch.frombuffer(bytearray(slot[:16]),dtype=torch.bfloat16)
                        self.assertTrue(torch.equal(decoded_scale, scale[ob*8:ob*8+8,group]))
                        for tile in range(4):
                            raw = slot[16+tile*64:80+tile*64]
                            expected = codes[ob*8:ob*8+8,group*32+tile*8:group*32+tile*8+8]
                            # Exact byte comparison includes both signed zeros.
                            self.assertEqual(raw, bytes(expected.flatten().tolist()))
                            self.assertTrue(all(math.isfinite(decode_byte(b,spec)) for b in raw))

    def test_quantization_zero_and_finite_groups(self):
        for spec in SPECS:
            with self.subTest(spec=spec):
                w = (torch.arange(8*64).reshape(8,64)%41-20).float()/32
                w[0,:32] = 0
                q,s = quantize(w,spec)
                self.assertEqual(q.dtype,torch.bfloat16)
                self.assertEqual(s.dtype,torch.bfloat16)
                self.assertEqual(float(s[0,0]),0. if spec == 'q8_0' else 1.)
                self.assertTrue(torch.equal(q[0,:32],torch.zeros(32)))
                self.assertTrue(torch.isfinite(q).all())
                self.assertEqual(len(pack_slots(q,s,spec)),2*272)
                for bad in (float('nan'),float('inf')):
                    invalid = w.clone()
                    invalid[0,0] = bad
                    with self.assertRaises(ValueError): quantize(invalid,spec)

    def test_reject_invalid_carriers_and_scales(self):
        for spec in SPECS:
            q = torch.zeros(8,32,dtype=torch.bfloat16)
            s = torch.ones(8,1,dtype=torch.bfloat16)
            bad_values = (.5,128.,-129.) if spec == 'q8_0' else (.1,float('inf'))
            for value in bad_values:
                invalid = q.clone()
                invalid[0,0] = value
                with self.subTest(spec=spec,value=value), self.assertRaises(ValueError):
                    pack_slots(invalid,s,spec)
            self.assertEqual(len(pack_slots(q,torch.zeros_like(s),spec)),272)
            for value in (-1.,float('nan'),float('inf')):
                with self.assertRaises(ValueError):
                    pack_slots(q,torch.full_like(s,value),spec)
            with self.assertRaises(ValueError): pack_slots(q.float(),s,spec)
            with self.assertRaises(ValueError): pack_slots(q,s.float(),spec)
            with self.assertRaises(ValueError): pack_slots(q,s.repeat(1,2),spec)
            with self.assertRaises(ValueError): quantize(q[:,:31],spec)

    def test_named_codegen_and_payload_sizes(self):
        root = Path(__file__).resolve().parents[1]
        for spec in ('fp4',*SPECS):
            with self.subTest(spec=spec):
                model,args = compiler.load_model_example(root/'example'/('linear_'+spec)/'model.py')
                program,weights = compiler.compile_model(model,args)
                node = program.nodes[0]
                self.assertEqual(node.op,'linear_'+spec)
                self.assertEqual(node.outputs[0].shape,(9,16))
                record = weights.records[node.attrs['packed']]
                self.assertEqual(record.nbytes,4*(144 if spec == 'fp4' else 272))
                self.assertEqual(program.nodes[-1].op,'add')
                self.assertEqual(record.offset%16,0)
                self.assertEqual(record.ctype,'uint8_t')
                abi = compiler.ForwardABI.from_example(model,args)
                code = compiler.ForwardRenderer(program,abi,weights).render()
                self.assertIn('nnedge::op::linear_'+spec+'(',code)
                self.assertEqual(len(node.dram_preferred),2)
                with self.assertRaises(SystemExit):
                    compiler.compile_model(model,(args[0][:,:32],))

    def test_modules_replace_linear_and_export_named_ops(self):
        classes = (nnc.LinearFp4, nnc.LinearQ8_0, nnc.LinearFp8E4M3FN, nnc.LinearFp8E5M2)
        for cls,spec in zip(classes,('fp4',*SPECS)):
            for bias in (False,True):
                with self.subTest(spec=spec,bias=bias):
                    original = torch.nn.Linear(64,16,bias=bias)
                    module = cls(original).eval()
                    x = torch.ones(9,64,dtype=torch.bfloat16)/8
                    self.assertEqual(original.weight.dtype,torch.float32)
                    self.assertEqual(module.weight.dtype,torch.bfloat16)
                    self.assertTrue(torch.equal(module.weight,original.weight.to(torch.bfloat16)))
                    self.assertNotEqual(module.weight.data_ptr(),original.weight.data_ptr())
                    output = module(x)
                    self.assertEqual(output.shape,(9,16))
                    self.assertEqual(output.dtype,torch.bfloat16)
                    exported = torch.export.export(module,(x,))
                    custom = [n for n in exported.graph.nodes if n.op == 'call_function'
                              and str(n.target).startswith('nnedge.linear_')]
                    self.assertEqual(len(custom),1)
                    self.assertEqual(str(custom[0].target),'nnedge.linear_'+spec+'.default')
                    self.assertEqual(custom[0].meta['val'].dtype,torch.bfloat16)
                    program,weights = compiler.compile_model(module,(x,))
                    self.assertEqual(program.nodes[0].op,'linear_'+spec)
                    self.assertEqual(len(program.nodes),2 if bias else 1)
                    record = weights.records[program.nodes[0].attrs['packed']]
                    if spec == 'fp4':
                        from nnc.fp4 import quantize as quantize_fp4, pack_slots as pack_fp4
                        q,s = quantize_fp4(module.weight)
                        expected = pack_fp4(q,s)
                    else:
                        q,s = quantize(module.weight,spec)
                        expected = pack_slots(q,s,spec)
                    self.assertEqual(bytes(weights.data[record.offset:record.offset+record.nbytes]),expected)
        with self.assertRaises(TypeError): nnc.LinearFp4(torch.nn.ReLU())
        with self.assertRaises(ValueError): nnc.LinearFp4(torch.nn.Linear(31,16))

    def test_public_compile_writes_cpp_and_payloads(self):
        module = nnc.LinearQ8_0(torch.nn.Linear(64,16)).eval()
        args = (torch.ones(1,64,dtype=torch.bfloat16),)
        with tempfile.TemporaryDirectory() as directory:
            destination = nnc.compile(module,args,out_dir=directory)
            self.assertEqual(destination,Path(directory))
            self.assertEqual({p.name for p in destination.iterdir()},
                             {'forward.hpp','init.hpp','input.bin','weight.bin'})
            self.assertIn('nnedge::op::linear_q8_0(', (destination/'forward.hpp').read_text())
            self.assertEqual(len((destination/'input.bin').read_bytes()),128)
            nnc.compile(module,args,out_dir=directory,save_reference=True)
            self.assertEqual(len((destination/'expected.bin').read_bytes()),32)
            expected = module(*args).view(torch.int16).flatten().tolist()
            self.assertEqual((destination/'expected.bin').read_bytes(),
                             struct.pack('<16H',*(v & 0xffff for v in expected)))

    def test_reference_applies_each_group_scale_before_sum(self):
        x = torch.ones(9,64,dtype=torch.bfloat16)
        q = torch.cat((torch.ones(8,32),torch.full((8,32),-2.)),dim=1).to(torch.bfloat16)
        scale = torch.tensor([[.5,2.]],dtype=torch.bfloat16).repeat(8,1)
        self.assertTrue(torch.equal(linear_q8_0(x,q,scale),torch.full((9,8),-112.,dtype=torch.bfloat16)))


if __name__ == '__main__':
    unittest.main()
