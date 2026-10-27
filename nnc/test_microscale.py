import math
import struct
import unittest
import torch
from nnc import compiler
from nnc.fp4 import encode_bf16
from nnc.microscale import decode_scales, group_k, quantize, pack_slots, reference, global_bits, rounded_global, combined_scales, tensor_dot8
from nnc.test.smoke_linear_mxfp4 import make_case


class MicroscaleTest(unittest.TestCase):
    def test_all_scale_codes_against_numeric_definition(self):
        for spec in ('mxfp4','nvfp4'):
            for code in range(256):
                invalid = code==255 if spec=='mxfp4' else (code&127)==127 or bool(code&128 and code&127)
                if invalid:
                    with self.assertRaises(ValueError): decode_scales(torch.tensor([code],dtype=torch.uint8),spec)
                    continue
                actual = float(decode_scales(torch.tensor([code],dtype=torch.uint8),spec))
                if spec=='mxfp4': expected=math.ldexp(1.,code-127)
                else:
                    exponent,fraction=(code>>3)&15,code&7
                    expected=math.ldexp(fraction/8 if exponent==0 else 1+fraction/8,-6 if exponent==0 else exponent-7)
                self.assertEqual(actual,expected)

    def test_original_codes_repack_and_group_size(self):
        for spec in ('mxfp4','nvfp4'):
            size=group_k(spec); n,k=16,2*size
            values=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
            q=values[(torch.arange(n*k)%8)].reshape(n,k).to(torch.bfloat16)
            scale=torch.full((n,2),127 if spec=='mxfp4' else 0x38,dtype=torch.uint8)
            payload=pack_slots(q,scale,spec)
            stride=136 if spec=='mxfp4' else 72
            self.assertEqual(len(payload),4*stride)
            codes=encode_bf16(q)
            for ob in range(2):
                for group in range(2):
                    slot=payload[(ob*2+group)*stride:(ob*2+group+1)*stride]
                    self.assertEqual(slot[:8],bytes(scale[ob*8:ob*8+8,group].tolist()))
                    self.assertEqual(len(slot),8+size*4)
                    for tile in range(size//8):
                        raw=slot[8+tile*32:40+tile*32]
                        unpacked=[nibble for byte in raw for nibble in (byte&15,byte>>4)]
                        self.assertEqual(unpacked,codes[ob*8:ob*8+8,group*size+tile*8:group*size+tile*8+8].flatten().tolist())

    def test_dense_sizes_with_single_final_transport_tail(self):
        for spec in ('mxfp4','nvfp4'):
            size=group_k(spec)
            for n,groups in ((8,1),(8,3),(24,1),(8,32)):
                model,_=make_case(spec,width=size*groups,outputs=n)
                q,scale,_=quantize(model.weight,spec)
                payload=pack_slots(q,scale,spec)
                native=n//8*groups*(8+size*4)
                self.assertEqual(len(payload),(native+15)//16*16)
                self.assertLess(len(payload)-native,16)
                self.assertEqual(payload[native:],bytes(len(payload)-native))

    def test_bf16_global_combined_scale_and_validation(self):
        x=torch.ones(1,16,dtype=torch.bfloat16)
        q=torch.ones(8,16,dtype=torch.bfloat16)
        codes=torch.full((8,1),0x38,dtype=torch.uint8)
        global_scale=torch.tensor(.1234567)
        self.assertTrue(torch.equal(reference(x,q,codes,global_scale,'nvfp4'),
                                    (torch.full((1,8),16.)*rounded_global(global_scale)).to(torch.bfloat16)))
        self.assertEqual(global_bits(global_scale),struct.unpack('<I',struct.pack('<f',float(rounded_global(global_scale))))[0])
        self.assertEqual(global_bits(global_scale) & 0xffff, 0)
        # A nontrivial BF16 x E4M3 product uses the low BF20 fraction bits.
        scales = combined_scales(torch.tensor([0x3f],dtype=torch.uint8), torch.tensor(1.9921875))
        self.assertEqual(float(scales[0]), 1.875*1.9921875)
        raw = scales.view(torch.int32)
        self.assertEqual(int(raw[0]) & 0xfff, 0)
        self.assertNotEqual(int(raw[0]) & 0xf000, 0)
        for value in (0.,-1.,math.nan,math.inf):
            with self.assertRaises(ValueError): global_bits(torch.tensor(value))

    def test_dot8_shared_exponent_alignment(self):
        x=torch.zeros(2,8,dtype=torch.bfloat16)
        x[:,0]=1;x[:,1]=-1;x[0,2]=2**-15;x[1,2]=3*2**-16
        q=torch.ones(1,8,dtype=torch.bfloat16)
        self.assertTrue(torch.equal(tensor_dot8(x,q),torch.tensor([[0.],[2**-14]])))

    def test_zero_groups_codegen_and_original_packed_payload(self):
        for spec in ('mxfp4','nvfp4'):
            width=64 if spec=='mxfp4' else 48
            model,args=make_case(spec,width=width)
            q,scale,global_scale=quantize(model.weight,spec)
            self.assertEqual(scale.dtype,torch.uint8)
            program,weights=compiler.compile_model(model,args)
            node=program.nodes[0]
            self.assertEqual(node.op,'linear_'+spec)
            record=weights.records[node.attrs['packed']]
            self.assertEqual(bytes(weights.data[record.offset:record.offset+record.nbytes]),pack_slots(q,scale,spec))
            code=compiler.ForwardRenderer(program,compiler.ForwardABI.from_example(model,args),weights).render()
            self.assertIn('nnedge::op::linear_'+spec+'<64, true>(',code)
            if spec=='nvfp4':self.assertIn(f'0x{global_bits(global_scale):08x}u',code)
            zero,_,_=quantize(torch.zeros(8,width),spec)
            self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))

    def test_static_checkpoint_codes_bypass_requantization(self):
        from nnc.microscale import linear_mxfp4,linear_nvfp4
        class Imported(torch.nn.Module):
            def __init__(self,spec):
                super().__init__(); self.spec=spec
                size=group_k(spec)
                self.q=torch.nn.Parameter(torch.full((8,size),1.5,dtype=torch.bfloat16),requires_grad=False)
                self.register_buffer('scales',torch.full((8,1),126 if spec=='mxfp4' else 0x34,dtype=torch.uint8))
                self.register_buffer('global_scale',torch.tensor(.1234567))
            def forward(self,x):
                return (linear_mxfp4(x,self.q,self.scales) if self.spec=='mxfp4' else
                        linear_nvfp4(x,self.q,self.scales,self.global_scale))
        for spec in ('mxfp4','nvfp4'):
            model=Imported(spec)
            args=(torch.ones(1,group_k(spec),dtype=torch.bfloat16),)
            program,weights=compiler.compile_model(model,args)
            node=program.nodes[0]; record=weights.records[node.attrs['packed']]
            self.assertEqual(bytes(weights.data[record.offset:record.offset+record.nbytes]),
                             pack_slots(model.q,model.scales,spec))
            if spec=='nvfp4':self.assertEqual(node.attrs['global_bits'],f'0x{global_bits(model.global_scale):08x}u')


if __name__=='__main__':unittest.main()
