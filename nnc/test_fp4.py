import struct
import unittest
import torch
from nnc.fp4 import quantize, encode_bf16, pack_slots, VALUES
from nnc import compiler

class FP4Test(unittest.TestCase):
    def test_all_codes_and_sign_zero(self):
        values = torch.tensor(VALUES + tuple(-v for v in VALUES), dtype=torch.bfloat16)
        self.assertEqual(encode_bf16(values).tolist(), list(range(16)))
        with self.assertRaises(ValueError):
            encode_bf16(torch.tensor([.25], dtype=torch.bfloat16))

    def test_rounding_scale_and_zero_group(self):
        w = torch.zeros(8,32)
        w[0,:8] = torch.tensor([6., .25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        q,s = quantize(w)
        self.assertEqual(q[0,:8].tolist(), [6.,0.,.5,1.,1.5,2.,3.,4.])
        self.assertTrue(torch.equal(s, torch.ones_like(s)))
        self.assertEqual(q.dtype, torch.bfloat16)
        for bad in [float('nan'), float('inf')]:
            w[1,0] = bad
            with self.assertRaises(ValueError): quantize(w)

    def test_slot_roundtrip_independent_decoder(self):
        n,k = 16,64
        code = torch.arange(n*k).reshape(n,k) % 16
        table = torch.tensor(VALUES + tuple(-v for v in VALUES), dtype=torch.bfloat16)
        q = table[code]
        s = (torch.arange(n*2).reshape(n,2)+1).to(torch.bfloat16)/8
        blob = pack_slots(q,s)
        self.assertEqual(len(blob), 4*144)
        for ob in range(2):
            for g in range(2):
                slot = blob[(ob*2+g)*144:(ob*2+g+1)*144]
                decoded_scales = torch.frombuffer(bytearray(slot[:16]),dtype=torch.bfloat16)
                self.assertTrue(torch.equal(decoded_scales,s[ob*8:ob*8+8,g]))
                for tile in range(4):
                    raw = slot[16+tile*32:48+tile*32]
                    decoded = [VALUES[v&7] * (-1 if v&8 else 1)
                               for byte in raw for v in (byte&15,byte>>4)]
                    expected = q[ob*8:ob*8+8,g*32+tile*8:g*32+tile*8+8]
                    self.assertEqual(decoded, expected.flatten().tolist())

    def test_bf22_partial_sums_keep_small_contributions(self):
        from nnc.test.smoke_linear_fp4_precision import make_model_and_args
        model,args = make_model_and_args()
        actual = model(*args)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 1.0078125)))
        narrow = torch.tensor(1., dtype=torch.bfloat16)
        for _ in range(31):
            narrow = (narrow.float() + 2**-12).to(torch.bfloat16)
        self.assertEqual(float(narrow), 1.)

    def test_compilation_and_reject_multi_token(self):
        from pathlib import Path
        model,args = compiler.load_model_example(Path(__file__).parent/'test/smoke_linear_fp4.py')
        program,weights = compiler.compile_model(model,args)
        self.assertEqual(len(weights.data), 12*144)
        self.assertEqual(program.nodes[-1].op,'linear_fp4')
        with self.assertRaises(SystemExit):
            compiler.compile_model(model,(args[0].repeat(2,1),))

if __name__ == '__main__': unittest.main()
