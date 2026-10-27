"""Reusable transformer modules preserve named export boundaries and semantics."""
from pathlib import Path
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F
import nnc
from nnc import compiler, Linear, RMSNorm, RoPE, KVCacheUpdate, Attention


class TransformerModulesTest(unittest.TestCase):
    def assert_named_export(self, module, args, name):
        exported = torch.export.export(module,args)
        custom = [node for node in exported.graph.nodes if node.op == 'call_function'
                  and str(node.target).startswith('nnedge.')]
        self.assertEqual(len(custom),1)
        self.assertEqual(str(custom[0].target),'nnedge.'+name+'.default')
        return custom[0]

    def test_linear_preserves_source_parameters_and_optional_bias(self):
        for bias in (False,True):
            layer = torch.nn.Linear(16,8,bias=bias,dtype=torch.bfloat16)
            module = Linear(layer)
            x = torch.randn(2,16,dtype=torch.bfloat16)
            self.assertIs(module.weight,layer.weight)
            self.assertIs(module.bias,layer.bias)
            expected = F.linear(x,layer.weight)
            if bias: expected = expected + layer.bias
            self.assertTrue(torch.equal(module(x),expected))
            node = self.assert_named_export(module,(x,),'linear')
            self.assertEqual(node.meta['val'].dtype,torch.bfloat16)
            program,_ = compiler.compile_model(module,(x,))
            self.assertEqual([n.op for n in program.nodes],['linear','add'] if bias else ['linear'])
        with self.assertRaises(TypeError): Linear(torch.nn.ReLU())

    def test_rms_norm_bf16_metadata_and_reference(self):
        module = RMSNorm(8,eps=1e-5)
        x = torch.arange(1,17).reshape(2,8).to(torch.bfloat16)
        expected = (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-5)).to(x.dtype)
        self.assertTrue(torch.equal(module(x),expected))
        node = self.assert_named_export(module,(x,),'rms_norm')
        self.assertEqual(node.meta['val'].dtype,torch.bfloat16)

    def test_rope_and_kv_shapes_and_no_alias(self):
        q = torch.randn(2,8,dtype=torch.bfloat16)
        k = torch.randn(2,8,dtype=torch.bfloat16)
        pos = torch.tensor(0)
        qr,kr = RoPE()(q,k,pos)
        self.assertTrue(torch.equal(qr[0],q[0]))
        self.assertTrue(torch.equal(kr[0],k[0]))
        node = self.assert_named_export(RoPE(),(q,k,pos),'rope')
        self.assertEqual([v.dtype for v in node.meta['val']],[torch.bfloat16]*2)
        keys,values = KVCacheUpdate()(q,k,pos)
        self.assertTrue(torch.equal(keys,q))
        self.assertTrue(torch.equal(values,k))
        self.assertNotEqual(keys.data_ptr(),q.data_ptr())
        self.assertNotEqual(values.data_ptr(),k.data_ptr())
        self.assert_named_export(KVCacheUpdate(),(q,k,pos),'kv_cache_update')

    def test_attention_matches_unscaled_two_head_reference(self):
        q = torch.randn(2,16,dtype=torch.bfloat16)/8
        k = torch.randn(3,16,dtype=torch.bfloat16)/8
        v = torch.randn(3,16,dtype=torch.bfloat16)/8
        params = torch.zeros(1)
        expected = torch.cat([torch.softmax((q[:,h:h+8]@k[:,h:h+8].T).float(),-1).to(q.dtype)@v[:,h:h+8]
                              for h in (0,8)],dim=1)
        self.assertTrue(torch.equal(Attention(2)(q,k,v,params),expected))
        node = self.assert_named_export(Attention(2),(q,k,v,params),'attention')
        self.assertEqual(node.meta['val'].dtype,torch.bfloat16)
        with self.assertRaises(ValueError): Attention(0)

    def test_llama_imports_modules_and_keeps_one_source(self):
        root = Path(__file__).resolve().parents[1]
        path = root/'example/llama/model.py'
        self.assertEqual((root/'nnc/test/llama3_source.py').resolve(),path)
        self.assertEqual((root/'example/llama/model/llama3_source.py').resolve(),path)
        self.assertNotIn('@torch.library.custom_op',path.read_text())
        self.assertNotIn('torch.ops.nnedge',path.read_text())
        model,args = compiler.load_model_example(path)
        program,_ = compiler.compile_model(model,args)
        self.assertEqual([n.op for n in program.nodes],
                         ['rms_norm','linear','linear','linear','rope','kv_cache_update',
                          'attention','linear','add','rms_norm','linear','linear','silu','mul','linear','add'])
        types = {type(m) for m in model.modules()}
        self.assertTrue({Linear,RMSNorm,RoPE,KVCacheUpdate,Attention} <= types)

    def test_prefix_debug_model_uses_wrapped_projections(self):
        root = Path(__file__).resolve().parents[1]
        model,args = compiler.load_model_example(root/'nnc/test/smoke_transformer_prefix.py')
        for stage in ('w1','w3','silu','gated','ffn','full'):
            model.stage = stage
            with torch.no_grad(): output = model(*args)
            self.assertEqual(output.shape,(32,64 if stage in ('ffn','full') else 192))
        with torch.no_grad(): expected = model.block(*args)
        self.assertTrue(torch.equal(output,expected))

    def test_transformer_registration_is_lazy_for_legacy_models(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable,'-c',
                                 "import nnc,sys; assert 'nnc.ops' not in sys.modules; "
                                 "from nnc import RoPE; assert 'nnc.ops' in sys.modules"],
                                cwd=root,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__ == '__main__':
    unittest.main()
