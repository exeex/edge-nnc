"""Native byte-scale FP4 Linear, including nontrivial scale/channel groups."""
import torch
import nnc


def make_case(spec='mxfp4', rows=9, width=64, outputs=16):
    linear = torch.nn.Linear(width,outputs,bias=False)
    with torch.no_grad():
        values = ((torch.arange(outputs*width).reshape(outputs,width)*7)%41-20).float()/16
        values *= torch.arange(1,outputs+1).reshape(outputs,1)/32
        values[0,:16] = 0
        linear.weight.copy_(values)
    cls = nnc.LinearMxFp4 if spec == 'mxfp4' else nnc.LinearNvFp4
    model = cls(linear).eval()
    x = ((torch.arange(rows*width).reshape(rows,width)%13-6).float()/32).to(torch.bfloat16)
    return model,(x,)


def make_model_and_args():
    return make_case()
