"""Different global scales around an MXFP4 op; contiguous BF16 op boundaries."""
import torch
import nnc


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        modules=[]
        for index,cls in enumerate((nnc.LinearNvFp4,nnc.LinearMxFp4,nnc.LinearNvFp4)):
            linear=torch.nn.Linear(32,32,bias=False)
            with torch.no_grad():
                weight=((torch.arange(32*32).reshape(32,32)*7)%29-14).float()/128
                linear.weight.copy_(weight*(index+1)/4)
            modules.append(cls(linear))
        self.layers=torch.nn.Sequential(*modules)

    def forward(self,x):
        return self.layers(x)


def make_model_and_args():
    x=((torch.arange(9*32).reshape(9,32)%13-6).float()/16).to(torch.bfloat16)
    return Model().eval(),(x,)
