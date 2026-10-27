"""Q8_0 native FP16 scale: single, circular wraps and public layout."""
import torch
import nnc


def make_model_and_args():
    torch.manual_seed(13)
    model = nnc.LinearQ8_0(torch.nn.Linear(32,8,bias=False)).eval()
    x = ((torch.arange(1*32).reshape(1,32)%17-8).float()/32).to(torch.bfloat16)
    return model,(x,)
