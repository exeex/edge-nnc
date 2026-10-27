"""Q8_0 native FP16 scale: chunks, circular wraps and public layout."""
import torch
import nnc


def make_model_and_args():
    torch.manual_seed(13)
    model = nnc.LinearQ8_0(torch.nn.Linear(96,24,bias=False)).eval()
    x = ((torch.arange(65*96).reshape(65,96)%17-8).float()/32).to(torch.bfloat16)
    return model,(x,)
