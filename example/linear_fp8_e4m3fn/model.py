"""BF16 [9,64] -> FP8_E4M3FN Linear -> BF16 [9,16]."""
from pathlib import Path
import torch
import nnc
from nnc import LinearFp8E4M3FN


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.linear = LinearFp8E4M3FN(torch.nn.Linear(64, 16))

    def forward(self, x):
        return self.linear(x)


def make_example_args():
    x = ((torch.arange(9*64).reshape(9,64) % 13 - 6).float() / 16).to(torch.bfloat16)
    return (x,)


def make_model_and_args():
    return Model().eval(), make_example_args()


if __name__ == "__main__":
    m = Model().eval()
    nnc.compile(m, make_example_args(), out_dir=Path(__file__).parent, save_reference=True)
