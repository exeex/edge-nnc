"""Exact Q8_0 Linear check before the separately rounded BF16 bias add."""
from nnc.test.smoke_linear_q8_0 import Model as Original, make_example_args
from nnc.quant import linear_q8_0


class Model(Original):
    def forward(self,x):
        return linear_q8_0(x,self.linear.weight)


def make_model_and_args():
    return Model().eval(),make_example_args()
