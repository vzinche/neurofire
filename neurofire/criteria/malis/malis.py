from torch.autograd import Function

from malis_impl import malis_impl


# Inherit from Function:
# https://github.com/pytorch/pytorch/blob/master/torch/autograd/function.py#L123
class MalisLoss(Function):
    """
    Malis loss
    """

    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
