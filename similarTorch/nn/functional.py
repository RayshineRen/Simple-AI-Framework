from similarTorch.autograd import Context
from .modules import Sigmoid

ctx = Context()


def sigmoid(x):
    return Sigmoid().forward(ctx, x)
