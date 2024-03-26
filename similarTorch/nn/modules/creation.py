import numpy as np
from similarTorch.tensor import Tensor


def empty(shape, dtype=np.float32, requires_grad=False):
    return Tensor(np.empty(shape, dtype=dtype), requires_grad=requires_grad)


def empty_like(other, dtype=None, requires_grad=False):
    if isinstance(other, Tensor):
        other = other.data
    return Tensor(np.empty_like(other, dtype=dtype), requires_grad=requires_grad)


def ones(shape, dtype=np.float32, requires_grad=False):
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)


def ones_like(other, dtype=None, requires_grad=False):
    if isinstance(other, Tensor):
        other = other.data
    return Tensor(np.ones_like(other, dtype=dtype), requires_grad=requires_grad)


def zeros(shape, dtype=np.float32, requires_grad=False):
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)


def zeros_like(other, dtype=None, requires_grad=False):
    if isinstance(other, Tensor):
        other = other.data
    return Tensor(np.zeros_like(other, dtype=dtype), requires_grad=requires_grad)


def rands(shape, requires_grad=False):
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)
