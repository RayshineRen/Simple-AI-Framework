import numpy as np
from typing import Type
from .autograd import Autograd
from similarTorch.nn.modules.mathematical import Add, Sub, Multiply, MatMul
from similarTorch.nn.modules.manipulation import SwapAxes, Reshape


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

        self.backward_function = None
        self.backward_tensor = []
        # self.shape = self.data.shape  # AttributeError: can't set attribute

    def backward(self, grad=np.array([1])):
        if self.requires_grad:
            self.grad = grad + self.grad
            sum_ax = tuple(range(len(self.grad.shape) - len(self.shape)))
            self.grad = np.sum(self.grad, axis=sum_ax)
        if self.backward_function is not None:
            accumulated = self.backward_function(grad)
            if len(self.backward_tensor) == 1:
                accumulated = (accumulated,)
            """
            对于 backward_tensor 中的每个后续节点张量 bv 和对应的 accumulated，
            依次调用其 backward 方法，并传入对应的梯度信息 ac，实现梯度的传播。
            """
            for bv, ac in zip(self.backward_tensor, accumulated):
                bv.backward(ac)

    @classmethod
    def _op(cls, Op: Type[Autograd], *input_vars):
        f = Op()
        return f(*input_vars)

    def __str__(self):
        return "<Tensor>\n" + self.data.__str__()

    def __add__(self, other):
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self._op(Add, other, self)

    def __sub__(self, other):
        return self._op(Sub, self, other)

    def __rsub__(self, other):
        return self._op(Sub, other, self)

    def __mul__(self, other):
        return self._op(Multiply, self, other)

    def __rmul__(self, other):
        return self._op(Multiply, other, self)

    def __matmul__(self, other):
        return self._op(MatMul, self, other)

    def __rmatmul__(self, other):
        return self._op(MatMul, other, self)

    def __copy__(self):
        copy = Tensor(np.copy(self.data), self.requires_grad)
        try:
            copy.grad[:] = self.grad[:]
        except:
            pass
        return copy

    def copy(self):
        return self.__copy__()

    def numpy(self):
        return self.data.copy()

    def __len__(self):
        return len(self.data)

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        copy = Tensor(np.copy(self.data.transpose()), self.requires_grad)
        try:
            copy.grad[:] = self.grad.transpose()[:]
        except:
            pass
        return copy

    def swapaxes(self, axis1, axis2):
        # return SwapAxes(axis1, axis2)(self)
        return self._op(SwapAxes, self, axis1, axis2)

    def reshape(self, *shape):
        # return Reshape(*shape)(self)
        return self._op(Reshape, self, *shape)
