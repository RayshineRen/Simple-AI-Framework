import numpy as np
from similarTorch.autograd import Autograd


class SwapAxes(Autograd):
    def __init__(self, axis1, axis2):
        self.axis1 = axis1
        self.axis2 = axis2

    def forward(self, ctx, input):
        return np.swapaxes(input, self.axis1, self.axis2)

    def backward(self, ctx, grad):
        return np.swapaxes(grad, self.axis1, self.axis2)


class Reshape(Autograd):
    def __init__(self, *shape):
        self.new_shape = shape

    def forward(self, ctx, input):
        ctx.save_for_backward(input.shape)
        return np.reshape(input, self.new_shape)

    def backward(self, ctx, grad):
        (back_shape,) = ctx.data_for_back
        if len(self.new_shape) + 1 == len(grad.shape):
            back_shape = (grad.shape[0], *back_shape)
        return np.reshape(grad, back_shape)


class GetItem(Autograd):
    def __init__(self, index):
        self.index = index

    def forward(self, ctx, input):
        ctx.save_for_backward(input.shape)
        return input[self.index]

    def backward(self, ctx, grad):
        (back_shape,) = ctx.data_for_back
        new_grad = np.zeros(back_shape, dtype=np.float32)
        new_grad[self.index] = grad
        return new_grad
