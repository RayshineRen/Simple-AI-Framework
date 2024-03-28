import numpy as np
from similarTorch.autograd import Autograd


class ReLU(Autograd):
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return np.clip(x, a_min=0, a_max=None)

    def backward(self, ctx, grad):
        (t,) = ctx.data_for_back
        return np.where(t < 0, 0, grad)


class Sigmoid(Autograd):
    def forward(self, ctx, x):
        sig = 1 / (1 + np.exp(-x))
        ctx.save_for_backward(sig)
        return sig

    def backward(self, ctx, grad):
        (sig,) = ctx.data_for_back
        return sig * (1 - sig) * grad


class Softmax(Autograd):
    def forward(self, ctx, x):
        softm = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        ctx.save_for_backward(softm)
        return softm

    def backward(self, ctx, grad):
        (softm,) = ctx.data_for_back
        return grad * softm * (1 - softm)  # 只计算了对角线上的元素


class Softplus(Autograd):
    # Softplus(x) = log(1 + exp(x))
    def forward(self, ctx, x):
        ctx.save_for_backward(1 + np.exp(-x))
        return np.log(1 + np.exp(-x))

    def backward(self, ctx, grad):
        (softp,) = ctx.data_for_back
        return grad / softp


class Softsign(Autograd):
    # Softsign(x) = x / (1 + |x|)
    def forward(self, ctx, x):
        ctx.save_for_backward(1 + np.abs(x))
        return x / (1 + np.abs(x))

    def backward(self, ctx, grad):
        (softs,) = ctx.data_for_back
        return grad / softs


class ArcTan(Autograd):
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return np.arctan(x)

    def backward(self, ctx, grad):
        (t,) = ctx.data_for_back
        return grad / (1 + t * t)


class Tanh(Autograd):
    def forward(self, ctx, x):
        tanh = np.tanh(x)
        ctx.save_for_backward(tanh)
        return tanh

    def backward(self, ctx, grad):
        (tanh,) = ctx.data_for_back
        return (1 - tanh * tanh) * grad
