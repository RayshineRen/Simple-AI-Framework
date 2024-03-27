import numpy as np
from similarTorch.autograd import Autograd


class MSELoss(Autograd):
    def forward(self, ctx, target, input):
        if target.shape != input.shape:
            raise ValueError("Wrong shape")
        ctx.save_for_backward(target, input)
        return ((target - input) ** 2).mean()

    def backward(self, ctx, grad):
        target, input = ctx.data_for_back
        batch = target.shape[0]
        dL_dinput = grad * 2 * (input - target) / batch
        dL_dtarget = grad * 2 * (target - input) / batch
        return dL_dtarget, dL_dinput


class CrossEntropyLoss(Autograd):
    def forward(self, ctx, target, input):
        ctx.save_for_backward(target, input)
        input = np.clip(input, 1e-15, 1 - 1e-15)
        return -target * np.log(input) - (1 - target) * np.log(1 - input)

    def backward(self, ctx, grad):
        target, input = ctx.data_for_back
        batch = target.shape[0]
        input = np.clip(input, 1e-15, 1 - 1e-15)
        dL_dinput = -target / input + (1 - target) / (1 - input)
        dL_dinput = grad * dL_dinput / batch
        dL_dtarget = np.log(1 - input) - np.log(input)
        dL_dtarget = grad * dL_dtarget / batch
        return dL_dtarget, dL_dinput
