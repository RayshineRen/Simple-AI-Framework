from abc import ABC, abstractmethod
from similarTorch import tensor


class Context:
    def __init__(self):
        self.data_for_back = None

    def save_for_backward(self, *data):
        self.data_for_back = tuple(data)


class Autograd(ABC):
    def apply(self, *tensor_list):
        ctx = Context()
        forward_tensor = self.forward(ctx, *map(lambda x: x.data, tensor_list))
        output_tensor = tensor.Tensor(forward_tensor, requires_grad=False)
        # output_tensor.backward_function = lambda x: self.backward(ctx, x)
        output_tensor.backward_function = self.backward
        output_tensor.backward_tensor = list(tensor_list)
        return output_tensor

    @abstractmethod
    def forward(self, ctx: Context, *tensor_lsit):
        raise NotImplementedError

    @abstractmethod
    def backward(self, ctx: Context, grad):
        raise NotImplementedError

    def __call__(self, *tensor_list):
        return self.apply(*tensor_list)
