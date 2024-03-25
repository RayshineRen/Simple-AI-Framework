import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict
from similarTorch.tensor import Tensor


class Module(ABC):
    def __init__(self):
        self._parameters = OrderedDict([])

    def register_parameter(self, *var_iterable):
        for var_name, var in var_iterable:
            self._parameters.update({var_name: var})  # 顺序正常

    def parameters(self) -> list:
        return list(self._parameters.values())

    def get_state_dict(self) -> OrderedDict:
        return self._parameters

    def load_state_dict(self, state_dict: OrderedDict):
        for k, v in state_dict.items():
            self._parameters[k].data = np.array(v)

    @abstractmethod
    def forward(self, *input) -> Tensor:
        raise NotImplementedError

    def __call__(self, *input) -> Tensor:
        return self.forward(*input)
