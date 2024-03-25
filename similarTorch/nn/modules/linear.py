import numpy as np
import similarTorch
from .module import Module
from . import init

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = 