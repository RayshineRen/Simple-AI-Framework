import numpy as np
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, param_list: list, learning_rate=0.01, momentum=0.0, decay=0.0):
        super(SGD, self).__init__(param_list)
        self.lr = learning_rate
        self.decay = decay
        self.momentum = momentum

    @staticmethod
    def initialize_state(state, param):
        state["v"] = np.zeros_like(
            param.grad
        )  # 状态 state 中存储了动量 v，其形状与参数梯度 param.grad 相同，但值初始化为全零。

    def step(self):
        for param in self.param_list:
            if param.grad is None:
                continue
            if param not in self.state:
                self.state[param] = {}
            state = self.state[param]
            if len(state) == 0:
                self.initialize_state(state, param)
            state["v"] = self.momentum * state["v"] - self.lr * param.grad
            param.data += state["v"]
        self.lr *= 1 / (1 + self.decay)
