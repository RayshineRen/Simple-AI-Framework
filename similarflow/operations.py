import numpy as np
from .graph import Operation


class matmul(Operation):
    def __init__(self, x, y):
        super(matmul, self).__init__(x, y)

    def compute(self, x_value, y_value):
        return np.dot(x_value, y_value)


class add(Operation):
    def __init__(self, x, y):
        super(add, self).__init__(x, y)

    def compute(self, x_value, y_value):
        return np.add(x_value, y_value)


class negative(Operation):
    def __init__(self, x):
        super(negative, self).__init__(x)

    def compute(self, x_value):
        return -x_value


class multiply(Operation):
    def __init__(self, x, y):
        super(multiply, self).__init__(x, y)

    def compute(self, x_value, y_value):
        return np.multiply(x_value, y_value)


class sigmoid(Operation):
    def __init__(self, x):
        super(sigmoid, self).__init__(x)

    def compute(self, x_value):
        return 1 / (1 + np.exp(-x_value))


class softmax(Operation):
    def __init__(self, x):
        super(softmax, self).__init__(x)

    def compute(self, x_value):
        return np.exp(x_value) / np.sum(np.exp(x_value), axis=1)[:, None]


class log(Operation):
    def __init__(self, x):
        super(log, self).__init__(x)

    def compute(self, x_value):
        return np.log(x_value)


class square(Operation):
    def __init__(self, x):
        super(square, self).__init__(x)

    def compute(self, x_value):
        return np.square(x_value)


class reduce_sum(Operation):
    def __init__(self, x, axis=None):
        super(reduce_sum, self).__init__(x)
        self.axis = axis

    def compute(self, x_value):
        return np.sum(x_value, self.axis)
