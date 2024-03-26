# import similarTorch
import similarTorch.nn as nn
from similarTorch.tensor import Tensor
import numpy as np

input_x = np.linspace(-1, 1, 100)
input_y = np.linspace(5, 10, 5)

model = nn.Linear(100, 5)
out = model(Tensor(input_x))
print(out)
