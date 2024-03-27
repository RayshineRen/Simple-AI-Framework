# import similarTorch
import similarTorch.nn as nn
import similarTorch.optim as optim
from similarTorch.tensor import Tensor
import numpy as np

input_x = np.linspace(-1, 1, 100)
input_y = np.linspace(5, 10, 100)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), learning_rate=0.01)

for epoch in range(1000):
    for i in range(0, 100, 10):
        x = Tensor(input_x[i : i + 10].reshape(-1, 1))
        y = Tensor(input_y[i : i + 10].reshape(-1, 1))
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

print(model(Tensor([0.5])))
