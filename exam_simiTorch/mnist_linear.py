import numpy as np
import similarTorch.nn as nn
import similarTorch.optim as optim
from similarTorch.tensor import Tensor
from similarTorch import CrossEntropyLoss
from similarTorch.utils import MNIST, Dataloader

batch_size = 64
epoches = 10
learning_rate = 0.0001

mnist_model = nn.Sequential(
    nn.Linear(28 * 28, 300),
    nn.ReLU(),
    nn.Linear(300, 300),
    nn.ReLU(),
    nn.Linear(300, 10),
    nn.Softmax(),
)

train_dataset = MNIST(
    images_path="./exam_simiTorch/data/train-images-idx3-ubyte",
    labels_path="./exam_simiTorch/data/train-labels-idx1-ubyte",
    flatten_input=True,
    one_hot_output=True,
    input_normalization=(0.1307, 0.3081),
)

test_dataset = MNIST(
    images_path="./exam_simiTorch/data/t10k-images-idx3-ubyte",
    labels_path="./exam_simiTorch/data/t10k-labels-idx1-ubyte",
    flatten_input=True,
    one_hot_output=True,
    input_normalization=(0.1307, 0.3081),
)

train_dataloader = Dataloader(train_dataset)
test_datasloader = Dataloader(test_dataset)

train_batches = train_dataloader.get_batch_iterable(batch_size)
test_batches = test_datasloader.get_batch_iterable(batch_size)

optimizer = optim.Adam(mnist_model.parameters(), learning_rate=learning_rate)
loss = CrossEntropyLoss()


def test_model_acc():
    correct = 0
    for test_batch_in, test_batch_out in test_batches:
        test_output = mnist_model(Tensor(test_batch_in)).data
        correct += np.sum(
            np.argmax(test_output, axis=1) == np.argmax(test_batch_out, axis=1)
        )

    my_acc = correct / len(test_dataset)
    return my_acc


if __name__ == "__main__":
    for it in range(epoches):
        train_batches.shuffle()
        for i_b, (batch_in, batch_out) in enumerate(train_batches):
            model_input = Tensor(batch_in)
            label = Tensor(batch_out)
            pred = mnist_model(model_input)
            errors = loss(pred, label)
            optimizer.zero_grad()
            errors.backward()
            optimizer.step()
        acc = test_model_acc()
        print(f"Epoch {it} - Accuracy: {acc}")
