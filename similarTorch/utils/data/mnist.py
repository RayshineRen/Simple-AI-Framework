import gzip
import numpy as np

from pathlib import Path
from urllib import request
from .dataset import Dataset
from abc import ABC


class MNIST(Dataset, ABC):
    train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    @staticmethod
    def _load_mnist(images_path: str, labels_path: str, flatten_input, one_hot_output):
        with open(images_path, "rb") as f:
            new_shape = (-1, 28 * 28) if flatten_input else (-1, 1, 28, 28)
            data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(new_shape)
        with open(labels_path, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

            if one_hot_output:
                b = np.zeros((labels.size, 10))
                b[np.arange(labels.size), labels] = 1
                labels = b

        return data, labels

    @staticmethod
    def download_if_not_exists(url, path):
        my_file = Path(path)
        if not my_file.exists():
            response = request.urlopen(url)
            with open(path, "wb") as f:
                f.write(response.read())

            with gzip.open(path, "rb") as zipped:
                out = zipped.read()
            with open(path, "wb") as f_out:
                f_out.write(out)

    def __init__(
        self,
        images_path: str = None,
        labels_path=None,
        train=True,
        flatten_input=True,
        one_hot_output=True,
        input_normalization=None,
    ):

        if train:
            self.download_if_not_exists(self.train_images_url, images_path)
            self.download_if_not_exists(self.train_labels_url, labels_path)
        else:
            self.download_if_not_exists(self.test_images_url, images_path)
            self.download_if_not_exists(self.test_labels_url, labels_path)

        self.loaded_data, self.loaded_labels = self._load_mnist(
            images_path, labels_path, flatten_input, one_hot_output
        )

        self.input_normalization = input_normalization

    def __getitem__(self, idx):
        data_in = self.loaded_data[idx] / 255
        data_out = self.loaded_labels[idx]
        if self.input_normalization is not None:
            data_in = np.where(
                data_in != 0,
                (data_in - self.input_normalization[0]) / self.input_normalization[1],
                0,
            )
        return data_in, data_out

    def __len__(self):
        return len(self.loaded_data)
