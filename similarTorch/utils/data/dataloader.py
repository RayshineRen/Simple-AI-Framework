import numpy as np
from .dataset import SingleDataset, BatchDataset


class Dataloader(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def get_single_iterable(self) -> SingleDataset:
        return SingleDataset(self.dataset)

    def get_batch_iterable(self, batch_size) -> BatchDataset:
        return BatchDataset(self.dataset, batch_size)

    def get_all_batches(self, shuffle=False):
        batches = [v for v in self.dataset]
        if shuffle:
            np.random.shuffle(batches)
        return np.array(batches)
