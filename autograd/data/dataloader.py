from typing import Callable, Any, List

import numpy as np
import math

import autograd
from dataset import Dataset


class DataLoader(object):
    """
    Dataloader class
    """

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True,
                 collate_fn: Callable[[List], Any] = None) -> None:
        """
        Constructor
        :param dataset: (Dataset) Dataset
        :param batch_size: (int) Batch size to be utilized
        :param shuffle: (bool) If true dataset is shuffled
        :param collate_fn: (Callable) Function to perform batching
        """
        # Check parameter
        assert batch_size > 0, 'Batch size must be bigger than 0.'
        # Save parameters
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        # Make indexes
        self.indexes = np.arange(self.dataset_len)
        # Shuffle indexes if utilized
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """
        Returns the length of the dataloader
        :return: (int) Length
        """
        return len(self.dataset) // self.batch_size

    def __iter__(self) -> Any:
        """
        Iter method iterates over the whole dataset and batches the dataset output
        :return: (Any) Batch objects
        """
        for index in range(math.ceil(self.dataset_len / self.batch_size)):
            for batch in range(self.batch_size):
                if index * self.batch_size + batch < self.dataset_len:
                    if batch == 0:
                        instances = self.dataset[self.indexes[index * self.batch_size + batch]]
                        return_values = list(instances)
                        for index in range(len(return_values)):
                            return_values[index] = [return_values[index]]
                    else:
                        instances = self.dataset[self.indexes[index * self.batch_size + batch]]
                        for index, instance in enumerate(instances):
                            return_values[index].append(instance)
            # Apply collate operation
            if self.collate_fn is None:
                yield tuple([autograd.stack(return_values[index]) for index in range(len(return_values))])
            else:
                yield self.collate_fn(return_values)
