import numpy as np
from typing import Tuple 

class Dataset:
    """
    A simple dataset class to hold features and labels.

    Attributes:
        X (np.ndarray): The input features.
        y (np.ndarray): The target labels.

    Methods:
        __getitem__(idx: int): Get the feature and label at the specified index.
        __len__(): Return the total number of samples in the dataset.
    """
    def __init__(self, X:np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.X)


class DataLoader:
    """
    DataLoader class to iterate over the dataset in batches.

    Attributes:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): The number of samples per batch.
        shuffle (bool): Indicates if to shuffle the data before each epoch.

    Methods:
        __iter__() -> Iterator['DataLoader']:
            Initialize the iteration and optionally shuffle the data.
        __next__() -> Tuple[np.ndarray, np.ndarray]:
            Return the next batch of data.
    """

    def __init__(self, dataset: Dataset, batch_size: int = 64, shuffle: bool = False):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.current_idx = 0

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.current_idx >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]

        self.current_idx += self.batch_size  # Move to the next batch

        batch_X, batch_y = zip(*batch)

        return np.array(batch_X), np.array(batch_y)

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
    