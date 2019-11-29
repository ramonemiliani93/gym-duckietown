import os
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import numpy as np


class SineDataset(Dataset):
    """ Sine function dataset given by:
        y = x · sin(x) + 0.3 · eps_1 + 0.3 · x · eps_2 , with  eps_1, eps_2 ∼ N(0,1)
    """

    def __init__(self, num_samples: int, domain: Tuple[float, float]):
        """
        Args:
            num_samples: Number of samples to draw from the domain of the function.
            domain: X range of the data to be generated.
        """
        super(SineDataset,  self).__init__()
        self.num_samples = num_samples
        self.domain = domain
        self.samples = np.random.uniform(*self.domain, self.num_samples)
        self.targets = self.function(self.samples)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value: int):
        if value <= 0:
            raise ValueError("Number of samples has to be a positive integer.")
        self._num_samples = value

    @property
    def domain(self) -> Tuple[float, float]:
        return self._domain

    @domain.setter
    def domain(self, value: Tuple[float, float]):
        low, high = value
        if high <= low:
            raise ValueError("Invalid domain specified.")
        self._domain = low, high

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.samples[item]).unsqueeze(0)
        target = torch.tensor(self.targets[item]).unsqueeze(0)

        return sample, target

    def __len__(self) -> int:
        return self.num_samples

    @staticmethod
    def function(x: np.ndarray) -> np.ndarray:
        eps_1 = np.random.normal(loc=0, scale=1, size=(len(x)))
        eps_2 = np.random.normal(loc=0, scale=1, size=(len(x)))
        y = x * np.sin(x) + 0.3 * eps_1 + 0.3 * x * eps_2

        return y


class MemoryMapDataset(Dataset):
    """Dataset to store multiple arrays on disk avoiding saturating the RAM"""
    def __init__(self, size: int, data_size: tuple, target_size: tuple, path: str):
        self.size = size
        self.data_size = data_size
        self.target_size = target_size
        self.path = path

        # Path for each array
        self.data_path = os.path.join(path, 'data.dat')
        self.target_path = os.path.join(path, 'target.dat')

        # Create arrays
        self.data = np.memmap(self.data_path, dtype='float32', mode='w+', shape=(self.size, *self.data_size))
        self.target = np.memmap(self.target_path, dtype='float32', mode='w+', shape=(self.size, *self.target_size))

        # Initialize number of saved records to zero
        self.length = 0

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.tensor(self.data[item, ...])
        target = torch.tensor(self.target[item, ...])

        return sample, target

    def __len__(self) -> int:
        return self.length

    def extend(self, observations: List[np.ndarray], actions: List[np.ndarray]):
        for index, (observation, action) in enumerate(zip(observations, actions)):
            self.data[self.length + index, ...] = observation.astype(np.float32)
            self.target[self.length + index, ...] = action.astype(np.float32)
        self.length += len(observations)

    def save(self):
        # TODO
        pass
