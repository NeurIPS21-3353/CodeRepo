from abc import ABC, abstractmethod

import numpy as np


class BaseSampleGenerator(ABC):
    def __init__(self, true_samples, sampler, device):
        self.true_samples = true_samples
        self.sampler = sampler
        self.device = device
        self.epoch = 0
        self.batch = 0

    def next_batch(self, true_samples):
        self.batch += 1
        return self._next_batch(true_samples)

    def next_epoch(self):
        self.batch = 0
        self.epoch += 1
        self._next_epoch()

    @abstractmethod
    def _next_batch(self, true_samples):
        pass

    @abstractmethod
    def _next_epoch(self):
        pass

    def get_index_true_samples(self, samples):
        return np.isin(self.true_samples, samples).all(axis=1).nonzero()