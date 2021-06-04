from abc import ABC, abstractmethod

import torch
import numpy as np


class BaseDistribution(ABC):
    @abstractmethod
    def log_prob(self, xs):
        pass

    @abstractmethod
    def sample(self, n):
        pass


class Plottable2DDistribution(BaseDistribution):
    @abstractmethod
    def log_prob(self, xs):
        pass

    def get_z_probabilities(self, x, y):
        x, y = np.meshgrid(x, y)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        z = self.log_prob(torch.tensor(pos.astype('float32'))).exp()

        return z
