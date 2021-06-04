from abc import abstractmethod

import torch

from samplers.generators.BaseGenerators import BaseSampleGenerator

class BaseBatchSampleGenerator(BaseSampleGenerator):
    def __init__(self, true_samples, sampler, device, persistent=False, persistent_reset=0.0):
        super().__init__(true_samples, sampler, device)
        self.persistent = persistent
        self.first_batch = True
        self.fake_sample_collection = []
        self.persistent_rest = persistent_reset

        self.first_run = True

    @abstractmethod
    def _get_starting_sample(self, true_samples):
        pass

    def _next_epoch(self):
        self.first_batch = True

    def _next_batch(self, true_samples):
        if torch.rand(1)[0] < self.persistent_rest:
            print("Resetting")
            self.first_run = True
        if not self.persistent or self.first_run:
            starting_sample = self._get_starting_sample(true_samples)
        else:
            starting_sample = self.previous

        fake_samples = self.sampler.sample(starting_sample, true_samples=true_samples, prefix=f"epoch_{self.epoch}_batch_{self.batch}_")

        self.previous = fake_samples
        self.first_run = False
        self.first_batch = False

        return fake_samples


class BatchStartingPointSampleGenerator(BaseBatchSampleGenerator):
    """
        Generates samples in batches
        Starting point is the original samples
        """
    def _get_starting_sample(self, true_samples):
        return true_samples


class BatchUniformSampleGenerator(BaseBatchSampleGenerator):
    """
        Generates samples in batches
        Starting point is the original samples
        """
    def __init__(self, true_samples, sampler, device, min, max, persistent=False, persistent_reset=0.0):
        super().__init__(true_samples, sampler, device, persistent, persistent_reset)
        self.uniform = torch.distributions.Uniform(torch.tensor(min, device=self.device), torch.tensor(max, device=self.device))

    def _get_starting_sample(self, true_samples):
        starting_samples = self.uniform.sample_n(true_samples.size(0))
        return starting_samples
