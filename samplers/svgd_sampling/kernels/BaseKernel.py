from abc import ABC, abstractmethod


class BaseKernel(ABC):
    def __init__(self):
        super().__init__()

    def __call__(self, x1, x2):
        return self.run(x1, x2)

    @abstractmethod
    def run(self, x1, x2):
        pass

    @abstractmethod
    def to(self, device):
        pass


class DifferentiableBaseKernel(BaseKernel):
    """
    Start for kernels with predefined grad path
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self, x1, x2):
        pass

    @abstractmethod
    def grad(self, x1, x2):
        pass



