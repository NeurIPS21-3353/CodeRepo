from abc import ABC, abstractmethod

import torch


class BaseSVGD(ABC):
    def __init__(self, prob, kernel, contrastive_SVGD=True):
        self.contrastive_SVGD = contrastive_SVGD
        self.p = prob
        self.k = kernel

    def construct(self, X, opt, true_samples=None, annealing=1.0):
        if true_samples is not None and self.contrastive_SVGD:
            x_comb = torch.cat([X, true_samples])
        else:
            x_comb = X

        grad = self.phi(self.p, self.k, x_comb, annealing)
        grad = grad[:X.size(0)]

        # Update X using the chosen optimizer
        opt.zero_grad()
        X.grad = -grad
        opt.step()

        return X

    @abstractmethod
    def phi(self, prob, kernel, X, annealing=1.0):
        pass