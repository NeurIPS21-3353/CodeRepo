
import torch

from samplers.svgd_sampling.SVGD.SVGD import SVGD


class SVGDSampler():
    def __init__(self, energy_model, kernel, lr, epochs, epsilon=None, loggers=[]):
        self.model = energy_model
        self.kernel = kernel
        self.lr = lr
        self.epochs = epochs
        self.epsilon = epsilon
        self.loggers = loggers

        self.svgd = SVGD(self.model, self.kernel)

    def to(self, device):
        return

    def sample(self, x, true_samples=None, prefix=""):
        Xs = x.detach().clone()
        opt = torch.optim.SGD([Xs], lr=self.lr)

        for i in range(0, self.epochs):
            start_Xs = Xs.detach().clone()
            Xs = self.svgd.construct(Xs, opt, true_samples)

            for logger in self.loggers:
                logger.execute(Xs, true_samples, i, prefix=prefix)

            if self.epsilon is not None:
                t = Xs - start_Xs
                dist = torch.norm(t, dim=1, p=2).mean()
                if dist < self.epsilon:
                    print(f"Sampling epsilon reached: {dist} < {self.epsilon}")
                    break


        return Xs

    def reset_loggers(self):
        for log in self.loggers:
            log.reset()
