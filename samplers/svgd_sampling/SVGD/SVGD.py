import torch

from samplers.svgd_sampling.SVGD.BaseSVGD import BaseSVGD


class SVGD(BaseSVGD):
    def __init__(self, prob, kernel):
        super().__init__(prob, kernel)

    def phi(self, prob, kernel, X, annealing=1.0):
        xs = X.detach().clone()
        xs.requires_grad = True

        # Get grad prob
        log_prob = prob.log_prob(xs)
        prob_grad = torch.autograd.grad(log_prob.sum(), xs)[0]

        # Get kernel value and grad
        kernel_value = kernel(xs, xs)
        kernel_grad = -torch.autograd.grad(kernel_value.sum(), xs)[0]

        update = (1 / xs.size(0)) * ((annealing * kernel_value.matmul(prob_grad)) + kernel_grad)

        return update