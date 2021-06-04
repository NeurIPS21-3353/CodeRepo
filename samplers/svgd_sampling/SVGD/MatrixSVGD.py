import torch

from samplers.svgd_sampling.SVGD.BaseSVGD import BaseSVGD

class MatrixSVGD(BaseSVGD):
    def __init__(self, prob, kernel, contrastive_SVGD=True):
        super().__init__(prob, kernel, contrastive_SVGD=contrastive_SVGD)

    def get_prob_grad(self, xs, prob):
        log_prob = prob.log_prob(xs)
        prob_grad = torch.autograd.grad(log_prob.sum(), xs)[0]
        return prob_grad

    def get_update_part1(self, xs, kernel, prob_grad):
        kernel_value = kernel(xs, xs)
        # t = kernel_value.sum(2).sum(2).detach().numpy()
        prob_grad = prob_grad[None, :, :, None]
        update_part_1 = torch.matmul(kernel_value, prob_grad).sum(1).squeeze()
        return update_part_1, kernel_value

    def get_update_part2_quick(self, xs, kernel_value):
        d = kernel_value.size(2)
        g = -torch.autograd.grad(kernel_value, xs, torch.ones_like(kernel_value))[0]
        return g / d

    def phi(self, prob, kernel, X, annealing=1.0):
        xs = X.detach().clone()
        xs.requires_grad = True

        prob_grad = self.get_prob_grad(xs, prob)

        update_part_1, kernel_value = self.get_update_part1(xs, kernel, prob_grad)
        update_part_2_quick = self.get_update_part2_quick(xs, kernel_value)

        update = (1 / xs.size(0)) * (annealing * update_part_1 + update_part_2_quick)

        return update