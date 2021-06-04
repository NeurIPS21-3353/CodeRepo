from abc import ABC

import numpy
import torch
import torch.distributions as D

import numpy as np

from distributions.BaseDistribution import Plottable2DDistribution


class RotationDistribution(Plottable2DDistribution):
    def __init__(self, skewness, n, mean=7):
        self.d = 2
        self.dimension = 2
        self.K = n
        self.mean = mean

        mix = D.Categorical(torch.ones(n))
        theta = torch.tensor([2 * np.pi / n] )
        U = torch.tensor([[torch.cos(theta), torch.sin(theta)], [-torch.sin(theta), torch.cos(theta)]])

        self.mu = torch.zeros([self.K, self.d])
        self.sigma = torch.zeros([self.K, self.d, self.d])

        self.mu[0, :] = self.mean * torch.tensor([1., 0.])
        self.sigma[0, :, :] = torch.diag(torch.tensor([1., 1. / skewness]))

        for i in range(1, n):
            self.mu[i, :] = torch.matmul(U, self.mu[i - 1, :])
            self.sigma[i, :, :] = torch.matmul(U, np.matmul(self.sigma[i - 1, :, :], U.T))

        comp = D.MultivariateNormal(self.mu, self.sigma)
        self.target = D.MixtureSameFamily(mix, comp)

    def log_prob(self, x):
        return self.target.log_prob(x)

    def sample(self, n):
        return  self.target.sample_n(n)


class TwoCircleDistribution(Plottable2DDistribution):
    def __init__(self, radius_1 = 4, radius_2 = 8, thickness_1 = 0.5, thickness_2=0.5, mixing = [1., 1.]):
        self.r1 = radius_1
        self.r2 = radius_2
        self.t1 = thickness_1
        self.t2 = thickness_2
        self.mixing = torch.tensor(mixing)

        # Radius distribution
        mix = D.Categorical(self.mixing)
        comp = D.Normal(torch.FloatTensor([self.r1, self.r2]), torch.FloatTensor([self.t1, self.t2]))
        self.radius_d = D.MixtureSameFamily(mix, comp)

        # Ring distribution
        self.ring = D.Uniform(torch.tensor([-numpy.pi]), torch.tensor([numpy.pi]))

    def log_prob(self, x):
        r = torch.norm(x, dim=-1)
        # print(r)

        log_prob = self.radius_d.log_prob(r)

        return log_prob

    def sample(self, n):
        r = self.radius_d.sample_n(n)
        u = self.ring.sample_n(n).squeeze()

        samples = torch.zeros((n, 2))
        samples[:, 0] = r * torch.cos(u)
        samples[:, 1] = r * torch.sin(u)

        return samples


class OneCircleDistribution(Plottable2DDistribution):
    def __init__(self, radius_1 = 4, thickness_1 = 0.5,):
        self.r1 = radius_1
        self.t1 = thickness_1

        # Radius distribution
        self.radius_d = D.Normal(torch.FloatTensor([self.r1]), torch.FloatTensor([self.t1]))

        # Ring distribution
        self.ring = D.Uniform(torch.tensor([-numpy.pi]), torch.tensor([numpy.pi]))

    def log_prob(self, x):
        r = torch.sqrt((x[:, 0] ** 2) + (x[:, 1] ** 2))
        print(r)

        log_prob = self.radius_d.log_prob(r)

        return log_prob

    def sample(self, n):
        r = self.radius_d.sample_n(n).squeeze()
        u = self.ring.sample_n(n).squeeze()

        samples = torch.zeros((n, 2))
        samples[:, 0] = r * torch.cos(u)
        samples[:, 1] = r * torch.sin(u)

        return samples


class TwoSphereDistribution(Plottable2DDistribution):
    def __init__(self, radius_1 = 2, radius_2 = 4, thickness_1 = 0.1, thickness_2=0.1, mixing = [1., 1.]):
        self.r1 = radius_1
        self.r2 = radius_2
        self.t1 = thickness_1
        self.t2 = thickness_2
        self.mixing = torch.tensor(mixing)

        # Radius distribution
        mix = D.Categorical(self.mixing)
        comp = D.Normal(torch.FloatTensor([self.r1, self.r2]), torch.FloatTensor([self.t1, self.t2]))
        self.radius_d = D.MixtureSameFamily(mix, comp)

        # Ring distribution
        self.phi_d = D.Uniform(torch.tensor([0.]), torch.tensor([1.]))
        self.theta_d = D.Uniform(torch.tensor([0.]), torch.tensor([2 * np.pi]))
        self.ring = D.Uniform(torch.tensor([0., 0]), torch.tensor([1., 2 * np.pi]))

        self.r = None
        self.u = None

    def log_prob(self, x):
        r = torch.norm(x, dim=-1)
        log_prob = self.radius_d.log_prob(r)

        return log_prob

    def sample(self, n, store=False):
        r = self.radius_d.sample_n(n)
        theta = self.theta_d.sample_n(n).squeeze()
        phi = self.phi_d.sample_n(n).squeeze()

        phi = torch.acos(1 - 2 * phi) # Prevent oversampling on the poles

        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)

        samples = torch.zeros((n, 3))
        samples[:, 0] = x
        samples[:, 1] = y
        samples[:, 2] = z

        # samples = torch.cat([xs, ys], dim=1)
        if store:
            self.theta = theta
            self.phi = phi

        return samples
