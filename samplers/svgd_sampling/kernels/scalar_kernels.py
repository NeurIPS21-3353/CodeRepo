import math

import torch
import numpy as np

from samplers.svgd_sampling.kernels.BaseKernel import BaseKernel


class RBFKernel(BaseKernel):
    def __init__(self, h=-1):
        super().__init__()
        self.h = h

    def run(self, x1, x2):
        dist = torch.cdist(x1, x2, p=2, compute_mode='use_mm_for_euclid_dist') ** 2 # This call fails in pytorch 1.15
        # DIST_NP = dist.view(x2.size(0) * x2.size(1), -1).detach().numpy()
        if self.h == -1:
            # Experimentally I found that using an adaptive h gave worse results
            h = torch.median(dist)
            h = torch.sqrt(0.5 * h / torch.log(torch.tensor([float(x1.size(0))], device=x1.device) + 1))
        else:
            h = self.h

        out = torch.exp(-dist / (h ** 2) / 2.0)

        return out

    def to(self, device):
        return


class InvariantScalarKernel(BaseKernel):
    def __init__(self, scalar_kernel, invariant_map):
        super().__init__()
        self.scalar_kernel = scalar_kernel
        self.invariant_map = invariant_map

    def run(self, x1, x2):
        inv_x1 = self.invariant_map(x1)
        inv_x2 = self.invariant_map(x2)

        return self.scalar_kernel(inv_x1, inv_x2)

    def to(self, device):
        return


class CNInvariantKernel(BaseKernel):
    def __init__(self, base_kernel, N=4):
        super().__init__()
        self.base_kernel = base_kernel
        self.N = N

        pis = torch.arange(0, self.N) * ((2. * np.pi) / self.N)
        self.g = torch.zeros((2, 2, self.N))
        for idx, pi in enumerate(pis):
            self.g[:, :, idx] = torch.tensor([[np.cos(pi), -np.sin(pi)],
                                               [np.sin(pi), np.cos(pi)]])

    def run(self, x1s, x2s):
        gx2 = torch.matmul(self.g.T[:, None, :, :], x2s[:, :, None]).squeeze()

        k = self.base_kernel(x1s, gx2)
        k = torch.mean(k, 0)

        return k

    def to(self, device):
        self.g.to(device)


class ScalarParticleKernel(BaseKernel):
    def __init__(self, base_kernel, types, n_particles=19, dim=3):
        super().__init__()
        self.base_kernel = base_kernel
        self.types = types
        self.n_particles = n_particles
        self.dim=dim

    def find_COM(self, inp):
        # Find CoM
        CoM = torch.mean(inp, dim=1)
        return CoM

    def center_around_0(self, inp, CoM):
        # Center around 0 for translation invariance
        centered = inp - CoM[:, None, :]
        return centered

    def order(self, centered, types):
        # Order by radius to make permutation invariant
        r = torch.norm(centered, dim=2)
        order = torch.argsort(r)
        sorted = centered[torch.arange(order.size(0)).unsqueeze(1).repeat((1, order.size(1))), order]
        types_sorted = types[torch.arange(order.size(0)).unsqueeze(1).repeat((1, order.size(1))), order]
        return sorted, types_sorted

    def rotate(self, sorted):
        top = sorted[:, -1]
        A = top / torch.norm(top, dim=1)[:, None]
        B = torch.tensor([1., 0., 0.], device=A.device)

        dot = (A @ B)
        cross = torch.cross(A, B.repeat(A.size(0), 1))
        cross_norm = torch.norm(cross, dim=1)

        zeros = torch.zeros_like(dot, device=dot.device)
        ones = torch.ones_like(dot, device=dot.device)
        G1 = torch.vstack((dot, -cross_norm, zeros)).T
        G2 = torch.vstack((cross_norm, dot, zeros)).T
        G3 = torch.vstack((zeros, zeros, ones)).T
        G = torch.stack((G1, G2, G3), dim=2)

        u = A
        v = (B - (dot[:, None] * A)) / torch.norm((B - (dot[:, None] * A)), dim=1)[:, None]
        w = torch.cross(B.repeat(A.size(0), 1), A)
        F_inv = torch.stack([u, v, w], dim=2)
        F = torch.inverse(F_inv)
        U = torch.matmul(torch.matmul(F_inv, G), F)

        rotated = torch.matmul(sorted, U)

        return rotated, U

    def run(self, x1s, x2s):
        inp_1 = x1s.view(-1, self.n_particles, self.dim)
        inp_2 = x2s.view(-1, self.n_particles, self.dim)

        CoM_1 = self.find_COM(inp_1)
        CoM_2 = self.find_COM(inp_2)

        centered_1 = self.center_around_0(inp_1, CoM_1)
        centered_2 = self.center_around_0(inp_2, CoM_2)

        sorted_1, sorted_types_1 = self.order(centered_1, types=self.types)
        sorted_2, sorted_types_2 = self.order(centered_2, types=self.types)

        rotated_1, U = self.rotate(sorted_1)
        rotated_2, U = self.rotate(sorted_2)

        s_1 = rotated_1.size()
        s_2 = rotated_1.size()

        input_1 = rotated_1.view((s_1[0], s_1[1] * s_1[2]))
        input_2 = rotated_2.view((s_2[0], s_2[1] * s_2[2]))

        k = self.base_kernel(input_1, input_2)

        return k

    def to(self, device):
        self.base_kernel.to(device)
