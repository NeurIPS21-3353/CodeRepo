import torch
import numpy as np

from samplers.svgd_sampling.kernels.BaseKernel import BaseKernel


class IdentityMatrixKernel(BaseKernel):
    def __init__(self, scalar_kernel):
        super().__init__()
        self.scalar_kernel = scalar_kernel

    def run(self, x1, x2):
        d = x1.size(1)
        scalar_output = self.scalar_kernel(x1, x2)
        id = (scalar_output * torch.eye(2, device=x1.device)[:, :, None, None]).T

        return id

    def to(self, device):
        return


class VectorizedRotationEquivariantMatrixKernel(BaseKernel):
    def __init__(self, scalar_kernel, r_group=4):
        super().__init__()
        self.scalar_kernel = scalar_kernel
        self.r_group = r_group

        pis = np.arange(0, self.r_group) * ((2. * np.pi) / self.r_group)
        self.gs = torch.zeros((2, 2, self.r_group))
        for idx, pi in enumerate(pis):
            self.gs[:, :, idx] = torch.tensor([[np.cos(pi), -np.sin(pi)],
                                               [np.sin(pi), np.cos(pi)]])

    def run(self, x1s, x2s):
        gx2 = torch.matmul(self.gs.T[:, None, :, :], x2s[:, :, None]).squeeze()

        k = self.scalar_kernel(x1s, gx2)

        k2 = k * torch.eye(2, device=x1s.device)[:, :, None, None, None] # This I is g in the paper
        k2t = torch.sum(k2, 2).permute([3, 2, 1, 0])

        return k2t

    def to(self, device):
        self.gs = self.gs.to(device)


class ApproximationRotationEquivariantMatrixKernel(BaseKernel):
    def __init__(self, scalar_kernel, samples=10):
        super().__init__()
        self.scalar_kernel = scalar_kernel
        self.n_samples = samples
        self.uniform = torch.distributions.Uniform(torch.tensor([0.]), torch.tensor([np.pi * 2.]))

    def get_rotation_matrices(self, pis, device):
        t0 = torch.cat([torch.cos(pis), -torch.sin(pis)], dim=1).to(device).T
        t1 = torch.cat([torch.sin(pis), torch.cos(pis)], dim=1).to(device).T
        t = torch.stack([t0, t1]).to(device)
        return t

    def run(self, x1s, x2s):
        d = x1s.size(1)

        pis = self.uniform.sample_n(self.n_samples).to(x1s.device)
        gs = self.get_rotation_matrices(pis, x1s.device)

        gx2 = torch.matmul(gs.T[:, None, :, :], x2s[:, :, None]).squeeze(3)

        k = self.scalar_kernel(x1s, gx2)

        k2 = k * torch.eye(d, device=x1s.device)[:, :, None, None, None] # This I is g in the paper
        k2t = torch.sum(k2, 2).permute([3, 2, 1, 0])

        for i in range(0, x1s.size(0)):
            k2t[i, i] = torch.eye(d, device=x1s.device)

        return k2t

    def to(self, device):
        self.uniform = torch.distributions.Uniform(torch.tensor([0.], device=device), torch.tensor([np.pi * 2.], device=device))
        return


class ContinousRotationEquivariantMatrixKernel(BaseKernel):
    def __init__(self, scalar_kernel, dim=2):
        super().__init__()
        self.scalar_kernel = scalar_kernel
        self.I = torch.eye(dim)[:, :, None, None]

    def run(self, x1s, x2s):
        inp_1 = torch.sqrt(x1s[:, 0] ** 2 + x1s[:, 1] ** 2).view((-1, 1))
        inp_2 = torch.sqrt(x2s[:, 0] ** 2 + x2s[:, 1] ** 2).view((-1, 1))

        k = self.scalar_kernel(inp_1, inp_2)

        k2 = k * self.I
        k2t = k2.permute([3, 2, 1, 0])

        return k2t

    def to(self, device):
        self.uniform = torch.distributions.Uniform(torch.tensor([0.], device=device), torch.tensor([np.pi * 2.], device=device))
        self.I = self.I.to(device)
        return


class CoupledParticleInvariantMatrixKernel(BaseKernel):
    def __init__(self, scalar_kernel, dim=2, n_particles=4, n_samples = 64):
        super().__init__()
        self.scalar_kernel = scalar_kernel
        self.dim = dim
        self.n_particles = n_particles

        self.n_samples = n_samples
        self.uniform = torch.distributions.Uniform(torch.tensor([0.]), torch.tensor([np.pi * 2.]))

    def get_rotation_matrices(self, pis, device):
        t0 = torch.cat([torch.cos(pis), -torch.sin(pis)], dim=1).to(device).T
        t1 = torch.cat([torch.sin(pis), torch.cos(pis)], dim=1).to(device).T
        t = torch.stack([t0, t1]).to(device)
        return t

    def find_COM(self, inp_1, inp_2):
        # Find CoM
        CoM_1 = torch.mean(inp_1, dim=1)
        CoM_2 = torch.mean(inp_2, dim=1)

        return CoM_1, CoM_2

    def center_around_0(self, inp_1, inp_2, CoM_1, CoM_2):
        # Center around 0 for translation invariance
        centered_1 = inp_1 - CoM_1[:, None, :]
        centered_2 = inp_2 - CoM_2[:, None, :]

        return centered_1, centered_2

    def order(self, centered_1, centered_2):
        # Order by radius to make permutation invariant
        r_1 = torch.sqrt(centered_1[:, :, 0] ** 2 + centered_1[:, :, 1] ** 2)  # [:, :, None]
        r_2 = torch.sqrt(centered_2[:, :, 0] ** 2 + centered_2[:, :, 1] ** 2)
        order_1 = torch.argsort(r_1)
        order_2 = torch.argsort(r_2)
        sorted_1 = centered_1[torch.arange(order_1.size(0)).unsqueeze(1).repeat((1, order_1.size(1))), order_1]
        sorted_2 = centered_2[torch.arange(order_2.size(0)).unsqueeze(1).repeat((1, order_2.size(1))), order_2]

        return sorted_1, sorted_2

    def rotate(self, centered_1, centered_2, sorted_1, sorted_2):
        r_1 = torch.sqrt(centered_1[:, :, 0] ** 2 + centered_1[:, :, 1] ** 2)  # [:, :, None]
        r_2 = torch.sqrt(centered_2[:, :, 0] ** 2 + centered_2[:, :, 1] ** 2)

        # Rotate to x axis
        top_1 = sorted_1[:, -1]
        top_r_1 = r_1.max(dim=1)[0]
        norm_1 = top_1 / top_r_1[:, None]
        t0 = torch.cat([norm_1[:, 0, None], -norm_1[:, 1, None]], dim=1).T
        t1 = torch.cat([norm_1[:, 1, None], norm_1[:, 0, None]], dim=1).T
        G_1 = torch.stack([t0, t1]).permute([2, 1, 0])[:, None, :, :]
        rotated_1 = torch.matmul(G_1, sorted_1[:, :, :, None]).squeeze()

        top_2 = sorted_2[:, -1]
        top_r_2 = r_2.max(dim=1)[0]
        norm_2 = top_2 / top_r_2[:, None]
        t0 = torch.cat([norm_2[:, 0, None], -norm_2[:, 1, None]], dim=1).T
        t1 = torch.cat([norm_2[:, 1, None], norm_2[:, 0, None]], dim=1).T
        G_2 = torch.stack([t0, t1]).permute([2, 1, 0])[:, None, :, :]
        rotated_2 = torch.matmul(G_2, sorted_2[:, :, :, None]).squeeze()
        # rotated_2 = sorted_2 * G

        return rotated_1, rotated_2

    def run(self, x1s, x2s):
        inp_1 = x1s.view(-1, self.n_particles, self.dim)
        inp_2 = x2s.view(-1, self.n_particles, self.dim)

        CoM_1, CoM_2 = self.find_COM(inp_1, inp_2)
        centered_1, centered_2 = self.center_around_0(inp_1, inp_2, CoM_1, CoM_2)
        sorted_1, sorted_2 = self.order(centered_1, centered_2)

        rotated_1, rotated_2 = self.rotate(centered_1, centered_2, sorted_1, sorted_2)

        s_1 = sorted_1.size()
        s_2 = sorted_2.size()
        input_1 = rotated_1.view((s_1[0], s_1[1]*s_1[2]))
        input_2 = rotated_2.view((s_2[0], s_2[1]*s_2[2]))

        k = self.scalar_kernel(input_1, input_2)

        # Make invariant
        k = k * (1 - torch.eye(k.size(0))) + torch.eye(k.size(0))
        k_eye = (k * torch.eye(8)[:, :, None, None]).permute([3, 2, 1, 0])

        return k_eye


    def to(self, device):
        self.uniform = torch.distributions.Uniform(torch.tensor([0.], device=device), torch.tensor([np.pi * 2.], device=device))
        self.I = self.I.to(device)
        return
