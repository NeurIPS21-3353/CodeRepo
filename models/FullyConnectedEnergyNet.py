import numpy as np
import torch

from models.FullyConnected import FCBaseNet


class FCRegularEnergyNet(FCBaseNet):
    def __init__(self, input_dim=2, output_dim=1, base_block=None):
        super().__init__(input_dim, output_dim, base_block)

    def forward(self, x):
        out = self.base_block(x)
        out = torch.log(1 + out ** 2)
        return out.squeeze()

    def log_prob(self, x):
        return -self.forward(x)


class FCONInvariantEnergyNet(FCBaseNet):
    def __init__(self, input_dim=1, output_dim=1, base_block=None):
        super().__init__(input_dim, output_dim, base_block)

    def forward(self, x):
        # Make invariant
        inp = torch.norm(x, dim=-1).unsqueeze(-1)

        out = self.base_block(inp)
        out = torch.log(1 + out ** 2)
        return out.squeeze()

    def log_prob(self, x):
        return -self.forward(x)


class FCCNInvariantEnergyNet(FCBaseNet):
    def __init__(self, input_dim=1, output_dim=1, N=4, base_block=None):
        super().__init__(input_dim, output_dim, base_block)
        self.N = N

        pis = torch.arange(0, self.N) * ((2. * np.pi) / self.N)
        self.g = torch.zeros((2, 2, self.N))
        for idx, pi in enumerate(pis):
            self.g[:, :, idx] = torch.tensor([[np.cos(pi), -np.sin(pi)],
                                              [np.sin(pi), np.cos(pi)]])

    def forward(self, x):
        # Make invariant
        gx2 = torch.matmul(self.g.T[:, None, :, :], x[:, :, None]).squeeze()
        inp = gx2.view((-1, gx2.size(2)))

        out = self.base_block(inp).view(gx2.size())
        out = torch.mean(out, dim=0)
        out = torch.log(1 + out ** 2)
        return out.squeeze()

    def log_prob(self, x):
        return -self.forward(x)


class FCENInvariantEnergyNet(FCBaseNet):
    def __init__(self, n_particles, dim_particles, input_dim=1, output_dim=1, base_block=None):
        super().__init__(input_dim, output_dim, base_block)
        self.n_particles = n_particles
        self.dim_particles = dim_particles

    def forward(self, x):
        # Make invariant
        inp_1 = x.view(-1, self.n_particles, self.dim_particles)

        # Center particle around (0,0), this makes it translation invariation
        CoM_1 = torch.mean(inp_1, dim=1)
        centered_1 = inp_1 - CoM_1[:, None, :]

        # Order the particles, this makes it permutation invariant
        r_1 = torch.sqrt(centered_1[:, :, 0] ** 2 + centered_1[:, :, 1] ** 2)  # [:, :, None]
        order_1 = torch.argsort(r_1)
        sorted_1 = centered_1[torch.arange(order_1.size(0)).unsqueeze(1).repeat((1, order_1.size(1))), order_1]

        # Rotate particles such that the furthest particle lies on the x-axis, this makes it rotation invariant
        top_1 = sorted_1[:, -1]
        top_r_1 = r_1.max(dim=1)[0]
        norm_1 = top_1 / top_r_1[:, None]
        t0 = torch.cat([norm_1[:, 0, None], -norm_1[:, 1, None]], dim=1).to(x.device).T
        t1 = torch.cat([norm_1[:, 1, None], norm_1[:, 0, None]], dim=1).to(x.device).T
        G_1 = torch.stack([t0, t1]).to(x.device).permute([2, 1, 0])[:, None, :, :]
        rotated_1 = torch.matmul(G_1, sorted_1[:, :, :, None]).squeeze()
        s_1 = sorted_1.size()
        input_1 = rotated_1.view((s_1[0], s_1[1] * s_1[2]))

        out = self.base_block(input_1)
        out = torch.log(1 + out ** 2)
        return out.squeeze()

    def log_prob(self, x):
        return -self.forward(x)