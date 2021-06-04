import numpy as np
import torch
from torch import nn


class FCBaseNet(torch.nn.Module):
    def __init__(self, input_dim=2, output_dim=1, base_block=None):
        super().__init__()
        self.d = input_dim
        self.output_dim = output_dim

        if base_block is None:
            self.base_block = nn.Sequential(
                nn.Linear(self.d, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, self.output_dim),
            )
        else:
            self.base_block = base_block


class FCRegularNet(FCBaseNet):
    def __init__(self, input_dim=2, output_dim=1, base_block=None):
        super().__init__(input_dim, output_dim, base_block)

    def forward(self, x):
        out = self.base_block(x)
        return out.squeeze()

    def log_prob(self, x):
        return -self.forward(x)


class FCCNInvariantNet(FCBaseNet):
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
        return out.squeeze()
