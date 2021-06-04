import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn

from loggers.plotting.DW4 import PlotAllSamples
from models.FullyConnectedEnergyNet import FCRegularEnergyNet, FCENInvariantEnergyNet
from samplers.svgd_sampling.kernels.maps.maps import ENInvariantMap
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, InvariantScalarKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed


class MultiDimerPotential(object):
    """This code is copied from "Equivariant Flows: Exact Likelihood Generative Learning for Symmetric Densities"""
    def __init__(
        self, n_particles, n_dims, distance_offset=0.0, double_well_coeffs=None,
        temperature=1.
    ):
        self._n_particles = n_particles
        self._n_dims = n_dims

        if double_well_coeffs is None:
            double_well_coeffs = {"a": 1.0, "b": -6.0, "c": 1.0}
        self._double_well_coeffs = double_well_coeffs

        self._distance_offset = distance_offset
        self._temperature = temperature

    def double_well_energy_torch(self, x, a=1.0, b=-6.0, c=1.0):
        double_well = a * x[:, 0] + b * (x[:, 0] ** 2) + c * (x[:, 0] ** 4)
        harmonic_oscillator = 0.5 * x[:, 1:].pow(2).sum(dim=-1)
        return double_well + harmonic_oscillator


    def distances_from_vectors(self, r, eps=1e-6):
        return (r.pow(2).sum(dim=-1) + eps).sqrt()

    def diagonal_filter(self, n, m, cuda_device=None):
        filt = (torch.eye(n, m) == 0).view(n, m)
        if cuda_device is not None:
            filt = filt.to(cuda_device)
        return filt

    def order_index(self, init_dim, n_tile, cuda_device=None):
        order_index = np.concatenate(
            [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
        )
        order_index = torch.LongTensor(order_index)
        if cuda_device is not None:
            order_index = order_index.to(cuda_device)
        return order_index

    def tile(self, a, dim, n_tile, reindex=False):
        """
        Tiles a pytorch tensor along one an arbitrary dimension.

        Parameters
        ----------
        a : PyTorch tensor
            the tensor which is to be tiled
        dim : integer like
            dimension along the tensor is tiled
        n_tile : integer like
            number of tiles

        Returns
        -------
        b : PyTorch tensor
            the tensor with dimension `dim` tiled `n_tile` times
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        if reindex:
            index = self.order_index(init_dim, n_tile, a.device)
            a = torch.index_select(a, dim, index)
        return a

    def distance_vectors(self, x, remove_diagonal=True):
        """
        Computes the matrix `d` of all distance vectors between
        given input points where

            ``d_{ij} = x_{i} - x{j}``

        Parameters
        ----------
        x : PyTorch tensor
            Tensor of shape `[n_batch, n_particles, n_dimensions]`
            containing input points.
        remove_diagonal : boolean
            Flag indicating whether the all-zero distance vectors
            `x_i - x_i` should be included in the result

        Returns
        -------
        d : PyTorch tensor
            All-distnance matrix d.
            If `remove_diagonal=True` this is a tensor of shape
                `[n_batch, n_particles, n_particles, n_dimensions]`.
            Otherwise this is a tensor of shape
                `[n_batch, n_particles, n_particles - 1, n_dimensions]`.
        """
        r = self.tile(x.unsqueeze(2), 2, x.shape[1])
        r = r - r.permute([0, 2, 1, 3])
        filt = self.diagonal_filter(x.shape[1], x.shape[1], r.device)
        if remove_diagonal:
            r = r[:, filt].view(
                -1, x.shape[1], x.shape[1] - 1, x.shape[2]
            )
        return r

    def _energy_torch(self, x):
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dims)

        dists = self.distances_from_vectors(
            self.distance_vectors(x.view(-1, self._n_particles, self._n_dims))
        )
        dists = dists.view(-1, 1)

        dists = dists - self._distance_offset

        energies = self.double_well_energy_torch(dists, **self._double_well_coeffs) / self._temperature
        return energies.view(n_batch, -1).sum(-1)

    def _energy_numpy(self, x):
        x = torch.Tensor(x)
        return self._energy_torch(x).cpu().numpy()

    def energy(self, x):
        return self._energy_torch(x)

    def forward(self, x):
        return self.energy(x)

    def log_prob(self, x):
        return -self.energy(x)

set_seed(42)

# Get device
print("#### VERSION INFORMATION ####")
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
print("#############################\n\n\n")

SAMPLER = "invariant" # or invariant
MODEL = "invariant"
fixed = False
correct = False
prefix = "fixed" if fixed else "original"

# Save parameters
BASE_PATH = "results/EBM/true/"
sampling_path = BASE_PATH + SAMPLER + "_" + MODEL

# Experiment parameters
BATCH_SIZE = 1024 # original
# BATCH_SIZE = 128 # fixed

# Sampling parameters
# SAMPLING_H = 0.1 # fixed regular
# SAMPLING_H = 0.001 # fixed invariant
SAMPLING_H = 0.0001 # original invariant -> Correct
# SAMPLING_H = 2.0 # original invariant -> Incorrect
SAMPLING_LR = {
    0: 0.1
}
SAMPLING_STEPS = 50000
SAMPLING_EPSILON = 0.0000001

# Initialize training samples
double_well_coeffs = {
    "a": 0,
    "b": -4,
    "c": 0.9

}

potential = MultiDimerPotential(4, 2, distance_offset=4, double_well_coeffs=double_well_coeffs)
if MODEL == "potential":
    target = potential
elif MODEL == "regular":
    base_block = nn.Sequential(
        nn.Linear(4 * 2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    target = FCRegularEnergyNet(base_block=base_block)
    target.load_state_dict(torch.load(f"models/weights/DW4/{prefix}/regular"))
else:
    base_block = nn.Sequential(
        nn.Linear(4 * 2, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    target = FCENInvariantEnergyNet(4, 2, base_block=base_block)
    target.load_state_dict(torch.load(f"models/weights/DW4/{prefix}/invariant"))

# Sample starting points
starting = torch.distributions.Uniform(torch.tensor([-5., -5., -5., -5., -5., -5., -5., -5.,]), torch.tensor([5., 5., 5., 5., 5., 5., 5., 5.])).sample_n(BATCH_SIZE)

# Get sampler
base_kernel = RBFKernel(h=SAMPLING_H)
if SAMPLER == "regular" or SAMPLER == "potential":
    kernel = base_kernel
else:
    kernel = InvariantScalarKernel(base_kernel, ENInvariantMap)

current_lr = SAMPLING_LR[0]
sampler = SVGDSampler(target, kernel, current_lr, SAMPLING_STEPS, epsilon=SAMPLING_EPSILON)
samples = sampler.sample(starting).detach()

potentials = potential.energy(samples)

minimimal_states = samples.detach().clone()
minimimal_states.requires_grad = True
opt = torch.optim.SGD([minimimal_states], lr=0.001)
for step in range(0, 1000):
    opt.zero_grad()
    en = potential.energy(minimimal_states)
    en.sum().backward()
    opt.step()
minimimal_states = minimimal_states.detach()
minimal_energy = potential.energy(minimimal_states)

plt.figure()
ax = plt.gca()
ax.set_facecolor('black')

plotter = PlotAllSamples(sampling_path + "/runs", interval=1)
plotter.execute(samples, None, 0)

plotter = PlotAllSamples(sampling_path + "/minimal", interval=1)
plotter.execute(minimimal_states, None, 0)
