import torch
import torch.distributions as D
import matplotlib.pyplot as plt

from distributions.symmetric_distributions import TwoSphereDistribution
from loggers.plotting.basics import ThreeDimPlotter
from samplers.svgd_sampling.kernels.maps.rotation_maps import ONInvariantMap
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, InvariantScalarKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

set_seed(42)

USE_INVARIANT_SAMPLER = True
USE_SAVE_FOLDER = "results/sampling/synthetic_data/O3/invariant"
N_SAMPLES = 100
H = 0.001

# Load the target distribution
target_distribution = TwoSphereDistribution(radius_1=4, radius_2=9, thickness_1=0.3, thickness_2=0.3, mixing=[0.5, 0.5])

# Sample starting points
sample_dist = D.Uniform(torch.tensor([-8.,-8., -8.]), torch.tensor([8.,8.,8.]))
Xs = sample_dist.sample_n(N_SAMPLES)

# Construct the SVGD sampler
base_kernel = RBFKernel(h=H)
if USE_INVARIANT_SAMPLER:
    kernel = InvariantScalarKernel(base_kernel, ONInvariantMap)
else:
    kernel = base_kernel

# Create plotter
fig = plt.figure()
plotter_rot = ThreeDimPlotter(target_distribution, USE_SAVE_FOLDER + "/with_rotation", True, interval=100, range=10.)
plotter_not_rot = ThreeDimPlotter(target_distribution, USE_SAVE_FOLDER + "/without_rotation", False, interval=100, range=10.)

sampler = SVGDSampler(target_distribution, kernel, 0.02, 25000, loggers=[plotter_not_rot, plotter_rot])
samples = sampler.sample(Xs)

# Save the gif
plotter_rot.convert_to_gif()
plotter_not_rot.convert_to_gif()
