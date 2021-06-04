import torch
import torch.distributions as D
import matplotlib.pyplot as plt

from distributions.symmetric_distributions import RotationDistribution
from loggers.plotting.basics import TwoDimPlotter
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, CNInvariantKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

set_seed(42)

USE_INVARIANT_SAMPLER = False
USE_SAVE_FOLDER = "./results/sampling/synthetic_data/C4/regular"
N_SAMPLES = 50
SHOW_ROTATION = True
H = 0.2

# Load the target distribution
target_distribution = RotationDistribution(5, 4, mean=3)

# Sample starting points
sample_dist = D.Normal(torch.tensor([0.,0.]), torch.tensor([2.,2.]))
Xs = sample_dist.sample_n(N_SAMPLES)

# Construct the SVGD sampler
base_kernel = RBFKernel(h=H)
if USE_INVARIANT_SAMPLER:
    kernel = CNInvariantKernel(base_kernel)
else:
    kernel = base_kernel

# Create plotter
fig = plt.figure()
plotter_rot = TwoDimPlotter(target_distribution, USE_SAVE_FOLDER + "/rotated", rotation='rotate', interval=100)
plotter_mode = TwoDimPlotter(target_distribution, USE_SAVE_FOLDER + "/mode", rotation='mode', interval=100)
plotter = TwoDimPlotter(target_distribution, USE_SAVE_FOLDER + "/normal",  interval=100)

sampler = SVGDSampler(target_distribution, kernel, 0.02, 25000, loggers=[plotter_rot, plotter_mode, plotter])
samples = sampler.sample(Xs)

# Save the gif
plotter_rot.convert_to_gif()
plotter_mode.convert_to_gif()
plotter.convert_to_gif()
