import json
import os

import torch
import torch.distributions as D
import matplotlib.pyplot as plt

from distributions.symmetric_distributions import TwoCircleDistribution
from loggers.plotting.basics import TwoDimPlotter
from samplers.svgd_sampling.kernels.maps.maps import ONInvariantMap
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, InvariantScalarKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

set_seed(42)

USE_SAVE_FOLDER = "results/ablation/ll/circles_3/"
N_SAMPLES = 20
EPOCHS = 5000
SHOW_ROTATION = False
H = 0.1

def get_starting_points(experiment, distribution, n_samples):
    # Reset the seed so that we have the same starting samples every time
    torch.manual_seed(42)
    if len(experiment.split(sep='x')) > 1:
        times = int(experiment.split(sep='x')[-1])
    else:
        times = 1

    initial_samples = distribution.sample_n(n_samples * times)
    return initial_samples

def get_kernel(experiment, base_kernel):
    if experiment == "invariant":
        return InvariantScalarKernel(base_kernel, ONInvariantMap)
    else:
        return base_kernel

def get_path(experiment, base_path):
    path = base_path + experiment
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if not os.path.exists(path + "/without_rotation"):
        os.makedirs(path + "/without_rotation", exist_ok=True)
    if not os.path.exists(path + "/with_rotation"):
        os.makedirs(path + "/with_rotation", exist_ok=True)
    return path


class LLLogger():
    def __init__(self, target_distribution, path):
        self.target = target_distribution
        self.path = path
        self.log = []

    def reset(self):
        self.log = []

    def execute(self, xs, original, epoch):
        ll = self.target.log_prob(xs).mean().item()
        self.log.append(ll)

    def save(self):
        data = {
            "log": self.log,
        }
        with open(path + '/ll.txt', 'w+') as outfile:
            json.dump(data, outfile)


# Load the target distribution
target_distribution = TwoCircleDistribution()

# Define starting distribution
sample_dist = D.Uniform(torch.tensor([-8.,-8.]), torch.tensor([8.,8.]))

# Construct the SVGD sampler
base_kernel = RBFKernel(h=H)

ll = {
    "regular": [],
    "regular x2": [],
    "regular x4": [],
    "regular x8": [],
    "regular x16": [],
    "regular x32": [],
    "invariant": [],
}

plotter_rot = TwoDimPlotter(target_distribution, USE_SAVE_FOLDER + "with_rotation", 'line', interval=100, range=10.)
plotter_not_rot = TwoDimPlotter(target_distribution, USE_SAVE_FOLDER + "without_rotation", None, interval=100, range=10.)
logger = LLLogger(target_distribution, USE_SAVE_FOLDER)

for exp in ll.keys():
    path = get_path(exp, USE_SAVE_FOLDER)
    plotter_rot.location = path + "/with_rotation"
    plotter_not_rot.location = path + "/without_rotation"
    logger.path = path

    Xs = get_starting_points(exp, sample_dist, N_SAMPLES)
    kernel = get_kernel(exp, base_kernel)

    sampler = SVGDSampler(target_distribution, kernel, 0.05, EPOCHS, loggers=[plotter_not_rot, plotter_rot, logger])
    samples = sampler.sample(Xs)

    # Save results
    plotter_rot.convert_to_gif()
    plotter_not_rot.convert_to_gif()
    logger.save()

    plotter_rot.reset()
    plotter_not_rot.reset()
    logger.reset()

# Create final plot
target_samples = target_distribution.sample(5000)
true_ll = target_distribution.log_prob(target_samples).mean()
colors = [
    "palevioletred",
    "magenta",
    "purple",
    "darkviolet",
    "mediumslateblue",
    "blue",
    "red"
]
plt.clf()
with plt.style.context("ggplot"):
    plt.figure(figsize=(8, 4))
    plt.plot(range(0, EPOCHS), [true_ll] * EPOCHS, label='true', linestyle='dashed', color='black')
    index = 0
    for exp in ll.keys():
        path = get_path(exp, USE_SAVE_FOLDER)
        with open(path + '/ll.txt') as json_file:
            data = json.load(json_file)
        plt.plot(range(0, EPOCHS), data['log'], label=exp, color=colors[index])
        index +=1

    plt.legend()
    plt.ylabel("Log-likelihood")
    plt.xlabel("Step")
    plt.tight_layout()
    plt.savefig(USE_SAVE_FOLDER + "ll.pdf")

