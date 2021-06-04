import json
import os

import torch
import torch.distributions as D
import matplotlib.pyplot as plt

from distributions.symmetric_distributions import RotationDistribution
from loggers.plotting.basics import TwoDimPlotter
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, CNInvariantKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

set_seed(42)


USE_SAVE_FOLDER = "results/ablation/initialization/4_gaussians/"
N_SAMPLES = 25
EPOCHS = 5000
H = 0.25
repeats = 25

def get_kernel(sampler, base_kernel):
    if sampler == "invariant":
        return CNInvariantKernel(base_kernel)
    else:
        return base_kernel

def get_path(sampler, run, base_path):
    path = base_path + sampler + "/" + str(run)
    if path != "" and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
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
target_distribution = RotationDistribution(5, 4, mean=3)

# Construct the SVGD sampler
base_kernel = RBFKernel(h=H)

plotter = TwoDimPlotter(target_distribution, USE_SAVE_FOLDER, None, interval=500)
logger = LLLogger(target_distribution, USE_SAVE_FOLDER)


for iteration in range(0, repeats):
    # Create random initialization
    mean = D.Uniform(torch.tensor([-5., -5.]), torch.tensor([5., 5.])).sample()
    variance = D.Uniform(torch.tensor([0.]), torch.tensor([4.])).sample()

    sample_dist = D.Normal(mean, torch.cat((variance, variance)))
    starting = sample_dist.sample_n(N_SAMPLES)

    for type in ['invariant', 'regular']:
        # Define starting distribution
        path = get_path(type, iteration, USE_SAVE_FOLDER)
        plotter.location = path
        logger.path = path

        kernel = get_kernel(type, base_kernel)
        sampler = SVGDSampler(target_distribution, kernel, 0.1, EPOCHS, loggers=[plotter, logger])
        samples = sampler.sample(starting.detach().clone())

        # Save results
        plotter.convert_to_gif()
        logger.save()

        plotter.reset()
        logger.reset()

data = []

for type in ['invariant', 'regular']:
    final_lls = []
    for iteration in range(0, repeats):
        path = get_path(type, iteration, USE_SAVE_FOLDER)
        with open(path + '/ll.txt') as json_file:
            loaded = json.load(json_file)
        print(loaded['log'][-1])
        final_lls.append(loaded['log'][-1])
    data.append(final_lls)

target_samples = target_distribution.sample(5000)
true_ll = target_distribution.log_prob(target_samples).mean()

with plt.style.context("ggplot"):
    plt.clf()
    fig, ax = plt.subplots(figsize=(3, 4))
    ax.boxplot(data)

    ax.set_ylabel("Log-Likelihood")
    ax.axhline(y=true_ll)
    plt.xticks([1, 2], ['Invariant', 'Regular'])
    plt.tight_layout()
    plt.savefig(USE_SAVE_FOLDER + "spread.pdf")

