import numpy as np
import torch
import matplotlib.pyplot as plt

from loggers.plotting.DW4 import Plot64Samples
from models.FullyConnectedEnergyNet import FCRegularEnergyNet, FCENInvariantEnergyNet
from samplers.generators.BatchSampleGenerators import BatchUniformSampleGenerator
from samplers.svgd_sampling.kernels.maps.maps import ENInvariantMap
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, InvariantScalarKernel
from math import ceil
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed
from torch import nn


set_seed(42)

# Get device
print("#### VERSION INFORMATION ####")
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
print("#############################\n\n\n")

SAMPLER = "invariant" # or invariant
MODEL = "invariant"
FIXED = False

# Save parameters
BASE_PATH = "./results/EBM/DW4/"
sampling_path = BASE_PATH + SAMPLER + "_" + MODEL

# Experiment parameters
N_SAMPLES = 1000
BATCH_SIZE = 64

# Sampling parameters
# SAMPLING_H = 0.1 # fixed regular
# SAMPLING_H = 0.001 # fixed invariant
# SAMPLING_H = 0.1 # original regular
SAMPLING_H = 0.001 # original invariant
SAMPLING_LR = {
    0: 0.1
}
SAMPLING_STEPS = 5001
SAMPLING_EPSILON = 0.00001

# Training parameters
TRAIN_LR = {
    0: 0.01
}
TRAIN_EPOCHS = 51

# Initialize training samples
if FIXED:
    data = np.load('data/fixed_DW4.npy')
    x_train = torch.tensor(data)
else:
    data = np.load("data/dimer_4particles.npy")[(1000-N_SAMPLES):]
    x_train = torch.tensor(data)

# Get model
base_block = nn.Sequential(
            nn.Linear(4 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
if MODEL == "regular":
    model = FCRegularEnergyNet(base_block=base_block)
else:
    model = FCENInvariantEnergyNet(4, 2, base_block=base_block)

# Setup logging and plotting
plt.figure()
plotter_true = Plot64Samples(BASE_PATH + "/true", interval=1)
plotter_true.execute(x_train, None, epoch=0)

# Get sampler
base_kernel = RBFKernel(h=SAMPLING_H)
if SAMPLER == "regular":
    kernel = base_kernel
else:
    kernel = InvariantScalarKernel(base_kernel, ENInvariantMap)

current_lr = SAMPLING_LR[0]
sampler = SVGDSampler(model, kernel, current_lr, SAMPLING_STEPS, epsilon=SAMPLING_EPSILON)

# Intitialize batch generator
sample_generator = BatchUniformSampleGenerator(x_train, sampler, device=device, persistent=True, persistent_reset=0.10,
                                               min=[-5., -5., -5., -5., -5., -5., -5., -5.], max=[5., 5., 5., 5., 5., 5., 5., 5.])

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR[0])

# Training loop
for epoch in range(0, TRAIN_EPOCHS):
    # Reset epoch variables
    batch_start = 0
    order = torch.tensor(np.random.permutation(np.arange(0, N_SAMPLES)), device=x_train.device)
    sample_generator.next_epoch()

    while batch_start < N_SAMPLES:
        batch_end = batch_start + BATCH_SIZE
        if batch_end > N_SAMPLES:
            batch_start = batch_end
            continue

        # Get true and false samples
        samples_true = x_train[order[batch_start:batch_end]].detach()
        samples_false = sample_generator.next_batch(samples_true).detach()

        # Train energy model
        optimizer.zero_grad()

        outputs_true = model(samples_true)
        outputs_false = model(samples_false)

        # Get Maximum likelihood gradients
        torch.autograd.backward(outputs_true.mean(), retain_graph=True)
        torch.autograd.backward(-outputs_false.mean())

        # Take step
        optimizer.step()

        batch_start = batch_end

        print(f"[{epoch}] {sample_generator.batch}/{ceil(N_SAMPLES/BATCH_SIZE)}")

    if (epoch % 10) == 0:
        plotter_rot_epochs = Plot64Samples(sampling_path + f"/{epoch}/", interval=1000)
        torch.save(model.state_dict(), f"{sampling_path}/{epoch}/model")

        # Starting samples
        starting = torch.distributions.Uniform(torch.tensor([-5., -5., -5., -5., -5., -5., -5., -5.,]), torch.tensor([5., 5., 5., 5., 5., 5., 5., 5.])).sample_n(BATCH_SIZE)

        # Sample uniform
        epoch_sampler = SVGDSampler(model, kernel, current_lr, SAMPLING_STEPS * 10, loggers=[plotter_rot_epochs])
        final_samples = epoch_sampler.sample(starting)

    # Learning rate scheduler
    if epoch + 1 in SAMPLING_LR.keys():
        current_lr = SAMPLING_LR[epoch + 1]
        sample_generator.sampler.lr = SAMPLING_LR[epoch + 1]
    if epoch + 1 in TRAIN_LR.keys():
        for param_group in optimizer.param_groups:
            param_group['lr'] = TRAIN_LR[epoch + 1]

