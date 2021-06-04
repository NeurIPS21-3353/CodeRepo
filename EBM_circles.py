import numpy as np
import torch
import matplotlib.pyplot as plt

from distributions.symmetric_distributions import TwoCircleDistribution
from loggers.plotting.EBM import TwoDimPlotter, SlicePlotter
from models.FullyConnectedEnergyNet import FCRegularEnergyNet, FCONInvariantEnergyNet
from samplers.generators.BatchSampleGenerators import BatchStartingPointSampleGenerator
from samplers.svgd_sampling.kernels.maps.rotation_maps import ONInvariantMap
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, InvariantScalarKernel
from math import ceil
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

set_seed(42)

# Get device
print("#### VERSION INFORMATION ####")
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
print("#############################\n\n\n")

SAMPLER = "regular" # or invariant
MODEL = "invariant"

# Save parameters
BASE_PATH = "./results/EBM/circles/"
sampling_path = BASE_PATH + SAMPLER + "_" + MODEL

# Experiment parameters
N_SAMPLES = 128
BATCH_SIZE = 32

# Sampling parameters
SAMPLING_H = 0.05
SAMPLING_LR = {
    0: 0.9,
    250: 0.5,
    400: 0.1
}
SAMPLING_STEPS = 10001

# Training parameters
TRAIN_LR = {
    0: 0.001,
    150: 0.0005,
    400: 0.0001,
}
TRAIN_EPOCHS = 501
TRAIN_MSE = 0.5

# Initialize training samples
target_distribution = TwoCircleDistribution(radius_1 = 3, radius_2 = 10, thickness_1 = 0.5, thickness_2=0.5, mixing = [1., 1.])
x_train = target_distribution.sample(N_SAMPLES)
y_train = -target_distribution.log_prob(x_train).detach()
x_train = x_train.to(device)
y_train = y_train.to(device)

# Get model
if MODEL == "regular":
    model = FCRegularEnergyNet()
else:
    model = FCONInvariantEnergyNet()

# Setup logging and plotting
plt.figure()
plotter_true_rot = TwoDimPlotter(target_distribution, BASE_PATH + "/true/2d", interval=1, range=15.)
plotter_true_slice = SlicePlotter(target_distribution, BASE_PATH + "/true/line", interval=1, range=15.)
plotter_true_rot.execute(torch.tensor([[]]), x_train, 0)
plotter_true_slice.execute(torch.tensor([[]]), x_train, 0)

plotter_rot = TwoDimPlotter(model, sampling_path + "/svgd/2d", interval=10000, range=15.)
plotter_slice = SlicePlotter(model, sampling_path + "/svgd/line", interval=10000, range=15.)

# Get sampler
base_kernel = RBFKernel(h=SAMPLING_H)
if SAMPLER == "regular":
    kernel = base_kernel
else:
    kernel = InvariantScalarKernel(base_kernel, ONInvariantMap)

current_lr = SAMPLING_LR[0]
sampler = SVGDSampler(model, kernel, current_lr, SAMPLING_STEPS, loggers=[plotter_rot, plotter_slice])

# Intitialize batch generator
sample_generator = BatchStartingPointSampleGenerator(x_train, sampler, device=device, persistent=True, persistent_reset=0.05)

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
        if batch_end > N_SAMPLES: batch_end = N_SAMPLES

        # Get true and false samples
        samples_true = x_train[order[batch_start:batch_end]].detach()
        samples_false = sample_generator.next_batch(samples_true).detach()
        labels = y_train[order[batch_start:batch_end]].detach()

        # Train energy model
        optimizer.zero_grad()

        outputs_true = model(samples_true)
        outputs_false = model(samples_false)

        # Get Maximum likelihood gradients
        torch.autograd.backward(outputs_true.mean(), retain_graph=True)
        torch.autograd.backward(-outputs_false.mean())

        # Get MSE gradients
        l = TRAIN_MSE * torch.nn.MSELoss()(outputs_true, labels)
        l.backward()

        # Take step
        optimizer.step()

        batch_start = batch_end

        print(f"[{epoch}] {sample_generator.batch}/{ceil(N_SAMPLES/BATCH_SIZE)}")

        plotter_rot.reset()
        plotter_slice.reset()

    if (epoch % 50) == 0:
        torch.save(model.state_dict(), f"{sampling_path}/model_{epoch}")

        plotter_rot_epochs = TwoDimPlotter(model, sampling_path + f"/2d/{epoch}/", interval=100, range=15.)
        plotter_slice_epochs = SlicePlotter(model, sampling_path + f"/line/{epoch}/", interval=100, range=15.)

        # Starting samples
        starting = torch.distributions.Uniform(torch.tensor([-10. -10.]), torch.tensor([10., 10.])).sample_n(BATCH_SIZE)

        # Sample uniform
        epoch_sampler = SVGDSampler(model, kernel, current_lr, SAMPLING_STEPS, loggers=[plotter_rot_epochs, plotter_slice_epochs])
        final_samples = epoch_sampler.sample(starting, true_samples=x_train)

        plotter_rot_epochs.convert_to_gif()
        plotter_slice_epochs.convert_to_gif()

    # Learning rate scheduler
    if epoch + 1 in SAMPLING_LR.keys():
        current_lr = SAMPLING_LR[epoch + 1]
        sample_generator.sampler.lr = SAMPLING_LR[epoch + 1]
    if epoch + 1 in TRAIN_LR.keys():
                # There are more advanced lr schedulers, but given that we are not really familiar with how it influences stuff, I thought that
                # it was best to use a very manual version.
        for param_group in optimizer.param_groups:
            param_group['lr'] = TRAIN_LR[epoch + 1]

