import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from distributions.symmetric_distributions import RotationDistribution
from loggers.plotting.JEM import ClassificationPlotter2Dim, JointEnergy
from models.FullyConnected import FCRegularNet, FCCNInvariantNet
from models.JointEnergyModels import JEM, CondJEM
from samplers.generators.BatchSampleGenerators import BatchStartingPointSampleGenerator
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, CNInvariantKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed


class TargetModel():
    def __init__(self):
        self.distribution_inner = RotationDistribution(4., 4, 7)
        self.distribution_outer = RotationDistribution(0.5, 4, 15)

    def forward(self, x, y=None):
        combined = self.classify(x)
        if y is None:
            out = combined.logsumexp(1)
            return out
        else:
            return combined[y]

    def energy(self, x, y=None):
        return self.forward(x, y)

    def classify(self, x):
        energy_1 = self.distribution_inner.log_prob(x)
        energy_2 = self.distribution_outer.log_prob(x)
        logits = torch.vstack([energy_1, energy_2]).T
        return logits


def generate_data(target_model, n_samples, device, plot=False):
    n_samples = int(n_samples / 2)

    x_train_1 = target_model.distribution_inner.sample(n_samples)
    x_train_2 = target_model.distribution_outer.sample(n_samples)
    y_train_1 = torch.zeros(n_samples)
    y_train_2 = torch.zeros(n_samples) + 1

    x_test_1 = target_model.distribution_inner.sample(n_samples)
    x_test_2 = target_model.distribution_outer.sample(n_samples)
    y_test_1 = torch.zeros(n_samples)
    y_test_2 = torch.zeros(n_samples) + 1

    x_train = torch.cat([x_train_1, x_train_2]).to(device)
    y_train = torch.cat([y_train_1, y_train_2]).to(device).long()

    x_test = torch.cat([x_test_1, x_test_2]).to(device)
    y_test = torch.cat([y_test_1, y_test_2]).to(device).long()

    order_train = torch.tensor(np.random.permutation(np.arange(0, n_samples * 2)), device=x_train.device)
    x_train = x_train[order_train]
    y_train = y_train[order_train]

    order_test = torch.tensor(np.random.permutation(np.arange(0, n_samples * 2)), device=x_train.device)
    x_test = x_test[order_test]
    y_test = y_test[order_test]

    if plot:
        plt.scatter(x_train_1[:, 0], x_train_1[:, 1], color='red', marker='o')
        plt.scatter(x_train_2[:, 0], x_train_2[:, 1], color='blue', marker='o')
        plt.scatter(x_test_1[:, 0], x_test_1[:, 1], color='red', marker='*')
        plt.scatter(x_test_2[:, 0], x_test_2[:, 1], color='blue', marker='*')
        plt.close('all')

    return x_train, y_train, x_test, y_test


def plot_final_visualizations(target, samples, labels, generated, path, epoch):
    generated = torch.cat(generated)
    base_path = f"{path}/{epoch}/"
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)

    split = int(len(samples)/2)
    steps = 200
    r = 20

    x_range = [-r, r]
    y_range = [-r, r]

    # Plot sampled samples
    mask_r = labels == 0
    mask_b = labels == 1

    # Filter samples outof range
    if generated.size(0) > 0:
        id = generated[(torch.abs(generated) <= r).all(dim=1)]
    else:
        id = generated

    x = torch.linspace(x_range[0], x_range[1], steps)
    y = torch.linspace(y_range[0], y_range[1], steps)

    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    inp = torch.tensor(pos.astype('float32')).view((-1, 2))

    logits = target.classify(inp)

    joint_prob = torch.exp(logits.logsumexp(1)).view(steps, steps).detach()
    class_0_prob = torch.exp(logits[:, 0]).view(steps, steps).detach()
    class_1_prob = torch.exp(logits[:, 1]).view(steps, steps).detach()

    # Joint
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches((6, 6))
    clev = torch.arange(0., float(joint_prob.max() + 0.001), .001)
    plt.contourf(x, y, joint_prob, clev, cmap='magma')
    plt.scatter(x_train[mask_r, 0], x_train[mask_r, 1], edgecolors='w', color='r', alpha=.4, s=100)
    plt.scatter(x_train[mask_b, 0], x_train[mask_b, 1], edgecolors='w', color='b', alpha=.4, s=100)
    if id.size(0) > 0:
        plt.scatter(id[:split, 0], id[:split, 1], marker='*', edgecolors='w', color="r", alpha=1., s=250)
        plt.scatter(id[split:, 0], id[split:, 1], marker='*', edgecolors='w', color="b", alpha=1., s=250)
    plt.axis('off')
    plt.tight_layout(True)
    path = f"{base_path}joint.png"
    plt.savefig(path)
    # plt.show()

    # class 0
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches((6, 6))
    clev = torch.arange(0., float(class_0_prob.max() + 0.001), .001)
    plt.contourf(x, y, class_0_prob, clev, cmap='magma')
    plt.scatter(x_train[mask_r, 0], x_train[mask_r, 1], edgecolors='w', color='r', alpha=.4, s=100)
    if id.size(0) > 0:
        plt.scatter(id[:split, 0], id[:split, 1], marker='*', edgecolors='w', color="r", alpha=1., s=250)
    plt.axis('off')
    plt.tight_layout(True)
    path = f"{base_path}class_0.png"
    plt.savefig(path)
    # plt.show()

    # class 1
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches((6, 6))
    clev = torch.arange(0., float(class_1_prob.max() + 0.001), .001)
    plt.contourf(x, y, class_1_prob, clev, cmap='magma')
    plt.scatter(x_train[mask_b, 0], x_train[mask_b, 1], edgecolors='w', color='b', alpha=.4, s=100)
    if id.size(0) > 0:
        plt.scatter(id[split:, 0], id[split:, 1], marker='*', edgecolors='w', color="b", alpha=1., s=250)
    plt.axis('off')
    plt.tight_layout(True)
    path = f"{base_path}class_1.png"
    plt.savefig(path)

    z = logits.argmax(dim=1) == 1
    z = z.view(steps, steps).detach()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches((6, 6))

    # Plot classification boundary of model
    plt.contourf(x, y, z, cmap=ListedColormap(['r', 'b']), alpha=.1)
    # Plot sampled samples
    plt.scatter(x_train[mask_r, 0], x_train[mask_r, 1], edgecolors='w', color='r', alpha=.4, s=100)
    plt.scatter(x_train[mask_b, 0], x_train[mask_b, 1], edgecolors='w', color='b', alpha=.4, s=100)

    # Plot training samples
    if id.size(0) > 0:
        plt.scatter(id[:split, 0], id[:split, 1], marker='*', edgecolors='w', color="r", alpha=1., s=250)
        plt.scatter(id[split:, 0], id[split:, 1], marker='*', edgecolors='w', color="b", alpha=1., s=250)

    plt.axis('off')
    plt.tight_layout(True)
    path = f"{base_path}boundary.png"
    plt.savefig(path)

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
BASE_PATH = "./results/JEM/4Gaussians/"
sampling_path = BASE_PATH + SAMPLER + "_" + MODEL

if not os.path.exists(sampling_path):
    os.makedirs(sampling_path, exist_ok=True)

# Experiment parameters
N_SAMPLES = 128
BATCH_SIZE = 32

# Sampling parameters
SAMPLING_H = 0.1
SAMPLING_LR = {
    0: 0.9
}
SAMPLING_STEPS = 10001
SAMPLING_EPSILON = 0.0001

# Training parameters
TRAIN_LR = {
    0: 0.001
}
TRAIN_EPOCHS = 501

# Get training samples
target_model = TargetModel()
x_train, y_train, x_test, y_test = generate_data(target_model, N_SAMPLES, device)
plot_final_visualizations(target_model, x_train, y_train, [torch.tensor([])], sampling_path, "true")

# Create models
if MODEL == "regular":
    base_model = FCRegularNet(output_dim=2)
else:
    base_model = FCCNInvariantNet(input_dim=2, output_dim=2)
jem_model = JEM(base_model)
jem_cond_model = CondJEM(jem_model)

# Setup plotting
plt.figure()
plotter_true_rot = ClassificationPlotter2Dim(target_model, x_train, y_train, BASE_PATH + "/true/class", interval=1, range=20., steps=200)
plotter_true_joint = JointEnergy(target_model, x_train, y_train, BASE_PATH + "/true/probs", interval=1, range=20., steps=200)
plotter_true_rot.execute(torch.tensor([[]]), x_train, 0)
plotter_true_joint.execute(torch.tensor([[]]), x_train, 0)

plotter = ClassificationPlotter2Dim(jem_model, x_train, y_train, sampling_path + "/svgd/class", interval=5000, range=20., steps=200)
plotter_joint = JointEnergy(jem_model, x_train, y_train, sampling_path + "/svgd/probs", interval=5000, range=20., steps=200)

epoch_plotter = ClassificationPlotter2Dim(jem_cond_model, x_train, y_train, sampling_path + f"/sampling/class", interval=1000, range=20., steps=200)
epoch_plotter_joint = JointEnergy(jem_cond_model, x_train, y_train, sampling_path + f"/sampling/probs", interval=1000, range=20., steps=200)

# Create sampler
base_kernel = RBFKernel(h=SAMPLING_H)
if SAMPLER == "regular":
    kernel = base_kernel
else:
    kernel = CNInvariantKernel(base_kernel)

current_lr = SAMPLING_LR[0]
sampler = SVGDSampler(jem_model, kernel, current_lr, SAMPLING_STEPS, epsilon=SAMPLING_EPSILON, loggers=[plotter, plotter_joint])
epoch_sampler = SVGDSampler(jem_cond_model, kernel, current_lr, SAMPLING_STEPS*5, loggers=[epoch_plotter, epoch_plotter_joint])

# Intitialize batch generator
sample_generator = BatchStartingPointSampleGenerator(x_train, sampler, device=device, persistent=True, persistent_reset=0.05)

# Setup training
optimizer = torch.optim.Adam(base_model.parameters(), lr=TRAIN_LR[0])

for epoch in range(0, TRAIN_EPOCHS):
    print(epoch)

    batch_start = 0
    order = torch.tensor(np.random.permutation(np.arange(0, N_SAMPLES)), device=x_train.device)

    sample_generator.next_epoch()

    while batch_start < N_SAMPLES:
        optimizer.zero_grad()

        batch_end = batch_start + BATCH_SIZE
        if batch_end > N_SAMPLES: batch_end = N_SAMPLES
        print(batch_start, batch_end)

        # Get the true and false samples
        samples_true = x_train[order[batch_start:batch_end]].detach()
        samples_false = sample_generator.next_batch(samples_true).detach()
        labels = y_train[order[batch_start:batch_end]].detach()

        # Train the energy model
        optimizer.zero_grad()

        outputs_true = jem_model.energy(samples_true)
        outputs_false = jem_model.energy(samples_false)

        torch.autograd.backward(outputs_true.mean(), retain_graph=True)
        torch.autograd.backward(-outputs_false.mean())

        predictions = jem_model.classify(samples_true)
        loss = torch.nn.CrossEntropyLoss()(predictions, labels)
        loss.backward()

        optimizer.step()

        # Update the batches
        batch_start = batch_end

    with torch.no_grad():
        # Train set
        outputs_train = jem_model.classify(x_train)
        predictions_train = torch.argmax(outputs_train, dim=1)
        acc_train = (predictions_train == y_train).sum() / predictions_train.size(0)
        loss_train = torch.nn.CrossEntropyLoss()(outputs_train, y_train)

        # Test set
        outputs_test = jem_model.classify(x_test)
        predictions_test = torch.argmax(outputs_test, dim=1)
        acc_test = (predictions_test == y_test).sum() / predictions_test.size(0)
        loss_test = torch.nn.CrossEntropyLoss()(outputs_test, y_test)

        print(f"[{epoch}] -- train_loss:{loss_train}, test_loss:{loss_test}, train_acc:{acc_train}, test_acc:{acc_test}")

    torch.save(jem_model.state_dict(), f"{sampling_path}/model_{epoch}")

    if epoch % 50 == 0:
        print("Starting sampling")
        generated = []
        for c in [0, 1]:
            print(f"Class {c}")
            jem_cond_model.set_class(c)

            # Starting samples
            starting = torch.distributions.Uniform(torch.tensor([-20., -20.]), torch.tensor([20., 20.])).sample_n(BATCH_SIZE * 2)

            final_samples = epoch_sampler.sample(starting, prefix=f"{epoch}_{c}_")
            generated.append(final_samples)

        plot_final_visualizations(jem_model, x_train, y_train, generated, sampling_path, epoch)

    # Learning rate scheduling
    if epoch + 1 in SAMPLING_LR.keys():
        current_lr = SAMPLING_LR[epoch + 1]
        sample_generator.sampler.lr = SAMPLING_LR[epoch + 1]
    if epoch + 1 in TRAIN_LR.keys():
        for param_group in optimizer.param_groups:
            param_group['lr'] = TRAIN_LR[epoch + 1]


