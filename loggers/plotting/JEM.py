import os

import matplotlib
import numpy as np
import torch

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
from matplotlib.colors import ListedColormap


class ClassificationPlotter2Dim():
    def __init__(self, target, samples, labels, location, interval=1, range=5.5, steps=500):
        self.steps = steps
        self.samples = samples
        self.labels = labels
        self.target = target
        self.location = location
        self.interval = interval
        self.range = range

        self.image_list = []

        if not os.path.exists(location):
            os.makedirs(location, exist_ok=True)

    def execute(self, samples, true_samples, epoch, prefix=""):
        if epoch % self.interval != 0:
            return

        r = self.range
        x_range =[-r, r]
        y_range =[-r, r]

        # Filter samples outof range
        id = samples[(torch.abs(samples) <= r).all(dim=1)]

        x = torch.linspace(x_range[0], x_range[1], self.steps)
        y = torch.linspace(y_range[0], y_range[1], self.steps)

        x, y = np.meshgrid(x, y)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        inp = torch.tensor(pos.astype('float32')).view((-1, 2))
        logits = self.target.classify(inp)
        z = logits.argmax(dim=1) == 1
        z = z.view(self.steps, self.steps).detach()

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches((6, 6))

        # Plot classification boundary of model
        plt.contourf(x, y, z, cmap=ListedColormap(['r', 'b']), alpha=.1)

        # Plot sampled samples
        mask_r = self.labels == 0
        mask_b = self.labels == 1
        plt.scatter(self.samples[mask_r, 0], self.samples[mask_r, 1], edgecolors='r', color='r', alpha=0.1)
        plt.scatter(self.samples[mask_b, 0], self.samples[mask_b, 1], edgecolors='b', color='b', alpha=0.1)

        # Plot training samples
        if id.size(1) > 0:
            plt.scatter(id[:, 0], id[:, 1], color="gray", alpha=0.8)

        plt.axis('off')
        plt.tight_layout(True)
        # plt.show()

        path = f"{self.location}/{prefix}epoch_{epoch}.png"
        plt.savefig(path)
        self.image_list.append(Image.open(path).convert("RGB"))

    def convert_to_gif(self):
        self.image_list[0].save(f'{self.location}/loop.gif',
                       save_all=True, append_images=self.image_list[1:], optimize=False, duration=40, loop=0)

    def reset(self):
        self.image_list = []


class JointEnergy():
    def __init__(self, target, samples, labels, location, interval=1, range=5.5, steps=500):
        self.steps = steps
        self.samples = samples
        self.labels = labels
        self.target = target
        self.location = location
        self.interval = interval
        self.range = range

        self.image_list = []

        if not os.path.exists(location):
            os.makedirs(location, exist_ok=True)

    def execute(self, samples, true_samples, epoch, prefix=""):
        if epoch % self.interval != 0:
            return

        r = self.range
        x_range =[-r, r]
        y_range =[-r, r]

        # Plot sampled samples
        mask_r = self.labels == 0
        mask_b = self.labels == 1

        # Filter samples outof range
        id = samples[(torch.abs(samples) <= r).all(dim=1)]

        x = torch.linspace(x_range[0], x_range[1], self.steps)
        y = torch.linspace(y_range[0], y_range[1], self.steps)

        x, y = np.meshgrid(x, y)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        inp = torch.tensor(pos.astype('float32')).view((-1, 2))

        logits = self.target.classify(inp)

        joint_prob = torch.exp(logits.logsumexp(1)).view(self.steps, self.steps).detach()
        class_0_prob = torch.exp(logits[:, 0]).view(self.steps, self.steps).detach()
        class_1_prob = torch.exp(logits[:, 1]).view(self.steps, self.steps).detach()

        # z = logits.argmax(dim=1) == 1
        # z = z.view(self.steps, self.steps).detach()

        # Joint
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches((6, 6))
        clev = torch.arange(0., float(joint_prob.max() + 0.001), .001)
        plt.contourf(x, y, joint_prob, clev, cmap='magma')
        plt.scatter(self.samples[mask_r, 0], self.samples[mask_r, 1], edgecolors='w', color='r', alpha=1.)
        plt.scatter(self.samples[mask_b, 0], self.samples[mask_b, 1], edgecolors='w', color='b', alpha=1.)
        if id.size(1) > 0:
            plt.scatter(id[:, 0], id[:, 1], color="gray", alpha=0.8)
        plt.axis('off')
        plt.tight_layout(True)
        path = f"{self.location}/{prefix}epoch_{epoch}.png"
        plt.savefig(path)
        # plt.show()

        # class 0
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches((6, 6))
        clev = torch.arange(0., float(class_0_prob.max() + 0.001), .001)
        plt.contourf(x, y, class_0_prob, clev, cmap='magma')
        plt.scatter(self.samples[mask_r, 0], self.samples[mask_r, 1], edgecolors='w', color='r', alpha=1.)
        if id.size(1) > 0:
            plt.scatter(id[:, 0], id[:, 1], color="gray", alpha=0.8)
        plt.axis('off')
        plt.tight_layout(True)
        path = f"{self.location}/{prefix}epoch_{epoch}_class_0.png"
        plt.savefig(path)
        # plt.show()

        # class 1
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches((6, 6))
        clev = torch.arange(0., float(class_1_prob.max() + 0.001), .001)
        plt.contourf(x, y, class_1_prob, clev, cmap='magma')
        plt.scatter(self.samples[mask_b, 0], self.samples[mask_b, 1], edgecolors='w', color='b', alpha=1.)
        if id.size(1) > 0:
            plt.scatter(id[:, 0], id[:, 1], color="gray", alpha=0.8)
        plt.axis('off')
        plt.tight_layout(True)
        path = f"{self.location}/{prefix}epoch_{epoch}_class_1.png"
        plt.savefig(path)
        # plt.show()

        # # Plot classification boundary of model
        # plt.contourf(x, y, z, cmap=ListedColormap(['r', 'b']), alpha=.1)
        #
        # # Plot sampled samples
        # mask_r = self.labels == 0
        # mask_b = self.labels == 1
        # plt.scatter(self.samples[mask_r, 0], self.samples[mask_r, 1], edgecolors='r', color='r', alpha=0.1)
        # plt.scatter(self.samples[mask_b, 0], self.samples[mask_b, 1], edgecolors='b', color='b', alpha=0.1)
        #
        # # Plot training samples
        # if id.size(1) > 0:
        #     plt.scatter(id[:, 0], id[:, 1], color="gray", alpha=0.8)
        #
        # plt.axis('off')
        # plt.tight_layout(True)
        # # plt.show()
        #
        # path = f"{self.location}/{prefix}epoch_{epoch}.png"
        # plt.savefig(path)

    def convert_to_gif(self):
        self.image_list[0].save(f'{self.location}/loop.gif',
                       save_all=True, append_images=self.image_list[1:], optimize=False, duration=40, loop=0)

    def reset(self):
        self.image_list = []
