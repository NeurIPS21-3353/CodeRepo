import os
import matplotlib.pyplot as plt
import numpy as np

import torch
from PIL import Image


class TwoDimPlotter():
    def __init__(self, target, location, interval=1, range=5.5, steps=200):
        self.steps = steps
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

        rot = torch.FloatTensor([[0, 1], [-1, 0]])

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
        z = self.target.log_prob(inp).exp().detach()
        z = z.view(self.steps, self.steps)

        plt.clf()
        clev = torch.arange(-0.02, float(z.max()+0.001), .001)
        fig = plt.gcf()
        fig.set_size_inches((5, 5))
        plt.contourf(x, y, z, clev, cmap='magma')

        if id.size(1) != 0:
            plt.scatter(id[:, 0], id[:, 1], color='firebrick', edgecolors='white', linewidths=1)
        if true_samples.size(1) != 0:
            plt.scatter(true_samples[:, 0], true_samples[:, 1], color='darkblue', edgecolors='white', linewidths=1)
        plt.axis('off')
        plt.tight_layout()
        # plt.show()

        path = f"{self.location}/{prefix}epoch_{epoch}.png"
        plt.savefig(path)
        self.image_list.append(Image.open(path).convert("RGB"))

    def convert_to_gif(self):
        self.image_list[0].save(f'{self.location}/loop.gif',
                       save_all=True, append_images=self.image_list[1:], optimize=False, duration=40, loop=0)

    def reset(self):
        self.image_list = []


class SlicePlotter():
    def __init__(self, target, location, interval=1, range=5.5, steps=200):
        self.steps = steps
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

        trim_samples = samples[(torch.norm(samples, dim=-1) <= self.range)]
        trim_true = true_samples[(torch.norm(true_samples, dim=-1) <= self.range)]

        ys = torch.zeros(self.steps, 2)
        ys[:, 0] = torch.linspace(0., self.range, self.steps)
        xs = self.target.log_prob(ys).exp().detach()

        xs_true = torch.norm(trim_true, dim=-1)
        ys_true = self.target.log_prob(trim_true).exp().detach()

        xs_samples = torch.norm(trim_samples, dim=-1)
        ys_samples = self.target.log_prob(trim_samples).exp().detach()

        with plt.style.context("ggplot"):
            plt.clf()
            fig = plt.gcf()
            fig.set_size_inches((6, 6))
            plt.plot(ys.norm(dim=-1), xs)
            if xs_samples.size(0) > 1:
                scat_1 = plt.scatter(xs_samples, ys_samples, color='firebrick', edgecolors='white', s=100)
            if xs_true.size(0) > 1:
                scat_2 = plt.scatter(xs_true, ys_true, color='darkblue', edgecolors='white', s=100)
            if xs_true.size(0) > 1 and xs_samples.size(0) > 1:
                plt.legend([scat_1, scat_2], ["Fake samples", "True samples"])
            plt.ylabel('Log-likelihood')
            plt.tight_layout()
            # plt.show()

        path = f"{self.location}/{prefix}epoch_{epoch}.png"
        plt.savefig(path)
        self.image_list.append(Image.open(path).convert("RGB"))

    def convert_to_gif(self):
        self.image_list[0].save(f'{self.location}/loop.gif',
                       save_all=True, append_images=self.image_list[1:], optimize=False, duration=40, loop=0)

    def reset(self):
        self.image_list = []