import os

import matplotlib
import numpy as np
import torch

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm


class TwoDimPlotter():
    def __init__(self, target, location, rotation=None, interval=1, range=5.5, steps=200):
        self.steps = steps
        self.target = target
        self.location = location
        self.rotation = rotation
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
        # z = self.target.get_z_probabilities(x, y)

        clev = torch.arange(-0.02, float(z.max()+0.001), .001)
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches((5, 5))
        plt.contourf(x, y, z, clev, cmap='magma')

        if id.size(1) > 0:
            if self.rotation is not None and self.rotation == "rotate":
                rot1 = id @ rot
                rot2 = id @ rot @ rot
                rot3 = id @ rot @ rot @ rot
                plt.scatter(id[:, 0], id[:, 1], color='firebrick', edgecolors='white', linewidths=1)
                plt.scatter(rot1[:, 0], rot1[:, 1], color='cornflowerblue', edgecolors='white', linewidths=1)
                plt.scatter(rot2[:, 0], rot2[:, 1], color='lightgreen', edgecolors='white', linewidths=1)
                plt.scatter(rot3[:, 0], rot3[:, 1], color='khaki', edgecolors='white', linewidths=1)
            elif self.rotation is not None and self.rotation == "mode":
                for idx, sample in enumerate(id):
                    x = sample[0]
                    y = sample[1]
                    while sample[0] < 0 or torch.abs(sample[1]) > sample[0]:
                        sample = sample @ rot
                    id[idx, :] = sample
                plt.scatter(id[:, 0], id[:, 1], color='firebrick', edgecolors='white', linewidths=1)
            elif self.rotation is not None and self.rotation == "line":
                norms = torch.norm(id, dim=1)
                id = torch.zeros_like(id)
                id[:, 0] = norms
                id = id[norms <= r]
                plt.scatter(id[:, 0], id[:, 1], color='firebrick', edgecolors='white', linewidths=1)
            else:
                plt.scatter(id[:, 0], id[:, 1], color='firebrick', edgecolors='white', linewidths=1)
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


class ThreeDimPlotter():
    def __init__(self, target, location, show_rotation, interval=1, range=5.5):
        self.target = target
        self.location = location
        self.show_rotation = show_rotation
        self.interval = interval
        self.range = range

        self.plotting_samples = self.target.sample(5000)

        self.image_list = []

        if not os.path.exists(location):
            os.makedirs(location, exist_ok=True)

    def execute(self, samples, true_samples, epoch):
        if epoch % self.interval != 0:
            return

        r = self.range
        cmap = cm.get_cmap('magma')
        dist_color = cmap(0.99)
        back_color = cmap(0.1)

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches((5, 5))
        matplotlib.rc('axes', edgecolor=(1, 1, 1, 0))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.plotting_samples[:, 0], self.plotting_samples[:, 1], self.plotting_samples[:, 2], color=dist_color, edgecolors=dist_color, s=10, alpha=0.1)
        if not self.show_rotation:
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], color='firebrick', edgecolors='white', linewidths=1, s=30)
        else:
            norms = samples.norm(dim=1)
            normed = torch.zeros_like(samples)
            normed[:, 0] = norms
            ax.scatter(normed[:, 0], normed[:, 1], normed[:, 2], color='firebrick', edgecolors='white', linewidths=1, s=30)

        ax.xaxis.set_pane_color((1, 1, 1, 0))
        ax.yaxis.set_pane_color((1, 1, 1, 0))
        ax.zaxis.set_pane_color((1, 1, 1, 0))

        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.)
        ax.set_facecolor(back_color)

        ax.tick_params(axis='x', labelsize=0, color=(1, 1, 1, 0))
        ax.tick_params(axis='y', labelsize=0, color=(1, 1, 1, 0))
        ax.tick_params(axis='z', labelsize=0, color=(1, 1, 1, 0))
        ax.set_xlim([-9., 9.])
        ax.set_ylim([-9., 9.])
        ax.set_zlim([-9., 9.])
        ax.dist = 7
        plt.tight_layout()
        # plt.show()

        path = f"{self.location}/epoch_{epoch}.png"
        plt.savefig(path, dpi=300)
        self.image_list.append(Image.open(path).convert("RGB"))

    def convert_to_gif(self):
        self.image_list[0].save(f'{self.location}/loop.gif',
                       save_all=True, append_images=self.image_list[1:], optimize=False, duration=40, loop=0)