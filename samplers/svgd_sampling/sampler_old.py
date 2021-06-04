from abc import ABC, abstractmethod

from samplers.svgd_sampling.SVGD.MatrixSVGD import MatrixSVGD
from samplers.svgd_sampling.kernels.matrix_kernels import ApproximationRotationEquivariantMatrixKernel, IdentityMatrixKernel, \
    VectorizedRotationEquivariantMatrixKernel, ContinousRotationEquivariantMatrixKernel, CoupledParticleInvariantMatrixKernel
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel

import torch

from utils.plotting.plot_2d_energy_model import plot_2d_energy_model, plot_2d_energy_model_ctu, plot_2d_JEM

import matplotlib.pyplot as plt

def plot_first_16_samples(samples, epoch, title="", save_file=None):
    fig, axs = plt.subplots(8, 8, figsize=(10, 10))
    # fig.tight_layout()
    sample_idx = 0
    samples = samples.view(-1, 4, 2).detach().numpy()
    for i in range(0, 8):
        for j in range(0, 8):
            axs[i, j].scatter(samples[sample_idx, :, 0], samples[sample_idx, :, 1])
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])
            max_ = max(axs[i, j].get_ylim()[1], axs[i, j].get_xlim()[1]) * 1.2
            min_ = min(axs[i, j].get_ylim()[0], axs[i, j].get_xlim()[0]) * 1.2
            axs[i, j].set_ylim([min_, max_])
            axs[i, j].set_xlim([min_, max_])
            sample_idx += 1

    fig.suptitle(title)
    if save_file is None or save_file == "":
        plt.draw()
    else:
        plt.savefig(f"{save_file}{epoch}.png")
        plt.close(fig)


class BaseSVGDSampler(ABC):
    def __init__(self, energy_model, lr, epochs, plot_type="reg", epsilon=None):
        self.model = energy_model
        self.lr = lr
        self.epochs = epochs
        self.plot_type = plot_type
        self.epsilon = epsilon

    @abstractmethod
    def svgd_step(self, xs, optimizer, true_samples):
        pass

    @abstractmethod
    def to(self, device):
        pass

    def sample(self, x, plot=None, plot_file="", burn_in=-1, true_samples=None):
        epochs = burn_in if burn_in != -1 else self.epochs
        plot_file = plot_file +"_burn_in_" if burn_in != -1 and plot_file != "" else plot_file

        original_samples = x.detach().clone()

        Xs = x.detach().clone()
        opt = torch.optim.SGD([Xs], lr=self.lr)

        for i in range(0, epochs):
            start_Xs = Xs.detach().clone()
            Xs = self.svgd_step(Xs, opt, true_samples)

            if plot is not None and i % plot == 0:
                if self.plot_type == "DW":
                    plot_first_16_samples(Xs.detach().clone(), epoch=i, title=f"Step {i}", save_file=plot_file)
                elif self.plot_type == "JEM":
                    plot_2d_JEM(200, self.model, original_samples, Xs.detach().clone(), i, title=f"Step {i}", save_file=plot_file)
                elif self.plot_type == "reg":
                    plot_2d_energy_model(200, self.model, original_samples, Xs.detach().clone(), i, title=f"Step {i}", save_file=plot_file)
                # plot_2d_energy_model_ctu(200, self.model, original_samples, Xs, i, title=f"Step {i}", save_file=plot_file)

                t = Xs - start_Xs
                dist = torch.norm(t, dim=1, p=2).mean()
                print(dist)

            if self.epsilon is not None:
                t = Xs - start_Xs
                dist = torch.norm(t, dim=1, p=2).mean()
                if dist < self.epsilon:
                    print("it happened")
                    break

        if plot is not None:
            if self.plot_type == "DW":
                plot_first_16_samples(Xs.detach().clone(), epoch="final", title=f"Final step", save_file=plot_file)
            elif self.plot_type == "JEM":
                plot_2d_JEM(200, self.model, original_samples, Xs.detach().clone(), "final", title=f"Final step", save_file=plot_file)
            elif self.plot_type == "reg":
                plot_2d_energy_model(200, self.model, original_samples, Xs.detach().clone(), "final", title=f"Final step", save_file=plot_file)
            # plot_2d_energy_model_ctu(200, self.model, original_samples, Xs, i, title=f"Step {i}", save_file=plot_file)

        return Xs



class DiscreteSVGDSampler(BaseSVGDSampler):
    def __init__(self, energy, lr, epochs, plot_type="reg", h=0.3, n_rot=4, contrastive_SVGD=True, epsilon=None):
        super(DiscreteSVGDSampler, self).__init__(energy, lr, epochs, plot_type, epsilon)
        k = RBFKernel(h=h)
        self.kmat = VectorizedRotationEquivariantMatrixKernel(k, n_rot)
        self.svgd = MatrixSVGD(self.model, self.kmat, contrastive_SVGD=contrastive_SVGD)

    def svgd_step(self, xs, optimizer, true_samples):
        return self.svgd.construct(xs, optimizer, true_samples)

    def to(self, device):
        self.kmat.to(device)


class ContinousSVGDSampler(BaseSVGDSampler):
    def __init__(self, energy, lr, epochs, h=0.3, plot_type="reg", epsilon=None):
        super(ContinousSVGDSampler, self).__init__(energy, lr, epochs, plot_type, epsilon)
        k = RBFKernel(h=h)
        self.kmat = ContinousRotationEquivariantMatrixKernel(k)
        self.svgd = MatrixSVGD(self.model, self.kmat)

    def svgd_step(self, xs, optimizer, true_samples):
        return self.svgd.construct(xs, optimizer, true_samples)

    def to(self, device):
        self.kmat.to(device)


class ContinousApproximationSVGDSampler(BaseSVGDSampler):
    def __init__(self, energy, lr, epochs, h=0.3, samples=250, plot_type="reg", epsilon=None):
        super(ContinousApproximationSVGDSampler, self).__init__(energy, lr, epochs, plot_type, epsilon)
        k = RBFKernel(h=h)
        self.kmat = ApproximationRotationEquivariantMatrixKernel(k, samples)
        self.svgd = MatrixSVGD(self.model, self.kmat)

    def svgd_step(self, xs, optimizer, true_samples):
        return self.svgd.construct(xs, optimizer, true_samples)

    def to(self, device):
        self.kmat.to(device)


class RegularSVGDSampler(BaseSVGDSampler):
    def __init__(self, energy, lr, epochs, h=0.3, plot_type="reg", epsilon=None):
        super(RegularSVGDSampler, self).__init__(energy, lr, epochs, plot_type, epsilon)
        k = RBFKernel(h=h)
        self.kmat = IdentityMatrixKernel(k)
        self.svgd = MatrixSVGD(self.model, self.kmat)

    def svgd_step(self, xs, optimizer, true_samples):
        return self.svgd.construct(xs, optimizer, true_samples)

    def to(self, device):
        self.kmat.to(device)


class ParticleSVGDSampler(BaseSVGDSampler):
    def __init__(self, energy, lr, epochs, h=0.3, dim=2, n_particles=4, plot_type="reg", epsilon=None):
        super(ParticleSVGDSampler, self).__init__(energy, lr, epochs, plot_type, epsilon)
        k = RBFKernel(h=h)
        self.kmat = CoupledParticleInvariantMatrixKernel(k, dim=dim, n_particles=n_particles)
        self.svgd = MatrixSVGD(self.model, self.kmat)

    def svgd_step(self, xs, optimizer, true_samples):
        return self.svgd.construct(xs, optimizer, true_samples)

    def to(self, device):
        self.kmat.to(device)

