import os
import matplotlib.pyplot as plt

class PlotAllSamples():
    def __init__(self, location, interval=1):
        self.location = location
        self.interval = interval

        if not os.path.exists(location):
            os.makedirs(location, exist_ok=True)

    def execute(self, samples, true_samples, epoch, prefix=""):
        if (epoch % self.interval) != 0 : return

        idx = 0
        for sample in samples:
            sample = sample.view((4,2))
            plt.clf()
            fig = plt.gcf()
            fig.set_size_inches((5, 5))

            plt.scatter(sample[:, 0], sample[:, 1])
            plt.yticks([])
            plt.xticks([])
            max_ = max(plt.ylim()[1], plt.xlim()[1]) * 1.2
            min_ = min(plt.ylim()[0], plt.xlim()[0]) * 1.2
            plt.ylim([min_, max_])
            plt.xlim([min_, max_])

            path = f"{self.location}/{prefix}_{epoch}_{idx}.png"
            plt.savefig(path)

            idx += 1

    def convert_to_gif(self):
        return

    def reset(self):
        return

class Plot64Samples():
    def __init__(self, location, interval=1):
        self.location = location
        self.interval = interval

        if not os.path.exists(location):
            os.makedirs(location, exist_ok=True)

    def execute(self, samples, true_samples, epoch, prefix=""):
        if (epoch % self.interval) != 0 : return

        plt.clf()
        fig = plt.gcf()

        idx = 0
        for i in range(0, 8):
            for j in range(0, 8):
                sample = samples[idx].view((4,2))
                fig.set_size_inches((10, 10))
                ax = fig.add_subplot(8, 8, idx+1)

                ax.scatter(sample[:, 0], sample[:, 1])
                max_ = max(ax.get_ylim()[1], ax.get_xlim()[1]) * 1.2
                min_ = min(ax.set_ylim()[0], ax.get_xlim()[0]) * 1.2
                ax.set_ylim([min_, max_])
                ax.set_xlim([min_, max_])

                idx += 1

        path = f"{self.location}/{prefix}{epoch}.png"
        plt.savefig(path)

    def convert_to_gif(self):
        return

    def reset(self):
        return
