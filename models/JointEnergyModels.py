import torch


class JEM(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.class_output = torch.nn.Linear(self.model.output_dim, 2)

    def forward(self, x):
        # Returns the energy
        logits = self.classify(x)
        return -logits.logsumexp(1)

    def energy(self, x):
        return self.forward(x)

    def classify(self, x):
        # Returns the class probability logits
        logits = self.model(x)
        # logits = self.class_output(penult)
        return logits

    def log_prob(self, x):
        return -self.energy(x)


class CondJEM(torch.nn.Module):
    def __init__(self, jem_model, y=0):
        super().__init__()
        self.jem_model = jem_model
        self.y = y

    def set_class(self, y):
        self.y = y

    def forward(self, x):
        logits = self.jem_model.classify(x)[:,self.y]
        return -logits

    def classify(self, x):
        logits = self.jem_model.classify(x)
        return logits

    def energy(self, x):
        return self.forward(x)

    def log_prob(self, x):
        return -self.energy(x)