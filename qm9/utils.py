import torch
import qm9.satorras.utils as qm9_utils

from samplers.generators.BatchSampleGenerators import BaseBatchSampleGenerator


class QMBatchSampleGenerator(BaseBatchSampleGenerator):
    def __init__(self, true_samples, sampler, n_samples, device, starting_samples, persistent=False, persistent_reset=0.0):
        super().__init__(true_samples, sampler, n_samples, device, persistent)
        self.starting_samples = starting_samples

    def _get_starting_sample(self, true_samples):
        return self.starting_samples


def generate_starting_samples(_n_samples, _n_nodes, _atoms, charge_power, _all_species, charge_scale, device):
    total = _n_samples * _n_nodes # These are the atom numbers for H, C, O, N and F

    # For if we want to start from the true_samples
    coords = torch.rand((_n_samples, _n_nodes, 3)) * 4.
    atoms = _atoms.repeat(_n_samples, 1)# torch.cat(_n_samples*[_atoms], dim=0)

    coords = coords.to(device)
    atoms = atoms.to(device)

    node_mask = torch.ones((_n_samples, _n_nodes))
    one_hot_atoms = atoms.unsqueeze(-1) == _all_species.unsqueeze(0).unsqueeze(0)
    node_embeddings = qm9_utils.preprocess_input(one_hot_atoms, atoms, charge_power, charge_scale, device).view((total, -1))  #

    edges = qm9_utils.get_adj_matrix(_n_nodes, _n_samples, device)
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)  # No self loops
    edge_mask *= diag_mask

    # Reshape
    node_mask = node_mask.view((total, 1))
    edge_mask = edge_mask.view((total * _n_nodes, 1))

    # Send to device
    node_embeddings = node_embeddings.to(device)
    node_mask = node_mask.to(device)
    edge_mask = edge_mask.to(device)

    return coords, atoms, node_embeddings, node_mask, edges, edge_mask


class ModelWrap(torch.nn.Module):
    def __init__(self, model, n_samples, node_embeddings, edges, node_mask, edge_mask, n_nodes):
        super().__init__()
        self.model = model
        self.n_samples = n_samples

        self.n_nodes = n_nodes
        self.edge_mask = edge_mask
        self.node_mask = node_mask
        self.edges = edges
        self.node_embeddings = node_embeddings

    def forward(self, x, h0=None, edges=None, edge_attr=None, node_mask=None, edge_mask=None, n_nodes=None):
        if h0 is not None:
            model_output = self.model(h0=h0, x=x.view((self.n_samples * self.n_nodes, -1)), edges=edges,
                                      edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        else:
            model_output = self.model(h0=self.node_embeddings, x=x.view((self.n_samples * self.n_nodes, -1)), edges=self.edges, edge_attr=None, node_mask=self.node_mask, edge_mask=self.edge_mask, n_nodes=self.n_nodes)
        out = torch.log(1 + model_output ** 2)
        return out

    def log_prob(self, x, h0=None, edges=None, edge_attr=None, node_mask=None, edge_mask=None, n_nodes=None):
        return -self.forward(h0=h0, x=x, edges=edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask, n_nodes=n_nodes)


def format_data(data, device, _charge_power, charge_scale):
    # Get true data
    batch_size, n_nodes, _ = data['positions'].size()
    atom_positions = data['positions']

    atom_positions = atom_positions.view(batch_size * n_nodes, -1).to(device)
    atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device)
    edge_mask = data['edge_mask'].to(device)
    one_hot = data['one_hot'].to(device)
    charges = data['charges'].to(device)
    nodes = qm9_utils.preprocess_input(one_hot, charges, _charge_power, charge_scale, device)

    nodes = nodes.view(batch_size * n_nodes, -1)
    edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
    label=None

    return atom_positions, nodes, atom_mask, edges, edge_mask, label, n_nodes
