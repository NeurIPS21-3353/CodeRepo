import torch

from qm9.satorras import dataset
from qm9.satorras.models import EGNN
from qm9.utils import generate_starting_samples, ModelWrap, QMBatchSampleGenerator, format_data
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, ScalarParticleKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

set_seed(42)

print("#### VERSION INFORMATION ####")
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
print("#############################\n\n\n")

# Dataloading parameters
_total_charge = 46
_batch_size = 16
_num_workers = 0
_all_species = torch.tensor([1, 6, 7, 8, 9], device=device)
_n_samples = _batch_size
_charge_power = 2

# Model parameters
_nf = 64
_n_layers = 4
_attention = 1
_batch_norm = 0
_node_attr = 0
_dropout = 0

# Sampling parameters
_sampling_lr_schedule = {
        0: 0.5
    }
_sampler_epochs = 2501
_sampler_burn_in = 0
_sampler_epsilon = 0.001
_sampler_h = 1.

# Training parameters
_training_epochs = 2500
_training_lr_schedule = {
        0: 0.01,
        250: 0.005,
        1000: 0.001,
}

dataloaders, charge_scale = dataset.retrieve_dataloaders(_batch_size, _num_workers, _total_charge)
_n_nodes = dataloaders['train'].dataset.data['num_atoms'][0]
_atoms = dataloaders['train'].dataset.data['charges'][0, :_n_nodes]
total = _n_samples * _n_nodes

coords, atoms, node_embeddings, node_mask, edges, edge_mask = generate_starting_samples(_n_samples, _n_nodes, _atoms, _charge_power, _all_species, charge_scale, device)

# Create model
original_model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=_nf, device=device,
                 n_layers=_n_layers, coords_weight=1.0, attention=_attention, bn=_batch_norm, node_attr=_node_attr, dropout=_dropout, output_dim=1)
original_model.to(device)

# Wrap Model for data
wrapped_model = ModelWrap(original_model, _n_samples, node_embeddings, edges, node_mask, edge_mask, _n_nodes)
wrapped_model.to(device)

# Define sampler
k = RBFKernel(h=_sampler_h)
kernel = ScalarParticleKernel(k, atoms, n_particles=_n_nodes)
kernel.to(device)

sampler = SVGDSampler(wrapped_model, kernel, _sampling_lr_schedule[0], _sampler_epochs, epsilon=_sampler_epsilon)
generator = QMBatchSampleGenerator(None, sampler, _n_samples, device, starting_samples=coords.clone().view((_batch_size, -1)), persistent=True)

# Setup training
optimizer = torch.optim.Adam(original_model.parameters(), lr=_training_lr_schedule[0])

# Train
for epoch in range(0, _training_epochs):
    print(epoch)
    # Statistics
    correct = 0
    count = 0

    generator.next_epoch()

    for i, data in enumerate(dataloaders['train']):
        if len(data['num_atoms']) != _batch_size:
            continue

        atom_positions, nodes, atom_mask, edges, edge_mask, labels, n_nodes_dataset = format_data(data, device, _charge_power, charge_scale)
        false_samples = generator.next_batch(atom_positions.detach()).detach()

        optimizer.zero_grad()
        energies_true = wrapped_model(x=atom_positions, h0=nodes, edges=edges, edge_attr=None,
                                                node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes_dataset)
        energies_false = wrapped_model(false_samples)

        print(f"[{i}] true:{energies_true.mean()}, false:{energies_false.mean()}")

        torch.autograd.backward(energies_true.mean()) # This takes the gradient and stores it in the grad variable of the parameters
        torch.autograd.backward(-energies_false.mean()) # The minus is there to make sure that this gradient is substracted instead of added.

        optimizer.step()

        # For every batch we have a 20% chance of resetting the persistence
        # We do not use the regular reset because we need to fix the model.
        if torch.rand(1)[0] < 0.20:
            print("Resetting")
            coords, atoms, node_embeddings, node_mask, edges, edge_mask = generate_starting_samples(_n_samples, _n_nodes, _atoms, _charge_power,
                                                                                                    _all_species, charge_scale, device)
            generator.previous = coords.view((_batch_size, -1))
            wrapped_model.edge_mask = edge_mask
            wrapped_model.node_mask = node_mask
            wrapped_model.edges = edges
            wrapped_model.node_embeddings = node_embeddings

    torch.save(wrapped_model.state_dict(), f"model_{epoch}")

            # Learning rate scheduling
    if epoch + 1 in _sampling_lr_schedule.keys():
        generator.sampler.lr = _sampling_lr_schedule[epoch + 1]
    if epoch + 1 in _training_lr_schedule.keys():
        for param_group in optimizer.param_groups:
            param_group['lr'] = _training_lr_schedule[epoch + 1]