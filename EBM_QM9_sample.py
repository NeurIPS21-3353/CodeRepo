import torch
import numpy

from qm9.satorras import dataset
from qm9.satorras.models import EGNN
from qm9.utils import generate_starting_samples, ModelWrap
from samplers.svgd_sampling.kernels.scalar_kernels import RBFKernel, ScalarParticleKernel
from samplers.svgd_sampling.sampler import SVGDSampler
from utils.seed import set_seed

import pyvista as pv
pv.set_plot_theme("document")

set_seed(42)

print("#### VERSION INFORMATION ####")
print(torch.__version__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
print("#############################\n\n\n")

# Dataloading parameters
_total_charge = 46
_batch_size = 64
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
        0: 0.9
    }
_sampler_epochs = 25
_sampler_epsilon = 0.000001
_sampler_h = 1.


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

# Load weights
state = torch.load("models/weights/qm9/model")
wrapped_model.load_state_dict(state)

# Define sampler
k = RBFKernel(h=_sampler_h)
kernel = ScalarParticleKernel(k, atoms, n_particles=_n_nodes)
kernel.to(device)

sampler = SVGDSampler(wrapped_model, kernel, _sampling_lr_schedule[0], _sampler_epochs, epsilon=_sampler_epsilon)

# final = sampler.sample(coords.view((_batch_size, -1)))
# numpy.save("models/weights/qm9/sampled_molecules_.npy", final.detach().numpy())
final = torch.tensor(numpy.load("models/weights/qm9/sampled_molecules.npy"))
coords = final.detach().view(_batch_size, 14, 3)

index_mol = -1
p = pv.Plotter(shape=(8, 8))
for x_axis in range(0, 8):
    if index_mol > len(coords): break
    for y_axis in range(0, 8):
        p.subplot(x_axis, y_axis)
        index_mol += 1
        if index_mol >= len(coords): break

        atom = atoms[index_mol]
        coords_atoms = coords[index_mol]


        # Draw atoms
        bonds = {}
        existing_bonds = {}
        for index_atom in range(0, 14):
            existing_bonds[index_atom] = []
            if atom[index_atom] == 6:
                color = 'dimgrey'
                radius = .5
                bonds[index_atom] = 4
            elif atom[index_atom] == 8:
                color = 'red'
                radius = .5
                bonds[index_atom] = 2
            elif atom[index_atom] == 1:
                color = 'white'
                radius = .2
                bonds[index_atom] = 1
            atom_location = coords[index_mol][index_atom]

            s = pv.Sphere(radius, atom_location)
            p.add_mesh(s, color=color, show_edges=False)

        # Start creating bonds
        C_indices = torch.where(atom == 6.)[0]
        O_indices = torch.where(atom == 8.)[0]
        H_indices = torch.where(atom == 1.)[0]
        CO_indices = torch.cat((C_indices, O_indices))
        most_center = torch.argmin(torch.norm(coords_atoms[CO_indices], dim=1))
        still_available = [i.item() for i in CO_indices if bonds[i.item()] > 0]

        # Select first carbon
        not_connected = [i.item() for i in CO_indices]
        connected = []

        connected.append(most_center.item())
        not_connected.remove(most_center)

        # Get one closest
        while len(not_connected) > 0:
            # connected_distance = distances[connected, not_connected].unsqueeze(dim=0)
            connected_and_available = [con for con in connected if con in still_available]
            connected_distance = torch.cdist(coords[index_mol][connected_and_available],coords[index_mol][not_connected]) #+ torch.eye(CO_indices.size(0)) * 99
            d0, i0 = torch.min(connected_distance, dim=1)
            closest_in_connected = torch.argmin(d0)
            new_closest = i0[closest_in_connected]

            start = coords[index_mol][connected_and_available[closest_in_connected]]
            end = coords[index_mol][not_connected[new_closest]]
            cyl = pv.Cylinder(center=(start+end)/2, direction=end-start, height=torch.min(d0).item(), radius=0.1)
            p.add_mesh(cyl, color='w', show_edges=False)


            bonds[connected_and_available[closest_in_connected]] -= 1
            bonds[not_connected[new_closest]] -= 1
            existing_bonds[connected_and_available[closest_in_connected]].append(not_connected[new_closest])
            existing_bonds[not_connected[new_closest]].append(connected_and_available[closest_in_connected])
            still_available = [i.item() for i in CO_indices if bonds[i.item()] > 0]

            connected.append(not_connected[new_closest])
            not_connected.remove(not_connected[new_closest])

        # Connect all hydrogen
        not_connected = [i.item() for i in H_indices]
        still_available = [i.item() for i in CO_indices if bonds[i.item()] > 0]
        while len(not_connected) > 0:
            connected_distance = torch.cdist(coords[index_mol][still_available], coords[index_mol][not_connected])
            # print(connected_distance)
            m, i = torch.min(connected_distance, dim=0)
            furthest_H = torch.argmax(m)
            closest_CO = i[furthest_H]

            start = coords[index_mol][not_connected[furthest_H]]
            end = coords[index_mol][still_available[closest_CO]]

            cyl = pv.Cylinder(center=(start + end) / 2, direction=end - start, height=torch.max(m).item(), radius=0.1)
            p.add_mesh(cyl, color='w', show_edges=False)

            bonds[still_available[closest_CO]] -= 1
            bonds[not_connected[furthest_H]] -= 1
            existing_bonds[still_available[closest_CO]].append(not_connected[furthest_H])
            existing_bonds[not_connected[furthest_H]].append(still_available[closest_CO])
            still_available = [i.item() for i in CO_indices if bonds[i.item()] > 0]

            not_connected.remove(not_connected[furthest_H])

        # Add remaining bonds
        # print(bonds)
        still_available = [i.item() for i in CO_indices if bonds[i.item()] > 0]
        while len(still_available) > 0:
            connected_distance = torch.cdist(coords[index_mol][still_available], coords[index_mol][still_available]) + torch.eye(len(still_available)) * 99
            # print(connected_distance)
            m, i = torch.min(connected_distance, dim=0)

            # Find true closest
            closest_true = torch.argmin(m)
            closest_true_ = i[closest_true]
            start = coords[index_mol][still_available[closest_true]]
            end = coords[index_mol][still_available[closest_true_]]
            dist = torch.norm(start - end)
            margin = dist + dist * 0.1

            # Find closest that is not yet bonded
            closest_order = torch.argsort(m)
            for closest_new in closest_order[1:]:
                closest_new_ = i[closest_new]
                start_new = coords[index_mol][still_available[closest_new]]
                end_new = coords[index_mol][still_available[closest_new_]]
                dist_new = torch.norm(start - end)
                if dist_new > margin:
                    break
                if still_available[closest_new_.item()] not in existing_bonds[still_available[closest_new.item()]]: # Found a new combination
                    closest_true = closest_new
                    closest_true_ = closest_new_
                    start = start_new
                    end = end_new
                    break

            radius_multiplier = len([i for i in existing_bonds[still_available[closest_true.item()]] if i == still_available[closest_true_.item()]]) + 1
            cyl = pv.Cylinder(center=(start + end) / 2, direction=end-start, height=torch.min(m).item(), radius=0.1 * radius_multiplier)
            p.add_mesh(cyl, color='w', show_edges=False)

            bonds[still_available[closest_true]] -= 1
            bonds[still_available[closest_true_]] -= 1
            existing_bonds[still_available[closest_true]].append(still_available[closest_true_])
            existing_bonds[still_available[closest_true_]].append(still_available[closest_true])
            still_available = [i.item() for i in CO_indices if bonds[i.item()] > 0]

        print(bonds)

p.show()