import torch
import numpy as np
import matplotlib.pyplot as plt
import logging

from qm9.satorras.data.dataset import ProcessedDataset
from qm9.satorras.data.prepare.download import prepare_dataset


def show_molecule(coordinates, atoms):
    CC = {
        1: "white",  # H
        6: "black",  # C
        7: "dark-blue",  # N
        8: "red",  # O
        9: "green"  # F
    }

    ax = plt.axes(projection='3d')
    # Hydrogen
    for n in [1, 6, 7, 8, 9]:
        ax.scatter3D(coordinates[(atoms == n), 0], coordinates[(atoms == n), 1], coordinates[(atoms == n), 2],
                     color=torch.sum((atoms == n))*[CC[n]], edgecolors='black', s=50)
    plt.show()

def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False, total_charge=68):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, dataset, subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    updated_datasets = {}
    profile = None
    for lab in ['train', 'valid', 'test']:
        dataset = datasets[lab]# Train has to be checked first
        # Get the right isomer
        t = (dataset['charges'].sum(dim=1) == total_charge)
        t = torch.where(t)[0]

        n_H = (dataset['charges'][t] == 1).sum(dim=1)
        n_C = (dataset['charges'][t] == 6).sum(dim=1)
        n_N = (dataset['charges'][t] == 7).sum(dim=1)
        n_O = (dataset['charges'][t] == 8).sum(dim=1)
        n_F = (dataset['charges'][t] == 9).sum(dim=1)
        stack = torch.vstack((n_H, n_C, n_N, n_O, n_F)).T
        uni, indices, counts = torch.unique(stack, dim=0, return_counts=True, return_inverse=True)
        if profile is None:
            profile = uni[torch.argmax(counts)]
        t0 = torch.where((stack == profile).all(dim=1))
        indices = t[t0]

        updated_datasets[lab] = {}
        for item in dataset.items():
            updated_datasets[lab][item[0]] = dataset[item[0]][indices]

    datasets = updated_datasets

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)
    all_species = torch.tensor([1, 6, 7, 8, 9])

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    return args, datasets, num_species, max_charge


def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels should be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
