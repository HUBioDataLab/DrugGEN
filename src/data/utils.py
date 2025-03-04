import os
import pickle

import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as geoutils

from rdkit import Chem, RDLogger



def label2onehot(labels, dim, device):
    """Convert label indices to one-hot vectors."""
    out = torch.zeros(list(labels.size())+[dim]).to(device)
    out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)

    return out.float()


def get_encoders_decoders(raw_file1, raw_file2, max_atom):
    """
    Given two raw SMILES files, either load the atom and bond encoders/decoders
    if they exist (naming them based on the file names) or create and save them.

    Parameters:
        raw_file1 (str): Path to the first SMILES file.
        raw_file2 (str): Path to the second SMILES file.
        max_atom (int): Maximum allowed number of atoms in a molecule.

    Returns:
        atom_encoder (dict): Mapping from atomic numbers to indices.
        atom_decoder (dict): Mapping from indices to atomic numbers.
        bond_encoder (dict): Mapping from bond types to indices.
        bond_decoder (dict): Mapping from indices to bond types.
    """
    # Determine unique suffix based on the two file names (alphabetically sorted for consistency)
    name1 = os.path.splitext(os.path.basename(raw_file1))[0]
    name2 = os.path.splitext(os.path.basename(raw_file2))[0]
    sorted_names = sorted([name1, name2])
    suffix = f"{sorted_names[0]}_{sorted_names[1]}"

    # Define encoder/decoder directories and file paths
    enc_dir = os.path.join("DrugGEN", "data", "encoders")
    dec_dir = os.path.join("DrugGEN", "data", "decoders")
    atom_encoder_path = os.path.join(enc_dir, f"atom_{suffix}.pkl")
    atom_decoder_path = os.path.join(dec_dir, f"atom_{suffix}.pkl")
    bond_encoder_path = os.path.join(enc_dir, f"bond_{suffix}.pkl")
    bond_decoder_path = os.path.join(dec_dir, f"bond_{suffix}.pkl")

    # If all files exist, load and return them
    if (os.path.exists(atom_encoder_path) and os.path.exists(atom_decoder_path) and 
        os.path.exists(bond_encoder_path) and os.path.exists(bond_decoder_path)):
        with open(atom_encoder_path, "rb") as f:
            atom_encoder = pickle.load(f)
        with open(atom_decoder_path, "rb") as f:
            atom_decoder = pickle.load(f)
        with open(bond_encoder_path, "rb") as f:
            bond_encoder = pickle.load(f)
        with open(bond_decoder_path, "rb") as f:
            bond_decoder = pickle.load(f)
        print("Loaded existing encoders/decoders!")
        return atom_encoder, atom_decoder, bond_encoder, bond_decoder

    # Otherwise, create the encoders/decoders
    print("Creating new encoders/decoders...")
    # Read SMILES from both files (assuming one SMILES per row, no header)
    smiles1 = pd.read_csv(raw_file1, header=None)[0].tolist()
    smiles2 = pd.read_csv(raw_file2, header=None)[0].tolist()
    smiles_combined = smiles1 + smiles2

    atom_labels = set()
    bond_labels = set()
    max_length = 0
    filtered_smiles = []
    
    # Process each SMILES: keep only valid molecules with <= max_atom atoms
    for smiles in tqdm(smiles_combined, desc="Processing SMILES"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        molecule_size = mol.GetNumAtoms()
        if molecule_size > max_atom:
            continue
        filtered_smiles.append(smiles)
        # Collect atomic numbers
        atom_labels.update([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        max_length = max(max_length, molecule_size)
        # Collect bond types
        bond_labels.update([bond.GetBondType() for bond in mol.GetBonds()])
    
    # Add a PAD symbol (here using 0 for atoms)
    atom_labels.add(0)
    atom_labels = sorted(atom_labels)
    
    # For bonds, prepend the PAD bond type (using rdkit's BondType.ZERO)
    bond_labels = sorted(bond_labels)
    bond_labels = [Chem.rdchem.BondType.ZERO] + bond_labels

    # Create encoder and decoder dictionaries
    atom_encoder = {l: i for i, l in enumerate(atom_labels)}
    atom_decoder = {i: l for i, l in enumerate(atom_labels)}
    bond_encoder = {l: i for i, l in enumerate(bond_labels)}
    bond_decoder = {i: l for i, l in enumerate(bond_labels)}

    # Ensure directories exist
    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(dec_dir, exist_ok=True)

    # Save the encoders/decoders to disk
    with open(atom_encoder_path, "wb") as f:
        pickle.dump(atom_encoder, f)
    with open(atom_decoder_path, "wb") as f:
        pickle.dump(atom_decoder, f)
    with open(bond_encoder_path, "wb") as f:
        pickle.dump(bond_encoder, f)
    with open(bond_decoder_path, "wb") as f:
        pickle.dump(bond_decoder, f)

    print("Encoders/decoders created and saved.")
    return atom_encoder, atom_decoder, bond_encoder, bond_decoder

def load_molecules(data=None, b_dim=32, m_dim=32, device=None, batch_size=32):
    data = data.to(device)
    a = geoutils.to_dense_adj(
        edge_index = data.edge_index,
        batch=data.batch,
        edge_attr=data.edge_attr,
        max_num_nodes=int(data.batch.shape[0]/batch_size)
    )
    x_tensor = data.x.view(batch_size,int(data.batch.shape[0]/batch_size),-1)
    a_tensor = label2onehot(a, b_dim, device)

    a_tensor_vec = a_tensor.reshape(batch_size,-1)
    x_tensor_vec = x_tensor.reshape(batch_size,-1)
    real_graphs = torch.concat((x_tensor_vec,a_tensor_vec),dim=-1)

    return real_graphs, a_tensor, x_tensor