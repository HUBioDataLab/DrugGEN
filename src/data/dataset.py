import os
import os.path as osp
import re
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset

from rdkit import Chem, RDLogger

from src.data.utils import label2onehot

RDLogger.DisableLog('rdApp.*') 


class DruggenDataset(InMemoryDataset):
    def __init__(self, root, dataset_file, raw_files, max_atom, features, 
                 atom_encoder, atom_decoder, bond_encoder, bond_decoder,
                 transform=None, pre_transform=None, pre_filter=None):
        """
        Initialize the DruggenDataset with pre-loaded encoder/decoder dictionaries.

        Parameters:
            root (str): Root directory.
            dataset_file (str): Name of the processed dataset file.
            raw_files (str): Path to the raw SMILES file.
            max_atom (int): Maximum number of atoms allowed in a molecule.
            features (bool): Whether to include additional node features.
            atom_encoder (dict): Pre-loaded atom encoder dictionary.
            atom_decoder (dict): Pre-loaded atom decoder dictionary.
            bond_encoder (dict): Pre-loaded bond encoder dictionary.
            bond_decoder (dict): Pre-loaded bond decoder dictionary.
            transform, pre_transform, pre_filter: See PyG InMemoryDataset.
        """
        self.dataset_name = dataset_file.split(".")[0]
        self.dataset_file = dataset_file
        self.raw_files = raw_files
        self.max_atom = max_atom
        self.features = features

        # Use the provided encoder/decoder mappings.
        self.atom_encoder_m = atom_encoder
        self.atom_decoder_m = atom_decoder
        self.bond_encoder_m = bond_encoder
        self.bond_decoder_m = bond_decoder

        self.atom_num_types = len(atom_encoder)
        self.bond_num_types = len(bond_encoder)

        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, dataset_file)
        self.data, self.slices = torch.load(path)
        self.root = root

    @property
    def processed_dir(self):
        """
        Returns the directory where processed dataset files are stored.
        """
        return self.root
    
    @property
    def raw_file_names(self):
        """
        Returns the raw SMILES file name.
        """
        return self.raw_files

    @property
    def processed_file_names(self):
        """
        Returns the name of the processed dataset file.
        """
        return self.dataset_file

    def _filter_smiles(self, smiles_list):
        """
        Filters the input list of SMILES strings to keep only valid molecules that:
         - Can be successfully parsed,
         - Have a number of atoms less than or equal to the maximum allowed (max_atom),
         - Contain only atoms present in the atom_encoder,
         - Contain only bonds present in the bond_encoder.

        Parameters:
            smiles_list (list): List of SMILES strings.

        Returns:
            max_length (int): Maximum number of atoms found in the filtered molecules.
            filtered_smiles (list): List of valid SMILES strings.
        """
        max_length = 0
        filtered_smiles = []
        for smiles in tqdm(smiles_list, desc="Filtering SMILES"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Check molecule size
            molecule_size = mol.GetNumAtoms()
            if molecule_size > self.max_atom:
                continue

            # Filter out molecules with atoms not in the atom_encoder
            if not all(atom.GetAtomicNum() in self.atom_encoder_m for atom in mol.GetAtoms()):
                continue

            # Filter out molecules with bonds not in the bond_encoder
            if not all(bond.GetBondType() in self.bond_encoder_m for bond in mol.GetBonds()):
                continue

            filtered_smiles.append(smiles)
            max_length = max(max_length, molecule_size)
        return max_length, filtered_smiles

    def _genA(self, mol, connected=True, max_length=None):
        """
        Generates the adjacency matrix for a molecule based on its bond structure.

        Parameters:
            mol (rdkit.Chem.Mol): The molecule.
            connected (bool): If True, ensures all atoms are connected.
            max_length (int, optional): The size of the matrix; if None, uses number of atoms in mol.

        Returns:
            np.array: Adjacency matrix with bond types as entries, or None if disconnected.
        """
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        A = np.zeros((max_length, max_length))
        begin = [b.GetBeginAtomIdx() for b in mol.GetBonds()]
        end = [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]
        A[begin, end] = bond_type
        A[end, begin] = bond_type
        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)
        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):
        """
        Generates the feature vector for each atom in a molecule by encoding their atomic numbers.

        Parameters:
            mol (rdkit.Chem.Mol): The molecule.
            max_length (int, optional): Length of the feature vector; if None, uses number of atoms in mol.

        Returns:
            np.array: Array of atom feature indices, padded with zeros if necessary, or None on error.
        """
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        try:
            return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] +
                            [0] * (max_length - mol.GetNumAtoms()))
        except KeyError as e:
            print(f"Skipping molecule with unsupported atom: {e}")
            print(f"Skipped SMILES: {Chem.MolToSmiles(mol)}")
            return None

    def _genF(self, mol, max_length=None):
        """
        Generates additional node features for a molecule using various atomic properties.

        Parameters:
            mol (rdkit.Chem.Mol): The molecule.
            max_length (int, optional): Number of rows in the features matrix; if None, uses number of atoms.

        Returns:
            np.array: Array of additional features for each atom, padded with zeros if necessary.
        """
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        features = np.array([[*[a.GetDegree() == i for i in range(5)],
                               *[a.GetExplicitValence() == i for i in range(9)],
                               *[int(a.GetHybridization()) == i for i in range(1, 7)],
                               *[a.GetImplicitValence() == i for i in range(9)],
                               a.GetIsAromatic(),
                               a.GetNoImplicit(),
                               *[a.GetNumExplicitHs() == i for i in range(5)],
                               *[a.GetNumImplicitHs() == i for i in range(5)],
                               *[a.GetNumRadicalElectrons() == i for i in range(5)],
                               a.IsInRing(),
                               *[a.IsInRingSize(i) for i in range(2, 9)]]
                              for a in mol.GetAtoms()], dtype=np.int32)
        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def decoder_load(self, dictionary_name, file):
        """
        Returns the pre-loaded decoder dictionary based on the dictionary name.

        Parameters:
            dictionary_name (str): Name of the dictionary ("atom" or "bond").
            file: Placeholder parameter for compatibility.

        Returns:
            dict: The corresponding decoder dictionary.
        """
        if dictionary_name == "atom":
            return self.atom_decoder_m
        elif dictionary_name == "bond":
            return self.bond_decoder_m
        else:
            raise ValueError("Unknown dictionary name.")

    def matrices2mol(self, node_labels, edge_labels, strict=True, file_name=None):
        """
        Converts graph representations (node labels and edge labels) back to an RDKit molecule.

        Parameters:
            node_labels (iterable): Encoded atom labels.
            edge_labels (np.array): Adjacency matrix with encoded bond types.
            strict (bool): If True, sanitizes the molecule and returns None on failure.
            file_name: Placeholder parameter for compatibility.

        Returns:
            rdkit.Chem.Mol: The resulting molecule, or None if sanitization fails.
        """
        mol = Chem.RWMol()
        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(self.atom_decoder_m[node_label]))
        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), self.bond_decoder_m[edge_labels[start, end]])
        if strict:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                mol = None
        return mol

    def check_valency(self, mol):
        """
        Checks that no atom in the molecule has exceeded its allowed valency.

        Parameters:
            mol (rdkit.Chem.Mol): The molecule.

        Returns:
            tuple: (True, None) if valid; (False, atomid_valence) if there is a valency issue.
        """
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True, None
        except ValueError as e:
            e = str(e)
            p = e.find('#')
            e_sub = e[p:]
            atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
            return False, atomid_valence

    def correct_mol(self, mol):
        """
        Corrects a molecule by removing bonds until all atoms satisfy their valency limits.

        Parameters:
            mol (rdkit.Chem.Mol): The molecule.

        Returns:
            rdkit.Chem.Mol: The corrected molecule.
        """
        while True:
            flag, atomid_valence = self.check_valency(mol)
            if flag:
                break
            else:
                # Expecting two numbers: atom index and its valence.
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                queue = []
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                queue.sort(key=lambda tup: tup[1], reverse=True)
                if queue:
                    start = queue[0][2]
                    end = queue[0][3]
                    mol.RemoveBond(start, end)
        return mol


    def process(self, size=None):
        """
        Processes the raw SMILES file by filtering and converting each valid SMILES into a PyTorch Geometric Data object.
        The resulting dataset is saved to disk.

        Parameters:
            size (optional): Placeholder parameter for compatibility.

        Side Effects:
            Saves the processed dataset as a file in the processed directory.
        """
        # Read raw SMILES from file (assuming CSV with no header)
        smiles_list = pd.read_csv(self.raw_files, header=None)[0].tolist()
        max_length, filtered_smiles = self._filter_smiles(smiles_list)
        data_list = []
        self.m_dim = len(self.atom_decoder_m)
        for smiles in tqdm(filtered_smiles, desc='Processing dataset', total=len(filtered_smiles)):
            mol = Chem.MolFromSmiles(smiles)
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                x_array = self._genX(mol, max_length=max_length)
                if x_array is None:
                    continue
                x = torch.from_numpy(x_array).to(torch.long).view(1, -1)
                x = label2onehot(x, self.m_dim).squeeze()
                if self.features:
                    f = torch.from_numpy(self._genF(mol, max_length=max_length)).to(torch.long).view(x.shape[0], -1)
                    x = torch.concat((x, f), dim=-1)
                adjacency = torch.from_numpy(A)
                edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adjacency[edge_index[0], edge_index[1]].to(torch.long)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)
        torch.save(self.collate(data_list), osp.join(self.processed_dir, self.dataset_file))