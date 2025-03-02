import pickle
import os
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import (Data, InMemoryDataset)
import os.path as osp
from tqdm import tqdm
import re
from rdkit import RDLogger
import pandas as pd

RDLogger.DisableLog('rdApp.*') 
class DruggenDataset(InMemoryDataset):
    
    def __init__(self, root, dataset_file, raw_files, max_atom, features, joint_raw_file=None, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_file.split(".")[0]
        self.dataset_file = dataset_file
        self.raw_files = raw_files
        self.joint_raw_file = joint_raw_file
        self.max_atom = max_atom
        self.features = features
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, dataset_file)
        self.data, self.slices = torch.load(path)
        self.root = root

        
    @property
    def processed_dir(self):
        
        return self.root
    
    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return self.dataset_file

    def _generate_encoders_decoders(self, data, create_joint=False):
        """
        Generate encoders and decoders for atoms and bonds.
        If create_joint=True, will create joint encoders using both datasets.
        Otherwise, will use existing joint encoders or create dataset-specific ones.
        """
        self.data = data
        
        # Get the name of the other dataset if this is a joint encoder
        other_dataset = None
        if self.joint_raw_file:
            other_dataset = os.path.basename(self.joint_raw_file).split('.')[0]
        
        # Create a consistent joint suffix by sorting dataset names
        if other_dataset:
            datasets = sorted([self.dataset_name, other_dataset])
            joint_suffix = f"joint_{datasets[0]}_{datasets[1]}"
        else:
            joint_suffix = "joint"
            
        joint_encoders_path = f"DrugGEN/data/encoders/atom_{joint_suffix}.pkl"
        
        # If joint encoders exist and we're not creating them, load them
        if os.path.exists(joint_encoders_path) and not create_joint:
            print(f'Loading existing joint encoders and decoders ({joint_suffix})...')
            with open(f"DrugGEN/data/encoders/atom_{joint_suffix}.pkl", "rb") as f:
                self.atom_encoder_m = pickle.load(f)
            with open(f"DrugGEN/data/decoders/atom_{joint_suffix}.pkl", "rb") as f:
                self.atom_decoder_m = pickle.load(f)
            with open(f"DrugGEN/data/encoders/bond_{joint_suffix}.pkl", "rb") as f:
                self.bond_encoder_m = pickle.load(f)
            with open(f"DrugGEN/data/decoders/bond_{joint_suffix}.pkl", "rb") as f:
                self.bond_decoder_m = pickle.load(f)
            
            # Still need to process the current dataset's SMILES
            max_length = 0
            filtered_smiles = []
            for smiles in tqdm(data):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                molecule_size = mol.GetNumAtoms()
                if molecule_size > self.max_atom:
                    continue
                filtered_smiles.append(smiles)
                max_length = max(max_length, molecule_size)
            
            return max_length, filtered_smiles

        # Creating new encoders (either joint or dataset-specific)
        print('Creating atoms and bonds encoder and decoder...')
        
        atom_labels = set()
        bond_labels = set()
        max_length = 0
        filtered_smiles = []
        
        # Process current dataset
        for smiles in tqdm(data):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            molecule_size = mol.GetNumAtoms()
            if molecule_size > self.max_atom:
                continue
            filtered_smiles.append(smiles)
            atom_labels.update([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            max_length = max(max_length, molecule_size)
            bond_labels.update([bond.GetBondType() for bond in mol.GetBonds()])

        # If creating joint encoders, process the other dataset too
        if create_joint and self.joint_raw_file:
            joint_smiles = pd.read_csv(self.joint_raw_file, header=None)[0].tolist()
            for smiles in tqdm(joint_smiles, desc='Processing joint dataset'):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                if mol.GetNumAtoms() > self.max_atom:
                    continue
                atom_labels.update([atom.GetAtomicNum() for atom in mol.GetAtoms()])
                bond_labels.update([bond.GetBondType() for bond in mol.GetBonds()])

        atom_labels.update([0])  # add PAD symbol
        atom_labels = sorted(atom_labels)
        
        bond_labels = sorted(bond_labels)
        bond_labels = [Chem.rdchem.BondType.ZERO] + bond_labels

        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        
        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)

        print(f'Created {"joint " if create_joint else ""}atoms encoder and decoder with {self.atom_num_types - 1} atom types and 1 PAD symbol!')
        print(f'Created {"joint " if create_joint else ""}bonds encoder and decoder with {self.bond_num_types - 1} bond types and 1 PAD symbol!')
        
        # Save encoders/decoders with unique naming
        suffix = joint_suffix if create_joint else self.dataset_name
        save_dir = "DrugGEN/data"
        os.makedirs(f"{save_dir}/encoders", exist_ok=True)
        os.makedirs(f"{save_dir}/decoders", exist_ok=True)
        
        with open(f"{save_dir}/encoders/atom_{suffix}.pkl", "wb") as f:
            pickle.dump(self.atom_encoder_m, f)
        with open(f"{save_dir}/decoders/atom_{suffix}.pkl", "wb") as f:
            pickle.dump(self.atom_decoder_m, f)
        with open(f"{save_dir}/encoders/bond_{suffix}.pkl", "wb") as f:
            pickle.dump(self.bond_encoder_m, f)
        with open(f"{save_dir}/decoders/bond_{suffix}.pkl", "wb") as f:
            pickle.dump(self.bond_decoder_m, f)
        
        return max_length, filtered_smiles

    def _genA(self, mol, connected=True, max_length=None):

        max_length = max_length if max_length is not None else mol.GetNumAtoms()

        A = np.zeros(shape=(max_length, max_length))

        begin, end = [b.GetBeginAtomIdx() for b in mol.GetBonds()], [b.GetEndAtomIdx() for b in mol.GetBonds()]
        bond_type = [self.bond_encoder_m[b.GetBondType()] for b in mol.GetBonds()]

        A[begin, end] = bond_type
        A[end, begin] = bond_type

        degree = np.sum(A[:mol.GetNumAtoms(), :mol.GetNumAtoms()], axis=-1)

        return A if connected and (degree > 0).all() else None

    def _genX(self, mol, max_length=None):
        max_length = max_length if max_length is not None else mol.GetNumAtoms()
        try:
            return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                        max_length - mol.GetNumAtoms()))
        except KeyError as e:
            print(f"Skipping molecule with unsupported atom: {e}")
            print(f"Skipped SMILES: {Chem.MolToSmiles(mol)}")
            return None

    def _genF(self, mol, max_length=None):

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
                              *[a.IsInRingSize(i) for i in range(2, 9)]] for a in mol.GetAtoms()], dtype=np.int32)

        return np.vstack((features, np.zeros((max_length - features.shape[0], features.shape[1]))))

    def decoder_load(self, dictionary_name, file):
        # Get the name of the other dataset if this is a joint encoder
        other_dataset = None
        if self.joint_raw_file:
            other_dataset = os.path.basename(self.joint_raw_file).split('.')[0]
        
        # Create a consistent joint suffix by sorting dataset names
        if other_dataset:
            datasets = sorted([self.dataset_name, other_dataset])
            joint_suffix = f"joint_{datasets[0]}_{datasets[1]}"
        else:
            joint_suffix = "joint"
        
        # Try loading joint encoders with specific naming first
        if os.path.exists(f"DrugGEN/data/decoders/{dictionary_name}_{joint_suffix}.pkl"):
            with open(f"DrugGEN/data/decoders/{dictionary_name}_{joint_suffix}.pkl", 'rb') as f:
                return pickle.load(f)    
        # Fall back to generic joint if not found
        elif os.path.exists(f"DrugGEN/data/decoders/{dictionary_name}_joint.pkl"):
            with open(f"DrugGEN/data/decoders/{dictionary_name}_joint.pkl", 'rb') as f:
                return pickle.load(f)    
        # Finally, try dataset-specific encoders
        else:
            with open(f"DrugGEN/data/decoders/{dictionary_name}_{file}.pkl", 'rb') as f:
                return pickle.load(f)    

    def matrices2mol(self, node_labels, edge_labels, strict=True, file_name=None):
        mol = Chem.RWMol()
        RDLogger.DisableLog('rdApp.*') 
        atom_decoders = self.decoder_load("atom", file_name)
        bond_decoders = self.decoder_load("bond", file_name)
        
        for node_label in node_labels:
            mol.AddAtom(Chem.Atom(atom_decoders[node_label]))

        for start, end in zip(*np.nonzero(edge_labels)):
            if start > end:
                mol.AddBond(int(start), int(end), bond_decoders[edge_labels[start, end]])
        #mol = self.correct_mol(mol)
        if strict:
            try:
                
                Chem.SanitizeMol(mol)
            except:
                mol = None

        return mol

    def check_valency(self,mol):
        """
        Checks that no atoms in the mol have exceeded their possible
        valency
        :return: True if no valency issues, False otherwise
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


    def correct_mol(self,x):
        xsm = Chem.MolToSmiles(x, isomericSmiles=True)
        mol = x
        while True:
            flag, atomid_valence = self.check_valency(mol)
            if flag:
                break
            else:
                assert len (atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                queue = []
                for b in mol.GetAtomWithIdx(idx).GetBonds():
                    queue.append(
                        (b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                    )
                queue.sort(key=lambda tup: tup[1], reverse=True)
                if len(queue) > 0:
                    start = queue[0][2]
                    end = queue[0][3]
                    t = queue[0][1] - 1
                    mol.RemoveBond(start, end)

                    #if t >= 1:
                        
                        #mol.AddBond(start, end, self.decoder_load('bond_decoders')[t])
                    # if '.' in Chem.MolToSmiles(mol, isomericSmiles=True):
                    #     mol.AddBond(start, end, self.decoder_load('bond_decoders')[t])
                    #     print(tt)
                    #     print(Chem.MolToSmiles(mol, isomericSmiles=True))
        return mol

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        out = torch.zeros(list(labels.size())+[dim])
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        
        return out.float() 

    def process(self, size=None):
        smiles_list = pd.read_csv(self.raw_files, header=None)[0].tolist()
        
        # Only create joint encoders if they don't exist yet
        create_joint = not os.path.exists("DrugGEN/data/encoders/atom_joint.pkl")
        max_length, smiles_list = self._generate_encoders_decoders(smiles_list, create_joint=create_joint)

        data_list = []
        self.m_dim = len(self.atom_decoder_m)
        for smiles in tqdm(smiles_list, desc='Processing chembl dataset', total=len(smiles_list)):
            mol = Chem.MolFromSmiles(smiles)
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                x_array = self._genX(mol, max_length=max_length)
                if x_array is None:  # Skip molecules with unsupported atoms
                    continue

                x = torch.from_numpy(x_array).to(torch.long).view(1, -1)
                x = self.label2onehot(x,self.m_dim).squeeze()
                if self.features: 
                    f = torch.from_numpy(self._genF(mol, max_length=max_length)).to(torch.long).view(x.shape[0], -1)
                    x = torch.concat((x,f), dim=-1)

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



   
if __name__ == '__main__':
    data = DruggenDataset("DrugGEN/data")
    
