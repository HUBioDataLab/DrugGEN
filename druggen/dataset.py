import pickle
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import (Data, InMemoryDataset)
import os
from tqdm import tqdm
import re
from rdkit import RDLogger
import pandas as pd
import json

def save_bonds_encoder(bonds, path):
    # save bonds as json
    bonds = {k.name: v for k, v in bonds.items()}    
    with open(path, "w") as f:
        json.dump(bonds, f)

def read_bonds_encoder(path):
    with open(path) as f:
        data = json.load(f)

    return {getattr(Chem.rdchem.BondType, k): v for k, v in data.items()}

RDLogger.DisableLog('rdApp.*')
class DruggenDataset(InMemoryDataset):
    def __init__(self, root, max_atom=45, add_features=True, transform=None, pre_transform=None, pre_filter=None):
        # self.dataset_name = dataset_file.split(".")[0]
        # self.dataset_file = dataset_file
        # self.raw_files = raw_files
        self.max_atom = max_atom
        self.add_features = add_features
        super().__init__(root, transform, pre_transform, pre_filter)
        # path = osp.join(self.processed_dir, dataset_file)
        self.data, self.slices = torch.load(os.path.join(root, "preprocessed.pt"))

        self.bond_encoder_m = read_bonds_encoder(f"{root}/bond-encoder.json")
        self.bond_decoder_m = {k: v for v, k in self.bond_encoder_m.items()}

        with open(f"{root}/bond-encoder.json", "r") as file:
            self.atom_encoder_m = json.load(file)
        self.atom_decoder_m = {k: v for v, k in self.atom_encoder_m.items()}
        
    @property
    def processed_dir(self):
        return self.root
    
    @property
    def raw_file_names(self):
        return ["raw.smi"]

    @property
    def processed_file_names(self):
        return ["preprocessed.pt"]

    def _generate_encoders_decoders(self, data):
        self.data = data
        print('Creating atoms and bonds  encoder and decoder..')
        
        atom_labels = set()
        bond_labels = set()
        max_length = 0
        smiles_list = []
        for smiles in tqdm(data):
            mol = Chem.MolFromSmiles(smiles)
            molecule_size = mol.GetNumAtoms()
            if molecule_size > self.max_atom:
                continue
            smiles_list.append(smiles)
            atom_labels.update([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            max_length = max(max_length, molecule_size)
            bond_labels.update([bond.GetBondType() for bond in mol.GetBonds()])

        atom_labels.update([0]) # add PAD symbol (for unknown atoms)
        atom_labels = sorted(atom_labels) # turn set into list and sort it

        bond_labels = sorted(bond_labels)
        bond_labels = [Chem.rdchem.BondType.ZERO] + bond_labels

        # atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        # print('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
        #     self.atom_num_types - 1))

        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        
        self.bond_num_types = len(bond_labels)
        # print('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
        #     self.bond_num_types - 1))        
        #dataset_names = str(self.dataset_name)

        save_bonds_encoder(self.bond_encoder_m, f"{self.root}/bond-encoder.json")
        print(self.atom_encoder_m)
        with open(f"{self.root}/atom-encoder.json","w") as file:
            json.dump(self.atom_encoder_m,file)
                         
        return max_length, smiles_list # data is filtered now
        
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

        return np.array([self.atom_encoder_m[atom.GetAtomicNum()] for atom in mol.GetAtoms()] + [0] * (
                    max_length - mol.GetNumAtoms()))

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

    # def decoder_load(self, dictionary_name, file):
    #     with open(f"{self.root}/decoders/atom.pkl", 'rb') as f:
    #         return pickle.load(f)    
        
    # def drugs_decoder_load(self, dictionary_name):
    #     with open(f"{self.root}/decoders/atom.pkl", 'rb') as f:
    #         return pickle.load(f)      

        
    
    # def drug_decoder_load(self, dictionary_name, file):
        
    #     ''' Loading the atom and bond decoders '''
        
    #     with open(f"{self.root}/decoders/" + dictionary_name +"_" + file +'.pkl', 'rb') as f:
    #         return pickle.load(f) 

    # def matrices2mol_drugs(self, node_labels, edge_labels, strict=True, file_name=None):
        # mol = Chem.RWMol()
        # RDLogger.DisableLog('rdApp.*') 
        # atom_decoders = self.drug_decoder_load("atom", file_name)
        # bond_decoders = self.drug_decoder_load("bond", file_name)
        
        # for node_label in node_labels:
            
        #     mol.AddAtom(Chem.Atom(atom_decoders[node_label]))

        # for start, end in zip(*np.nonzero(edge_labels)):
        #     if start > end:
        #         mol.AddBond(int(start), int(end), bond_decoders[edge_labels[start, end]])
        # #mol = self.correct_mol(mol)
        # if strict:
        #     try:
        #         Chem.SanitizeMol(mol)
        #     except:
        #         mol = None

    #     return mol
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

        return mol
    
    
        
    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        
        out = torch.zeros(list(labels.size())+[dim])
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        
        return out.float() 
       
    def process(self, size= None):
        smiles_list = pd.read_csv(os.path.join(self.root, "raw.smi"), header=None)[0].tolist()
        max_length, smiles_list = self._generate_encoders_decoders(smiles_list)

        data_list = []
      
        self.m_dim = len(self.atom_decoder_m)
        for smiles in tqdm(smiles_list, desc='Processing chembl dataset', total=len(smiles_list)):
            mol = Chem.MolFromSmiles(smiles)
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                

                x = torch.from_numpy(self._genX(mol, max_length=max_length)).to(torch.long).view(1, -1)
          
                x = self.label2onehot(x,self.m_dim).squeeze()
                if self.add_features: 
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


        torch.save(self.collate(data_list), os.path.join(self.root, "preprocessed.pt"))

