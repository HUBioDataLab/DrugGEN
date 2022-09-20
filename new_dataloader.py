import pickle
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import (Data, InMemoryDataset)
import os.path as osp
import pickle
import torch
from tqdm import tqdm
import re 

class DruggenDataset(InMemoryDataset):
    
    def __init__(self, root, dataset_file, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.dataset_file = dataset_file
        self.dataset_name = self.dataset_file.split(".")[0]
        path = osp.join(self.processed_dir, dataset_file)
        self.data, self.slices = torch.load(path)

    @property
    def processed_dir(self):
        
        return self.root
    
    @property
    def raw_file_names(self):
        return ["chembl25.pt","chembl45.pt","qm9.pt","zinc_250k.pt", "chembl35.pt"]

    @property
    def processed_file_names(self):
        return ['chembl25.pt','chembl45.pt','qm9.pt',"zinc_250k.pt", "chembl35.pt"]

    def _generate_encoders_decoders(self, data):
        self.data = data
        print('Creating atoms encoder and decoder..')
        atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        print('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))
        print("atom_labels", atom_labels)
        print('Creating bonds encoder and decoder..')
        bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
                                                                    for mol in self.data
                                                                    for bond in mol.GetBonds())))
        print("bond labels", bond_labels)
        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        print('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))        
        #dataset_names = str(self.dataset_name)
        atom_encoders = open("DrugGEN/data/atom_encoders" + "chembl35" + ".pkl","wb")
        pickle.dump(self.atom_encoder_m,atom_encoders)
        atom_encoders.close()
        
        atom_decoders = open("DrugGEN/data/atom_decoders" + "chembl35" + ".pkl","wb")
        pickle.dump(self.atom_decoder_m,atom_decoders)
        atom_decoders.close() 
               
        bond_encoders = open("DrugGEN/data/bond_encoders" + "chembl35" + ".pkl","wb")
        pickle.dump(self.bond_encoder_m,bond_encoders)
        bond_encoders.close()  
              
        bond_decoders = open("DrugGEN/data/bond_decoders" + "chembl35" + ".pkl","wb")
        pickle.dump(self.bond_decoder_m,bond_decoders)
        bond_decoders.close()        
        
        
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

    def decoder_load(self, dictionary_name):
        with open("DrugGEN/data/" + dictionary_name + self.dataset_name + '.pkl', 'rb') as f:
            return pickle.load(f)    
    def drugs_decoder_load(self, dictionary_name):
        with open("DrugGEN/data/" + "drugs_" + dictionary_name + '.pkl', 'rb') as f:
            return pickle.load(f)      
    def matrices2mol(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()
        atom_decoders = self.decoder_load("atom_decoders")
        bond_decoders = self.decoder_load("bond_decoders")
        
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
    def matrices2mol_drugs(self, node_labels, edge_labels, strict=False):
        mol = Chem.RWMol()
        atom_decoders = self.drugs_decoder_load("atom_decoders")
        bond_decoders = self.drugs_decoder_load("bond_decoders")
        
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
                    #     print(tt)
                    #     print(Chem.MolToSmiles(mol, isomericSmiles=True))

        return mol
    
    def process(self, size= None):
        
        mols = [Chem.MolFromSmiles(line) for line in open("DrugGEN/data/chembl_smiles.smi", 'r').readlines()]
        mols = list(filter(lambda x: x.GetNumAtoms() <= 35, mols))
        mols = mols[:size]
        indices = range(len(mols))
        
        self._generate_encoders_decoders(mols)
        
  
    
        pbar = tqdm(total=len(indices))
        pbar.set_description(f'Processing chembl dataset')
        max_length = max(mol.GetNumAtoms() for mol in mols)
        data_list = []
        for idx in indices:
            mol = mols[idx]
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                

                x = torch.from_numpy(self._genX(mol, max_length=max_length)).to(torch.long).view(-1, 1)
                
                adjacency = torch.from_numpy(A)
                
                edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adjacency[edge_index[0], edge_index[1]].to(torch.long)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)
                pbar.update(1)

        pbar.close()

        torch.save(self.collate(data_list), osp.join(self.processed_dir, "chembl35.pt"))



   
if __name__ == '__main__':
    data = DruggenDataset("DrugGEN/data")
    
