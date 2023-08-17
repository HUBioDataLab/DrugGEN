import pickle
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import (Data, InMemoryDataset)
import os.path as osp
from tqdm import tqdm
import re 
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*') 
class DruggenDataset(InMemoryDataset):
    
    def __init__(self, root, dataset_file, raw_files, max_atom, features, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_file.split(".")[0]
        self.dataset_file = dataset_file
        self.raw_files = raw_files
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

    def _generate_encoders_decoders(self, data):
        
        self.data = data
        print('Creating atoms encoder and decoder..')
        
        atom_labels = set()
        # bond_labels = set()
        for smile in tqdm(data):
            mol = Chem.MolFromSmiles(smile)
            atom_labels.update([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            # bond_labels.update([bond.GetBondType() for bond in mol.GetBonds()])
        atom_labels.update([0]) # add PAD symbol (for unknown atoms)
        atom_labels = sorted(atom_labels) # turn set into list and sort it

        # atom_labels = sorted(set([atom.GetAtomicNum() for mol in self.data for atom in mol.GetAtoms()] + [0]))
        self.atom_encoder_m = {l: i for i, l in enumerate(atom_labels)}
        self.atom_decoder_m = {i: l for i, l in enumerate(atom_labels)}
        self.atom_num_types = len(atom_labels)
        print('Created atoms encoder and decoder with {} atom types and 1 PAD symbol!'.format(
            self.atom_num_types - 1))
        print("atom_labels", atom_labels)
        print('Creating bonds encoder and decoder..')
        # bond_labels = [Chem.rdchem.BondType.ZERO] + list(sorted(set(bond.GetBondType()
        #                                                             for mol in self.data
        #                                                             for bond in mol.GetBonds())))
        bond_labels = [
            Chem.rdchem.BondType.ZERO,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

        print("bond labels", bond_labels)
        self.bond_encoder_m = {l: i for i, l in enumerate(bond_labels)}
        self.bond_decoder_m = {i: l for i, l in enumerate(bond_labels)}
        self.bond_num_types = len(bond_labels)
        print('Created bonds encoder and decoder with {} bond types and 1 PAD symbol!'.format(
            self.bond_num_types - 1))        
        #dataset_names = str(self.dataset_name)
        with open("DrugGEN/data/encoders/" +"atom_" + self.dataset_name + ".pkl","wb") as atom_encoders:
            pickle.dump(self.atom_encoder_m,atom_encoders)
        
        
        with open("DrugGEN/data/decoders/" +"atom_" +  self.dataset_name + ".pkl","wb") as atom_decoders:
            pickle.dump(self.atom_decoder_m,atom_decoders)
        
               
        with open("DrugGEN/data/encoders/" +"bond_" +  self.dataset_name + ".pkl","wb") as bond_encoders:
            pickle.dump(self.bond_encoder_m,bond_encoders)
        
              
        with open("DrugGEN/data/decoders/" +"bond_" +  self.dataset_name + ".pkl","wb") as bond_decoders:
            pickle.dump(self.bond_decoder_m,bond_decoders)
         
        
        
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

    def decoder_load(self, dictionary_name, file):
        with open("DrugGEN/data/decoders/" + dictionary_name + "_" + file + '.pkl', 'rb') as f:
            return pickle.load(f)    
        
    def drugs_decoder_load(self, dictionary_name):
        with open("DrugGEN/data/decoders/" + dictionary_name +'.pkl', 'rb') as f:
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
    
    def drug_decoder_load(self, dictionary_name, file):
        
        ''' Loading the atom and bond decoders '''
        
        with open("DrugGEN/data/decoders/" + dictionary_name +"_" + file +'.pkl', 'rb') as f:
            
            return pickle.load(f) 
    def matrices2mol_drugs(self, node_labels, edge_labels, strict=True, file_name=None):
        mol = Chem.RWMol()
        RDLogger.DisableLog('rdApp.*') 
        atom_decoders = self.drug_decoder_load("atom", file_name)
        bond_decoders = self.drug_decoder_load("bond", file_name)
        
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
       
    def process(self, size= None):
        
        mols = [Chem.MolFromSmiles(line) for line in open(self.raw_files, 'r').readlines()]
     
        mols = list(filter(lambda x: x.GetNumAtoms() <= self.max_atom, mols))
        mols = mols[:size]
        indices = range(len(mols))
        
        self._generate_encoders_decoders(mols)
        
  
    
        pbar = tqdm(total=len(indices))
        pbar.set_description(f'Processing chembl dataset')
        max_length = max(mol.GetNumAtoms() for mol in mols)
        data_list = []
      
        self.m_dim = len(self.atom_decoder_m)
        for idx in indices:
            mol = mols[idx]
            A = self._genA(mol, connected=True, max_length=max_length)
            if A is not None:
                

                x = torch.from_numpy(self._genX(mol, max_length=max_length)).to(torch.long).view(1, -1)
          
                x = self.label2onehot(x,self.m_dim).squeeze()
                if self.features: 
                    f = torch.from_numpy(self._genF(mol, max_length=max_length)).to(torch.long).view(x.shape[0], -1)
                    x = torch.concat((x,f), dim=-1)
             
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

        torch.save(self.collate(data_list), osp.join(self.processed_dir, self.dataset_file))



   
if __name__ == '__main__':
    data = DruggenDataset("DrugGEN/data")
    
