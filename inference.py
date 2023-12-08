import os
import time
import pickle
import random
from tqdm import tqdm
import argparse

import torch
from torch_geometric.loader import DataLoader
import torch.utils.data
from rdkit import RDLogger
torch.set_num_threads(5)
RDLogger.DisableLog('rdApp.*')

from utils import *
from models import Generator
from new_dataloader import DruggenDataset
from loss import generator_loss
from training_data import load_molecules


class Inference(object):
    """Inference class for DrugGEN."""

    def __init__(self, config):
        if config.set_seed:
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(config.seed)

            print(f'Using seed {config.seed}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        # Initialize configurations
        self.targeted = config.targeted

        if targeted:
            self.submodel = "DrugGEN"
        else:
            self.submodel = "DrugGEN-NoTarget"
        
        self.inference_model = config.inference_model
        self.sample_num = config.sample_num

        # Data loader.
        self.inf_raw_file = config.inf_raw_file  # SMILES containing text file for first dataset. 
                                         # Write the full path to file.
        self.inf_dataset_file = config.inf_dataset_file    # Dataset file name for the first GAN. 
                                                   # Contains large number of molecules.
        self.inf_batch_size = config.inf_batch_size
        self.mol_data_dir = config.mol_data_dir  # Directory where the dataset files are stored.
        self.dataset_name = self.inf_dataset_file.split(".")[0]
        self.max_atom = config.max_atom  # Model is based on one-shot generation. 
                                         # Max atom number for molecules must be specified.
        self.features = config.features  # Small model uses atom types as node features. (Boolean, False uses atom types only.)
                                         # Additional node features can be added. Please check new_dataloarder.py Line 102.

        self.inf_dataset = DruggenDataset(self.mol_data_dir,
                                      self.inf_dataset_file,
                                      self.inf_raw_file,
                                      self.max_atom,
                                      self.features) # Dataset for the first GAN. Custom dataset class from PyG parent class.
                                                     # Can create any molecular graph dataset given smiles string.
                                                     # Nonisomeric SMILES are suggested but not necessary.
                                                     # Uses sparse matrix representation for graphs,
                                                     # For computational and speed efficiency.
        
        self.inf_loader = DataLoader(self.inf_dataset,
                                 shuffle=True,
                                 batch_size=self.inf_batch_size,
                                 drop_last=True)  # PyG dataloader for the first GAN.


        # Atom and bond type dimensions for the construction of the model.
        self.atom_decoders = self.decoder_load("atom")  # Atom type decoders for first GAN. 
                                                        # eg. 0:0, 1:6 (C), 2:7 (N), 3:8 (O), 4:9 (F)
        self.bond_decoders = self.decoder_load("bond")  # Bond type decoders for first GAN.
                                                        # eg. 0: (no-bond), 1: (single), 2: (double), 3: (triple), 4: (aromatic)
        self.m_dim = len(self.atom_decoders) if not self.features else int(self.inf_loader.dataset[0].x.shape[1]) # Atom type dimension.
        self.b_dim = len(self.bond_decoders) # Bond type dimension.
        self.vertexes = int(self.inf_loader.dataset[0].x.shape[0]) # Number of nodes in the graph.

        # Transformer and Convolution configurations.
        self.act = config.act
        self.dim = config.dim
        self.depth = config.depth
        self.heads = config.heads
        self.mlp_ratio = config.mlp_ratio
        self.dropout = config.dropout

        self.build_model()


    def build_model(self):
        """Create generators and discriminators."""
        self.G = Generator(self.act,
                           self.vertexes,
                           self.b_dim,
                           self.m_dim,
                           self.dropout,
                           dim=self.dim,
                           depth=self.depth,
                           heads=self.heads,
                           mlp_ratio=self.mlp_ratio,
                           submodel = self.submodel)

        self.print_network(self.G, 'G')

        self.G.to(self.device)


    def decoder_load(self, dictionary_name):
        ''' Loading the atom and bond decoders'''
        with open("DrugGEN/data/decoders/" + dictionary_name + "_" + self.dataset_name + '.pkl', 'rb') as f:
            return pickle.load(f)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel() 
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def restore_model(self, submodel, model_directory):
        """Restore the trained generator and discriminator."""
        print('Loading the model...')
        G_path = os.path.join(model_directory, '{}-G.ckpt'.format(submodel))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))


    def inference(self):
        # Load the trained generator.
        self.restore_model(self.submodel, self.inference_model)

        # smiles data for metrics calculation.
        chembl_smiles = [line for line in open("DrugGEN/data/chembl_train.smi", 'r').read().splitlines()]
        chembl_test = [line for line in open("DrugGEN/data/chembl_test.smi", 'r').read().splitlines()]
        drug_smiles = [line for line in open("DrugGEN/data/akt_inhibitors.smi", 'r').read().splitlines()]
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in drug_mols if x is not None]


        # Make directories if not exist.
        if not os.path.exists("DrugGEN/experiments/inference/{}".format(self.submodel)):
            os.makedirs("DrugGEN/experiments/inference/{}".format(self.submodel))


        self.G.eval()

        start_time = time.time()
        metric_calc_dr = []
        uniqueness_calc = []
        real_smiles_snn = []
        nodes_sample = torch.Tensor(size=[1,45,1]).to(self.device)

        val_counter = 0
        none_counter = 0
        # Inference mode
        with torch.inference_mode():
            pbar = tqdm(range(self.sample_num))
            pbar.set_description('Inference mode for {} model started'.format(self.submodel))
            for i, data in enumerate(self.inf_loader):

                val_counter += 1
                # Preprocess dataset 
                _, a_tensor, x_tensor = load_molecules(
                    data=data, 
                    batch_size=self.inf_batch_size,
                    device=self.device,
                    b_dim=self.b_dim,
                    m_dim=self.m_dim,
                )

                _, _, node_sample, edge_sample = self.G(a_tensor, x_tensor)

                g_edges_hat_sample = torch.max(edge_sample, -1)[1]
                g_nodes_hat_sample = torch.max(node_sample, -1)[1]

                fake_mol_g = [self.inf_dataset.matrices2mol_drugs(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=self.dataset_name) 
                        for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]

                a_tensor_sample = torch.max(a_tensor, -1)[1]
                x_tensor_sample = torch.max(x_tensor, -1)[1]
                real_mols = [self.inf_dataset.matrices2mol_drugs(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=self.dataset_name) 
                        for e_, n_ in zip(a_tensor_sample, x_tensor_sample)]

                inference_drugs = [None if line is None else Chem.MolToSmiles(line) for line in fake_mol_g]
                inference_drugs = [None if x is None else max(x.split('.'), key=len) for x in inference_drugs]

                for molecules in inference_drugs:
                            if molecules is None:
                                none_counter += 1

                with open("DrugGEN/experiments/inference/{}/inference_drugs.txt".format(self.submodel), "a") as f:
                    for molecules in inference_drugs:
                        if molecules is not None:
                            molecules = molecules.replace("*", "C") 
                            f.write(molecules)
                            f.write("\n")
                            uniqueness_calc.append(molecules)
                            nodes_sample = torch.cat((nodes_sample, g_nodes_hat_sample.view(1,45,1)), 0)
                            pbar.update(1)
                        metric_calc_dr.append(molecules)


                generation_number = len([x for x in metric_calc_dr if x is not None])
                if generation_number == self.sample_num or none_counter == self.sample_num:
                    break
                real_smiles_snn.append(real_mols[0])

        et = time.time() - start_time
        gen_vecs = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in uniqueness_calc if Chem.MolFromSmiles(x) is not None]
        real_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in real_smiles_snn if x is not None]
        print("Inference mode is lasted for {:.2f} seconds".format(et))

        print("Metrics calculation started using MOSES.")
        # post-process * to Carbon atom in valid molecules

        print("Validity: ", fraction_valid(metric_calc_dr), "\n")
        print("Uniqueness: ", fraction_unique(uniqueness_calc), "\n")
        print("Novelty: ", novelty(metric_calc_dr, chembl_smiles), "\n")
        print("Novelty_test: ", novelty(metric_calc_dr, chembl_test), "\n")
        print("AKT_novelty: ", novelty(metric_calc_dr, drug_smiles), "\n")
        print("max_len: ", Metrics.max_component(uniqueness_calc, self.vertexes), "\n")
        print("mean_atom_type: ", Metrics.mean_atom_type(nodes_sample), "\n")
        print("snn_chembl: ", average_agg_tanimoto(np.array(real_vecs), np.array(gen_vecs)), "\n")
        print("snn_akt: ", average_agg_tanimoto(np.array(drug_vecs), np.array(gen_vecs)), "\n")

        print("Metrics are calculated.")


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Inference configuration.
    parser.add_argument('--targeted', type=bool, default=True, help="Whether to use targeted model.")
    parser.add_argument('--inference_model', type=str, help="Path to the model for inference")
    parser.add_argument('--sample_num', type=int, default=10000, help='inference samples')

    # Data configuration.
    parser.add_argument('--inf_dataset_file', type=str, default='chembl45_test.pt')
    parser.add_argument('--inf_raw_file', type=str, default='DrugGEN/data/chembl_test.smi')
    parser.add_argument('--inf_batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--mol_data_dir', type=str, default='DrugGEN/data')
    parser.add_argument('--features', type=str2bool, default=False, help='features dimension for nodes')

    # Model configuration.
    parser.add_argument('--act', type=str, default="relu", help="Activation function for the model.", choices=['relu', 'tanh', 'leaky', 'sigmoid'])
    parser.add_argument('--max_atom', type=int, default=45, help='Max atom number for molecules must be specified.')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the Transformer Encoder model for GAN1.')
    parser.add_argument('--depth', type=int, default=1, help='Depth of the Transformer model from the first GAN.')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the first GAN.')
    parser.add_argument('--mlp_ratio', type=int, default=3, help='MLP ratio for the Transformers.')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')

    # Seed configuration.
    parser.add_argument('--set_seed', type=bool, default=False, help='set seed for reproducibility')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility')

    config = parser.parse_args()
    inference = Inference(config)
    inference.inference()
