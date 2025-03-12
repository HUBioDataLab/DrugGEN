import os
import sys
import time
import random
import pickle
import argparse
import os.path as osp

import torch
import torch.utils.data
from torch_geometric.loader import DataLoader

import pandas as pd
from tqdm import tqdm

from rdkit import RDLogger, Chem
from rdkit.Chem import QED, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from src.util.utils import *
from src.model.models import Generator
from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.loss import generator_loss
from src.util.smiles_cor import smi_correct


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
        self.submodel = config.submodel
        self.inference_model = config.inference_model
        self.sample_num = config.sample_num
        self.disable_correction = config.disable_correction

        # Data loader.
        self.inf_smiles = config.inf_smiles  # SMILES containing text file for first dataset. 
                                         # Write the full path to file.
        
        inf_smiles_basename = osp.basename(self.inf_smiles)
        
        # Get the base name without extension and add max_atom to it
        self.max_atom = config.max_atom  # Model is based on one-shot generation.
        inf_smiles_base = os.path.splitext(inf_smiles_basename)[0]
        
        # Change extension from .smi to .pt and add max_atom to the filename
        self.inf_dataset_file = f"{inf_smiles_base}{self.max_atom}.pt"

        self.inf_batch_size = config.inf_batch_size
        self.train_smiles = config.train_smiles
        self.train_drug_smiles = config.train_drug_smiles
        self.mol_data_dir = config.mol_data_dir  # Directory where the dataset files are stored.
        self.dataset_name = self.inf_dataset_file.split(".")[0]
        self.features = config.features  # Small model uses atom types as node features. (Boolean, False uses atom types only.)
                                         # Additional node features can be added. Please check new_dataloarder.py Line 102.

        # Get atom and bond encoders/decoders
        self.atom_encoder, self.atom_decoder, self.bond_encoder, self.bond_decoder = get_encoders_decoders(
            self.train_smiles,
            self.train_drug_smiles,
            self.max_atom
        )

        self.inf_dataset = DruggenDataset(self.mol_data_dir,
                                      self.inf_dataset_file,
                                      self.inf_smiles,
                                      self.max_atom,
                                      self.features,
                                      atom_encoder=self.atom_encoder,
                                      atom_decoder=self.atom_decoder,
                                      bond_encoder=self.bond_encoder,
                                      bond_decoder=self.bond_decoder)

        self.inf_loader = DataLoader(self.inf_dataset,
                                 shuffle=True,
                                 batch_size=self.inf_batch_size,
                                 drop_last=True)  # PyG dataloader for the first GAN.

        self.m_dim = len(self.atom_decoder) if not self.features else int(self.inf_loader.dataset[0].x.shape[1]) # Atom type dimension.
        self.b_dim = len(self.bond_decoder) # Bond type dimension.
        self.vertexes = int(self.inf_loader.dataset[0].x.shape[0]) # Number of nodes in the graph.

        # Model configurations.
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
                           mlp_ratio=self.mlp_ratio)
        self.G.to(self.device)
        self.print_network(self.G, 'G')

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
        chembl_smiles = [line for line in open(self.train_smiles, 'r').read().splitlines()]
        chembl_test = [line for line in open(self.inf_smiles, 'r').read().splitlines()]
        drug_smiles = [line for line in open(self.train_drug_smiles, 'r').read().splitlines()]
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in drug_mols if x is not None]


        # Make directories if not exist.
        if not os.path.exists("experiments/inference/{}".format(self.submodel)):
            os.makedirs("experiments/inference/{}".format(self.submodel))

        if not self.disable_correction:
            correct = smi_correct(self.submodel, "experiments/inference/{}".format(self.submodel))

        search_res = pd.DataFrame(columns=["submodel", "validity",
                                           "uniqueness", "novelty",
                                           "novelty_test", "drug_novelty",
                                           "max_len", "mean_atom_type",
                                           "snn_chembl", "snn_drug", "IntDiv", "qed", "sa"])

        self.G.eval()

        start_time = time.time()
        metric_calc_dr = []
        uniqueness_calc = []
        real_smiles_snn = []
        nodes_sample = torch.Tensor(size=[1, self.vertexes, 1]).to(self.device)
        f = open("experiments/inference/{}/inference_drugs.txt".format(self.submodel), "w")
        f.write("SMILES")
        f.write("\n")
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

                fake_mol_g = [self.inf_dataset.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=False, file_name=self.dataset_name) 
                        for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]

                a_tensor_sample = torch.max(a_tensor, -1)[1]
                x_tensor_sample = torch.max(x_tensor, -1)[1]
                real_mols = [self.inf_dataset.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=self.dataset_name) 
                        for e_, n_ in zip(a_tensor_sample, x_tensor_sample)]

                inference_drugs = [None if line is None else Chem.MolToSmiles(line) for line in fake_mol_g]
                inference_drugs = [None if x is None else max(x.split('.'), key=len) for x in inference_drugs]

                for molecules in inference_drugs:
                            if molecules is None:
                                none_counter += 1

                for molecules in inference_drugs:
                    if molecules is not None:
                        molecules = molecules.replace("*", "C") 
                        f.write(molecules)
                        f.write("\n")
                        uniqueness_calc.append(molecules)
                        nodes_sample = torch.cat((nodes_sample, g_nodes_hat_sample.view(1, self.vertexes, 1)), 0)
                        pbar.update(1)
                    metric_calc_dr.append(molecules)

                real_smiles_snn.append(real_mols[0])
                generation_number = len([x for x in metric_calc_dr if x is not None])
                if generation_number == self.sample_num or none_counter == self.sample_num:
                    break

        f.close()
        print("Inference completed, starting metrics calculation.")
        if not self.disable_correction:
            corrected = correct.correct("experiments/inference/{}/inference_drugs.txt".format(self.submodel))
            gen_smi = corrected["SMILES"].tolist()
            
        else:
            gen_smi = pd.read_csv("experiments/inference/{}/inference_drugs.txt".format(self.submodel))["SMILES"].tolist()
            
            
        et = time.time() - start_time
        
        gen_vecs = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=1024) for x in uniqueness_calc if Chem.MolFromSmiles(x) is not None]
        real_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in real_smiles_snn if x is not None]
        print("Inference mode is lasted for {:.2f} seconds".format(et))

        print("Metrics calculation started using MOSES.")
        
        if not self.disable_correction:
            val = round(len(gen_smi)/self.sample_num, 3)
            print("Validity: ", val, "\n")
        else: 
            val = round(fraction_valid(gen_smi), 3)
            print("Validity: ", val, "\n")

        uniq = round(fraction_unique(gen_smi), 3)
        nov = round(novelty(gen_smi, chembl_smiles), 3)
        nov_test = round(novelty(gen_smi, chembl_test), 3)
        drug_nov = round(novelty(gen_smi, drug_smiles), 3)
        max_len = round(Metrics.max_component(gen_smi, self.vertexes), 3)
        mean_atom = round(Metrics.mean_atom_type(nodes_sample), 3)
        snn_chembl = round(average_agg_tanimoto(np.array(real_vecs), np.array(gen_vecs)), 3)
        snn_drug = round(average_agg_tanimoto(np.array(drug_vecs), np.array(gen_vecs)), 3)
        int_div = round((internal_diversity(np.array(gen_vecs)))[0], 3)
        qed = round(np.mean([QED.qed(Chem.MolFromSmiles(x)) for x in gen_smi if Chem.MolFromSmiles(x) is not None]), 3)
        sa = round(np.mean([sascorer.calculateScore(Chem.MolFromSmiles(x)) for x in gen_smi if Chem.MolFromSmiles(x) is not None]), 3)

        print("Uniqueness: ", uniq, "\n")
        print("Novelty: ", nov, "\n")
        print("Novelty_test: ", nov_test, "\n")
        print("Drug_novelty: ", drug_nov, "\n")
        print("max_len: ", max_len, "\n")
        print("mean_atom_type: ", mean_atom, "\n")
        print("snn_chembl: ", snn_chembl, "\n")
        print("snn_drug: ", snn_drug, "\n")
        print("IntDiv: ", int_div, "\n")
        print("QED: ", qed, "\n")
        print("SA: ", sa, "\n")

        print("Metrics are calculated.")
        model_res = pd.DataFrame({"submodel": [self.submodel], "validity": [val],
                        "uniqueness": [uniq], "novelty": [nov],
                        "novelty_test": [nov_test], "drug_novelty": [drug_nov],
                        "max_len": [max_len], "mean_atom_type": [mean_atom],
                        "snn_chembl": [snn_chembl], "snn_drug": [snn_drug], 
                        "IntDiv": [int_div], "qed": [qed], "sa": [sa]})
        search_res = pd.concat([search_res, model_res], axis=0)
        os.remove("experiments/inference/{}/inference_drugs.txt".format(self.submodel))
        search_res.to_csv("experiments/inference/{}/inference_results.csv".format(self.submodel), index=False)
        generatedsmiles = pd.DataFrame({"SMILES": gen_smi})
        generatedsmiles.to_csv("experiments/inference/{}/inference_drugs.csv".format(self.submodel), index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # Inference configuration.
    parser.add_argument('--submodel', type=str, default="DrugGEN", help="Chose model subtype: DrugGEN, NoTarget", choices=['DrugGEN', 'NoTarget'])
    parser.add_argument('--inference_model', type=str, help="Path to the model for inference")
    parser.add_argument('--sample_num', type=int, default=100, help='inference samples')
    parser.add_argument('--disable_correction', action='store_true', help='Disable SMILES correction')
   
    # Data configuration.
    parser.add_argument('--inf_smiles', type=str, required=True)
    parser.add_argument('--train_smiles', type=str, required=True)
    parser.add_argument('--train_drug_smiles', type=str, required=True)
    parser.add_argument('--inf_batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--mol_data_dir', type=str, default='data')
    parser.add_argument('--features', action='store_true', help='features dimension for nodes')

    # Model configuration.
    parser.add_argument('--act', type=str, default="relu", help="Activation function for the model.", choices=['relu', 'tanh', 'leaky', 'sigmoid'])
    parser.add_argument('--max_atom', type=int, default=45, help='Max atom number for molecules must be specified.')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the Transformer Encoder model for the GAN.')
    parser.add_argument('--depth', type=int, default=1, help='Depth of the Transformer model from the GAN.')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the GAN.')
    parser.add_argument('--mlp_ratio', type=int, default=3, help='MLP ratio for the Transformer.')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')

    # Seed configuration.
    parser.add_argument('--set_seed', action='store_true', help='set seed for reproducibility')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility')

    config = parser.parse_args()
    inference = Inference(config)
    inference.inference()
