import os
import time
import torch.nn
import torch

import torch_geometric.utils as geoutils

from rdkit import RDLogger  
import pickle

torch.set_num_threads(5)
RDLogger.DisableLog('rdApp.*') 
import random
from tqdm import tqdm

from .models import NoTargetDiscriminator, NoTargetGenerator
from .loss import discriminator_loss, generator_loss
from ...dataset import DruggenDataset
from ...training_data import generate_z_values, load_molecules
from ...utils import *

class NoTargetInferencerConfig:
    def __init__(
        self,
        folder: str =None,
        seed: int or None=None,
        batch_size=128,
        epoch=50,
        num_molecules=1000,
    ):
        self.seed = seed
        self.batch_size = batch_size
        self.folder = folder
        self.epoch = epoch



class NoTargetInferencer:
    """inferencer for training and testing DrugGEN No Target Model."""
    def __init__(
        self,
        model_config=None,
        inferencer_config: NoTargetInferencerConfig=None,
    ):
        self.model_config = model_config
        self.inferencer_config = inferencer_config

        if inferencer_config.seed is not None:
            np.random.seed(inferencer_config.seed)
            random.seed(inferencer_config.seed)
            torch.manual_seed(inferencer_config.seed)
            torch.cuda.manual_seed(inferencer_config.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(inferencer_config.seed)
            print(f'Using seed {inferencer_config.seed}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.build_model()


    def build_model(self):
        """Create generators and discriminators."""
        self.generator = NoTargetGenerator(self.model_config)
        self.generator.to(self.device)


    def restore_model(self, epoch, iteration, model_directory ):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from epoch / iteration {}-{}...'.format(epoch, iteration))
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(epoch, iteration))
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def save_model(self, folder=None):
        if folder is None:
            generator_path = os.path.join(self.inferencer_config.inferencer_folder, "generator.pt")
            
        else:
            generator_path = os.path.join(folder, "generator.pt")

        torch.save(self.generator.state_dict(), generator_path)

  
    def inference(self, dataloader):
        
        # Load the trained generator.
        self.generator.to(self.device)
        
        G_path = os.path.join(self.inference_model, '{}-G.ckpt'.format(self.submodel))
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        
        
        drug_smiles = [line for line in open("DrugGEN/data/chembl_train.smi", 'r').read().splitlines()]
            
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_scaf = [MurckoScaffold.GetScaffoldForMol(x) for x in drug_mols]
        fps_r = [Chem.RDKFingerprint(x) for x in drug_scaf]

        akt1_human_adj = torch.load("DrugGEN/data/akt/AKT1_human_adj.pt").reshape(1,-1).to(self.device).float() 
        akt1_human_annot = torch.load("DrugGEN/data/akt/AKT1_human_annot.pt").reshape(1,-1).to(self.device).float() 
        
        self.generator.eval()
        
        step = self.inference_iterations
        
        
        start_time = time.time()
        #metric_calc_mol = []
        metric_calc_dr = []
        date = time.time()
        if not os.path.exists("DrugGEN/experiments/inference/{}".format(self.submodel)):
            os.makedirs("DrugGEN/experiments/inference/{}".format(self.submodel))
        with torch.inference_mode():
            
            dataloader_iterator = iter(self.inf_drugs_loader)
            pbar = tqdm(range(self.inference_sample_num))
            pbar.set_description('Inference mode for {} model started'.format(self.submodel))
            for i, data in enumerate(self.inf_loader):   
                
                z, z_edge, z_node = generate_z_values(
                    batch_size=self.batch_size,
                    z_dim=self.z_dim,
                    vertexes=self.vertexes,
                    device=self.device,
                )

                real_graphs, a_tensor, x_tensor = load_molecules(
                    data=data, 
                    batch_size=self.batch_size,
                    device=self.device,
                    b_dim=self.b_dim,
                    m_dim=self.m_dim,
                )

                drug_graphs, drugs_a_tensor, drugs_x_tensor = load_molecules(
                    data=drugs, 
                    batch_size=self.batch_size,
                    device=self.device,
                    b_dim=self.drugs_b_dim,
                    m_dim=self.drugs_m_dim,
                )

                
                GAN1_input_e = a_tensor
                GAN1_input_x = x_tensor
                # =================================================================================== #
                #                             2. GAN1 Inference                                       #
                # =================================================================================== #            
                generator_output = generator_loss(self.G,
                                                    self.D,
                                                    self.V,
                                                    GAN1_input_e,
                                                    GAN1_input_x,
                                                    self.inf_batch_size,
                                                    sim_reward,
                                                    self.dataset.matrices2mol,
                                                    fps_r,
                                                    self.submodel,
                                                    self.dataset_name)   
                
                _, fake_mol_g, _, _, node, edge = generator_output  

                
                inference_drugs = [Chem.MolToSmiles(line) for line in fake_mol_g if line is not None]   
                inference_drugs = [None if x is None else max(x.split('.'), key=len) for x in inference_drugs]

                with open("DrugGEN/experiments/inference/{}/inference_drugs.txt".format(self.submodel), "a") as f:
                    for molecules in inference_drugs:
                        
                        f.write(molecules)
                        f.write("\n")
                        metric_calc_dr.append(molecules)
            
                if len(inference_drugs) > 0:
                    pbar.update(1)
                              
                if len(metric_calc_dr) == self.inference_sample_num:
                    break
        
        et = time.time() - start_time
        
        print("Inference mode is lasted for {:.2f} seconds".format(et))
        
        print("Metrics calculation started using MOSES.")
                   
        print("Validity: ", fraction_valid(metric_calc_dr), "\n")
        print("Uniqueness: ", fraction_unique(metric_calc_dr), "\n")
        print("Novelty: ", novelty(metric_calc_dr, drug_smiles), "\n")

        print("Metrics are calculated.")
