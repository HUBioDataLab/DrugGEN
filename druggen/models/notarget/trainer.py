import os
import time
import torch.nn
import torch

# from models import Generator, Generator2, simple_disc
import torch_geometric.utils as geoutils
#import wandb
# import re
from torch_geometric.loader import DataLoader

from rdkit import RDLogger  
import pickle
from rdkit.Chem.Scaffolds import MurckoScaffold
torch.set_num_threads(5)
RDLogger.DisableLog('rdApp.*') 
import random
from tqdm import tqdm

from .model import NoTargetDiscriminator, NoTargetGenerator
from loss import discriminator_loss, generator_loss, discriminator2_loss, generator2_loss
from ...dataset import DruggenDataset
# from ...training_data import load_data
from ...training_data import generate_z_values, load_molecules
from ...utils import *

class NoTargetTrainerConfig:
    def __init__(
        self,
        trainer_folder: str =None,
        seed: int or None=None,
        batch_size=128,
        epoch=50,
        g_lr=0.00001,
        d_lr=0.00001,
        log_step=30,

    ):
        self.seed = seed
        self.batch_size = batch_size
        self.epoch = epoch
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.trainer_folder = trainer_folder
        self.log_step = log_step


class NoTargetTrainer:
    """Trainer for training and testing DrugGEN No Target Model."""
    def __init__(
        self,
        model_config=None,
        trainer_config: NoTargetTrainerConfig=None,
    ):
        self.model_config = model_config
        self.trainer_config = trainer_config

        if trainer_config.seed is not None:
            np.random.seed(trainer_config.seed)
            random.seed(trainer_config.seed)
            torch.manual_seed(trainer_config.seed)
            torch.cuda.manual_seed(trainer_config.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(trainer_config.seed)
            print(f'Using seed {trainer_config.seed}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        # self.raw_file = config.raw_file  # SMILES containing text file for first dataset. 
        #                                  # Write the full path to file.
        
                                                    
        # self.dataset_file = config.dataset_file    # Dataset file name for the first GAN. 
        #                                            # Contains large number of molecules.
        
        # self.drugs_dataset_file = config.drug_dataset_file  # Drug dataset file name for the second GAN. 
        #                                                     # Contains drug molecules only. (In this case AKT1 inhibitors.)
        
        # self.inf_raw_file = config.inf_raw_file  # SMILES containing text file for first dataset. 
        #                                  # Write the full path to file.
        

        # self.inf_dataset_file = config.inf_dataset_file    # Dataset file name for the first GAN. 
        #                                            # Contains large number of molecules.
        
        # self.inf_drugs_dataset_file = config.inf_drug_dataset_file  # Drug dataset file name for the second GAN. 
        #                                                     # Contains drug molecules only. (In this case AKT1 inhibitors.)
        # self.inference_iterations = config.inference_iterations
        
        # self.inf_batch_size = config.inf_batch_size
        
        # self.mol_data_dir = config.mol_data_dir  # Directory where the dataset files are stored.
        
        # self.drug_data_dir = config.drug_data_dir  # Directory where the drug dataset files are stored.
                                                         
        
        # self.max_atom = config.max_atom  # Model is based on one-shot generation. 
        #                                  # Max atom number for molecules must be specified.
                                         
        # self.features = config.features  # Small model uses atom types as node features. (Boolean, False uses atom types only.)
        #                                  # Additional node features can be added. Please check new_dataloarder.py Line 102.
        
        
        # self.batch_size = config.batch_size  # Batch size for training.
        
        # self.dataset = DruggenDataset(self.mol_data_dir,
        #                               self.dataset_file, 
        #                               self.raw_file, 
        #                               self.max_atom, 
        #                               self.features) # Dataset for the first GAN. Custom dataset class from PyG parent class. 
        #                                              # Can create any molecular graph dataset given smiles string. 
        #                                              # Nonisomeric SMILES are suggested but not necessary.
        #                                              # Uses sparse matrix representation for graphs, 
        #                                              # For computational and speed efficiency.
        
        self.loader = DataLoader(self.dataset, 
                                 shuffle=True,
                                 batch_size=self.batch_size, 
                                 drop_last=True)  # PyG dataloader for the first GAN.
        
        # self.atom_decoders = self.decoder_load("atom")  # Atom type decoders for first GAN. 
        #                                                 # eg. 0:0, 1:6 (C), 2:7 (N), 3:8 (O), 4:9 (F)
                                                        
        # self.bond_decoders = self.decoder_load("bond")  # Bond type decoders for first GAN.
        #                                                 # eg. 0: (no-bond), 1: (single), 2: (double), 3: (triple), 4: (aromatic)
     
        self.m_dim = len(self.atom_decoders) if not self.features else int(self.loader.dataset[0].x.shape[1]) # Atom type dimension.
        
        self.b_dim = len(self.bond_decoders) # Bond type dimension.
        
        self.vertexes = int(self.loader.dataset[0].x.shape[0]) # Number of nodes in the graph.
               
        # self.drugs_atom_decoders = self.drug_decoder_load("atom") # Atom type decoders for second GAN.
        #                                                           # eg. 0:0, 1:6 (C), 2:7 (N), 3:8 (O), 4:9 (F)
        
        # self.drugs_bond_decoders = self.drug_decoder_load("bond") # Bond type decoders for second GAN.
        #                                                           # eg. 0: (no-bond), 1: (single), 2: (double), 3: (triple), 4: (aromatic)
        
        
        # self.drugs_m_dim = len(self.drugs_atom_decoders) if not self.features else int(self.drugs_loader.dataset[0].x.shape[1]) # Atom type dimension.
        
        # self.drugs_b_dim = len(self.drugs_bond_decoders)    # Bond type dimension.
        
        # self.drug_vertexes = int(self.drugs_loader.dataset[0].x.shape[0])  # Number of nodes in the graph.
        
        # self.act = config.act         
        # self.z_dim = config.z_dim 
        # self.lambda_gp = config.lambda_gp  
        # self.dim = config.dim 
        # self.depth = config.depth 
        # self.heads = config.heads 
        # self.mlp_ratio = config.mlp_ratio 
        # self.dec_depth = config.dec_depth 
        # self.dec_heads = config.dec_heads 
        # self.dec_dim = config.dec_dim
        # self.epoch = config.epoch 
        # self.g_lr = config.g_lr  
        # self.d_lr = config.d_lr  
        # self.g2_lr = config.g2_lr 
        # self.d2_lr = config.d2_lr
        # self.dropout = config.dropout
        # self.dec_dropout = config.dec_dropout
        # self.n_critic = config.n_critic 
        # self.resume_iters = config.resume_iters 
        # self.warm_up_steps = config.warm_up_steps
        # self.test_iters = config.test_iters
        # self.inference_sample_num = config.inference_sample_num     
        # self.log_dir = config.log_dir
        # self.sample_dir = config.sample_dir
        # self.model_save_dir = config.model_save_dir
        # self.result_dir = config.result_dir        
        # self.log_step = config.log_sample_step
        # self.clipping_value = config.clipping_value
        # self.resume = config.resume
        # self.resume_epoch = config.resume_epoch
        # self.resume_iter = config.resume_iter
        # self.resume_directory = config.resume_directory
        # self.init_type = config.init_type
        self.build_model()


    def build_model(self):
        """Create generators and discriminators."""
        self.generator = NoTargetGenerator(self.model_config)
        self.discriminator = NoTargetDiscriminator(self.model_config)

        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            self.trainer_config.g_lr,
            [self.trainer_config.beta1, self.trainer_config.beta2]
        )
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            self.trainer_config.d_lr,
            [self.trainer_config.beta1, self.trainer_config.beta2]
        )

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
    # def decoder_load(self, dictionary_name):
    #     ''' Loading the atom and bond decoders'''
    #     with open("DrugGEN/data/decoders/" + dictionary_name + "_" + self.dataset_name + '.pkl', 'rb') as f:    
    #         return pickle.load(f)    

    # def drug_decoder_load(self, dictionary_name):
    #     ''' Loading the atom and bond decoders'''
    #     with open("DrugGEN/data/decoders/" + dictionary_name +"_" + self.drugs_name +'.pkl', 'rb') as f:
    #         return pickle.load(f)    
 
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel() 
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def restore_model(self, epoch, iteration, model_directory):
        
        """Restore the trained generator and discriminator."""
        
        print('Loading the trained models from epoch / iteration {}-{}...'.format(epoch, iteration))
        
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(epoch, iteration))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(epoch, iteration))
        
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
         
    def save_model(self, model_directory, idx,i):
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(idx+1,i+1))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(idx+1,i+1))
        torch.save(self.generator.state_dict(), G_path)     
        torch.save(self.discriminator.state_dict(), D_path) 
   
        
    def reset_grad(self):        
        """Reset the gradient buffers."""

        self.g_optimizer.zero_grad()            
        self.d_optimizer.zero_grad()

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        
        weight = torch.ones(y.size(),requires_grad=False).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        gradient_penalty = ((dydx.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def train(self):
        
        ''' Training Script starts from here'''
        
       
        self.arguments = "{}_glr{}_dlr{}_g2lr{}_d2lr{}_dim{}_depth{}_heads{}_decdepth{}_decheads{}_ncritic{}_batch{}_epoch{}_warmup{}_dataset{}_dropout{}".format(self.submodel,self.g_lr,self.d_lr,self.g2_lr,self.d2_lr,self.dim,self.depth,self.heads,self.dec_depth,self.dec_heads,self.n_critic,self.batch_size,self.epoch,self.warm_up_steps,self.dataset_name,self.dropout)
       
        self.model_directory= os.path.join(self.model_save_dir,self.arguments)
        self.sample_directory=os.path.join(self.sample_dir,self.arguments)
        self.log_path = os.path.join(self.log_dir, "{}.txt".format(self.arguments))
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)         

        if self.resume:
            self.restore_model(self.resume_epoch, self.resume_iter, self.resume_directory)

        # Start training.
        
        print('Start training...')
        self.start_time = time.time()
        for idx in range(self.epoch):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            
            for i, data in enumerate(self.loader):   
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

                GAN1_input_e = a_tensor 
                GAN1_input_x = x_tensor 
                GAN1_disc_e = a_tensor
                GAN1_disc_x = x_tensor                                                  
                
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                loss = {}
                self.reset_grad()
                
                # Compute discriminator loss.
     
                node, edge, d_loss = discriminator_loss(self.generator, 
                                            self.discriminator, 
                                            real_graphs, 
                                            GAN1_disc_e, 
                                            GAN1_disc_x, 
                                            self.batch_size, 
                                            self.device, 
                                            self.gradient_penalty, 
                                            self.lambda_gp,
                                            GAN1_input_e,
                                            GAN1_input_x)
        
                d_total = d_loss
                
                loss["d_total"] = d_total.item()
                d_total.backward()
                self.d_optimizer.step()
                self.reset_grad()
                generator_output = generator_loss(self.generator,
                                                    self.discriminator,
                                                    self.V,
                                                    GAN1_input_e,
                                                    GAN1_input_x,
                                                    self.batch_size,
                                                    sim_reward,
                                                    self.dataset.matrices2mol,
                                                    fps_r,
                                                    self.submodel,
                                                    self.dataset_name)        
                
                g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge = generator_output    
            
                self.reset_grad()
                g_total = g_loss
              
                loss["g_total"] = g_total.item()
                g_total.backward()
                self.g_optimizer.step()
  
    def inference(self):
        
        # Load the trained generator.
        self.generator.to(self.device)
        
        
        G_path = os.path.join(self.inference_model, '{}-G.ckpt'.format(self.submodel))
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        
        
        
        drug_smiles = [line for line in open("DrugGEN/data/chembl_train.smi", 'r').read().splitlines()]
        
        self.generator.eval()
        
        step = self.inference_iterations
        
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

        start_time = time.time()
        #metric_calc_mol = []
        metric_calc_dr = []
        date = time.time()
        if not os.path.exists("DrugGEN/experiments/inference/{}".format(self.submodel)):
            os.makedirs("DrugGEN/experiments/inference/{}".format(self.submodel))
        with torch.inference_mode():
            
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

                GAN1_input_e = a_tensor
                GAN1_input_x = x_tensor     
                # =================================================================================== #
                #                             2. GAN1 Inference                                       #
                # =================================================================================== #            
                generator_output = generator_loss(self.generator,
                                                    self.discriminator,
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

                # =================================================================================== #
                #                             3. GAN2 Inference                                       #
                # =================================================================================== #   

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
