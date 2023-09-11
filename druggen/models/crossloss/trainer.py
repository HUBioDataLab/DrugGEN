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

from .models import CrossLossDiscriminator, CrossLossGenerator
from .loss import discriminator_loss, generator_loss
from ...dataset import DruggenDataset
from ...training_data import generate_z_values, load_molecules
from ...utils import *

class CrossLossTrainerConfig:
    def __init__(
        self,
        trainer_folder: str =None,
        seed: int or None=None,
        batch_size=128,
        epoch=50,
        g_lr=0.00001,
        d_lr=0.00001,
        log_step=30,
        beta1=0.9,
        beta2=0.999,
        resume_from_checkpoint=False,
    ):
        self.seed = seed
        self.batch_size = batch_size
        self.epoch = epoch
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.trainer_folder = trainer_folder
        self.log_step = log_step
        self.beta1 = beta1
        self.beta2 = beta2
        self.resume_from_checkpoint = resume_from_checkpoint


class CrossLossTrainer:
    """Trainer for training and testing DrugGEN."""
    def __init__(self, model_config=None, trainer_config: CrossLossTrainerConfig=None):
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

        self.build_model()


             
    def build_model(self):
        self.generator = CrossLossGenerator(self.model_config)
        
        self.discriminator = CrossLossDiscriminator(self.model_config)
        
        self.g_optimizer = torch.optim.AdamW(self.generator1.parameters(), self.g_lr, [self.beta1, self.beta2])
        
        self.d_optimizer = torch.optim.AdamW(self.discriminator1.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        

    def restore_model(self, epoch, iteration, model_directory):
        
        """Restore the trained generator and discriminator."""
        
        print('Loading the trained models from epoch / iteration {}-{}...'.format(epoch, iteration))
        
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(epoch, iteration))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(epoch, iteration))
        
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
      
        
        G2_path = os.path.join(model_directory, '{}-{}-G2.ckpt'.format(epoch, iteration))
        D2_path = os.path.join(model_directory, '{}-{}-D2.ckpt'.format(epoch, iteration))
        
        self.G2.load_state_dict(torch.load(G2_path, map_location=lambda storage, loc: storage))
        self.D2.load_state_dict(torch.load(D2_path, map_location=lambda storage, loc: storage))

   
    def save_model(self, model_directory, idx,i):
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(idx+1,i+1))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(idx+1,i+1))
        torch.save(self.G.state_dict(), G_path)     
        torch.save(self.D.state_dict(), D_path) 
        
        if self.submodel != "NoTarget" and self.submodel != "CrossLoss":
            G2_path = os.path.join(model_directory, '{}-{}-G2.ckpt'.format(idx+1,i+1))
            D2_path = os.path.join(model_directory, '{}-{}-D2.ckpt'.format(idx+1,i+1))
    
            torch.save(self.G2.state_dict(), G2_path)         
            torch.save(self.D2.state_dict(), D2_path)  
        
    def reset_grad(self):
        """Reset the gradient buffers."""
        
        self.g_optimizer.zero_grad()
        self.g2_optimizer.zero_grad()            
        self.d_optimizer.zero_grad()
        self.d2_optimizer.zero_grad()

    def train(self):
        ''' Training Script starts from here'''
        

        # protein data
        full_smiles = [line for line in open("DrugGEN/data/chembl_train.smi", 'r').read().splitlines()]
        drug_smiles = [line for line in open("DrugGEN/data/akt_train.smi", 'r').read().splitlines()]
        
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_scaf = [MurckoScaffold.GetScaffoldForMol(x) for x in drug_mols]
        fps_r = [Chem.RDKFingerprint(x) for x in drug_scaf]

        akt1_human_adj = torch.load("DrugGEN/data/akt/AKT1_human_adj.pt").reshape(1,-1).to(self.device).float() 
        akt1_human_annot = torch.load("DrugGEN/data/akt/AKT1_human_annot.pt").reshape(1,-1).to(self.device).float() 

        if self.resume:
            self.restore_model(self.resume_epoch, self.resume_iter, self.resume_directory)

        # Start training.
        
        print('Start training...')
        self.start_time = time.time()
        for idx in range(self.epoch):
            dataloader_iterator = iter(self.drugs_loader)
            for i, data in enumerate(self.loader):   
                try:
                    drugs = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.drugs_loader)
                    drugs = next(dataloader_iterator)

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
                GAN1_disc_e = drugs_a_tensor
                GAN1_disc_x = drugs_x_tensor

                loss = {}
                self.reset_grad()
                
                # Compute discriminator loss.
     
                node, edge, d_loss = discriminator_loss(self.G, 
                                            self.D, 
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
                generator_output = generator_loss(self.G,
                                                    self.D,
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
                
                if (i+1) % self.log_step == 0:
                    logging(self.log_path, self.start_time, fake_mol, drug_smiles, i, idx, loss, 1, self.sample_directory)
                    mol_sample(self.sample_directory,"GAN1",fake_mol, g_edges_hat_sample.detach(), g_nodes_hat_sample.detach(), idx, i)
                                  

            if (idx+1) % 10 == 0:   
                self.save_model(self.model_directory,idx,i) 
                print("model saved at epoch {} and iteration {}".format(idx,i))       

