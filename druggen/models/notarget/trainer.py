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

    def restore_model(self, folder=None):
        """Restore the trained generator and discriminator."""
        
        if folder is None:
            generator_path = os.path.join(self.trainer_config.trainer_folder, "generator.pt")
            discriminator_path = os.path.join(self.trainer_config.trainer_folder, "discriminator.pt")
        else:
            generator_path = os.path.join(folder, "generator.pt")
            discriminator_path = os.path.join(folder, "discriminator.pt")
        
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(epoch, iteration))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(epoch, iteration))
        
        self.generator.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.discriminator.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def save_model(self, folder=None):
        if folder is None:
            generator_path = os.path.join(self.trainer_config.trainer_folder, "generator.pt")
            discriminator_path = os.path.join(self.trainer_config.trainer_folder, "discriminator.pt")

        else:
            generator_path = os.path.join(folder, "generator.pt")
            discriminator_path = os.path.join(folder, "discriminator.pt")

        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
   
        
    def reset_grad(self):        
        """Reset the gradient buffers."""

        self.g_optimizer.zero_grad()            
        self.d_optimizer.zero_grad()



    def train(self, dataloader):
        ''' Training Script starts from here'''

        if self.trainer_config.resume_from_checkpoint:
            raise NotImplementedError
            self.restore_model(self.resume_epoch, self.resume_iter, self.resume_directory)
        
        print('Start training...')
        self.start_time = time.time()
        for epoch_idx in range(self.trainer_config.epoch):
            for batch_idx, data in enumerate(dataloader):
                z, z_edge, z_node = generate_z_values(
                    batch_size=self.trainer_config.batch_size,
                    z_dim=self.model_config.z_dim,
                    vertexes=self.model_config.vertexes,
                    device=self.device,
                )

                real_graphs, a_tensor, x_tensor = load_molecules(
                    data=data, 
                    batch_size=self.trainer_config.batch_size,
                    device=self.device,
                    b_dim=self.model_config.edges,
                    m_dim=self.model_config.nodes,
                )

                self.reset_grad()

                # Compute discriminator loss.
                node, edge, d_loss = discriminator_loss(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    mol_graph=real_graphs,
                    batch_size=self.trainer_config.batch_size,
                    device=self.device,
                    lambda_gp=self.trainer_config.lambda_gp,
                    z_edge=z_edge,
                    z_node=z_node
                )


                d_loss.backward()
                self.d_optimizer.step()
                self.reset_grad()
                g_loss = generator_loss(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    adj=a_tensor,
                    annot=x_tensor,
                    batch_size=self.batch_size,
                )
                # g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge = generator_output
                # none of these outputs are being used, so we can just ignore them

                self.reset_grad()

                g_loss.backward()
                self.g_optimizer.step()
