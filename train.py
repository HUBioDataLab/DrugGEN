import os
import time
import random
import pickle
import argparse
import os.path as osp

import torch
import torch.utils.data
from torch import nn
from torch_geometric.loader import DataLoader

import wandb
from rdkit import RDLogger

torch.set_num_threads(5)
RDLogger.DisableLog('rdApp.*')

from src.util.utils import *
from src.model.models import Generator, Discriminator, simple_disc
from src.data.dataset import DruggenDataset
from src.data.utils import get_encoders_decoders, load_molecules
from src.model.loss import discriminator_loss, generator_loss

class Train(object):
    """Trainer for DrugGEN."""

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

        # Data loader.
        self.raw_file = config.raw_file  # SMILES containing text file for dataset. 
                                         # Write the full path to file.
        self.drug_raw_file = config.drug_raw_file  # SMILES containing text file for second dataset. 
                                                   # Write the full path to file.
        
        # Automatically infer dataset file names from raw file names
        raw_file_basename = osp.basename(self.raw_file)
        drug_raw_file_basename = osp.basename(self.drug_raw_file)
        
        # Get the base name without extension and add max_atom to it
        self.max_atom = config.max_atom  # Model is based on one-shot generation.
        raw_file_base = os.path.splitext(raw_file_basename)[0]
        drug_raw_file_base = os.path.splitext(drug_raw_file_basename)[0]

        # Change extension from .smi to .pt and add max_atom to the filename
        self.dataset_file = f"{raw_file_base}{self.max_atom}.pt"
        self.drugs_dataset_file = f"{drug_raw_file_base}{self.max_atom}.pt"

        self.mol_data_dir = config.mol_data_dir  # Directory where the dataset files are stored.
        self.drug_data_dir = config.drug_data_dir  # Directory where the drug dataset files are stored.
        self.dataset_name = self.dataset_file.split(".")[0]
        self.drugs_dataset_name = self.drugs_dataset_file.split(".")[0]
        self.features = config.features  # Small model uses atom types as node features. (Boolean, False uses atom types only.)
                                         # Additional node features can be added. Please check new_dataloarder.py Line 102.
        self.batch_size = config.batch_size  # Batch size for training.
        
        self.parallel = config.parallel

        # Get atom and bond encoders/decoders
        atom_encoder, atom_decoder, bond_encoder, bond_decoder = get_encoders_decoders(
            self.raw_file,
            self.drug_raw_file,
            self.max_atom
        )
        self.atom_encoder = atom_encoder
        self.atom_decoder = atom_decoder
        self.bond_encoder = bond_encoder
        self.bond_decoder = bond_decoder

        self.dataset = DruggenDataset(self.mol_data_dir,
                                     self.dataset_file,
                                     self.raw_file,
                                     self.max_atom,
                                     self.features,
                                     atom_encoder=atom_encoder,
                                     atom_decoder=atom_decoder,
                                     bond_encoder=bond_encoder,
                                     bond_decoder=bond_decoder)

        self.loader = DataLoader(self.dataset,
                                 shuffle=True,
                                 batch_size=self.batch_size,
                                 drop_last=True)  # PyG dataloader for the GAN.

        self.drugs = DruggenDataset(self.drug_data_dir, 
                                 self.drugs_dataset_file, 
                                 self.drug_raw_file, 
                                 self.max_atom, 
                                 self.features,
                                 atom_encoder=atom_encoder,
                                 atom_decoder=atom_decoder,
                                 bond_encoder=bond_encoder,
                                 bond_decoder=bond_decoder)

        self.drugs_loader = DataLoader(self.drugs, 
                                       shuffle=True,
                                       batch_size=self.batch_size, 
                                       drop_last=True)  # PyG dataloader for the second GAN.

        self.m_dim = len(self.atom_decoder) if not self.features else int(self.loader.dataset[0].x.shape[1]) # Atom type dimension.
        self.b_dim = len(self.bond_decoder) # Bond type dimension.
        self.vertexes = int(self.loader.dataset[0].x.shape[0]) # Number of nodes in the graph.

        # Model configurations.
        self.act = config.act
        self.lambda_gp = config.lambda_gp
        self.dim = config.dim
        self.depth = config.depth
        self.heads = config.heads
        self.mlp_ratio = config.mlp_ratio
        self.ddepth = config.ddepth
        self.ddropout = config.ddropout

        # Training configurations.
        self.epoch = config.epoch
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.dropout = config.dropout
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir

        # Step size.
        self.log_step = config.log_sample_step

        # resume training
        self.resume = config.resume
        self.resume_epoch = config.resume_epoch
        self.resume_iter = config.resume_iter
        self.resume_directory = config.resume_directory

        # wandb configuration
        self.use_wandb = config.use_wandb
        self.online = config.online
        self.exp_name = config.exp_name

        # Arguments for the model.
        self.arguments = "{}_{}_glr{}_dlr{}_dim{}_depth{}_heads{}_batch{}_epoch{}_dataset{}_dropout{}".format(self.exp_name, self.submodel, self.g_lr, self.d_lr, self.dim, self.depth, self.heads, self.batch_size, self.epoch, self.dataset_name, self.dropout)

        self.build_model(self.model_save_dir, self.arguments)


    def build_model(self, model_save_dir, arguments):
        """Create generators and discriminators."""
        
        ''' Generator is based on Transformer Encoder: 
            
            @ g_conv_dim: Dimensions for MLP layers before Transformer Encoder
            @ vertexes: maximum length of generated molecules (atom length)
            @ b_dim: number of bond types
            @ m_dim: number of atom types (or number of features used)
            @ dropout: dropout possibility
            @ dim: Hidden dimension of Transformer Encoder
            @ depth: Transformer layer number
            @ heads: Number of multihead-attention heads
            @ mlp_ratio: Read-out layer dimension of Transformer
            @ drop_rate: depricated  
            @ tra_conv: Whether module creates output for TransformerConv discriminator
            '''
        self.G = Generator(self.act,
                           self.vertexes,
                           self.b_dim,
                           self.m_dim,
                           self.dropout,
                           dim=self.dim,
                           depth=self.depth,
                           heads=self.heads,
                           mlp_ratio=self.mlp_ratio)

        ''' Discriminator implementation with Transformer Encoder:
            
            @ act: Activation function for MLP
            @ vertexes: maximum length of generated molecules (molecule length)
            @ b_dim: number of bond types
            @ m_dim: number of atom types (or number of features used)
            @ dropout: dropout possibility
            @ dim: Hidden dimension of Transformer Encoder
            @ depth: Transformer layer number
            @ heads: Number of multihead-attention heads
            @ mlp_ratio: Read-out layer dimension of Transformer'''

        self.D = Discriminator(self.act,
                                self.vertexes,
                                self.b_dim,
                                self.m_dim,
                                self.ddropout,
                                dim=self.dim,
                                depth=self.ddepth,
                                heads=self.heads,
                                mlp_ratio=self.mlp_ratio)

        self.g_optimizer = torch.optim.AdamW(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.AdamW(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        network_path = os.path.join(model_save_dir, arguments)
        self.print_network(self.G, 'G', network_path)
        self.print_network(self.D, 'D', network_path)

        if self.parallel and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name, save_dir):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        network_path = os.path.join(save_dir, "{}_modules.txt".format(name))
        with open(network_path, "w+") as file:
            for module in model.modules():
                file.write(f"{module.__class__.__name__}:\n")
                print(module.__class__.__name__)
                for n, param in module.named_parameters():
                    if param is not None:
                        file.write(f"  - {n}: {param.size()}\n")
                        print(f"  - {n}: {param.size()}")
                break
            file.write(f"Total number of parameters: {num_params}\n")
            print(f"Total number of parameters: {num_params}\n\n")

    def restore_model(self, epoch, iteration, model_directory):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from epoch / iteration {}-{}...'.format(epoch, iteration))

        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(epoch, iteration))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(epoch, iteration))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def save_model(self, model_directory, idx,i):
        G_path = os.path.join(model_directory, '{}-{}-G.ckpt'.format(idx+1,i+1))
        D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(idx+1,i+1))
        torch.save(self.G.state_dict(), G_path)
        torch.save(self.D.state_dict(), D_path)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self, config):
        ''' Training Script starts from here'''
        if self.use_wandb:
            mode = 'online' if self.online else 'offline'
        else:
            mode = 'disabled'
        kwargs = {'name': self.exp_name, 'project': 'druggen', 'config': config,
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
        wandb.init(**kwargs)

        wandb.save(os.path.join(self.model_save_dir, self.arguments, "G_modules.txt"))
        wandb.save(os.path.join(self.model_save_dir, self.arguments, "D_modules.txt"))

        self.model_directory = os.path.join(self.model_save_dir, self.arguments)
        self.sample_directory = os.path.join(self.sample_dir, self.arguments)
        self.log_path = os.path.join(self.log_dir, "{}.txt".format(self.arguments))
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)

        # smiles data for metrics calculation.
        drug_smiles = [line for line in open(self.drug_raw_file, 'r').read().splitlines()]
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in drug_mols if x is not None]

        if self.resume:
            self.restore_model(self.resume_epoch, self.resume_iter, self.resume_directory)

        # Start training.
        print('Start training...')
        self.start_time = time.time()
        for idx in range(self.epoch):
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
            # Load the data
            dataloader_iterator = iter(self.drugs_loader)

            wandb.log({"epoch": idx})

            for i, data in enumerate(self.loader):
                try:
                    drugs = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.drugs_loader)
                    drugs = next(dataloader_iterator)

                wandb.log({"iter": i})

                # Preprocess both dataset
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
                    b_dim=self.b_dim,
                    m_dim=self.m_dim,
                )

                # Training configuration.
                GEN_node = x_tensor             # Generator input node features (annotation matrix of real molecules)
                GEN_edge = a_tensor             # Generator input edge features (adjacency matrix of real molecules)
                if self.submodel == "DrugGEN":
                    DISC_node = drugs_x_tensor  # Discriminator input node features (annotation matrix of drug molecules)
                    DISC_edge = drugs_a_tensor  # Discriminator input edge features (adjacency matrix of drug molecules)
                elif self.submodel == "NoTarget":
                    DISC_node = x_tensor      # Discriminator input node features (annotation matrix of real molecules)
                    DISC_edge = a_tensor      # Discriminator input edge features (adjacency matrix of real molecules)

                # =================================================================================== #
                #                                     2. Train the GAN                                #
                # =================================================================================== #

                loss = {}
                self.reset_grad()
                # Compute discriminator loss.
                node, edge, d_loss = discriminator_loss(self.G,
                                            self.D,
                                            DISC_edge,
                                            DISC_node,
                                            GEN_edge,
                                            GEN_node,
                                            self.batch_size,
                                            self.device,
                                            self.lambda_gp)
                d_total = d_loss
                wandb.log({"d_loss": d_total.item()})

                loss["d_total"] = d_total.item()
                d_total.backward()
                self.d_optimizer.step()

                self.reset_grad()

                # Compute generator loss.
                generator_output = generator_loss(self.G,
                                                    self.D,
                                                    GEN_edge,
                                                    GEN_node,
                                                    self.batch_size)
                g_loss, node, edge, node_sample, edge_sample = generator_output
                g_total = g_loss
                wandb.log({"g_loss": g_total.item()})

                loss["g_total"] = g_total.item()
                g_total.backward()
                self.g_optimizer.step()

                # Logging.
                if (i+1) % self.log_step == 0:
                    logging(self.log_path, self.start_time, i, idx, loss, self.sample_directory,
                            drug_smiles,edge_sample, node_sample, self.dataset.matrices2mol,
                            self.dataset_name, a_tensor, x_tensor, drug_vecs)

                    mol_sample(self.sample_directory, edge_sample.detach(), node_sample.detach(),
                               idx, i, self.dataset.matrices2mol, self.dataset_name)
                    print("samples saved at epoch {} and iteration {}".format(idx,i))

                    self.save_model(self.model_directory, idx, i)
                    print("model saved at epoch {} and iteration {}".format(idx,i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data configuration.
    parser.add_argument('--raw_file', type=str, required=True)
    parser.add_argument('--drug_raw_file', type=str, required=False, help='Required for DrugGEN model, optional for NoTarget')
    parser.add_argument('--drug_data_dir', type=str, default='data')
    parser.add_argument('--mol_data_dir', type=str, default='data')
    parser.add_argument('--features', action='store_true', help='features dimension for nodes')

    # Model configuration.
    parser.add_argument('--submodel', type=str, default="DrugGEN", help="Chose model subtype: DrugGEN, NoTarget", choices=['DrugGEN', 'NoTarget'])
    parser.add_argument('--act', type=str, default="relu", help="Activation function for the model.", choices=['relu', 'tanh', 'leaky', 'sigmoid'])
    parser.add_argument('--max_atom', type=int, default=45, help='Max atom number for molecules must be specified.')
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the Transformer Encoder model for the GAN.')
    parser.add_argument('--depth', type=int, default=1, help='Depth of the Transformer model from the GAN.')
    parser.add_argument('--ddepth', type=int, default=1, help='Depth of the Transformer model from the discriminator.')
    parser.add_argument('--heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the GAN.')
    parser.add_argument('--mlp_ratio', type=int, default=3, help='MLP ratio for the Transformer.')
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    parser.add_argument('--ddropout', type=float, default=0., help='dropout rate for the discriminator')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Gradient penalty lambda multiplier for the GAN.')

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the training.')
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number for Training.')
    parser.add_argument('--g_lr', type=float, default=0.00001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.00001, help='learning rate for D')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--log_dir', type=str, default='experiments/logs')
    parser.add_argument('--sample_dir', type=str, default='experiments/samples')
    parser.add_argument('--model_save_dir', type=str, default='experiments/models')
    parser.add_argument('--log_sample_step', type=int, default=1000, help='step size for sampling during training')

    # Resume training.
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    parser.add_argument('--resume_epoch', type=int, default=None, help='resume training from this epoch')
    parser.add_argument('--resume_iter', type=int, default=None, help='resume training from this step')
    parser.add_argument('--resume_directory', type=str, default=None, help='load pretrained weights from this directory')

    # Seed configuration.
    parser.add_argument('--set_seed', action='store_true', help='set seed for reproducibility')
    parser.add_argument('--seed', type=int, default=1, help='seed for reproducibility')

    # wandb configuration.
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--online', action='store_true', help='use wandb online')
    parser.add_argument('--exp_name', type=str, default='druggen', help='experiment name')
    parser.add_argument('--parallel', action='store_true', help='Parallelize training')

    config = parser.parse_args()

    # Check if drug_raw_file is provided when using DrugGEN model
    if config.submodel == "DrugGEN" and not config.drug_raw_file:
        parser.error("--drug_raw_file is required when using DrugGEN model")

    # If using NoTarget model and drug_raw_file is not provided, use a dummy file
    if config.submodel == "NoTarget" and not config.drug_raw_file:
        config.drug_raw_file = "data/akt_train.smi"  # Use a reference file for NoTarget model (AKT) (not used for training for ease of use and encoder/decoder's)

    trainer = Train(config)
    trainer.train(config)
