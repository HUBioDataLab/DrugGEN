import os
import time
import torch.nn
import torch

from utils import *
from models import Generator, Generator2, simple_disc
import torch_geometric.utils as geoutils
#import #wandb
import re
from torch_geometric.loader import DataLoader
from new_dataloader import DruggenDataset
import torch.utils.data
from rdkit import RDLogger  
import pickle
from rdkit.Chem.Scaffolds import MurckoScaffold
torch.set_num_threads(5)
RDLogger.DisableLog('rdApp.*') 
from loss import discriminator_loss, generator_loss, discriminator2_loss, generator2_loss
from training_data import load_data
import random
torch.manual_seed(1337)
random.seed(1337)

class Trainer(object):
    
    """Trainer for training and testing DrugGEN."""

    def __init__(self, config):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        """Initialize configurations."""
        self.submodel = config.submodel
        self.inference_model = config.inference_model
        # Data loader.
        self.raw_file = config.raw_file  # SMILES containing text file for first dataset. 
                                         # Write the full path to file.
        
        self.drug_raw_file = config.drug_raw_file  # SMILES containing text file for second dataset. 
                                                   # Write the full path to file.       
                                                   
                                                    
        self.dataset_file = config.dataset_file    # Dataset file name for the first GAN. 
                                                   # Contains large number of molecules.
        
        self.drugs_dataset_file = config.drug_dataset_file  # Drug dataset file name for the second GAN. 
                                                            # Contains drug molecules only. (In this case AKT1 inhibitors.)
        
        self.inf_raw_file = config.inf_raw_file  # SMILES containing text file for first dataset. 
                                         # Write the full path to file.
        
        self.inf_drug_raw_file = config.inf_drug_raw_file  # SMILES containing text file for second dataset. 
                                                   # Write the full path to file.       
                                                   
                                                    
        self.inf_dataset_file = config.inf_dataset_file    # Dataset file name for the first GAN. 
                                                   # Contains large number of molecules.
        
        self.inf_drugs_dataset_file = config.inf_drug_dataset_file  # Drug dataset file name for the second GAN. 
                                                            # Contains drug molecules only. (In this case AKT1 inhibitors.)

        self.mol_data_dir = config.mol_data_dir  # Directory where the dataset files are stored.
        
        self.drug_data_dir = config.drug_data_dir  # Directory where the drug dataset files are stored.
                                                         
        self.dataset_name = self.dataset_file.split(".")[0]
        self.drugs_name = self.drugs_dataset_file.split(".")[0]
        
        self.max_atom = config.max_atom  # Model is based on one-shot generation. 
                                         # Max atom number for molecules must be specified.
                                         
        self.features = config.features  # Small model uses atom types as node features. (Boolean, False uses atom types only.)
                                         # Additional node features can be added. Please check new_dataloarder.py Line 102.
        
        
        self.batch_size = config.batch_size  # Batch size for training.
        
        self.dataset = DruggenDataset(self.mol_data_dir,
                                      self.dataset_file, 
                                      self.raw_file, 
                                      self.max_atom, 
                                      self.features) # Dataset for the first GAN. Custom dataset class from PyG parent class. 
                                                     # Can create any molecular graph dataset given smiles string. 
                                                     # Nonisomeric SMILES are suggested but not necessary.
                                                     # Uses sparse matrix representation for graphs, 
                                                     # For computational and speed efficiency.
        
        self.loader = DataLoader(self.dataset, 
                                 shuffle=True,
                                 batch_size=self.batch_size, 
                                 drop_last=True)  # PyG dataloader for the first GAN.

        self.drugs = DruggenDataset(self.drug_data_dir, 
                                    self.drugs_dataset_file, 
                                    self.drug_raw_file, 
                                    self.max_atom, 
                                    self.features)   # Dataset for the second GAN. Custom dataset class from PyG parent class. 
                                                     # Can create any molecular graph dataset given smiles string. 
                                                     # Nonisomeric SMILES are suggested but not necessary.
                                                     # Uses sparse matrix representation for graphs, 
                                                     # For computational and speed efficiency.
        
        self.drugs_loader = DataLoader(self.drugs, 
                                       shuffle=True,
                                       batch_size=self.batch_size, 
                                       drop_last=True)  # PyG dataloader for the second GAN.
        
        # Atom and bond type dimensions for the construction of the model.
        
        self.atom_decoders = self.decoder_load("atom")  # Atom type decoders for first GAN. 
                                                        # eg. 0:0, 1:6 (C), 2:7 (N), 3:8 (O), 4:9 (F)
                                                        
        self.bond_decoders = self.decoder_load("bond")  # Bond type decoders for first GAN.
                                                        # eg. 0: (no-bond), 1: (single), 2: (double), 3: (triple), 4: (aromatic)
     
        self.m_dim = len(self.atom_decoders) if not self.features else int(self.loader.dataset[0].x.shape[1]) # Atom type dimension.
        
        self.b_dim = len(self.bond_decoders) # Bond type dimension.
        
        self.vertexes = int(self.loader.dataset[0].x.shape[0]) # Number of nodes in the graph.
               
        self.drugs_atom_decoders = self.drug_decoder_load("atom") # Atom type decoders for second GAN.
                                                                  # eg. 0:0, 1:6 (C), 2:7 (N), 3:8 (O), 4:9 (F)
        
        self.drugs_bond_decoders = self.drug_decoder_load("bond") # Bond type decoders for second GAN.
                                                                  # eg. 0: (no-bond), 1: (single), 2: (double), 3: (triple), 4: (aromatic)
        
        self.drugs_m_dim = len(self.drugs_atom_decoders) if not self.features else int(self.drugs_loader.dataset[0].x.shape[1]) # Atom type dimension.
        
        self.drugs_b_dim = len(self.drugs_bond_decoders)    # Bond type dimension.
        
        self.drug_vertexes = int(self.drugs_loader.dataset[0].x.shape[0])  # Number of nodes in the graph.

        # Transformer and Convolution configurations.
        
        self.act = config.act 
        
        self.z_dim = config.z_dim 
        
        self.lambda_gp = config.lambda_gp  
        
        self.dim = config.dim 
        
        self.depth = config.depth 
        
        self.heads = config.heads 
        
        self.mlp_ratio = config.mlp_ratio 
        
        self.dec_depth = config.dec_depth 
        
        self.dec_heads = config.dec_heads 
        
        self.dec_dim = config.dec_dim
        
        self.dis_select = config.dis_select 
        
        """self.la = config.la
        self.la2 = config.la2
        self.gcn_depth = config.gcn_depth
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim"""
        """# PNA config
        
        self.agg = config.aggregators
        self.sca = config.scalers
        self.pna_in_ch = config.pna_in_ch
        self.pna_out_ch = config.pna_out_ch
        self.edge_dim = config.edge_dim
        self.towers = config.towers
        self.pre_lay = config.pre_lay
        self.post_lay = config.post_lay
        self.pna_layer_num = config.pna_layer_num
        self.graph_add = config.graph_add"""
        
        # Training configurations.
        
        self.epoch = config.epoch 
        
        self.g_lr = config.g_lr  
        
        self.d_lr = config.d_lr  
        
        self.g2_lr = config.g2_lr 
        
        self.d2_lr = config.d2_lr
              
        self.dropout = config.dropout
        
        self.dec_dropout = config.dec_dropout
        
        self.n_critic = config.n_critic 
        
        self.beta1 = config.beta1
        
        self.beta2 = config.beta2 
        
        self.resume_iters = config.resume_iters 
        
        self.warm_up_steps = config.warm_up_steps 
        
        # Test configurations.
        
        self.num_test_epoch = config.num_test_epoch
        
        self.test_iters = config.test_iters
        
        self.inference_sample_num = config.inference_sample_num     
        
        # Directories.
        
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir
        
        # Step size.
        
        self.log_step = config.log_sample_step
        self.clipping_value = config.clipping_value
        # Miscellaneous.
        
        self.mode = config.mode

        self.noise_strength_0 = torch.nn.Parameter(torch.zeros([]))
        self.noise_strength_1 = torch.nn.Parameter(torch.zeros([]))        
        self.noise_strength_2 = torch.nn.Parameter(torch.zeros([]))
        self.noise_strength_3 = torch.nn.Parameter(torch.zeros([]))

        self.init_type = config.init_type
        self.build_model()


             
    def build_model(self):
        """Create generators and discriminators."""
        
        ''' Generator is based on Transformer Encoder: 
            
            @ g_conv_dim: Dimensions for first MLP layers before Transformer Encoder
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
            
        self.G = Generator(self.z_dim,
                           self.act,
                           self.vertexes,
                           self.b_dim,
                           self.m_dim,
                           self.dropout,
                           dim=self.dim, 
                           depth=self.depth, 
                           heads=self.heads, 
                           mlp_ratio=self.mlp_ratio)
         
        self.G2 = Generator2(self.dim,
                           self.dec_dim,
                           self.depth,
                           self.heads,
                           self.mlp_ratio,
                           self.dec_dropout,
                           self.drugs_m_dim,
                           self.drugs_b_dim,
                           self.b_dim, 
                           self.m_dim,
                           self.submodel)
        
        
        
        ''' Discriminator implementation with PNA:
        
            @ deg: Degree distribution based on used data. (Created with _genDegree() function)
            @ agg: aggregators used in PNA
            @ sca: scalers used in PNA
            @ pna_in_ch: First PNA hidden dimension
            @ pna_out_ch: Last PNA hidden dimension
            @ edge_dim: Edge hidden dimension
            @ towers: Number of towers (Splitting the hidden dimension to multiple parallel processes)
            @ pre_lay: Pre-transformation layer
            @ post_lay: Post-transformation layer 
            @ pna_layer_num: number of PNA layers
            @ graph_add: global pooling layer selection
            '''

      
        ''' Discriminator implementation with Graph Convolution:
            
            @ d_conv_dim: convolution dimensions for GCN
            @ m_dim: number of atom types (or number of features used)
            @ b_dim: number of bond types
            @ dropout: dropout possibility 
            '''

        ''' Discriminator implementation with MLP:
            
            @ act: Activation function for MLP
            @ m_dim: number of atom types (or number of features used)
            @ b_dim: number of bond types
            @ dropout: dropout possibility 
            @ vertexes: maximum length of generated molecules (molecule length)
            '''        
        
        #self.D = Discriminator_old(self.d_conv_dim, self.m_dim , self.b_dim, self.dropout, self.gcn_depth) 
        self.D2 = simple_disc("tanh", self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)
        self.D = simple_disc("tanh", self.m_dim, self.vertexes, self.b_dim)
        self.V = simple_disc("tanh", self.m_dim, self.vertexes, self.b_dim)
        self.V2 = simple_disc("tanh", self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)
        
        ''' Optimizers for G1, G2, D1, and D2:
            
            Adam Optimizer is used and different beta1 and beta2s are used for GAN1 and GAN2
            '''
        
        self.g_optimizer = torch.optim.AdamW(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g2_optimizer = torch.optim.AdamW(self.G2.parameters(), self.g2_lr, [self.beta1, self.beta2])
        
        self.d_optimizer = torch.optim.AdamW(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.d2_optimizer = torch.optim.AdamW(self.D2.parameters(), self.d2_lr, [self.beta1, self.beta2])
        
        
        
        self.v_optimizer = torch.optim.AdamW(self.V.parameters(), self.d_lr, [self.beta1, self.beta2])       
        self.v2_optimizer = torch.optim.AdamW(self.V2.parameters(), self.d2_lr, [self.beta1, self.beta2]) 
        ''' Learning rate scheduler:
            
            Changes learning rate based on loss.
            '''
        
        #self.scheduler_g = ReduceLROnPlateau(self.g_optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
        
        
        #self.scheduler_d = ReduceLROnPlateau(self.d_optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
   
        #self.scheduler_v = ReduceLROnPlateau(self.v_optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)        
        #self.scheduler_g2 = ReduceLROnPlateau(self.g2_optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
        #self.scheduler_d2 = ReduceLROnPlateau(self.d2_optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)
        #self.scheduler_v2 = ReduceLROnPlateau(self.v2_optimizer, mode='min', factor=0.5, patience=10, min_lr=0.00001)                  
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        
        self.print_network(self.G2, 'G2')
        self.print_network(self.D2, 'D2')
        
        self.G.to(self.device)
        self.D.to(self.device)

        self.V.to(self.device)
        self.V2.to(self.device)
        self.G2.to(self.device)
        self.D2.to(self.device)
  
        #self.V2.to(self.device)      
        #self.modules_of_the_model = (self.G, self.D, self.G2, self.D2)
        """for p in self.G.parameters():
            if p.dim() > 1:
                if self.init_type == 'uniform':
                    torch.nn.init.xavier_uniform_(p)
                elif self.init_type == 'normal':
                    torch.nn.init.xavier_normal_(p)
                elif self.init_type == 'random_normal':
                     torch.nn.init.normal_(p, 0.0, 0.02)
        for p in self.G2.parameters():
            if p.dim() > 1:
                if self.init_type == 'uniform':
                    torch.nn.init.xavier_uniform_(p)
                elif self.init_type == 'normal':
                    torch.nn.init.xavier_normal_(p)   
                elif self.init_type == 'random_normal':
                     torch.nn.init.normal_(p, 0.0, 0.02)                 
        if self.dis_select == "conv":
            for p in self.D.parameters():
                if p.dim() > 1:
                    if self.init_type == 'uniform':
                        torch.nn.init.xavier_uniform_(p)
                    elif self.init_type == 'normal':
                        torch.nn.init.xavier_normal_(p)      
                    elif self.init_type == 'random_normal':
                        torch.nn.init.normal_(p, 0.0, 0.02) 

        if self.dis_select == "conv":
            for p in self.D2.parameters():
                if p.dim() > 1:
                    if self.init_type == 'uniform':
                        torch.nn.init.xavier_uniform_(p)
                    elif self.init_type == 'normal':
                        torch.nn.init.xavier_normal_(p)  
                    elif self.init_type == 'random_normal':
                        torch.nn.init.normal_(p, 0.0, 0.02)"""     

        
    def decoder_load(self, dictionary_name):
        
        ''' Loading the atom and bond decoders'''
        
        with open("DrugGEN/data/decoders/" + dictionary_name + "_" + self.dataset_name + '.pkl', 'rb') as f:
            
            return pickle.load(f)    

    def drug_decoder_load(self, dictionary_name):
        
        ''' Loading the atom and bond decoders'''
        
        with open("DrugGEN/data/decoders/" + dictionary_name +"_" + self.drugs_name +'.pkl', 'rb') as f:
            
            return pickle.load(f)    
 
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
        #D_path = os.path.join(model_directory, '{}-{}-D.ckpt'.format(epoch, iteration))
        
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        #self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
      
        
        G2_path = os.path.join(model_directory, '{}-{}-G2.ckpt'.format(epoch, iteration))
        #D2_path = os.path.join(model_directory, '{}-{}-D2.ckpt'.format(epoch, iteration))
        
        self.G2.load_state_dict(torch.load(G2_path, map_location=lambda storage, loc: storage))
        #self.D2.load_state_dict(torch.load(D2_path, map_location=lambda storage, loc: storage))

   
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
        self.v_optimizer.zero_grad()
        self.g2_optimizer.zero_grad()
        self.v2_optimizer.zero_grad()
            
        self.d_optimizer.zero_grad()
        self.d2_optimizer.zero_grad()

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
        
        #wandb.config = {'beta2': 0.999}
        #wandb.init(project="DrugGEN2", entity="atabeyunlu")
        
        # Defining sampling paths and creating logger
       
        self.arguments = "{}_glr{}_dlr{}_g2lr{}_d2lr{}_dim{}_depth{}_heads{}_decdepth{}_decheads{}_ncritic{}_batch{}_epoch{}_warmup{}_dataset{}_dropout{}".format(self.submodel,self.g_lr,self.d_lr,self.g2_lr,self.d2_lr,self.dim,self.depth,self.heads,self.dec_depth,self.dec_heads,self.n_critic,self.batch_size,self.epoch,self.warm_up_steps,self.dataset_name,self.dropout)
       
        self.model_directory= os.path.join(self.model_save_dir,self.arguments)
        self.sample_directory=os.path.join(self.sample_dir,self.arguments)
        self.log_path = os.path.join(self.log_dir, "{}.txt".format(self.arguments))
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)         
        
        # Learning rate cache for decaying.


        # protein data
        full_smiles = [line for line in open("DrugGEN/data/chembl_train.smi", 'r').read().splitlines()]
        drug_smiles = [line for line in open("DrugGEN/data/akt_train.smi", 'r').read().splitlines()]
        
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_scaf = [MurckoScaffold.GetScaffoldForMol(x) for x in drug_mols]
        fps_r = [Chem.RDKFingerprint(x) for x in drug_scaf]

        akt1_human_adj = torch.load("DrugGEN/data/akt/AKT1_human_adj.pt").reshape(1,-1).to(self.device).float() 
        akt1_human_annot = torch.load("DrugGEN/data/akt/AKT1_human_annot.pt").reshape(1,-1).to(self.device).float() 
      
        # Start training.
        
        print('Start training...')
        self.start_time = time.time()
        for idx in range(self.epoch):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #
          
            # Load the data 
            
            dataloader_iterator = iter(self.drugs_loader)
            
            for i, data in enumerate(self.loader):   
                try:
                    drugs = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.drugs_loader)
                    drugs = next(dataloader_iterator)

                # Preprocess both dataset 
                
                bulk_data = load_data(data,
                                     drugs,
                                     self.batch_size, 
                                     self.device,
                                     self.b_dim,
                                     self.m_dim,
                                     self.drugs_b_dim,
                                     self.drugs_m_dim,
                                     self.z_dim,
                                     self.vertexes)   
                
                drug_graphs, real_graphs, a_tensor, x_tensor, drugs_a_tensor, drugs_x_tensor, z, z_edge, z_node = bulk_data
                
                if self.submodel == "CrossLoss":
                    GAN1_input_e = drugs_a_tensor
                    GAN1_input_x = drugs_x_tensor
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                elif self.submodel == "Ligand":
                    GAN1_input_e = a_tensor
                    GAN1_input_x = x_tensor
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                    GAN2_input_e = drugs_a_tensor
                    GAN2_input_x = drugs_x_tensor
                    GAN2_disc_e = drugs_a_tensor
                    GAN2_disc_x = drugs_x_tensor            
                elif self.submodel == "Prot":        
                    GAN1_input_e = z_edge + a_tensor
                    GAN1_input_x = z_node + x_tensor
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                    GAN2_input_e = akt1_human_adj
                    GAN2_input_x = akt1_human_annot
                    GAN2_disc_e = drugs_a_tensor
                    GAN2_disc_x = drugs_x_tensor        
                elif self.submodel == "RL":
                    GAN1_input_e = z_edge
                    GAN1_input_x = z_node
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                    GAN2_input_e = drugs_a_tensor
                    GAN2_input_x = drugs_x_tensor
                    GAN2_disc_e = drugs_a_tensor
                    GAN2_disc_x = drugs_x_tensor    
                elif self.submodel == "NoTarget":
                    GAN1_input_e = z_edge 
                    GAN1_input_x = z_node 
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor                                                  
                    
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
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
                if self.submodel != "NoTarget" and self.submodel != "CrossLoss":
                    d2_loss = discriminator2_loss(self.G2, 
                                                    self.D2, 
                                                    drug_graphs,
                                                    edge, 
                                                    node, 
                                                    self.batch_size, 
                                                    self.device,
                                                    self.gradient_penalty, 
                                                    self.lambda_gp,
                                                    GAN2_input_e,
                                                    GAN2_input_x)
                    d_total = d_loss + d2_loss
                
                loss["d_total"] = d_total.item()
                d_total.backward()
                self.d_optimizer.step()
                if self.submodel != "NoTarget" and self.submodel != "CrossLoss":
                    self.d2_optimizer.step()
                self.reset_grad()
                generator_output = generator_loss(self.G,
                                                    self.D,
                                                    self.V,
                                                    GAN1_input_e,
                                                    GAN1_input_x,
                                                    self.batch_size,
                                                    sim_reward,
                                                    self.dataset.matrices2mol_drugs,
                                                    fps_r,
                                                    self.submodel)        
                
                g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge = generator_output    
            
                self.reset_grad()
                g_total = g_loss
                if self.submodel != "NoTarget" and self.submodel != "CrossLoss":
                    output = generator2_loss(self.G2,
                                                self.D2,
                                                self.V2,
                                                edge,
                                                node,
                                                self.batch_size,
                                                sim_reward,
                                                self.dataset.matrices2mol_drugs,
                                                fps_r,
                                                GAN2_input_e,
                                                GAN2_input_x,
                                                self.submodel)
                
                    g2_loss, fake_mol_g, dr_g_edges_hat_sample, dr_g_nodes_hat_sample = output     
                
                    g_total = g_loss + g2_loss     
              
                loss["g_total"] = g_total.item()
                g_total.backward()
                self.g_optimizer.step()
                if self.submodel != "NoTarget" and self.submodel != "CrossLoss":
                    self.g2_optimizer.step()
                
                if self.submodel == "RL":
                    self.v_optimizer.step()
                    self.v2_optimizer.step()
                  
                
                if (i+1) % self.log_step == 0:
              
                    logging(self.log_path, self.start_time, fake_mol, full_smiles, i, idx, loss, 1,self.sample_directory) 
                    mol_sample(self.sample_directory,"GAN1",fake_mol, g_edges_hat_sample.detach(), g_nodes_hat_sample.detach(), idx, i)
                    if self.submodel != "NoTarget" and self.submodel != "CrossLoss":
                        logging(self.log_path, self.start_time, fake_mol_g, drug_smiles, i, idx, loss, 2,self.sample_directory)     
                        mol_sample(self.sample_directory,"GAN2",fake_mol_g, dr_g_edges_hat_sample.detach(), dr_g_nodes_hat_sample.detach(), idx, i)
                                  

            if (idx+1) % 10 == 0:   
                self.save_model(self.model_directory,idx,i) 
                print("model saved at epoch {} and iteration {}".format(idx,i))       
                            
                      
  
    def inference(self):
        
        # Load the trained generator.
        self.G.to(self.device)
        #self.D.to(self.device)
        self.G2.to(self.device)
        #self.D2.to(self.device)        
        
        G_path = os.path.join(self.inference_model, '{}-G.ckpt'.format(self.submodel))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        G2_path = os.path.join(self.inference_model, '{}-G2.ckpt'.format(self.submodel))
        self.G2.load_state_dict(torch.load(G2_path, map_location=lambda storage, loc: storage))        
        
        
        drug_smiles = [line for line in open("DrugGEN/data/akt_test.smi", 'r').read().splitlines()]
        
        drug_mols = [Chem.MolFromSmiles(smi) for smi in drug_smiles]
        drug_scaf = [MurckoScaffold.GetScaffoldForMol(x) for x in drug_mols]
        fps_r = [Chem.RDKFingerprint(x) for x in drug_scaf]

        akt1_human_adj = torch.load("DrugGEN/akt/AKT1_human_adj.pt").reshape(1,-1).to(self.device).float() 
        akt1_human_annot = torch.load("DrugGEN/akt/AKT1_human_annot.pt").reshape(1,-1).to(self.device).float() 
        
        self.G.eval()
        #self.D.eval()
        self.G2.eval()
        #self.D2.eval()
        
        self.inf_batch_size =256
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

        self.inf_drugs = DruggenDataset(self.drug_data_dir, 
                                    self.inf_drugs_dataset_file, 
                                    self.inf_drug_raw_file, 
                                    self.max_atom, 
                                    self.features)   # Dataset for the second GAN. Custom dataset class from PyG parent class. 
                                                     # Can create any molecular graph dataset given smiles string. 
                                                     # Nonisomeric SMILES are suggested but not necessary.
                                                     # Uses sparse matrix representation for graphs, 
                                                     # For computational and speed efficiency.
        
        self.inf_drugs_loader = DataLoader(self.inf_drugs, 
                                       shuffle=True,
                                       batch_size=self.inf_batch_size, 
                                       drop_last=True)  # PyG dataloader for the second GAN.        
        start_time = time.time()
        #metric_calc_mol = []
        metric_calc_dr = []
        date = time.time()
        if not os.path.exists("DrugGEN/experiments/inference/{}".format(self.submodel)):
            os.makedirs("DrugGEN/experiments/inference/{}".format(self.submodel))
        with torch.inference_mode():
            
            dataloader_iterator = iter(self.drugs_loader)
            
            for i, data in enumerate(self.loader):   
                try:
                    drugs = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.drugs_loader)
                    drugs = next(dataloader_iterator)

                # Preprocess both dataset 
                
                bulk_data = load_data(data,
                                     drugs,
                                     self.batch_size, 
                                     self.device,
                                     self.b_dim,
                                     self.m_dim,
                                     self.drugs_b_dim,
                                     self.drugs_m_dim,
                                     self.z_dim,
                                     self.vertexes)   
                
                drug_graphs, real_graphs, a_tensor, x_tensor, drugs_a_tensor, drugs_x_tensor, z, z_edge, z_node = bulk_data
                
                if self.submodel == "CrossLoss":
                    GAN1_input_e = a_tensor
                    GAN1_input_x = x_tensor
                    GAN1_disc_e = drugs_a_tensor
                    GAN1_disc_x = drugs_x_tensor
                    GAN2_input_e = drugs_a_tensor
                    GAN2_input_x = drugs_x_tensor
                    GAN2_disc_e = a_tensor
                    GAN2_disc_x = x_tensor
                elif self.submodel == "Ligand":
                    GAN1_input_e = a_tensor
                    GAN1_input_x = x_tensor
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                    GAN2_input_e = drugs_a_tensor
                    GAN2_input_x = drugs_x_tensor
                    GAN2_disc_e = drugs_a_tensor
                    GAN2_disc_x = drugs_x_tensor            
                elif self.submodel == "Prot":        
                    GAN1_input_e = a_tensor 
                    GAN1_input_x = x_tensor
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                    GAN2_input_e = akt1_human_adj
                    GAN2_input_x = akt1_human_annot
                    GAN2_disc_e = drugs_a_tensor
                    GAN2_disc_x = drugs_x_tensor        
                elif self.submodel == "RL":
                    GAN1_input_e = z_edge
                    GAN1_input_x = z_node
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor
                    GAN2_input_e = drugs_a_tensor
                    GAN2_input_x = drugs_x_tensor
                    GAN2_disc_e = drugs_a_tensor
                    GAN2_disc_x = drugs_x_tensor    
                elif self.submodel == "NoTarget":
                    GAN1_input_e = z_edge
                    GAN1_input_x = z_node
                    GAN1_disc_e = a_tensor
                    GAN1_disc_x = x_tensor      
                # =================================================================================== #
                #                             2. GAN1 Inference                                       #
                # =================================================================================== #            
                generator_output = generator_loss(self.G,
                                                    self.D,
                                                    self.V,
                                                    GAN1_input_e,
                                                    GAN1_input_x,
                                                    self.batch_size,
                                                    sim_reward,
                                                    self.dataset.matrices2mol_drugs,
                                                    fps_r,
                                                    self.submodel)   
                
                _, fake_mol, _, _, node, edge = generator_output  

                # =================================================================================== #
                #                             3. GAN2 Inference                                       #
                # =================================================================================== #   
                        
                output = generator2_loss(self.G2,
                                            self.D2,
                                            self.V2,
                                            edge,
                                            node,
                                            self.batch_size,
                                            sim_reward,
                                            self.dataset.matrices2mol_drugs,
                                            fps_r,
                                            GAN2_input_e,
                                            GAN2_input_x,
                                            self.submodel)
            
                _, fake_mol_g, _, _ = output     

                inference_drugs = [Chem.MolToSmiles(line) for line in fake_mol_g if line is not None]   


                
                #inference_smiles = [Chem.MolToSmiles(line) for line in fake_mol]  
                 
                

                print("molecule batch {} inferred, {} % Valid".format(i, (len(inference_drugs)/self.batch_size)*100))  

                with open("DrugGEN/experiments/inference/{}/inference_drugs.txt".format(self.submodel), "a") as f:
                    for molecules in inference_drugs:
                        
                        f.write(molecules)
                        f.write("\n")
                        metric_calc_dr.append(molecules)
            
                 
                                            
                if i == 120:
                    break
        
        et = time.time() - start_time
        
        print("Inference mode is lasted for {:.2f} seconds".format(et))
        
        print("Metrics calculation started using MOSES.")
                   
        print("Validity: ", fraction_valid(inference_drugs), "\n")
        print("Uniqueness: ", fraction_unique(inference_drugs), "\n")
        print("Validity: ", novelty(inference_drugs, drug_smiles), "\n")

        print("Metrics are calculated.")
