import numpy as np
import os
import time
import datetime
import torch.nn
import torch
import torch.nn.functional as F
from utils import *
from models import Generator, Generator2, simple_disc
import torch_geometric.utils as geoutils
#import #wandb
import re
from torch_geometric.loader import DataLoader
from new_dataloader import DruggenDataset
import torch.utils.data
from moses.metrics.metrics import get_all_metrics
from rdkit import RDLogger  
torch.set_num_threads(3)
RDLogger.DisableLog('rdApp.*') 
 
class Solver(object):
    
    """Solver for training and testing DrugGEN."""

    def __init__(self, config):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        """Initialize configurations."""
        
        # Data loader.
        self.raw_file = config.raw_file  # SMILES containing text file for first dataset. 
                                         # Write the full path to file.
        
        self.drug_raw_file = config.drug_raw_file  # SMILES containing text file for second dataset. 
                                                   # Write the full path to file.       
                                                   
                                                    
        self.dataset_file = config.dataset_file    # Dataset file name for the first GAN. 
                                                   # Contains large number of molecules.
        
        self.drugs_dataset_file = config.drug_dataset_file  # Drug dataset file name for the second GAN. 
                                                            # Contains drug molecules only. (In this case AKT1 inhibitors.)
                                                            
        self.dataset_name = self.dataset_file.split(".")[0]
        self.drugs_name = self.drugs_dataset_file.split(".")[0]
        
        self.max_atom = config.max_atom  # Model is based on one-shot generation. 
                                         # Max atom number for molecules must be specified.
                                         
        self.features = config.features  # Small model uses atom types as node features. (Boolean, False uses atom types only.)
                                         # Additional node features can be added. Please check new_dataloarder.py Line 102.
        
        
        self.batch_size = config.batch_size  # Batch size for training.
        
        self.dataset = DruggenDataset(config.mol_data_dir,
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

        self.drugs = DruggenDataset(config.drug_data_dir, 
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
        self.tra_conv = False    
        if self.dis_select == "TraConv":
            self.tra_conv = True
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
                           self.depth,
                           self.heads,
                           self.mlp_ratio,
                           self.dropout,
                           self.drugs_m_dim,
                           self.drugs_b_dim,
                           self.b_dim, 
                           self.m_dim)
        
        self.softmax = torch.nn.Softmax(dim = -1) 
        
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
        self.D2 = simple_disc(self.act, self.drugs_m_dim, self.drug_vertexes, self.drugs_b_dim)
        self.D = simple_disc(self.act, self.m_dim, self.vertexes, self.b_dim)
        #self.V = Discriminator_old(self.d_conv_dim, self.m_dim , self.b_dim, self.dropout, self.gcn_depth)
        #self.V2 = Discriminator_old2(self.d_conv_dim, self.drugs_m_dim , self.drugs_b_dim, self.dropout, self.gcn_depth)
        
        ''' Optimizers for G1, G2, D1, and D2:
            
            Adam Optimizer is used and different beta1 and beta2s are used for GAN1 and GAN2
            '''
        
        self.g_optimizer = torch.optim.AdamW(self.G.parameters(),
                                            self.g_lr, [self.beta1, self.beta2])
        self.g2_optimizer = torch.optim.AdamW(self.G2.parameters(),
                                            self.g2_lr, [self.beta1, self.beta2])
        
        self.d_optimizer = torch.optim.AdamW(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.d2_optimizer = torch.optim.AdamW(self.D2.parameters(), self.d2_lr, [self.beta1, self.beta2])
        
        
        
        #self.v_optimizer = torch.optim.AdamW(self.V.parameters(), self.d_lr, [self.beta1, self.beta2])       
        #self.v2_optimizer = torch.optim.AdamW(self.V2.parameters(), self.d2_lr, [self.beta1, self.beta2]) 
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

        #self.V.to(self.device)
        
        self.G2.to(self.device)
        self.D2.to(self.device)
  
        #self.V2.to(self.device)      
        #self.modules_of_the_model = (self.G, self.D, self.G2, self.D2)
        for p in self.G.parameters():
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
                        torch.nn.init.normal_(p, 0.0, 0.02)     

        
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


    def restore_model(self, resume_iters):
        
        """Restore the trained generator and discriminator."""
        
        print('Loading the trained models from step {}...'.format(resume_iters))
        
        G_path = os.path.join(self.model_directory, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_directory, '{}-D.ckpt'.format(resume_iters))
        V_path = os.path.join(self.model_directory, '{}-V.ckpt'.format(resume_iters))
        
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        self.V.load_state_dict(torch.load(V_path, map_location=lambda storage, loc: storage))
        
        G2_path = os.path.join(self.model_directory, '{}-G2.ckpt'.format(resume_iters))
        D2_path = os.path.join(self.model_directory, '{}-D2.ckpt'.format(resume_iters))
        V2_path = os.path.join(self.model_directory, '{}-V2.ckpt'.format(resume_iters))
        
        self.G2.load_state_dict(torch.load(G2_path, map_location=lambda storage, loc: storage))
        self.D2.load_state_dict(torch.load(D2_path, map_location=lambda storage, loc: storage))
        self.V2.load_state_dict(torch.load(V2_path, map_location=lambda storage, loc: storage))
   

    def reset_grad(self):
        
        """Reset the gradient buffers."""
        
        self.g_optimizer.zero_grad()
        self.g2_optimizer.zero_grad()
            
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


    def label2onehot(self, labels, dim):
        
        """Convert label indices to one-hot vectors."""
        
        out = torch.zeros(list(labels.size())+[dim]).to(self.device)
        out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)
        
        return out.float()


    def sample_z_node(self, batch_size):
        
        ''' Random noise for nodes logits. '''
        
        return np.random.normal(0,1, size=(batch_size,self.vertexes, self.z_dim))  #  128, 9, 5
    
    
    def sample_z_edge(self, batch_size):
        
        ''' Random noise for edges logits. '''
        
        return np.random.normal(0,1, size=(batch_size,self.vertexes,self.vertexes,self.z_dim)) # 128, 9, 9, 5
    
    def sample_z(self, batch_size):
        
        ''' Random noise. '''
        
        return np.random.normal(0,1, size=(batch_size,self.z_dim))  #  128, 9, 5       
  


    def model_save(self, model,model_name, idx, i, dire):
        
        path = os.path.join(self.model_directory, '{}_{}_{}.ckpt'.format(model_name,idx+1,i+1))
        torch.save(model.state_dict(), path)
        print('Saved model checkpoints into {}...'.format(dire))

    def mol_sample(self,model_name, mol, edges, nodes, idx, i):
        sample_path = os.path.join(self.sample_directory,"{}-{}_{}-epoch_iteration".format(model_name,idx+1, i+1))
        
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
            
        mols2grid_image(mol,sample_path)
        
        save_smiles_matrices(mol,edges.detach(), nodes.detach(), sample_path)
        
        if len(os.listdir(sample_path)) == 0:
            os.rmdir(sample_path)
                                
        print("Valid molecules are saved.")
        print("Valid matrices and smiles are saved")
        
    def logging(self, mols, gen_smiles, train_smiles, i,idx, loss, batch_size,model_num):
        
            
        et = time.time() - self.start_time
        et = str(datetime.timedelta(seconds=et))[:-7]
        log = "Elapsed [{}], Epoch/Iteration [{}/{}] for GAN{}".format(et, idx,  i+1, model_num)
        
        # Log update
        m0 = get_all_metrics(gen = gen_smiles, train = train_smiles, batch_size=batch_size, k = None)
        #m0= all_scores_val(fake_mol, mols, full_mols, full_smiles, vert, norm=True)     # 'mols' is output of Fake Reward
        #m1 =all_scores_chem(fake_mol, mols, vert, norm=True)
        #m0.update(m1)
        maxlen = MolecularMetrics.max_component(mols, 45)
        m0 = {k: np.array(v).mean() for k, v in m0.items()}
        loss.update(m0)
        loss.update({'maxlen': maxlen})
        #wandb.log({"maxlen": maxlen})
  
        for tag, value in loss.items():
            
            log += ", {}: {:.4f}".format(tag, value)
        with open(self.log_path, "a") as f:
            f.write(log)                                 
        print(log) 
        print("\n")  
        
    def train(self):
        
        ''' Training Script starts from here'''
        
        #wandb.config = {'beta2': 0.999}
        #wandb.init(project="DrugGEN2", entity="atabeyunlu")
        
        # Defining sampling paths and creating logger
       
        self.arguments = "glr{}_dlr{}_g2lr{}_d2lr{}_dim{}_depth{}_heads{}_decdepth{}_decheads{}_ncritic{}_batch{}_epoch{}_warmup{}_dataset{}_disc-{}_dropout{}".format(self.g_lr,self.d_lr,self.g2_lr,self.d2_lr,self.dim,self.depth,self.heads,self.dec_depth,self.dec_heads,self.n_critic,self.batch_size,self.epoch,self.warm_up_steps,self.dataset_name,self.dis_select,self.dropout)
       
        self.model_directory= os.path.join(self.model_save_dir,self.arguments)
        self.sample_directory=os.path.join(self.sample_dir,self.arguments)
        self.log_path = os.path.join(self.log_dir, "{}.txt".format(self.arguments))
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        if not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)         
        
        # Learning rate cache for decaying.


        # protein data
        full_smiles = [line in line for line in open("DrugGEN/data/chembl_smiles.smi", 'r').readlines()]
        akt1_human_adj = torch.load("DrugGEN/data/akt/AKT1_human_adj.pt")
        akt1_human_annot = torch.load("DrugGEN/data/akt/AKT1_human_annot.pt")  
      
        # Start training.
        ##wandb.watch(self.modules_of_the_model, log="all", log_freq=100)
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
                z = self.sample_z(self.batch_size)                                                   # (batch,max_len)          
            
                z = torch.from_numpy(z).to(self.device).float().requires_grad_(True)                
                data = data.to(self.device)
                drugs = drugs.to(self.device)                                               
                z_e = self.sample_z_edge(self.batch_size)                                                   # (batch,max_len,max_len)    
                z_n = self.sample_z_node(self.batch_size)                                                   # (batch,max_len)          
                z_edge = torch.from_numpy(z_e).to(self.device).float().requires_grad_(True)                                      # Edge noise.(batch,max_len,max_len)
                z_node = torch.from_numpy(z_n).to(self.device).float().requires_grad_(True)                                      # Node noise.(batch,max_len)       
                a = geoutils.to_dense_adj(edge_index = data.edge_index,batch=data.batch,edge_attr=data.edge_attr, max_num_nodes=int(data.batch.shape[0]/self.batch_size)) 
                x = data.x.view(self.batch_size,int(data.batch.shape[0]/self.batch_size),-1)

                a_tensor = self.label2onehot(a, self.b_dim)
                #x_tensor = self.label2onehot(x, self.m_dim)
                x_tensor = x
           
                #a_tensor = a_tensor + torch.randn([a_tensor.size(0), a_tensor.size(1), a_tensor.size(2),1], device=a_tensor.device) * self.noise_strength_0
                #x_tensor = x_tensor + torch.randn([x_tensor.size(0), x_tensor.size(1),1], device=x_tensor.device) * self.noise_strength_1
                
                drugs_a = geoutils.to_dense_adj(edge_index = drugs.edge_index,batch=drugs.batch,edge_attr=drugs.edge_attr, max_num_nodes=int(drugs.batch.shape[0]/self.batch_size))
    
                drugs_x = drugs.x.view(self.batch_size,int(drugs.batch.shape[0]/self.batch_size),-1)
                
                drugs_a = drugs_a.to(self.device).long() 
                drugs_x = drugs_x.to(self.device).long() 
                drugs_a_tensor = self.label2onehot(drugs_a, self.drugs_b_dim)
                drugs_x_tensor = drugs_x
   
                drugs_a_tensor = drugs_a_tensor + torch.randn([drugs_a_tensor.size(0), drugs_a_tensor.size(1), drugs_a_tensor.size(2),1], device=drugs_a_tensor.device) * self.noise_strength_2
                drugs_x_tensor = drugs_x_tensor + torch.randn([drugs_x_tensor.size(0), drugs_x_tensor.size(1),1], device=drugs_x_tensor.device) * self.noise_strength_3
                prot_n = akt1_human_annot[None,:].to(self.device).float()        
                prot_e = akt1_human_adj[None,None,:].view(1,546,546,1).to(self.device).float()

                zeros = torch.zeros(self.batch_size,1,requires_grad=True).to(self.device).float()
                ones = torch.ones(self.batch_size,1,requires_grad=True).to(self.device).float() 
                       
                
                a_tensor_vec = a_tensor.reshape(self.batch_size,-1)
                x_tensor_vec = x_tensor.reshape(self.batch_size,-1)               
                real_graphs = torch.concat((x_tensor_vec,a_tensor_vec),dim=-1)                      
                
                a_drug_vec = drugs_a_tensor.reshape(self.batch_size,-1)
                x_drug_vec = drugs_x_tensor.reshape(self.batch_size,-1)               
                drug_graphs = torch.concat((x_drug_vec,a_drug_vec),dim=-1)                 
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #
                self.reset_grad()
                # Compute loss with real molecules.
                
     
            
                logits_real = self.D(real_graphs)
              
                                  
                prediction_real =  - torch.mean(logits_real)

                #d_loss_real = bce_loss(torch.sigmoid(logits_real), y_real)
                
                

                # Compute loss with fake molecules.

                node, edge= self.G(z_edge, z_node)
    
                edges_hat = self.softmax(edge).view(self.batch_size, -1)
                
                nodes_hat = self.softmax(node).view(self.batch_size, -1)

                graph = torch.cat((nodes_hat, edges_hat), dim=-1)
                

     
                    
                logits_fake = self.D(graph)
                    
                prediction_fake = torch.mean(logits_fake)
                
                #d_loss_fake = bce_loss(torch.sigmoid(logits_fake),y_fake)
                
                # Compute gradient loss.
                #print(a_tensor.shape, edges_hat.shape, x_tensor.shape, nodes_hat.shape)
                eps = torch.rand(graph.size(0),1).to(self.device)
                x_int0 = (eps * real_graphs + (1. - eps) * graph).requires_grad_(True)
               

           
                grad0 = self.D(x_int0)
                d_loss_gp = self.gradient_penalty(grad0, x_int0) 
               
                # Calculate total loss
               
                d_loss = prediction_fake + prediction_real +  d_loss_gp * self.lambda_gp
                
                # Feed the loss
                
                d_loss.backward()
                if self.dis_select == "conv":
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.clipping_value) 
                 

                

                    
                self.d_optimizer.step()
              
                  
                # Logging.
                
                loss = {}
                #loss['D/d_loss_real'] = prediction_real.item()
                #loss['D/d_loss_fake'] = prediction_fake.item()

                #loss['D/loss_gp'] = d_loss_gp.item()
                loss["D/d_loss"] = d_loss.item()
                #wandb.log({"d_loss": d_loss, "iteration/epoch":[i,idx]})       
                # =================================================================================== #
                #                            3. Train the generator                                   #
                # =================================================================================== #                
                if (i+1) % self.n_critic == 0:
                    self.reset_grad()
                    # Generate fake molecules.

                    node, edge = self.G(z_edge, z_node)
                    
                    edges = self.softmax(edge).view(self.batch_size, -1)
                    
                    nodes = self.softmax(node).view(self.batch_size, -1)
                 
                    graph = torch.cat((nodes, edges), dim=-1)
            
                    edges_hat = self.softmax(edge)
                    
                    nodes_hat= self.softmax(node) 

                    g_edges_hat_sample, g_nodes_hat_sample = torch.max(edges_hat, -1)[1], torch.max(nodes_hat, -1)[1]
                    


                        
                    g_logits_fake = self.D(graph)
                
                                        
                    g_prediction_fake = - torch.mean(g_logits_fake) 
                    #g_loss_fake = bce_loss(torch.sigmoid(g_logits_fake),y_real)
                    
              
                   
                    # Real Reward
                    
                    #rewardR = torch.from_numpy(self.reward(mols)).to(self.device)
                    
                    # Fake Reward
                    
                    #rewardF = torch.from_numpy(self.reward(fake_mol)).to(self.device)
                    
                    
                
                    # Reinforcement Loss
                    
                    #value_logit_real,_ = self.V(a_tensor, None, x_tensor,torch.sigmoid)
                    #value_logit_fake,_ = self.V(g_edges_hat, None, g_nodes_hat, torch.sigmoid)
                    #g_loss_value_pred =  torch.mean((value_logit_real - rewardR) ** 2 + (value_logit_fake - rewardF) ** 2)
                    #g_loss_value_pred =  (1 - rewardF) ** 2
                    #g_loss_value = bce_loss(torch.sigmoid(g_loss_value_pred),y_fake_value)
                 

                    # Clone edge and node logits for GAN2
                    
                    
         
                    # Backward and optimize. 
                    
                    g_loss =   g_prediction_fake # + (1. - self.la) * g_loss_value_pred 
                    
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.clipping_value)
                    #torch.nn.utils.clip_grad_norm_(self.V.parameters(), self.clipping_value)
                
                    self.g_optimizer.step()
                    #self.v_optimizer.step()
                    #self.scheduler_g.step(g_loss)
                    #self.scheduler_v.step(g_loss)
                    
                    
                    # Logging.
                    
                    #loss['G/g_loss_fake'] =  g_prediction_fake.item()
                    #loss['G/g_loss_value'] = g_loss_value_pred.item() 
                    
                    loss["G/g_loss"] = g_loss.item() 
                    #wandb.log({ "g_loss": g_loss, "iteration/epoch":[i,idx]})
                    #g_edges_logits_forGAN2, g_nodes_logits_forGAN2 =  edges_logits.detach().clone(), nodes_logits.detach().clone()
                    if (i+1) % 1000 == 0:
                
                        fake_mol = [self.dataset.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                                    for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]    
                                
            
                        gen_smiles =  []    
                        for line in fake_mol:
                            if line is not None:
                                gen_smiles.append(Chem.MolToSmiles(line))
                            elif line is None:
                                gen_smiles.append(None)
                   
                        #print(gen_smiles) 
                        et = time.time() - self.start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Epoch/Iteration [{}/{}] for GAN1".format(et, idx,  i+1)
                        #gen_smiles = [re.sub('\*', '', line)  for line in gen_smiles]
                        #gen_smiles = [re.sub('\.', '', line)  for line in gen_smiles]  
                        #print(gen_smiles) 
                        #if str in gen_smiles:   
                        if not all(isinstance(x, type(None)) for x in gen_smiles):  
                      
      
                            self.mol_sample("GAN1",fake_mol, g_edges_hat_sample.detach(), g_nodes_hat_sample.detach(), idx, i)
                            self.logging(fake_mol, gen_smiles, full_smiles,i,idx,loss,self.batch_size,1) 
                        
                        else:
                            for tag, value in loss.items():
            
                                log += ", {}: {:.4f}".format(tag, value)
                            with open(self.log_path, "a") as f:
                                f.write(log)     
                            
                            print(log) 
                GAN2_edges, GAN2_nodes = edge.detach(), node.detach()              
                loss2 = {}
                
                if (idx+1 > self.warm_up_steps) & (i+1 > 10):
                    
                    # =================================================================================== #
                    #                             4. Train the discriminator - 2                          #
                    # =================================================================================== #
  
                    dr_edges, dr_nodes = self.G2(GAN2_edges, 
                                                GAN2_nodes, 
                                                drugs_x_tensor,
                                                drugs_a_tensor)
                    
     
                    dr_edges_hat = self.softmax(dr_edges).view(self.batch_size, -1)
                    
                    dr_nodes_hat = self.softmax(dr_nodes).view(self.batch_size, -1)

                    graph = torch.cat((dr_nodes_hat, dr_edges_hat), dim=-1)                    

                    
                    dr_logits_fake = self.D2(graph)
      
                    d2_loss_fake = torch.mean(dr_logits_fake)
                    
                    
                    dr_logits_real2 = self.D2(drug_graphs)
                    

            
                    d2_loss_real = - torch.mean(dr_logits_real2)
                    
                    
                    
                    eps = torch.rand(drug_graphs.size(0),1).to(self.device)
                    x_int0 = (eps * drug_graphs + (1. - eps) * graph).requires_grad_(True)
                
             
                    grad0 = self.D2(x_int0)
                    d2_loss_gp = self.gradient_penalty(grad0, x_int0) 
                    d2_loss = d2_loss_fake + d2_loss_real + d2_loss_gp * self.lambda_gp
         

                
                    self.reset_grad()
                    d2_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.D2.parameters(), self.clipping_value)      
           
                    self.d2_optimizer.step()
                                         
              
                    
                    #loss2['D2/d2_loss_real'] = d2_loss_real.item()
                    #loss2['D2/d2_loss_fake'] = d2_loss_fake.item()
                    #loss['D2/loss_gp'] = d2_loss_gp_tra.item()           
                    loss2["D2/d2_loss"] = d2_loss.item()     
                    ##wandb.log({"d2_loss_fake": d2_loss_fake, "d2_loss_real": d2_loss_real, "d2_loss": d2_loss, "iteration/epoch":[i,idx]})       
                    # =================================================================================== #
                    #                             5. Train the generator - 2                              #
                    # =================================================================================== #
                                   
                if ((idx+1 > self.warm_up_steps) & ((i+1) % self.n_critic == 0)):
                
                    
                    dr_edges, dr_nodes = self.G2(GAN2_edges, 
                                                GAN2_nodes, 
                                                drugs_x_tensor,
                                                drugs_a_tensor)

                    
                    
                    dr_edges_hat = self.softmax(dr_edges).view(self.batch_size, -1)
                    
                    dr_nodes_hat = self.softmax(dr_nodes).view(self.batch_size, -1)

                    graph = torch.cat((dr_nodes_hat, dr_edges_hat), dim=-1)      
                    
                    edges_hat = self.softmax(dr_edges)
                    
                    nodes_hat= self.softmax(dr_nodes) 

                    g_edges_hat_sample, g_nodes_hat_sample = torch.max(edges_hat, -1)[1], torch.max(nodes_hat, -1)[1]
                    
                    
                            
                    g_tra_logits_fake2 = self.D2(graph)
                        

                    g2_loss_fake = - torch.mean(g_tra_logits_fake2)                                                  
                              
                    #rewardR2 = torch.from_numpy(self.reward(drugs_mol)).to(self.device)
                    
                    #rewardF2 = torch.from_numpy(self.reward(fake_mol2)).to(self.device)                
                
                    #value_logit_real2,value_f_real2 = self.V2(drugs_a_tensor, None, drugs_x_tensor, torch.sigmoid)
                
                    #value_logit_fake2,value_f_fake2 = self.V2(g_tra_edges_hat, None, g_tra_nodes_hat, torch.sigmoid)
                
                    #g2_loss_value = torch.mean(torch.abs(value_logit_real2 - rewardR2)) + torch.mean(torch.abs(value_logit_fake2 - rewardF2))
        
                    g2_loss =  g2_loss_fake 
                
                    self.reset_grad()
                
                    g2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.G2.parameters(), self.clipping_value)      
                    self.g2_optimizer.step()
                
                    

                    #loss2["G2/g2_loss_fake"] = g2_loss_fake.item()
                    loss2["G2/g2_loss"] = g2_loss.item()        
                    ##wandb.log({ "g2_loss": g2_loss, "iteration/epoch":[i,idx]})
                    if (i+1) % 1000 == 0:
                
                        fake_mol = [self.dataset.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                                    for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]    
                                
            
                        gen_smiles_dr =  []    
                        for line in fake_mol:
                            if line is not None:
                                gen_smiles_dr.append(Chem.MolToSmiles(line))
                            elif line is None:
                                gen_smiles_dr.append(None)
                   
                        #print(gen_smiles) 
                        et = time.time() - self.start_time
                        et = str(datetime.timedelta(seconds=et))[:-7]
                        log = "Elapsed [{}], Epoch/Iteration [{}/{}] for GAN2".format(et, idx,  i+1)
                        #gen_smiles = [re.sub('\*', '', line)  for line in gen_smiles]
                        #gen_smiles = [re.sub('\.', '', line)  for line in gen_smiles]  
                        #print(gen_smiles) 
                        #if str in gen_smiles:   
                        if not all(isinstance(x, type(None)) for x in gen_smiles_dr):  
                      
      
                            self.mol_sample("GAN2",fake_mol, g_edges_hat_sample.detach(), g_nodes_hat_sample.detach(), idx, i)
                            self.logging(fake_mol, gen_smiles, full_smiles,i,idx,loss2,self.batch_size,2) 
                        
                        else:
                            for tag, value in loss2.items():
            
                                log += ", {}: {:.4f}".format(tag, value)
                            with open(self.log_path, "a") as f:
                                f.write(log)     
                            
                            print(log) 
        G_path = os.path.join(self.model_directory, '{}-G.ckpt'.format(i+1))
        D_path = os.path.join(self.model_directory, '{}-D.ckpt'.format(i+1))
        G2_path = os.path.join(self.model_directory, '{}-G2.ckpt'.format(i+1))
        D2_path = os.path.join(self.model_directory, '{}-D2.ckpt'.format(i+1))
        torch.save(self.G.state_dict(), G_path)     
        torch.save(self.D.state_dict(), D_path)     
        torch.save(self.G2.state_dict(), G2_path)         
        torch.save(self.D2.state_dict(), D2_path)                        
  
    def test(self):
        # Load the trained generator.
        self.restore_model(self.test_iters)
        akt1_human_adj = torch.load("DrugGEN/data/akt/AKT1_human_adj.pt")
        akt1_human_annot = torch.load("DrugGEN/data/akt/AKT1_human_annot.pt")  
        prot_n = akt1_human_annot[None,:].to(self.device)          
        prot_e = akt1_human_adj[None,None,:].view(1,546,546,1).to(self.device).long()
        date = time.time()
        with torch.no_grad():
            for idx in range(self.num_test_epoch):

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
                        
                    data = data.to(self.device)
                            
                    drugs = drugs.to(self.device)                                               
                    z_e = self.sample_z_edge(self.batch_size)                                                   # (batch,max_len,max_len)    
                    z_n = self.sample_z_node(self.batch_size)                                                   # (batch,max_len)          
                    z_edge = torch.from_numpy(z_e).to(self.device).float()                                      # Edge noise.(batch,max_len,max_len)
                    z_node = torch.from_numpy(z_n).to(self.device).float()                                      # Node noise.(batch,max_len)       
                    a = geoutils.to_dense_adj(edge_index = data.edge_index,batch=data.batch,edge_attr=data.edge_attr, max_num_nodes=int(data.batch.shape[0]/self.batch_size))
                    x = data.x.view(-1,int(data.batch.shape[0]/self.batch_size))
                
                    a_tensor = self.label2onehot(a, self.b_dim)
                    x_tensor = self.label2onehot(x, self.m_dim)
                    
                    drugs_a = geoutils.to_dense_adj(edge_index = drugs.edge_index,batch=drugs.batch,edge_attr=drugs.edge_attr, max_num_nodes=int(drugs.batch.shape[0]/self.batch_size))
                    drugs_x = drugs.x.view(-1,int(drugs.batch.shape[0]/self.batch_size))
                    
                    drugs_a = drugs_a.to(self.device).long() 
                    drugs_x = drugs_x.to(self.device).long() 
                    drugs_a_tensor = self.label2onehot(drugs_a, self.drugs_b_dim)
                    drugs_x_tensor = self.label2onehot(drugs_x, self.drugs_m_dim)
                    drugs_a_tensor = drugs_a_tensor + torch.randn([drugs_a_tensor.size(0), drugs_a_tensor.size(1), drugs_a_tensor.size(2),1], device=drugs_a_tensor.device) * self.noise_strength_2
                    drugs_x_tensor = drugs_x_tensor + torch.randn([drugs_x_tensor.size(0), drugs_x_tensor.size(1),1], device=drugs_x_tensor.device) * self.noise_strength_3
                    real_edge_attr_for_traconv = self.attr_mlp(data.edge_attr.view(-1,1).float())
                    real_nodes_for_traconv = self.nodes_mlp(data.x.view(-1,1).float())        
                    mols = [self.dataset.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=False) 
                                for e_, n_ in zip(a, x)]                     
            
                    # =================================================================================== #
                    #                             2. GAN1 Inference                                       #
                    # =================================================================================== #            
                    edges_hat, nodes_hat, edges_logits, nodes_logits, nodes_fake,fake_edge_index,fake_edge_attr, attr_for_traconv, nodes_for_traconv = self.G(z_edge,z_node,a,a_tensor,x_tensor)   
                    edges_hat = edges_hat.view(-1, self.vertexes,self.vertexes,self.b_dim)
                    if self.dis_select == "conv":
                    
                        logits_real, features_real = self.D(a_tensor, None, x_tensor)
                        
                        

                    if self.dis_select == "conv":
                    
                        logits_fake, features_fake = self.D(edges_hat, None, nodes_hat)
                        
                    g_edges_hard, g_nodes_hard = torch.max(edges_hat, 1)[1], torch.max(nodes_hat, -1)[1] 
                    fake_mol = [self.dataset.matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                                for e_, n_ in zip(g_edges_hard, g_nodes_hard)] 
                                           
                    d_loss_fake = torch.mean(logits_fake)
                    g_loss = - torch.mean(logits_fake)
                    # =================================================================================== #
                    #                             3. GAN2 Inference                                       #
                    # =================================================================================== #           
                    edges_logits_forGAN2, nodes_logits_forGAN2 = edges_logits.detach().clone(), nodes_logits.detach().clone()
                    tra_edges_hat, tra_nodes_hat, edges_hard, nodes_hard, nodes_fake2, fake_edge_index2, fake_edge_attr2,g2_attr_for_traconv,g2_nodes_for_traconv = self.G2(edges_logits_forGAN2, nodes_logits_forGAN2,prot_n,prot_e)
   
                    
                    if self.dis_select == "conv":
                        
                        logits_fake2, features_fake2 = self.D2(tra_edges_hat, None, tra_nodes_hat)


                        
                    d2_loss_fake = torch.mean(logits_fake2)
                    
                    if self.dis_select == "conv":
      
                        logits_real2, features_real2 = self.D2(drugs_a_tensor, None,drugs_x_tensor)
                    

                    d2_loss_real = - torch.mean(logits_real2)
                    d2_loss = d2_loss_fake + d2_loss_real
                    g2_loss = - torch.mean(logits_fake2)
                    fake_mol2 = [self.dataset.matrices2mol_drugs(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                                        for e_, n_ in zip(edges_hard, nodes_hard)]     
                    
                    drugs_mol = [self.dataset.matrices2mol_drugs(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True) 
                                        for e_, n_ in zip(drugs_a, drugs_x)]        
                                          
                # Log update
                m0, m1 = all_scores_val(fake_mol, mols, norm=True)     # 'mols' is output of Fake Reward
                m0 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m0.items()}
                m0.update(m1)
                m2, m3 = all_scores_chem(fake_mol, mols, norm=True)
                m2 = {k: np.array(v)[np.nonzero(v)].mean() for k, v in m2.items()}
                m2.update(m3)
                
                sample_path = os.path.join(self.sample_path,"inference",date)
                sample_path_GAN2 = os.path.join(self.sample_path_GAN2,"inference",date)
                if not os.path.exists(sample_path_GAN2):
                            os.makedirs(sample_path_GAN2)  
                if not os.path.exists(sample_path):
                            os.makedirs(sample_path)                          
                mols2grid_image(fake_mol,self.sample_path)
                save_smiles_matrices(fake_mol,g_edges_hard.detach(), g_nodes_hard.detach(), sample_path)
                mols2grid_image(fake_mol2,self.sample_path_GAN2)
                save_smiles_matrices(fake_mol2,edges_hard.detach(), nodes_hard.detach(), sample_path_GAN2)                           
                log = "Test started [{}]".format("now")
                for tag, value in m0.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)
                if ((i * self.batch_size) // self.inference_sample_num) >= 1 :
                    break



  # Compute loss for gradient penalty. 
            
            #    eps_node = torch.rand(nodes.size(0)).to(self.device)
            #    eps_attr = torch.rand(logits_real.size(0),1,1,).to(self.device)
            #    eps_idx = torch.rand(logits_real.size(0),1,1,).to(self.device)
            #    x_int0 = (eps * edge_attr + (1. - eps) * fake_edge_attr).requires_grad_(True)
            #    x_int1 = (eps.squeeze(-1) * nodes + (1. - eps.squeeze(-1)) * nodes_fake).requires_grad_(True)
            #    x_int2 = ((eps.squeeze(-1) * edge_index + (1. - eps.squeeze(-1)) * fake_edge_index).requires_grad_(True)) 
            #
            #    grad0= self.D(x_int1, x_int0, x_int2,node_index)
            #    d_loss_gp = self.gradient_penalty(grad0, x_int0) + self.gradient_penalty(grad1, x_int1) 
            
