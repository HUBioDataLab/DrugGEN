import os
import argparse
from xmlrpc.client import boolean
from trainer import Trainer
from torch.backends import cudnn
import torch


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    
    # Trainer for training and inference.
    trainer = Trainer(config) 

    if config.mode == 'train':
        trainer.train()
    elif config.mode == 'inference':
        trainer.inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--submodel', type=str, default="NoTarget", help="Chose model subtype: Prot, CrossLoss, Ligand, Prot, NoTarget ")

    # Model configuration.
    parser.add_argument('--act', type=str, default="relu", help="Activation function for the model.")
    
    parser.add_argument('--z_dim', type=int, default=16, help='Prior noise for the first GAN')
    
    parser.add_argument('--max_atom', type=int, default=45, help='Max atom number for molecules must be specified.')    
    
    parser.add_argument('--lambda_gp', type=float, default=1, help='Gradient penalty lambda multiplier for the first GAN.')
    
    parser.add_argument('--dim', type=int, default=128, help='Dimension of the Transformer Encoder model for GAN1.')
    
    parser.add_argument('--depth', type=int, default=1, help='Depth of the Transformer model from the first GAN.')
    
    parser.add_argument('--heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the first GAN.')
    
    parser.add_argument('--dec_depth', type=int, default=10, help='Depth of the Transformer model from the second GAN.')
    
    parser.add_argument('--dec_heads', type=int, default=8, help='Number of heads for the MultiHeadAttention module from the second GAN.')   
    
    parser.add_argument('--dec_dim', type=int, default=512, help='Dimension of the Transformer Decoder model for GAN2.')
     
    parser.add_argument('--mlp_ratio', type=int, default=3, help='MLP ratio for the Transformers.')
    
    parser.add_argument('--warm_up_steps', type=float, default=0, help=' Warm up steps for the first GAN.')
    
    parser.add_argument('--dis_select', type=str, default="conv", help="Select the discriminator for the first and second GAN.")
    
    parser.add_argument('--init_type', type=str, default="normal", help="Initialization type for the model.")
    
    """parser.add_argument('--g_conv_dim',default=[128, 256, 512, 1024], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--la', type=float, default=0.5, help="lambda value for Total Discriminator loss balance")
    parser.add_argument('--la2', type=float, default=0.5, help="lambda value for Total Generator loss balance") 
    parser.add_argument('--gcn_depth', type=int, default=0, help="GCN layer depth")""" 

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the training.')
    
    parser.add_argument('--epoch', type=int, default=122, help='Epoch number for Training.')
    
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    
    parser.add_argument('--d_lr', type=float, default=0.00001, help='learning rate for D')
    
    parser.add_argument('--g2_lr', type=float, default=0.0001, help='learning rate for G2')
    
    parser.add_argument('--d2_lr', type=float, default=0.00001, help='learning rate for D2')
    
    parser.add_argument('--dropout', type=float, default=0., help='dropout rate')
    
    parser.add_argument('--dec_dropout', type=float, default=0.01, help='dropout rate')
    
    parser.add_argument('--n_critic', type=int, default=1, help='number of D updates per each G update')
    
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    
    parser.add_argument('--clipping_value', type=int, default=2, help='1,2, or 5 suggested but not strictly')
    
    parser.add_argument('--features', type=str2bool, default=False, help='features dimension for nodes')  
      
    
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=10000, help='test model from this step')
    
    parser.add_argument('--num_test_epoch', type=int, default=30000, help='inference epoch')
    
    parser.add_argument('--inference_sample_num', type=int, default=30000, help='inference samples')
    
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    
    # Directories.
    parser.add_argument('--protein_data_dir', type=str, default='DrugGEN/data/akt')      
    
    parser.add_argument('--drug_index', type=str, default='DrugGEN/data/drug_smiles.index')  
    
    parser.add_argument('--drug_data_dir', type=str, default='DrugGEN/data')    
    
    parser.add_argument('--mol_data_dir', type=str, default='DrugGEN/data')
    
    parser.add_argument('--log_dir', type=str, default='DrugGEN/experiments/logs')
    
    parser.add_argument('--model_save_dir', type=str, default='DrugGEN/experiments/models')
    
    parser.add_argument('--sample_dir', type=str, default='DrugGEN/experiments/samples')
    
    parser.add_argument('--result_dir', type=str, default='DrugGEN/experiments/results')
    
    parser.add_argument('--dataset_file', type=str, default='chembl45.pt')    
    
    parser.add_argument('--drug_dataset_file', type=str, default='drugs.pt')        
    
    parser.add_argument('--raw_file', type=str, default='DrugGEN/data/chembl_smiles.smi')     
      
    parser.add_argument('--drug_raw_file', type=str, default='DrugGEN/data/drugs_smiles.smi')   
       
    # Step size.
    parser.add_argument('--log_sample_step', type=int, default=100)

    """# PNA configurations
    parser.add_argument('--aggregators', type=str, default="max,mean,min,std", help='aggregator identifiers - "min","max","std","var","mean","sum"')
    parser.add_argument('--scalers', type=str, default="identity,attenuation,amplification", help='scaler identifiers - "attenuation","amplification","identity","linear", "inverse_linear')
    parser.add_argument('--pna_in_ch',type = int, default=50, help='PNA in channel dimension')
    parser.add_argument('--pna_out_ch', type=int, default=50, help='PNA out channel dimension')
    parser.add_argument('--edge_dim', type=int, default=50, help='PNA edge dimension')
    parser.add_argument('--towers', type=int, default=1, help='PNA towers')
    parser.add_argument('--pre_lay', type=int, default=1, help='Pre-transformation layer number')
    parser.add_argument('--post_lay', type=int, default=1, help='Post-transformation layer number')
    parser.add_argument('--pna_layer_num', type=int, default=2, help='PNA layers')
    parser.add_argument('--graph_add', type=str, default="global_add", help='global_add,set2set,graph_multitrans')"""

    config = parser.parse_args()
    print(config)
    main(config)
